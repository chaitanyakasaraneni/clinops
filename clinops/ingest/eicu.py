"""
eICU Collaborative Research Database (eICU-CRD) loader and feature builder.

This module adds eICU support to clinops as the external-validation counterpart
to the MIMIC-III / MIMIC-IV loaders. It targets the JBHI follow-on to the ICHI
2026 multi-organ deterioration study: a researcher with PhysioNet eICU access
should be able to reproduce the cross-site validation in a few lines of code.

Two classes are provided:

``EicuLoader``
    Low-level, one-method-per-table reader. Returns lightly-validated
    DataFrames straight from the CSV files. Large tables (``vitalPeriodic``,
    ``nurseCharting``, ``lab``, ``medication``, ``vitalAperiodic``) are read in
    chunks so they never have to fit in memory at once.

``EicuTableLoader``
    High-level builder that turns the raw tables into the ICHI-compatible
    22-feature matrix, the five physiology-only organ-deterioration labels, and
    sliding-window sequences ready for model training — with a programmatic
    leakage guard that refuses to emit label-defining variables as features.

Key eICU schema facts (verified against an eICU-CRD v2.0 download, not docs):

- **Time is an integer offset in minutes from ICU admission**
  (``observationoffset``, ``labresultoffset``, ``nursingchartoffset`` …), not a
  wall-clock timestamp. All temporal logic uses ``offset // 60`` hour bins.
- **Columns are already lowercase** — no MIMIC-III-style normalisation needed.
- ``age`` is a **string**; ages over 89 are de-identified as ``"> 89"`` and are
  mapped to 90.
- ``uniquepid`` is the patient-level id; ``patientunitstayid`` is the stay-level
  id (one patient may have several stays). Splitting must group by ``uniquepid``.
- Lab results are **long-format free text** (``lab.labname``) — harmonised via
  :mod:`clinops.ingest.mappings.eicu_lab_map`.
- GCS is in ``nurseCharting`` under category ``Scores`` /
  ``Glasgow coma score`` (valname ``GCS Total``), **not** in ``lab``.
- Vasopressor use comes from charted infusion rates in ``infusionDrug`` (see
  ``EicuCohortConfig.vasopressor_source``). ``medication`` holds *orders*, not
  administrations, and about 40% of its rows have no ``drugname`` (only a
  ``drughiclseqno`` drug code), so it is used only as a fallback for hospitals
  that do not chart infusions. Mechanical ventilation comes from
  ``respiratoryCare`` vent offsets; renal replacement from the ``treatment``
  table and medication text.
- DNR / care-limitation status lives in ``carePlanGeneral`` (``cplgroup ==
  'Care Limitation'``), **not** in the ``diagnosis`` table.

Known gaps (documented rather than silently handled):

- eICU does **not** ship a pre-computed per-timestep SOFA score. APACHE
  (II/IVa) is available per stay in ``apachePatientResult`` but is an
  admission-level severity score, not a temporal feature. Where the ICHI model
  used SOFA as a derived feature, this loader substitutes the shock index
  (HR / SBP) and exposes the stay-level APACHE score as an optional join.

References
----------
Pollard, T.J. et al. The eICU Collaborative Research Database, a freely
available multi-center database for critical care research. Sci Data 5, 180178
(2018). https://doi.org/10.1038/sdata.2018.178
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

from clinops.ingest.mappings.eicu_lab_map import canonical_lab_features, map_labname
from clinops.ingest.schema import LeakageError, SchemaValidationError

logger = logging.getLogger(__name__)

#: A per-chunk row filter, ``(chunk) -> chunk``, applied while reading large tables.
RowFilter = Callable[[pd.DataFrame], pd.DataFrame]

# ---------------------------------------------------------------------------
# Table registry — filenames and minimum required columns
# ---------------------------------------------------------------------------

#: All eICU-CRD tables live as flat CSV files in a single directory.
_TABLE_STEMS: dict[str, str] = {
    "patient": "patient",
    "vital_periodic": "vitalPeriodic",
    "vital_aperiodic": "vitalAperiodic",
    "lab": "lab",
    "medication": "medication",
    "infusion_drug": "infusionDrug",
    "nurse_charting": "nurseCharting",
    "diagnosis": "diagnosis",
    "apache_patient_result": "apachePatientResult",
    "respiratory_care": "respiratoryCare",
    "treatment": "treatment",
    "care_plan_general": "carePlanGeneral",
}

#: Minimum required columns per logical table (post-load, lowercase).
_REQUIRED_COLS: dict[str, list[str]] = {
    "patient": [
        "patientunitstayid",
        "uniquepid",
        "age",
        "gender",
        "unittype",
        "unitdischargeoffset",
    ],
    "vital_periodic": ["patientunitstayid", "observationoffset"],
    "vital_aperiodic": ["patientunitstayid", "observationoffset"],
    "lab": ["patientunitstayid", "labresultoffset", "labname", "labresult"],
    "medication": ["patientunitstayid", "drugstartoffset", "drugname"],
    "infusion_drug": ["patientunitstayid", "infusionoffset", "drugname", "drugrate"],
    "nurse_charting": [
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypecat",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ],
    "diagnosis": ["patientunitstayid", "diagnosisoffset", "diagnosisstring"],
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class EicuConfig(BaseModel):
    """
    Validated configuration for :class:`EicuLoader` / :class:`EicuTableLoader`.

    Parameters
    ----------
    data_dir:
        Directory containing the eICU-CRD CSV files (flat layout). Must exist.
    strict_validation:
        If ``True`` (default), raise :exc:`SchemaValidationError` when a table
        is missing a required column. If ``False``, log a warning and continue.
    chunk_size:
        Row count per chunk when reading large tables (``vitalPeriodic``,
        ``nurseCharting``, ``lab``, ``medication``, ``vitalAperiodic``).
        Default 100_000.
    """

    data_dir: Path
    strict_validation: bool = True
    chunk_size: int = 100_000

    @field_validator("data_dir")
    @classmethod
    def _dir_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"eICU data_dir does not exist: {v}")
        return v

    @field_validator("chunk_size")
    @classmethod
    def _chunk_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"chunk_size must be positive, got {v}")
        return v


# ---------------------------------------------------------------------------
# EicuLoader — low-level table access
# ---------------------------------------------------------------------------


class EicuLoader:
    """
    Low-level reader for eICU-CRD CSV tables.

    One method per table. Each returns a DataFrame with lowercase columns
    (eICU CSVs are already lowercase, so no normalisation is applied) and a
    minimal required-column check. Large tables are read in chunks; an optional
    ``patient_unit_stay_ids`` filter is pushed down into each chunk so a cohort
    subset never materialises the full table.

    Parameters
    ----------
    data_dir:
        Directory of eICU-CRD CSV files.
    strict_validation:
        Raise on missing required columns when ``True`` (default).
    chunk_size:
        Rows per chunk for large tables. Default 100_000.

    Examples
    --------
    >>> loader = EicuLoader("/data/eicu-crd")  # doctest: +SKIP
    >>> pt = loader.load_patient()  # doctest: +SKIP
    >>> vitals = loader.load_vital_periodic(patient_unit_stay_ids=[141168])  # doctest: +SKIP
    """

    def __init__(
        self,
        data_dir: str | Path,
        strict_validation: bool = True,
        chunk_size: int = 100_000,
    ) -> None:
        self._cfg = EicuConfig(
            data_dir=Path(data_dir),
            strict_validation=strict_validation,
            chunk_size=chunk_size,
        )
        logger.info("EicuLoader initialised — data_dir=%s", self._cfg.data_dir)

    # -- public table accessors --------------------------------------------

    def load_patient(self) -> pd.DataFrame:
        """
        Load ``patient`` (one row per ICU stay). Small table, read whole.

        Returns
        -------
        pd.DataFrame
            Includes ``patientunitstayid``, ``uniquepid``, ``hospitalid``,
            ``age`` (a string; see :func:`parse_eicu_age`), and
            ``unitdischargeoffset``.
        """
        return self._load_small("patient")

    def load_diagnosis(self, patient_unit_stay_ids: Sequence[int] | None = None) -> pd.DataFrame:
        """
        Load ``diagnosis`` (free-text diagnosis strings and ICD-9 codes).

        Parameters
        ----------
        patient_unit_stay_ids:
            Restrict to these ICU stays. ``None`` (default) returns all stays.
        """
        df = self._load_small("diagnosis")
        return self._filter_stays(df, patient_unit_stay_ids)

    def load_vital_periodic(
        self,
        patient_unit_stay_ids: Sequence[int] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load ``vitalPeriodic`` (~5-min resolution vitals). Chunked read.

        Parameters
        ----------
        patient_unit_stay_ids:
            Restrict to these ICU stays (applied to each chunk).
        columns:
            Subset of columns to read (``usecols``) — cuts memory and time on
            this wide table. ``patientunitstayid`` and ``observationoffset`` are
            always included.
        """
        return self._load_large("vital_periodic", patient_unit_stay_ids, columns=columns)

    def load_vital_aperiodic(
        self,
        patient_unit_stay_ids: Sequence[int] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load ``vitalAperiodic`` (non-invasive blood pressure etc.). Chunked read.

        Parameters
        ----------
        patient_unit_stay_ids:
            Restrict to these ICU stays (applied to each chunk).
        columns:
            Subset of columns to read. ``patientunitstayid`` and
            ``observationoffset`` are always included.
        """
        return self._load_large("vital_aperiodic", patient_unit_stay_ids, columns=columns)

    def load_lab(
        self,
        patient_unit_stay_ids: Sequence[int] | None = None,
        labnames: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load ``lab`` (long-format lab results). Chunked read.

        Parameters
        ----------
        patient_unit_stay_ids:
            Restrict to these ICU stays (applied to each chunk).
        labnames:
            If given, keep only these raw ``labname`` values (case-insensitive),
            applied per chunk to avoid materialising unrelated labs.
        """
        lab_lc = {s.strip().lower() for s in labnames} if labnames else None

        def _row_filter(chunk: pd.DataFrame) -> pd.DataFrame:
            if lab_lc is not None and "labname" in chunk.columns:
                chunk = chunk[chunk["labname"].str.strip().str.lower().isin(lab_lc)]
            return chunk

        return self._load_large("lab", patient_unit_stay_ids, extra_filter=_row_filter)

    def load_medication(
        self,
        patient_unit_stay_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Load ``medication`` (drug *orders*, not administrations). Chunked read.

        About 40% of rows have no ``drugname`` and identify the drug only by
        its ``drughiclseqno`` code; ``drugordercancelled`` marks cancelled
        orders.

        Parameters
        ----------
        patient_unit_stay_ids:
            Restrict to these ICU stays (applied to each chunk).
        """
        return self._load_large("medication", patient_unit_stay_ids)

    def load_infusion_drug(
        self,
        patient_unit_stay_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Load ``infusionDrug`` (charted continuous-infusion rates). Chunked read.

        Each row is one charting of an infusion: ``drugname`` (e.g.
        ``"Norepinephrine (mcg/min)"``), ``drugrate`` (a string; numeric for most
        rows), and ``infusionoffset`` in minutes from ICU admission.

        Parameters
        ----------
        patient_unit_stay_ids:
            Restrict to these ICU stays (applied to each chunk).
        """
        return self._load_large("infusion_drug", patient_unit_stay_ids)

    def load_nurse_charting(
        self,
        patient_unit_stay_ids: Sequence[int] | None = None,
        categories: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load ``nurseCharting`` (~11 GB). Chunked read — never read whole.

        Parameters
        ----------
        patient_unit_stay_ids:
            Restrict to these ICU stays (applied to each chunk).
        categories:
            If given, keep only rows whose ``nursingchartcelltypecat`` is in
            this set (e.g. ``["Scores"]`` for GCS). Applied per chunk.
        """
        cats = set(categories) if categories else None

        def _row_filter(chunk: pd.DataFrame) -> pd.DataFrame:
            if cats is not None and "nursingchartcelltypecat" in chunk.columns:
                chunk = chunk[chunk["nursingchartcelltypecat"].isin(cats)]
            return chunk

        return self._load_large("nurse_charting", patient_unit_stay_ids, extra_filter=_row_filter)

    # -- internal helpers ---------------------------------------------------

    def _resolve_path(self, table: str) -> Path:
        stem = _TABLE_STEMS[table]
        names = [f"{stem}{ext}" for ext in (".csv", ".csv.gz", ".parquet")]
        for name in names:
            candidate = self._cfg.data_dir / name
            if candidate.exists():
                return candidate
        # File-name case differs between releases (the demo ships "infusiondrug",
        # the full release "infusionDrug"); match case-insensitively so this also
        # works on case-sensitive file systems.
        by_lower = {f.name.lower(): f for f in self._cfg.data_dir.iterdir() if f.is_file()}
        for name in names:
            if name.lower() in by_lower:
                return by_lower[name.lower()]
        raise FileNotFoundError(
            f"eICU table '{table}' ({stem}) not found in {self._cfg.data_dir}. "
            f"Expected one of: {stem}.csv, {stem}.csv.gz, {stem}.parquet"
        )

    def _validate(self, table: str, df: pd.DataFrame) -> None:
        required = _REQUIRED_COLS.get(table, [])
        missing = [c for c in required if c not in df.columns]
        if not missing:
            return
        msg = f"eICU table '{table}' missing required columns: {missing}"
        if self._cfg.strict_validation:
            raise SchemaValidationError(msg)
        logger.warning(msg)

    def _load_small(self, table: str) -> pd.DataFrame:
        path = self._resolve_path(table)
        logger.debug("Loading eICU %s from %s (whole)", table, path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        self._validate(table, df)
        logger.debug("Loaded %s: %d rows", table, len(df))
        return df

    def _load_large(
        self,
        table: str,
        patient_unit_stay_ids: Sequence[int] | None = None,
        columns: Sequence[str] | None = None,
        extra_filter: RowFilter | None = None,
    ) -> pd.DataFrame:
        """
        Chunked read of a large table with optional stay/row filtering.

        ``extra_filter`` is a callable ``(chunk) -> chunk`` applied to each
        chunk before concatenation (used for labname / category filters).
        """
        path = self._resolve_path(table)
        stay_set = set(patient_unit_stay_ids) if patient_unit_stay_ids else None

        usecols = None
        if columns is not None:
            wanted = {"patientunitstayid", "observationoffset", *columns}
            # Resolve against the actual file schema so we only request columns
            # that exist — and never hand a Parquet file to the CSV reader.
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq

                # pyarrow.parquet.read_schema lacks a return annotation, so
                # mypy --strict flags the call; the result is a pyarrow Schema.
                available: list[str] = list(
                    pq.read_schema(path).names  # type: ignore[no-untyped-call, unused-ignore]
                )
            else:
                available = list(pd.read_csv(path, nrows=0).columns)
            usecols = [c for c in available if c in wanted]

        logger.debug(
            "Loading eICU %s from %s (chunked, chunk_size=%d)",
            table,
            path,
            self._cfg.chunk_size,
        )

        if path.suffix == ".parquet":
            df = pd.read_parquet(path, columns=usecols)
            df = self._apply_chunk_filters(df, stay_set, extra_filter)
            self._validate(table, df)
            return df.reset_index(drop=True)

        processed: list[pd.DataFrame] = []
        reader = pd.read_csv(
            path,
            low_memory=False,
            chunksize=self._cfg.chunk_size,
            usecols=usecols,
        )
        for chunk in reader:
            chunk = self._apply_chunk_filters(chunk, stay_set, extra_filter)
            if len(chunk):
                processed.append(chunk)

        if processed:
            df = pd.concat(processed, ignore_index=True)
        else:
            # Empty result — preserve columns from the header for a clean schema.
            df = pd.read_csv(path, nrows=0, usecols=usecols)

        self._validate(table, df)
        logger.debug("Loaded %s: %d rows (filtered)", table, len(df))
        return df

    @staticmethod
    def _apply_chunk_filters(
        chunk: pd.DataFrame,
        stay_set: set[int] | None,
        extra_filter: RowFilter | None,
    ) -> pd.DataFrame:
        if stay_set is not None and "patientunitstayid" in chunk.columns:
            chunk = chunk[chunk["patientunitstayid"].isin(stay_set)]
        if extra_filter is not None and len(chunk):
            chunk = extra_filter(chunk)
        return chunk

    @classmethod
    def _filter_stays(
        cls, df: pd.DataFrame, patient_unit_stay_ids: Sequence[int] | None
    ) -> pd.DataFrame:
        """Whole-table version of the per-chunk stay filter."""
        stay_set = set(patient_unit_stay_ids) if patient_unit_stay_ids is not None else None
        return cls._apply_chunk_filters(df, stay_set, None).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers for value parsing
# ---------------------------------------------------------------------------


def _text(series: pd.Series) -> pd.Series:
    """
    A column as strings with "" for missing values.

    A text column that is empty in every row is read as float64, and pandas 3.1
    warns on ``fillna("")`` for it, so cast to a string dtype first.
    """
    return series.astype("string").fillna("")


def parse_eicu_age(value: object) -> float:
    """
    Parse an eICU ``age`` cell into a float.

    eICU de-identifies ages over 89 as the string ``"> 89"``; those map to 90.
    Empty / unparseable values map to ``NaN``.

    Examples
    --------
    >>> parse_eicu_age("> 89")
    90.0
    >>> parse_eicu_age("67")
    67.0
    >>> import math; math.isnan(parse_eicu_age(""))
    True
    """
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    if s.startswith(">"):
        # "> 89" -> 90 (one above the de-identification threshold)
        return 90.0
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Feature / label schema for the ICHI-compatible builder
# ---------------------------------------------------------------------------

#: 7 vital features (canonical names) and their eICU source columns.
_VITAL_SOURCES: dict[str, str] = {
    "heart_rate": "heartrate",
    "sbp": "systemicsystolic",
    "dbp": "systemicdiastolic",
    "map": "systemicmean",  # invasive; non-invasive fallback merged in separately
    "resp_rate": "respiration",
    "temperature": "temperature",
    "spo2": "sao2",
}
_VITAL_FEATURES: list[str] = list(_VITAL_SOURCES.keys())

#: 9 lab features. ``gcs_total`` is sourced from nurseCharting, not lab; the
#: rest come from the long-format lab table via ``eicu_lab_map``.
_LAB_FEATURES: list[str] = [
    "creatinine",
    "bilirubin",
    "platelets",
    "wbc",
    "lactate",
    "ph",
    "pao2",
    "paco2",
    "gcs_total",
]

# Variables whose own measurement triggers an organ label. Under
# label_mode="observed" a criterion fires only in hours where its trigger
# was measured, not where it was carried forward.
_LABEL_TRIGGERS = [
    "map",
    "lactate",
    "pao2",
    "resp_rate",
    "spo2",
    "creatinine",
    "bilirubin",
    "gcs_total",
]

#: 3 derived features. SOFA is not pre-computed in eICU (documented gap) and is
#: emitted as an all-NaN column for schema parity with the MIMIC pipeline;
#: shock index is computed; APACHE is joined per stay from apachePatientResult.
_DERIVED_FEATURES: list[str] = ["shock_index", "sofa", "apache_ii"]

#: 3 treatment-indicator features (model *inputs*, not label criteria).
_TREATMENT_FEATURES: list[str] = [
    "vasopressor_flag",
    "mechanical_ventilation_flag",
    "rrt_flag",
]

#: The full 22-feature matrix, in deterministic order (matches ICHI 2026).
FEATURE_COLUMNS: list[str] = (
    _VITAL_FEATURES + _LAB_FEATURES + _DERIVED_FEATURES + _TREATMENT_FEATURES
)

#: The five organ-system deterioration labels.
ORGAN_LABELS: list[str] = [
    "cardiovascular",
    "respiratory",
    "renal",
    "hepatic",
    "neurological",
]

#: Programmatic leakage-exclusion list. These are derived *only* to define
#: labels and must never appear as model inputs. (Note: the treatment flags
#: ``vasopressor_flag`` / ``mechanical_ventilation_flag`` are deliberately NOT
#: here — in the ICHI feature set they are legitimate treatment-indicator
#: inputs, not label criteria. The genuinely label-defining derived quantities
#: are the per-patient baselines and the PaO2/FiO2 ratio with its FiO2 input.)
LEAKAGE_FORBIDDEN: frozenset[str] = frozenset(
    {
        "baseline_creatinine",
        "baseline_gcs",
        "pao2_fio2_ratio",
        "fio2",
    }
)

#: Vasopressor drug names, matched as whole words so "epinephrine" does not match
#: "racepinephrine". Brand names map to the generic drug.
_VASOPRESSOR_NAME = re.compile(
    r"\b(norepinephrine|levophed|epinephrine|adrenalin|phenylephrine|neo-?synephrine"
    r"|vasopressin|dopamine)\b",
    re.IGNORECASE,
)
_VASOPRESSOR_BRANDS = {
    "levophed": "norepinephrine",
    "adrenalin": "epinephrine",
    "neo-synephrine": "phenylephrine",
    "neosynephrine": "phenylephrine",
}
#: Orders whose name contains a vasopressor but which are not systemic vasopressor
#: therapy: local-anaesthetic combinations, eye and nasal drops, inhaled forms.
_NOT_VASOPRESSOR_THERAPY = re.compile(
    r"lidocaine|bupivacaine|ropivacaine|articaine|prilocaine|cyclopentolate|tropicamide"
    r"|ophth|nasal|inhal|\bneb",
    re.IGNORECASE,
)
#: A charted infusion rate counts as running until the drug's next charting, but
#: for at most this long. In the eICU-CRD demo, half of consecutive chartings of a
#: running vasopressor are 60 minutes apart and 90% are at most 200 minutes apart.
_INFUSION_MAX_GAP_MINUTES = 240

#: Legacy (``vasopressor_source="medication_name"``): substrings of
#: ``medication.drugname`` (case-insensitive), the rule of clinops <= 0.6.
_VASOPRESSOR_KEYWORDS: tuple[str, ...] = (
    "norepinephrine",
    "epinephrine",  # also matches norepinephrine; harmless for a boolean flag
    "dopamine",
    "vasopressin",
    "phenylephrine",
)

#: Substrings used to infer renal-replacement therapy from ``treatment`` /
#: ``medication`` free text.
_RRT_KEYWORDS: tuple[str, ...] = (
    "dialysis",
    "renal replacement",
    "crrt",
    "cvvh",
    "hemodialysis",
)


class EicuCohortConfig(BaseModel):
    """
    Cohort-selection and windowing configuration for :class:`EicuTableLoader`.

    Defaults mirror the ICHI 2026 study design. All fields are keyword
    arguments to the constructor.

    Attributes
    ----------
    min_age:
        Minimum age in years (adults). Default 18.
    min_los_hours:
        Minimum ICU length of stay in hours. Default 48.
    max_icu_hours:
        Hours of each stay to materialise on the hourly grid. Default 72.
    observation_hours:
        Length of the input window. Default 24.
    prediction_hours:
        Length of the forward label horizon. Default 6.
    baseline_hours:
        Window (from ICU admission) used to compute each patient's physiological
        baseline (lowest creatinine, best GCS) for the renal and neurological
        labels. Default 6. Kept separate from ``prediction_hours`` — they are
        unrelated concepts that merely share a default.
    missingness_threshold:
        Stays whose critical-variable missingness on the hourly grid exceeds
        this fraction are excluded. Default 0.30.
    dnr_window_hours:
        Stays with a DNR / care-limitation order within this many hours of
        admission are excluded. Default 6.
    first_stay_only:
        Keep only the first ICU stay per ``uniquepid``. Default ``True``.
    hospital_ids:
        Optional restriction to specific ``hospitalid`` values (multi-site
        subgroup analysis).
    limit_stays:
        Optional cap on the number of cohort stays (development / testing on a
        subset). ``None`` (default) uses the full cohort.
    random_state:
        Deprecated and unused: cohort construction involves no random sampling
        (``limit_stays`` keeps the first stays). Setting it emits a
        ``DeprecationWarning``; it will be removed in a future release.
    label_mode:
        How organ labels treat carried-forward values. ``"observed"`` (default)
        marks an hour positive only when the variable that triggers the
        criterion was actually measured in that hour, so a label reflects new
        evidence rather than a value carried in from before the prediction
        horizon. Companion values (FiO2 for the PaO2/FiO2 ratio, baselines)
        may still be carried forward. ``"locf"`` evaluates the criteria on the
        forward-filled grid, the behaviour of clinops <= 0.5.0; use it only to
        reproduce results produced with that definition.
    vasopressor_source:
        Where ``vasopressor_flag`` comes from. ``"infusion"`` (default) uses
        charted vasopressor infusion rates in ``infusionDrug``: an hour is
        flagged while a norepinephrine, epinephrine, phenylephrine,
        vasopressin, or dopamine infusion runs at a rate above zero (a charted
        rate holds until the drug's next charting, for at most 4 hours). Stays
        at hospitals that chart no infusions fall back to ``medication``
        orders, matched by drug name or ``drughiclseqno`` code, skipping
        cancelled, as-needed (PRN), and one-time orders and non-systemic
        products such as lidocaine-epinephrine. ``"medication_name"`` is the
        rule of clinops <= 0.6: any ``medication`` order whose name contains a
        vasopressor keyword, over its order interval. On the eICU-CRD demo it
        caught only about a third of the stays with a charted vasopressor
        infusion, because ~40% of ``medication`` rows have no name; use it only
        to reproduce earlier results.
    """

    min_age: int = 18
    min_los_hours: int = 48
    max_icu_hours: int = 72
    observation_hours: int = 24
    prediction_hours: int = 6
    baseline_hours: int = 6
    missingness_threshold: float = 0.30
    dnr_window_hours: int = 6
    first_stay_only: bool = True
    hospital_ids: list[int] | None = None
    limit_stays: int | None = None
    random_state: int = 42
    label_mode: Literal["observed", "locf"] = "observed"
    vasopressor_source: Literal["infusion", "medication_name"] = "infusion"

    @model_validator(mode="after")
    def _check_consistency(self) -> EicuCohortConfig:
        if "random_state" in self.model_fields_set:
            warnings.warn(
                "EicuCohortConfig.random_state is unused and deprecated; cohort "
                "construction involves no random sampling.",
                DeprecationWarning,
                stacklevel=2,
            )
        positive = {
            "min_age": self.min_age,
            "min_los_hours": self.min_los_hours,
            "max_icu_hours": self.max_icu_hours,
            "observation_hours": self.observation_hours,
            "prediction_hours": self.prediction_hours,
            "baseline_hours": self.baseline_hours,
            "dnr_window_hours": self.dnr_window_hours,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.missingness_threshold <= 1.0:
            raise ValueError(
                f"missingness_threshold must be in [0, 1], got {self.missingness_threshold}"
            )
        if self.observation_hours + self.prediction_hours > self.max_icu_hours:
            raise ValueError(
                f"observation_hours + prediction_hours "
                f"({self.observation_hours} + {self.prediction_hours}) must be "
                f"<= max_icu_hours ({self.max_icu_hours}); otherwise no window fits."
            )
        return self


# ---------------------------------------------------------------------------
# EicuTableLoader — ICHI-compatible feature/label/sequence builder
# ---------------------------------------------------------------------------


class EicuTableLoader:
    """
    Build the ICHI-compatible feature matrix, labels, and sequences from eICU.

    Wraps :class:`EicuLoader` and produces, for the selected cohort:

    * a 22-feature hourly matrix (7 vitals, 9 labs, 3 derived, 3 treatment),
    * five physiology-only organ-deterioration labels, and
    * sliding-window sequences ``(X, y)`` ready for model training,

    with a programmatic :class:`~clinops.ingest.schema.LeakageError` guard that
    refuses to emit any label-defining variable as a feature.

    Downstream code (``clinops.temporal``, ``clinops.split``) does not need to
    know the data came from eICU — column names are the canonical clinops names.

    Parameters
    ----------
    data_dir:
        Directory of eICU-CRD CSV files.
    config:
        :class:`EicuCohortConfig`. If ``None``, ICHI-study defaults are used.
    strict_validation:
        Passed through to the underlying :class:`EicuLoader`.
    chunk_size:
        Passed through to the underlying :class:`EicuLoader` for large tables.

    Examples
    --------
    >>> loader = EicuTableLoader("/data/eicu-crd")  # doctest: +SKIP
    >>> X, y = loader.build_sequences(observation_hours=24, prediction_hours=6)  # doctest: +SKIP
    >>> meta = loader.build_cohort()[  # doctest: +SKIP
    ...     ["uniquepid", "patientunitstayid", "unittype", "age", "gender"]
    ... ]
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: EicuCohortConfig | None = None,
        strict_validation: bool = True,
        chunk_size: int = 100_000,
    ) -> None:
        self._cfg = config or EicuCohortConfig()
        self._loader = EicuLoader(
            data_dir=data_dir,
            strict_validation=strict_validation,
            chunk_size=chunk_size,
        )
        self._cohort: pd.DataFrame | None = None
        logger.info("EicuTableLoader ready — data_dir=%s", data_dir)

    # -- cohort -------------------------------------------------------------

    def build_cohort(self) -> pd.DataFrame:
        """
        Select the analysis cohort (one row per included ICU stay).

        Applies, in order: adult age filter, first-stay-per-patient, minimum
        ICU LOS, optional hospital restriction, DNR-within-6h exclusion, and an
        optional development cap. The hourly-missingness exclusion is applied
        later (in :meth:`build_feature_matrix`) because it needs the hourly grid.

        Returns
        -------
        pd.DataFrame
            Columns include ``patientunitstayid``, ``uniquepid``, ``hospitalid``,
            ``unittype``, ``age`` (float, ``> 89`` → 90), ``gender``,
            ``los_hours``, and ``apache_ii`` (may be NaN).
        """
        if self._cohort is not None:
            return self._cohort

        pt = self._loader.load_patient().copy()
        n0 = len(pt)

        pt["age"] = pt["age"].map(parse_eicu_age)
        pt = pt[pt["age"] >= self._cfg.min_age]
        logger.info("Cohort: age>=%d → %d/%d stays", self._cfg.min_age, len(pt), n0)

        if self._cfg.hospital_ids is not None and "hospitalid" in pt.columns:
            pt = pt[pt["hospitalid"].isin(self._cfg.hospital_ids)]
            logger.info("Cohort: hospital filter → %d stays", len(pt))

        # Minimum ICU LOS from the unit discharge offset (minutes → hours).
        pt["los_hours"] = pd.to_numeric(pt["unitdischargeoffset"], errors="coerce") / 60.0
        pt = pt[pt["los_hours"] >= self._cfg.min_los_hours]
        logger.info("Cohort: LOS>=%dh → %d stays", self._cfg.min_los_hours, len(pt))

        # First ICU stay per patient.
        if self._cfg.first_stay_only:
            if "unitvisitnumber" in pt.columns:
                order = ["unitvisitnumber", "patientunitstayid"]
            else:
                order = ["patientunitstayid"]
            pt = pt.sort_values(order).groupby("uniquepid", as_index=False, sort=False).first()
            logger.info("Cohort: first stay per patient → %d stays", len(pt))

        # DNR within the configured window (default 6h) of admission → exclude.
        dnr_stays = self._dnr_stays(within_minutes=self._cfg.dnr_window_hours * 60)
        if dnr_stays:
            before = len(pt)
            pt = pt[~pt["patientunitstayid"].isin(dnr_stays)]
            logger.info(
                "Cohort: DNR<=%dh excluded %d stays",
                self._cfg.dnr_window_hours,
                before - len(pt),
            )

        if self._cfg.limit_stays is not None:
            pt = pt.head(self._cfg.limit_stays)
            logger.info("Cohort: dev limit_stays → %d stays", len(pt))

        # Stay-level APACHE severity (optional join).
        pt = pt.merge(self._apache_scores(), on="patientunitstayid", how="left")

        self._cohort = pt.reset_index(drop=True)
        logger.info("Cohort finalised: %d stays", len(self._cohort))
        return self._cohort

    def _dnr_stays(self, within_minutes: int) -> set[int]:
        """
        Stays with a DNR / care-limitation order within ``within_minutes``.

        In eICU, code-status / care-limitation lives in ``carePlanGeneral``
        (``cplgroup == 'Care Limitation'``, ``cplitemvalue`` such as
        ``'Do not resuscitate'``) with offset column ``cplitemoffset`` — *not*
        in the ``diagnosis`` table (verified against a real eICU-CRD v2.0
        download: the diagnosis strings contain no DNR entries).
        """
        try:
            cpg = self._loader._load_small("care_plan_general")
        except FileNotFoundError:
            logger.warning("carePlanGeneral not found; skipping DNR exclusion")
            return set()
        required = {"cplitemvalue", "cplitemoffset", "patientunitstayid"}
        if not required.issubset(cpg.columns):
            logger.warning("carePlanGeneral missing expected columns; skipping DNR exclusion")
            return set()
        # Restrict to the care-limitation group so the same text appearing in an
        # unrelated care-plan group cannot over-exclude stays (matches docstring).
        if "cplgroup" in cpg.columns:
            cpg = cpg[cpg["cplgroup"].astype(str).str.strip().str.lower() == "care limitation"]
        item = cpg["cplitemvalue"].astype(str).str.lower()
        is_dnr = item.str.contains("do not resuscitate", na=False) | item.str.contains(
            r"\bdnr\b|\bdnar\b", na=False, regex=True
        )
        early = pd.to_numeric(cpg["cplitemoffset"], errors="coerce") <= within_minutes
        return set(cpg.loc[is_dnr & early, "patientunitstayid"].unique().tolist())

    def _apache_scores(self) -> pd.DataFrame:
        """Per-stay APACHE score (best-effort; empty frame if unavailable)."""
        try:
            ap = self._loader._load_small("apache_patient_result")
        except FileNotFoundError:
            return pd.DataFrame(columns=["patientunitstayid", "apache_ii"])
        if "apachescore" not in ap.columns:
            return pd.DataFrame(columns=["patientunitstayid", "apache_ii"])
        ap = ap[ap["apachescore"] >= 0]  # eICU uses -1 for missing
        # Prefer APACHE IV/IVa rows when both versions exist; take max score.
        return ap.groupby("patientunitstayid", as_index=False).agg(apache_ii=("apachescore", "max"))

    # -- hourly assembly ----------------------------------------------------

    def _hour_grid(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Dense (stay, hour) grid covering [0, min(LOS, max_icu_hours)) per stay."""
        max_h = self._cfg.max_icu_hours
        # Vectorised: avoids building one DataFrame per stay (slow at 60k+ stays).
        los = cohort["los_hours"].to_numpy()
        n_hours = np.clip(np.floor(los), 1, max_h).astype(int)
        stays = np.repeat(cohort["patientunitstayid"].to_numpy(), n_hours)
        hours = np.concatenate([np.arange(n, dtype=int) for n in n_hours])
        return pd.DataFrame({"patientunitstayid": stays, "hour": hours})

    def _hourly_vitals(self, stay_ids: list[int]) -> pd.DataFrame:
        """Mean-aggregate vitalPeriodic to hourly bins, with NIBP MAP fallback."""
        vp = self._loader.load_vital_periodic(
            patient_unit_stay_ids=stay_ids,
            columns=list(_VITAL_SOURCES.values()),
        )
        vp = vp.rename(columns={src: feat for feat, src in _VITAL_SOURCES.items()})
        vp["hour"] = (pd.to_numeric(vp["observationoffset"], errors="coerce") // 60).astype("Int64")
        vp = vp[(vp["hour"] >= 0) & (vp["hour"] < self._cfg.max_icu_hours)]
        agg = vp.groupby(["patientunitstayid", "hour"], as_index=False)[_VITAL_FEATURES].mean()

        # Non-invasive BP fallback (vitalAperiodic) where invasive is missing.
        try:
            va = self._loader.load_vital_aperiodic(
                patient_unit_stay_ids=stay_ids,
                columns=["noninvasivemean", "noninvasivesystolic", "noninvasivediastolic"],
            )
            va["hour"] = (pd.to_numeric(va["observationoffset"], errors="coerce") // 60).astype(
                "Int64"
            )
            va = va[(va["hour"] >= 0) & (va["hour"] < self._cfg.max_icu_hours)]
            va_agg = va.groupby(["patientunitstayid", "hour"], as_index=False).mean(
                numeric_only=True
            )
            agg = agg.merge(va_agg, on=["patientunitstayid", "hour"], how="outer")
            for feat, nib in [
                ("map", "noninvasivemean"),
                ("sbp", "noninvasivesystolic"),
                ("dbp", "noninvasivediastolic"),
            ]:
                if nib in agg.columns:
                    agg[feat] = agg[feat].fillna(agg[nib])
            agg = agg.drop(
                columns=[
                    c
                    for c in [
                        "noninvasivemean",
                        "noninvasivesystolic",
                        "noninvasivediastolic",
                        "observationoffset",
                    ]
                    if c in agg.columns
                ]
            )
        except FileNotFoundError:
            logger.warning("vitalAperiodic not found; skipping NIBP fallback")

        return agg

    def _hourly_labs(self, stay_ids: list[int]) -> pd.DataFrame:
        """Pivot long-format labs to a wide hourly table using the lab map."""
        # Read every lab row for the cohort; map names, warn on unmapped.
        # load_lab returns a freshly-built frame owned by this method, so it is
        # safe to mutate in place without an extra copy.
        lab = self._loader.load_lab(patient_unit_stay_ids=stay_ids)
        lab["feature"] = lab["labname"].map(map_labname)

        unmapped = sorted(
            lab.loc[lab["feature"].isna(), "labname"].astype(str).str.strip().unique()
        )
        if unmapped:
            logger.warning(
                "eICU lab: %d unmapped labname(s) ignored (extend eicu_lab_map "
                "to include them): %s",
                len(unmapped),
                unmapped[:25],
            )
        lab = lab.dropna(subset=["feature"])
        lab["hour"] = (pd.to_numeric(lab["labresultoffset"], errors="coerce") // 60).astype("Int64")
        lab = lab[(lab["hour"] >= 0) & (lab["hour"] < self._cfg.max_icu_hours)]
        lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")

        wide = lab.pivot_table(
            index=["patientunitstayid", "hour"],
            columns="feature",
            values="labresult",
            aggfunc="mean",
        ).reset_index()
        wide.columns.name = None
        return wide

    def _hourly_gcs(self, stay_ids: list[int]) -> pd.DataFrame:
        """Extract hourly GCS total from nurseCharting (category 'Scores')."""
        nc = self._loader.load_nurse_charting(patient_unit_stay_ids=stay_ids, categories=["Scores"])
        if nc.empty:
            return pd.DataFrame(columns=["patientunitstayid", "hour", "gcs_total"])

        label = nc["nursingchartcelltypevallabel"].astype(str)
        is_gcs = label.str.contains("Glasgow", case=False, na=False) | label.str.contains(
            "GCS", case=False, na=False
        )
        gcs = nc[is_gcs].copy()
        if gcs.empty:
            return pd.DataFrame(columns=["patientunitstayid", "hour", "gcs_total"])

        gcs["value"] = pd.to_numeric(gcs["nursingchartvalue"], errors="coerce")
        gcs["hour"] = (pd.to_numeric(gcs["nursingchartoffset"], errors="coerce") // 60).astype(
            "Int64"
        )
        gcs = gcs[(gcs["hour"] >= 0) & (gcs["hour"] < self._cfg.max_icu_hours)]

        valname = gcs["nursingchartcelltypevalname"].astype(str)
        total = gcs[valname.str.contains("Total", case=False, na=False)]
        if not total.empty:
            return total.groupby(["patientunitstayid", "hour"], as_index=False).agg(
                gcs_total=("value", "mean")
            )

        # Fall back to summing Eyes + Verbal + Motor components per (stay, hour).
        comp = gcs[valname.str.contains("Eyes|Verbal|Motor", case=False, na=False, regex=True)]
        return comp.groupby(["patientunitstayid", "hour"], as_index=False).agg(
            gcs_total=("value", "sum")
        )

    def _interval_flag(
        self,
        intervals: pd.DataFrame,
        start_col: str,
        end_col: str | None,
        flag_name: str,
    ) -> pd.DataFrame:
        """
        Expand offset intervals into per-(stay, hour) boolean flags.

        For each row, flag every hour in ``[start//60, end//60]`` (inclusive),
        clipped to ``[0, max_icu_hours)``. When ``end_col`` is missing/None the
        flag is open-ended to ``max_icu_hours``.
        """
        max_h = self._cfg.max_icu_hours
        rows: list[tuple[int, int]] = []
        starts = pd.to_numeric(intervals[start_col], errors="coerce") // 60
        if end_col is not None and end_col in intervals.columns:
            ends = pd.to_numeric(intervals[end_col], errors="coerce") // 60
        else:
            ends = pd.Series([np.nan] * len(intervals), index=intervals.index)
        for stay, s, e in zip(intervals["patientunitstayid"], starts, ends, strict=True):
            if pd.isna(s):
                continue
            s_h = max(0, int(s))
            e_h = max_h - 1 if pd.isna(e) else min(max_h - 1, int(e))
            for h in range(s_h, e_h + 1):
                rows.append((stay, h))
        if not rows:
            return pd.DataFrame(columns=["patientunitstayid", "hour", flag_name])
        out = pd.DataFrame(rows, columns=["patientunitstayid", "hour"]).drop_duplicates()
        out[flag_name] = 1
        return out

    def _hourly_vasopressors(self, stay_ids: list[int], med: pd.DataFrame) -> pd.DataFrame:
        """Per-(stay, hour) ``vasopressor_flag`` from the configured source."""
        if self._cfg.vasopressor_source == "medication_name":
            return self._legacy_vasopressor_orders(med)

        try:
            inf = self._loader.load_infusion_drug(patient_unit_stay_ids=stay_ids)
        except FileNotFoundError:
            logger.warning("infusionDrug not found; vasopressor flag uses medication orders")
            inf = pd.DataFrame(columns=["patientunitstayid"])

        # Hospitals that chart infusions use them exclusively; the rest fall back
        # to medication orders.
        cohort = self.build_cohort()
        charted = set(inf["patientunitstayid"])
        if "hospitalid" in cohort.columns:
            charting_hospitals = set(
                cohort.loc[cohort["patientunitstayid"].isin(charted), "hospitalid"]
            )
            uses_infusion = cohort["hospitalid"].isin(charting_hospitals)
        else:
            uses_infusion = cohort["patientunitstayid"].isin(charted)
        fallback = set(cohort.loc[~uses_infusion, "patientunitstayid"])
        if fallback:
            logger.info(
                "Vasopressors: %d stays at hospitals without infusion charting use "
                "medication orders",
                len(fallback),
            )
        frames = [
            self._vasopressor_infusions(inf),
            self._vasopressor_orders(med[med["patientunitstayid"].isin(fallback)]),
        ]
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame(columns=["patientunitstayid", "hour", "vasopressor_flag"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(["patientunitstayid", "hour"])

    def _vasopressor_infusions(self, inf: pd.DataFrame) -> pd.DataFrame:
        """Hours with a vasopressor infusion charted at a rate above zero."""
        if inf.empty or "drugname" not in inf.columns:
            return pd.DataFrame(columns=["patientunitstayid", "hour", "vasopressor_flag"])
        drug = _text(inf["drugname"]).str.extract(_VASOPRESSOR_NAME, expand=False)
        v = inf.assign(
            drug=drug.str.lower().replace(_VASOPRESSOR_BRANDS),
            rate=pd.to_numeric(inf["drugrate"], errors="coerce"),
            start=pd.to_numeric(inf["infusionoffset"], errors="coerce"),
        )
        v = v[v["drug"].notna() & v["start"].notna()].sort_values(
            ["patientunitstayid", "drug", "start"]
        )
        # A charting holds until the same drug's next charting, at most the cap.
        nxt = v.groupby(["patientunitstayid", "drug"])["start"].shift(-1)
        end = np.minimum(nxt.fillna(np.inf), v["start"] + _INFUSION_MAX_GAP_MINUTES)
        v = v.assign(stop=np.maximum(end - 1, v["start"]))
        running = v[v["rate"] > 0]
        return self._interval_flag(running, "start", "stop", "vasopressor_flag")

    def _vasopressor_orders(self, med: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback: vasopressor ``medication`` orders over their order interval.

        An order counts when its name is a vasopressor or, for the ~40% of rows
        without a name, its ``drughiclseqno`` is a code that only ever carries
        vasopressor names in this data. Cancelled, as-needed (PRN), one-time, and
        non-systemic orders are skipped.
        """
        if med.empty:
            return pd.DataFrame(columns=["patientunitstayid", "hour", "vasopressor_flag"])
        name = _text(med["drugname"])
        is_vaso_name = name.str.extract(_VASOPRESSOR_NAME, expand=False).notna() & ~(
            name.str.contains(_NOT_VASOPRESSOR_THERAPY)
        )
        code = med.get("drughiclseqno", pd.Series(np.nan, index=med.index))
        named = name.ne("")
        vaso_codes = set(code[named & is_vaso_name].dropna()) - set(
            code[named & ~is_vaso_name].dropna()
        )
        is_vaso = is_vaso_name | (~named & code.isin(vaso_codes))

        def text(col: str) -> pd.Series:
            if col not in med.columns:
                return pd.Series("", index=med.index)
            return _text(med[col]).str.strip().str.upper()

        given = (
            text("drugordercancelled").ne("YES")
            & text("prn").ne("YES")
            & ~text("frequency").str.contains(r"\bONCE\b", regex=True)
        )
        return self._interval_flag(
            med[is_vaso & given], "drugstartoffset", "drugstopoffset", "vasopressor_flag"
        )

    def _legacy_vasopressor_orders(self, med: pd.DataFrame) -> pd.DataFrame:
        """``vasopressor_source="medication_name"``: the clinops <= 0.6 rule."""
        if med.empty:
            return pd.DataFrame(columns=["patientunitstayid", "hour", "vasopressor_flag"])
        name = med["drugname"].astype(str).str.lower()
        mask = pd.Series(False, index=med.index)
        for kw in _VASOPRESSOR_KEYWORDS:
            mask |= name.str.contains(kw, na=False)
        return self._interval_flag(
            med[mask], "drugstartoffset", "drugstopoffset", "vasopressor_flag"
        )

    def _hourly_treatments(self, stay_ids: list[int]) -> pd.DataFrame:
        """Per-(stay, hour) vasopressor, mechanical-ventilation, and RRT flags."""
        frames: list[pd.DataFrame] = []

        med = self._loader.load_medication(patient_unit_stay_ids=stay_ids)
        frames.append(self._hourly_vasopressors(stay_ids, med))
        if not med.empty:
            name = med["drugname"].astype(str).str.lower()
            # RRT can also appear as a drug order.
            rrt_mask = pd.Series(False, index=med.index)
            for kw in _RRT_KEYWORDS:
                rrt_mask |= name.str.contains(kw, na=False)
            if rrt_mask.any():
                frames.append(
                    self._interval_flag(
                        med[rrt_mask], "drugstartoffset", "drugstopoffset", "rrt_flag"
                    )
                )

        # Mechanical ventilation from respiratoryCare vent offsets.
        try:
            rc = self._loader._load_small("respiratory_care")
            if "ventstartoffset" in rc.columns:
                vent = rc[pd.to_numeric(rc["ventstartoffset"], errors="coerce") > 0]
                frames.append(
                    self._interval_flag(
                        vent,
                        "ventstartoffset",
                        "ventendoffset",
                        "mechanical_ventilation_flag",
                    )
                )
        except FileNotFoundError:
            logger.warning("respiratoryCare not found; MV flag will be 0")

        # RRT from the treatment table.
        try:
            tr = self._loader._load_small("treatment")
            tr = tr[tr["patientunitstayid"].isin(set(stay_ids))]
            tstr = tr["treatmentstring"].astype(str).str.lower()
            rrt_mask = pd.Series(False, index=tr.index)
            for kw in _RRT_KEYWORDS:
                rrt_mask |= tstr.str.contains(kw, na=False)
            if rrt_mask.any():
                frames.append(
                    self._interval_flag(tr[rrt_mask], "treatmentoffset", None, "rrt_flag")
                )
        except FileNotFoundError:
            logger.warning("treatment table not found; RRT flag may be incomplete")

        if not frames:
            return pd.DataFrame(columns=["patientunitstayid", "hour", *_TREATMENT_FEATURES])

        out: pd.DataFrame | None = None
        for f in frames:
            out = f if out is None else out.merge(f, on=["patientunitstayid", "hour"], how="outer")
        assert out is not None
        # Collapse duplicate flag columns (e.g. rrt_flag from two sources).
        out = out.groupby(["patientunitstayid", "hour"], as_index=False).max()
        return out

    def _build_hourly(self) -> pd.DataFrame:
        """
        Assemble the dense hourly table with features + label-support columns.

        Returns a DataFrame with one row per (stay, hour) containing the 22
        canonical features, the label-only derived columns, and the five organ
        labels. Per-stay forward-fill densifies sparse measurements.
        """
        cohort = self.build_cohort()
        stay_ids = cohort["patientunitstayid"].astype(int).tolist()

        grid = self._hour_grid(cohort)
        vitals = self._hourly_vitals(stay_ids)
        labs = self._hourly_labs(stay_ids)
        gcs = self._hourly_gcs(stay_ids)
        treat = self._hourly_treatments(stay_ids)

        df = grid.merge(vitals, on=["patientunitstayid", "hour"], how="left")
        df = df.merge(labs, on=["patientunitstayid", "hour"], how="left")
        df = df.merge(gcs, on=["patientunitstayid", "hour"], how="left")
        df = df.merge(treat, on=["patientunitstayid", "hour"], how="left")

        # Guarantee every measured vital/lab column exists (a feature or
        # label-support variable never observed in this cohort becomes an
        # all-NaN column). This keeps the downstream label/derived logic free of
        # ``.get()`` / ``None`` handling and makes the schema deterministic.
        for col in (*_VITAL_FEATURES, *_LAB_FEATURES, "fio2"):
            if col not in df.columns:
                df[col] = np.nan

        # Treatment flags: absence means 0, not missing.
        for col in _TREATMENT_FEATURES:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)

        df = df.sort_values(["patientunitstayid", "hour"])

        # Critical-variable missingness is measured on the RAW grid, *before*
        # carry-forward — otherwise ffill would densify a sparse stay and hide
        # genuine gaps. Stash a per-stay value for _drop_high_missingness.
        critical = [
            c for c in ["heart_rate", "map", "resp_rate", "spo2", "creatinine"] if c in df.columns
        ]
        if critical:
            per_row_miss = df[critical].isna().mean(axis=1)
            stay_miss = per_row_miss.groupby(df["patientunitstayid"]).transform("mean")
            df["_raw_critical_missingness"] = stay_miss.to_numpy()
        else:
            df["_raw_critical_missingness"] = 0.0

        # Record which label-trigger variables were actually measured each hour,
        # before carry-forward erases the distinction (see label_mode).
        for var in _LABEL_TRIGGERS:
            if var in df.columns:
                df[f"_observed_{var}"] = df[var].notna()

        # Carry-forward measured values within each stay (clinical standard).
        measured = [c for c in (_VITAL_FEATURES + _LAB_FEATURES + ["fio2"]) if c in df.columns]
        df[measured] = df.groupby("patientunitstayid")[measured].ffill()

        # Stay-level constants.
        df = df.merge(
            cohort[["patientunitstayid", "apache_ii"]],
            on="patientunitstayid",
            how="left",
        )

        # Derived features.
        df["shock_index"] = df["heart_rate"] / df["sbp"].replace(0, np.nan)
        df["sofa"] = np.nan  # not pre-computed in eICU (documented gap)

        # Label-support derived columns (kept out of the feature matrix).
        df = self._add_label_support(df)
        df = self._add_labels(df)
        df = df.drop(columns=[c for c in df.columns if c.startswith("_observed_")])
        return df.reset_index(drop=True)

    def _add_label_support(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute baselines and PaO2/FiO2 ratio used only to derive labels.

        ``creatinine``, ``gcs_total``, ``pao2`` and ``fio2`` are guaranteed to
        exist by :meth:`_build_hourly` (as NaN columns if never observed).
        """
        # Per-stay baselines from the first ``baseline_hours`` of the stay:
        # lowest creatinine and best (max) GCS.
        early = df[df["hour"] < self._cfg.baseline_hours]
        base_cr = (
            early.groupby("patientunitstayid")["creatinine"].min().rename("baseline_creatinine")
        )
        base_gcs = early.groupby("patientunitstayid")["gcs_total"].max().rename("baseline_gcs")
        df = df.merge(base_cr, on="patientunitstayid", how="left")
        df = df.merge(base_gcs, on="patientunitstayid", how="left")

        # PaO2/FiO2 ratio. eICU FiO2 may be a percentage (40) or fraction (0.4).
        fio2_pct = np.where(df["fio2"] <= 1.0, df["fio2"] * 100.0, df["fio2"])
        with np.errstate(divide="ignore", invalid="ignore"):
            df["pao2_fio2_ratio"] = df["pao2"] * 100.0 / fio2_pct
        df["pao2_fio2_ratio"] = df["pao2_fio2_ratio"].replace([np.inf, -np.inf], np.nan)
        return df

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Physiology-only organ-deterioration labels (per stay-hour).

        With ``label_mode="observed"`` each criterion only fires in hours where
        its trigger variable was measured; see :class:`EicuCohortConfig`.
        """
        observed_only = self._cfg.label_mode == "observed"

        def seen(var: str) -> pd.Series[bool]:
            col = f"_observed_{var}"
            if observed_only and col in df.columns:
                return df[col].astype(bool)
            return pd.Series(True, index=df.index)

        cr = df["creatinine"]
        rr = df["resp_rate"]
        gcs = df["gcs_total"]

        df["cardiovascular"] = (
            (seen("map") & (df["map"] < 65)) | (seen("lactate") & (df["lactate"] > 2.0))
        ).astype(int)

        df["respiratory"] = (
            (seen("pao2") & (df["pao2_fio2_ratio"] < 300))
            | (seen("resp_rate") & ((rr > 30) | (rr < 8)))
            | (seen("spo2") & (df["spo2"] < 90))
        ).astype(int)

        df["renal"] = (
            seen("creatinine") & ((cr > 2.0) | (cr >= 1.5 * df["baseline_creatinine"]))
        ).astype(int)

        df["hepatic"] = (seen("bilirubin") & (df["bilirubin"] > 2.0)).astype(int)

        df["neurological"] = (
            seen("gcs_total") & ((gcs < 13) | ((df["baseline_gcs"] - gcs) >= 2))
        ).astype(int)
        return df

    # -- public builders ----------------------------------------------------

    def build_feature_matrix(self) -> pd.DataFrame:
        """
        Return the hourly 22-feature matrix, leakage-guarded.

        Raises
        ------
        LeakageError
            If any column on the programmatic exclusion list
            (:data:`LEAKAGE_FORBIDDEN`) is present in the returned feature set.

        Returns
        -------
        pd.DataFrame
            Columns: ``patientunitstayid``, ``hour``, then :data:`FEATURE_COLUMNS`.
        """
        hourly = self._build_hourly()
        hourly = self._drop_high_missingness(hourly)

        feature_cols = list(FEATURE_COLUMNS)
        leaked = LEAKAGE_FORBIDDEN & set(feature_cols)
        if leaked:
            raise LeakageError(
                f"Label-defining variable(s) {sorted(leaked)} found in the "
                f"feature matrix. These are derived only to compute labels and "
                f"must never be model inputs."
            )
        for col in feature_cols:
            if col not in hourly.columns:
                hourly[col] = np.nan
        return hourly[["patientunitstayid", "hour", *feature_cols]].reset_index(drop=True)

    def build_labels(self) -> pd.DataFrame:
        """
        Return the per-(stay, hour) organ-deterioration labels.

        Returns
        -------
        pd.DataFrame
            Columns: ``patientunitstayid``, ``hour``, then :data:`ORGAN_LABELS`.
        """
        hourly = self._build_hourly()
        hourly = self._drop_high_missingness(hourly)
        return hourly[["patientunitstayid", "hour", *ORGAN_LABELS]].reset_index(drop=True)

    def build_sequences(
        self,
        observation_hours: int | None = None,
        prediction_hours: int | None = None,
        max_icu_hours: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build sliding-window sequences for model training.

        For each stay, an hourly window of ``observation_hours`` produces one
        sample; its label is whether each organ deteriorates at any point in the
        following ``prediction_hours``. Windows slide hourly. Labels count only
        values measured within the horizon unless ``label_mode="locf"``; see
        :class:`EicuCohortConfig`.

        Parameters
        ----------
        observation_hours, prediction_hours, max_icu_hours:
            Override the configured values for this call.

        Returns
        -------
        (X, y) : tuple of np.ndarray
            ``X`` has shape ``(n_windows, observation_hours, 22)``;
            ``y`` has shape ``(n_windows, 5)`` (one binary label per organ).
        """
        # Apply per-call overrides for the duration of the build, then restore —
        # so a one-off call never silently mutates the loader's state, and the
        # overrides reach every consumer consistently (the grid via
        # max_icu_hours, the windowing via observation/prediction_hours, and the
        # baselines via baseline_hours, all read from self._cfg). Overrides are
        # routed through a fresh EicuCohortConfig so they are validated the same
        # way as the constructor (positive values; observation + prediction <=
        # max_icu_hours) rather than silently producing empty/invalid windows.
        overrides = {
            "observation_hours": observation_hours,
            "prediction_hours": prediction_hours,
            "max_icu_hours": max_icu_hours,
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if not overrides:
            return self._build_sequences_impl()

        saved_cfg = self._cfg
        self._cfg = EicuCohortConfig(**{**saved_cfg.model_dump(exclude_unset=True), **overrides})
        try:
            return self._build_sequences_impl()
        finally:
            self._cfg = saved_cfg

    def _build_sequences_impl(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self._cfg.observation_hours
        pred = self._cfg.prediction_hours

        hourly = self._build_hourly()
        hourly = self._drop_high_missingness(hourly)

        # Impute remaining gaps (post-ffill) with 0 so arrays are dense.
        feats = hourly[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float)
        labels = hourly[ORGAN_LABELS].to_numpy(dtype=int)
        hourly = hourly.reset_index(drop=True)

        x_windows: list[np.ndarray] = []
        y_windows: list[np.ndarray] = []
        for _, idx in hourly.groupby("patientunitstayid").groups.items():
            pos = np.asarray(idx)
            n = len(pos)
            # Need obs hours of features + pred hours of forward label horizon.
            for start in range(0, n - obs - pred + 1):
                win = pos[start : start + obs]
                horizon = pos[start + obs : start + obs + pred]
                x_windows.append(feats[win])
                y_windows.append(labels[horizon].max(axis=0))

        if not x_windows:
            logger.warning(
                "build_sequences produced 0 windows (cohort too short for obs=%d + pred=%d)",
                obs,
                pred,
            )
            return (
                np.empty((0, obs, len(FEATURE_COLUMNS))),
                np.empty((0, len(ORGAN_LABELS))),
            )

        x_arr = np.stack(x_windows)
        y_arr = np.stack(y_windows)
        logger.info("build_sequences: X=%s, y=%s", x_arr.shape, y_arr.shape)
        return x_arr, y_arr

    def _drop_high_missingness(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Exclude stays whose critical-variable missingness exceeds threshold.

        Uses the pre-forward-fill missingness computed in :meth:`_build_hourly`
        (``_raw_critical_missingness``); the helper column is dropped here so it
        never reaches the feature matrix.
        """
        if "_raw_critical_missingness" not in hourly.columns:
            return hourly
        miss = hourly.groupby("patientunitstayid")["_raw_critical_missingness"].first()
        keep = set(miss[miss <= self._cfg.missingness_threshold].index)
        before = hourly["patientunitstayid"].nunique()
        out = hourly[hourly["patientunitstayid"].isin(keep)].drop(
            columns=["_raw_critical_missingness"]
        )
        logger.info(
            "Missingness filter (<=%.0f%%): kept %d/%d stays",
            self._cfg.missingness_threshold * 100,
            len(keep),
            before,
        )
        return out.reset_index(drop=True)


__all__ = [
    "EicuConfig",
    "EicuCohortConfig",
    "EicuLoader",
    "EicuTableLoader",
    "FEATURE_COLUMNS",
    "ORGAN_LABELS",
    "LEAKAGE_FORBIDDEN",
    "parse_eicu_age",
    "canonical_lab_features",
    "map_labname",
]
