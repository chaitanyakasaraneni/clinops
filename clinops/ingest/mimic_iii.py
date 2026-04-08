"""
MIMIC-III clinical database loader.

``MimicIIILoader`` provides the same interface as ``MimicLoader`` (MIMIC-IV)
but targets the MIMIC-III schema, which differs in three important ways:

1. **Flat directory structure** — all tables live in a single directory,
   no ``hosp/`` or ``icu/`` subdirectories.
2. **Uppercase column names** — ``SUBJECT_ID``, ``HADM_ID``, ``ICUSTAY_ID``,
   ``CHARTTIME``, etc. Normalised to lowercase on load so downstream code
   can treat MIMIC-III and MIMIC-IV DataFrames identically.
3. **ICD-9-CM only** — no ICD-10 codes; ``icd_version`` column is always 9.

This loader is intended for researchers benchmarking against the large body
of existing MIMIC-III literature (MIMIC-Extract, DeepPatient, AIM-III, etc.)
who want to reuse clinops preprocessing pipelines without rewriting filters.

Supported tables
----------------
- ``chartevents``    — ICU charted observations (vitals, GCS, vents)
- ``labevents``      — Hospital lab results
- ``admissions``     — Hospital admission / discharge records
- ``patients``       — Patient-level demographics
- ``diagnoses_icd``  — ICD-9-CM diagnosis codes (seq_num + icd9_code)
- ``icustays``       — ICU stay metadata including LOS
- ``prescriptions``  — Medication orders
- ``inputevents_mv`` — MetaVision fluid inputs (CareVue: ``inputevents_cv``)
- ``d_items``        — Item dictionary (maps itemid → label)
- ``d_labitems``     — Lab item dictionary

Examples
--------
>>> from clinops.ingest import MimicIIILoader
>>>
>>> loader = MimicIIILoader("/data/mimic-iii-clinical-database-1.4")
>>>
>>> charts = loader.chartevents(subject_ids=[40124, 41914])
>>> labs   = loader.labevents(subject_ids=[40124], with_ref_range=True)
>>> dx     = loader.diagnoses_icd(subject_ids=[40124], primary_only=True)

Authors
-------
Chaitanya Kasaraneni

References
----------
Johnson, A.E.W. et al. MIMIC-III, a freely accessible critical care database.
Sci Data 3, 160035 (2016). https://doi.org/10.1038/sdata.2016.35
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator

from clinops.ingest.schema import SchemaValidationError

# ---------------------------------------------------------------------------
# Column name normalisation
# ---------------------------------------------------------------------------


#: MIMIC-III uses uppercase column names; normalise to lowercase so DataFrames
#: are compatible with MIMIC-IV loaders and downstream clinops transforms.
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Table registry
# ---------------------------------------------------------------------------

#: All MIMIC-III tables live in a flat directory (no module subdirectories).
_TABLE_STEMS: dict[str, str] = {
    "chartevents": "CHARTEVENTS",
    "labevents": "LABEVENTS",
    "admissions": "ADMISSIONS",
    "diagnoses_icd": "DIAGNOSES_ICD",
    "icustays": "ICUSTAYS",
    "prescriptions": "PRESCRIPTIONS",
    "inputevents_mv": "INPUTEVENTS_MV",
    "inputevents_cv": "INPUTEVENTS_CV",
    "d_items": "D_ITEMS",
    "d_labitems": "D_LABITEMS",
    "patients": "PATIENTS",
}

#: Required columns per table (post-normalisation, lowercase).
_REQUIRED_COLS: dict[str, list[str]] = {
    "chartevents": ["subject_id", "hadm_id", "icustay_id", "itemid", "charttime", "valuenum"],
    "labevents": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
    "diagnoses_icd": ["subject_id", "hadm_id", "seq_num", "icd9_code"],
    "icustays": ["subject_id", "hadm_id", "icustay_id", "intime", "outtime", "los"],
    "prescriptions": ["subject_id", "hadm_id", "startdate", "drug"],
    "inputevents_mv": [
        "subject_id",
        "hadm_id",
        "icustay_id",
        "itemid",
        "starttime",
        "amount",
        "amountuom",
    ],
}

#: Datetime columns per table (post-normalisation, lowercase).
_DATETIME_COLS: dict[str, list[str]] = {
    "chartevents": ["charttime"],
    "labevents": ["charttime"],
    "admissions": ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    "diagnoses_icd": [],
    "icustays": ["intime", "outtime"],
    "prescriptions": ["startdate", "enddate"],
    "inputevents_mv": ["starttime", "endtime"],
    "inputevents_cv": ["charttime"],
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class MimicIIIConfig(BaseModel):
    """Validated configuration for MimicIIILoader."""

    mimic_path: Path
    strict_validation: bool = True
    chunk_size: int | None = None

    @field_validator("mimic_path")
    @classmethod
    def path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"MIMIC-III path does not exist: {v}")
        return v


# ---------------------------------------------------------------------------
# MimicIIILoader
# ---------------------------------------------------------------------------


class MimicIIILoader:
    """
    Loader for the MIMIC-III Clinical Database.

    Provides the same filtering interface as ``MimicLoader`` (MIMIC-IV) so
    that analysis code can be reused across both datasets with minimal changes.

    Key differences from MIMIC-IV:
    - Flat directory — no ``hosp/`` / ``icu/`` split.
    - Uppercase column names in source files — normalised to lowercase on load.
    - ICU stay key is ``icustay_id`` (not ``stay_id``).
    - ICD-9-CM only — ``diagnoses_icd`` has ``icd9_code``, not ``icd_code``.
    - ``inputevents`` is split into ``inputevents_mv`` (MetaVision) and
      ``inputevents_cv`` (CareVue); use :meth:`inputevents` to get both merged.

    Parameters
    ----------
    mimic_path:
        Root directory of the MIMIC-III dataset. Should contain files like
        ``CHARTEVENTS.csv.gz``, ``ADMISSIONS.csv.gz``, etc.
    strict_validation:
        If ``True`` (default), raise :exc:`SchemaValidationError` when
        required columns are missing. If ``False``, log a warning and continue.
    chunk_size:
        If set, large tables (``chartevents``, ``labevents``) return a chunked
        reader instead of loading fully into memory.

    Examples
    --------
    >>> loader = MimicIIILoader("/data/mimic-iii-clinical-database-1.4")
    >>> charts = loader.chartevents(subject_ids=[40124])
    >>> labs   = loader.labevents(hadm_ids=[198765])
    >>> dx     = loader.diagnoses_icd(subject_ids=[40124], primary_only=True)
    """

    def __init__(
        self,
        mimic_path: str | Path,
        strict_validation: bool = True,
        chunk_size: int | None = None,
    ) -> None:
        self._cfg = MimicIIIConfig(
            mimic_path=Path(mimic_path),
            strict_validation=strict_validation,
            chunk_size=chunk_size,
        )
        logger.info(f"MimicIIILoader initialised — path={self._cfg.mimic_path}")

    # ------------------------------------------------------------------
    # Public table accessors
    # ------------------------------------------------------------------

    def chartevents(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        icustay_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        """
        Load ICU charted observations.

        Parameters
        ----------
        subject_ids:
            Restrict to these patient IDs.
        hadm_ids:
            Restrict to these hospital admission IDs.
        icustay_ids:
            Restrict to these ICU stay IDs (``icustay_id`` in MIMIC-III,
            equivalent to ``stay_id`` in MIMIC-IV).
        item_ids:
            Restrict to these ``itemid`` values. See :meth:`d_items`.
        start_time:
            Exclude rows with ``charttime`` before this ISO datetime string.
        end_time:
            Exclude rows with ``charttime`` after this ISO datetime string.

        Returns
        -------
        pd.DataFrame
            Columns (lowercase): ``subject_id``, ``hadm_id``,
            ``icustay_id``, ``itemid``, ``charttime`` (datetime),
            ``valuenum``, ``valueuom``.
        """
        df = self._load("chartevents")
        return self._filter(
            df,
            subject_ids,
            hadm_ids,
            icustay_ids=icustay_ids,
            item_ids=item_ids,
            start_time=start_time,
            end_time=end_time,
            time_col="charttime",
        )

    def labevents(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        with_ref_range: bool = False,
    ) -> pd.DataFrame:
        """
        Load hospital laboratory results.

        Parameters
        ----------
        subject_ids, hadm_ids, item_ids, start_time, end_time:
            Standard filters — see :meth:`chartevents`.
        with_ref_range:
            If ``True``, retain ``ref_range_lower`` and ``ref_range_upper``.
            Dropped by default to reduce memory footprint.

        Returns
        -------
        pd.DataFrame
            Columns: ``subject_id``, ``hadm_id``, ``itemid``,
            ``charttime`` (datetime), ``valuenum``, ``valueuom``.
        """
        df = self._load("labevents")
        df = self._filter(
            df,
            subject_ids,
            hadm_ids,
            item_ids=item_ids,
            start_time=start_time,
            end_time=end_time,
            time_col="charttime",
        )
        if not with_ref_range:
            drop = [c for c in ["ref_range_lower", "ref_range_upper"] if c in df.columns]
            df = df.drop(columns=drop)
        return df

    def admissions(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Load hospital admission and discharge records.

        Returns
        -------
        pd.DataFrame
            Columns: ``subject_id``, ``hadm_id``, ``admittime``,
            ``dischtime``, ``deathtime``, ``admission_type``, and others.
        """
        df = self._load("admissions")
        return self._filter(df, subject_ids, hadm_ids)

    def diagnoses_icd(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        icd9_codes: Sequence[str] | None = None,
        primary_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load ICD-9-CM diagnosis codes.

        MIMIC-III uses ICD-9-CM exclusively. The column name is ``icd9_code``
        (not ``icd_code`` as in MIMIC-IV). For cross-dataset compatibility,
        a synthetic ``icd_version`` column (always 9) is added on load.

        Parameters
        ----------
        subject_ids, hadm_ids:
            Standard filters.
        icd9_codes:
            Restrict to these ICD-9-CM codes (exact match,
            case-insensitive).
        primary_only:
            If ``True``, return only the primary diagnosis (``seq_num == 1``)
            per admission.

        Returns
        -------
        pd.DataFrame
            Columns: ``subject_id``, ``hadm_id``, ``seq_num``,
            ``icd9_code``, ``icd_version`` (always 9).
        """
        df = self._load("diagnoses_icd")
        df = self._filter(df, subject_ids, hadm_ids)

        if icd9_codes is not None:
            if "icd9_code" in df.columns:
                df = df[df["icd9_code"].str.upper().isin([c.upper() for c in icd9_codes])]
            else:
                logger.warning(
                    "Column 'icd9_code' missing from diagnoses_icd; skipping icd9_codes filter."
                )

        if primary_only:
            if "seq_num" in df.columns:
                df = df[df["seq_num"] == 1]
            else:
                logger.warning(
                    "Column 'seq_num' missing from diagnoses_icd; "
                    "cannot restrict to primary diagnoses."
                )

        # Add synthetic icd_version for cross-dataset compatibility
        if "icd_version" not in df.columns:
            df = df.assign(icd_version=9)

        return df.reset_index(drop=True)

    def icustays(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        icustay_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Load ICU stay metadata including length of stay.

        Note: The ICU stay key in MIMIC-III is ``icustay_id``, not
        ``stay_id`` as in MIMIC-IV.

        Returns
        -------
        pd.DataFrame
            Columns: ``subject_id``, ``hadm_id``, ``icustay_id``,
            ``first_careunit``, ``last_careunit``, ``intime`` (datetime),
            ``outtime`` (datetime), ``los`` (days, float).
        """
        df = self._load("icustays")
        return self._filter(df, subject_ids, hadm_ids, icustay_ids=icustay_ids)

    def prescriptions(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        drugs: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load medication prescriptions.

        Parameters
        ----------
        drugs:
            Restrict to these drug names (case-insensitive substring match).

        Returns
        -------
        pd.DataFrame
            Columns: ``subject_id``, ``hadm_id``, ``startdate``,
            ``enddate``, ``drug``, ``dose_val_rx``, ``dose_unit_rx``.
        """
        df = self._load("prescriptions")
        df = self._filter(df, subject_ids, hadm_ids)
        if drugs is not None:
            pattern = "|".join(drugs)
            df = df[df["drug"].str.contains(pattern, case=False, na=False)]
        return df

    def inputevents(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        icustay_ids: Sequence[int] | None = None,
        source: str = "mv",
    ) -> pd.DataFrame:
        """
        Load ICU fluid input events.

        MIMIC-III stores MetaVision (``INPUTEVENTS_MV``) and CareVue
        (``INPUTEVENTS_CV``) inputs in separate tables.

        Parameters
        ----------
        source:
            ``"mv"`` (MetaVision, default), ``"cv"`` (CareVue), or
            ``"both"`` to load and concatenate both tables.

        Returns
        -------
        pd.DataFrame
            For MetaVision: ``subject_id``, ``hadm_id``, ``icustay_id``,
            ``itemid``, ``starttime``, ``amount``, ``amountuom``.
            When ``source="both"``, also includes ``event_time`` (unified
            timestamp) and ``source`` (``"mv"`` or ``"cv"``).
        """
        if source not in {"mv", "cv", "both"}:
            raise ValueError(f"source must be 'mv', 'cv', or 'both' — got {source!r}")

        dfs: list[pd.DataFrame] = []
        if source in {"mv", "both"}:
            df_mv = self._load("inputevents_mv")
            dfs.append(
                self._filter(
                    df_mv,
                    subject_ids,
                    hadm_ids,
                    icustay_ids=icustay_ids,
                    time_col="starttime",
                )
            )
        if source in {"cv", "both"}:
            df_cv = self._load("inputevents_cv")
            dfs.append(
                self._filter(
                    df_cv,
                    subject_ids,
                    hadm_ids,
                    icustay_ids=icustay_ids,
                    time_col="charttime",
                )
            )

        # When combining MetaVision and CareVue input events, normalise to a
        # common timestamp column and record the source system so callers have
        # a consistent schema regardless of the underlying table.
        if len(dfs) > 1:
            normalised: list[pd.DataFrame] = []
            for df in dfs:
                df_norm = df.copy()
                if "starttime" in df_norm.columns:
                    df_norm["event_time"] = df_norm["starttime"]
                    df_norm["source"] = "mv"
                elif "charttime" in df_norm.columns:
                    df_norm["event_time"] = df_norm["charttime"]
                    df_norm["source"] = "cv"
                else:
                    # Fallback: no recognised time column; keep schema
                    # consistent but leave event_time/source missing.
                    df_norm["event_time"] = pd.NaT
                    df_norm["source"] = pd.NA
                normalised.append(df_norm)
            result = pd.concat(normalised, ignore_index=True)
        else:
            result = dfs[0]
        return result

    def patients(
        self,
        subject_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Load patient demographics.

        Returns
        -------
        pd.DataFrame
            Columns: ``subject_id``, ``gender``, ``dob``, ``dod``,
            ``dod_hosp``, ``dod_ssn``, ``expire_flag``.
        """
        df = self._load("patients")
        return self._filter(df, subject_ids)

    def d_items(self) -> pd.DataFrame:
        """
        Load the item dictionary (``itemid`` → label).

        Use this to resolve ``itemid`` values in :meth:`chartevents`.

        Returns
        -------
        pd.DataFrame
            Columns: ``itemid``, ``label``, ``abbreviation``,
            ``dbsource``, ``linksto``, ``category``, ``unitname``.
        """
        return self._load("d_items")

    def d_labitems(self) -> pd.DataFrame:
        """
        Load the lab item dictionary (``itemid`` → label).

        Use this to resolve ``itemid`` values in :meth:`labevents`.

        Returns
        -------
        pd.DataFrame
            Columns: ``itemid``, ``label``, ``fluid``, ``category``,
            ``loinc_code``.
        """
        return self._load("d_labitems")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, table_name: str) -> Path:
        """Find the filesystem path for a MIMIC-III table."""
        stem = _TABLE_STEMS[table_name]
        for ext in (".csv.gz", ".csv", ".parquet"):
            candidate = self._cfg.mimic_path / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Table '{table_name}' not found in {self._cfg.mimic_path}. "
            f"Expected one of: {stem}.csv.gz, {stem}.csv, {stem}.parquet"
        )

    def _load(self, table_name: str) -> pd.DataFrame:
        """Load, normalise columns, parse datetimes, validate schema."""
        path = self._resolve_path(table_name)
        logger.debug(f"Loading MIMIC-III {table_name} from {path}")

        read_kwargs: dict = {}
        if self._cfg.chunk_size and table_name in {"chartevents", "labevents"}:
            read_kwargs["chunksize"] = self._cfg.chunk_size

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif self._cfg.chunk_size and table_name in {"chartevents", "labevents"}:
            chunks = pd.read_csv(path, low_memory=False, chunksize=self._cfg.chunk_size)
            processed = []
            for chunk in chunks:
                chunk = _normalise_columns(chunk)
                chunk = self._parse_datetimes(chunk, table_name)
                self._validate_schema(chunk, table_name)
                processed.append(chunk)
            df = pd.concat(processed, ignore_index=True)
        else:
            df = pd.read_csv(path, low_memory=False)

        if isinstance(df, pd.DataFrame) and not (
            self._cfg.chunk_size and table_name in {"chartevents", "labevents"}
        ):
            df = _normalise_columns(df)
            if table_name == "diagnoses_icd" and "icd9_code" in df.columns:
                df["icd9_code"] = df["icd9_code"].astype(str)
            df = self._parse_datetimes(df, table_name)
            self._validate_schema(df, table_name)

        return df

    def _parse_datetimes(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        for col in _DATETIME_COLS.get(table_name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def _validate_schema(self, df: pd.DataFrame, table_name: str) -> None:
        required = _REQUIRED_COLS.get(table_name, [])
        missing = [c for c in required if c not in df.columns]
        if not missing:
            return
        msg = f"MIMIC-III table '{table_name}' missing required columns: {missing}"
        if self._cfg.strict_validation:
            raise SchemaValidationError(msg)
        logger.warning(msg)

    def _filter(
        self,
        df: pd.DataFrame,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        icustay_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        time_col: str | None = None,
    ) -> pd.DataFrame:
        if subject_ids is not None and "subject_id" in df.columns:
            df = df[df["subject_id"].isin(subject_ids)]
        if hadm_ids is not None and "hadm_id" in df.columns:
            df = df[df["hadm_id"].isin(hadm_ids)]
        if icustay_ids is not None and "icustay_id" in df.columns:
            df = df[df["icustay_id"].isin(icustay_ids)]
        if item_ids is not None and "itemid" in df.columns:
            df = df[df["itemid"].isin(item_ids)]

        col = time_col or next(
            (c for c in ["charttime", "starttime", "admittime"] if c in df.columns),
            None,
        )
        if col and col in df.columns:
            if start_time:
                df = df[df[col] >= pd.Timestamp(start_time)]
            if end_time:
                df = df[df[col] <= pd.Timestamp(end_time)]

        return df.reset_index(drop=True)
