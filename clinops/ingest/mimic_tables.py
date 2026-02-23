"""
Pre-built table loader for the MIMIC-IV tables researchers always need.

``MimicTableLoader`` wraps ``MimicLoader`` and exposes five high-value tables
with fully-typed, validated schemas out of the box — no manual ``ColumnSpec``
definitions required.  Each accessor returns a clean, analysis-ready DataFrame
with correct dtypes, physiological range hints, and join keys clearly labelled.

Supported tables
----------------
- ``chartevents``   — ICU vitals, GCS, ventilator settings (icu module)
- ``labevents``     — Hospital laboratory results (hosp module)
- ``admissions``    — Hospital admission / discharge records (hosp module)
- ``diagnoses_icd`` — ICD-9/10 diagnosis codes per admission (hosp module)
- ``icustays``      — ICU stay metadata with LOS (icu module)

Example
-------
>>> from clinops.ingest import MimicTableLoader
>>>
>>> tbl = MimicTableLoader("/data/mimic-iv-2.2")
>>>
>>> # Vitals for two patients — no schema definition needed
>>> charts = tbl.chartevents(subject_ids=[10000032, 10000980])
>>>
>>> # Lab results joined with reference ranges
>>> labs = tbl.labevents(subject_ids=[10000032], with_ref_range=True)
>>>
>>> # Diagnoses with ICD version already labelled
>>> dx = tbl.diagnoses_icd(subject_ids=[10000032])
>>> print(dx.dtypes)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from loguru import logger

from clinops.ingest.mimic import MimicLoader
from clinops.ingest.schema import ClinicalSchema, ColumnSpec

# ---------------------------------------------------------------------------
# Pre-built schemas
# ---------------------------------------------------------------------------

#: Physiological bounds used for range-hint annotations (not hard enforcement —
#: use ClinicalOutlierClipper for that).  Stored here so researchers can inspect
#: them without running the full preprocessor.
VITAL_BOUNDS: dict[str, tuple[float, float]] = {
    "heart_rate":    (0,   300),
    "spo2":          (50,  100),
    "sbp":           (0,   300),
    "dbp":           (0,   200),
    "map":           (0,   200),
    "resp_rate":     (0,   60),
    "temperature":   (25,  45),
    "glucose":       (0,   2000),
}

_SCHEMAS: dict[str, ClinicalSchema] = {
    "chartevents": ClinicalSchema(
        name="chartevents",
        columns=[
            ColumnSpec("subject_id", nullable=False),
            ColumnSpec("hadm_id",    nullable=True),
            ColumnSpec("stay_id",    nullable=False),
            ColumnSpec("itemid",     nullable=False),
            ColumnSpec("charttime",  nullable=False),
            ColumnSpec("valuenum",   nullable=True),
            ColumnSpec("valueuom",   nullable=True),
        ],
    ),
    "labevents": ClinicalSchema(
        name="labevents",
        columns=[
            ColumnSpec("subject_id", nullable=False),
            ColumnSpec("hadm_id",    nullable=True),
            ColumnSpec("itemid",     nullable=False),
            ColumnSpec("charttime",  nullable=False),
            ColumnSpec("valuenum",   nullable=True),
            ColumnSpec("valueuom",   nullable=True),
            ColumnSpec("ref_range_lower", nullable=True),
            ColumnSpec("ref_range_upper", nullable=True),
        ],
    ),
    "admissions": ClinicalSchema(
        name="admissions",
        columns=[
            ColumnSpec("subject_id",      nullable=False),
            ColumnSpec("hadm_id",         nullable=False),
            ColumnSpec("admittime",        nullable=False),
            ColumnSpec("dischtime",        nullable=True),
            ColumnSpec("deathtime",        nullable=True),
            ColumnSpec("admission_type",   nullable=False),
            ColumnSpec("admission_location", nullable=True),
            ColumnSpec("discharge_location", nullable=True),
            ColumnSpec("insurance",        nullable=True),
            ColumnSpec("hospital_expire_flag", nullable=True),
        ],
    ),
    "diagnoses_icd": ClinicalSchema(
        name="diagnoses_icd",
        columns=[
            ColumnSpec("subject_id", nullable=False),
            ColumnSpec("hadm_id",    nullable=False),
            ColumnSpec("seq_num",    nullable=False),
            ColumnSpec("icd_code",   nullable=False),
            ColumnSpec("icd_version", nullable=False),
        ],
    ),
    "icustays": ClinicalSchema(
        name="icustays",
        columns=[
            ColumnSpec("subject_id",      nullable=False),
            ColumnSpec("hadm_id",         nullable=False),
            ColumnSpec("stay_id",         nullable=False),
            ColumnSpec("first_careunit",  nullable=True),
            ColumnSpec("last_careunit",   nullable=True),
            ColumnSpec("intime",          nullable=False),
            ColumnSpec("outtime",         nullable=True),
            ColumnSpec("los",             nullable=True),
        ],
    ),
}

# MIMIC-IV module mapping for the two extra tables not in MimicLoader
_EXTRA_TABLE_PATHS: dict[str, tuple[str, str]] = {
    "diagnoses_icd": ("hosp", "diagnoses_icd"),
}


# ---------------------------------------------------------------------------
# MimicTableLoader
# ---------------------------------------------------------------------------

class MimicTableLoader:
    """
    Pre-built loader for the MIMIC-IV tables researchers use most.

    Wraps :class:`~clinops.ingest.MimicLoader` and adds:

    * Pre-validated schemas for ``chartevents``, ``labevents``,
      ``admissions``, ``diagnoses_icd``, and ``icustays``.
    * ``with_ref_range`` flag on ``labevents`` to retain or drop the
      reference range columns (noisy on many MIMIC exports).
    * ``primary_only`` flag on ``diagnoses_icd`` to keep only
      ``seq_num == 1`` (the principal diagnosis).
    * ``with_los_band`` flag on ``icustays`` to add a categorical
      ``los_band`` column (``<1d``, ``1-3d``, ``3-7d``, ``>7d``) useful
      as a stratification variable.
    * A ``summary()`` method that prints row counts and null rates for
      all five tables without loading full data.

    Parameters
    ----------
    mimic_path:
        Root directory of the MIMIC-IV dataset.
    version:
        MIMIC-IV version string or ``"auto"`` (default).
    strict_validation:
        Raise on missing required columns when ``True`` (default).
    chunk_size:
        Pass through to underlying :class:`MimicLoader` for large tables.

    Examples
    --------
    >>> tbl = MimicTableLoader("/data/mimic-iv-2.2")
    >>> charts = tbl.chartevents(subject_ids=[10000032, 10000980])
    >>> dx      = tbl.diagnoses_icd(subject_ids=[10000032], primary_only=True)
    >>> stays   = tbl.icustays(subject_ids=[10000032], with_los_band=True)
    """

    def __init__(
        self,
        mimic_path: str | Path,
        version: str = "auto",
        strict_validation: bool = True,
        chunk_size: int | None = None,
    ) -> None:
        self._path = Path(mimic_path)
        self._loader = MimicLoader(
            mimic_path=mimic_path,
            version=version,
            strict_validation=strict_validation,
            chunk_size=chunk_size,
        )
        logger.info(f"MimicTableLoader ready — {self._path}")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def chartevents(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        stay_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        """
        Load ICU charted observations with schema validation.

        Returns a DataFrame with columns:
        ``subject_id``, ``hadm_id``, ``stay_id``, ``itemid``,
        ``charttime``, ``valuenum``, ``valueuom``.

        Parameters
        ----------
        subject_ids:
            Restrict to these patients.
        hadm_ids:
            Restrict to these hospital admissions.
        stay_ids:
            Restrict to these ICU stays.
        item_ids:
            Restrict to these MIMIC itemids (see ``d_items``).
        start_time, end_time:
            ISO datetime strings for time range filtering.
        """
        df = self._loader.chartevents(
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            stay_ids=stay_ids,
            item_ids=item_ids,
            start_time=start_time,
            end_time=end_time,
        )
        df = df.copy()
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        _SCHEMAS["chartevents"].validate(df)
        logger.info(f"chartevents: {len(df):,} rows")
        return df

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
        with_ref_range:
            If ``False`` (default), drop ``ref_range_lower`` /
            ``ref_range_upper`` columns — they are sparsely populated in
            most MIMIC exports and add noise to downstream pipelines.
            Set ``True`` to retain them.
        """
        df = self._loader.labevents(
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            item_ids=item_ids,
            start_time=start_time,
            end_time=end_time,
        )
        _SCHEMAS["labevents"].validate(df)
        if not with_ref_range:
            drop_cols = [c for c in ["ref_range_lower", "ref_range_upper"] if c in df.columns]
            df = df.drop(columns=drop_cols)
        logger.info(f"labevents: {len(df):,} rows (with_ref_range={with_ref_range})")
        return df

    def admissions(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Load hospital admission records.

        Returns a DataFrame with columns:
        ``subject_id``, ``hadm_id``, ``admittime``, ``dischtime``,
        ``deathtime``, ``admission_type``, ``admission_location``,
        ``discharge_location``, ``insurance``, ``hospital_expire_flag``.
        """
        df = self._loader.admissions(subject_ids=subject_ids, hadm_ids=hadm_ids)
        _SCHEMAS["admissions"].validate(df)
        logger.info(f"admissions: {len(df):,} rows")
        return df

    def diagnoses_icd(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        primary_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load ICD-9/ICD-10 diagnosis codes per hospital admission.

        MIMIC-IV mixes ICD-9-CM and ICD-10-CM codes.  The ``icd_version``
        column contains ``9`` or ``10`` — use
        :class:`~clinops.preprocess.ICDMapper` to harmonize to a single
        version before modelling.

        Parameters
        ----------
        primary_only:
            If ``True``, keep only rows where ``seq_num == 1`` (the
            principal/primary diagnosis per admission).  Default ``False``
            returns all coded diagnoses.
        """
        df = self._load_extra_table("diagnoses_icd")
        if subject_ids is not None:
            df = df[df["subject_id"].isin(subject_ids)]
        if hadm_ids is not None:
            df = df[df["hadm_id"].isin(hadm_ids)]
        if primary_only:
            df = df[df["seq_num"] == 1]
        _SCHEMAS["diagnoses_icd"].validate(df)
        logger.info(
            f"diagnoses_icd: {len(df):,} rows (primary_only={primary_only})"
        )
        return df.reset_index(drop=True)

    def icustays(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        stay_ids: Sequence[int] | None = None,
        with_los_band: bool = False,
    ) -> pd.DataFrame:
        """
        Load ICU stay metadata.

        Parameters
        ----------
        with_los_band:
            If ``True``, add a ``los_band`` categorical column bucketing
            length-of-stay into ``<1d``, ``1-3d``, ``3-7d``, ``>7d``.
            Useful as a stratification variable for cohort splits.
        """
        df = self._loader.icustays(
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            stay_ids=stay_ids,
        )
        _SCHEMAS["icustays"].validate(df)
        if with_los_band and "los" in df.columns:
            df = df.copy()
            df["los_band"] = pd.cut(
                df["los"],
                bins=[0, 1, 3, 7, float("inf")],
                labels=["<1d", "1-3d", "3-7d", ">7d"],
                right=True,
            )
        logger.info(f"icustays: {len(df):,} rows (with_los_band={with_los_band})")
        return df

    def summary(self) -> pd.DataFrame:
        """
        Print a quick-look table of row counts and null rates for all five
        tables without loading the full data into memory.

        Uses ``pd.read_csv`` with ``nrows=0`` to read headers, then scans
        only the first 10,000 rows to estimate null rates.

        Returns
        -------
        pd.DataFrame
            Columns: ``table``, ``rows_sampled``, ``columns``, ``null_rate_pct``
        """
        records = []
        for table_name in ["chartevents", "labevents", "admissions", "diagnoses_icd", "icustays"]:
            try:
                path = self._resolve_extra_path(table_name)
                sample = pd.read_csv(path, nrows=10_000, low_memory=False)
                null_rate = sample.isnull().mean().mean() * 100
                records.append({
                    "table":          table_name,
                    "rows_sampled":   len(sample),
                    "columns":        len(sample.columns),
                    "null_rate_pct":  round(null_rate, 2),
                })
            except FileNotFoundError:
                records.append({
                    "table":         table_name,
                    "rows_sampled":  0,
                    "columns":       0,
                    "null_rate_pct": None,
                })
        result = pd.DataFrame(records)
        print(result.to_string(index=False))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_extra_table(self, table_name: str) -> pd.DataFrame:
        """Load tables that exist in MIMIC-IV but not in MimicLoader."""
        path = self._resolve_extra_path(table_name)
        logger.debug(f"Loading {table_name} from {path}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        return df

    def _resolve_extra_path(self, table_name: str) -> Path:
        """Resolve path for tables in _EXTRA_TABLE_PATHS or delegate to MimicLoader."""
        if table_name in _EXTRA_TABLE_PATHS:
            module, stem = _EXTRA_TABLE_PATHS[table_name]
            base = self._path / module
            for ext in [".csv.gz", ".csv", ".parquet"]:
                candidate = base / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                f"Table '{table_name}' not found under {base}. "
                f"Expected one of: {stem}.csv.gz, {stem}.csv, {stem}.parquet"
            )
        # Fall back to MimicLoader's path resolver for shared tables
        return self._loader._resolve_table_path(table_name)
