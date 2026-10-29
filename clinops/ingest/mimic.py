"""
MIMIC-IV data loader with table-level validation and lazy loading.

Supports MIMIC-IV v2.0, v2.1, and v2.2. Handles both the hosp and icu
modules. Validates column presence, dtypes, and value ranges on load.

Example
-------
>>> from clinops.ingest import MimicLoader  # doctest: +SKIP
>>> loader = MimicLoader("/data/mimic-iv-2.2")  # doctest: +SKIP
>>> charts = loader.chartevents(subject_ids=[10000032, 10000980])  # doctest: +SKIP
>>> labs   = loader.labevents(subject_ids=[10000032, 10000980])  # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from clinops.ingest.schema import SchemaValidationError

logger = logging.getLogger(__name__)

# Tables large enough that chunk_size applies.
_CHUNKED_TABLES = frozenset({"chartevents", "labevents"})

# ---------------------------------------------------------------------------
# Table schemas — minimum required columns per MIMIC-IV table
# ---------------------------------------------------------------------------

_REQUIRED_COLS: dict[str, list[str]] = {
    "chartevents": ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"],
    "labevents": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
    "prescriptions": ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "dose_val_rx"],
    "inputevents": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "amount",
        "amountuom",
    ],
    "d_items": ["itemid", "label", "category"],
    "d_labitems": ["itemid", "label", "fluid", "category"],
    "icustays": ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
}

_TABLE_PATHS: dict[str, tuple[str, str]] = {
    # table_name: (module, filename_stem)
    "chartevents": ("icu", "chartevents"),
    "labevents": ("hosp", "labevents"),
    "admissions": ("hosp", "admissions"),
    "patients": ("hosp", "patients"),
    "prescriptions": ("hosp", "prescriptions"),
    "inputevents": ("icu", "inputevents"),
    "d_items": ("icu", "d_items"),
    "d_labitems": ("hosp", "d_labitems"),
    "icustays": ("icu", "icustays"),
}


@dataclass
class MimicLoaderConfig:
    """Configuration for MimicLoader."""

    mimic_path: Path
    version: Literal["2.2", "2.1", "2.0", "auto"] = "auto"
    strict_validation: bool = True
    chunk_size: int | None = None  # set for chunked reading on large tables

    def __post_init__(self) -> None:
        if not self.mimic_path.exists():
            raise ValueError(f"MIMIC path does not exist: {self.mimic_path}")


class MimicLoader:
    """
    Loader for MIMIC-IV clinical database tables.

    Parameters
    ----------
    mimic_path:
        Root directory of the MIMIC-IV dataset. Should contain
        ``hosp/`` and ``icu/`` subdirectories.
    version:
        MIMIC-IV version string (e.g. ``"2.2"``). Use ``"auto"`` to
        detect from the directory structure.
    strict_validation:
        If True (default), raise ``SchemaValidationError`` when required
        columns are missing. If False, log a warning and continue.
    chunk_size:
        If set, ``chartevents`` and ``labevents`` CSV files are read in chunks
        of this many rows, and the requested filters are applied to each chunk
        before the next is read. Only matching rows are kept in memory, so
        pass ``subject_ids``, ``item_ids`` or a time range to benefit.
        Parquet files are always read in full.

    Examples
    --------
    >>> loader = MimicLoader("/data/mimic-iv-2.2")  # doctest: +SKIP
    >>> charts = loader.chartevents(subject_ids=[10000032])  # doctest: +SKIP
    >>> labs   = loader.labevents(hadm_ids=[20000019])  # doctest: +SKIP
    """

    def __init__(
        self,
        mimic_path: str | Path,
        version: Literal["2.2", "2.1", "2.0", "auto"] = "auto",
        strict_validation: bool = True,
        chunk_size: int | None = None,
    ) -> None:
        self._cfg = MimicLoaderConfig(
            mimic_path=Path(mimic_path),
            version=version,
            strict_validation=strict_validation,
            chunk_size=chunk_size,
        )
        self._version = self._detect_version() if version == "auto" else version
        logger.info(
            f"MimicLoader initialised — path={self._cfg.mimic_path}, version={self._version}"
        )

    # ------------------------------------------------------------------
    # Public table accessors
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
        Load ICU charted observations (vitals, GCS, ventilator settings, etc.).

        Parameters
        ----------
        subject_ids:
            Filter to these patients.
        hadm_ids:
            Filter to these hospital admissions.
        stay_ids:
            Filter to these ICU stays.
        item_ids:
            Filter to these MIMIC-IV itemids (see d_items).
        start_time:
            ISO datetime string — exclude rows before this time.
        end_time:
            ISO datetime string — exclude rows after this time.
        """
        return self._load_table(
            "chartevents",
            lambda df: self._filter(
                df, subject_ids, hadm_ids, stay_ids, item_ids, start_time, end_time
            ),
        )

    def labevents(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        """Load hospital laboratory results."""
        return self._load_table(
            "labevents",
            lambda df: self._filter(
                df, subject_ids, hadm_ids, None, item_ids, start_time, end_time
            ),
        )

    def admissions(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Load hospital admission records."""
        df = self._load_table("admissions")
        return self._filter(df, subject_ids, hadm_ids)

    def patients(
        self,
        subject_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Load patient demographics."""
        df = self._load_table("patients")
        return self._filter(df, subject_ids)

    def icustays(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        stay_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Load ICU stay metadata including LOS."""
        df = self._load_table("icustays")
        return self._filter(df, subject_ids, hadm_ids, stay_ids)

    def prescriptions(
        self,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        drugs: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Load medication prescriptions."""
        df = self._load_table("prescriptions")
        df = self._filter(df, subject_ids, hadm_ids)
        if drugs:
            df = df[df["drug"].str.lower().isin([d.lower() for d in drugs])]
        return df

    def inputevents(
        self,
        subject_ids: Sequence[int] | None = None,
        stay_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Load ICU fluid input events."""
        df = self._load_table("inputevents")
        return self._filter(df, subject_ids, None, stay_ids)

    def d_items(self) -> pd.DataFrame:
        """Load ICU item dictionary (maps itemid → label)."""
        return self._load_table("d_items")

    def d_labitems(self) -> pd.DataFrame:
        """Load lab item dictionary (maps itemid → label)."""
        return self._load_table("d_labitems")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_version(self) -> str:
        """Attempt to infer MIMIC-IV version from directory name or files."""
        path_str = str(self._cfg.mimic_path).lower()
        for v in ["2.2", "2.1", "2.0"]:
            if v in path_str:
                return v
        logger.warning("Could not detect MIMIC-IV version from path; assuming 2.2")
        return "2.2"

    def _resolve_table_path(self, table_name: str) -> Path:
        """Resolve the filesystem path for a MIMIC table, supporting csv and csv.gz."""
        module, stem = _TABLE_PATHS[table_name]
        base = self._cfg.mimic_path / module
        for ext in [".csv.gz", ".csv", ".parquet"]:
            candidate = base / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Table '{table_name}' not found under {base}. "
            f"Expected one of: {stem}.csv.gz, {stem}.csv, {stem}.parquet"
        )

    def _load_table(
        self,
        table_name: str,
        row_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Load a table, validate its schema, parse datetimes, and apply ``row_filter``.

        With ``chunk_size`` set, large CSV tables are read chunk by chunk and
        ``row_filter`` is applied to each chunk, so only matching rows are kept.
        """
        path = self._resolve_table_path(table_name)
        logger.debug(f"Loading {table_name} from {path}")

        chunk_size = self._cfg.chunk_size
        if chunk_size and path.suffix != ".parquet" and table_name in _CHUNKED_TABLES:
            parts = []
            with pd.read_csv(path, low_memory=False, chunksize=chunk_size) as reader:
                for i, chunk in enumerate(reader):
                    if i == 0:
                        self._validate_schema(table_name, chunk)
                    chunk = self._parse_datetimes(table_name, chunk)
                    parts.append(row_filter(chunk) if row_filter else chunk)
            if parts:
                df = pd.concat(parts, ignore_index=True)
            else:
                # Header-only file: no chunks are yielded.
                df = pd.read_csv(path, nrows=0)
                self._validate_schema(table_name, df)
        else:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, low_memory=False)
            self._validate_schema(table_name, df)
            df = self._parse_datetimes(table_name, df)
            if row_filter:
                df = row_filter(df)

        logger.debug(f"Loaded {table_name}: {len(df):,} rows, {len(df.columns)} cols")
        return df

    def _validate_schema(self, table_name: str, df: pd.DataFrame) -> None:
        required = _REQUIRED_COLS.get(table_name, [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            msg = f"Table '{table_name}' missing required columns: {missing}"
            if self._cfg.strict_validation:
                raise SchemaValidationError(msg)
            logger.warning(msg)

    def _parse_datetimes(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        time_cols = [c for c in df.columns if "time" in c or c == "dod"]
        for col in time_cols:
            # is_string_dtype covers both object (pandas 2) and the default
            # "str" dtype that pandas 3 infers for text columns read from CSV.
            if col in df.columns and pd.api.types.is_string_dtype(df[col].dtype):
                with contextlib.suppress(Exception):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def _filter(
        self,
        df: pd.DataFrame,
        subject_ids: Sequence[int] | None = None,
        hadm_ids: Sequence[int] | None = None,
        stay_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        if subject_ids is not None and "subject_id" in df.columns:
            df = df[df["subject_id"].isin(subject_ids)]
        if hadm_ids is not None and "hadm_id" in df.columns:
            df = df[df["hadm_id"].isin(hadm_ids)]
        if stay_ids is not None and "stay_id" in df.columns:
            df = df[df["stay_id"].isin(stay_ids)]
        if item_ids is not None and "itemid" in df.columns:
            df = df[df["itemid"].isin(item_ids)]

        time_col = next(
            (c for c in ["charttime", "starttime", "admittime"] if c in df.columns), None
        )
        if time_col:
            if start_time:
                df = df[df[time_col] >= pd.Timestamp(start_time)]
            if end_time:
                df = df[df[time_col] <= pd.Timestamp(end_time)]

        return df.reset_index(drop=True)
