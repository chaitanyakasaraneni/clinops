"""
Flat file loader for CSV and Parquet clinical data exports.

Handles common messy-data patterns: mixed datetime formats, inconsistent
null representations, duplicate rows, and configurable schema validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from clinops.ingest.schema import ClinicalSchema

_NULL_VALUES = ["", "NA", "N/A", "n/a", "nan", "NaN", "NULL", "null", "None", "UNKNOWN", "."]


class FlatFileLoader:
    """
    Load clinical data from CSV or Parquet flat files with validation.

    Parameters
    ----------
    path:
        Path to a CSV (.csv, .csv.gz) or Parquet (.parquet, .pq) file.
    schema:
        Optional ClinicalSchema for validation after loading.
    id_col:
        Name of the patient/subject identifier column. Used for
        deduplication reporting.
    datetime_cols:
        Column names to parse as datetimes. If None, auto-detection
        is attempted for columns with "time", "date", or "dt" in the name.
    strict:
        If True, raise on schema violations. If False, warn and continue.

    Examples
    --------
    >>> loader = FlatFileLoader("vitals_export.csv", id_col="patient_id")
    >>> df = loader.load()
    >>> print(loader.summary())
    """

    def __init__(
        self,
        path: str | Path,
        schema: ClinicalSchema | None = None,
        id_col: str | None = None,
        datetime_cols: list[str] | None = None,
        strict: bool = True,
    ) -> None:
        self._path = Path(path)
        self._schema = schema
        self._id_col = id_col
        self._datetime_cols = datetime_cols
        self._strict = strict
        self._loaded_df: pd.DataFrame | None = None

        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")

    def load(self) -> pd.DataFrame:
        """
        Load the file, apply cleaning and validation, return DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        logger.info(f"Loading flat file: {self._path}")
        df = self._read_file()
        df = self._clean(df)
        df = self._parse_datetimes(df)

        if self._schema:
            violations = self._schema.validate(df, strict=self._strict)
            if violations and not self._strict:
                for v in violations:
                    logger.warning(v)

        self._loaded_df = df
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} cols from {self._path.name}")
        return df

    def summary(self) -> str:
        """Return a human-readable summary of the loaded DataFrame."""
        if self._loaded_df is None:
            return "No data loaded yet. Call .load() first."
        df = self._loaded_df
        null_counts = df.isna().sum()
        null_summary = null_counts[null_counts > 0].to_dict()
        lines = [
            f"File:        {self._path.name}",
            f"Rows:        {len(df):,}",
            f"Columns:     {len(df.columns)}",
            f"Dtypes:      {df.dtypes.value_counts().to_dict()}",
            f"Null cols:   {null_summary if null_summary else 'None'}",
        ]
        if self._id_col and self._id_col in df.columns:
            lines.append(f"Unique IDs:  {df[self._id_col].nunique():,}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_file(self) -> pd.DataFrame:
        suffix = "".join(self._path.suffixes).lower()
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(self._path)
        return pd.read_csv(
            self._path,
            low_memory=False,
            na_values=_NULL_VALUES,
            keep_default_na=True,
        )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalise column names: strip whitespace, lowercase, replace spaces
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # Drop fully empty columns and rows
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")
        return df.reset_index(drop=True)

    def _parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._datetime_cols:
            target_cols = self._datetime_cols
        else:
            # Auto-detect: columns with "time", "date", or "dt" in name
            target_cols = [
                c for c in df.columns if any(kw in c for kw in ["time", "date", "_dt", "ts", "_at"])
            ]

        for col in target_cols:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    # Only replace if conversion was meaningful (>50% non-null)
                    if converted.notna().mean() > 0.5:
                        df[col] = converted
                except Exception:  # noqa: BLE001
                    logger.debug(f"Could not parse datetime column: {col}")
        return df
