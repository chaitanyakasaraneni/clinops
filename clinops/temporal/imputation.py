"""
Gap-aware imputation strategies for clinical time-series data.

Clinical data has distinctive missingness patterns — labs are drawn
infrequently, vitals may flatline during device disconnects, and some
features are only measured once per admission. Standard ML imputation
often fails silently on these patterns. This module provides strategies
tuned for clinical context.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

import numpy as np
import pandas as pd
from loguru import logger


class ImputationStrategy(StrEnum):
    """
    Supported imputation strategies.

    FORWARD_FILL
        Carry the last observed value forward in time. Appropriate for
        slowly-changing vitals (heart rate, SpO2) where repeated
        measurements are assumed stable until updated.

    BACKWARD_FILL
        Fill from the next observed value backward. Useful when a
        measurement is known to have been taken but not yet recorded.

    LINEAR
        Linear interpolation between surrounding observations. Use for
        continuous physiological signals with regular sampling.

    MEAN
        Replace missing values with the column mean (global, per patient,
        or per cohort depending on ``fit`` scope).

    MEDIAN
        Replace with column median. More robust than mean for skewed
        lab values.

    ZERO
        Fill with zero. Use only for count-based features where absence
        genuinely means zero (e.g. number of interventions).

    INDICATOR
        Add a binary missingness indicator column (``{col}_missing``)
        and fill values with zero. Lets the model learn from
        missingness patterns directly.

    NONE
        Do not impute — leave NaN values in place.
    """

    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR = "linear"
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"
    INDICATOR = "indicator"
    NONE = "none"


class Imputer:
    """
    Apply a chosen imputation strategy to a DataFrame.

    Parameters
    ----------
    strategy:
        Imputation strategy to apply.
    max_gap_hours:
        For FORWARD_FILL and BACKWARD_FILL: maximum gap (in hours) to
        fill across. Gaps larger than this are left as NaN to avoid
        propagating stale values across long time periods. Requires a
        ``time_col`` to be set.
    time_col:
        Name of the datetime column (used with max_gap_hours).
    per_patient:
        If True and strategy is MEAN/MEDIAN, compute statistics per
        patient group rather than globally.
    id_col:
        Patient identifier column (required when per_patient=True).
    """

    def __init__(
        self,
        strategy: ImputationStrategy = ImputationStrategy.FORWARD_FILL,
        max_gap_hours: float | None = None,
        time_col: str | None = None,
        per_patient: bool = False,
        id_col: str | None = None,
    ) -> None:
        self.strategy = strategy
        self.max_gap_hours = max_gap_hours
        self.time_col = time_col
        self.per_patient = per_patient
        self.id_col = id_col
        self._fill_values: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> Imputer:
        """Compute imputation statistics from a reference DataFrame (training set)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.strategy == ImputationStrategy.MEAN:
            if self.per_patient and self.id_col:
                # store per-patient means — used in transform
                self._patient_means = df.groupby(self.id_col)[numeric_cols].mean()
            else:
                self._fill_values = {str(k): float(v) for k, v in df[numeric_cols].mean().items()}
        elif self.strategy == ImputationStrategy.MEDIAN:
            if self.per_patient and self.id_col:
                self._patient_medians = df.groupby(self.id_col)[numeric_cols].median()
            else:
                self._fill_values = {str(k): float(v) for k, v in df[numeric_cols].median().items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to df. Call fit() first for MEAN/MEDIAN strategies."""
        df = df.copy()
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        if self.strategy == ImputationStrategy.NONE:
            return df

        elif self.strategy == ImputationStrategy.ZERO:
            df[numeric_cols] = df[numeric_cols].fillna(0.0)

        elif self.strategy == ImputationStrategy.FORWARD_FILL:
            if self.max_gap_hours is not None and self.time_col and self.time_col in df.columns:
                # Use a UUID-based sentinel to avoid clobbering any user column
                # and to guarantee uniqueness. try/finally ensures the column is
                # always removed, even if an exception is raised mid-transform.
                _sentinel = f"__clinops_pos_{uuid.uuid4().hex}__"
                try:
                    df[_sentinel] = np.arange(len(df))
                    df = df.sort_values(self.time_col).reset_index(drop=True)
                    original_nulls = df[numeric_cols].isna()
                    df[numeric_cols] = df[numeric_cols].ffill()
                    df = self._mask_large_gaps(
                        df, numeric_cols, forward=True, original_nulls=original_nulls
                    )
                    df = df.sort_values(_sentinel).reset_index(drop=True)
                finally:
                    df = df.drop(columns=[_sentinel], errors="ignore")
            else:
                df[numeric_cols] = df[numeric_cols].ffill()

        elif self.strategy == ImputationStrategy.BACKWARD_FILL:
            if self.max_gap_hours is not None and self.time_col and self.time_col in df.columns:
                # Use a UUID-based sentinel to avoid clobbering any user column
                # and to guarantee uniqueness. try/finally ensures the column is
                # always removed, even if an exception is raised mid-transform.
                _sentinel = f"__clinops_pos_{uuid.uuid4().hex}__"
                try:
                    df[_sentinel] = np.arange(len(df))
                    df = df.sort_values(self.time_col).reset_index(drop=True)
                    original_nulls = df[numeric_cols].isna()
                    df[numeric_cols] = df[numeric_cols].bfill()
                    df = self._mask_large_gaps(
                        df, numeric_cols, forward=False, original_nulls=original_nulls
                    )
                    df = df.sort_values(_sentinel).reset_index(drop=True)
                finally:
                    df = df.drop(columns=[_sentinel], errors="ignore")
            else:
                df[numeric_cols] = df[numeric_cols].bfill()

        elif self.strategy == ImputationStrategy.LINEAR:
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

        elif self.strategy == ImputationStrategy.MEAN:
            if self._fill_values:
                df[numeric_cols] = df[numeric_cols].fillna(self._fill_values)
            else:
                logger.warning("Imputer not fitted — using column means from current df")
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        elif self.strategy == ImputationStrategy.MEDIAN:
            if self._fill_values:
                df[numeric_cols] = df[numeric_cols].fillna(self._fill_values)
            else:
                logger.warning("Imputer not fitted — using column medians from current df")
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        elif self.strategy == ImputationStrategy.INDICATOR:
            for col in numeric_cols:
                df[f"{col}_missing"] = df[col].isna().astype(int)
            df[numeric_cols] = df[numeric_cols].fillna(0.0)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit then transform."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mask_large_gaps(
        self, df: pd.DataFrame, numeric_cols: list[str], forward: bool,
        original_nulls: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Re-introduce NaN where the gap between actual observations exceeds
        max_gap_hours, preventing stale carry-forward over long intervals.

        Callers are expected to sort by time_col and reset_index before
        calling this method, and to capture original_nulls after that sort
        so that row indices are aligned.

        Raises
        ------
        ValueError
            If time_col is not set, not present in df, or max_gap_hours is None.
        """
        if self.max_gap_hours is None:
            raise ValueError("_mask_large_gaps called with max_gap_hours=None")
        if not self.time_col:
            raise ValueError("_mask_large_gaps called without time_col set")
        if self.time_col not in df.columns:
            raise ValueError(
                f"_mask_large_gaps: time_col '{self.time_col}' not found in DataFrame"
            )
        time = pd.to_datetime(df[self.time_col])
        gap_hours = time.diff().dt.total_seconds() / 3600

        for col in numeric_cols:
            # Use pre-fill nulls when provided; fall back to current column state
            was_null_before_fill = (
                original_nulls[col] if original_nulls is not None else df[col].isna()
            )
            max_gap = self.max_gap_hours
            large_gap = gap_hours > max_gap if forward else gap_hours.shift(-1).fillna(0) > max_gap
            df.loc[was_null_before_fill & large_gap, col] = np.nan

        return df
