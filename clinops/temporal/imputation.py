"""
Gap-aware imputation strategies for clinical time-series data.

Clinical data has distinctive missingness patterns — labs are drawn
infrequently, vitals may flatline during device disconnects, and some
features are only measured once per admission. Standard ML imputation
often fails silently on these patterns. This module provides strategies
tuned for clinical context.
"""

from __future__ import annotations

import logging
import uuid
from enum import StrEnum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
        For FORWARD_FILL and BACKWARD_FILL: fill a missing value only if
        the observation it is filled from is at most this many hours away,
        measured per column from the last (or next) actual observation.
        Requires ``time_col``.
    time_col:
        Name of the datetime column (used with max_gap_hours).
    per_patient:
        If True and strategy is MEAN/MEDIAN, ``fit`` learns per-patient
        statistics. ``transform`` fills each patient from their own fitted
        statistic, falling back to the global fitted statistic for patients
        not seen in ``fit``. Requires ``id_col``.
    id_col:
        Patient identifier column. FORWARD_FILL, BACKWARD_FILL, and LINEAR
        are applied within each patient when set. Without it, the whole
        DataFrame is treated as one series, so set it on multi-patient data.

    Examples
    --------
    Fit per-patient means on the training split and apply them to the test
    split. Patient 3 was not in training, so it gets the training-set mean:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from clinops.temporal import ImputationStrategy, Imputer
    >>> train = pd.DataFrame({"subject_id": [1, 1, 2, 2], "heart_rate": [60.0, 80.0, 100.0, 100.0]})
    >>> test = pd.DataFrame({"subject_id": [1, 3], "heart_rate": [np.nan, np.nan]})
    >>> imputer = Imputer(ImputationStrategy.MEAN, per_patient=True, id_col="subject_id")
    >>> imputer.fit(train).transform(test)["heart_rate"].tolist()
    [70.0, 85.0]

    Forward fill within each patient, but never across a gap longer than 2 hours:

    >>> df = pd.DataFrame({
    ...     "subject_id": 1,
    ...     "charttime": pd.date_range("2150-01-01", periods=5, freq="h"),
    ...     "lactate": [4.0, np.nan, np.nan, np.nan, np.nan],
    ... })
    >>> Imputer(
    ...     ImputationStrategy.FORWARD_FILL,
    ...     max_gap_hours=2,
    ...     time_col="charttime",
    ...     id_col="subject_id",
    ... ).transform(df)["lactate"].tolist()
    [4.0, 4.0, 4.0, nan, nan]
    """

    def __init__(
        self,
        strategy: ImputationStrategy = ImputationStrategy.FORWARD_FILL,
        max_gap_hours: float | None = None,
        time_col: str | None = None,
        per_patient: bool = False,
        id_col: str | None = None,
    ) -> None:
        if max_gap_hours is not None and not time_col:
            raise ValueError("max_gap_hours requires time_col to be set")
        if per_patient and not id_col:
            raise ValueError("per_patient=True requires id_col to be set")
        self.strategy = strategy
        self.max_gap_hours = max_gap_hours
        self.time_col = time_col
        self.per_patient = per_patient
        self.id_col = id_col
        self._fitted = False
        self._fill_values: dict[str, float] = {}
        self._patient_stats: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> Imputer:
        """Compute imputation statistics from a reference DataFrame (training set)."""
        if self.strategy in (ImputationStrategy.MEAN, ImputationStrategy.MEDIAN):
            self._fill_values, self._patient_stats = self._compute_stats(df)
            # Kept for backwards compatibility with code that inspects these attributes.
            if self.strategy == ImputationStrategy.MEAN:
                self._patient_means = self._patient_stats
            else:
                self._patient_medians = self._patient_stats
            self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to df. Call fit() first for MEAN/MEDIAN strategies."""
        df = df.copy()
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != self.id_col
        ]
        has_id = bool(self.id_col) and self.id_col in df.columns

        if self.strategy == ImputationStrategy.NONE:
            return df

        elif self.strategy == ImputationStrategy.ZERO:
            df[numeric_cols] = df[numeric_cols].fillna(0.0)

        elif self.strategy in (ImputationStrategy.FORWARD_FILL, ImputationStrategy.BACKWARD_FILL):
            forward = self.strategy == ImputationStrategy.FORWARD_FILL
            if self.max_gap_hours is not None:
                assert self.time_col is not None  # enforced in __init__
                if self.time_col not in df.columns:
                    raise ValueError(
                        f"time_col {self.time_col!r} not found; it is required for max_gap_hours"
                    )
                # Use a UUID-based sentinel to avoid clobbering any user column
                # and to guarantee uniqueness. try/finally ensures the column is
                # always removed, even if an exception is raised mid-transform.
                _sentinel = f"__clinops_pos_{uuid.uuid4().hex}__"
                try:
                    df[_sentinel] = np.arange(len(df))
                    df = self._fill_with_gap_mask(df, numeric_cols, self.time_col, forward)
                    df = df.sort_values(_sentinel).reset_index(drop=True)
                finally:
                    df = df.drop(columns=[_sentinel], errors="ignore")
            elif has_id:
                grouped = df.groupby(self.id_col, sort=False, dropna=False)[numeric_cols]
                df[numeric_cols] = grouped.ffill() if forward else grouped.bfill()
            else:
                df[numeric_cols] = df[numeric_cols].ffill() if forward else df[numeric_cols].bfill()

        elif self.strategy == ImputationStrategy.LINEAR:
            if has_id:
                # Interpolate within each entity so no value is drawn from a neighbouring patient.
                df[numeric_cols] = df.groupby(self.id_col, sort=False, dropna=False)[
                    numeric_cols
                ].transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            else:
                df[numeric_cols] = df[numeric_cols].interpolate(
                    method="linear", limit_direction="both"
                )

        elif self.strategy in (ImputationStrategy.MEAN, ImputationStrategy.MEDIAN):
            if self._fitted:
                fill_values, patient_stats = self._fill_values, self._patient_stats
            else:
                logger.warning(
                    f"Imputer not fitted; using column {self.strategy.value}s from the current "
                    "df. Call fit() on training data to avoid leakage."
                )
                fill_values, patient_stats = self._compute_stats(df)
            for col in numeric_cols:
                series = df[col]
                if patient_stats is not None and has_id and col in patient_stats.columns:
                    series = series.fillna(df[self.id_col].map(patient_stats[col]))
                # Patients unseen at fit time (e.g. a held-out test set) fall back to the
                # global statistic from the fitted data.
                if col in fill_values:
                    series = series.fillna(fill_values[col])
                df[col] = series

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

    def _compute_stats(self, df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame | None]:
        """Return global per-column statistics and, if per_patient, per-entity statistics."""
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != self.id_col
        ]
        stat = "mean" if self.strategy == ImputationStrategy.MEAN else "median"
        global_stats = df[numeric_cols].agg(stat)
        fill_values = {str(k): float(v) for k, v in global_stats.items() if pd.notna(v)}
        patient_stats = None
        if self.per_patient and self.id_col and self.id_col in df.columns:
            patient_stats = df.groupby(self.id_col)[numeric_cols].agg(stat)
        return fill_values, patient_stats

    def _fill_with_gap_mask(
        self, df: pd.DataFrame, numeric_cols: list[str], time_col: str, forward: bool
    ) -> pd.DataFrame:
        """
        Apply ffill/bfill, filling a missing value only if the observation it
        is filled from is at most ``max_gap_hours`` away in time.

        The distance is measured from the last (or next) actual observation of
        that column, not from the neighbouring row, so dense rows from other
        features cannot extend a stale value. When ``id_col`` is set, fill and
        masking are applied within each entity group.
        """
        assert self.max_gap_hours is not None
        id_col = self.id_col if self.id_col and self.id_col in df.columns else None
        sort_cols = [id_col, time_col] if id_col is not None else [time_col]
        df = df.sort_values(sort_cols, kind="stable")
        time = pd.to_datetime(df[time_col])
        limit = pd.Timedelta(hours=self.max_gap_hours)
        keys = df[id_col] if id_col is not None else pd.Series(0, index=df.index)

        for col in numeric_cols:
            observed = df[col].notna()
            obs_time = time.where(observed)
            if forward:
                filled = df[col].groupby(keys, dropna=False).ffill()
                source_time = obs_time.groupby(keys, dropna=False).ffill()
                distance = time - source_time
            else:
                filled = df[col].groupby(keys, dropna=False).bfill()
                source_time = obs_time.groupby(keys, dropna=False).bfill()
                distance = source_time - time
            df[col] = filled.where(observed | (distance <= limit))

        return df
