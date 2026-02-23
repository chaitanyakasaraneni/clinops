"""
Lag feature construction and cohort alignment utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class LagFeatureBuilder:
    """
    Add lag and rolling-statistics features to windowed clinical data.

    Parameters
    ----------
    lags:
        List of lag steps (in window units) to include.
        e.g. [1, 2, 4] adds features at t-1, t-2, t-4 windows back.
    rolling_windows:
        List of rolling window sizes to compute mean/std over.
    feature_cols:
        Columns to create lags for. If None, all numeric columns are used.
    id_col:
        Patient identifier column — lags are computed within each patient.

    Examples
    --------
    >>> builder = LagFeatureBuilder(lags=[1, 2], rolling_windows=[4])
    >>> enriched = builder.fit_transform(windows_df, id_col="subject_id")
    """

    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        feature_cols: list[str] | None = None,
        id_col: str = "subject_id",
    ) -> None:
        self.lags = lags or [1, 2]
        self.rolling_windows = rolling_windows or []
        self.feature_cols = feature_cols
        self.id_col = id_col

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag and rolling features. Returns enriched DataFrame."""
        df = df.copy().sort_values([self.id_col, "window_start"])
        numeric_cols = self.feature_cols or [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in [self.id_col, "label"]
        ]

        logger.info(
            f"Building lag features: lags={self.lags}, "
            f"rolling={self.rolling_windows}, cols={len(numeric_cols)}"
        )

        for col in numeric_cols:
            # Lag features
            for lag in self.lags:
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = df.groupby(self.id_col)[col].shift(lag)

            # Rolling statistics
            for window in self.rolling_windows:
                grouped = df.groupby(self.id_col)[col]
                df[f"{col}_roll{window}_mean"] = grouped.transform(
                    lambda s, w=window: s.rolling(w, min_periods=1).mean()
                )
                df[f"{col}_roll{window}_std"] = grouped.transform(
                    lambda s, w=window: s.rolling(w, min_periods=1).std()
                )

        logger.info(f"Added {len(df.columns) - len(df.columns):,} lag/rolling features")
        return df


class CohortAligner:
    """
    Align multiple patients' time-series to a common reference event.

    In clinical research it's common to align patients relative to an
    anchor event (e.g. ICU admission, ventilation start, first sepsis
    flag) rather than using wall-clock time. This class handles the
    realignment so downstream models see time-relative-to-event rather
    than absolute timestamps.

    Parameters
    ----------
    anchor_col:
        Column containing the anchor event timestamp for each patient.
    id_col:
        Patient identifier column.
    max_hours_before:
        Include data up to this many hours before the anchor event.
    max_hours_after:
        Include data up to this many hours after the anchor event.

    Examples
    --------
    >>> aligner = CohortAligner(anchor_col="icu_intime", max_hours_after=48)
    >>> aligned = aligner.align(chartevents, admissions)
    """

    def __init__(
        self,
        anchor_col: str = "icu_intime",
        id_col: str = "subject_id",
        max_hours_before: float = 0.0,
        max_hours_after: float = 48.0,
        time_col: str = "charttime",
    ) -> None:
        self.anchor_col = anchor_col
        self.id_col = id_col
        self.max_hours_before = max_hours_before
        self.max_hours_after = max_hours_after
        self.time_col = time_col

    def align(
        self,
        events_df: pd.DataFrame,
        anchor_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align events to anchor timestamps from a reference DataFrame.

        Parameters
        ----------
        events_df:
            Long-format events with id_col and time_col.
        anchor_df:
            One row per patient with id_col and anchor_col.

        Returns
        -------
        pd.DataFrame
            events_df filtered to the alignment window with a new
            ``hours_from_anchor`` column (negative = before anchor).
        """
        anchor_map = anchor_df.set_index(self.id_col)[self.anchor_col]
        anchor_map = pd.to_datetime(anchor_map)

        df = events_df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])

        anchors = df[self.id_col].map(anchor_map)
        df["hours_from_anchor"] = (df[self.time_col] - anchors).dt.total_seconds() / 3600

        df = df[
            (df["hours_from_anchor"] >= -self.max_hours_before) &
            (df["hours_from_anchor"] <= self.max_hours_after)
        ]
        df = df.drop(columns=[], errors="ignore").reset_index(drop=True)

        logger.info(
            f"CohortAligner: retained {len(df):,} rows "
            f"(window: -{self.max_hours_before}h to +{self.max_hours_after}h from anchor)"
        )
        return df
