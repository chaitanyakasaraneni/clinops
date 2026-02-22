"""
Sliding and tumbling window extraction for clinical time-series data.

Handles the most common pattern in ICU research: given a long-format
DataFrame of timestamped observations for multiple patients, extract
fixed-size feature windows suitable for ML model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from clinops.temporal.imputation import ImputationStrategy, Imputer


@dataclass
class WindowConfig:
    """
    Configuration for temporal window extraction.

    Parameters
    ----------
    window_hours:
        Duration of each feature window in hours.
    step_hours:
        Step size between consecutive windows (< window_hours = overlapping).
        Set equal to window_hours for tumbling (non-overlapping) windows.
    min_observations:
        Minimum number of non-null observations required per window.
        Windows below this threshold are dropped.
    imputation:
        Strategy used to fill gaps within each window.
    aggregations:
        Dict mapping feature column names to aggregation functions.
        Defaults to mean for all numeric columns.
    label_col:
        Optional column name containing outcome labels. If provided,
        labels are extracted per window using label_fn.
    label_fn:
        Function (window_df) → label applied to each window's rows
        within the label_col. Defaults to last observed value.
    """

    window_hours: float = 24.0
    step_hours: float = 6.0
    min_observations: int = 1
    imputation: ImputationStrategy = ImputationStrategy.FORWARD_FILL
    aggregations: dict[str, str | Callable] = field(default_factory=dict)
    label_col: str | None = None
    label_fn: Callable | None = None


class TemporalWindower:
    """
    Extract fixed-size feature windows from long-format clinical time-series.

    Parameters
    ----------
    window_hours:
        Duration of each window in hours.
    step_hours:
        Step between window starts in hours. Use same value as window_hours
        for non-overlapping (tumbling) windows.
    imputation:
        Imputation strategy for within-window missing values.
    min_observations:
        Drop windows with fewer non-null observations than this threshold.
    aggregations:
        Column → aggregation function mapping. If empty, mean is used for
        all numeric columns.
    label_col:
        Column name of binary/multi-class outcome labels (optional).
    label_fn:
        How to derive the label for a window. Default: last non-null value.

    Examples
    --------
    >>> windower = TemporalWindower(window_hours=24, step_hours=6)
    >>> windows = windower.fit_transform(
    ...     df=chartevents,
    ...     id_col="subject_id",
    ...     time_col="charttime",
    ...     feature_cols=["heart_rate", "spo2", "resp_rate", "map"],
    ... )
    >>> windows.shape
    (4820, 6)   # (n_windows, n_features + id + window_start)
    """

    def __init__(
        self,
        window_hours: float = 24.0,
        step_hours: float = 6.0,
        imputation: ImputationStrategy = ImputationStrategy.FORWARD_FILL,
        min_observations: int = 1,
        aggregations: dict[str, str | Callable] | None = None,
        label_col: str | None = None,
        label_fn: Callable | None = None,
    ) -> None:
        self.config = WindowConfig(
            window_hours=window_hours,
            step_hours=step_hours,
            imputation=imputation,
            min_observations=min_observations,
            aggregations=aggregations or {},
            label_col=label_col,
            label_fn=label_fn,
        )
        self._imputer = Imputer(imputation)

    def fit_transform(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        feature_cols: list[str] | None = None,
        value_col: str | None = None,
        item_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Extract windows from a long-format clinical DataFrame.

        Supports two input formats:

        **Wide format** (one column per feature):
            ``df`` has columns like ``heart_rate``, ``spo2``, ``map``.
            Pass ``feature_cols`` to select which columns to use.

        **Long format** (item × value pairs):
            ``df`` has an ``item_col`` (e.g. ``itemid``) and a
            ``value_col`` (e.g. ``valuenum``).  Data is pivoted before
            windowing.

        Parameters
        ----------
        df:
            Input DataFrame in long or wide format.
        id_col:
            Column identifying each patient/subject.
        time_col:
            Datetime column for temporal ordering.
        feature_cols:
            Columns to include as features (wide format).
        value_col:
            Numeric value column (long format with item_col).
        item_col:
            Item identifier column (long format, e.g. itemid).

        Returns
        -------
        pd.DataFrame
            One row per (patient, window_start). Columns:
            ``id_col``, ``window_start``, ``window_end``, feature columns,
            and optionally ``label``.
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        # Pivot long → wide if item/value columns given
        if item_col and value_col:
            df = self._pivot_long_to_wide(df, id_col, time_col, item_col, value_col)
            feature_cols = [c for c in df.columns if c not in [id_col, time_col]]

        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                            if c not in [id_col]]

        logger.info(
            f"Windowing {len(df):,} rows for {df[id_col].nunique()} subjects "
            f"| window={self.config.window_hours}h step={self.config.step_hours}h "
            f"| {len(feature_cols)} features"
        )

        results = []
        window_td = pd.Timedelta(hours=self.config.window_hours)
        step_td = pd.Timedelta(hours=self.config.step_hours)

        for subject_id, subject_df in df.groupby(id_col):
            subject_df = subject_df.sort_values(time_col)
            t_start = subject_df[time_col].min()
            t_end = subject_df[time_col].max()

            window_start = t_start
            while window_start + window_td <= t_end + step_td:
                window_end = window_start + window_td
                mask = (subject_df[time_col] >= window_start) & (subject_df[time_col] < window_end)
                window_df = subject_df[mask]

                if len(window_df) < self.config.min_observations:
                    window_start += step_td
                    continue

                row = self._aggregate_window(window_df, feature_cols, subject_id, window_start, window_end, id_col)
                results.append(row)
                window_start += step_td

        if not results:
            logger.warning("No windows extracted — check min_observations threshold or data coverage")
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        logger.info(f"Extracted {len(result_df):,} windows across {result_df[id_col].nunique()} subjects")
        return result_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aggregate_window(
        self,
        window_df: pd.DataFrame,
        feature_cols: list[str],
        subject_id: object,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        id_col: str,
    ) -> dict:
        row: dict = {
            id_col: subject_id,
            "window_start": window_start,
            "window_end": window_end,
        }
        for col in feature_cols:
            if col not in window_df.columns:
                row[col] = np.nan
                continue
            series = window_df[col].dropna()
            if series.empty:
                row[col] = np.nan
                continue
            agg_fn = self.config.aggregations.get(col, "mean")
            if callable(agg_fn):
                row[col] = agg_fn(series)
            elif agg_fn == "mean":
                row[col] = series.mean()
            elif agg_fn == "median":
                row[col] = series.median()
            elif agg_fn == "min":
                row[col] = series.min()
            elif agg_fn == "max":
                row[col] = series.max()
            elif agg_fn == "last":
                row[col] = series.iloc[-1]
            elif agg_fn == "first":
                row[col] = series.iloc[0]
            else:
                row[col] = series.mean()

        if self.config.label_col and self.config.label_col in window_df.columns:
            label_series = window_df[self.config.label_col].dropna()
            if not label_series.empty:
                fn = self.config.label_fn or (lambda s: s.iloc[-1])
                row["label"] = fn(label_series)
            else:
                row["label"] = np.nan

        return row

    def _pivot_long_to_wide(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        item_col: str,
        value_col: str,
    ) -> pd.DataFrame:
        pivoted = (
            df.pivot_table(
                index=[id_col, time_col],
                columns=item_col,
                values=value_col,
                aggfunc="mean",
            )
            .reset_index()
        )
        pivoted.columns.name = None
        return pivoted
