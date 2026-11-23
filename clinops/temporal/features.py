"""
Lag feature construction and cohort alignment utilities.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LagFeatureBuilder:
    """
    Add lag and rolling-statistics features to windowed clinical data.

    Rows are ordered by ``id_col`` and ``time_col`` before lags are taken, so
    the input needs both columns. :class:`TemporalWindower` output has them
    (``window_start``); for other data, pass the column that orders each
    patient's rows as ``time_col``.

    Parameters
    ----------
    lags:
        Lag steps, in rows (windows), to add: ``[1, 2, 4]`` adds the values 1, 2,
        and 4 windows back. Default ``None`` means ``[1, 2]``; pass ``[]`` for no
        lags (e.g. only rolling statistics). Each step must be a positive
        integer: a lag of 0 or below would copy the current or a later value.
    rolling_windows:
        Window sizes, in rows, for rolling mean and standard deviation, each a
        positive integer. Default ``None`` means none. A rolling window covers
        the current row and the ones before it, never later rows.
    feature_cols:
        Columns to create features for. Default ``None`` uses every numeric
        column except ``id_col``, ``time_col``, and ``label``. That includes any
        numeric outcome column, whose lags would leak the label into the
        features, so pass ``feature_cols`` whenever ``df`` carries an outcome.
        ``[]`` means no columns.
    id_col:
        Patient identifier; lags never cross patients. Default ``"subject_id"``.
    time_col:
        Column that orders each patient's rows. Default ``"window_start"``.

    Examples
    --------
    >>> import pandas as pd
    >>> from clinops.temporal import LagFeatureBuilder
    >>> windows_df = pd.DataFrame({
    ...     "subject_id": [1, 1, 1, 1],
    ...     "window_start": pd.date_range("2150-01-01", periods=4, freq="6h"),
    ...     "heart_rate": [80.0, 85.0, 90.0, 95.0],
    ... })
    >>> builder = LagFeatureBuilder(lags=[1], rolling_windows=[2], feature_cols=["heart_rate"])
    >>> enriched = builder.fit_transform(windows_df)
    >>> enriched[["heart_rate", "heart_rate_lag1", "heart_rate_roll2_mean"]]
       heart_rate  heart_rate_lag1  heart_rate_roll2_mean
    0        80.0              NaN                   80.0
    1        85.0             80.0                   82.5
    2        90.0             85.0                   87.5
    3        95.0             90.0                   92.5
    """

    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        feature_cols: list[str] | None = None,
        id_col: str = "subject_id",
        time_col: str = "window_start",
    ) -> None:
        self.lags = _positive_steps("lags", [1, 2] if lags is None else lags)
        self.rolling_windows = _positive_steps(
            "rolling_windows", [] if rolling_windows is None else rolling_windows
        )
        self.feature_cols = feature_cols
        self.id_col = id_col
        self.time_col = time_col

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag and rolling features. Returns the enriched DataFrame.

        Raises
        ------
        ValueError
            If ``id_col`` or ``time_col`` is missing from ``df``.
        """
        missing = [c for c in (self.id_col, self.time_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"LagFeatureBuilder needs {missing} to order each patient's rows. "
                "TemporalWindower output has 'window_start'; for other data, pass the "
                "column that orders rows as time_col."
            )
        n_before = len(df.columns)
        df = df.copy().sort_values([self.id_col, self.time_col])
        if self.feature_cols is not None:
            numeric_cols = list(self.feature_cols)
        else:
            numeric_cols = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c not in (self.id_col, self.time_col, "label")
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

        logger.info(f"Added {len(df.columns) - n_before:,} lag/rolling features")
        return df


def _positive_steps(name: str, steps: list[int]) -> list[int]:
    """Check that lag steps / window sizes are positive integers."""
    bad = [s for s in steps if isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 1]
    if bad:
        raise ValueError(
            f"{name} must be positive integers (a lag of 0 or below would use the current "
            f"or a later row), got {bad}"
        )
    return [int(s) for s in steps]


class CohortAligner:
    """
    Align multiple patients' time-series to a common reference event.

    In clinical research it's common to align patients relative to an
    anchor event (e.g. ICU admission, ventilation start, first sepsis
    flag) rather than using wall-clock time. This class handles the
    realignment so downstream models see time-relative-to-event rather
    than absolute timestamps.

    Each ``id_col`` value must have exactly one anchor. For patients with
    several ICU stays, either align per stay by passing the stay identifier
    as ``id_col`` (e.g. ``"stay_id"``, which MIMIC-IV ``chartevents`` and
    ``icustays`` both carry), or reduce ``anchor_df`` to one row per patient
    first, e.g. the first stay::

        first_stays = stays.sort_values("intime").drop_duplicates("subject_id")

    Parameters
    ----------
    anchor_col:
        Column containing the anchor event timestamp.
    id_col:
        Identifier shared by ``events_df`` and ``anchor_df``. Use a patient
        id for one anchor per patient or a stay id for one anchor per stay.
    max_hours_before:
        Include data up to this many hours before the anchor event.
    max_hours_after:
        Include data up to this many hours after the anchor event.
    time_col:
        Event timestamp column in ``events_df``.

    Examples
    --------
    Keep only events in the 48 hours after ICU admission:

    >>> import pandas as pd
    >>> from clinops.temporal import CohortAligner
    >>> chartevents = pd.DataFrame({
    ...     "subject_id": [1, 1, 1],
    ...     "charttime": pd.to_datetime(
    ...         ["2150-01-01 06:00", "2150-01-01 20:00", "2150-01-04 08:00"]
    ...     ),
    ...     "heart_rate": [88.0, 92.0, 75.0],
    ... })
    >>> stays = pd.DataFrame(
    ...     {"subject_id": [1], "icu_intime": pd.to_datetime(["2150-01-01 08:00"])}
    ... )
    >>> aligner = CohortAligner(anchor_col="icu_intime", max_hours_after=48)
    >>> aligner.align(chartevents, stays)
       subject_id           charttime  heart_rate  hours_from_anchor
    0           1 2150-01-01 20:00:00        92.0               12.0

    A patient with two ICU stays: align each event to its own stay.

    >>> events = pd.DataFrame({
    ...     "subject_id": [1, 1],
    ...     "stay_id": [10, 11],
    ...     "charttime": pd.to_datetime(["2150-01-01 09:00", "2150-06-01 10:00"]),
    ...     "heart_rate": [80.0, 95.0],
    ... })
    >>> stays = pd.DataFrame({
    ...     "subject_id": [1, 1],
    ...     "stay_id": [10, 11],
    ...     "intime": pd.to_datetime(["2150-01-01 08:00", "2150-06-01 08:00"]),
    ... })
    >>> CohortAligner(anchor_col="intime", id_col="stay_id").align(events, stays)
       subject_id  stay_id           charttime  heart_rate  hours_from_anchor
    0           1       10 2150-01-01 09:00:00        80.0                1.0
    1           1       11 2150-06-01 10:00:00        95.0                2.0
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
            One anchor per ``id_col`` value, in ``id_col`` and ``anchor_col``.
            Repeated identical rows are allowed.

        Returns
        -------
        pd.DataFrame
            events_df filtered to the alignment window with a new
            ``hours_from_anchor`` column (negative = before anchor). Events
            whose ``id_col`` value has no anchor are dropped.

        Raises
        ------
        ValueError
            If a required column is missing or ``anchor_df`` has more than one
            row for an ``id_col`` value.
        """
        for frame, name, cols in (
            (anchor_df, "anchor_df", (self.id_col, self.anchor_col)),
            (events_df, "events_df", (self.id_col, self.time_col)),
        ):
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise ValueError(f"CohortAligner: {name} is missing column(s) {missing}")

        # Exact duplicate rows are harmless; different anchors for one id are not.
        anchors_by_id = anchor_df[[self.id_col, self.anchor_col]].drop_duplicates()
        duplicated = anchors_by_id[self.id_col].duplicated(keep=False)
        if duplicated.any():
            n_ids = anchors_by_id.loc[duplicated, self.id_col].nunique()
            raise ValueError(
                f"CohortAligner: anchor_df has more than one {self.anchor_col!r} for "
                f"{n_ids} {self.id_col!r} value(s), but each needs exactly one anchor. "
                f"If these are patients with several ICU stays, align per stay with "
                f"id_col='stay_id', or keep one row per patient with "
                f"anchor_df.sort_values({self.anchor_col!r}).drop_duplicates({self.id_col!r})"
            )

        anchor_map = pd.to_datetime(anchors_by_id.set_index(self.id_col)[self.anchor_col])

        df = events_df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])

        anchors = df[self.id_col].map(anchor_map)
        n_unanchored = int(anchors.isna().sum())
        if n_unanchored:
            logger.warning(
                f"CohortAligner: dropping {n_unanchored:,} events whose {self.id_col!r} "
                "has no anchor in anchor_df"
            )
        df["hours_from_anchor"] = (df[self.time_col] - anchors).dt.total_seconds() / 3600

        df = df[
            (df["hours_from_anchor"] >= -self.max_hours_before)
            & (df["hours_from_anchor"] <= self.max_hours_after)
        ].reset_index(drop=True)

        logger.info(
            f"CohortAligner: retained {len(df):,} rows "
            f"(window: -{self.max_hours_before}h to +{self.max_hours_after}h from anchor)"
        )
        return df
