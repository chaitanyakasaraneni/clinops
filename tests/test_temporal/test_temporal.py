"""Tests for clinops.temporal module."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from clinops.temporal import (
    CohortAligner,
    ImputationStrategy,
    Imputer,
    LagFeatureBuilder,
    TemporalWindower,
)


def make_vitals_df(n_patients=3, n_hours=48, freq_minutes=60) -> pd.DataFrame:
    """Generate synthetic ICU vitals DataFrame for testing."""
    rows = []
    base_time = datetime(2023, 1, 1)
    for pid in range(1, n_patients + 1):
        for h in range(0, n_hours * 60, freq_minutes):
            rows.append(
                {
                    "subject_id": pid,
                    "charttime": base_time + timedelta(minutes=h),
                    "heart_rate": 70 + np.random.normal(0, 5),
                    "spo2": 97 + np.random.normal(0, 1),
                    "resp_rate": 16 + np.random.normal(0, 2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TemporalWindower tests
# ---------------------------------------------------------------------------


class TestTemporalWindower:
    def test_basic_windowing(self):
        df = make_vitals_df(n_patients=2, n_hours=24, freq_minutes=60)
        windower = TemporalWindower(window_hours=6, step_hours=6)
        result = windower.fit_transform(
            df, id_col="subject_id", time_col="charttime", feature_cols=["heart_rate", "spo2"]
        )
        assert len(result) > 0
        assert "window_start" in result.columns
        assert "heart_rate" in result.columns
        assert "spo2" in result.columns

    def test_overlapping_windows(self):
        df = make_vitals_df(n_patients=1, n_hours=24)
        w_tumbling = TemporalWindower(window_hours=6, step_hours=6)
        w_sliding = TemporalWindower(window_hours=6, step_hours=3)
        r_tumbling = w_tumbling.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        r_sliding = w_sliding.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert len(r_sliding) >= len(r_tumbling)

    def test_min_observations_filter(self):
        df = make_vitals_df(n_patients=1, n_hours=6, freq_minutes=60)
        windower = TemporalWindower(window_hours=1, step_hours=1, min_observations=10)
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        # Each 1h window has only 1 observation, all should be filtered
        assert len(result) == 0

    def test_custom_aggregation(self):
        df = make_vitals_df(n_patients=1, n_hours=12)
        windower = TemporalWindower(
            window_hours=6, step_hours=6, aggregations={"heart_rate": "max", "spo2": "min"}
        )
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate", "spo2"])
        assert len(result) > 0

    def test_empty_result_on_no_coverage(self):
        df = make_vitals_df(n_patients=1, n_hours=1, freq_minutes=60)
        windower = TemporalWindower(window_hours=24, step_hours=24, min_observations=20)
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert len(result) == 0

    def test_multi_patient_independence(self):
        df = make_vitals_df(n_patients=5, n_hours=24)
        windower = TemporalWindower(window_hours=6, step_hours=6)
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert result["subject_id"].nunique() == 5


# ---------------------------------------------------------------------------
# Imputer tests
# ---------------------------------------------------------------------------


class TestImputer:
    def test_forward_fill(self):
        df = pd.DataFrame({"hr": [70.0, np.nan, np.nan, 75.0]})
        imputed = Imputer(ImputationStrategy.FORWARD_FILL).fit_transform(df)
        assert imputed["hr"].isna().sum() == 0
        assert imputed["hr"].iloc[1] == 70.0

    def test_backward_fill(self):
        df = pd.DataFrame({"hr": [np.nan, np.nan, 75.0, 80.0]})
        imputed = Imputer(ImputationStrategy.BACKWARD_FILL).fit_transform(df)
        assert imputed["hr"].isna().sum() == 0
        assert imputed["hr"].iloc[0] == 75.0

    def test_linear_interpolation(self):
        df = pd.DataFrame({"hr": [60.0, np.nan, 80.0]})
        imputed = Imputer(ImputationStrategy.LINEAR).fit_transform(df)
        assert imputed["hr"].iloc[1] == pytest.approx(70.0)

    def test_mean_imputation(self):
        df = pd.DataFrame({"hr": [60.0, 80.0, 70.0]})
        imputer = Imputer(ImputationStrategy.MEAN)
        imputer.fit(df)
        test_df = pd.DataFrame({"hr": [np.nan, 75.0]})
        result = imputer.transform(test_df)
        assert result["hr"].iloc[0] == pytest.approx(70.0)

    def test_indicator_strategy(self):
        df = pd.DataFrame({"hr": [70.0, np.nan, 75.0]})
        imputed = Imputer(ImputationStrategy.INDICATOR).fit_transform(df)
        assert "hr_missing" in imputed.columns
        assert imputed["hr_missing"].iloc[1] == 1
        assert imputed["hr_missing"].iloc[0] == 0
        assert imputed["hr"].isna().sum() == 0

    def test_zero_fill(self):
        df = pd.DataFrame({"count": [1.0, np.nan, 3.0]})
        imputed = Imputer(ImputationStrategy.ZERO).fit_transform(df)
        assert imputed["count"].iloc[1] == 0.0

    def test_none_strategy_leaves_nans(self):
        df = pd.DataFrame({"hr": [70.0, np.nan, 75.0]})
        imputed = Imputer(ImputationStrategy.NONE).fit_transform(df)
        assert imputed["hr"].isna().sum() == 1


# ---------------------------------------------------------------------------
# LagFeatureBuilder tests
# ---------------------------------------------------------------------------


class TestLagFeatureBuilder:
    def test_lag_columns_created(self):
        df = make_vitals_df(n_patients=1, n_hours=12)
        windower = TemporalWindower(window_hours=2, step_hours=2)
        windows = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])

        builder = LagFeatureBuilder(lags=[1, 2], id_col="subject_id")
        enriched = builder.fit_transform(windows)
        assert "heart_rate_lag1" in enriched.columns
        assert "heart_rate_lag2" in enriched.columns

    def test_rolling_columns_created(self):
        df = make_vitals_df(n_patients=1, n_hours=24)
        windower = TemporalWindower(window_hours=3, step_hours=3)
        windows = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])

        builder = LagFeatureBuilder(lags=[], rolling_windows=[4], id_col="subject_id")
        enriched = builder.fit_transform(windows)
        assert "heart_rate_roll4_mean" in enriched.columns
        assert "heart_rate_roll4_std" in enriched.columns


# ---------------------------------------------------------------------------
# CohortAligner tests
# ---------------------------------------------------------------------------


class TestCohortAligner:
    def test_basic_alignment(self):
        base = datetime(2023, 1, 1, 12, 0)
        events = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 1],
                "charttime": [
                    base - timedelta(hours=2),
                    base - timedelta(hours=1),
                    base + timedelta(hours=6),
                    base + timedelta(hours=50),  # outside window
                ],
                "heart_rate": [70, 72, 75, 80],
            }
        )
        anchors = pd.DataFrame({"subject_id": [1], "icu_intime": [base]})
        aligner = CohortAligner(anchor_col="icu_intime", max_hours_before=4, max_hours_after=24)
        aligned = aligner.align(events, anchors)
        assert len(aligned) == 3  # row at +50h excluded
        assert "hours_from_anchor" in aligned.columns
        assert aligned["hours_from_anchor"].min() >= -4
        assert aligned["hours_from_anchor"].max() <= 24
