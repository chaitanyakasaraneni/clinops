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


def _make_long_df(n_items: int = 2, n_hours: int = 6) -> pd.DataFrame:
    """
    Build a minimal long-format DataFrame with multiple itemids so that
    pd.pivot_table() produces a named columns axis that needs clearing.
    """
    base = datetime(2023, 1, 1)
    item_ids = [220045 + i for i in range(n_items)]
    rows = [
        {
            "subject_id": 1,
            "charttime": base + timedelta(hours=h),
            "itemid": item_id,
            "valuenum": float(70 + h),
        }
        for h in range(n_hours)
        for item_id in item_ids
    ]
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

    def test_long_format_pivot_produces_wide_output(self):
        base = datetime(2023, 1, 1)
        rows = []
        for h in range(12):
            t = base + timedelta(hours=h)
            rows.append({"subject_id": 1, "charttime": t, "itemid": 220045, "valuenum": 70.0 + h})
            rows.append({"subject_id": 1, "charttime": t, "itemid": 220277, "valuenum": 97.0})
        df = pd.DataFrame(rows)
        windower = TemporalWindower(window_hours=6, step_hours=6)
        result = windower.fit_transform(
            df, id_col="subject_id", time_col="charttime",
            item_col="itemid", value_col="valuenum",
        )
        assert len(result) > 0
        col_names = [str(c) for c in result.columns]
        assert "220045" in col_names or 220045 in result.columns

    def test_pivot_long_to_wide_clears_columns_name(self):
        """
        pd.pivot_table(..., columns="itemid") sets columns.name = "itemid".
        _pivot_long_to_wide() must clear it to None.

        The original test called fit_transform() and checked result.columns.name,
        but fit_transform() builds its output via pd.DataFrame(list_of_dicts) whose
        columns.name is always None — the pivot output is fully consumed before the
        final DataFrame is created. This test calls _pivot_long_to_wide() directly
        so it will fail if the `pivoted.columns.name = None` line is removed.
        """
        df = _make_long_df(n_items=3)
        windower = TemporalWindower(window_hours=6, step_hours=6)

        pivoted = windower._pivot_long_to_wide(
            df,
            id_col="subject_id",
            time_col="charttime",
            item_col="itemid",
            value_col="valuenum",
        )

        assert pivoted.columns.name is None, (
            f"Expected columns.name=None after _pivot_long_to_wide(), "
            f"got {pivoted.columns.name!r}. "
            f"The line `pivoted.columns.name = None` may have been removed."
        )

    def test_pivot_preserves_all_item_columns(self):
        """
        Sanity: pivoted output has one column per unique itemid.
        Ensures the pivot actually ran and columns.name = None
        isn't just from an early return or passthrough.
        """
        n_items = 4
        df = _make_long_df(n_items=n_items)
        windower = TemporalWindower(window_hours=6, step_hours=6)

        pivoted = windower._pivot_long_to_wide(
            df,
            id_col="subject_id",
            time_col="charttime",
            item_col="itemid",
            value_col="valuenum",
        )

        assert pivoted.columns.name is None
        feature_cols = [c for c in pivoted.columns if c not in ("subject_id", "charttime")]
        assert len(feature_cols) == n_items, (
            f"Expected {n_items} feature columns, got: {feature_cols}"
        )

    def test_pivot_columns_name_none_survives_reset_index(self):
        """
        reset_index() is called inside _pivot_long_to_wide() after the pivot.
        Verify that columns.name = None is set AFTER reset_index() (which can
        re-introduce a named axis), not before, and that index columns are
        restored as regular columns.
        """
        df = _make_long_df(n_items=2)
        windower = TemporalWindower(window_hours=6, step_hours=6)

        pivoted = windower._pivot_long_to_wide(
            df,
            id_col="subject_id",
            time_col="charttime",
            item_col="itemid",
            value_col="valuenum",
        )

        assert pivoted.columns.name is None
        assert "subject_id" in pivoted.columns
        assert "charttime" in pivoted.columns

    def test_feature_cols_auto_detected(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1] * 6,
            "charttime": [base + timedelta(hours=h) for h in range(6)],
            "heart_rate": [70.0] * 6,
            "spo2": [97.0] * 6,
        })
        windower = TemporalWindower(window_hours=6, step_hours=6)
        result = windower.fit_transform(df, id_col="subject_id", time_col="charttime")
        assert "heart_rate" in result.columns
        assert "spo2" in result.columns

    def test_missing_feature_col_produces_nan(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1] * 6,
            "charttime": [base + timedelta(hours=h) for h in range(6)],
            "heart_rate": [70.0] * 6,
        })
        windower = TemporalWindower(window_hours=6, step_hours=6)
        result = windower.fit_transform(
            df, id_col="subject_id", time_col="charttime",
            feature_cols=["heart_rate", "nonexistent"],
        )
        assert "nonexistent" in result.columns
        assert result["nonexistent"].isna().all()

    def test_all_nan_feature_within_window_gives_nan(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "charttime": [base + timedelta(hours=h) for h in range(3)],
            "heart_rate": [np.nan, np.nan, np.nan],
        })
        windower = TemporalWindower(window_hours=6, step_hours=6, min_observations=1)
        result = windower.fit_transform(
            df, id_col="subject_id", time_col="charttime", feature_cols=["heart_rate"]
        )
        assert result["heart_rate"].isna().all()

    def test_callable_aggregation(self):
        df = make_vitals_df(n_patients=1, n_hours=12)
        windower = TemporalWindower(
            window_hours=6, step_hours=6,
            aggregations={"heart_rate": lambda s: s.quantile(0.9)},
        )
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert len(result) > 0
        assert pd.notna(result["heart_rate"].iloc[0])

    def test_median_aggregation(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "charttime": [base + timedelta(hours=h) for h in range(3)],
            "heart_rate": [70.0, 80.0, 90.0],
        })
        windower = TemporalWindower(
            window_hours=6, step_hours=6, aggregations={"heart_rate": "median"}
        )
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert result["heart_rate"].iloc[0] == pytest.approx(80.0)

    def test_last_aggregation(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "charttime": [base + timedelta(hours=h) for h in range(3)],
            "heart_rate": [70.0, 75.0, 80.0],
        })
        windower = TemporalWindower(
            window_hours=6, step_hours=6, aggregations={"heart_rate": "last"}
        )
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert result["heart_rate"].iloc[0] == pytest.approx(80.0)

    def test_first_aggregation(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "charttime": [base + timedelta(hours=h) for h in range(3)],
            "heart_rate": [70.0, 75.0, 80.0],
        })
        windower = TemporalWindower(
            window_hours=6, step_hours=6, aggregations={"heart_rate": "first"}
        )
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert result["heart_rate"].iloc[0] == pytest.approx(70.0)

    def test_unknown_aggregation_falls_back_to_mean(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "charttime": [base + timedelta(hours=h) for h in range(3)],
            "heart_rate": [70.0, 80.0, 90.0],
        })
        windower = TemporalWindower(
            window_hours=6, step_hours=6, aggregations={"heart_rate": "p95"}
        )
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert result["heart_rate"].iloc[0] == pytest.approx(80.0)

    def test_label_col_extracted_as_last_value(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1] * 4,
            "charttime": [base + timedelta(hours=h) for h in range(4)],
            "heart_rate": [70.0] * 4,
            "outcome": [0, 0, 0, 1],
        })
        windower = TemporalWindower(window_hours=6, step_hours=6, label_col="outcome")
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert "label" in result.columns
        assert result["label"].iloc[0] == pytest.approx(1)

    def test_custom_label_fn_applied(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1] * 4,
            "charttime": [base + timedelta(hours=h) for h in range(4)],
            "heart_rate": [70.0] * 4,
            "outcome": [0, 1, 0, 1],
        })
        windower = TemporalWindower(
            window_hours=6, step_hours=6,
            label_col="outcome",
            label_fn=lambda s: int(s.max()),
        )
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert result["label"].iloc[0] == 1

    def test_label_col_all_nan_gives_nan_label(self):
        base = datetime(2023, 1, 1)
        df = pd.DataFrame({
            "subject_id": [1] * 4,
            "charttime": [base + timedelta(hours=h) for h in range(4)],
            "heart_rate": [70.0] * 4,
            "outcome": [np.nan, np.nan, np.nan, np.nan],
        })
        windower = TemporalWindower(window_hours=6, step_hours=6, label_col="outcome")
        result = windower.fit_transform(df, "subject_id", "charttime", ["heart_rate"])
        assert "label" in result.columns
        assert result["label"].isna().all()


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

    def test_per_patient_mean_stores_patient_means(self):
        df = pd.DataFrame({"subject_id": [1, 1, 2, 2], "hr": [60.0, 80.0, 90.0, 100.0]})
        imputer = Imputer(ImputationStrategy.MEAN, per_patient=True, id_col="subject_id")
        imputer.fit(df)
        assert hasattr(imputer, "_patient_means")
        assert imputer._patient_means.loc[1, "hr"] == pytest.approx(70.0)
        assert imputer._patient_means.loc[2, "hr"] == pytest.approx(95.0)

    def test_per_patient_median_stores_patient_medians(self):
        df = pd.DataFrame({"subject_id": [1, 1, 2, 2], "hr": [60.0, 80.0, 90.0, 100.0]})
        imputer = Imputer(ImputationStrategy.MEDIAN, per_patient=True, id_col="subject_id")
        imputer.fit(df)
        assert hasattr(imputer, "_patient_medians")
        assert imputer._patient_medians.loc[1, "hr"] == pytest.approx(70.0)

    def test_global_median_fit_populates_fill_values(self):
        df = pd.DataFrame({"hr": [60.0, 80.0, 100.0]})
        imputer = Imputer(ImputationStrategy.MEDIAN)
        imputer.fit(df)
        assert imputer._fill_values["hr"] == pytest.approx(80.0)

    def test_global_median_transform_uses_fitted_median(self):
        imputer = Imputer(ImputationStrategy.MEDIAN)
        imputer.fit(pd.DataFrame({"hr": [60.0, 80.0, 100.0]}))
        result = imputer.transform(pd.DataFrame({"hr": [np.nan, 75.0]}))
        assert result["hr"].iloc[0] == pytest.approx(80.0)

    def test_forward_fill_gap_exceeding_threshold_renulled(self):
        # 6h gap > 4h threshold: ffill fills NaN then gap masking re-nulls it
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 6), datetime(2023, 1, 1, 12)])
        df = pd.DataFrame({"time": times, "hr": [70.0, np.nan, 80.0]})
        imputer = Imputer(ImputationStrategy.FORWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        assert pd.isna(result["hr"].iloc[1])

    def test_forward_fill_gap_within_threshold_kept(self):
        # 2h gap < 4h threshold: fill should persist
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 2), datetime(2023, 1, 1, 4)])
        df = pd.DataFrame({"time": times, "hr": [70.0, np.nan, 80.0]})
        imputer = Imputer(ImputationStrategy.FORWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        assert result["hr"].iloc[1] == pytest.approx(70.0)

    def test_backward_fill_gap_exceeding_threshold_renulled(self):
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 6), datetime(2023, 1, 1, 12)])
        df = pd.DataFrame({"time": times, "hr": [np.nan, np.nan, 80.0]})
        imputer = Imputer(ImputationStrategy.BACKWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        assert pd.isna(result["hr"].iloc[0])
        assert pd.isna(result["hr"].iloc[1])

    def test_backward_fill_gap_within_threshold_kept(self):
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 2), datetime(2023, 1, 1, 4)])
        df = pd.DataFrame({"time": times, "hr": [np.nan, np.nan, 80.0]})
        imputer = Imputer(ImputationStrategy.BACKWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        assert result["hr"].iloc[0] == pytest.approx(80.0)

    def test_mean_unfitted_falls_back_to_column_mean(self):
        imputer = Imputer(ImputationStrategy.MEAN)  # fit() never called
        df = pd.DataFrame({"hr": [60.0, np.nan, 80.0]})
        result = imputer.transform(df)
        assert result["hr"].iloc[1] == pytest.approx(70.0)

    def test_median_fitted_uses_stored_fill_values(self):
        imputer = Imputer(ImputationStrategy.MEDIAN)
        imputer.fit(pd.DataFrame({"hr": [60.0, 80.0, 100.0]}))
        result = imputer.transform(pd.DataFrame({"hr": [np.nan]}))
        assert result["hr"].iloc[0] == pytest.approx(80.0)

    def test_median_unfitted_falls_back_to_column_median(self):
        imputer = Imputer(ImputationStrategy.MEDIAN)  # fit() never called
        df = pd.DataFrame({"hr": [60.0, np.nan, 80.0, 100.0]})
        result = imputer.transform(df)
        assert result["hr"].iloc[1] == pytest.approx(80.0)

    def test_mask_large_gaps_skipped_when_time_col_absent(self):
        df = pd.DataFrame({"hr": [70.0, np.nan, 80.0]})
        imputer = Imputer(ImputationStrategy.FORWARD_FILL, max_gap_hours=1, time_col="charttime")
        result = imputer.fit_transform(df)  # charttime not in df → gap masking skipped
        assert result["hr"].iloc[1] == pytest.approx(70.0)  # ffill still ran

    def test_originally_non_null_values_not_masked(self):
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 6), datetime(2023, 1, 1, 12)])
        df = pd.DataFrame({"time": times, "hr": [70.0, 75.0, 80.0]})  # no NaNs
        imputer = Imputer(ImputationStrategy.FORWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        assert result["hr"].iloc[1] == pytest.approx(75.0)  # original value preserved

    def test_forward_fill_gap_masking_correct_on_unsorted_input(self):
        # Rows delivered in reverse chronological order.
        # Gap masking must apply correctly (based on sorted time gaps) AND
        # the result must come back in the original input order.
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 12), datetime(2023, 1, 1, 6), datetime(2023, 1, 1, 0)])
        df = pd.DataFrame({"time": times, "hr": [80.0, np.nan, 70.0]})  # reversed order
        imputer = Imputer(ImputationStrategy.FORWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        # Input order preserved: row 0=12:00, row 1=06:00, row 2=00:00
        assert result["time"].iloc[0] == pd.Timestamp("2023-01-01 12:00")
        assert result["time"].iloc[1] == pd.Timestamp("2023-01-01 06:00")
        assert result["time"].iloc[2] == pd.Timestamp("2023-01-01 00:00")
        # Gap masking: 06:00 NaN has 6h gap from 00:00 > 4h → stays NaN
        assert result["hr"].iloc[0] == pytest.approx(80.0)   # 12:00 — original
        assert pd.isna(result["hr"].iloc[1])                  # 06:00 — gap too large, re-nulled
        assert result["hr"].iloc[2] == pytest.approx(70.0)   # 00:00 — original

    def test_forward_fill_gap_within_threshold_correct_on_unsorted_input(self):
        # Same unsorted setup but gap is within threshold — fill should persist.
        # Input order must be preserved in the result.
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 4), datetime(2023, 1, 1, 2), datetime(2023, 1, 1, 0)])
        df = pd.DataFrame({"time": times, "hr": [80.0, np.nan, 70.0]})  # reversed
        imputer = Imputer(ImputationStrategy.FORWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        # Input order preserved: row 0=04:00, row 1=02:00, row 2=00:00
        assert result["time"].iloc[0] == pd.Timestamp("2023-01-01 04:00")
        assert result["time"].iloc[1] == pd.Timestamp("2023-01-01 02:00")
        assert result["time"].iloc[2] == pd.Timestamp("2023-01-01 00:00")
        assert result["hr"].iloc[0] == pytest.approx(80.0)   # 04:00 — original
        assert result["hr"].iloc[1] == pytest.approx(70.0)   # 02:00 — filled (2h < 4h)
        assert result["hr"].iloc[2] == pytest.approx(70.0)   # 00:00 — original

    def test_backward_fill_gap_masking_correct_on_unsorted_input(self):
        # Rows delivered in reverse chronological order.
        # Gap masking must apply correctly AND result must come back in input order.
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 12), datetime(2023, 1, 1, 6), datetime(2023, 1, 1, 0)])
        df = pd.DataFrame({"time": times, "hr": [80.0, np.nan, np.nan]})  # reversed
        imputer = Imputer(ImputationStrategy.BACKWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        # Input order preserved: row 0=12:00, row 1=06:00, row 2=00:00
        assert result["time"].iloc[0] == pd.Timestamp("2023-01-01 12:00")
        assert result["time"].iloc[1] == pd.Timestamp("2023-01-01 06:00")
        assert result["time"].iloc[2] == pd.Timestamp("2023-01-01 00:00")
        # Both NaNs: 06:00 is 6h from 12:00 > 4h → stays NaN;
        #            00:00 is 12h from 12:00 > 4h → stays NaN
        assert result["hr"].iloc[0] == pytest.approx(80.0)   # 12:00 — original
        assert pd.isna(result["hr"].iloc[1])                  # 06:00 — gap too large
        assert pd.isna(result["hr"].iloc[2])                  # 00:00 — gap too large

    def test_backward_fill_gap_within_threshold_correct_on_unsorted_input(self):
        # Gap within threshold — fill should persist. Input order must be preserved.
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 4), datetime(2023, 1, 1, 2), datetime(2023, 1, 1, 0)])
        df = pd.DataFrame({"time": times, "hr": [80.0, np.nan, np.nan]})  # reversed
        imputer = Imputer(ImputationStrategy.BACKWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        # Input order preserved: row 0=04:00, row 1=02:00, row 2=00:00
        assert result["time"].iloc[0] == pd.Timestamp("2023-01-01 04:00")
        assert result["time"].iloc[1] == pd.Timestamp("2023-01-01 02:00")
        assert result["time"].iloc[2] == pd.Timestamp("2023-01-01 00:00")
        assert result["hr"].iloc[0] == pytest.approx(80.0)   # 04:00 — original
        assert result["hr"].iloc[1] == pytest.approx(80.0)   # 02:00 — filled (2h < 4h)
        assert result["hr"].iloc[2] == pytest.approx(80.0)   # 00:00 — filled (4h <= 4h)

    def test_forward_fill_gap_masking_preserves_input_row_order(self):
        # transform() must return rows in the same order they were passed in,
        # even when gap masking internally sorts by time. Before the fix,
        # sort_values() was not reversed, so callers relying on positional
        # iloc would get silently wrong results.
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 12), datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 6)])
        # Input order: 12:00, 00:00, 06:00 (deliberately shuffled)
        df = pd.DataFrame({"time": times, "hr": [80.0, 70.0, np.nan]})
        imputer = Imputer(ImputationStrategy.FORWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        # Row 0 (12:00), row 1 (00:00), row 2 (06:00) — input order must be preserved
        assert result["time"].iloc[0] == pd.Timestamp("2023-01-01 12:00")
        assert result["time"].iloc[1] == pd.Timestamp("2023-01-01 00:00")
        assert result["time"].iloc[2] == pd.Timestamp("2023-01-01 06:00")
        # And gap masking must still work: 06:00 NaN is 6h from 00:00 > 4h → stays NaN
        assert pd.isna(result["hr"].iloc[2])

    def test_backward_fill_gap_masking_preserves_input_row_order(self):
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 12), datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 6)])
        # Input order: 12:00, 00:00, 06:00 (shuffled)
        df = pd.DataFrame({"time": times, "hr": [80.0, np.nan, np.nan]})
        imputer = Imputer(ImputationStrategy.BACKWARD_FILL, max_gap_hours=4, time_col="time")
        result = imputer.fit_transform(df)
        # Row order must match input
        assert result["time"].iloc[0] == pd.Timestamp("2023-01-01 12:00")
        assert result["time"].iloc[1] == pd.Timestamp("2023-01-01 00:00")
        assert result["time"].iloc[2] == pd.Timestamp("2023-01-01 06:00")
        # 00:00 NaN: 12h from 12:00 > 4h → stays NaN; 06:00 NaN: 6h > 4h → stays NaN
        assert pd.isna(result["hr"].iloc[1])
        assert pd.isna(result["hr"].iloc[2])

    def test_max_gap_hours_zero_enables_masking_not_disables_it(self):
        # max_gap_hours=0 means "never fill across any gap" — every filled value
        # should be re-nulled. With truthiness check (if self.max_gap_hours),
        # 0 was treated as falsey and gap masking was silently skipped entirely.
        times = pd.to_datetime(
            [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 1), datetime(2023, 1, 1, 2)])
        df = pd.DataFrame({"time": times, "hr": [70.0, np.nan, 80.0]})
        imputer = Imputer(ImputationStrategy.FORWARD_FILL, max_gap_hours=0, time_col="time")
        result = imputer.fit_transform(df)
        # Any gap > 0h means re-null: the filled NaN at 01:00 must be restored
        assert pd.isna(result["hr"].iloc[1])


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
