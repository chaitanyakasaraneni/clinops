"""Tests for clinops.monitor — drift detection and data quality."""

import numpy as np
import pandas as pd
import pytest

from clinops.monitor import (
    DataQualityChecker,
    DistributionDriftDetector,
    DriftSeverity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "heart_rate": rng.normal(75, 10, n),
            "spo2": rng.normal(97, 2, n),
            "glucose": rng.normal(120, 30, n),
        }
    )


# ---------------------------------------------------------------------------
# DistributionDriftDetector
# ---------------------------------------------------------------------------


class TestDistributionDriftDetector:
    def test_fit_returns_self(self):
        detector = DistributionDriftDetector()
        ref = make_df()
        assert detector.fit(ref) is detector

    def test_detect_requires_fit(self):
        detector = DistributionDriftDetector()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            detector.detect(make_df())

    def test_stable_distribution_is_low_severity(self):
        # Use a large sample to keep PSI well below the 0.1 MEDIUM threshold
        ref = make_df(n=2000, seed=0)
        cur = make_df(n=2000, seed=1)
        detector = DistributionDriftDetector()
        detector.fit(ref)
        report = detector.detect(cur)
        assert report.n_high == 0
        assert report.n_medium == 0
        assert report.n_columns_checked == 3

    def test_shifted_distribution_detected_as_high(self):
        ref = make_df(n=500, seed=0)
        rng = np.random.default_rng(42)
        # Shift heart_rate mean by 4 standard deviations — should be HIGH drift
        cur = ref.copy()
        cur["heart_rate"] = rng.normal(75 + 40, 10, len(cur))
        detector = DistributionDriftDetector()
        detector.fit(ref)
        report = detector.detect(cur)
        hr_result = next(r for r in report.results if r.column == "heart_rate")
        assert hr_result.severity == DriftSeverity.HIGH

    def test_psi_is_zero_for_identical_data(self):
        df = make_df()
        detector = DistributionDriftDetector(run_ks_test=False)
        detector.fit(df)
        report = detector.detect(df)
        for result in report.results:
            assert result.psi == pytest.approx(0.0, abs=1e-6)

    def test_ks_test_disabled(self):
        df = make_df()
        detector = DistributionDriftDetector(run_ks_test=False)
        detector.fit(df)
        report = detector.detect(df)
        for result in report.results:
            assert result.ks_statistic is None
            assert result.ks_pvalue is None

    def test_ks_test_enabled_returns_values(self):
        df = make_df()
        detector = DistributionDriftDetector(run_ks_test=True)
        detector.fit(df)
        report = detector.detect(df)
        for result in report.results:
            assert result.ks_statistic is not None
            assert result.ks_pvalue is not None

    def test_missing_column_in_current_is_skipped(self):
        ref = make_df()
        cur = make_df().drop(columns=["heart_rate"])
        detector = DistributionDriftDetector()
        detector.fit(ref)
        report = detector.detect(cur)
        assert all(r.column != "heart_rate" for r in report.results)

    def test_explicit_columns_restricts_monitoring(self):
        ref = make_df()
        detector = DistributionDriftDetector(columns=["heart_rate"])
        detector.fit(ref)
        report = detector.detect(make_df(seed=99))
        assert report.n_columns_checked == 1
        assert report.results[0].column == "heart_rate"

    def test_drifted_columns_filters_by_severity(self):
        ref = make_df(n=500, seed=0)
        cur = ref.copy()
        cur["heart_rate"] = np.random.default_rng(7).normal(200, 10, len(cur))
        detector = DistributionDriftDetector()
        detector.fit(ref)
        report = detector.detect(cur)
        high = report.drifted_columns(DriftSeverity.HIGH)
        assert "heart_rate" in high

    def test_to_dataframe_returns_sorted_by_psi(self):
        ref = make_df(n=500, seed=0)
        cur = ref.copy()
        cur["heart_rate"] = np.random.default_rng(7).normal(200, 10, len(cur))
        detector = DistributionDriftDetector()
        detector.fit(ref)
        report = detector.detect(cur)
        df = report.to_dataframe()
        assert df["psi"].iloc[0] >= df["psi"].iloc[-1]
        assert "column" in df.columns

    def test_summary_string(self):
        ref = make_df()
        detector = DistributionDriftDetector()
        detector.fit(ref)
        report = detector.detect(make_df(seed=1))
        s = report.summary()
        assert "Columns checked" in s

    def test_n_bins_validation(self):
        with pytest.raises(ValueError):
            DistributionDriftDetector(n_bins=1)

    def test_mean_shift_property(self):
        ref = make_df(n=500, seed=0)
        cur = ref.copy()
        cur["heart_rate"] = cur["heart_rate"] + 10.0
        detector = DistributionDriftDetector()
        detector.fit(ref)
        report = detector.detect(cur)
        hr = next(r for r in report.results if r.column == "heart_rate")
        assert hr.mean_shift == pytest.approx(10.0, abs=0.5)

    def test_all_null_column_in_current_is_skipped(self):
        ref = make_df()
        cur = make_df()
        cur["heart_rate"] = np.nan
        detector = DistributionDriftDetector()
        detector.fit(ref)
        report = detector.detect(cur)
        # heart_rate skipped — should not appear in results
        assert all(r.column != "heart_rate" for r in report.results)


# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------


class TestDataQualityChecker:
    def test_no_issues_on_clean_data(self):
        df = make_df()
        report = DataQualityChecker().check(df)
        assert report.passed
        assert len(report.errors) == 0

    def test_high_null_rate_is_warning(self):
        df = make_df()
        df.loc[:150, "heart_rate"] = np.nan  # > 50% null
        report = DataQualityChecker(max_null_rate=0.5).check(df)
        issue_types = [i.issue_type for i in report.warnings]
        assert "high_null_rate" in issue_types

    def test_missing_required_column_is_error(self):
        df = make_df().drop(columns=["heart_rate"])
        report = DataQualityChecker(required_columns=["heart_rate"]).check(df)
        assert not report.passed
        assert any(i.issue_type == "column_removed" for i in report.errors)

    def test_all_null_required_column_is_error(self):
        df = make_df()
        df["heart_rate"] = np.nan
        report = DataQualityChecker(required_columns=["heart_rate"]).check(df)
        assert not report.passed
        assert any(i.issue_type == "all_null" for i in report.errors)

    def test_min_rows_violation_is_error(self):
        df = make_df(n=10)
        report = DataQualityChecker(min_rows=100).check(df)
        assert not report.passed
        assert any(i.issue_type == "row_count_anomaly" for i in report.errors)

    def test_max_rows_violation_is_warning(self):
        df = make_df(n=200)
        report = DataQualityChecker(max_rows=50).check(df)
        assert any(i.issue_type == "row_count_anomaly" for i in report.warnings)

    def test_dtype_mismatch_is_warning(self):
        df = make_df()
        df["heart_rate"] = df["heart_rate"].astype(str)
        report = DataQualityChecker(expected_dtypes={"heart_rate": "float64"}).check(df)
        assert any(i.issue_type == "dtype_changed" for i in report.warnings)

    def test_fit_detects_removed_column(self):
        ref = make_df()
        cur = make_df().drop(columns=["heart_rate"])
        checker = DataQualityChecker()
        checker.fit(ref)
        report = checker.check(cur)
        assert any(i.issue_type == "column_removed" for i in report.issues)

    def test_fit_detects_added_column(self):
        ref = make_df()
        cur = make_df()
        cur["new_col"] = 1.0
        checker = DataQualityChecker()
        checker.fit(ref)
        report = checker.check(cur)
        assert any(i.issue_type == "column_added" for i in report.issues)

    def test_fit_detects_dtype_change(self):
        ref = make_df()
        cur = make_df()
        cur["heart_rate"] = cur["heart_rate"].astype("float32")
        checker = DataQualityChecker()
        checker.fit(ref)
        report = checker.check(cur)
        assert any(
            i.issue_type == "dtype_changed" and i.column == "heart_rate" for i in report.issues
        )

    def test_fit_warns_on_low_row_count(self):
        ref = make_df(n=200)
        cur = make_df(n=50)  # 25% of reference — below 50% threshold
        checker = DataQualityChecker()
        checker.fit(ref)
        report = checker.check(cur)
        assert any(i.issue_type == "row_count_anomaly" for i in report.warnings)

    def test_null_rates_populated(self):
        df = make_df()
        df.loc[:49, "heart_rate"] = np.nan
        report = DataQualityChecker().check(df)
        assert "heart_rate" in report.null_rates
        assert report.null_rates["heart_rate"] == pytest.approx(0.25)

    def test_to_dataframe_returns_issues(self):
        df = make_df()
        df.loc[:, "heart_rate"] = np.nan
        report = DataQualityChecker(required_columns=["heart_rate"]).check(df)
        result_df = report.to_dataframe()
        assert "issue_type" in result_df.columns

    def test_summary_string(self):
        df = make_df()
        report = DataQualityChecker().check(df)
        s = report.summary()
        assert "Passed" in s

    def test_fit_returns_self(self):
        checker = DataQualityChecker()
        assert checker.fit(make_df()) is checker
