"""Tests for clinops.split — temporal, patient-level, and stratified splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinops.split import (
    PatientSplitter,
    SplitResult,
    StratifiedPatientSplitter,
    TemporalSplitter,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def temporal_df() -> pd.DataFrame:
    """50 rows spanning 2 years with a binary outcome."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2150-01-01", periods=50, freq="W")
    return pd.DataFrame(
        {
            "subject_id": rng.integers(1, 20, size=50),
            "charttime": dates,
            "heart_rate": rng.normal(80, 15, 50),
            "outcome": rng.integers(0, 2, size=50),
        }
    )


@pytest.fixture
def patient_df() -> pd.DataFrame:
    """100 rows, 20 unique patients, 5 rows each."""
    rng = np.random.default_rng(1)
    patients = np.repeat(np.arange(1, 21), 5)
    return pd.DataFrame(
        {
            "subject_id": patients,
            "value": rng.normal(0, 1, 100),
            "outcome": rng.integers(0, 2, size=100),
        }
    )


@pytest.fixture
def stratified_df() -> pd.DataFrame:
    """200 rows, 40 patients; 10 are positive (outcome=1 on all rows)."""
    rows = []
    for pid in range(1, 41):
        outcome = 1 if pid <= 10 else 0
        for _ in range(5):
            rows.append({"subject_id": pid, "value": pid * 1.0, "outcome": outcome})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# SplitResult
# -----------------------------------------------------------------------


class TestSplitResult:

    def test_train_size_and_test_size(self):
        train = pd.DataFrame({"x": range(80)})
        test = pd.DataFrame({"x": range(20)})
        result = SplitResult(train=train, test=test)
        assert result.train_size == 80
        assert result.test_size == 20

    def test_train_frac(self):
        train = pd.DataFrame({"x": range(80)})
        test = pd.DataFrame({"x": range(20)})
        result = SplitResult(train=train, test=test)
        assert abs(result.train_frac - 0.8) < 0.001

    def test_summary_returns_string(self):
        train = pd.DataFrame({"x": range(80)})
        test = pd.DataFrame({"x": range(20)})
        result = SplitResult(train=train, test=test, metadata={"cutoff": "2155-01-01"})
        s = result.summary()
        assert "Train" in s
        assert "Test" in s
        assert "cutoff" in s


# -----------------------------------------------------------------------
# TemporalSplitter
# -----------------------------------------------------------------------


class TestTemporalSplitter:

    def test_explicit_cutoff_splits_correctly(self, temporal_df):
        cutoff = "2150-07-01"
        splitter = TemporalSplitter(cutoff=cutoff, time_col="charttime")
        result = splitter.split(temporal_df)
        assert pd.to_datetime(result.train["charttime"]).max() < pd.Timestamp(cutoff)
        assert pd.to_datetime(result.test["charttime"]).min() >= pd.Timestamp(cutoff)

    def test_no_overlap_between_splits(self, temporal_df):
        splitter = TemporalSplitter(cutoff="2150-07-01", time_col="charttime")
        result = splitter.split(temporal_df)
        train_times = set(result.train["charttime"].astype(str))
        test_times = set(result.test["charttime"].astype(str))
        assert train_times.isdisjoint(test_times)

    def test_auto_cutoff_uses_train_frac(self, temporal_df):
        splitter = TemporalSplitter(train_frac=0.8, time_col="charttime")
        result = splitter.split(temporal_df)
        actual_frac = result.train_size / (result.train_size + result.test_size)
        # Allow ±15% tolerance due to weekly frequency rounding
        assert 0.65 <= actual_frac <= 0.95

    def test_metadata_contains_cutoff(self, temporal_df):
        splitter = TemporalSplitter(cutoff="2150-07-01", time_col="charttime")
        result = splitter.split(temporal_df)
        assert "cutoff" in result.metadata

    def test_missing_time_col_raises(self, temporal_df):
        splitter = TemporalSplitter(cutoff="2150-07-01", time_col="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            splitter.split(temporal_df)

    def test_train_test_cover_all_rows(self, temporal_df):
        splitter = TemporalSplitter(cutoff="2150-07-01", time_col="charttime")
        result = splitter.split(temporal_df)
        assert result.train_size + result.test_size == len(temporal_df)

    def test_empty_test_when_cutoff_after_all_data(self, temporal_df):
        splitter = TemporalSplitter(cutoff="2200-01-01", time_col="charttime")
        result = splitter.split(temporal_df)
        assert result.test_size == 0
        assert result.train_size == len(temporal_df)


# -----------------------------------------------------------------------
# PatientSplitter
# -----------------------------------------------------------------------


class TestPatientSplitter:

    def test_no_patient_leakage(self, patient_df):
        splitter = PatientSplitter(id_col="subject_id", test_size=0.2)
        result = splitter.split(patient_df)
        train_patients = set(result.train["subject_id"])
        test_patients = set(result.test["subject_id"])
        assert train_patients.isdisjoint(test_patients)

    def test_all_patients_accounted_for(self, patient_df):
        splitter = PatientSplitter(id_col="subject_id", test_size=0.2)
        result = splitter.split(patient_df)
        all_patients_in = set(result.train["subject_id"]) | set(result.test["subject_id"])
        all_patients_orig = set(patient_df["subject_id"])
        assert all_patients_in == all_patients_orig

    def test_all_rows_accounted_for(self, patient_df):
        splitter = PatientSplitter(id_col="subject_id", test_size=0.2)
        result = splitter.split(patient_df)
        assert result.train_size + result.test_size == len(patient_df)

    def test_test_size_approximately_correct(self, patient_df):
        splitter = PatientSplitter(id_col="subject_id", test_size=0.2)
        result = splitter.split(patient_df)
        n_test_patients = result.metadata["n_test_patients"]
        n_total_patients = patient_df["subject_id"].nunique()
        actual_test_frac = n_test_patients / n_total_patients
        assert 0.1 <= actual_test_frac <= 0.35

    def test_random_state_reproducible(self, patient_df):
        s1 = PatientSplitter(random_state=42).split(patient_df)
        s2 = PatientSplitter(random_state=42).split(patient_df)
        assert list(s1.train["subject_id"]) == list(s2.train["subject_id"])

    def test_different_random_states_give_different_splits(self, patient_df):
        s1 = PatientSplitter(random_state=0).split(patient_df)
        s2 = PatientSplitter(random_state=99).split(patient_df)
        assert set(s1.test["subject_id"]) != set(s2.test["subject_id"])

    def test_invalid_test_size_raises(self):
        with pytest.raises(ValueError, match="test_size must be between"):
            PatientSplitter(test_size=1.5)

    def test_missing_id_col_raises(self, patient_df):
        splitter = PatientSplitter(id_col="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            splitter.split(patient_df)

    def test_metadata_contains_patient_counts(self, patient_df):
        splitter = PatientSplitter(id_col="subject_id")
        result = splitter.split(patient_df)
        assert "n_train_patients" in result.metadata
        assert "n_test_patients" in result.metadata


# -----------------------------------------------------------------------
# StratifiedPatientSplitter
# -----------------------------------------------------------------------


class TestStratifiedPatientSplitter:

    def test_no_patient_leakage(self, stratified_df):
        splitter = StratifiedPatientSplitter(
            id_col="subject_id", outcome_col="outcome", test_size=0.2
        )
        result = splitter.split(stratified_df)
        train_patients = set(result.train["subject_id"])
        test_patients = set(result.test["subject_id"])
        assert train_patients.isdisjoint(test_patients)

    def test_all_rows_accounted_for(self, stratified_df):
        splitter = StratifiedPatientSplitter(id_col="subject_id", outcome_col="outcome")
        result = splitter.split(stratified_df)
        assert result.train_size + result.test_size == len(stratified_df)

    def test_outcome_rate_approximately_preserved(self, stratified_df):
        # Population: 10/40 = 25% positive
        splitter = StratifiedPatientSplitter(
            id_col="subject_id", outcome_col="outcome", test_size=0.25
        )
        result = splitter.split(stratified_df)
        pop_rate = stratified_df["outcome"].mean()
        test_rate = result.metadata["test_outcome_rate"]
        # Allow ±10pp tolerance
        assert abs(test_rate - pop_rate) < 0.10

    def test_metadata_contains_outcome_rates(self, stratified_df):
        splitter = StratifiedPatientSplitter(id_col="subject_id", outcome_col="outcome")
        result = splitter.split(stratified_df)
        assert "population_outcome_rate" in result.metadata
        assert "train_outcome_rate" in result.metadata
        assert "test_outcome_rate" in result.metadata

    def test_invalid_test_size_raises(self):
        with pytest.raises(ValueError):
            StratifiedPatientSplitter(test_size=0.0)

    def test_missing_outcome_col_raises(self, stratified_df):
        splitter = StratifiedPatientSplitter(outcome_col="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            splitter.split(stratified_df)

    def test_reproducible_with_same_seed(self, stratified_df):
        s1 = StratifiedPatientSplitter(
            id_col="subject_id", outcome_col="outcome", random_state=42
        ).split(stratified_df)
        s2 = StratifiedPatientSplitter(
            id_col="subject_id", outcome_col="outcome", random_state=42
        ).split(stratified_df)
        assert set(s1.test["subject_id"]) == set(s2.test["subject_id"])

    def test_summary_includes_outcome_rates(self, stratified_df):
        splitter = StratifiedPatientSplitter(id_col="subject_id", outcome_col="outcome")
        result = splitter.split(stratified_df)
        summary = result.summary()
        assert "outcome_rate" in summary
