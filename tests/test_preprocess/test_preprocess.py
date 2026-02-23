"""Tests for clinops.preprocess — outliers, unit normalization, ICD mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from clinops.preprocess import ClinicalOutlierClipper, ICDMapper, UnitNormalizer
from clinops.preprocess.outliers import BoundSpec
from clinops.preprocess.units import (
    UNIT_CONVERSIONS,
    celsius_to_fahrenheit,
    creatinine_mgdl_to_umol,
    fahrenheit_to_celsius,
    glucose_mgdl_to_mmol,
    glucose_mmol_to_mgdl,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def vitals_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "heart_rate": [75.0, 301.0, 0.0, 120.0, -5.0],  # 301 and -5 are outliers
            "spo2": [98.0, 102.0, 95.0, 49.0, 97.0],  # 102 and 49 are outliers
            "temperature": [37.0, 45.5, 36.5, 24.0, 38.2],  # 45.5 and 24.0 are outliers
        }
    )


@pytest.fixture
def glucose_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "glucose": [126.0, 5.5, 180.0, 7.2],
            "glucose_unit": ["mg/dL", "mmol/L", "mg/dL", "mmol/L"],
        }
    )


@pytest.fixture
def icd_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "icd_code": ["41401", "4280", "42731", "I10", "49121"],
            "icd_version": ["9", "9", "9", "10", "9"],
        }
    )


# -----------------------------------------------------------------------
# ClinicalOutlierClipper
# -----------------------------------------------------------------------


class TestClinicalOutlierClipper:
    def test_clip_action_replaces_with_boundary(self, vitals_df):
        clipper = ClinicalOutlierClipper(action="clip")
        result = clipper.fit_transform(vitals_df)
        assert result["heart_rate"].max() <= 300
        assert result["heart_rate"].min() >= 0
        assert result["spo2"].max() <= 100
        assert result["spo2"].min() >= 50

    def test_clip_preserves_valid_values(self, vitals_df):
        clipper = ClinicalOutlierClipper(action="clip")
        result = clipper.fit_transform(vitals_df)
        # Valid heart rates should be unchanged
        assert result.loc[0, "heart_rate"] == 75.0
        assert result.loc[3, "heart_rate"] == 120.0

    def test_null_action_replaces_with_nan(self, vitals_df):
        clipper = ClinicalOutlierClipper(action="null")
        result = clipper.fit_transform(vitals_df)
        # heart_rate of 300 and -5 should become NaN
        assert pd.isna(result.loc[1, "heart_rate"])  # 300 → NaN
        assert pd.isna(result.loc[4, "heart_rate"])  # -5 → NaN
        # Valid value preserved
        assert result.loc[0, "heart_rate"] == 75.0

    def test_flag_action_adds_indicator_column(self, vitals_df):
        clipper = ClinicalOutlierClipper(action="flag")
        result = clipper.fit_transform(vitals_df)
        assert "heart_rate_outlier" in result.columns
        assert result.loc[1, "heart_rate_outlier"] == 1  # 300 is outlier
        assert result.loc[0, "heart_rate_outlier"] == 0  # 75 is not

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action must be"):
            ClinicalOutlierClipper(action="remove")

    def test_report_contains_outlier_summary(self, vitals_df):
        clipper = ClinicalOutlierClipper(action="clip")
        clipper.fit_transform(vitals_df)
        report = clipper.report()
        assert isinstance(report, pd.DataFrame)
        assert "column" in report.columns
        assert "total_outliers" in report.columns
        assert len(report) > 0

    def test_no_outliers_returns_empty_report(self):
        df = pd.DataFrame(
            {
                "heart_rate": [60.0, 75.0, 80.0],
                "spo2": [97.0, 98.0, 99.0],
            }
        )
        clipper = ClinicalOutlierClipper(action="clip")
        clipper.fit_transform(df)
        report = clipper.report()
        assert len(report) == 0

    def test_custom_bounds_override_defaults(self):
        custom = {"glucose": BoundSpec("glucose", low=50, high=400, unit="mg/dL")}
        df = pd.DataFrame({"glucose": [45.0, 200.0, 450.0]})
        clipper = ClinicalOutlierClipper(bounds=custom, action="clip")
        result = clipper.fit_transform(df)
        assert result["glucose"].min() >= 50
        assert result["glucose"].max() <= 400

    def test_extra_bounds_merge_with_defaults(self):
        extra = {"custom_score": BoundSpec("custom_score", 0, 10)}
        df = pd.DataFrame(
            {
                "heart_rate": [75.0, 350.0],
                "custom_score": [5.0, 15.0],
            }
        )
        clipper = ClinicalOutlierClipper(action="clip", extra_bounds=extra)
        result = clipper.fit_transform(df)
        assert result["custom_score"].max() <= 10
        assert result["heart_rate"].max() <= 300

    def test_strict_mode_raises_on_missing_column(self):
        df = pd.DataFrame({"unknown_col": [1.0, 2.0]})
        bounds = {"heart_rate": BoundSpec("heart_rate", 0, 300)}
        clipper = ClinicalOutlierClipper(bounds=bounds, action="clip", strict=True)
        with pytest.raises(ValueError, match="not found in DataFrame"):
            clipper.fit_transform(df)

    def test_non_numeric_columns_skipped(self):
        df = pd.DataFrame(
            {
                "heart_rate": [75.0, 350.0],
                "label": ["normal", "abnormal"],
            }
        )
        clipper = ClinicalOutlierClipper(action="clip")
        result = clipper.fit_transform(df)
        assert list(result["label"]) == ["normal", "abnormal"]

    def test_add_bounds_updates_registry(self):
        clipper = ClinicalOutlierClipper(action="clip")
        clipper.add_bounds("custom_var", 0, 100)
        df = pd.DataFrame({"custom_var": [50.0, 150.0, -10.0]})
        result = clipper.fit_transform(df)
        assert result["custom_var"].max() <= 100
        assert result["custom_var"].min() >= 0


# -----------------------------------------------------------------------
# UnitNormalizer
# -----------------------------------------------------------------------


class TestUnitNormalizer:
    def test_explicit_celsius_to_fahrenheit(self):
        df = pd.DataFrame({"temperature": [0.0, 37.0, 100.0]})
        spec = UNIT_CONVERSIONS["temperature__c__f"]
        normalizer = UnitNormalizer(explicit_conversions={"temperature": spec})
        result = normalizer.transform(df)
        assert abs(result.loc[0, "temperature"] - 32.0) < 0.01
        assert abs(result.loc[1, "temperature"] - 98.6) < 0.01
        assert abs(result.loc[2, "temperature"] - 212.0) < 0.01

    def test_explicit_fahrenheit_to_celsius(self):
        df = pd.DataFrame({"temperature": [32.0, 98.6, 212.0]})
        spec = UNIT_CONVERSIONS["temperature__f__c"]
        normalizer = UnitNormalizer(explicit_conversions={"temperature": spec})
        result = normalizer.transform(df)
        assert abs(result.loc[0, "temperature"] - 0.0) < 0.01
        assert abs(result.loc[1, "temperature"] - 37.0) < 0.1
        assert abs(result.loc[2, "temperature"] - 100.0) < 0.01

    def test_unit_column_mixed_glucose(self, glucose_df):
        normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
        result = normalizer.transform(glucose_df)
        # mmol/L rows should now be in mg/dL (~18x)
        assert result.loc[1, "glucose"] == pytest.approx(5.5 * 18.018, rel=0.01)
        assert result.loc[3, "glucose"] == pytest.approx(7.2 * 18.018, rel=0.01)
        # mg/dL rows should be unchanged
        assert result.loc[0, "glucose"] == 126.0
        assert result.loc[2, "glucose"] == 180.0

    def test_unit_column_updates_unit_col_after_conversion(self, glucose_df):
        normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
        result = normalizer.transform(glucose_df)
        # All unit rows should now be mg/dL
        assert (result["glucose_unit"] == "mg/dL").all()

    def test_missing_value_column_skips_gracefully(self):
        df = pd.DataFrame({"not_glucose": [100.0]})
        normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
        # Should not raise — gracefully skips missing column
        result = normalizer.transform(df)
        assert "not_glucose" in result.columns

    def test_report_returns_conversion_summary(self, glucose_df):
        normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
        normalizer.transform(glucose_df)
        report = normalizer.report()
        assert isinstance(report, pd.DataFrame)
        assert "n_converted" in report.columns
        assert len(report) > 0

    def test_no_conversions_returns_empty_report(self):
        normalizer = UnitNormalizer()
        normalizer.transform(pd.DataFrame({"x": [1.0]}))
        report = normalizer.report()
        assert len(report) == 0

    def test_available_conversions_lists_keys(self):
        keys = UnitNormalizer.available_conversions()
        assert "glucose__mg_dl__mmol_l" in keys
        assert "temperature__c__f" in keys

    def test_convenience_functions_roundtrip(self):
        s = pd.Series([0.0, 37.0, 100.0])
        assert (celsius_to_fahrenheit(fahrenheit_to_celsius(celsius_to_fahrenheit(s)))).equals(
            celsius_to_fahrenheit(s)
        )

    def test_glucose_conversion_roundtrip(self):
        mg_dl = pd.Series([126.0, 180.0, 54.0])
        result = glucose_mmol_to_mgdl(glucose_mgdl_to_mmol(mg_dl))
        for orig, conv in zip(mg_dl, result, strict=False):
            assert abs(orig - conv) < 0.01

    def test_creatinine_mgdl_to_umol(self):
        s = pd.Series([1.0])
        result = creatinine_mgdl_to_umol(s)
        assert abs(result.iloc[0] - 88.42) < 0.01


# -----------------------------------------------------------------------
# ICDMapper
# -----------------------------------------------------------------------


class TestICDMapper:
    def test_map_known_code(self):
        mapper = ICDMapper()
        assert mapper.map_code("4280") == "I509"

    def test_map_unknown_code_returns_default(self):
        mapper = ICDMapper(default_value="UNKNOWN")
        assert mapper.map_code("99999") == "UNKNOWN"

    def test_map_code_strips_decimal(self):
        mapper = ICDMapper()
        # "414.01" should map same as "41401"
        assert mapper.map_code("414.01") == mapper.map_code("41401")

    def test_map_series_converts_column(self):
        mapper = ICDMapper()
        s = pd.Series(["41401", "4280", "42731"])
        result = mapper.map_series(s)
        assert result.iloc[0] == "I2510"
        assert result.iloc[1] == "I509"
        assert result.iloc[2] == "I4891"

    def test_map_series_unmapped_becomes_nan(self):
        mapper = ICDMapper()
        s = pd.Series(["00000"])
        result = mapper.map_series(s)
        assert pd.isna(result.iloc[0])

    def test_harmonize_converts_icd9_rows_only(self, icd_df):
        mapper = ICDMapper()
        result = mapper.harmonize(icd_df, code_col="icd_code", version_col="icd_version")
        # ICD-9 row 0: "41401" → "I2510"
        assert result.loc[0, "icd_code"] == "I2510"
        # ICD-10 row 3: "I10" should be unchanged
        assert result.loc[3, "icd_code"] == "I10"

    def test_harmonize_preserves_original_df(self, icd_df):
        mapper = ICDMapper()
        mapper.harmonize(icd_df, code_col="icd_code", version_col="icd_version")
        # Original should be unchanged (harmonize returns copy)
        assert icd_df.loc[0, "icd_code"] == "41401"

    def test_chapter_cardiovascular(self):
        mapper = ICDMapper()
        assert mapper.chapter("I2510") == "Diseases of the circulatory system"

    def test_chapter_respiratory(self):
        mapper = ICDMapper()
        assert mapper.chapter("J189") == "Diseases of the respiratory system"

    def test_chapter_invalid_code_returns_unknown(self):
        mapper = ICDMapper()
        assert mapper.chapter("XXXXX") == "Unknown"

    def test_chapter_series(self):
        mapper = ICDMapper()
        s = pd.Series(["I2510", "J189", "K259"])
        result = mapper.chapter_series(s)
        assert result.iloc[0] == "Diseases of the circulatory system"
        assert result.iloc[1] == "Diseases of the respiratory system"
        assert result.iloc[2] == "Diseases of the digestive system"

    def test_n_mappings_gt_zero(self):
        mapper = ICDMapper()
        assert mapper.n_mappings > 0

    def test_custom_mappings(self):
        custom = [("99999", "Z9999", "Test mapping")]
        mapper = ICDMapper(mappings=custom)
        assert mapper.map_code("99999") == "Z9999"
        assert mapper.n_mappings == 1

    def test_describe_known_code(self):
        mapper = ICDMapper()
        desc = mapper.describe("4280")
        assert desc != "No description available"
        assert len(desc) > 0
