"""Tests for clinops.ingest module."""

import pandas as pd
import pytest

from clinops.ingest.fhir import FHIRLoader
from clinops.ingest.flat import FlatFileLoader
from clinops.ingest.schema import ClinicalSchema, ColumnSpec, SchemaValidationError

# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestClinicalSchema:
    def test_valid_df_passes(self):
        schema = ClinicalSchema(
            name="test", columns=[ColumnSpec("subject_id", nullable=False), ColumnSpec("value")]
        )
        df = pd.DataFrame({"subject_id": [1, 2], "value": [10.0, 20.0]})
        violations = schema.validate(df, strict=False)
        assert violations == []

    def test_missing_column_raises_strict(self):
        schema = ClinicalSchema(
            name="test", columns=[ColumnSpec("subject_id"), ColumnSpec("missing_col")]
        )
        df = pd.DataFrame({"subject_id": [1, 2]})
        with pytest.raises(SchemaValidationError, match="missing_col"):
            schema.validate(df, strict=True)

    def test_missing_column_returns_violation_non_strict(self):
        schema = ClinicalSchema(
            name="test", columns=[ColumnSpec("subject_id"), ColumnSpec("missing_col")]
        )
        df = pd.DataFrame({"subject_id": [1, 2]})
        violations = schema.validate(df, strict=False)
        assert len(violations) == 1
        assert "missing_col" in violations[0]

    def test_null_check(self):
        schema = ClinicalSchema(name="test", columns=[ColumnSpec("subject_id", nullable=False)])
        df = pd.DataFrame({"subject_id": [1, None]})
        violations = schema.validate(df, strict=False)
        assert any("null" in v.lower() for v in violations)

    def test_range_check_below_min(self):
        schema = ClinicalSchema(
            name="test", columns=[ColumnSpec("heart_rate", min_value=0, max_value=300)]
        )
        df = pd.DataFrame({"heart_rate": [-5.0, 70.0, 80.0]})
        violations = schema.validate(df, strict=False)
        assert any("heart_rate" in v and "min_value" in v for v in violations)

    def test_allowed_values_check(self):
        schema = ClinicalSchema(
            name="test", columns=[ColumnSpec("gender", allowed_values=["M", "F"])]
        )
        df = pd.DataFrame({"gender": ["M", "F", "X"]})
        violations = schema.validate(df, strict=False)
        assert any("gender" in v for v in violations)

    def test_extra_columns_allowed_by_default(self):
        schema = ClinicalSchema(name="test", columns=[ColumnSpec("subject_id")])
        df = pd.DataFrame({"subject_id": [1], "extra_col": ["foo"]})
        violations = schema.validate(df, strict=False)
        assert violations == []


# ---------------------------------------------------------------------------
# FlatFileLoader tests
# ---------------------------------------------------------------------------


class TestFlatFileLoader:
    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "patient_id,heart_rate,charttime\n1,72,2023-01-01 08:00:00\n2,85,2023-01-01 09:00:00\n"
        )
        loader = FlatFileLoader(csv_file, id_col="patient_id")
        df = loader.load()
        assert len(df) == 2
        assert "heart_rate" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["charttime"])

    def test_column_name_normalisation(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("Patient ID,Heart Rate\n1,72\n")
        loader = FlatFileLoader(csv_file)
        df = loader.load()
        assert "patient_id" in df.columns
        assert "heart_rate" in df.columns

    def test_null_value_handling(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("patient_id,value\n1,N/A\n2,NULL\n3,72.0\n")
        loader = FlatFileLoader(csv_file)
        df = loader.load()
        assert df["value"].isna().sum() == 2
        assert df["value"].iloc[2] == 72.0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            FlatFileLoader("/nonexistent/path.csv")

    def test_summary_output(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("patient_id,value\n1,10\n2,20\n")
        loader = FlatFileLoader(csv_file, id_col="patient_id")
        loader.load()
        summary = loader.summary()
        assert "Rows" in summary
        assert "Unique IDs" in summary


# ---------------------------------------------------------------------------
# FHIRLoader tests
# ---------------------------------------------------------------------------


class TestFHIRLoader:
    def test_load_bundle(self, tmp_path):
        import json

        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "p1",
                        "gender": "male",
                        "birthDate": "1980-01-01",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "p2",
                        "gender": "female",
                        "birthDate": "1975-06-15",
                    }
                },
            ],
        }
        bundle_file = tmp_path / "bundle.json"
        bundle_file.write_text(json.dumps(bundle))
        loader = FHIRLoader(bundle_file)
        patients = loader.patients()
        assert len(patients) == 2
        assert set(patients["patient_id"]) == {"p1", "p2"}

    def test_load_observations(self, tmp_path):
        import json

        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "o1",
                        "subject": {"reference": "Patient/p1"},
                        "code": {
                            "coding": [{"system": "http://loinc.org", "code": "8867-4"}],
                            "text": "Heart rate",
                        },
                        "valueQuantity": {"value": 72.0, "unit": "/min"},
                        "effectiveDateTime": "2023-01-01T08:00:00Z",
                        "status": "final",
                    }
                }
            ],
        }
        obs_file = tmp_path / "obs.json"
        obs_file.write_text(json.dumps(bundle))
        loader = FHIRLoader(obs_file)
        obs = loader.observations()
        assert len(obs) == 1
        assert obs.iloc[0]["loinc_code"] == "8867-4"
        assert obs.iloc[0]["value"] == 72.0

    def test_source_not_found(self):
        with pytest.raises(FileNotFoundError):
            FHIRLoader("/nonexistent/fhir")
