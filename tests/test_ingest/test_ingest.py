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

    def test_observations_filter_by_loinc_code(self, tmp_path):
        import json

        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "o1",
                        "subject": {"reference": "Patient/p1"},
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
                        "valueQuantity": {"value": 72.0, "unit": "/min"},
                        "effectiveDateTime": "2023-01-01T08:00:00Z",
                        "status": "final",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "o2",
                        "subject": {"reference": "Patient/p1"},
                        "code": {"coding": [{"system": "http://loinc.org", "code": "59408-5"}]},
                        "valueQuantity": {"value": 98.0, "unit": "%"},
                        "effectiveDateTime": "2023-01-01T08:00:00Z",
                        "status": "final",
                    }
                },
            ],
        }
        f = tmp_path / "obs.json"
        f.write_text(json.dumps(bundle))
        loader = FHIRLoader(f)
        df = loader.observations(loinc_codes=["8867-4"])
        assert len(df) == 1
        assert df.iloc[0]["loinc_code"] == "8867-4"

    def test_conditions_basic(self, tmp_path):
        import json

        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "c1",
                        "subject": {"reference": "Patient/p1"},
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "73211009",
                                    "display": "Diabetes mellitus",
                                }
                            ]
                        },
                        "clinicalStatus": {"coding": [{"code": "active"}]},
                        "onsetDateTime": "2020-03-01",
                    }
                }
            ],
        }
        f = tmp_path / "conditions.json"
        f.write_text(json.dumps(bundle))
        loader = FHIRLoader(f)
        df = loader.conditions()
        assert len(df) == 1
        assert df.iloc[0]["condition_id"] == "c1"
        assert df.iloc[0]["patient_id"] == "p1"
        assert df.iloc[0]["code"] == "73211009"
        assert df.iloc[0]["system"] == "http://snomed.info/sct"
        assert df.iloc[0]["display"] == "Diabetes mellitus"
        assert df.iloc[0]["clinical_status"] == "active"
        assert df.iloc[0]["onset"] == "2020-03-01"

    def test_conditions_no_codings_gives_none(self, tmp_path):
        import json

        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "c2",
                        "subject": {"reference": "Patient/p2"},
                        "code": {},  # no coding list
                    }
                }
            ],
        }
        f = tmp_path / "conditions_empty.json"
        f.write_text(json.dumps(bundle))
        loader = FHIRLoader(f)
        df = loader.conditions()
        assert len(df) == 1
        assert df.iloc[0]["code"] is None
        assert df.iloc[0]["system"] is None
        assert df.iloc[0]["display"] is None

    def test_conditions_returns_dataframe(self, tmp_path):
        import json

        bundle = {"resourceType": "Bundle", "entry": []}
        f = tmp_path / "empty.json"
        f.write_text(json.dumps(bundle))
        loader = FHIRLoader(f)
        df = loader.conditions()
        assert isinstance(df, pd.DataFrame)

    def test_load_from_directory_json(self, tmp_path):
        import json

        (tmp_path / "patients.json").write_text(
            json.dumps(
                {
                    "resourceType": "Bundle",
                    "entry": [
                        {"resource": {"resourceType": "Patient", "id": "p1", "gender": "male"}}
                    ],
                }
            )
        )
        loader = FHIRLoader(tmp_path)
        df = loader.patients()
        assert len(df) == 1
        assert df.iloc[0]["patient_id"] == "p1"

    def test_load_from_directory_ndjson(self, tmp_path):
        (tmp_path / "patients.ndjson").write_text(
            '{"resourceType": "Patient", "id": "p2", "gender": "female"}\n'
        )
        loader = FHIRLoader(tmp_path)
        df = loader.patients()
        assert len(df) == 1
        assert df.iloc[0]["patient_id"] == "p2"

    def test_load_from_directory_mixed_files(self, tmp_path):
        import json

        (tmp_path / "patients.json").write_text(
            json.dumps(
                {
                    "resourceType": "Bundle",
                    "entry": [
                        {"resource": {"resourceType": "Patient", "id": "p1", "gender": "male"}}
                    ],
                }
            )
        )
        (tmp_path / "more_patients.ndjson").write_text(
            '{"resourceType": "Patient", "id": "p2", "gender": "female"}\n'
        )
        loader = FHIRLoader(tmp_path)
        df = loader.patients()
        assert len(df) == 2
        assert set(df["patient_id"]) == {"p1", "p2"}

    def test_load_ndjson_file_directly(self, tmp_path):
        f = tmp_path / "patients.ndjson"
        f.write_text(
            '{"resourceType": "Patient", "id": "p1", "gender": "male"}\n'
            '{"resourceType": "Patient", "id": "p2", "gender": "female"}\n'
        )
        loader = FHIRLoader(f)
        df = loader.patients()
        assert len(df) == 2
        assert set(df["patient_id"]) == {"p1", "p2"}

    def test_load_jsonl_file_directly(self, tmp_path):
        f = tmp_path / "patients.jsonl"
        f.write_text('{"resourceType": "Patient", "id": "p3", "gender": "male"}\n')
        loader = FHIRLoader(f)
        df = loader.patients()
        assert len(df) == 1
        assert df.iloc[0]["patient_id"] == "p3"

    def test_parse_single_resource_json(self, tmp_path):
        import json

        f = tmp_path / "patient.json"
        f.write_text(json.dumps({"resourceType": "Patient", "id": "p1", "gender": "male"}))
        loader = FHIRLoader(f)
        df = loader.patients()
        assert len(df) == 1
        assert df.iloc[0]["patient_id"] == "p1"

    def test_parse_file_wrong_resource_type_returns_empty(self, tmp_path):
        import json

        # File is a single Observation — loader.patients() should return empty (line 155)
        f = tmp_path / "obs.json"
        f.write_text(json.dumps({"resourceType": "Observation", "id": "o1"}))
        loader = FHIRLoader(f)
        df = loader.patients()
        assert len(df) == 0

    def test_ndjson_skips_blank_lines(self, tmp_path):
        f = tmp_path / "patients.ndjson"
        f.write_text(
            '{"resourceType": "Patient", "id": "p1"}\n\n{"resourceType": "Patient", "id": "p2"}\n'
        )
        loader = FHIRLoader(f)
        df = loader.patients()
        assert len(df) == 2

    def test_ndjson_skips_wrong_resource_type(self, tmp_path):
        f = tmp_path / "mixed.ndjson"
        f.write_text(
            '{"resourceType": "Patient", "id": "p1"}\n{"resourceType": "Observation", "id": "o1"}\n'
        )
        loader = FHIRLoader(f)
        df = loader.patients()
        assert len(df) == 1
        assert df.iloc[0]["patient_id"] == "p1"

    def test_ndjson_skips_malformed_lines(self, tmp_path):
        # line 168-169: JSONDecodeError caught, warning logged, parsing continues
        f = tmp_path / "bad.ndjson"
        f.write_text(
            '{"resourceType": "Patient", "id": "p1"}\n'
            "THIS IS NOT JSON\n"
            '{"resourceType": "Patient", "id": "p2"}\n'
        )
        loader = FHIRLoader(f)
        df = loader.patients()
        assert len(df) == 2
        assert set(df["patient_id"]) == {"p1", "p2"}


class TestReviewedParameters:
    """Documented parameters that used to be ignored (clinops#17)."""

    def _obs(self, oid, loinc, category):
        return {
            "resource": {
                "resourceType": "Observation",
                "id": oid,
                "subject": {"reference": "Patient/p1"},
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": category,
                            }
                        ]
                    }
                ],
                "code": {"coding": [{"system": "http://loinc.org", "code": loinc}]},
                "valueQuantity": {"value": 1.0},
            }
        }

    def _loader(self, tmp_path):
        import json

        bundle = {
            "resourceType": "Bundle",
            "entry": [
                self._obs("hr", "8867-4", "vital-signs"),
                self._obs("cr", "2160-0", "laboratory"),
            ],
        }
        path = tmp_path / "obs.json"
        path.write_text(json.dumps(bundle))
        return FHIRLoader(path)

    def test_fhir_category_filter(self, tmp_path):
        loader = self._loader(tmp_path)
        assert loader.observations()["observation_id"].tolist() == ["hr", "cr"]
        assert loader.observations(category="vital-signs")["observation_id"].tolist() == ["hr"]
        assert loader.observations(category="laboratory")["observation_id"].tolist() == ["cr"]

    def test_fhir_unknown_category_returns_empty_frame_with_columns(self, tmp_path):
        obs = self._loader(tmp_path).observations(category="no-such-cat", loinc_codes=["8867-4"])
        assert obs.empty
        assert "loinc_code" in obs.columns

    def test_dtype_mismatch_reported(self):
        schema = ClinicalSchema(name="v", columns=[ColumnSpec("heart_rate", dtype="float64")])
        violations = schema.validate(pd.DataFrame({"heart_rate": ["80", "90"]}), strict=False)
        assert any("dtype" in v for v in violations)
        with pytest.raises(SchemaValidationError, match="dtype"):
            schema.validate(pd.DataFrame({"heart_rate": ["80"]}))

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [
            ("float64", [80, 90]),  # integer readings satisfy a float spec
            ("float64", [80.5, None]),
            ("int64", pd.array([1, None], dtype="Int64")),
            ("str", ["a", "b"]),
            ("datetime64[ns]", pd.to_datetime(["2150-01-01"] * 2)),
            ("bool", [True, False]),
        ],
    )
    def test_dtype_family_matches(self, dtype, values):
        schema = ClinicalSchema(name="v", columns=[ColumnSpec("c", dtype=dtype)])
        assert schema.validate(pd.DataFrame({"c": values})) == []

    def test_allow_extra_columns_false(self):
        schema = ClinicalSchema(
            name="v", columns=[ColumnSpec("heart_rate")], allow_extra_columns=False
        )
        df = pd.DataFrame({"heart_rate": [80.0], "note": ["a"]})
        with pytest.raises(SchemaValidationError, match="note"):
            schema.validate(df)
        assert schema.validate(df[["heart_rate"]]) == []

    def test_extra_columns_allowed_by_default(self):
        schema = ClinicalSchema(name="v", columns=[ColumnSpec("heart_rate")])
        assert schema.validate(pd.DataFrame({"heart_rate": [80.0], "note": ["a"]})) == []

    def test_range_on_text_column_reported(self):
        # The reviewer's example: every check used to pass silently.
        schema = ClinicalSchema(
            name="vitals",
            columns=[ColumnSpec("heart_rate", dtype="float64", min_value=0, max_value=300)],
            allow_extra_columns=False,
        )
        df = pd.DataFrame({"heart_rate": ["80", "9999"], "note": ["a", "b"]})
        violations = schema.validate(df, strict=False)
        assert len(violations) == 3  # extra column, dtype, non-numeric range check
