"""Tests for clinops.orchestrate — store and Step Functions pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from clinops.orchestrate import PipelineStep, StepFunctionsPipeline, StorageFormat
from clinops.orchestrate.store import GCSPipelineStore, S3PipelineStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "heart_rate": rng.normal(75, 10, n),
            "spo2": rng.normal(97, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# StorageFormat
# ---------------------------------------------------------------------------


class TestStorageFormat:
    def test_parquet_value(self):
        assert StorageFormat.PARQUET == "parquet"

    def test_csv_value(self):
        assert StorageFormat.CSV == "csv"


# ---------------------------------------------------------------------------
# GCSPipelineStore (no real GCS — test non-IO methods and lazy import error)
# ---------------------------------------------------------------------------


class TestGCSPipelineStore:
    def test_blob_path_no_prefix(self):
        store = GCSPipelineStore("my-bucket")
        assert store._blob_path("features/windows") == "features/windows.parquet"

    def test_blob_path_with_prefix(self):
        store = GCSPipelineStore("my-bucket", prefix="clinops/prod")
        assert store._blob_path("features/windows") == "clinops/prod/features/windows.parquet"

    def test_blob_path_csv_format(self):
        store = GCSPipelineStore("my-bucket", format=StorageFormat.CSV)
        assert store._blob_path("data") == "data.csv"

    def test_serialize_deserialize_parquet_roundtrip(self):
        store = GCSPipelineStore("my-bucket", format=StorageFormat.PARQUET)
        df = make_df()
        buf = store._serialize(df)
        result = store._deserialize(buf.read())
        pd.testing.assert_frame_equal(df, result)

    def test_serialize_deserialize_csv_roundtrip(self):
        store = GCSPipelineStore("my-bucket", format=StorageFormat.CSV)
        df = make_df()
        buf = store._serialize(df)
        result = store._deserialize(buf.read())
        # CSV does not preserve int64 dtypes exactly for subject_id, so compare values
        assert list(result["subject_id"]) == list(df["subject_id"])

    def test_upload_raises_import_error_without_gcp(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("google"):
                raise ImportError("No module named 'google'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        store = GCSPipelineStore("my-bucket")
        store._client = None
        with pytest.raises(ImportError, match="clinops\\[gcp\\]"):
            store._get_client()

    def test_content_type_parquet(self):
        store = GCSPipelineStore("b", format=StorageFormat.PARQUET)
        assert store._content_type() == "application/octet-stream"

    def test_content_type_csv(self):
        store = GCSPipelineStore("b", format=StorageFormat.CSV)
        assert store._content_type() == "text/csv"


# ---------------------------------------------------------------------------
# S3PipelineStore (no real S3 — test non-IO methods and lazy import error)
# ---------------------------------------------------------------------------


class TestS3PipelineStore:
    def test_key_no_prefix(self):
        store = S3PipelineStore("my-bucket")
        assert store._key("features/windows") == "features/windows.parquet"

    def test_key_with_prefix(self):
        store = S3PipelineStore("my-bucket", prefix="clinops/prod")
        assert store._key("features/windows") == "clinops/prod/features/windows.parquet"

    def test_serialize_deserialize_parquet_roundtrip(self):
        store = S3PipelineStore("my-bucket", format=StorageFormat.PARQUET)
        df = make_df()
        buf = store._serialize(df)
        result = store._deserialize(buf.read())
        pd.testing.assert_frame_equal(df, result)

    def test_upload_raises_import_error_without_aws(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        store = S3PipelineStore("my-bucket")
        store._client = None
        with pytest.raises(ImportError, match="clinops\\[aws\\]"):
            store._get_client()


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------


class TestPipelineStep:
    def test_to_asl_state_with_next(self):
        step = PipelineStep(name="Ingest", resource="arn:aws:lambda:::function:f")
        state = step.to_asl_state(next_state="Preprocess")
        assert state["Type"] == "Task"
        assert state["Next"] == "Preprocess"
        assert "End" not in state

    def test_to_asl_state_terminal(self):
        step = PipelineStep(name="Train", resource="arn:aws:lambda:::function:f")
        state = step.to_asl_state(next_state=None)
        assert state["End"] is True
        assert "Next" not in state

    def test_to_asl_state_includes_parameters(self):
        step = PipelineStep(
            name="Preprocess",
            resource="arn:aws:lambda:::function:f",
            parameters={"max_null_rate": 0.5},
        )
        state = step.to_asl_state(next_state=None)
        assert state["Parameters"] == {"max_null_rate": 0.5}

    def test_to_asl_state_no_parameters_key_when_empty(self):
        step = PipelineStep(name="S", resource="arn:aws:lambda:::function:f")
        state = step.to_asl_state(next_state=None)
        assert "Parameters" not in state

    def test_to_asl_state_timeout(self):
        step = PipelineStep(name="S", resource="arn:aws:lambda:::function:f", timeout_seconds=7200)
        state = step.to_asl_state(next_state=None)
        assert state["TimeoutSeconds"] == 7200

    def test_to_asl_state_retry_attempts(self):
        step = PipelineStep(name="S", resource="arn:aws:lambda:::function:f", retry_attempts=5)
        state = step.to_asl_state(next_state=None)
        assert state["Retry"][0]["MaxAttempts"] == 5

    def test_to_asl_state_comment(self):
        step = PipelineStep(
            name="S", resource="arn:aws:lambda:::function:f", comment="Load raw data"
        )
        state = step.to_asl_state(next_state=None)
        assert state["Comment"] == "Load raw data"

    def test_to_asl_state_no_comment_key_when_empty(self):
        step = PipelineStep(name="S", resource="arn:aws:lambda:::function:f")
        state = step.to_asl_state(next_state=None)
        assert "Comment" not in state


# ---------------------------------------------------------------------------
# StepFunctionsPipeline
# ---------------------------------------------------------------------------


class TestStepFunctionsPipeline:
    def _make_pipeline(self) -> StepFunctionsPipeline:
        return StepFunctionsPipeline(
            name="test-pipeline",
            role_arn="arn:aws:iam::123:role/SFRole",
        )

    def _add_steps(self, pipeline: StepFunctionsPipeline) -> StepFunctionsPipeline:
        pipeline.add_step(PipelineStep("Ingest", "arn:aws:lambda:::function:ingest"))
        pipeline.add_step(PipelineStep("Preprocess", "arn:aws:lambda:::function:preprocess"))
        pipeline.add_step(PipelineStep("Train", "arn:aws:lambda:::function:train"))
        return pipeline

    def test_definition_requires_steps(self):
        with pytest.raises(ValueError, match="no steps"):
            self._make_pipeline().definition()

    def test_definition_start_at_first_step(self):
        pipeline = self._add_steps(self._make_pipeline())
        defn = pipeline.definition()
        assert defn["StartAt"] == "Ingest"

    def test_definition_last_step_is_terminal(self):
        pipeline = self._add_steps(self._make_pipeline())
        defn = pipeline.definition()
        assert defn["States"]["Train"]["End"] is True

    def test_definition_intermediate_steps_have_next(self):
        pipeline = self._add_steps(self._make_pipeline())
        defn = pipeline.definition()
        assert defn["States"]["Ingest"]["Next"] == "Preprocess"
        assert defn["States"]["Preprocess"]["Next"] == "Train"

    def test_definition_all_steps_present(self):
        pipeline = self._add_steps(self._make_pipeline())
        defn = pipeline.definition()
        assert set(defn["States"].keys()) == {"Ingest", "Preprocess", "Train"}

    def test_definition_json_is_valid_json(self):
        pipeline = self._add_steps(self._make_pipeline())
        parsed = json.loads(pipeline.definition_json())
        assert "StartAt" in parsed

    def test_add_step_duplicate_name_raises(self):
        pipeline = self._make_pipeline()
        pipeline.add_step(PipelineStep("Ingest", "arn:aws:lambda:::function:ingest"))
        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_step(PipelineStep("Ingest", "arn:aws:lambda:::function:other"))

    def test_add_step_returns_self_for_chaining(self):
        pipeline = self._make_pipeline()
        result = pipeline.add_step(PipelineStep("S", "arn:aws:lambda:::function:f"))
        assert result is pipeline

    def test_deploy_raises_import_error_without_aws(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        pipeline = self._add_steps(self._make_pipeline())
        pipeline._client = None
        with pytest.raises(ImportError, match="clinops\\[aws\\]"):
            pipeline._get_client()

    def test_single_step_pipeline(self):
        pipeline = self._make_pipeline()
        pipeline.add_step(PipelineStep("OnlyStep", "arn:aws:lambda:::function:f"))
        defn = pipeline.definition()
        assert defn["StartAt"] == "OnlyStep"
        assert defn["States"]["OnlyStep"]["End"] is True
