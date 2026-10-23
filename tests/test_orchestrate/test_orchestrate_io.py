"""I/O paths of clinops.orchestrate, exercised against in-memory fake clients.

No cloud access is needed: each test injects a fake GCS, S3, or Step Functions
client, so upload/download/list and deploy/execute logic is tested end to end.
"""

from __future__ import annotations

import io
import json
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from clinops.orchestrate import PipelineStep, StepFunctionsPipeline, StorageFormat
from clinops.orchestrate.store import GCSPipelineStore, S3PipelineStore


def make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({"subject_id": np.arange(n), "heart_rate": rng.normal(75, 10, n)})


# ---------------------------------------------------------------------------
# Fake clients
# ---------------------------------------------------------------------------


class FakeGCSClient:
    """Minimal google.cloud.storage.Client: bucket().blob() and list_blobs()."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.content_types: dict[tuple[str, str], str] = {}

    def bucket(self, name: str) -> Any:
        client = self

        class Blob:
            def __init__(self, path: str) -> None:
                self.name = path

            def upload_from_file(self, fileobj: io.BytesIO, content_type: str) -> None:
                client.objects[(name, self.name)] = fileobj.read()
                client.content_types[(name, self.name)] = content_type

            def download_as_bytes(self) -> bytes:
                return client.objects[(name, self.name)]

        return types.SimpleNamespace(blob=Blob)

    def list_blobs(self, bucket: str, prefix: str) -> list[Any]:
        return [
            types.SimpleNamespace(name=path)
            for (b, path) in self.objects
            if b == bucket and path.startswith(prefix)
        ]


class FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.pages


class FakeS3Client:
    """Minimal boto3 S3 client: upload_fileobj, download_fileobj, list_objects_v2."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def upload_fileobj(self, fileobj: io.BytesIO, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = fileobj.read()

    def download_fileobj(self, bucket: str, key: str, fileobj: io.BytesIO) -> None:
        fileobj.write(self.objects[(bucket, key)])

    def get_paginator(self, op: str) -> FakePaginator:
        assert op == "list_objects_v2"
        keys = [{"Key": k} for (_, k) in self.objects]
        # Two pages, to check that every page is read.
        return FakePaginator([{"Contents": keys[:1]}, {"Contents": keys[1:]}, {}])


class FakeSFNClient:
    """Minimal boto3 Step Functions client."""

    def __init__(self, existing: dict[str, str] | None = None) -> None:
        self.existing = existing or {}
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.started: list[dict[str, Any]] = []

    def get_paginator(self, op: str) -> FakePaginator:
        assert op == "list_state_machines"
        machines = [{"name": n, "stateMachineArn": a} for n, a in self.existing.items()]
        return FakePaginator([{"stateMachines": machines}])

    def create_state_machine(self, **kwargs: Any) -> dict[str, str]:
        self.created.append(kwargs)
        return {"stateMachineArn": f"arn:aws:states:us-east-1:123:stateMachine:{kwargs['name']}"}

    def update_state_machine(self, **kwargs: Any) -> None:
        self.updated.append(kwargs)

    def start_execution(self, **kwargs: Any) -> dict[str, str]:
        self.started.append(kwargs)
        return {"executionArn": f"arn:aws:states:us-east-1:123:execution:{kwargs['name']}"}


# ---------------------------------------------------------------------------
# GCSPipelineStore
# ---------------------------------------------------------------------------


class TestGCSPipelineStoreIO:
    @pytest.mark.parametrize("fmt", [StorageFormat.PARQUET, StorageFormat.CSV])
    def test_upload_download_roundtrip(self, fmt):
        store = GCSPipelineStore("bkt", prefix="clinops/prod", format=fmt)
        store._client = FakeGCSClient()
        df = make_df()
        uri = store.upload(df, "features/windows")
        assert uri == f"gs://bkt/clinops/prod/features/windows.{fmt}"
        result = store.download("features/windows")
        pd.testing.assert_frame_equal(result, df, check_dtype=fmt == StorageFormat.PARQUET)

    def test_upload_sets_content_type(self):
        store = GCSPipelineStore("bkt", format=StorageFormat.CSV)
        client = FakeGCSClient()
        store._client = client
        store.upload(make_df(), "a")
        assert client.content_types[("bkt", "a.csv")] == "text/csv"

    def test_list_artifacts_strips_prefix_and_extension(self):
        store = GCSPipelineStore("bkt", prefix="clinops")
        client = FakeGCSClient()
        store._client = client
        store.upload(make_df(), "features/b")
        store.upload(make_df(), "features/a")
        store.upload(make_df(), "labels")
        client.objects[("bkt", "clinops/notes.txt")] = b"ignored: wrong extension"
        assert store.list_artifacts() == ["features/a", "features/b", "labels"]
        assert store.list_artifacts("features") == ["features/a", "features/b"]

    def test_client_uses_service_account_when_credentials_path_set(self, monkeypatch):
        calls: dict[str, Any] = {}

        class FakeCreds:
            @staticmethod
            def from_service_account_file(path: str) -> str:
                calls["path"] = path
                return "creds"

        def fake_client(credentials: Any = None) -> str:
            calls["credentials"] = credentials
            return "client"

        google = types.ModuleType("google")
        cloud = types.ModuleType("google.cloud")
        storage = types.ModuleType("google.cloud.storage")
        oauth2 = types.ModuleType("google.oauth2")
        sa = types.ModuleType("google.oauth2.service_account")
        storage.Client = fake_client  # type: ignore[attr-defined]
        sa.Credentials = FakeCreds  # type: ignore[attr-defined]
        cloud.storage = storage  # type: ignore[attr-defined]
        oauth2.service_account = sa  # type: ignore[attr-defined]
        google.cloud = cloud  # type: ignore[attr-defined]
        google.oauth2 = oauth2  # type: ignore[attr-defined]
        for name, mod in {
            "google": google,
            "google.cloud": cloud,
            "google.cloud.storage": storage,
            "google.oauth2": oauth2,
            "google.oauth2.service_account": sa,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)

        store = GCSPipelineStore("bkt", credentials_path="/secrets/sa.json")
        assert store._get_client() == "client"
        assert calls == {"path": "/secrets/sa.json", "credentials": "creds"}
        assert store._get_client() == "client"  # cached, not rebuilt


# ---------------------------------------------------------------------------
# S3PipelineStore
# ---------------------------------------------------------------------------


class TestS3PipelineStoreIO:
    @pytest.mark.parametrize("fmt", [StorageFormat.PARQUET, StorageFormat.CSV])
    def test_upload_download_roundtrip(self, fmt):
        store = S3PipelineStore("bkt", prefix="clinops", format=fmt)
        store._client = FakeS3Client()
        df = make_df()
        uri = store.upload(df, "features/windows")
        assert uri == f"s3://bkt/clinops/features/windows.{fmt}"
        result = store.download("features/windows")
        pd.testing.assert_frame_equal(result, df, check_dtype=fmt == StorageFormat.PARQUET)

    def test_list_artifacts_reads_every_page(self):
        store = S3PipelineStore("bkt", prefix="clinops")
        client = FakeS3Client()
        store._client = client
        store.upload(make_df(), "b")
        store.upload(make_df(), "a")
        client.objects[("bkt", "clinops/readme.md")] = b"ignored"
        assert store.list_artifacts() == ["a", "b"]

    def test_client_built_from_boto3_session(self, monkeypatch):
        calls: dict[str, Any] = {}

        class FakeSession:
            def __init__(self, profile_name: str | None, region_name: str | None) -> None:
                calls["session"] = (profile_name, region_name)

            def client(self, service: str) -> str:
                calls["service"] = service
                return "s3-client"

        boto3 = types.ModuleType("boto3")
        boto3.Session = FakeSession  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "boto3", boto3)

        store = S3PipelineStore("bkt", region="eu-west-1", profile="research")
        assert store._get_client() == "s3-client"
        assert calls == {"session": ("research", "eu-west-1"), "service": "s3"}


# ---------------------------------------------------------------------------
# StepFunctionsPipeline deploy / execute
# ---------------------------------------------------------------------------


def make_pipeline(client: FakeSFNClient) -> StepFunctionsPipeline:
    pipeline = StepFunctionsPipeline(name="clinops-daily", role_arn="arn:aws:iam::123:role/SFNRole")
    pipeline.add_step(PipelineStep(name="Ingest", resource="arn:aws:lambda:::function:ingest"))
    pipeline.add_step(PipelineStep(name="Split", resource="arn:aws:lambda:::function:split"))
    pipeline._client = client
    return pipeline


class TestStepFunctionsIO:
    def test_deploy_creates_when_missing(self):
        client = FakeSFNClient()
        arn = make_pipeline(client).deploy()
        assert arn.endswith(":stateMachine:clinops-daily")
        (call,) = client.created
        assert call["name"] == "clinops-daily"
        assert call["roleArn"] == "arn:aws:iam::123:role/SFNRole"
        assert call["type"] == "STANDARD"
        assert json.loads(call["definition"])["StartAt"] == "Ingest"
        assert client.updated == []

    def test_deploy_updates_when_existing(self):
        existing = "arn:aws:states:us-east-1:123:stateMachine:clinops-daily"
        client = FakeSFNClient(existing={"clinops-daily": existing, "other": "arn:other"})
        assert make_pipeline(client).deploy() == existing
        (call,) = client.updated
        assert call["stateMachineArn"] == existing
        assert client.created == []

    def test_execute_looks_up_arn_and_serializes_input(self):
        existing = "arn:aws:states:us-east-1:123:stateMachine:clinops-daily"
        client = FakeSFNClient(existing={"clinops-daily": existing})
        execution = make_pipeline(client).execute({"date": "2150-01-01"}, execution_name="run-1")
        assert execution.endswith(":execution:run-1")
        (call,) = client.started
        assert call["stateMachineArn"] == existing
        assert json.loads(call["input"]) == {"date": "2150-01-01"}

    def test_execute_with_explicit_arn_and_default_name(self):
        client = FakeSFNClient()
        make_pipeline(client).execute(state_machine_arn="arn:explicit")
        (call,) = client.started
        assert call["stateMachineArn"] == "arn:explicit"
        assert call["name"].startswith("clinops-daily-")
        assert json.loads(call["input"]) == {}

    def test_execute_raises_when_not_deployed(self):
        with pytest.raises(RuntimeError, match="deploy"):
            make_pipeline(FakeSFNClient()).execute()
