# Orchestrate

`clinops.orchestrate` provides GCS/S3 pipeline artifact storage and AWS Step Functions support for running clinops pipelines at scale.

---

## GCSPipelineStore

Upload and download DataFrames to Google Cloud Storage between pipeline steps. Uses Parquet by default for efficient storage of wide clinical DataFrames.

Requires `pip install clinops[gcp]`.

```python
from clinops.orchestrate import GCSPipelineStore, StorageFormat

store = GCSPipelineStore(
    bucket="my-clinical-bucket",
    prefix="clinops/prod/v2",
    format=StorageFormat.PARQUET,
)

# Upload a DataFrame after preprocessing
uri = store.upload(windows_df, "features/windows_2024_01")
# → 'gs://my-clinical-bucket/clinops/prod/v2/features/windows_2024_01.parquet'

# Download in a downstream step
df = store.download("features/windows_2024_01")

# List all artifacts under a sub-prefix
artifacts = store.list_artifacts("features/")
```

### Service account credentials

```python
store = GCSPipelineStore(
    bucket="my-clinical-bucket",
    credentials_path="/secrets/sa-key.json",
)
```

If `credentials_path` is None, falls back to Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS` env var or gcloud CLI).

---

## S3PipelineStore

Equivalent interface for Amazon S3.

Requires `pip install clinops[aws]`.

```python
from clinops.orchestrate import S3PipelineStore

store = S3PipelineStore(
    bucket="my-clinical-bucket",
    prefix="clinops/prod",
    region="us-east-1",
)

uri = store.upload(windows_df, "features/windows_2024_01")
# → 's3://my-clinical-bucket/clinops/prod/features/windows_2024_01.parquet'

df = store.download("features/windows_2024_01")
```

Uses the default AWS credential chain (env vars, instance profile, `~/.aws/credentials`). Pass `profile` to use a named profile:

```python
store = S3PipelineStore("my-bucket", profile="clinical-prod")
```

---

## StepFunctionsPipeline

Build, deploy, and execute a sequential AWS Step Functions state machine from a list of `PipelineStep` objects.

Requires `pip install clinops[aws]`.

```python
from clinops.orchestrate import StepFunctionsPipeline, PipelineStep

pipeline = StepFunctionsPipeline(
    name="clinops-daily-ingest",
    role_arn="arn:aws:iam::123456789:role/StepFunctionsRole",
    region="us-east-1",
)

pipeline.add_step(PipelineStep(
    name="Ingest",
    resource="arn:aws:lambda:us-east-1:123456789:function:clinops-ingest",
    parameters={"source": "mimic-iv", "date": "2024-01-01"},
    timeout_seconds=1800,
))
pipeline.add_step(PipelineStep(
    name="Preprocess",
    resource="arn:aws:lambda:us-east-1:123456789:function:clinops-preprocess",
    retry_attempts=2,
))
pipeline.add_step(PipelineStep(
    name="FeatureExtraction",
    resource="arn:aws:lambda:us-east-1:123456789:function:clinops-features",
))
```

### Inspect the definition before deploying

```python
print(pipeline.definition_json())
```

```json
{
  "Comment": "clinops-daily-ingest — built by clinops.orchestrate",
  "StartAt": "Ingest",
  "States": {
    "Ingest": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...",
      "Next": "Preprocess",
      ...
    },
    ...
  }
}
```

### Deploy and execute

```python
# Create or update the state machine in AWS
arn = pipeline.deploy()

# Start an execution
execution_arn = pipeline.execute(
    input_data={"date": "2024-01-15"},
    execution_name="daily-2024-01-15",
)
```

Each step's output is passed as the input to the next step. All steps retry up to `retry_attempts` times on Lambda service errors.

---

## StorageFormat

| Value | Description |
|---|---|
| `StorageFormat.PARQUET` | Apache Parquet (default) — efficient for wide clinical DataFrames |
| `StorageFormat.CSV` | CSV — use for interoperability with tools that cannot read Parquet |
