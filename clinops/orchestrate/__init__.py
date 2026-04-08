"""
clinops.orchestrate — Cloud storage and workflow orchestration for clinical pipelines.

Provides GCS/S3 pipeline artifact storage and AWS Step Functions pipeline
definition and execution. Targets cloud-based clinical data engineering
workflows where pipeline steps need to be coordinated across distributed
infrastructure.

Optional dependencies
---------------------
GCS support requires ``pip install clinops[gcp]``.
S3 and Step Functions support requires ``pip install clinops[aws]``.

Core classes
------------
- GCSPipelineStore: upload/download pipeline artifacts to Google Cloud Storage
- S3PipelineStore: upload/download pipeline artifacts to Amazon S3
- PipelineStep: a single named step in a Step Functions state machine
- StepFunctionsPipeline: build, deploy, and execute Step Functions workflows
"""

from clinops.orchestrate.stepfunctions import PipelineStep, StepFunctionsPipeline
from clinops.orchestrate.store import GCSPipelineStore, S3PipelineStore, StorageFormat

__all__ = [
    "GCSPipelineStore",
    "S3PipelineStore",
    "StorageFormat",
    "PipelineStep",
    "StepFunctionsPipeline",
]
