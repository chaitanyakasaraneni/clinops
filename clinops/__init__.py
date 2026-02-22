"""
clinops — Clinical ML Pipeline Toolkit

Bridging raw clinical data and production-ready machine learning pipelines.

v0.1 modules
------------
- clinops.ingest   : MIMIC-IV, FHIR R4, flat file loaders with schema validation
- clinops.temporal : Time-series windowing, imputation, lag features, cohort alignment

Coming in v0.2
--------------
- clinops.monitor     : Drift detection, data quality, pipeline health
- clinops.orchestrate : GCS/S3 uploads, Step Functions helpers
- clinops.explain     : LLM-generated clinical rationales for model outputs
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("clinops")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__author__ = "Chaitanya Kasaraneni"
__email__ = "kc.kasaraneni@gmail.com"
__license__ = "Apache 2.0"

__all__ = ["__version__", "__author__"]
