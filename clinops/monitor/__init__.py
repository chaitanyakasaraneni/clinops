"""
clinops.monitor — Drift detection and data quality alerting for production pipelines.

Intended for teams running clinops-based pipelines in scheduled or streaming
environments where input data distribution can shift over time without warning.

Core classes
------------
- DistributionDriftDetector: fit on reference data; detect PSI and KS drift on new batches
- DataQualityChecker: null rates, schema drift, dtype changes, row count anomalies
- DriftReport: structured result from drift detection with per-column metrics
- QualityReport: structured result from quality checks with typed issue list
"""

from clinops.monitor.drift import (
    ColumnDriftResult,
    DistributionDriftDetector,
    DriftReport,
    DriftSeverity,
)
from clinops.monitor.quality import (
    DataQualityChecker,
    QualityIssue,
    QualityReport,
)

__all__ = [
    "DistributionDriftDetector",
    "DriftReport",
    "DriftSeverity",
    "ColumnDriftResult",
    "DataQualityChecker",
    "QualityReport",
    "QualityIssue",
]
