"""
clinops.temporal — Time-series windowing, imputation, and feature engineering
for ICU and clinical time-series data.

Core classes
------------
- TemporalWindower: sliding/tumbling window extraction from long-format data
- ImputationStrategy: enum of supported imputation methods
- LagFeatureBuilder: add lag/rolling statistics to windowed data
- CohortAligner: align multiple patients' time-series to a common reference event
"""

from clinops.temporal.cohort import CohortAligner
from clinops.temporal.features import LagFeatureBuilder
from clinops.temporal.imputation import ImputationStrategy, Imputer
from clinops.temporal.windower import TemporalWindower, WindowConfig

__all__ = [
    "TemporalWindower",
    "WindowConfig",
    "ImputationStrategy",
    "Imputer",
    "LagFeatureBuilder",
    "CohortAligner",
]
