"""
clinops.preprocess — Clinical data preprocessing utilities.

Handles the gap between raw ingested data and ML-ready features:
outlier detection and clipping using physiologically-grounded bounds,
clinical unit normalization (mg/dL ↔ mmol/L, °F ↔ °C, etc.), and
ICD-9 to ICD-10 code mapping for multi-site compatibility.

Typical usage
-------------
>>> from clinops.preprocess import ClinicalOutlierClipper, UnitNormalizer, ICDMapper
"""

from clinops.preprocess.icd import ICDMapper
from clinops.preprocess.outliers import LAB_BOUNDS, VITAL_BOUNDS, ClinicalOutlierClipper
from clinops.preprocess.units import UNIT_CONVERSIONS, UnitNormalizer

__all__ = [
    "ClinicalOutlierClipper",
    "VITAL_BOUNDS",
    "LAB_BOUNDS",
    "UnitNormalizer",
    "UNIT_CONVERSIONS",
    "ICDMapper",
]
