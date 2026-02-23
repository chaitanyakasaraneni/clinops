"""
clinops.split — Clinical-aware train/test splitting.

Standard random splits are inappropriate for clinical ML because:
- Random splits leak future information (a patient's t+1 data in train,
  t-1 data in test)
- Patient-level splits are required to prevent label leakage across
  admissions
- Outcome prevalence varies over time and must be preserved

This module provides three splitters that handle these concerns:

TemporalSplitter
    Splits on a datetime cutoff — all rows before the cutoff go to
    train, all rows after go to test. The correct approach for
    time-series clinical data.

PatientSplitter
    Ensures all admissions/visits for a given patient are in the same
    split. Prevents data leakage in multi-admission datasets.

StratifiedPatientSplitter
    Combines patient-level splitting with outcome stratification.
    Ensures the train/test outcome rate matches the population rate
    while respecting patient boundaries.
"""

from clinops.split.splitters import (
    PatientSplitter,
    SplitResult,
    StratifiedPatientSplitter,
    TemporalSplitter,
)

__all__ = [
    "TemporalSplitter",
    "PatientSplitter",
    "StratifiedPatientSplitter",
    "SplitResult",
]
