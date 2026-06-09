"""
clinops.ingest.mappings — Vocabulary maps for harmonising free-text clinical
source data into clinops's canonical feature names.

Currently provides:

- ``eicu_lab_map`` : eICU-CRD ``lab.labname`` strings → canonical lab features
  used by the multi-organ deterioration model (ICHI 2026 / JBHI).
"""

from clinops.ingest.mappings.eicu_lab_map import (
    EICU_LAB_MAP,
    canonical_lab_features,
    map_labname,
)

__all__ = [
    "EICU_LAB_MAP",
    "map_labname",
    "canonical_lab_features",
]
