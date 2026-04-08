"""
clinops.ingest — Clinical data loaders with schema validation.

Supported sources:
- MIMIC-IV (v2.0–v2.2)
- FHIR R4
- HL7 v2 (ADT, ORU)
- Flat CSV / Parquet with configurable schemas
"""

from clinops.ingest.fhir import FHIRLoader
from clinops.ingest.flat import FlatFileLoader
from clinops.ingest.mimic import MimicLoader
from clinops.ingest.mimic_iii import MimicIIILoader
from clinops.ingest.mimic_tables import MimicTableLoader
from clinops.ingest.schema import ClinicalSchema, ColumnSpec, SchemaValidationError

__all__ = [
    "MimicLoader",
    "MimicIIILoader",
    "MimicTableLoader",
    "FHIRLoader",
    "FlatFileLoader",
    "ClinicalSchema",
    "ColumnSpec",
    "SchemaValidationError",
]
