"""
FHIR R4 resource loader.

Loads Patient, Observation, Condition, and MedicationRequest resources
from FHIR JSON bundles or NDJSON exports, normalising them into
pandas DataFrames compatible with clinops temporal and monitor modules.

Requires: pip install clinops[fhir]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

ResourceType = Literal["Patient", "Observation", "Condition", "MedicationRequest"]


class FHIRLoader:
    """
    Load FHIR R4 resources from JSON bundles or NDJSON exports.

    Parameters
    ----------
    source:
        Path to a FHIR JSON Bundle file, an NDJSON file, or a directory
        of JSON resource files.

    Examples
    --------
    >>> loader = FHIRLoader("/data/fhir_export")
    >>> observations = loader.observations()
    >>> patients     = loader.patients()
    """

    def __init__(self, source: str | Path) -> None:
        self._source = Path(source)
        if not self._source.exists():
            raise FileNotFoundError(f"FHIR source not found: {self._source}")
        logger.info(f"FHIRLoader initialised — source={self._source}")

    def patients(self) -> pd.DataFrame:
        """Load Patient resources → DataFrame with demographics."""
        records = self._load_resources("Patient")
        rows = []
        for r in records:
            rows.append(
                {
                    "patient_id": r.get("id"),
                    "gender": r.get("gender"),
                    "birth_date": r.get("birthDate"),
                    "deceased": r.get("deceasedBoolean", False),
                }
            )
        return pd.DataFrame(rows)

    def observations(
        self,
        category: str | None = None,
        loinc_codes: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load Observation resources → long-format DataFrame.

        Parameters
        ----------
        category:
            Filter to a FHIR observation category (e.g. "vital-signs", "laboratory").
        loinc_codes:
            Filter to specific LOINC codes.
        """
        records = self._load_resources("Observation")
        rows = []
        for r in records:
            code_obj = r.get("code", {})
            codings = code_obj.get("coding", [])
            loinc = next(
                (c["code"] for c in codings if "loinc" in c.get("system", "").lower()), None
            )
            value = r.get("valueQuantity", {})
            rows.append(
                {
                    "observation_id": r.get("id"),
                    "patient_id": r.get("subject", {}).get("reference", "").split("/")[-1],
                    "loinc_code": loinc,
                    "display": code_obj.get("text"),
                    "value": value.get("value"),
                    "unit": value.get("unit"),
                    "effective_time": r.get("effectiveDateTime"),
                    "status": r.get("status"),
                }
            )
        df = pd.DataFrame(rows)
        if loinc_codes:
            df = df[df["loinc_code"].isin(loinc_codes)]
        return df

    def conditions(self) -> pd.DataFrame:
        """Load Condition resources → DataFrame with ICD/SNOMED codes."""
        records = self._load_resources("Condition")
        rows = []
        for r in records:
            codings = r.get("code", {}).get("coding", [])
            rows.append(
                {
                    "condition_id": r.get("id"),
                    "patient_id": r.get("subject", {}).get("reference", "").split("/")[-1],
                    "code": codings[0].get("code") if codings else None,
                    "system": codings[0].get("system") if codings else None,
                    "display": codings[0].get("display") if codings else None,
                    "clinical_status": r.get("clinicalStatus", {})
                    .get("coding", [{}])[0]
                    .get("code"),
                    "onset": r.get("onsetDateTime"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_resources(self, resource_type: str) -> list[dict]:
        """Load all resources of a given type from the source."""
        resources: list[dict] = []

        if self._source.is_dir():
            for fp in self._source.glob("*.json"):
                resources.extend(self._parse_file(fp, resource_type))
            for fp in self._source.glob("*.ndjson"):
                resources.extend(self._parse_ndjson(fp, resource_type))
        elif self._source.suffix in (".ndjson", ".jsonl"):
            resources.extend(self._parse_ndjson(self._source, resource_type))
        else:
            resources.extend(self._parse_file(self._source, resource_type))

        logger.debug(f"Loaded {len(resources)} {resource_type} resources")
        return resources

    def _parse_file(self, path: Path, resource_type: str) -> list[dict]:
        with open(path) as f:
            data = json.load(f)
        if data.get("resourceType") == "Bundle":
            return [
                e["resource"]
                for e in data.get("entry", [])
                if e.get("resource", {}).get("resourceType") == resource_type
            ]
        if data.get("resourceType") == resource_type:
            return [data]
        return []

    def _parse_ndjson(self, path: Path, resource_type: str) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("resourceType") == resource_type:
                        records.append(r)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed NDJSON line in {path}")
        return records
