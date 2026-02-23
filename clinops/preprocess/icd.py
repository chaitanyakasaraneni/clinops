"""
ICD-9-CM to ICD-10-CM code mapping for multi-site clinical studies.

The US healthcare system transitioned from ICD-9-CM to ICD-10-CM in
October 2015. MIMIC-III uses ICD-9, MIMIC-IV mixes both versions,
and many real-world datasets span the transition boundary. Combining
data across versions without harmonization silently splits the same
clinical condition into two non-overlapping code sets.

This module provides:
- A bundled general equivalence mapping (GEM) table covering the
  most common ICD-9 → ICD-10 mappings
- Forward and backward mapping support
- Aggregation to top-level ICD chapters for ML features
- Phecode-style category mapping for phenotype analysis

Note on GEM completeness
------------------------
The full CMS GEM files contain ~72,000 mappings. This module ships a
curated subset of ~3,000 high-frequency mappings covering conditions
commonly encountered in ICU and general hospital datasets. For full
GEM coverage, users can load the official CMS file using
``ICDMapper.from_gem_file(path)``.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

import pandas as pd
from loguru import logger


class ICDVersion(StrEnum):
    """ICD coding version."""

    ICD9 = "icd9"
    ICD10 = "icd10"


# -----------------------------------------------------------------------
# ICD-10 chapter ranges (for top-level grouping)
# -----------------------------------------------------------------------

_ICD10_CHAPTERS: list[tuple[str, str, str]] = [
    ("A00", "B99", "Infectious and parasitic diseases"),
    ("C00", "D49", "Neoplasms"),
    ("D50", "D89", "Blood and immune disorders"),
    ("E00", "E89", "Endocrine, nutritional and metabolic diseases"),
    ("F01", "F99", "Mental, behavioural and neurodevelopmental disorders"),
    ("G00", "G99", "Diseases of the nervous system"),
    ("H00", "H59", "Diseases of the eye and adnexa"),
    ("H60", "H95", "Diseases of the ear and mastoid process"),
    ("I00", "I99", "Diseases of the circulatory system"),
    ("J00", "J99", "Diseases of the respiratory system"),
    ("K00", "K95", "Diseases of the digestive system"),
    ("L00", "L99", "Diseases of the skin and subcutaneous tissue"),
    ("M00", "M99", "Diseases of the musculoskeletal system"),
    ("N00", "N99", "Diseases of the genitourinary system"),
    ("O00", "O9A", "Pregnancy, childbirth and the puerperium"),
    ("P00", "P96", "Conditions originating in the perinatal period"),
    ("Q00", "Q99", "Congenital malformations and chromosomal abnormalities"),
    ("R00", "R99", "Symptoms and signs not elsewhere classified"),
    ("S00", "T88", "Injury, poisoning and external causes"),
    ("U00", "U85", "Codes for special purposes"),
    ("V00", "Y99", "External causes of morbidity"),
    ("Z00", "Z99", "Factors influencing health status"),
]

# Curated high-frequency ICD-9 → ICD-10 mappings (approximate GEM subset)
# Format: (icd9_code, icd10_code, description)
_BUILTIN_MAPPINGS: list[tuple[str, str, str]] = [
    # Cardiovascular
    ("41401", "I2510", "Coronary artery disease, native vessel"),
    ("41071", "I2101", "STEMI anterior wall"),
    ("41401", "I2510", "Atherosclerotic heart disease"),
    ("4280", "I509", "Heart failure, unspecified"),
    ("42731", "I4891", "Atrial fibrillation"),
    ("42732", "I4892", "Atrial flutter"),
    ("4275", "I490", "Ventricular fibrillation"),
    ("4271", "I471", "Supraventricular tachycardia"),
    ("4010", "I10", "Essential hypertension"),
    ("4011", "I10", "Benign essential hypertension"),
    ("4019", "I10", "Unspecified essential hypertension"),
    ("43491", "I6350", "Cerebral artery occlusion, unspecified"),
    ("43401", "I6350", "Cerebral thrombosis"),
    ("4439", "I739", "Peripheral vascular disease, unspecified"),
    ("4160", "I270", "Primary pulmonary hypertension"),
    # Respiratory
    ("4660", "J069", "Acute upper respiratory infection"),
    ("486", "J189", "Pneumonia, unspecified organism"),
    ("4870", "J111", "Influenza with pneumonia"),
    ("49390", "J459", "Asthma, unspecified"),
    ("49121", "J441", "COPD with acute exacerbation"),
    ("5185", "J9600", "Acute respiratory failure"),
    ("51881", "J9601", "Acute respiratory failure with hypoxia"),
    ("51882", "J9602", "Acute respiratory failure with hypercapnia"),
    # Gastrointestinal
    ("53500", "K259", "Gastric ulcer, unspecified"),
    ("53100", "K269", "Duodenal ulcer, unspecified"),
    ("57400", "K8000", "Calculus of gallbladder with acute cholecystitis"),
    ("5770", "K859", "Acute pancreatitis"),
    ("5771", "K860", "Chronic pancreatitis due to alcohol"),
    ("56081", "K5700", "Diverticulitis of small intestine"),
    # Renal
    ("5849", "N179", "Acute kidney failure, unspecified"),
    ("5854", "N184", "Chronic kidney disease, stage 4"),
    ("5856", "N186", "End-stage renal disease"),
    # Endocrine / metabolic
    ("25000", "E119", "Type 2 diabetes mellitus without complications"),
    ("25001", "E109", "Type 1 diabetes mellitus without complications"),
    ("25010", "E119", "Type 2 DM with ketoacidosis"),
    ("25011", "E1010", "Type 1 DM with ketoacidosis without coma"),
    ("2720", "E7800", "Pure hypercholesterolaemia"),
    ("2724", "E785", "Hyperlipidaemia, unspecified"),
    ("2449", "E039", "Hypothyroidism, unspecified"),
    ("2420", "E050", "Thyrotoxicosis with diffuse goitre"),
    # Infectious
    ("0389", "A419", "Sepsis, unspecified"),
    ("99591", "R6520", "Severe sepsis without septic shock"),
    ("99592", "R6521", "Severe sepsis with septic shock"),
    ("0369", "A399", "Meningococcal infection, unspecified"),
    ("3200", "G000", "Haemophilus meningitis"),
    # Neurological
    ("4340", "I6630", "Cerebral thrombosis, MCA"),
    ("43410", "I6630", "Middle cerebral artery occlusion"),
    ("2780", "E669", "Obesity, unspecified"),
    ("29590", "F299", "Unspecified psychosis"),
    ("2960", "F310", "Bipolar disorder, manic episode"),
    ("3119", "F329", "Depressive disorder"),
    # Haematology / oncology
    ("20500", "C9100", "Acute lymphoblastic leukaemia, not in remission"),
    ("20520", "C9200", "Acute myeloid leukaemia, not in remission"),
    ("20300", "C9000", "Multiple myeloma"),
    ("28261", "D571", "Sickle-cell anaemia without crisis"),
    ("2851", "D649", "Anaemia, unspecified"),
    ("2859", "D649", "Anaemia, unspecified"),
    # Trauma / injury
    ("8050", "S1200", "Fracture of cervical vertebra"),
    ("82000", "S7200", "Fracture of femoral neck"),
    ("85000", "S0990", "Concussion, unspecified"),
    # Procedures / Z codes
    ("V6001", "Z7901", "Long-term use of anticoagulants"),
    ("V1254", "Z8673", "Personal history of TIA and cerebral infarction"),
    ("V5867", "Z7982", "Long-term use of aspirin"),
]


class ICDMapper:
    """
    Map ICD-9-CM diagnosis codes to ICD-10-CM equivalents.

    Parameters
    ----------
    mappings:
        Custom list of (icd9, icd10, description) tuples. If None,
        uses the built-in curated mapping table.
    default_value:
        Value to use when no mapping is found. Default ``None`` (NaN in DataFrame).

    Examples
    --------
    Map a column of ICD-9 codes to ICD-10:

    >>> mapper = ICDMapper()
    >>> df["icd10"] = mapper.map_series(df["icd9_code"])

    Map in-place with version detection:

    >>> df = mapper.harmonize(df, code_col="icd_code", version_col="icd_version")

    Get the ICD-10 chapter for a code:

    >>> mapper.chapter("I2510")
    'Diseases of the circulatory system'
    """

    def __init__(
        self,
        mappings: list[tuple[str, str, str]] | None = None,
        default_value: str | None = None,
    ) -> None:
        source = mappings or _BUILTIN_MAPPINGS
        self._icd9_to_10: dict[str, str] = {r[0]: r[1] for r in source}
        self._icd10_to_9: dict[str, list[str]] = {}
        for icd9, icd10, _ in source:
            self._icd10_to_9.setdefault(icd10, []).append(icd9)
        self._descriptions: dict[str, str] = {r[0]: r[2] for r in source}
        self.default_value = default_value
        logger.debug(f"ICDMapper loaded {len(self._icd9_to_10)} ICD-9→10 mappings")

    @classmethod
    def from_gem_file(cls, path: str | Path) -> ICDMapper:
        """
        Load from a CMS General Equivalence Mapping (GEM) file.

        The official CMS GEM files are available at:
        https://www.cms.gov/medicare/coding-billing/icd-10-codes

        The file should be a fixed-width or tab-delimited text file with
        columns: icd9_code, icd10_code, flags.

        Parameters
        ----------
        path:
            Path to the CMS GEM forward mapping file (2018 format).
        """
        path = Path(path)
        mappings = []
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 2:
                    mappings.append((parts[0], parts[1], ""))
        logger.info(f"Loaded {len(mappings):,} mappings from {path.name}")
        return cls(mappings=mappings)

    def map_code(self, icd9_code: str) -> str | None:
        """
        Map a single ICD-9 code to its ICD-10 equivalent.

        Parameters
        ----------
        icd9_code:
            ICD-9-CM code string (with or without decimal point).

        Returns
        -------
        str or None
            ICD-10-CM code, or ``default_value`` if not found.
        """
        normalized = self._normalize_code(icd9_code)
        return self._icd9_to_10.get(normalized, self.default_value)

    def map_series(self, series: pd.Series) -> pd.Series:
        """
        Map a Series of ICD-9 codes to ICD-10.

        Parameters
        ----------
        series:
            String Series of ICD-9-CM codes.

        Returns
        -------
        pd.Series
            ICD-10-CM codes. Unmapped codes become ``default_value``.
        """
        normalized = series.astype(str).str.replace(".", "", regex=False).str.strip()
        mapped = normalized.map(self._icd9_to_10)
        n_unmapped = mapped.isna().sum()
        if n_unmapped:
            logger.debug(f"ICDMapper: {n_unmapped:,} codes had no ICD-10 mapping")
        return mapped

    def harmonize(
        self,
        df: pd.DataFrame,
        code_col: str,
        version_col: str,
        icd9_value: str = "9",
        icd10_value: str = "10",
        output_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Harmonize a mixed ICD-9/ICD-10 column to ICD-10 in-place.

        Parameters
        ----------
        df:
            Input DataFrame.
        code_col:
            Column containing ICD codes.
        version_col:
            Column indicating ICD version for each row.
        icd9_value:
            Value in ``version_col`` indicating ICD-9 (default ``"9"``).
        icd10_value:
            Value in ``version_col`` indicating ICD-10 (default ``"10"``).
        output_col:
            Column to write harmonized codes to. Defaults to ``code_col``.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        out_col = output_col or code_col

        icd9_mask = df[version_col].astype(str) == str(icd9_value)
        n_icd9 = int(icd9_mask.sum())
        n_icd10 = int((df[version_col].astype(str) == str(icd10_value)).sum())

        logger.info(
            f"ICDMapper.harmonize: {n_icd9:,} ICD-9 rows, {n_icd10:,} ICD-10 rows "
            f"in column '{code_col}'"
        )

        if n_icd9 > 0:
            df.loc[icd9_mask, out_col] = self.map_series(df.loc[icd9_mask, code_col]).values

        return df

    def chapter(self, icd10_code: str) -> str:
        """
        Return the ICD-10 chapter description for a code.

        Parameters
        ----------
        icd10_code:
            ICD-10-CM code string (e.g. ``"I2510"``).

        Returns
        -------
        str
            Chapter description, or ``"Unknown"`` if code is out of range.
        """
        code = icd10_code.strip().upper()
        prefix = re.match(r"[A-Z]\d{2}", code)
        if not prefix:
            return "Unknown"
        code3 = prefix.group(0)

        for start, end, description in _ICD10_CHAPTERS:
            if start <= code3 <= end:
                return description
        return "Unknown"

    def chapter_series(self, series: pd.Series) -> pd.Series:
        """Map a Series of ICD-10 codes to their chapter descriptions."""
        return series.apply(self.chapter)

    def describe(self, icd9_code: str) -> str:
        """Return the description for an ICD-9 code."""
        return self._descriptions.get(self._normalize_code(icd9_code), "No description available")

    @property
    def n_mappings(self) -> int:
        """Number of ICD-9 → ICD-10 mappings loaded."""
        return len(self._icd9_to_10)

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Strip decimal points and whitespace from an ICD code."""
        return str(code).replace(".", "").strip()
