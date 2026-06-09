"""
eICU-CRD lab-name → canonical feature mapping.

eICU stores laboratory results in long format with a free-text ``labname``
column (there is no LOINC/itemid coding as in MIMIC-IV). To assemble the
feature matrix used by the multi-organ deterioration model, those free-text
names must be harmonised to a small set of canonical feature keys.

The keys below match the lab features in the ICHI 2026 model. ``gcs_total`` is
deliberately *not* here — in eICU the Glasgow Coma Score lives in
``nurseCharting`` (category ``Scores``), not in ``lab``; it is handled
separately by :class:`~clinops.ingest.eicu.EicuTableLoader`.

The right-hand-side strings were taken from an actual eICU-CRD v2.0 download
(value-count scan of ``lab.csv``), not from documentation — e.g. eICU spells
white-cell count ``"WBC x 1000"`` and platelets ``"platelets x 1000"``.

Matching is **case-insensitive** and whitespace-insensitive; see
:func:`map_labname`. Unmapped names are returned as ``None`` so the caller can
log a warning rather than silently dropping data.
"""

from __future__ import annotations

#: Canonical feature key -> list of known eICU ``labname`` spellings (lowercased).
#: Multiple raw spellings collapse onto one canonical feature.
_CANONICAL_TO_RAW: dict[str, list[str]] = {
    "creatinine": [
        "creatinine",
        "creatinine (point-of-care)",
    ],
    "bilirubin": [
        "total bilirubin",
        "bilirubin total",
        "bilirubin, total",
        "tbil",
    ],
    "platelets": [
        "platelets x 1000",
        "platelet count",
        "platelets",
    ],
    "wbc": [
        "wbc x 1000",
        "white blood cell count",
        "wbc",
        "wbc's in body fluid",
    ],
    "lactate": [
        "lactate",
        "lactic acid",
        "lactate (mmol/l)",
    ],
    "ph": [
        "ph",
        "arterial ph",
        "venous ph",
    ],
    "pao2": [
        "pao2",
        "po2 (arterial)",
        "po2",
    ],
    "paco2": [
        "paco2",
        "pco2 (arterial)",
        "pco2",
    ],
    # FiO2 is a derived/label-supporting variable (used for the PaO2/FiO2 ratio).
    # It is mapped here so the loader can read it from the lab table, but the
    # leakage guard in EicuTableLoader keeps it out of the final feature matrix.
    "fio2": [
        "fio2",
    ],
}

#: Flattened reverse lookup: raw lowercased ``labname`` -> canonical feature key.
EICU_LAB_MAP: dict[str, str] = {
    raw: canonical for canonical, raws in _CANONICAL_TO_RAW.items() for raw in raws
}


def map_labname(raw: str | None) -> str | None:
    """
    Map a raw eICU ``labname`` to its canonical feature key.

    Parameters
    ----------
    raw:
        Free-text lab name as it appears in ``lab.labname``. May be ``None``.

    Returns
    -------
    str or None
        The canonical feature key (e.g. ``"creatinine"``) or ``None`` if the
        name is not recognised. Matching is case- and whitespace-insensitive.

    Examples
    --------
    >>> map_labname("WBC x 1000")
    'wbc'
    >>> map_labname("  Total Bilirubin ")
    'bilirubin'
    >>> map_labname("troponin - I") is None
    True
    """
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return EICU_LAB_MAP.get(key)


def canonical_lab_features() -> list[str]:
    """
    Return the canonical lab feature keys in a stable, deterministic order.

    Useful for building feature-matrix columns with consistent ordering across
    runs (reproducibility).
    """
    return list(_CANONICAL_TO_RAW.keys())
