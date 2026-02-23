"""
Clinical unit normalization.

Labs and vitals are frequently reported in different units across
sites, instruments, and time periods. A glucose of 5.5 mmol/L and
126 mg/dL represent the same value — but most ML pipelines silently
treat them as different features, introducing large systematic errors
in multi-site studies.

This module provides:
- A registry of common clinical unit conversion factors
- A UnitNormalizer class that detects and converts non-standard units
  to a canonical form
- Convenience functions for the most common conversions

Supported conversion families
------------------------------
- Glucose: mg/dL ↔ mmol/L
- Creatinine: mg/dL ↔ μmol/L
- Urea/BUN: mg/dL ↔ mmol/L
- Bilirubin: mg/dL ↔ μmol/L
- Haemoglobin: g/dL ↔ mmol/L
- Temperature: °C ↔ °F ↔ K
- Weight: kg ↔ lb ↔ g
- Height: cm ↔ inches ↔ m
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass
class ConversionSpec:
    """Specification for a single unit conversion."""

    from_unit: str
    to_unit: str
    factor: float | None  # result = value * factor  (None if fn is used)
    fn: Callable[[pd.Series], pd.Series] | None = None  # for non-linear conversions

    def convert(self, series: pd.Series) -> pd.Series:
        if self.fn is not None:
            return self.fn(series)
        if self.factor is not None:
            return series * self.factor
        raise ValueError("ConversionSpec must have either factor or fn")


# -----------------------------------------------------------------------
# Conversion registry
# Key format: "{lab_or_vital}__{from_unit}__{to_unit}"
# -----------------------------------------------------------------------

UNIT_CONVERSIONS: dict[str, ConversionSpec] = {
    # Glucose
    "glucose__mg_dl__mmol_l": ConversionSpec("mg/dL", "mmol/L", 1 / 18.018),
    "glucose__mmol_l__mg_dl": ConversionSpec("mmol/L", "mg/dL", 18.018),
    # Creatinine
    "creatinine__mg_dl__umol_l": ConversionSpec("mg/dL", "μmol/L", 88.42),
    "creatinine__umol_l__mg_dl": ConversionSpec("μmol/L", "mg/dL", 1 / 88.42),
    # BUN (blood urea nitrogen) — reported as urea in mmol/L in Europe
    "bun__mg_dl__mmol_l": ConversionSpec("mg/dL", "mmol/L", 1 / 2.801),
    "bun__mmol_l__mg_dl": ConversionSpec("mmol/L", "mg/dL", 2.801),
    # Bilirubin
    "bilirubin__mg_dl__umol_l": ConversionSpec("mg/dL", "μmol/L", 17.104),
    "bilirubin__umol_l__mg_dl": ConversionSpec("μmol/L", "mg/dL", 1 / 17.104),
    # Haemoglobin
    "hgb__g_dl__mmol_l": ConversionSpec("g/dL", "mmol/L", 0.6206),
    "hgb__mmol_l__g_dl": ConversionSpec("mmol/L", "g/dL", 1 / 0.6206),
    # Calcium (total)
    "calcium__mg_dl__mmol_l": ConversionSpec("mg/dL", "mmol/L", 0.2495),
    "calcium__mmol_l__mg_dl": ConversionSpec("mmol/L", "mg/dL", 4.008),
    # Phosphate
    "phosphate__mg_dl__mmol_l": ConversionSpec("mg/dL", "mmol/L", 0.3229),
    "phosphate__mmol_l__mg_dl": ConversionSpec("mmol/L", "mg/dL", 3.097),
    # Temperature (non-linear)
    "temperature__f__c": ConversionSpec("°F", "°C", None, fn=lambda s: (s - 32) * 5 / 9),
    "temperature__c__f": ConversionSpec("°C", "°F", None, fn=lambda s: s * 9 / 5 + 32),
    "temperature__k__c": ConversionSpec("K", "°C", None, fn=lambda s: s - 273.15),
    "temperature__c__k": ConversionSpec("°C", "K", None, fn=lambda s: s + 273.15),
    # Weight
    "weight__lb__kg": ConversionSpec("lb", "kg", 0.453592),
    "weight__kg__lb": ConversionSpec("kg", "lb", 2.20462),
    "weight__g__kg": ConversionSpec("g", "kg", 0.001),
    "weight__kg__g": ConversionSpec("kg", "g", 1000.0),
    # Height
    "height__in__cm": ConversionSpec("in", "cm", 2.54),
    "height__cm__in": ConversionSpec("cm", "in", 1 / 2.54),
    "height__m__cm": ConversionSpec("m", "cm", 100.0),
    "height__cm__m": ConversionSpec("cm", "m", 0.01),
}

# Canonical target units for common variables
_CANONICAL_UNITS: dict[str, str] = {
    "glucose": "mg/dL",
    "creatinine": "mg/dL",
    "bun": "mg/dL",
    "bilirubin": "mg/dL",
    "hgb": "g/dL",
    "calcium": "mg/dL",
    "phosphate": "mg/dL",
    "temperature": "°C",
    "weight": "kg",
    "height": "cm",
}


class UnitNormalizer:
    """
    Normalize clinical measurements to canonical units.

    Detects non-standard units via a companion unit column or explicit
    mapping and converts values in-place.

    Parameters
    ----------
    column_unit_map:
        Dict mapping value column name → unit column name.
        e.g. ``{"glucose": "glucose_unit"}`` tells the normalizer to
        read units from the ``glucose_unit`` column.
    explicit_conversions:
        Dict mapping value column name → ConversionSpec to apply
        unconditionally (ignores unit columns).
        e.g. ``{"temperature": UNIT_CONVERSIONS["temperature__f__c"]}``
    target_units:
        Dict mapping column name → target unit string. Defaults to
        ``_CANONICAL_UNITS`` for known columns.

    Examples
    --------
    Normalize a glucose column that is mixed mg/dL and mmol/L:

    >>> normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
    >>> df = normalizer.transform(df)

    Convert all temperatures from °F to °C unconditionally:

    >>> normalizer = UnitNormalizer(
    ...     explicit_conversions={"temperature": UNIT_CONVERSIONS["temperature__f__c"]}
    ... )
    >>> df = normalizer.transform(df)
    """

    def __init__(
        self,
        column_unit_map: dict[str, str] | None = None,
        explicit_conversions: dict[str, ConversionSpec] | None = None,
        target_units: dict[str, str] | None = None,
    ) -> None:
        self._column_unit_map = column_unit_map or {}
        self._explicit = explicit_conversions or {}
        self._target_units = {**_CANONICAL_UNITS, **(target_units or {})}
        self._converted: list[dict] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply unit normalization to df.

        Parameters
        ----------
        df:
            Input DataFrame. Modified copy is returned.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        self._converted = []

        # Explicit unconditional conversions
        for col, spec in self._explicit.items():
            if col not in df.columns:
                logger.warning(f"UnitNormalizer: column '{col}' not found — skipping")
                continue
            n_non_null = df[col].notna().sum()
            df[col] = spec.convert(df[col])
            self._converted.append(
                {
                    "column": col,
                    "from_unit": spec.from_unit,
                    "to_unit": spec.to_unit,
                    "n_converted": int(n_non_null),
                    "method": "explicit",
                }
            )
            logger.info(f"UnitNormalizer: converted {col} from {spec.from_unit} → {spec.to_unit}")

        # Unit-column-aware conversions
        for value_col, unit_col in self._column_unit_map.items():
            if value_col not in df.columns:
                logger.warning(f"UnitNormalizer: value column '{value_col}' not found — skipping")
                continue
            if unit_col not in df.columns:
                logger.warning(f"UnitNormalizer: unit column '{unit_col}' not found — skipping")
                continue

            target_unit = self._target_units.get(value_col)
            if target_unit is None:
                logger.warning(
                    f"UnitNormalizer: no target unit configured for '{value_col}' — skipping"
                )
                continue

            unique_units = df[unit_col].dropna().unique()
            for from_unit in unique_units:
                if from_unit == target_unit:
                    continue

                key = self._make_key(value_col, from_unit, target_unit)
                if key not in UNIT_CONVERSIONS:
                    logger.warning(
                        f"UnitNormalizer: no conversion registered for "
                        f"'{value_col}' {from_unit} → {target_unit} (key={key!r})"
                    )
                    continue

                spec = UNIT_CONVERSIONS[key]
                mask = df[unit_col] == from_unit
                n = int(mask.sum())
                df.loc[mask, value_col] = spec.convert(df.loc[mask, value_col])
                df.loc[mask, unit_col] = target_unit

                self._converted.append(
                    {
                        "column": value_col,
                        "from_unit": from_unit,
                        "to_unit": target_unit,
                        "n_converted": n,
                        "method": "unit_column",
                    }
                )
                logger.info(
                    f"UnitNormalizer: converted {n:,} rows of '{value_col}' "
                    f"from {from_unit} → {target_unit}"
                )

        return df

    def report(self) -> pd.DataFrame:
        """Return a summary of all conversions applied."""
        if not self._converted:
            return pd.DataFrame(columns=["column", "from_unit", "to_unit", "n_converted", "method"])
        return pd.DataFrame(self._converted)

    @staticmethod
    def _make_key(variable: str, from_unit: str, to_unit: str) -> str:
        """Normalise a unit string to a registry key segment."""

        def _norm(u: str) -> str:
            return (
                u.lower()
                .replace("/", "_")
                .replace("μ", "u")
                .replace("°", "")
                .replace(" ", "_")
                .strip("_")
            )

        return f"{variable}__{_norm(from_unit)}__{_norm(to_unit)}"

    @staticmethod
    def available_conversions() -> list[str]:
        """Return all registered conversion keys."""
        return sorted(UNIT_CONVERSIONS.keys())


# -----------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------


def celsius_to_fahrenheit(series: pd.Series) -> pd.Series:
    """Convert temperature from °C to °F."""
    return series * 9 / 5 + 32


def fahrenheit_to_celsius(series: pd.Series) -> pd.Series:
    """Convert temperature from °F to °C."""
    return (series - 32) * 5 / 9


def glucose_mgdl_to_mmol(series: pd.Series) -> pd.Series:
    """Convert glucose from mg/dL to mmol/L."""
    return series / 18.018


def glucose_mmol_to_mgdl(series: pd.Series) -> pd.Series:
    """Convert glucose from mmol/L to mg/dL."""
    return series * 18.018


def creatinine_mgdl_to_umol(series: pd.Series) -> pd.Series:
    """Convert creatinine from mg/dL to μmol/L."""
    return series * 88.42


def creatinine_umol_to_mgdl(series: pd.Series) -> pd.Series:
    """Convert creatinine from μmol/L to mg/dL."""
    return series / 88.42
