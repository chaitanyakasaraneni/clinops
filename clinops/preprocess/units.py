"""
Clinical unit normalization.

Labs and vitals are frequently reported in different units across
sites, instruments, and time periods. A glucose of 5.5 mmol/L and
126 mg/dL represent the same value — but most ML pipelines silently
treat them as different features, introducing large systematic errors
in multi-site studies.

This module provides:

- ``UNIT_CONVERSIONS``, a registry of conversion factors. Each linear
  conversion is defined once from its physical constant (molar mass, or the
  exact definition of the pound and inch), and the reverse direction is its
  exact reciprocal, so converting there and back returns the original value.
- ``UnitNormalizer``, for DataFrames whose rows mix units: it reads a
  companion unit column per row and converts every value to a canonical unit.
- Convenience functions (``glucose_mgdl_to_mmol`` etc.) for a single Series
  whose unit is known for the whole column. They use the same registry.

Supported conversion families
------------------------------
- Glucose: mg/dL ↔ mmol/L
- Creatinine: mg/dL ↔ μmol/L
- Urea/BUN: mg/dL ↔ mmol/L
- Bilirubin: mg/dL ↔ μmol/L
- Haemoglobin: g/dL ↔ mmol/L
- Calcium: mg/dL ↔ mmol/L
- Phosphate: mg/dL ↔ mmol/L
- Temperature: °C ↔ °F ↔ K
- Weight: kg ↔ lb ↔ g
- Height: cm ↔ inches ↔ m
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


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

#: Linear conversions, each defined once, in the direction its constant is
#: usually quoted: (variable, unit A key, unit B key, A label, B label, factor)
#: with ``B = A * factor``. The reverse direction is derived as ``1 / factor``,
#: so every registered pair is an exact inverse.
_LINEAR_CONVERSIONS: list[tuple[str, str, str, str, str, float]] = [
    # Molar conversions: factor = molar mass (g/mol) / 10 for mg/dL per mmol/L.
    # Glucose: the conventional 18.018 (its molar mass, 180.16 g/mol, gives 18.016).
    ("glucose", "mmol_l", "mg_dl", "mmol/L", "mg/dL", 18.018),
    ("creatinine", "mg_dl", "umol_l", "mg/dL", "μmol/L", 88.42),  # 113.12 g/mol
    ("bun", "mmol_l", "mg_dl", "mmol/L", "mg/dL", 2.801),  # urea nitrogen (2 N) 28.01 g/mol
    ("bilirubin", "mg_dl", "umol_l", "mg/dL", "μmol/L", 17.104),  # 584.66 g/mol
    ("hgb", "g_dl", "mmol_l", "g/dL", "mmol/L", 0.6206),  # per haem monomer, 16,114 g/mol
    ("calcium", "mmol_l", "mg_dl", "mmol/L", "mg/dL", 4.008),  # 40.08 g/mol
    ("phosphate", "mmol_l", "mg_dl", "mmol/L", "mg/dL", 3.097),  # as phosphorus, 30.97 g/mol
    # Exact unit definitions.
    ("weight", "lb", "kg", "lb", "kg", 0.45359237),
    ("weight", "kg", "g", "kg", "g", 1000.0),
    ("height", "in", "cm", "in", "cm", 2.54),
    ("height", "m", "cm", "m", "cm", 100.0),
]


def _celsius_to_fahrenheit(s: pd.Series) -> pd.Series:
    return s * 9 / 5 + 32


def _fahrenheit_to_celsius(s: pd.Series) -> pd.Series:
    return (s - 32) * 5 / 9


def _kelvin_to_celsius(s: pd.Series) -> pd.Series:
    return s - 273.15


def _celsius_to_kelvin(s: pd.Series) -> pd.Series:
    return s + 273.15


def _build_registry() -> dict[str, ConversionSpec]:
    registry: dict[str, ConversionSpec] = {}
    for var, a, b, a_label, b_label, factor in _LINEAR_CONVERSIONS:
        registry[f"{var}__{a}__{b}"] = ConversionSpec(a_label, b_label, factor)
        registry[f"{var}__{b}__{a}"] = ConversionSpec(b_label, a_label, 1 / factor)
    # Temperature (affine, so a function rather than a factor).
    registry["temperature__f__c"] = ConversionSpec("°F", "°C", None, fn=_fahrenheit_to_celsius)
    registry["temperature__c__f"] = ConversionSpec("°C", "°F", None, fn=_celsius_to_fahrenheit)
    registry["temperature__k__c"] = ConversionSpec("K", "°C", None, fn=_kelvin_to_celsius)
    registry["temperature__c__k"] = ConversionSpec("°C", "K", None, fn=_celsius_to_kelvin)
    return registry


#: All registered conversions, keyed ``"{variable}__{from}__{to}"``.
UNIT_CONVERSIONS: dict[str, ConversionSpec] = _build_registry()

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
    Normalize a glucose column that mixes mg/dL and mmol/L:

    >>> import pandas as pd
    >>> from clinops.preprocess import UNIT_CONVERSIONS, UnitNormalizer
    >>> df = pd.DataFrame({"glucose": [126.0, 7.0], "glucose_unit": ["mg/dL", "mmol/L"]})
    >>> normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
    >>> normalizer.transform(df).round(1)
       glucose glucose_unit
    0    126.0        mg/dL
    1    126.1        mg/dL

    Convert all temperatures from °F to °C unconditionally:

    >>> normalizer = UnitNormalizer(
    ...     explicit_conversions={"temperature": UNIT_CONVERSIONS["temperature__f__c"]}
    ... )
    >>> normalizer.transform(pd.DataFrame({"temperature": [98.6, 101.3]})).round(1)
       temperature
    0         37.0
    1         38.5
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
        self._converted: list[dict[str, Any]] = []

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
                converted = spec.convert(df.loc[mask, value_col])
                # Converted values are fractional; pandas 3 refuses to write them into
                # an integer column through .loc (pandas 2 upcast with a FutureWarning).
                if pd.api.types.is_integer_dtype(df[value_col].dtype):
                    df[value_col] = df[value_col].astype(converted.dtype)
                df.loc[mask, value_col] = converted
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
            return pd.DataFrame(
                columns=["column", "from_unit", "to_unit", "n_converted", "method"]
            )
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
# For a single Series whose unit is known for the whole column, e.g. a lab
# exported from one site. They use the same registry as UnitNormalizer, which is
# for DataFrames whose rows mix units and carry a companion unit column.


def _convert(series: pd.Series, key: str) -> pd.Series:
    return UNIT_CONVERSIONS[key].convert(series)


def celsius_to_fahrenheit(series: pd.Series) -> pd.Series:
    """Convert temperature from °C to °F (for a column known to be in °C)."""
    return _convert(series, "temperature__c__f")


def fahrenheit_to_celsius(series: pd.Series) -> pd.Series:
    """Convert temperature from °F to °C (for a column known to be in °F)."""
    return _convert(series, "temperature__f__c")


def glucose_mgdl_to_mmol(series: pd.Series) -> pd.Series:
    """Convert glucose from mg/dL to mmol/L (for a column known to be in mg/dL)."""
    return _convert(series, "glucose__mg_dl__mmol_l")


def glucose_mmol_to_mgdl(series: pd.Series) -> pd.Series:
    """Convert glucose from mmol/L to mg/dL (for a column known to be in mmol/L)."""
    return _convert(series, "glucose__mmol_l__mg_dl")


def creatinine_mgdl_to_umol(series: pd.Series) -> pd.Series:
    """Convert creatinine from mg/dL to μmol/L (for a column known to be in mg/dL)."""
    return _convert(series, "creatinine__mg_dl__umol_l")


def creatinine_umol_to_mgdl(series: pd.Series) -> pd.Series:
    """Convert creatinine from μmol/L to mg/dL (for a column known to be in μmol/L)."""
    return _convert(series, "creatinine__umol_l__mg_dl")
