"""
Outlier detection and clipping using physiologically-grounded bounds.

Clinical data contains a high rate of physiologically impossible values —
transcription errors, unit mix-ups, device artefacts, and data entry
mistakes. Standard statistical outlier methods (z-score, IQR) are
inappropriate here because true extreme values (e.g., a heart rate of
180 in a patient with SVT) are clinically meaningful and should not be
removed.

This module uses published physiological bounds from clinical literature
to clip or flag values that are impossible regardless of patient state,
while preserving extreme-but-plausible values.

References
----------
Johnson et al. (2023). MIMIC-IV, a freely accessible electronic health
record dataset. Scientific Data, 10(1), 1.

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation, 101(23), e215–e220.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BoundSpec:
    """Physiological bounds for a single clinical variable."""

    col: str
    low: float
    high: float
    unit: str = ""
    description: str = ""


# -----------------------------------------------------------------------
# Published physiological bounds
# -----------------------------------------------------------------------

#: Vital sign bounds — values outside these ranges are physiologically
#: impossible and indicate data entry errors or unit mix-ups.
VITAL_BOUNDS: dict[str, BoundSpec] = {
    "heart_rate": BoundSpec("heart_rate", 0, 300, "bpm", "Asystole to extreme tachycardia"),
    "resp_rate": BoundSpec("resp_rate", 0, 80, "breaths/min", "Apnoea to extreme tachypnoea"),
    "spo2": BoundSpec("spo2", 50, 100, "%", "Incompatible with life below 50%"),
    "sbp": BoundSpec("sbp", 0, 300, "mmHg", "Systolic blood pressure"),
    "dbp": BoundSpec("dbp", 0, 200, "mmHg", "Diastolic blood pressure"),
    "map": BoundSpec("map", 0, 250, "mmHg", "Mean arterial pressure"),
    "temperature": BoundSpec("temperature", 25.0, 45.0, "°C", "Severe hypothermia to hyperthermia"),
    "temperature_f": BoundSpec("temperature_f", 77.0, 113.0, "°F", "°F equivalent of 25–45°C"),
    "weight": BoundSpec("weight", 0.5, 500.0, "kg", "Neonate to extreme obesity"),
    "height": BoundSpec("height", 30.0, 250.0, "cm", "Infant to extreme height"),
    "bmi": BoundSpec("bmi", 10.0, 100.0, "kg/m²", "Severe underweight to extreme obesity"),
    "gcs": BoundSpec("gcs", 3, 15, "score", "Glasgow Coma Scale: 3 (worst) to 15 (normal)"),
}

#: Laboratory value bounds — physiologically impossible ranges.
LAB_BOUNDS: dict[str, BoundSpec] = {
    # Metabolic
    "glucose": BoundSpec("glucose", 0, 2000, "mg/dL", "Hypoglycaemia to extreme hyperglycaemia"),
    "glucose_mmol": BoundSpec("glucose_mmol", 0, 111, "mmol/L", "mmol/L equivalent"),
    "sodium": BoundSpec("sodium", 90, 200, "mEq/L", "Compatible with life"),
    "potassium": BoundSpec("potassium", 1.0, 10.0, "mEq/L", "Severe hypo- to hyperkalaemia"),
    "chloride": BoundSpec("chloride", 60, 150, "mEq/L"),
    "bicarbonate": BoundSpec("bicarbonate", 0, 60, "mEq/L"),
    "bun": BoundSpec("bun", 0, 300, "mg/dL", "Blood urea nitrogen"),
    "creatinine": BoundSpec("creatinine", 0, 50, "mg/dL"),
    "calcium": BoundSpec("calcium", 1.0, 20.0, "mg/dL"),
    "magnesium": BoundSpec("magnesium", 0, 10, "mg/dL"),
    "phosphate": BoundSpec("phosphate", 0, 30, "mg/dL"),
    # Haematology
    "wbc": BoundSpec("wbc", 0, 500, "K/uL", "White blood cell count"),
    "hgb": BoundSpec("hgb", 0, 25, "g/dL", "Haemoglobin"),
    "hct": BoundSpec("hct", 0, 100, "%", "Haematocrit"),
    "platelets": BoundSpec("platelets", 0, 3000, "K/uL"),
    # Liver / coagulation
    "alt": BoundSpec("alt", 0, 30000, "U/L", "Alanine aminotransferase"),
    "ast": BoundSpec("ast", 0, 30000, "U/L", "Aspartate aminotransferase"),
    "bilirubin": BoundSpec("bilirubin", 0, 100, "mg/dL"),
    "inr": BoundSpec("inr", 0, 20, "ratio"),
    "pt": BoundSpec("pt", 0, 200, "seconds"),
    "ptt": BoundSpec("ptt", 0, 400, "seconds"),
    # ABG
    "ph": BoundSpec("ph", 6.5, 7.9, "pH", "Arterial pH — incompatible outside this range"),
    "pao2": BoundSpec("pao2", 0, 700, "mmHg", "PaO2 on 100% O2"),
    "paco2": BoundSpec("paco2", 0, 150, "mmHg"),
    "lactate": BoundSpec("lactate", 0, 30, "mmol/L"),
}


class ClinicalOutlierClipper:
    """
    Detect and clip physiologically impossible values in clinical DataFrames.

    Uses published physiological bounds to identify values that are
    impossible regardless of patient state. Values outside bounds are
    either clipped to the boundary (default) or replaced with NaN.

    Parameters
    ----------
    bounds:
        Dict mapping column name to BoundSpec. Defaults to combined
        VITAL_BOUNDS + LAB_BOUNDS. Pass a custom dict to override.
    action:
        What to do with out-of-range values:
        - ``"clip"``  : replace with the boundary value (default)
        - ``"null"``  : replace with NaN
        - ``"flag"``  : add a boolean ``{col}_outlier`` column, do not modify value
    extra_bounds:
        Additional BoundSpec entries to merge with the default bounds.
        Useful for site-specific or assay-specific ranges.
    strict:
        If True, raise ValueError when a column in bounds is not found
        in the DataFrame. If False (default), silently skip missing cols.

    Examples
    --------
    >>> clipper = ClinicalOutlierClipper()
    >>> clean_df = clipper.fit_transform(vitals_df)
    >>> print(clipper.report())
    """

    def __init__(
        self,
        bounds: dict[str, BoundSpec] | None = None,
        action: str = "clip",
        extra_bounds: dict[str, BoundSpec] | None = None,
        strict: bool = False,
    ) -> None:
        if action not in ("clip", "null", "flag"):
            raise ValueError(f"action must be 'clip', 'null', or 'flag' — got {action!r}")

        self._bounds = {**VITAL_BOUNDS, **LAB_BOUNDS}
        if bounds is not None:
            self._bounds = bounds
        if extra_bounds:
            self._bounds.update(extra_bounds)

        self.action = action
        self.strict = strict
        self._report: list[dict] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip or flag outliers in df using the configured bounds.

        Parameters
        ----------
        df:
            Input DataFrame. Only columns present in bounds are processed.

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers handled according to ``action``.
        """
        df = df.copy()
        self._report = []

        for col, spec in self._bounds.items():
            if col not in df.columns:
                if self.strict:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                continue

            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue

            low_mask = series < spec.low
            high_mask = series > spec.high
            n_low = int(low_mask.sum())
            n_high = int(high_mask.sum())

            if n_low == 0 and n_high == 0:
                continue

            self._report.append({
                "column": col,
                "low_outliers": n_low,
                "high_outliers": n_high,
                "total_outliers": n_low + n_high,
                "pct_outliers": round(100 * (n_low + n_high) / len(series), 3),
                "bound_low": spec.low,
                "bound_high": spec.high,
                "unit": spec.unit,
            })

            logger.debug(
                f"{col}: {n_low} below {spec.low}{spec.unit}, "
                f"{n_high} above {spec.high}{spec.unit} → action={self.action}"
            )

            if self.action == "clip":
                df[col] = series.clip(lower=spec.low, upper=spec.high)
            elif self.action == "null":
                df.loc[low_mask | high_mask, col] = np.nan
            elif self.action == "flag":
                df[f"{col}_outlier"] = (low_mask | high_mask).astype(int)

        n_affected = sum(r["total_outliers"] for r in self._report)
        if n_affected:
            logger.info(
                f"ClinicalOutlierClipper: {n_affected:,} outlier values across "
                f"{len(self._report)} columns (action={self.action})"
            )
        else:
            logger.info("ClinicalOutlierClipper: no outliers detected")

        return df

    def report(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of all detected outliers.

        Returns an empty DataFrame if fit_transform has not been called
        or no outliers were detected.
        """
        if not self._report:
            return pd.DataFrame(columns=[
                "column", "low_outliers", "high_outliers",
                "total_outliers", "pct_outliers", "bound_low", "bound_high", "unit",
            ])
        return pd.DataFrame(self._report).sort_values("total_outliers", ascending=False)

    def add_bounds(self, col: str, low: float, high: float, unit: str = "") -> None:
        """Add or replace a bound for a specific column."""
        self._bounds[col] = BoundSpec(col, low, high, unit)
