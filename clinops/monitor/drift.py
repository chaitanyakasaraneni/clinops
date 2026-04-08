"""
Distribution drift detection for clinical ML pipelines.

Clinical datasets shift in ways that general-purpose drift detectors miss.
A new patient cohort may have different comorbidity distributions; a lab
analyser replacement shifts creatinine values by a fixed offset; seasonal
admission patterns change the mix of diagnoses. This module provides two
complementary metrics:

Population Stability Index (PSI)
    Widely used in healthcare model validation. Measures how much a
    variable's distribution has shifted relative to a reference.
    Interpretable thresholds: PSI < 0.1 is stable, 0.1–0.2 warrants
    review, > 0.2 indicates significant drift.

Kolmogorov–Smirnov test
    A non-parametric two-sample test that detects any distributional
    difference. Complements PSI by providing a p-value and works well
    for small samples where PSI binning is unreliable.

References
----------
Yurdakul (2018). Statistical properties of population stability index.
Working paper, Western Michigan University.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class DriftSeverity(StrEnum):
    """
    Severity levels for distribution drift.

    Based on standard PSI thresholds used in healthcare model validation:

    LOW
        PSI < 0.1 — distribution is stable; no action required.
    MEDIUM
        0.1 <= PSI < 0.2 — moderate shift; review the column and
        investigate whether the change is clinically meaningful.
    HIGH
        PSI >= 0.2 — significant drift; model retraining or pipeline
        investigation is strongly recommended.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ColumnDriftResult:
    """
    Drift metrics for a single column.

    Attributes
    ----------
    column:
        Column name.
    psi:
        Population Stability Index (lower is more stable).
    ks_statistic:
        KS two-sample test statistic, or None if not computed.
    ks_pvalue:
        KS test p-value, or None if not computed. Values below 0.05
        indicate a statistically significant distributional difference.
    severity:
        Drift severity based on PSI thresholds.
    reference_mean:
        Mean of the column in the reference (training) dataset.
    current_mean:
        Mean of the column in the current (production) dataset.
    reference_std:
        Standard deviation in the reference dataset.
    current_std:
        Standard deviation in the current dataset.
    n_reference:
        Number of non-null observations in the reference dataset.
    n_current:
        Number of non-null observations in the current dataset.
    """

    column: str
    psi: float
    ks_statistic: float | None
    ks_pvalue: float | None
    severity: DriftSeverity
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    n_reference: int
    n_current: int

    @property
    def mean_shift(self) -> float:
        """Absolute shift in the column mean."""
        return self.current_mean - self.reference_mean

    @property
    def mean_shift_pct(self) -> float:
        """Mean shift as a percentage of the reference mean (0 if reference mean is 0)."""
        if self.reference_mean == 0:
            return 0.0
        return 100.0 * self.mean_shift / abs(self.reference_mean)


@dataclass
class DriftReport:
    """
    Structured result from :class:`DistributionDriftDetector`.

    Attributes
    ----------
    results:
        Per-column drift metrics.
    n_columns_checked:
        Total number of numeric columns evaluated.
    """

    results: list[ColumnDriftResult]
    n_columns_checked: int

    @property
    def n_low(self) -> int:
        """Number of columns with LOW severity drift."""
        return sum(1 for r in self.results if r.severity == DriftSeverity.LOW)

    @property
    def n_medium(self) -> int:
        """Number of columns with MEDIUM severity drift."""
        return sum(1 for r in self.results if r.severity == DriftSeverity.MEDIUM)

    @property
    def n_high(self) -> int:
        """Number of columns with HIGH severity drift."""
        return sum(1 for r in self.results if r.severity == DriftSeverity.HIGH)

    def drifted_columns(self, min_severity: DriftSeverity = DriftSeverity.MEDIUM) -> list[str]:
        """
        Return column names with drift at or above ``min_severity``.

        Parameters
        ----------
        min_severity:
            Minimum severity level to include. Default: MEDIUM.

        Returns
        -------
        list[str]
        """
        order = {DriftSeverity.LOW: 0, DriftSeverity.MEDIUM: 1, DriftSeverity.HIGH: 2}
        threshold = order[min_severity]
        return [r.column for r in self.results if order[r.severity] >= threshold]

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-column results as a DataFrame sorted by PSI descending."""
        rows: list[dict[str, Any]] = [
            {
                "column": r.column,
                "psi": round(r.psi, 4),
                "severity": r.severity.value,
                "ks_statistic": round(r.ks_statistic, 4) if r.ks_statistic is not None else None,
                "ks_pvalue": round(r.ks_pvalue, 4) if r.ks_pvalue is not None else None,
                "reference_mean": round(r.reference_mean, 4),
                "current_mean": round(r.current_mean, 4),
                "mean_shift_pct": round(r.mean_shift_pct, 2),
                "n_reference": r.n_reference,
                "n_current": r.n_current,
            }
            for r in self.results
        ]
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    def summary(self) -> str:
        """Human-readable drift summary."""
        lines = [
            f"Columns checked : {self.n_columns_checked}",
            f"HIGH drift      : {self.n_high}",
            f"MEDIUM drift    : {self.n_medium}",
            f"LOW drift       : {self.n_low}",
        ]
        high_cols = self.drifted_columns(DriftSeverity.HIGH)
        if high_cols:
            lines.append(f"HIGH columns    : {', '.join(high_cols)}")
        med_cols = [c for c in self.drifted_columns(DriftSeverity.MEDIUM) if c not in high_cols]
        if med_cols:
            lines.append(f"MEDIUM columns  : {', '.join(med_cols)}")
        return "\n".join(lines)


class DistributionDriftDetector:
    """
    Detect distribution drift between a reference dataset and a current batch.

    Fit on a reference dataset (typically the training set), then call
    ``detect()`` on each new batch to get per-column PSI and KS statistics.

    Parameters
    ----------
    n_bins:
        Number of equal-frequency bins used to compute PSI. Default 10.
        Use fewer bins for small datasets (< 500 rows).
    psi_threshold_medium:
        PSI threshold for MEDIUM severity. Default 0.1.
    psi_threshold_high:
        PSI threshold for HIGH severity. Default 0.2.
    run_ks_test:
        If True, run a KS two-sample test in addition to PSI. Default True.
    columns:
        Explicit list of columns to monitor. If None, all numeric columns
        in the reference DataFrame are monitored.

    Examples
    --------
    >>> detector = DistributionDriftDetector()
    >>> detector.fit(train_df)
    >>> report = detector.detect(production_batch_df)
    >>> print(report.summary())
    >>> print(report.to_dataframe())

    >>> # Only alert on high-severity drift
    >>> drifted = report.drifted_columns(DriftSeverity.HIGH)
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold_medium: float = 0.1,
        psi_threshold_high: float = 0.2,
        run_ks_test: bool = True,
        columns: list[str] | None = None,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins = n_bins
        self.psi_threshold_medium = psi_threshold_medium
        self.psi_threshold_high = psi_threshold_high
        self.run_ks_test = run_ks_test
        self.columns = columns
        self._reference_data: dict[str, np.ndarray] = {}
        self._bin_edges: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> DistributionDriftDetector:
        """
        Compute reference statistics from a training/baseline DataFrame.

        Parameters
        ----------
        df:
            Reference DataFrame (typically the training set).

        Returns
        -------
        DistributionDriftDetector
            Self, for method chaining.
        """
        cols = self.columns or list(df.select_dtypes(include=[np.number]).columns)
        self._reference_data = {}
        self._bin_edges = {}

        for col in cols:
            if col not in df.columns:
                logger.warning(f"DriftDetector.fit: column '{col}' not in DataFrame — skipping")
                continue
            values = df[col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                logger.warning(
                    f"DriftDetector.fit: column '{col}' has no non-null values — skipping"
                )
                continue
            self._reference_data[col] = values
            # Build equal-frequency bin edges from the reference distribution
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.unique(np.percentile(values, quantiles))
            # Ensure open-ended outer bins to cover any current values outside reference range
            edges[0] = -np.inf
            edges[-1] = np.inf
            self._bin_edges[col] = edges

        logger.info(f"DistributionDriftDetector fitted on {len(self._reference_data)} columns")
        return self

    def detect(self, df: pd.DataFrame) -> DriftReport:
        """
        Compute drift metrics for each fitted column.

        Parameters
        ----------
        df:
            Current DataFrame to compare against the reference.

        Returns
        -------
        DriftReport

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if not self._reference_data:
            raise RuntimeError("Call fit() before detect()")

        results: list[ColumnDriftResult] = []

        for col, ref_values in self._reference_data.items():
            if col not in df.columns:
                logger.warning(f"DriftDetector.detect: column '{col}' missing from current batch")
                continue

            cur_values = df[col].dropna().to_numpy(dtype=float)
            if len(cur_values) == 0:
                logger.warning(
                    f"DriftDetector.detect: column '{col}' has no non-null values in current batch"
                )
                continue

            psi = self._compute_psi(ref_values, cur_values, self._bin_edges[col])
            severity = self._severity(psi)

            ks_stat: float | None = None
            ks_pval: float | None = None
            if self.run_ks_test:
                ks_result = stats.ks_2samp(ref_values, cur_values)
                ks_stat = float(ks_result.statistic)
                ks_pval = float(ks_result.pvalue)

            results.append(
                ColumnDriftResult(
                    column=col,
                    psi=psi,
                    ks_statistic=ks_stat,
                    ks_pvalue=ks_pval,
                    severity=severity,
                    reference_mean=float(np.mean(ref_values)),
                    current_mean=float(np.mean(cur_values)),
                    reference_std=float(np.std(ref_values)),
                    current_std=float(np.std(cur_values)),
                    n_reference=len(ref_values),
                    n_current=len(cur_values),
                )
            )

        report = DriftReport(results=results, n_columns_checked=len(results))

        logger.info(
            f"DriftDetector: {report.n_high} HIGH, {report.n_medium} MEDIUM, "
            f"{report.n_low} LOW across {report.n_columns_checked} columns"
        )
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_psi(
        self, reference: np.ndarray, current: np.ndarray, bin_edges: np.ndarray
    ) -> float:
        """Compute PSI using pre-computed equal-frequency bin edges."""
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions; add small epsilon to avoid log(0)
        eps = 1e-4
        ref_pct = ref_counts / len(reference) + eps
        cur_pct = cur_counts / len(current) + eps

        # Normalise so proportions sum to 1 (epsilon shifts them slightly off)
        ref_pct /= ref_pct.sum()
        cur_pct /= cur_pct.sum()

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(0.0, psi)

    def _severity(self, psi: float) -> DriftSeverity:
        if psi >= self.psi_threshold_high:
            return DriftSeverity.HIGH
        if psi >= self.psi_threshold_medium:
            return DriftSeverity.MEDIUM
        return DriftSeverity.LOW
