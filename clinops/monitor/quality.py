"""
Data quality checking for clinical ML pipelines.

Clinical pipelines fail silently in ways that standard data validation
misses: a new hospital site sends creatinine in mmol/L instead of mg/dL,
a required ID column becomes all-null after a join, or a schema change
drops the outcome column entirely. This module provides lightweight,
production-oriented checks that can be run at pipeline ingestion time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass
class QualityIssue:
    """
    A single data quality issue detected by :class:`DataQualityChecker`.

    Attributes
    ----------
    column:
        Affected column name, or ``"__dataframe__"`` for row-level issues.
    issue_type:
        One of ``"high_null_rate"``, ``"all_null"``, ``"column_added"``,
        ``"column_removed"``, ``"dtype_changed"``, ``"row_count_anomaly"``.
    severity:
        ``"error"`` for issues that will break downstream steps;
        ``"warning"`` for issues worth investigating but not fatal.
    detail:
        Human-readable description of the issue.
    """

    column: str
    issue_type: str
    severity: str
    detail: str


@dataclass
class QualityReport:
    """
    Structured result from :class:`DataQualityChecker`.

    Attributes
    ----------
    issues:
        All detected quality issues.
    n_rows:
        Number of rows in the checked DataFrame.
    n_columns:
        Number of columns in the checked DataFrame.
    null_rates:
        Per-column null rate (fraction 0–1).
    """

    issues: list[QualityIssue]
    n_rows: int
    n_columns: int
    null_rates: dict[str, float]

    @property
    def errors(self) -> list[QualityIssue]:
        """Issues with severity ``"error"``."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        """Issues with severity ``"warning"``."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        """True if there are no error-severity issues."""
        return len(self.errors) == 0

    def to_dataframe(self) -> pd.DataFrame:
        """Return issues as a DataFrame."""
        if not self.issues:
            return pd.DataFrame(columns=["column", "issue_type", "severity", "detail"])
        return pd.DataFrame(
            [
                {
                    "column": i.column,
                    "issue_type": i.issue_type,
                    "severity": i.severity,
                    "detail": i.detail,
                }
                for i in self.issues
            ]
        )

    def summary(self) -> str:
        """Human-readable quality summary."""
        lines = [
            f"Rows     : {self.n_rows:,}",
            f"Columns  : {self.n_columns}",
            f"Errors   : {len(self.errors)}",
            f"Warnings : {len(self.warnings)}",
            f"Passed   : {self.passed}",
        ]
        for issue in self.issues:
            prefix = "  [ERROR]  " if issue.severity == "error" else "  [WARN]   "
            lines.append(f"{prefix}{issue.detail}")
        return "\n".join(lines)


class DataQualityChecker:
    """
    Run data quality checks on a clinical DataFrame.

    Can be used standalone (``check(df)`` only) or fitted on a reference
    DataFrame to also detect schema drift between pipeline runs.

    Parameters
    ----------
    max_null_rate:
        Null rate above which a column is flagged as a warning. Default 0.5.
    required_columns:
        Columns that must be present and non-null. Any missing column is
        an error; any all-null required column is also an error.
    expected_dtypes:
        Dict mapping column name to expected dtype string (e.g.
        ``{"subject_id": "int64", "charttime": "datetime64[ns]"}``).
        Dtype mismatches are reported as warnings.
    min_rows:
        Minimum number of rows expected. Fewer rows triggers an error.
    max_rows:
        Maximum number of rows expected. More rows triggers a warning.

    Examples
    --------
    >>> checker = DataQualityChecker(required_columns=["subject_id", "charttime"])
    >>> checker.fit(train_df)          # learn reference schema and row count
    >>> report = checker.check(df)
    >>> print(report.summary())
    >>> if not report.passed:
    ...     raise RuntimeError("Data quality check failed")

    >>> # Standalone (no reference schema)
    >>> report = DataQualityChecker(max_null_rate=0.3).check(df)
    """

    def __init__(
        self,
        max_null_rate: float = 0.5,
        required_columns: list[str] | None = None,
        expected_dtypes: dict[str, str] | None = None,
        min_rows: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.max_null_rate = max_null_rate
        self.required_columns: list[str] = required_columns or []
        self.expected_dtypes: dict[str, str] = expected_dtypes or {}
        self.min_rows = min_rows
        self.max_rows = max_rows
        self._reference_schema: dict[str, str] = {}
        self._reference_row_count: int | None = None

    def fit(self, df: pd.DataFrame) -> DataQualityChecker:
        """
        Learn the reference schema and row count from a baseline DataFrame.

        After fitting, ``check()`` will also report columns that were added
        or removed relative to this reference.

        Parameters
        ----------
        df:
            Reference DataFrame (typically the training set).

        Returns
        -------
        DataQualityChecker
            Self, for method chaining.
        """
        self._reference_schema = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        self._reference_row_count = len(df)
        logger.info(
            f"DataQualityChecker fitted: {len(df):,} rows, {len(self._reference_schema)} columns"
        )
        return self

    def check(self, df: pd.DataFrame) -> QualityReport:
        """
        Run all configured quality checks against ``df``.

        Parameters
        ----------
        df:
            DataFrame to check.

        Returns
        -------
        QualityReport
        """
        issues: list[QualityIssue] = []
        null_rates: dict[str, float] = {}

        # Row count checks
        issues.extend(self._check_row_counts(df))

        # Required column presence
        for col in self.required_columns:
            if col not in df.columns:
                issues.append(
                    QualityIssue(
                        column=col,
                        issue_type="column_removed",
                        severity="error",
                        detail=f"Required column '{col}' is missing from DataFrame",
                    )
                )

        # Schema drift vs reference
        if self._reference_schema:
            issues.extend(self._check_schema_drift(df))

        # Per-column checks
        for col in df.columns:
            null_rate = float(df[col].isna().mean())
            null_rates[col] = null_rate

            # All-null required column → error
            if col in self.required_columns and null_rate == 1.0:
                issues.append(
                    QualityIssue(
                        column=col,
                        issue_type="all_null",
                        severity="error",
                        detail=f"Required column '{col}' is entirely null",
                    )
                )
            # High null rate → warning
            elif null_rate > self.max_null_rate:
                issues.append(
                    QualityIssue(
                        column=col,
                        issue_type="high_null_rate",
                        severity="warning",
                        detail=(
                            f"Column '{col}' null rate {null_rate:.1%} "
                            f"exceeds threshold {self.max_null_rate:.1%}"
                        ),
                    )
                )

            # Expected dtype mismatch
            if col in self.expected_dtypes:
                actual = str(df[col].dtype)
                expected = self.expected_dtypes[col]
                if actual != expected:
                    issues.append(
                        QualityIssue(
                            column=col,
                            issue_type="dtype_changed",
                            severity="warning",
                            detail=(f"Column '{col}' dtype is '{actual}', expected '{expected}'"),
                        )
                    )

        n_errors = sum(1 for i in issues if i.severity == "error")
        n_warnings = sum(1 for i in issues if i.severity == "warning")
        status = "FAILED" if n_errors else "PASSED"
        logger.info(
            f"DataQualityChecker [{status}]: "
            f"{n_errors} errors, {n_warnings} warnings "
            f"on {len(df):,} rows × {len(df.columns)} columns"
        )

        return QualityReport(
            issues=issues,
            n_rows=len(df),
            n_columns=len(df.columns),
            null_rates=null_rates,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_row_counts(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        n = len(df)

        if self.min_rows is not None and n < self.min_rows:
            issues.append(
                QualityIssue(
                    column="__dataframe__",
                    issue_type="row_count_anomaly",
                    severity="error",
                    detail=f"DataFrame has {n:,} rows, below minimum of {self.min_rows:,}",
                )
            )
        if self.max_rows is not None and n > self.max_rows:
            issues.append(
                QualityIssue(
                    column="__dataframe__",
                    issue_type="row_count_anomaly",
                    severity="warning",
                    detail=f"DataFrame has {n:,} rows, above maximum of {self.max_rows:,}",
                )
            )

        if self._reference_row_count is not None:
            ratio = n / self._reference_row_count if self._reference_row_count > 0 else 0.0
            if ratio < 0.5:
                issues.append(
                    QualityIssue(
                        column="__dataframe__",
                        issue_type="row_count_anomaly",
                        severity="warning",
                        detail=(
                            f"DataFrame has {n:,} rows — only {ratio:.0%} of the "
                            f"reference row count ({self._reference_row_count:,})"
                        ),
                    )
                )

        return issues

    def _check_schema_drift(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        ref_cols = set(self._reference_schema.keys())
        cur_cols = set(df.columns)

        for col in sorted(ref_cols - cur_cols):
            severity = "error" if col in self.required_columns else "warning"
            issues.append(
                QualityIssue(
                    column=col,
                    issue_type="column_removed",
                    severity=severity,
                    detail=f"Column '{col}' present in reference but missing from current",
                )
            )

        for col in sorted(cur_cols - ref_cols):
            issues.append(
                QualityIssue(
                    column=col,
                    issue_type="column_added",
                    severity="warning",
                    detail=f"Column '{col}' not present in reference DataFrame",
                )
            )

        for col in sorted(ref_cols & cur_cols):
            ref_dtype = self._reference_schema[col]
            cur_dtype = str(df[col].dtype)
            if ref_dtype != cur_dtype:
                issues.append(
                    QualityIssue(
                        column=col,
                        issue_type="dtype_changed",
                        severity="warning",
                        detail=(
                            f"Column '{col}' dtype changed from '{ref_dtype}' "
                            f"(reference) to '{cur_dtype}' (current)"
                        ),
                    )
                )

        return issues
