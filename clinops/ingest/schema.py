"""
Schema validation primitives used across clinops.ingest loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when a loaded table does not match the expected schema."""

    pass


@dataclass
class ColumnSpec:
    """Specification for a single column in a clinical table."""

    name: str
    dtype: str | None = None  # e.g. "int64", "float64", "datetime64[ns]"
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] = field(default_factory=list)


@dataclass
class ClinicalSchema:
    """
    Declarative schema for a clinical data table.

    Parameters
    ----------
    name:
        Human-readable name for this schema (used in error messages).
    columns:
        List of ColumnSpec objects describing required and optional columns.
    allow_extra_columns:
        If True (default), columns not in the spec are silently allowed.

    Example
    -------
    >>> schema = ClinicalSchema(
    ...     name="vitals",
    ...     columns=[
    ...         ColumnSpec("subject_id", dtype="int64", nullable=False),
    ...         ColumnSpec("heart_rate", dtype="float64", min_value=0, max_value=300),
    ...     ]
    ... )
    >>> schema.validate(df)
    """

    name: str
    columns: list[ColumnSpec] = field(default_factory=list)
    allow_extra_columns: bool = True

    def validate(self, df: pd.DataFrame, strict: bool = True) -> list[str]:
        """
        Validate a DataFrame against this schema.

        Parameters
        ----------
        df:
            DataFrame to validate.
        strict:
            If True, raise SchemaValidationError on the first violation.
            If False, collect all violations and return them as a list.

        Returns
        -------
        list[str]
            Empty list if valid; list of violation messages otherwise.
        """
        violations: list[str] = []

        for spec in self.columns:
            if spec.name not in df.columns:
                msg = f"[{self.name}] Missing required column: '{spec.name}'"
                violations.append(msg)
                if strict:
                    raise SchemaValidationError(msg)
                continue

            col = df[spec.name]

            # Nullability check
            if not spec.nullable and col.isna().any():
                null_count = col.isna().sum()
                msg = (
                    f"[{self.name}] Column '{spec.name}' has "
                    f"{null_count} null values (nullable=False)"
                )
                violations.append(msg)
                if strict:
                    raise SchemaValidationError(msg)

            # Range checks (numeric only)
            if spec.min_value is not None and pd.api.types.is_numeric_dtype(col):
                out_of_range = (col < spec.min_value).sum()
                if out_of_range:
                    msg = (
                        f"[{self.name}] Column '{spec.name}' has {out_of_range} values "
                        f"below min_value={spec.min_value}"
                    )
                    violations.append(msg)
                    if strict:
                        raise SchemaValidationError(msg)

            if spec.max_value is not None and pd.api.types.is_numeric_dtype(col):
                out_of_range = (col > spec.max_value).sum()
                if out_of_range:
                    msg = (
                        f"[{self.name}] Column '{spec.name}' has {out_of_range} values "
                        f"above max_value={spec.max_value}"
                    )
                    violations.append(msg)
                    if strict:
                        raise SchemaValidationError(msg)

            # Allowed values check
            if spec.allowed_values:
                invalid = ~col.isin(spec.allowed_values)
                invalid_count = invalid.sum()
                if invalid_count:
                    msg = (
                        f"[{self.name}] Column '{spec.name}' has {invalid_count} values "
                        f"not in allowed_values={spec.allowed_values}"
                    )
                    violations.append(msg)
                    if strict:
                        raise SchemaValidationError(msg)

        return violations
