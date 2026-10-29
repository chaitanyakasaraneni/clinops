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


class LeakageError(ValueError):
    """
    Raised when a label-defining variable would leak into the feature matrix.

    clinops enforces a programmatic exclusion list at feature-assembly time:
    variables that are used only to *derive* outcome labels (e.g. a patient's
    baseline creatinine, or the PaO2/FiO2 ratio) must never appear as model
    inputs. If one is found in the assembled feature set, this is raised with a
    message naming the offending column.
    """

    pass


@dataclass
class ColumnSpec:
    """Specification for a single column in a clinical table."""

    name: str
    # Expected dtype family, e.g. "int64", "float64", "datetime64[ns]", "str",
    # "bool", "category". Checked by kind: any integer dtype satisfies "int64";
    # integer or float satisfies "float64"; object or string satisfies "str".
    dtype: str | None = None
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
        If True (default), columns not in the spec are allowed. If False,
        any column not listed in ``columns`` is a violation.

    Examples
    --------
    >>> import pandas as pd
    >>> from clinops.ingest import ClinicalSchema, ColumnSpec
    >>> schema = ClinicalSchema(
    ...     name="vitals",
    ...     columns=[
    ...         ColumnSpec("subject_id", dtype="int64", nullable=False),
    ...         ColumnSpec("heart_rate", dtype="float64", min_value=0, max_value=300),
    ...     ],
    ... )
    >>> df = pd.DataFrame({"subject_id": [1, 2], "heart_rate": [72.0, 350.0]})
    >>> schema.validate(df, strict=False)
    ["[vitals] Column 'heart_rate' has 1 values above max_value=300"]
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

        def report(msg: str) -> None:
            violations.append(msg)
            if strict:
                raise SchemaValidationError(msg)

        if not self.allow_extra_columns:
            expected = {spec.name for spec in self.columns}
            extra = [c for c in df.columns if c not in expected]
            if extra:
                report(f"[{self.name}] Unexpected columns {extra} (allow_extra_columns=False)")

        for spec in self.columns:
            if spec.name not in df.columns:
                report(f"[{self.name}] Missing required column: '{spec.name}'")
                continue

            col = df[spec.name]

            # Nullability check
            if not spec.nullable and col.isna().any():
                null_count = col.isna().sum()
                report(
                    f"[{self.name}] Column '{spec.name}' has "
                    f"{null_count} null values (nullable=False)"
                )

            # Dtype check
            if spec.dtype is not None and not _dtype_matches(col, spec.dtype):
                report(
                    f"[{self.name}] Column '{spec.name}' has dtype '{col.dtype}', "
                    f"expected '{spec.dtype}'"
                )

            # Range checks need a numeric column; a text column cannot be range-checked.
            has_range = spec.min_value is not None or spec.max_value is not None
            if has_range and not pd.api.types.is_numeric_dtype(col) and col.notna().any():
                report(
                    f"[{self.name}] Column '{spec.name}' has non-numeric dtype "
                    f"'{col.dtype}', so min_value/max_value cannot be checked"
                )

            if spec.min_value is not None and pd.api.types.is_numeric_dtype(col):
                out_of_range = (col < spec.min_value).sum()
                if out_of_range:
                    report(
                        f"[{self.name}] Column '{spec.name}' has {out_of_range} values "
                        f"below min_value={spec.min_value}"
                    )

            if spec.max_value is not None and pd.api.types.is_numeric_dtype(col):
                out_of_range = (col > spec.max_value).sum()
                if out_of_range:
                    report(
                        f"[{self.name}] Column '{spec.name}' has {out_of_range} values "
                        f"above max_value={spec.max_value}"
                    )

            # Allowed values check
            if spec.allowed_values:
                invalid = ~col.isin(spec.allowed_values)
                invalid_count = invalid.sum()
                if invalid_count:
                    report(
                        f"[{self.name}] Column '{spec.name}' has {invalid_count} values "
                        f"not in allowed_values={spec.allowed_values}"
                    )

        return violations


def _dtype_matches(col: pd.Series[Any], expected: str) -> bool:
    """Check ``col``'s dtype against the dtype family named by ``expected``."""
    actual = col.dtype
    api = pd.api.types
    name = expected.lower()
    if name.startswith(("int", "uint")):
        return bool(api.is_integer_dtype(col))
    if name.startswith("float"):
        return bool(api.is_numeric_dtype(col)) and not api.is_bool_dtype(col)
    if name.startswith("datetime"):
        return bool(api.is_datetime64_any_dtype(col))
    if name.startswith("timedelta"):
        return bool(api.is_timedelta64_dtype(col))
    if name in {"str", "string", "object"}:
        return bool(api.is_string_dtype(col) or api.is_object_dtype(col))
    if name in {"bool", "boolean"}:
        return bool(api.is_bool_dtype(col))
    if name == "category":
        return isinstance(actual, pd.CategoricalDtype)
    return str(actual).lower() == name
