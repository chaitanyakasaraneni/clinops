"""
Clinical-aware train/test splitting.

Standard sklearn.train_test_split is inappropriate for clinical ML:

1. Random splits violate temporal ordering — a model trained on t+1
   data and tested on t-1 data will appear to perform well but will
   fail in production where the future is not available.

2. Patient leakage — if patient 123 has three admissions and they end
   up in both train and test, the model can memorise patient-specific
   patterns rather than generalising across patients.

3. Outcome imbalance — random splits on small datasets can produce
   train/test splits with very different outcome rates, making
   evaluation unreliable.

All three splitters here address one or more of these problems.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SplitResult:
    """
    The result of a train/test split operation.

    Attributes
    ----------
    train:
        Training set DataFrame.
    test:
        Test set DataFrame.
    metadata:
        Dict with split statistics (sizes, outcome rates, cutoff, etc.)
    """

    train: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)

    @property
    def train_frac(self) -> float:
        total = self.train_size + self.test_size
        return self.train_size / total if total > 0 else 0.0

    def summary(self) -> str:
        """Return a human-readable summary of the split."""
        lines = [
            f"Train: {self.train_size:,} rows ({self.train_frac:.1%})",
            f"Test:  {self.test_size:,} rows ({1 - self.train_frac:.1%})",
        ]
        for k, v in self.metadata.items():
            if isinstance(v, float):
                lines.append(f"{k}: {v:.4f}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)


class TemporalSplitter:
    """
    Split clinical data on a datetime cutoff.

    All rows with ``time_col`` < ``cutoff`` go to train; all rows with
    ``time_col`` >= ``cutoff`` go to test. This is the only split
    strategy that respects temporal ordering and prevents future leakage.

    Parameters
    ----------
    cutoff:
        Datetime string, pd.Timestamp, or None. If None, ``train_frac``
        is used to compute the cutoff automatically from the data.
    train_frac:
        Fraction of the time range to use for training when ``cutoff``
        is None. Default 0.8 (use the earliest 80% of time as train).
    time_col:
        Name of the datetime column. Default ``"charttime"``.

    Examples
    --------
    >>> splitter = TemporalSplitter(cutoff="2155-01-01")
    >>> result = splitter.split(df)
    >>> print(result.summary())

    >>> # Auto-cutoff at 80% of the time range
    >>> splitter = TemporalSplitter(train_frac=0.8, time_col="admittime")
    >>> result = splitter.split(df)
    """

    def __init__(
        self,
        cutoff: str | pd.Timestamp | None = None,
        train_frac: float = 0.8,
        time_col: str = "charttime",
    ) -> None:
        self.cutoff = pd.Timestamp(cutoff) if cutoff is not None else None
        self.train_frac = train_frac
        self.time_col = time_col

    def split(self, df: pd.DataFrame) -> SplitResult:
        """
        Split df into train and test sets.

        Parameters
        ----------
        df:
            Input DataFrame. Must contain ``time_col``.

        Returns
        -------
        SplitResult
        """
        if self.time_col not in df.columns:
            raise ValueError(f"time_col '{self.time_col}' not found in DataFrame")

        times = pd.to_datetime(df[self.time_col])
        cutoff = self.cutoff

        if cutoff is None:
            t_min = times.min()
            t_max = times.max()
            duration = t_max - t_min
            cutoff = t_min + duration * self.train_frac
            logger.info(
                f"TemporalSplitter: auto-cutoff at {cutoff} "
                f"({self.train_frac:.0%} of [{t_min}, {t_max}])"
            )

        train_mask = times < cutoff
        test_mask = times >= cutoff

        train = df[train_mask].reset_index(drop=True)
        test = df[test_mask].reset_index(drop=True)

        logger.info(f"TemporalSplitter: cutoff={cutoff} → train={len(train):,}, test={len(test):,}")

        return SplitResult(
            train=train,
            test=test,
            metadata={
                "cutoff": str(cutoff),
                "time_col": self.time_col,
                "train_rows": len(train),
                "test_rows": len(test),
            },
        )


class PatientSplitter:
    """
    Split clinical data at the patient level.

    Ensures all rows for a given patient are entirely in train or
    entirely in test — no patient appears in both splits. This is
    required to prevent label leakage in multi-admission datasets.

    Parameters
    ----------
    id_col:
        Patient identifier column. Default ``"subject_id"``.
    test_size:
        Fraction of patients to hold out for testing. Default 0.2.
    random_state:
        Random seed for reproducibility. Default 42.

    Examples
    --------
    >>> splitter = PatientSplitter(id_col="subject_id", test_size=0.2)
    >>> result = splitter.split(df)
    >>> # Verify no patient leakage
    >>> assert not set(result.train["subject_id"]) & set(result.test["subject_id"])
    """

    def __init__(
        self,
        id_col: str = "subject_id",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        self.id_col = id_col
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> SplitResult:
        """
        Split df into train and test sets at the patient level.

        Parameters
        ----------
        df:
            Input DataFrame. Must contain ``id_col``.

        Returns
        -------
        SplitResult
        """
        if self.id_col not in df.columns:
            raise ValueError(f"id_col '{self.id_col}' not found in DataFrame")

        rng = np.random.default_rng(self.random_state)
        all_patients = df[self.id_col].unique()
        n_patients = len(all_patients)
        n_test = max(1, round(n_patients * self.test_size))

        shuffled = rng.permutation(all_patients)
        test_patients = set(shuffled[:n_test])
        train_patients = set(shuffled[n_test:])

        train = df[df[self.id_col].isin(train_patients)].reset_index(drop=True)
        test = df[df[self.id_col].isin(test_patients)].reset_index(drop=True)

        logger.info(
            f"PatientSplitter: {n_patients} patients → "
            f"train={len(train_patients)} patients ({len(train):,} rows), "
            f"test={len(test_patients)} patients ({len(test):,} rows)"
        )

        # Verify no leakage
        overlap = set(train[self.id_col].unique()) & set(test[self.id_col].unique())
        if overlap:
            logger.error(f"PatientSplitter: {len(overlap)} patients leaked across splits!")

        return SplitResult(
            train=train,
            test=test,
            metadata={
                "id_col": self.id_col,
                "n_train_patients": len(train_patients),
                "n_test_patients": len(test_patients),
                "train_rows": len(train),
                "test_rows": len(test),
                "random_state": self.random_state,
            },
        )


class StratifiedPatientSplitter:
    """
    Patient-level split with outcome stratification.

    Combines the patient-boundary guarantee of PatientSplitter with
    stratification on a binary or multi-class outcome column. Ensures
    the outcome rate in train and test approximately matches the
    population rate, which is important for imbalanced clinical outcomes
    (e.g., in-hospital mortality typically 5–15%).

    The algorithm:
    1. Compute per-patient outcome (e.g., any positive in admissions)
    2. Separately sample positive and negative patients at ``test_size``
    3. Combine → test set has ~same positive rate as full population

    Parameters
    ----------
    id_col:
        Patient identifier column. Default ``"subject_id"``.
    outcome_col:
        Binary outcome column (0/1 or bool). Default ``"mortality"``.
    test_size:
        Fraction of patients to hold out. Default 0.2.
    patient_outcome_fn:
        Function that maps a per-patient group of outcome values → scalar label.
        Default: any positive observation → patient is positive.
    random_state:
        Random seed. Default 42.

    Examples
    --------
    >>> splitter = StratifiedPatientSplitter(
    ...     id_col="subject_id",
    ...     outcome_col="hospital_expire_flag",
    ...     test_size=0.2,
    ... )
    >>> result = splitter.split(df)
    >>> print(result.summary())
    """

    def __init__(
        self,
        id_col: str = "subject_id",
        outcome_col: str = "mortality",
        test_size: float = 0.2,
        patient_outcome_fn: Callable[[pd.Series], int] | None = None,
        random_state: int = 42,
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        self.id_col = id_col
        self.outcome_col = outcome_col
        self.test_size = test_size
        self._outcome_fn: Callable[[pd.Series], int] = patient_outcome_fn or (
            lambda s: int(s.max())
        )
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> SplitResult:
        """
        Split df with patient-level stratification on outcome.

        Parameters
        ----------
        df:
            Input DataFrame. Must contain ``id_col`` and ``outcome_col``.

        Returns
        -------
        SplitResult
        """
        for col in (self.id_col, self.outcome_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        rng = np.random.default_rng(self.random_state)

        # Derive per-patient outcome label
        patient_labels = (
            df.groupby(self.id_col)[self.outcome_col]
            .apply(lambda s: self._outcome_fn(s))
            .reset_index()
        )
        patient_labels.columns = [self.id_col, "_label"]

        positives = patient_labels[patient_labels["_label"] == 1][self.id_col].to_numpy()
        negatives = patient_labels[patient_labels["_label"] == 0][self.id_col].to_numpy()

        n_pos_test = max(1, round(len(positives) * self.test_size))
        n_neg_test = max(1, round(len(negatives) * self.test_size))

        test_pos = set(rng.choice(positives, size=n_pos_test, replace=False))
        test_neg = set(rng.choice(negatives, size=n_neg_test, replace=False))
        test_patients = test_pos | test_neg
        train_patients = set(patient_labels[self.id_col].values) - test_patients

        train = df[df[self.id_col].isin(train_patients)].reset_index(drop=True)
        test = df[df[self.id_col].isin(test_patients)].reset_index(drop=True)

        # Compute outcome rates for reporting
        pop_rate = float(df[self.outcome_col].mean())
        train_rate = float(train[self.outcome_col].mean()) if len(train) else 0.0
        test_rate = float(test[self.outcome_col].mean()) if len(test) else 0.0

        logger.info(
            f"StratifiedPatientSplitter: population rate={pop_rate:.3f} | "
            f"train rate={train_rate:.3f} ({len(train_patients)} patients, {len(train):,} rows) | "
            f"test rate={test_rate:.3f} ({len(test_patients)} patients, {len(test):,} rows)"
        )

        return SplitResult(
            train=train,
            test=test,
            metadata={
                "id_col": self.id_col,
                "outcome_col": self.outcome_col,
                "population_outcome_rate": round(pop_rate, 4),
                "train_outcome_rate": round(train_rate, 4),
                "test_outcome_rate": round(test_rate, 4),
                "n_train_patients": len(train_patients),
                "n_test_patients": len(test_patients),
                "train_rows": len(train),
                "test_rows": len(test),
            },
        )
