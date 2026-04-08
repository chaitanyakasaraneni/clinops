# Split

`clinops.split` provides train/test splitting strategies that are correct for clinical ML. Standard `sklearn.train_test_split` is inappropriate for clinical data for two reasons:

1. **Future leakage** — random splits can put a patient's `t+1` observation in training and `t-1` in test.
2. **Patient leakage** — when the same patient appears in both train and test, the model memorises patient-specific patterns rather than generalising to new patients.

---

## TemporalSplitter

Splits on a datetime cutoff. All rows before the cutoff go to train, all rows after go to test. The correct approach for any time-series clinical data.

```python
from clinops.split import TemporalSplitter

# Explicit cutoff
result = TemporalSplitter(cutoff="2155-01-01", time_col="charttime").split(df)

# Auto-compute cutoff from the data
result = TemporalSplitter(train_frac=0.8, time_col="charttime").split(df)

print(result.summary())
# Train: 38,400 rows (80.0%)
# Test:   9,600 rows (20.0%)
# cutoff: 2155-01-01 00:00:00

train_df = result.train
test_df  = result.test
```

---

## PatientSplitter

Ensures all rows for a given patient appear in the same split. Prevents data leakage in datasets where patients have multiple admissions or time-points.

```python
from clinops.split import PatientSplitter

result = PatientSplitter(id_col="subject_id", test_size=0.2).split(df)

# Guaranteed: no patient appears in both splits
assert not set(result.train["subject_id"]) & set(result.test["subject_id"])
```

---

## StratifiedPatientSplitter

Combines patient-level splitting with outcome stratification. Ensures the train/test outcome rate matches the population rate while respecting patient boundaries.

Critical for imbalanced clinical endpoints — in-hospital mortality is typically 5–15% in MIMIC.

```python
from clinops.split import StratifiedPatientSplitter

result = StratifiedPatientSplitter(
    id_col="subject_id",
    outcome_col="hospital_expire_flag",
    test_size=0.2,
).split(df)

print(result.summary())
# Train: 32,000 rows (80.0%)
# Test:   8,000 rows (20.0%)
# population_outcome_rate: 0.0821
# train_outcome_rate:      0.0819
# test_outcome_rate:       0.0826
```

---

## SplitResult

All splitters return a `SplitResult` with consistent fields:

| Field | Type | Description |
|---|---|---|
| `train` | `pd.DataFrame` | Training split |
| `test` | `pd.DataFrame` | Test split |
| `summary()` | `str` | Human-readable split statistics |

---

## Choosing the right splitter

| Scenario | Recommended splitter |
|---|---|
| Time-series data, no patient ID | `TemporalSplitter` |
| Multiple rows per patient, balanced outcome | `PatientSplitter` |
| Multiple rows per patient, rare outcome | `StratifiedPatientSplitter` |
