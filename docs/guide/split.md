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

Combines patient-level splitting with outcome stratification. Each patient gets one label (by default the maximum outcome value, so any positive row makes the patient positive), and `test_size` of the patients in each class go to test. The train/test class proportions therefore match the population while respecting patient boundaries.

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
# train_class_counts: {0: 7344, 1: 656}
# test_class_counts: {0: 1836, 1: 164}
```

Multiclass outcomes work the same way. Labels can be any values (`-1/1`, integer classes, strings), and `metadata` reports patients per class in each split. The row-level `*_outcome_rate` entries are only added for 0/1 outcomes.

```python
# Ordinal severity 0-3; each patient is labelled with their worst severity.
result = StratifiedPatientSplitter(outcome_col="sofa_band", test_size=0.2).split(df)
result.metadata["test_class_counts"]
# {0: 410, 1: 233, 2: 98, 3: 31}
```

Pass `patient_outcome_fn` to label patients differently, for example `lambda s: s.iloc[-1]` for the last recorded value. A class with a single patient can't be split; it goes to test and a warning is logged.

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
