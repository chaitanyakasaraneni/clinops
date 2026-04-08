# Monitor

`clinops.monitor` provides drift detection and data quality alerting for production pipelines. Intended for teams running clinops-based pipelines in scheduled or streaming environments where input data distribution can shift over time without warning.

---

## DistributionDriftDetector

Fits on a reference dataset (training set), then computes per-column drift metrics on each new batch.

Two complementary metrics are computed:

- **PSI (Population Stability Index)** — widely used in healthcare model validation. Interpretable thresholds: PSI < 0.1 is stable, 0.1–0.2 warrants review, > 0.2 indicates significant drift.
- **KS test** — a non-parametric two-sample test that provides a p-value and works well for small samples where PSI binning is unreliable.

```python
from clinops.monitor import DistributionDriftDetector, DriftSeverity

detector = DistributionDriftDetector(n_bins=10, run_ks_test=True)
detector.fit(train_df)

report = detector.detect(production_batch_df)
print(report.summary())
# Columns checked : 12
# HIGH drift      : 1
# MEDIUM drift    : 2
# LOW drift       : 9
# HIGH columns    : glucose
# MEDIUM columns  : heart_rate, creatinine

# DataFrame with per-column metrics sorted by PSI
print(report.to_dataframe())

# Get column names drifting at or above MEDIUM
drifted = report.drifted_columns(DriftSeverity.MEDIUM)
```

### Monitoring specific columns

```python
detector = DistributionDriftDetector(
    columns=["heart_rate", "spo2", "glucose"],
)
detector.fit(train_df)
```

### Tuning thresholds

```python
# Tighter thresholds for a high-stakes mortality model
detector = DistributionDriftDetector(
    psi_threshold_medium=0.05,
    psi_threshold_high=0.1,
)
```

---

## DataQualityChecker

Checks null rates, schema drift (added/removed/retyped columns), and row count anomalies. Can be used standalone or fitted on a reference DataFrame to detect schema changes between pipeline runs.

```python
from clinops.monitor import DataQualityChecker

checker = DataQualityChecker(
    required_columns=["subject_id", "charttime", "heart_rate"],
    max_null_rate=0.3,
    min_rows=100,
)
checker.fit(train_df)   # learn reference schema and row count

report = checker.check(incoming_df)
print(report.summary())

if not report.passed:
    raise RuntimeError(f"Data quality check failed:\n{report.summary()}")
```

### Issue types

| Issue type | Severity | Trigger |
|---|---|---|
| `high_null_rate` | warning | Column null rate > `max_null_rate` |
| `all_null` | error | Required column is entirely null |
| `column_removed` | error/warning | Column present in reference but missing from current |
| `column_added` | warning | Column not present in reference |
| `dtype_changed` | warning | Column dtype differs from reference or `expected_dtypes` |
| `row_count_anomaly` | error/warning | Rows below `min_rows`, above `max_rows`, or < 50% of reference |

### Standalone use (no reference schema)

```python
report = DataQualityChecker(
    required_columns=["subject_id"],
    max_null_rate=0.5,
).check(df)
```

---

## Typical pipeline integration

```python
from clinops.monitor import DistributionDriftDetector, DataQualityChecker

# Fit once on training data
quality_checker = DataQualityChecker(required_columns=["subject_id", "charttime"])
quality_checker.fit(train_df)

drift_detector = DistributionDriftDetector()
drift_detector.fit(train_df)

# Run on each new batch
def validate_batch(batch_df):
    quality_report = quality_checker.check(batch_df)
    if not quality_report.passed:
        raise RuntimeError(quality_report.summary())

    drift_report = drift_detector.detect(batch_df)
    high_drift = drift_report.drifted_columns(DriftSeverity.HIGH)
    if high_drift:
        # alert, log, or block depending on your policy
        logger.warning(f"HIGH drift detected: {high_drift}")
```
