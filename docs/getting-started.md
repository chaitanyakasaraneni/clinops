# Getting Started

## Installation

Requires Python **3.12+**.

```bash
pip install clinops
```

Install optional extras for FHIR or cloud support:

```bash
pip install clinops[fhir]   # FHIR R4 loader
pip install clinops[gcp]    # Google Cloud Storage / BigQuery
pip install clinops[aws]    # AWS S3 / boto3
```

For development (includes tests, linting, and docs):

```bash
git clone https://github.com/chaitanyakasaraneni/clinops
cd clinops
pip install -e ".[dev]"
```

## Verify the install

```python
import clinops
print(clinops.__version__)
```

---

## End-to-end example

This walkthrough mirrors a typical research workflow: load MIMIC-IV data, preprocess it, build temporal windows, and produce a patient-level train/test split without data leakage.

### 1. Load data

```python
from clinops.ingest import MimicTableLoader

tbl = MimicTableLoader("/data/mimic-iv-2.2")

# ICU chartevents — charttime parsed as datetime automatically
charts = tbl.chartevents(subject_ids=list(range(10000032, 10000132)))

# ICU stays — needed to align windows to admission time
stays = tbl.icustays(subject_ids=list(range(10000032, 10000132)))

# Admissions — contains hospital_expire_flag outcome
adm = tbl.admissions(subject_ids=list(range(10000032, 10000132)))
```

### 2. Preprocess

```python
from clinops.preprocess import ClinicalOutlierClipper, UnitNormalizer

# Clip values outside physiological bounds (heart_rate, spo2, sbp, ...)
charts = ClinicalOutlierClipper(action="clip").fit_transform(charts)

# Normalize any mixed-unit columns (e.g. glucose in mmol/L → mg/dL)
if "glucose_unit" in charts.columns:
    charts = UnitNormalizer(
        column_unit_map={"glucose": "glucose_unit"}
    ).transform(charts)
```

### 3. Align to ICU admission

```python
from clinops.temporal import CohortAligner

# Keep only measurements within 48 hours of ICU admission
aligned = CohortAligner(
    anchor_col="intime",
    max_hours_before=0,
    max_hours_after=48,
).align(events_df=charts, anchor_df=stays)
```

### 4. Build temporal windows

```python
from clinops.temporal import TemporalWindower, Imputer, ImputationStrategy

# 24-hour sliding windows, stepped every 6 hours
windower = TemporalWindower(window_hours=24, step_hours=6)
windows = windower.fit_transform(
    df=aligned,
    id_col="subject_id",
    time_col="charttime",
    feature_cols=["heart_rate", "spo2", "resp_rate", "map"],
)

# Gap-aware forward fill — does not propagate across patients or
# across gaps longer than 6 hours
imputer = Imputer(
    ImputationStrategy.FORWARD_FILL,
    max_gap_hours=6,
    time_col="charttime",
    id_col="subject_id",
)
windows = imputer.fit_transform(windows)
```

### 5. Add outcome and split

```python
from clinops.split import StratifiedPatientSplitter

# Attach outcome from admissions table
windows = windows.merge(
    adm[["subject_id", "hospital_expire_flag"]],
    on="subject_id",
    how="left",
)

# Stratified patient split — preserves outcome rate, no cross-patient leakage
result = StratifiedPatientSplitter(
    id_col="subject_id",
    outcome_col="hospital_expire_flag",
    test_size=0.2,
).split(windows)

print(result.summary())
train_df = result.train
test_df  = result.test
```

---

## Next steps

- [Ingest guide](guide/ingest.md) — all loader options including FHIR and flat files
- [Preprocess guide](guide/preprocess.md) — outlier bounds, unit conversions, ICD mapping
- [Temporal guide](guide/temporal.md) — windowing strategies, imputation, lag features
- [Split guide](guide/split.md) — temporal, patient, and stratified splits
- [API Reference](api/ingest.md) — full class and method signatures
