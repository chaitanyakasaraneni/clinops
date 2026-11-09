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

This walkthrough mirrors a typical research workflow: load MIMIC-IV data, preprocess it, build temporal windows, and produce a patient-level train/test split without data leakage. It predicts in-hospital mortality from the first 48 hours of each ICU stay.

The code runs as written against MIMIC-IV 2.2; CI runs it on a synthetic copy of the tables.

### 1. Load data

```python
from clinops.ingest import MimicTableLoader

tbl = MimicTableLoader("/data/mimic-iv-2.2")
subjects = list(range(10000032, 10000132))

# MIMIC-IV item IDs for four vital signs
VITALS = {220045: "heart_rate", 220277: "spo2", 220210: "resp_rate", 220052: "map"}

# ICU chartevents (long format: one row per measurement, charttime parsed)
charts = tbl.chartevents(subject_ids=subjects, item_ids=list(VITALS))

# ICU stays. A patient can have several, each with its own intime
stays = tbl.icustays(subject_ids=subjects)

# Admissions contain the hospital_expire_flag outcome
adm = tbl.admissions(subject_ids=subjects)
```

### 2. Reshape and preprocess

`chartevents` has one row per measurement. The preprocessing transformers work on named columns, so pivot to one column per vital sign first.

```python
from clinops.preprocess import ClinicalOutlierClipper

vitals = (
    charts.assign(vital=charts["itemid"].map(VITALS))
    .pivot_table(
        index=["subject_id", "stay_id", "charttime"],
        columns="vital",
        values="valuenum",
        aggfunc="mean",
    )
    .reset_index()
    .rename_axis(columns=None)
)

# Clip values outside physiological bounds (heart_rate, spo2, resp_rate, map, ...)
vitals = ClinicalOutlierClipper(action="clip").fit_transform(vitals)
```

If a lab is recorded in mixed units, add a unit column and use `UnitNormalizer` here too; see the [preprocess guide](guide/preprocess.md).

### 3. Align to ICU admission

`CohortAligner` needs exactly one anchor time per id. Because a patient can have several ICU stays, align on `stay_id` so each measurement is measured from the start of its own stay.

```python
from clinops.temporal import CohortAligner

# Keep only measurements within 48 hours of each ICU admission
aligned = CohortAligner(
    anchor_col="intime",
    id_col="stay_id",
    max_hours_before=0,
    max_hours_after=48,
).align(events_df=vitals, anchor_df=stays[["stay_id", "intime"]])
```

To use only each patient's first stay instead, pass
`stays.sort_values("intime").drop_duplicates("subject_id")` as `anchor_df` with `id_col="subject_id"`.

### 4. Build temporal windows

```python
from clinops.temporal import ImputationStrategy, TemporalWindower

# 24-hour sliding windows, stepped every 6 hours, one series per stay.
# Windows with no measurement of a vital take the previous window's value.
windower = TemporalWindower(
    window_hours=24,
    step_hours=6,
    imputation=ImputationStrategy.FORWARD_FILL,
)
windows = windower.fit_transform(
    df=aligned,
    id_col="stay_id",
    time_col="charttime",
    feature_cols=["heart_rate", "spo2", "resp_rate", "map"],
)
```

Values still missing at the start of a stay stay `NaN`. Fill them with an `Imputer` fitted on the training split only (after step 5) so test data never influences the fill values.

### 5. Add outcome and split

```python
from clinops.split import StratifiedPatientSplitter

# Outcome is per admission; map it to each stay, then to its windows
stay_outcome = stays[["stay_id", "subject_id", "hadm_id"]].merge(
    adm[["hadm_id", "hospital_expire_flag"]], on="hadm_id", how="left"
)
windows = windows.merge(
    stay_outcome[["stay_id", "subject_id", "hospital_expire_flag"]],
    on="stay_id",
    how="left",
)

# Split by patient, not by stay, so a patient's stays stay together.
# A patient who died in any admission counts as positive for stratification.
result = StratifiedPatientSplitter(
    id_col="subject_id",
    outcome_col="hospital_expire_flag",
    test_size=0.2,
).split(windows)

print(result.summary())
train_df = result.train
test_df = result.test
```

---

## Next steps

- [Ingest guide](guide/ingest.md) — all loader options including FHIR and flat files
- [Preprocess guide](guide/preprocess.md) — outlier bounds, unit conversions, ICD mapping
- [Temporal guide](guide/temporal.md) — windowing strategies, imputation, lag features
- [Split guide](guide/split.md) — temporal, patient, and stratified splits
- [API Reference](api/ingest.md) — full class and method signatures
