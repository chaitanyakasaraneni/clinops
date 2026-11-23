# clinops

**Clinical ML Pipeline Toolkit** — production-grade data loading, preprocessing, and time-series feature engineering for healthcare AI research.

[![PyPI version](https://img.shields.io/pypi/v/clinops.svg)](https://pypi.org/project/clinops/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/chaitanyakasaraneni/clinops/actions/workflows/ci.yml/badge.svg)](https://github.com/chaitanyakasaraneni/clinops/actions)

---

Every healthcare AI project starts with the same two weeks of plumbing: loading MIMIC tables without hitting memory limits, clipping physiologically impossible values before they corrupt your model, normalizing glucose from mmol/L to mg/dL across sites, building time-series windows that handle clinical missingness correctly, and splitting data without leaking patients across folds. `clinops` packages those hard-won patterns into a single, well-tested library so your first notebook is actual science.

## Modules

| Module | What it does |
|---|---|
| [`clinops.ingest`](guide/ingest.md) | Loaders for MIMIC-IV, MIMIC-III, FHIR R4, and flat CSV/Parquet with schema validation |
| [`clinops.preprocess`](guide/preprocess.md) | Outlier clipping with physiological bounds, unit normalization, ICD-9→10 mapping |
| [`clinops.temporal`](guide/temporal.md) | Sliding/tumbling windows, gap-aware imputation, lag features, cohort alignment |
| [`clinops.split`](guide/split.md) | Temporal, patient-level, and stratified patient train/test splitting |
| [`clinops.monitor`](guide/monitor.md) | Distribution drift detection (PSI + KS) and data quality alerting for production pipelines |
| [`clinops.orchestrate`](guide/orchestrate.md) | GCS/S3 artifact storage and AWS Step Functions pipeline builder |

## Quickstart

```bash
pip install clinops
```

```python
from clinops.ingest import MimicTableLoader
from clinops.preprocess import ClinicalOutlierClipper
from clinops.split import StratifiedPatientSplitter
from clinops.temporal import TemporalWindower

VITALS = {220045: "heart_rate", 220277: "spo2", 220210: "resp_rate"}

# Load MIMIC-IV vital signs (long format: one row per measurement)
tbl = MimicTableLoader("/data/mimic-iv-2.2")
charts = tbl.chartevents(item_ids=list(VITALS))

# One column per vital sign, then clip physiologically impossible values
vitals = (
    charts.assign(vital=charts["itemid"].map(VITALS))
    .pivot_table(index=["hadm_id", "charttime"], columns="vital", values="valuenum")
    .reset_index()
)
vitals = ClinicalOutlierClipper(action="clip").fit_transform(vitals)

# 24-hour windows with a 6-hour stride, one series per admission
windows = TemporalWindower(window_hours=24, step_hours=6).fit_transform(
    df=vitals,
    id_col="hadm_id",
    time_col="charttime",
    feature_cols=["heart_rate", "spo2", "resp_rate"],
)

# Attach each admission's patient and outcome, then split by patient
adm = tbl.admissions()
windows = windows.merge(adm[["hadm_id", "subject_id", "hospital_expire_flag"]], on="hadm_id")
result = StratifiedPatientSplitter(
    id_col="subject_id",
    outcome_col="hospital_expire_flag",
    test_size=0.2,
).split(windows)
print(result.summary())
```

[Getting started](getting-started.md) builds the same pipeline step by step, per ICU
stay and limited to the first 48 hours.

## Examples

- [Getting started notebook](examples.md#01--getting-started) — module-by-module walkthrough with synthetic data, no MIMIC access needed
- [ICU mortality pipeline notebook](examples.md#02--icu-mortality-prediction-pipeline) — end-to-end pipeline from raw EHR tables to ML-ready arrays

---

## Installation

Requires Python **3.12+**.

```bash
pip install clinops           # core, including the MIMIC, eICU, FHIR R4, and flat-file loaders
pip install clinops[gcp]      # adds GCS storage for clinops.orchestrate
pip install clinops[aws]      # adds S3 storage and Step Functions for clinops.orchestrate
pip install -e ".[dev]"       # development (includes docs, linting, tests)
```
