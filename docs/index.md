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

## Quickstart

```bash
pip install clinops
```

```python
from clinops.ingest import MimicTableLoader
from clinops.preprocess import ClinicalOutlierClipper
from clinops.temporal import TemporalWindower, ImputationStrategy
from clinops.split import StratifiedPatientSplitter

# Load MIMIC-IV vitals
tbl = MimicTableLoader("/data/mimic-iv-2.2")
charts = tbl.chartevents(subject_ids=[10000032, 10000980])

# Clip physiologically impossible values
charts = ClinicalOutlierClipper(action="clip").fit_transform(charts)

# Build 24-hour windows with 6-hour stride
windows = TemporalWindower(window_hours=24, step_hours=6).fit_transform(
    df=charts,
    id_col="subject_id",
    time_col="charttime",
    feature_cols=["heart_rate", "spo2", "resp_rate"],
)

# Patient-stratified split — no leakage
result = StratifiedPatientSplitter(
    id_col="subject_id",
    outcome_col="hospital_expire_flag",
    test_size=0.2,
).split(windows)
```

## Installation

Requires Python **3.12+**.

```bash
pip install clinops           # core
pip install clinops[fhir]     # adds FHIR R4 loader
pip install clinops[gcp]      # adds GCP extras
pip install -e ".[dev]"       # development (includes docs, linting, tests)
```
