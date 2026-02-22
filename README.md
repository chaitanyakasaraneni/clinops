# clinops 🏥

**Clinical ML Pipeline Toolkit** — production-grade data loading and time-series feature engineering for healthcare AI research.

[![PyPI version](https://img.shields.io/pypi/v/clinops.svg)](https://pypi.org/project/clinops/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/chaitanyakasaraneni/clinops/actions/workflows/ci.yml/badge.svg)](https://github.com/chaitanyakasaraneni/clinops/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psych/black)

---

Every healthcare AI project starts with the same two weeks of plumbing: loading MIMIC-IV tables without hitting memory limits, building time-series windows that handle clinical missingness correctly, aligning multi-patient cohorts to anchor events. `clinops` packages those hard-won patterns into a single, well-tested library so your first notebook is actual science.

Built from production experience in clinical and genomic data engineering across multi-cloud environments.

---

## v0.1 Modules

| Module | What it does |
|---|---|
| `clinops.ingest` | Loaders for MIMIC-IV, FHIR R4, and flat CSV/Parquet with schema validation |
| `clinops.temporal` | Sliding/tumbling windows, gap-aware imputation, lag features, cohort alignment |

**Roadmap:** `clinops.monitor` (drift detection, data quality) and `clinops.orchestrate` (GCS/S3, Step Functions) are planned for v0.2.

---

## Quickstart

```bash
pip install clinops
```

### Load MIMIC-IV ICU data

```python
from clinops.ingest import MimicLoader

loader = MimicLoader("/data/mimic-iv-2.2")

charts = loader.chartevents(
    subject_ids=[10000032, 10000980],
    start_time="2150-01-01",
    end_time="2150-01-10",
)
labs   = loader.labevents(subject_ids=[10000032, 10000980])
stays  = loader.icustays(subject_ids=[10000032, 10000980])
```

### Build temporal feature windows

```python
from clinops.temporal import TemporalWindower, ImputationStrategy

windower = TemporalWindower(
    window_hours=24,
    step_hours=6,
    imputation=ImputationStrategy.FORWARD_FILL,
    min_observations=3,
)

windows = windower.fit_transform(
    df=charts,
    id_col="subject_id",
    time_col="charttime",
    feature_cols=["heart_rate", "spo2", "resp_rate", "map"],
)
# → DataFrame: subject_id | window_start | window_end | heart_rate | spo2 | ...
```

### Long-format input (MIMIC native itemid × valuenum)

```python
windows = windower.fit_transform(
    df=charts,
    id_col="subject_id",
    time_col="charttime",
    item_col="itemid",     # auto-pivots to wide format
    value_col="valuenum",
)
```

### Add lag and rolling features

```python
from clinops.temporal import LagFeatureBuilder

enriched = LagFeatureBuilder(
    lags=[1, 2, 4],
    rolling_windows=[4, 8],
    id_col="subject_id",
).fit_transform(windows)
# Adds: heart_rate_lag1, heart_rate_roll4_mean, heart_rate_roll4_std, ...
```

### Align a cohort to an anchor event (e.g. ICU admission)

```python
from clinops.temporal import CohortAligner

aligned = CohortAligner(
    anchor_col="intime",
    max_hours_before=0,
    max_hours_after=48,
).align(events_df=charts, anchor_df=stays)
# → filtered to 48h post-admission, with hours_from_anchor column
```

### Load FHIR R4 resources

```python
from clinops.ingest import FHIRLoader

loader = FHIRLoader("/data/fhir_export")
obs      = loader.observations(category="vital-signs")
patients = loader.patients()
```

### Validate any flat clinical export

```python
from clinops.ingest import FlatFileLoader, ClinicalSchema, ColumnSpec

schema = ClinicalSchema(
    name="vitals",
    columns=[
        ColumnSpec("subject_id", nullable=False),
        ColumnSpec("heart_rate", min_value=0, max_value=300),
        ColumnSpec("spo2",       min_value=50, max_value=100),
    ]
)
df = FlatFileLoader("vitals.csv", schema=schema).load()
```

---

## Imputation strategies

Clinical data has unique missingness patterns that standard ML windowing gets wrong. `clinops` provides strategies tuned for clinical context:

| Strategy | Best for |
|---|---|
| `FORWARD_FILL` | Slowly-changing vitals — carry last observation forward |
| `BACKWARD_FILL` | Values recorded with lag |
| `LINEAR` | Continuous signals with regular sampling |
| `MEAN` / `MEDIAN` | Fit on training set, apply to test (no leakage) |
| `INDICATOR` | Adds `{col}_missing` binary column — lets model learn from missingness |
| `NONE` | Leave NaN in place |

```python
from clinops.temporal import Imputer, ImputationStrategy

imputer = Imputer(ImputationStrategy.MEAN, per_patient=True, id_col="subject_id")
imputer.fit(train_windows)
test_windows = imputer.transform(test_windows)
```

---

## Installation

Requires Python **3.12+**.

```bash
pip install clinops           # core
pip install clinops[fhir]     # adds FHIR R4 loader
pip install clinops[gcp]      # adds GCP extras (for v0.2)
pip install -e ".[dev]"       # development
```

---

## Supported sources

| Source | Format |
|---|---|
| MIMIC-IV v2.0–v2.2 | CSV, CSV.GZ, Parquet |
| FHIR R4 | JSON Bundle, NDJSON |
| Flat files | CSV, CSV.GZ, Parquet |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `pytest tests/ -v` and `ruff check clinops/` before opening a PR.

---

## Citation

```bibtex
@software{kasaraneni2026clinops,
  author  = {Kasaraneni, Chaitanya},
  title   = {clinops: Clinical ML Pipeline Toolkit},
  year    = {2026},
  url     = {https://github.com/chaitanyakasaraneni/clinops},
  version = {0.1.0}
}
```

A companion JOSS paper is in preparation.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
