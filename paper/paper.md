---
title: 'clinops: A Python Toolkit for Clinical ML Data Pipelines'
tags:
  - Python
  - healthcare
  - machine learning
  - MIMIC-IV
  - clinical data
  - bioinformatics
  - EHR
authors:
  - name: Chaitanya Kasaraneni
    orcid: 0000-0001-5792-1095
    affiliation: 1
    url: https://ckasaraneni.com
affiliations:
  - name: Independent Researcher, Chicago, Illinois, USA
    index: 1
date: 23 February 2026
bibliography: paper.bib
---

# Summary

`clinops` is an open-source Python library that provides production-grade building
blocks for clinical machine learning (ML) data pipelines. It addresses the
recurring engineering overhead in healthcare AI research by packaging validated
patterns for data loading, preprocessing, feature engineering, and data splitting
into a single, well-tested toolkit. The library supports MIMIC-IV [@johnson2023mimic],
FHIR R4 [@hl7fhir], and arbitrary flat clinical exports, and is designed to prevent
the class of silent data errors — patient leakage, unit heterogeneity, ICD version
discontinuities, and physiologically impossible values — that commonly affect
clinical ML workflows.

# Statement of Need

Building a clinical ML model requires solving a set of data engineering problems
that are largely orthogonal to the scientific question being asked, yet consume a
disproportionate share of researcher time. Published surveys of healthcare AI
projects consistently identify data preparation as the dominant cost
[@wornow2023shaky; @johnson2018mimic]. Despite this, no general-purpose library
has emerged that addresses the clinical-specific failure modes that distinguish
healthcare data from general tabular ML data.

**Patient leakage** is a widely underreported source of inflated performance
metrics in clinical ML. When `sklearn.train_test_split` [@sklearn] is applied to
admission-level records, a patient with multiple admissions may appear in both
training and test sets. The model memorises patient-specific physiology rather than
generalising across patients, producing optimistic held-out performance that does
not transfer to deployment. Systematic reviews of published clinical prediction
models have identified data leakage as one of the most prevalent methodological
flaws in the field [@roberts2019machine; @wynants2020prediction].

**Unit heterogeneity** in multi-site studies is a source of silent numerical
errors. Glucose reported in mmol/L from a European site is numerically
indistinguishable from mg/dL values from a US site without explicit unit
normalisation; the conversion factor of 18 is never applied automatically by
generic ML pipelines.

**ICD version discontinuity** affects any study spanning the October 2015 US
transition from ICD-9-CM to ICD-10-CM. MIMIC-IV contains both versions; MIMIC-III
is exclusively ICD-9. Without harmonisation, the same diagnosis appears as two
distinct codes, artificially inflating apparent feature cardinality and suppressing
apparent diagnosis prevalence after the transition date.

**Clinical outlier removal** using standard statistical methods (z-score, IQR) is
inappropriate for physiological measurements. A heart rate of 180 bpm is plausible
for a patient in supraventricular tachycardia and carries clinical signal; it is
not an outlier. A heart rate of 800 bpm is a data entry error. The distinction
requires physiological reference knowledge, not statistical thresholds.

`clinops` directly addresses each of these failure modes with validated, tested
abstractions. The library targets clinical ML researchers who need reliable data
pipelines without re-implementing these safeguards from scratch on every project.

# Software Description

`clinops` is organised into four modules:

**`clinops.ingest`** provides loaders for MIMIC-IV (v2.0–v2.2), FHIR R4, and
flat CSV/Parquet files with configurable schema validation. The `MimicTableLoader`
class exposes the five tables most commonly needed in ICU cohort studies —
`chartevents`, `labevents`, `admissions`, `diagnoses_icd`, and `icustays` — with
pre-built schemas, automatic datetime parsing, and convenience flags such as
`primary_only` (restricts diagnoses to the principal diagnosis per admission) and
`with_los_band` (adds a categorical length-of-stay column).

**`clinops.preprocess`** provides three preprocessing transformers.
`ClinicalOutlierClipper` applies published physiological bounds [@normal_ranges]
to clip, null, or flag values that are physiologically implausible, covering 20
vitals and laboratory measurements. `UnitNormalizer` converts between 30 registered
unit pairs (glucose mg/dL ↔ mmol/L, temperature °F ↔ °C, etc.) using either a
companion unit column or unconditional explicit conversions. `ICDMapper` harmonises
ICD-9-CM and ICD-10-CM diagnosis codes using the CMS General Equivalence Mappings
[@cms_gem], ships with approximately 60 curated high-frequency mappings, and can
load the full CMS GEM file for complete coverage.

**`clinops.temporal`** provides tools for time-series feature extraction from
clinical event data. `TemporalWindower` extracts fixed-size sliding or tumbling
windows with gap-aware imputation that prevents stale carry-forward across
observation gaps exceeding a configurable threshold. `LagFeatureBuilder` adds
lagged and rolling-window features per patient. `CohortAligner` filters events to
a specified time window relative to an anchor event (e.g., ICU admission), adding
an `hours_from_anchor` column for downstream temporal modelling. `Imputer` provides
fit/transform imputation (mean, median, forward fill, backward fill, linear,
indicator) fitted on training data and applied to test data without leakage.

**`clinops.split`** provides three patient-aware splitting strategies.
`TemporalSplitter` partitions by a time cutoff, preventing future observations
from appearing in training. `PatientSplitter` assigns complete patient records to
either train or test, with a guarantee that no `subject_id` appears in both
partitions. `StratifiedPatientSplitter` extends patient-level splitting with
stratification on a binary outcome column, preserving the population outcome rate
in both splits — critical for imbalanced clinical endpoints such as in-hospital
mortality (typically 5–15% in ICU cohorts).

All transformers follow the scikit-learn `fit`/`transform`/`fit_transform`
interface where applicable. The library is fully typed, passes `mypy --strict`,
and is tested with 118 unit tests achieving 85% line coverage.

# Example Usage

```python
from clinops.ingest import MimicTableLoader
from clinops.preprocess import ClinicalOutlierClipper, UnitNormalizer, ICDMapper
from clinops.split import StratifiedPatientSplitter
from clinops.temporal import TemporalWindower, ImputationStrategy

# Load MIMIC-IV tables with pre-built schemas
tbl    = MimicTableLoader("/data/mimic-iv-2.2")
charts = tbl.chartevents(subject_ids=[10000032, 10000980])
dx     = tbl.diagnoses_icd(subject_ids=[10000032], primary_only=True)
stays  = tbl.icustays(with_los_band=True)

# Clip physiologically impossible values
clipper = ClinicalOutlierClipper(action="clip")
charts  = clipper.fit_transform(charts)

# Harmonise ICD codes and add chapter labels
mapper        = ICDMapper()
dx            = mapper.harmonize(dx, code_col="icd_code", version_col="icd_version")
dx["chapter"] = mapper.chapter_series(dx["icd_code"])

# Extract 24-hour feature windows with gap-aware imputation
windower = TemporalWindower(
    window_hours=24,
    step_hours=6,
    imputation=ImputationStrategy.FORWARD_FILL,
    max_gap_hours=4,
)
windows = windower.fit_transform(
    df=charts,
    id_col="subject_id",
    time_col="charttime",
    feature_cols=["heart_rate", "spo2", "resp_rate"],
)

# Patient-stratified split preserving mortality rate
result = StratifiedPatientSplitter(
    id_col="subject_id",
    outcome_col="hospital_expire_flag",
    test_size=0.2,
).split(windows)
print(result.summary())
```

# Availability

`clinops` is available on PyPI (`pip install clinops`) and GitHub at
[https://github.com/chaitanyakasaraneni/clinops](https://github.com/chaitanyakasaraneni/clinops)
under the Apache 2.0 license. The library requires Python 3.12+ and has no
dependency on proprietary data sources; the included end-to-end example
(`examples/e2e_example.py`) runs entirely on synthetic data.

# AI Usage Disclosure

The initial draft of this paper was developed with assistance from Claude
(Anthropic), a large language model. The author reviewed, edited, and verified
all content for accuracy and correctness. The software itself — including all
source code, tests, and documentation — was written by the author independently.

# Acknowledgements

The author thanks the MIMIC project team at the MIT Laboratory for Computational
Physiology for maintaining the MIMIC-IV database and making it available to the
research community.

# References