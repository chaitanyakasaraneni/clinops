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
MIMIC-III [@johnson2016mimic], the eICU Collaborative Research Database
[@pollard2018eicu], FHIR R4 [@hl7fhir], and arbitrary flat clinical exports, and is designed to prevent
the class of silent data errors — patient leakage, unit heterogeneity, ICD version
discontinuities, and physiologically impossible values — that commonly affect
clinical ML workflows.

# Statement of Need

Building a clinical ML model requires solving a set of data engineering problems
that are largely orthogonal to the scientific question being asked. Research
groups working with the same critical care databases repeatedly re-implement the
same extraction and cleaning steps, which costs time and makes results hard to
reproduce and compare [@johnson2018mimic]. General-purpose ML libraries do not
address the clinical-specific failure modes that distinguish healthcare data from
general tabular data, so each project has to guard against them itself.

**Patient leakage** inflates performance metrics in clinical ML. When
`sklearn.train_test_split` [@sklearn] is applied to admission- or record-level
data, a patient with several admissions can appear in both the training and test
sets, so the model can exploit patient-specific patterns rather than generalising
across patients. @saeb2017need show that record-wise cross-validation, which lets
one subject's records fall on both sides of the split, substantially overestimates
prediction accuracy compared with subject-wise cross-validation, which matches how
a clinical model is used on new patients. More broadly, @kapoor2023leakage identify leakage, including
non-independence between training and test data, as a widespread cause of
irreproducible results in ML-based science.

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

`clinops` is organised into four core modules for building datasets, plus two
for running pipelines in production:

**`clinops.ingest`** provides loaders for MIMIC-IV (tested with v2.0–v2.2),
MIMIC-III, eICU-CRD, FHIR R4, and flat CSV/Parquet files with configurable schema
validation. Large event tables are read in chunks with filters applied to each
chunk, so only matching rows are held in memory. The `MimicTableLoader`
class exposes the five tables most commonly needed in ICU cohort studies —
`chartevents`, `labevents`, `admissions`, `diagnoses_icd`, and `icustays` — with
pre-built schemas, automatic datetime parsing, and convenience flags such as
`primary_only` (restricts diagnoses to the principal diagnosis per admission) and
`with_los_band` (adds a categorical length-of-stay column).

**`clinops.preprocess`** provides three preprocessing transformers.
`ClinicalOutlierClipper` clips, nulls, or flags values outside plausibility bounds
for 37 vital signs and laboratory measurements. The bounds are deliberately wide
limits, chosen by the authors rather than taken from a published source, that
separate physically impossible values and recording errors from extreme but real
measurements; all of them are documented and can be overridden. `UnitNormalizer` converts between 26 registered
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
stratification on a binary or multiclass outcome column, sampling each class at the
test fraction so both splits approximately keep the population class proportions
(a class with a single patient can only be placed in one split) — critical for imbalanced clinical endpoints such as in-hospital
mortality (typically 5–15% in ICU cohorts).

**`clinops.monitor`** checks incoming batches for data quality problems and
distribution drift against a reference dataset, and **`clinops.orchestrate`**
stores pipeline artifacts on Google Cloud Storage or Amazon S3 and deploys
pipelines as AWS Step Functions state machines.

All transformers follow the scikit-learn `fit`/`transform`/`fit_transform`
interface where applicable. The library is fully typed and passes `mypy --strict`.
Continuous integration runs the test suite (over 450 tests, about 94% line
coverage) on Python 3.12 to 3.14 with pandas 2 and 3, executes every docstring
example, and runs the getting-started walkthrough and the example below on
synthetic MIMIC-IV data.

# Example Usage

The example below builds a leakage-safe mortality-prediction dataset from the first
48 hours of each ICU stay in MIMIC-IV.

```python
from clinops.ingest import MimicTableLoader
from clinops.preprocess import ClinicalOutlierClipper, ICDMapper
from clinops.split import StratifiedPatientSplitter
from clinops.temporal import CohortAligner, TemporalWindower

VITALS = {220045: "heart_rate", 220277: "spo2", 220210: "resp_rate"}

# Load tables with pre-built schemas; chartevents is filtered chunk by chunk
tbl = MimicTableLoader("/data/mimic-iv-2.2", chunk_size=1_000_000)
charts = tbl.chartevents(item_ids=list(VITALS))
stays, adm = tbl.icustays(), tbl.admissions()

# Harmonise ICD-9 and ICD-10 diagnoses and add chapter labels
mapper = ICDMapper()
dx = mapper.harmonize(
    tbl.diagnoses_icd(primary_only=True), code_col="icd_code", version_col="icd_version"
)
dx["chapter"] = mapper.chapter_series(dx["icd_code"])

# One column per vital sign, then clip physiologically impossible values
vitals = (
    charts.assign(vital=charts["itemid"].map(VITALS))
    .pivot_table(index=["stay_id", "charttime"], columns="vital", values="valuenum")
    .reset_index()
)
vitals = ClinicalOutlierClipper(action="clip").fit_transform(vitals)

# First 48 hours of each ICU stay, in 24-hour windows every 6 hours
aligned = CohortAligner(anchor_col="intime", id_col="stay_id", max_hours_after=48).align(
    vitals, stays[["stay_id", "intime"]]
)
windows = TemporalWindower(window_hours=24, step_hours=6).fit_transform(
    df=aligned, id_col="stay_id", time_col="charttime", feature_cols=list(VITALS.values())
)

# Attach in-hospital mortality, then split by patient, stratified on the outcome
outcome = stays[["stay_id", "subject_id", "hadm_id"]].merge(
    adm[["hadm_id", "hospital_expire_flag"]], on="hadm_id"
)
result = StratifiedPatientSplitter(
    id_col="subject_id", outcome_col="hospital_expire_flag", test_size=0.2
).split(windows.merge(outcome, on="stay_id"))
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