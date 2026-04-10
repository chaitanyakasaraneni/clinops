# Examples

All examples use **synthetic data** and run without MIMIC-IV access. Swap in a real
`MimicTableLoader` path to run against real data — the rest of the pipeline is identical.

---

## Jupyter notebooks

### [01 — Getting started](../examples/notebooks/01_getting_started.ipynb)

A module-by-module walkthrough that demonstrates every `clinops` component in
isolation. Good starting point if you are new to the library.

Topics covered:

| Section | Module | Key class |
|---------|--------|-----------|
| Schema validation | `clinops.ingest` | `FlatFileLoader`, `ClinicalSchema` |
| Outlier clipping | `clinops.preprocess` | `ClinicalOutlierClipper` |
| Unit normalization | `clinops.preprocess` | `UnitNormalizer` |
| ICD harmonization | `clinops.preprocess` | `ICDMapper` |
| Temporal windowing | `clinops.temporal` | `TemporalWindower` |
| Lag/rolling features | `clinops.temporal` | `LagFeatureBuilder` |
| Imputation | `clinops.temporal` | `Imputer` |
| Splitting | `clinops.split` | `PatientSplitter`, `TemporalSplitter`, `StratifiedPatientSplitter` |

---

### [02 — ICU mortality prediction pipeline](../examples/notebooks/02_icu_mortality_pipeline.ipynb)

An end-to-end pipeline that takes raw (synthetic) MIMIC-IV-style tables and produces
ML-ready `X_train / y_train / X_test / y_test` arrays for a 48-hour ICU mortality
prediction task.

Pipeline stages:

```
raw EHR data
  → cohort definition  (ICU stays with ≥48h data)
  → ingest & validate  (FlatFileLoader + ClinicalSchema)
  → clip outliers      (ClinicalOutlierClipper)
  → ICD harmonization  (ICDMapper → chapter-level comorbidity flags)
  → align to admission (CohortAligner → first 48h only)
  → temporal windows   (TemporalWindower, 8h window / 4h stride)
  → patient split      (StratifiedPatientSplitter)
  → imputation         (Imputer fitted on train only)
  → lag features       (LagFeatureBuilder)
  → merge comorbidities
  → scikit-learn ready arrays
```

---

## Script example

### [e2e_example.py](../examples/e2e_example.py)

A single runnable Python script that exercises the full v0.1 pipeline. Useful for
quick smoke-testing after installation.

```bash
pip install clinops
python examples/e2e_example.py
```

---

## Running the notebooks locally

```bash
pip install clinops jupyter
jupyter notebook examples/notebooks/
```

Or with JupyterLab:

```bash
pip install clinops jupyterlab
jupyter lab examples/notebooks/
```

The notebooks install no additional dependencies beyond `clinops` itself.
The optional scikit-learn baseline in notebook 02 requires `pip install scikit-learn`.

---

## Adapting to real MIMIC-IV data

Replace the synthetic data generation block in either notebook with:

```python
from clinops.ingest import MimicTableLoader

tbl      = MimicTableLoader("/data/mimic-iv-2.2")
cohort   = [10000032, 10000980, ...]   # your subject_id list

charts   = tbl.chartevents(subject_ids=cohort)
icustays = tbl.icustays(subject_ids=cohort, with_los_band=True)
adm      = tbl.admissions(subject_ids=cohort)
dx       = tbl.diagnoses_icd(subject_ids=cohort)
```

Everything from the preprocessing step onwards is identical.
