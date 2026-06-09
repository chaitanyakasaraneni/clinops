# Ingest

`clinops.ingest` provides loaders for MIMIC-IV, MIMIC-III, FHIR R4, and flat CSV/Parquet files with schema validation built in.

---

## MimicTableLoader

The fastest way to work with MIMIC-IV. Pre-built schemas for the five most-used tables — no `ColumnSpec` definitions required.

```python
from clinops.ingest import MimicTableLoader

tbl = MimicTableLoader("/data/mimic-iv-2.2")
```

### Available tables

```python
# ICU chartevents — charttime parsed as datetime automatically
charts = tbl.chartevents(subject_ids=[10000032, 10000980])

# Lab results
labs = tbl.labevents(subject_ids=[10000032], with_ref_range=True)

# Hospital admissions — includes hospital_expire_flag mortality outcome
adm = tbl.admissions(subject_ids=[10000032])

# ICD-9/10 diagnoses — primary_only keeps only seq_num == 1
dx = tbl.diagnoses_icd(subject_ids=[10000032], primary_only=True)

# ICU stays — with_los_band adds a <1d / 1-3d / 3-7d / >7d length-of-stay column
stays = tbl.icustays(subject_ids=[10000032], with_los_band=True)
```

### Audit a new MIMIC download

Check row counts, column counts, and null rates without loading full tables:

```python
tbl.summary()
#        table  rows_sampled  columns  null_rate_pct
#  chartevents         10000       23           8.41
#    labevents         10000       12           4.17
#   admissions         10000       15           6.02
# diagnoses_icd        10000        5           0.00
#     icustays         10000        8           2.31
```

---

## MimicLoader

For custom filtering and chunk-based loading of large tables.

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

Large tables (`chartevents`, `labevents`) are loaded in chunks when `chunk_size` is set to avoid memory issues:

```python
loader = MimicLoader("/data/mimic-iv-2.2", chunk_size=100_000)
charts = loader.chartevents()   # streams in 100k-row chunks internally
```

---

## MimicIIILoader

Equivalent loader for MIMIC-III (ICD-9 codes, slightly different schema).

```python
from clinops.ingest import MimicIIILoader

loader = MimicIIILoader("/data/mimic-iii-1.4")
charts = loader.chartevents(subject_ids=[10006])
```

---

## FHIRLoader

Load FHIR R4 resources from a JSON Bundle or NDJSON export.

```python
from clinops.ingest import FHIRLoader

loader   = FHIRLoader("/data/fhir_export")
obs      = loader.observations(category="vital-signs")
patients = loader.patients()
```

!!! note
    Requires the `fhir` extra: `pip install clinops[fhir]`

---

## eICU-CRD (`EicuLoader` / `EicuTableLoader`)

The [eICU Collaborative Research Database](https://eicu-crd.mit.edu/) is a
multi-centre ICU dataset (208 US hospitals, ~200k unit stays). It differs from
MIMIC in ways that break naïve loaders, all handled here:

| eICU quirk | How clinops handles it |
|---|---|
| Time is **minutes from ICU admission** (`observationoffset`), not datetimes | All temporal logic uses `offset // 60` hour bins |
| `age` is a **string**; `> 89` for de-identified elders | `parse_eicu_age` → 90.0 |
| `uniquepid` (patient) ≠ `patientunitstayid` (stay) | `GroupedPatientSplitter` groups by patient |
| Labs are **long-format free text** (`labname`) | Harmonised via `eicu_lab_map`; unmapped names warn, never crash |
| GCS lives in `nurseCharting` (`Scores`), not `lab` | Extracted and merged automatically |
| DNR status lives in `carePlanGeneral`, not `diagnosis` | Cohort excludes DNR-within-6h from the correct table |
| `vitalPeriodic` / `nurseCharting` are 6–11 GB | Chunked reads; files never fully held in memory |

### Low-level table access

```python
from clinops.ingest import EicuLoader

loader = EicuLoader("/data/eicu-crd", chunk_size=100_000)
pt     = loader.load_patient()                                  # small, read whole
vitals = loader.load_vital_periodic(patient_unit_stay_ids=[141168])  # chunked
labs   = loader.load_lab(patient_unit_stay_ids=[141168],
                         labnames=["creatinine", "lactate"])    # chunked + filtered
```

### ICHI-compatible feature/label/sequence builder

```python
from clinops.ingest import EicuTableLoader, EicuCohortConfig

cfg    = EicuCohortConfig(min_los_hours=48, max_icu_hours=72,
                          observation_hours=24, prediction_hours=6,
                          hospital_ids=None)        # restrict sites for subgroup analysis
loader = EicuTableLoader("/data/eicu-crd", config=cfg)

cohort   = loader.build_cohort()          # one row per included stay (+ apache_ii)
features = loader.build_feature_matrix()  # hourly 22-feature matrix (leakage-guarded)
labels   = loader.build_labels()          # 5 organ-deterioration labels per stay-hour
X, y     = loader.build_sequences()       # (n, 24, 22) and (n, 5)
```

The feature matrix is checked against a programmatic exclusion list before it is
returned — if a label-defining variable (e.g. a patient's baseline creatinine or
the PaO2/FiO2 ratio) ever reaches the feature set, a `LeakageError` is raised.

---

## FlatFileLoader

Load and validate any flat CSV or Parquet file with a custom schema.

```python
from clinops.ingest import FlatFileLoader, ClinicalSchema, ColumnSpec

schema = ClinicalSchema(
    name="vitals",
    columns=[
        ColumnSpec("subject_id", nullable=False),
        ColumnSpec("heart_rate", min_value=0,  max_value=300),
        ColumnSpec("spo2",       min_value=50, max_value=100),
    ]
)
df = FlatFileLoader("vitals.csv", schema=schema).load()
```

`SchemaValidationError` is raised if any `nullable=False` column contains nulls, or if values fall outside the declared bounds.

---

## Supported file formats

| Format | Extension |
|---|---|
| CSV | `.csv` |
| Compressed CSV | `.csv.gz` |
| Parquet | `.parquet` |
