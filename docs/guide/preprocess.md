# Preprocess

`clinops.preprocess` handles the gap between raw ingested data and ML-ready features: physiological outlier clipping, clinical unit normalization, and ICD code harmonization.

---

## ClinicalOutlierClipper

Standard statistical outlier methods (z-score, IQR) are wrong for clinical data — a heart rate of 180 in a patient with SVT is clinically meaningful and should not be removed. `ClinicalOutlierClipper` uses published physiological bounds to remove values that are impossible regardless of patient state.

```python
from clinops.preprocess import ClinicalOutlierClipper

clipper = ClinicalOutlierClipper(action="clip")
clean_df = clipper.fit_transform(vitals_df)
```

### Actions

| Action | Behaviour |
|---|---|
| `"clip"` | Replace out-of-range values with the bound value |
| `"null"` | Replace out-of-range values with `NaN` |
| `"flag"` | Add a `{col}_outlier` boolean column, leave values in place |

### Outlier report

```python
print(clipper.report())
#    column  low_outliers  high_outliers  pct_outliers  bound_low  bound_high
#  heart_rate             0              3         0.012          0         300
#        spo2             1              0         0.004         50         100
```

### Built-in bounds

Built-in bounds cover 20 vitals and labs, including:

| Column | Low | High |
|---|---|---|
| `heart_rate` | 0 | 300 |
| `spo2` | 50 | 100 |
| `sbp` | 40 | 300 |
| `dbp` | 20 | 200 |
| `resp_rate` | 0 | 80 |
| `glucose` | 1 | 1500 |
| `creatinine` | 0 | 50 |
| `ph` | 6.5 | 8.0 |
| `wbc` | 0 | 200 |
| `temperature` | 25 | 45 |

### Adding custom bounds

```python
from clinops.preprocess import ClinicalOutlierClipper

clipper = ClinicalOutlierClipper(action="null")
clipper.add_bounds("lactate", low=0.0, high=30.0)
clipper.add_bounds("inr",     low=0.5, high=20.0)
clean_df = clipper.fit_transform(df)
```

---

## UnitNormalizer

Multi-site studies routinely mix mg/dL and mmol/L for the same lab, or °F and °C for temperature. `UnitNormalizer` detects non-standard units via a companion unit column and converts in-place.

```python
from clinops.preprocess import UnitNormalizer

# df has "glucose" and "glucose_unit" columns (mixed "mg/dL" / "mmol/L")
normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
df = normalizer.transform(df)
# All glucose values now in mg/dL; glucose_unit column updated

print(normalizer.report())
#   column from_unit to_unit  n_converted
#  glucose    mmol/L   mg/dL          142
```

### Registered conversions

30 built-in conversions covering:

- **Glucose / creatinine / bilirubin / haemoglobin / calcium** — mmol/L ↔ mg/dL
- **Temperature** — °F ↔ °C
- **Weight** — lb ↔ kg
- **Height** — in ↔ cm

---

## ICDMapper

MIMIC-III uses ICD-9, MIMIC-IV mixes both versions, and many real-world datasets span the October 2015 transition. `ICDMapper` converts ICD-9-CM codes to ICD-10-CM and adds chapter-level groupings for ML features.

```python
from clinops.preprocess import ICDMapper

mapper = ICDMapper()

# Harmonize a mixed-version DataFrame to ICD-10
df = mapper.harmonize(df, code_col="icd_code", version_col="icd_version")

# Add chapter-level grouping (e.g. "Diseases of the circulatory system")
df["chapter"] = mapper.chapter_series(df["icd_code"])

# Map a single code
mapper.map_code("4280")   # → "I509"
```

Ships with ~60 curated high-frequency mappings. Load the full CMS GEM file (~72,000 mappings) with:

```python
mapper = ICDMapper.from_gem_file("/data/2018_I9gem.txt")
```
