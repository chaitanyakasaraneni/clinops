# Preprocess

`clinops.preprocess` handles the gap between raw ingested data and ML-ready features: physiological outlier clipping, clinical unit normalization, and ICD code harmonization.

---

## ClinicalOutlierClipper

Standard statistical outlier methods (z-score, IQR) are wrong for clinical data — a heart rate of 180 in a patient with SVT is clinically meaningful and should not be removed. `ClinicalOutlierClipper` instead uses fixed plausibility bounds to catch values that are impossible regardless of patient state.

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

Built-in bounds cover 37 variables (12 vitals and 25 labs).

These are plausibility limits, not reference ranges: wide bounds outside which a value is physically impossible or almost certainly a recording, unit, or device error. They were chosen by the clinops authors to be deliberately permissive and are not taken from a single published source, so review them for your setting and override any of them with `bounds=`, `extra_bounds=`, or `add_bounds()`.

<!-- Generated from VITAL_BOUNDS and LAB_BOUNDS; tests/test_preprocess/test_bounds_doc.py keeps it in sync. -->
| Column | Group | Low | High | Unit | Note |
|---|---|---|---|---|---|
| `heart_rate` | vital | 0 | 300 | bpm | Asystole to extreme tachycardia |
| `resp_rate` | vital | 0 | 80 | breaths/min | Apnoea to extreme tachypnoea |
| `spo2` | vital | 50 | 100 | % | Readings below 50% treated as artefact |
| `sbp` | vital | 0 | 300 | mmHg | Systolic blood pressure |
| `dbp` | vital | 0 | 200 | mmHg | Diastolic blood pressure |
| `map` | vital | 0 | 250 | mmHg | Mean arterial pressure |
| `temperature` | vital | 25 | 45 | °C | Severe hypothermia to hyperthermia |
| `temperature_f` | vital | 77 | 113 | °F | °F equivalent of 25–45°C |
| `weight` | vital | 0.5 | 500 | kg | Neonate to extreme obesity |
| `height` | vital | 30 | 250 | cm | Infant to extreme height |
| `bmi` | vital | 10 | 100 | kg/m² | Severe underweight to extreme obesity |
| `gcs` | vital | 3 | 15 | score | Glasgow Coma Scale: 3 (worst) to 15 (normal) |
| `glucose` | lab | 0 | 2000 | mg/dL | Hypoglycaemia to extreme hyperglycaemia |
| `glucose_mmol` | lab | 0 | 111 | mmol/L | mmol/L equivalent |
| `sodium` | lab | 90 | 200 | mEq/L | Values outside 90–200 treated as errors |
| `potassium` | lab | 1 | 10 | mEq/L | Severe hypo- to hyperkalaemia |
| `chloride` | lab | 60 | 150 | mEq/L |  |
| `bicarbonate` | lab | 0 | 60 | mEq/L |  |
| `bun` | lab | 0 | 300 | mg/dL | Blood urea nitrogen |
| `creatinine` | lab | 0 | 50 | mg/dL |  |
| `calcium` | lab | 1 | 20 | mg/dL |  |
| `magnesium` | lab | 0 | 10 | mg/dL |  |
| `phosphate` | lab | 0 | 30 | mg/dL |  |
| `wbc` | lab | 0 | 500 | K/uL | White blood cell count |
| `hgb` | lab | 0 | 25 | g/dL | Haemoglobin |
| `hct` | lab | 0 | 100 | % | Haematocrit |
| `platelets` | lab | 0 | 3000 | K/uL |  |
| `alt` | lab | 0 | 30000 | U/L | Alanine aminotransferase |
| `ast` | lab | 0 | 30000 | U/L | Aspartate aminotransferase |
| `bilirubin` | lab | 0 | 100 | mg/dL |  |
| `inr` | lab | 0 | 20 | ratio |  |
| `pt` | lab | 0 | 200 | seconds |  |
| `ptt` | lab | 0 | 400 | seconds |  |
| `ph` | lab | 6.5 | 7.9 | pH | Arterial pH; values outside 6.5–7.9 treated as errors |
| `pao2` | lab | 0 | 700 | mmHg | PaO2 on 100% O2 |
| `paco2` | lab | 0 | 150 | mmHg |  |
| `lactate` | lab | 0 | 30 | mmol/L |  |

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

26 built-in conversions (each direction counts once). Run
`UnitNormalizer.available_conversions()` to list them.

| Measurement | Units |
|---|---|
| Glucose, BUN, calcium, phosphate | mg/dL ↔ mmol/L |
| Creatinine, bilirubin | mg/dL ↔ µmol/L |
| Haemoglobin | g/dL ↔ mmol/L |
| Temperature | °C ↔ °F, °C ↔ K |
| Weight | kg ↔ lb, kg ↔ g |
| Height | cm ↔ in, cm ↔ m |

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
