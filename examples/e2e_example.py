"""
clinops v0.1 — End-to-End Example
==================================

Demonstrates the full v0.1 pipeline:
  ingest → temporal windowing → lag features → cohort alignment

Uses synthetic ICU data — no MIMIC-IV access required.

Run: python examples/e2e_example.py
"""

import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from clinops.ingest import ClinicalSchema, ColumnSpec, FlatFileLoader
from clinops.preprocess import ClinicalOutlierClipper, ICDMapper, UnitNormalizer
from clinops.split import PatientSplitter, StratifiedPatientSplitter, TemporalSplitter
from clinops.temporal import (
    CohortAligner,
    ImputationStrategy,
    Imputer,
    LagFeatureBuilder,
    TemporalWindower,
)

# ── Synthetic MIMIC-IV-style data ───────────────────────────────────────────
print("=" * 60)
print("clinops v0.1 — End-to-End Example")
print("=" * 60)

rng = np.random.default_rng(42)
N_PATIENTS = 10
N_HOURS = 48
BASE_TIME = datetime(2023, 6, 1, 8, 0)

rows = []
for pid in range(1, N_PATIENTS + 1):
    for h in range(N_HOURS):
        rows.append({
            "subject_id": pid,
            "hadm_id":    1000 + pid,
            "stay_id":    2000 + pid,
            "charttime":  BASE_TIME + timedelta(hours=h),
            "heart_rate": float(np.clip(rng.normal(72, 8),   40, 200)),
            "spo2":       float(np.clip(rng.normal(97, 1.5), 80, 100)),
            "resp_rate":  float(np.clip(rng.normal(16, 3),    5,  60)),
            "map":        float(np.clip(rng.normal(85, 12),  30, 180)),
        })

chartevents = pd.DataFrame(rows)
print(f"\n[ingest] Synthetic chartevents: {len(chartevents):,} rows, "
      f"{chartevents['subject_id'].nunique()} patients")

# ── FlatFileLoader + schema validation ──────────────────────────────────────

schema = ClinicalSchema(
    name="vitals",
    columns=[
        ColumnSpec("subject_id", nullable=False),
        ColumnSpec("heart_rate", min_value=0, max_value=300),
        ColumnSpec("spo2",       min_value=50, max_value=100),
        ColumnSpec("resp_rate",  min_value=0,  max_value=80),
        ColumnSpec("map",        min_value=0,  max_value=300),
    ]
)

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    chartevents.to_csv(f.name, index=False)
    tmp_path = f.name

print("\n[ingest] Validating with ClinicalSchema...")
loader = FlatFileLoader(tmp_path, schema=schema, id_col="subject_id", strict=False)
loader.load()
print(loader.summary())
os.unlink(tmp_path)

# ── Temporal windowing ───────────────────────────────────────────────────────

print("\n[temporal] Extracting 6-hour windows (step=3h, imputation=FORWARD_FILL)...")
windower = TemporalWindower(
    window_hours=6,
    step_hours=3,
    imputation=ImputationStrategy.FORWARD_FILL,
    min_observations=3,
)
windows = windower.fit_transform(
    df=chartevents,
    id_col="subject_id",
    time_col="charttime",
    feature_cols=["heart_rate", "spo2", "resp_rate", "map"],
)
print(f"[temporal] {len(windows):,} windows — shape: {windows.shape}")
print(windows.head(3).to_string(index=False))

# ── Imputer (train/test split — no leakage) ─────────────────────────────────

split  = int(len(windows) * 0.7)
train_w = windows.iloc[:split].copy()
test_w  = windows.iloc[split:].copy()

# Introduce artificial missingness to show imputation working
rng2 = np.random.default_rng(7)
missing_idx = rng2.choice(test_w.index, size=max(1, len(test_w) // 10), replace=False)
test_w.loc[missing_idx, "heart_rate"] = np.nan

print(f"\n[temporal] Imputing {test_w['heart_rate'].isna().sum()} missing heart_rate values "
      f"(MEAN, fitted on train set)...")
imputer = Imputer(ImputationStrategy.MEAN)
imputer.fit(train_w)
test_imputed = imputer.transform(test_w)
print(f"[temporal] After imputation: {test_imputed['heart_rate'].isna().sum()} missing remaining")

# ── Lag features ─────────────────────────────────────────────────────────────

print("\n[temporal] Building lag (t-1, t-2) and 4-window rolling features...")
enriched = LagFeatureBuilder(
    lags=[1, 2],
    rolling_windows=[4],
    id_col="subject_id",
).fit_transform(windows)

new_cols = [c for c in enriched.columns if c not in windows.columns]
print(f"[temporal] Added {len(new_cols)} features: {new_cols}")

# ── Cohort alignment to anchor event ─────────────────────────────────────────

admissions = pd.DataFrame({
    "subject_id": range(1, N_PATIENTS + 1),
    "intime":     [BASE_TIME + timedelta(hours=4)] * N_PATIENTS,
})

print("\n[temporal] Aligning cohort to ICU admission (0–36h post-admission window)...")
aligned = CohortAligner(
    anchor_col="intime",
    id_col="subject_id",
    time_col="charttime",
    max_hours_before=0,
    max_hours_after=36,
).align(events_df=chartevents, anchor_df=admissions)

print(f"[temporal] {len(aligned):,} events retained | "
      f"hours_from_anchor: [{aligned['hours_from_anchor'].min():.1f}, "
      f"{aligned['hours_from_anchor'].max():.1f}]")

# ── Outlier clipping ─────────────────────────────────────────────────────────

print("\n[preprocess] Clipping physiologically impossible values...")
clipper = ClinicalOutlierClipper(action="clip")
chartevents_clean = clipper.fit_transform(chartevents)
report = clipper.report()
if len(report):
    print(report.to_string(index=False))
else:
    print(
        "[preprocess] No outliers detected in synthetic data \n"
        "(expected — data was clipped at generation)"
    )

# ── Unit normalization ────────────────────────────────────────────────────────

print("\n[preprocess] Demonstrating unit normalization (mixed glucose units)...")
glucose_df = pd.DataFrame({
    "subject_id":   [1, 2, 3, 4],
    "glucose":      [126.0, 5.5, 180.0, 7.2],
    "glucose_unit": ["mg/dL", "mmol/L", "mg/dL", "mmol/L"],
})
print(f"  Before: {glucose_df['glucose'].tolist()} {glucose_df['glucose_unit'].tolist()}")
normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
glucose_norm = normalizer.transform(glucose_df)
print(f"  After:  {[round(v, 1) for v in glucose_norm['glucose'].tolist()]} "
      f"{glucose_norm['glucose_unit'].tolist()}")
print(normalizer.report().to_string(index=False))

# ── ICD mapping ───────────────────────────────────────────────────────────────

print("\n[preprocess] Harmonizing mixed ICD-9/ICD-10 diagnosis codes...")
dx_df = pd.DataFrame({
    "subject_id":  [1, 2, 3, 4, 5],
    "icd_code":    ["41401", "4280", "42731", "I10", "49121"],
    "icd_version": ["9", "9", "9", "10", "9"],
})
mapper = ICDMapper()
dx_harmonized = mapper.harmonize(dx_df, code_col="icd_code", version_col="icd_version")
dx_harmonized["chapter"] = mapper.chapter_series(dx_harmonized["icd_code"])
print(dx_harmonized[["subject_id", "icd_code", "chapter"]].to_string(index=False))
print(f"  Mapper loaded {mapper.n_mappings} ICD-9→10 mappings")

# ── Splitting ─────────────────────────────────────────────────────────────────

print("\n[split] TemporalSplitter — cutoff at 24h mark...")
temporal_splitter = TemporalSplitter(
    cutoff=BASE_TIME + timedelta(hours=24), time_col="charttime"
)
t_result = temporal_splitter.split(chartevents)
print(t_result.summary())

print("\n[split] PatientSplitter — patient-level 80/20 split...")
patient_splitter = PatientSplitter(id_col="subject_id", test_size=0.2, random_state=42)
p_result = patient_splitter.split(chartevents)
print(p_result.summary())
train_ids = set(p_result.train["subject_id"].unique())
test_ids  = set(p_result.test["subject_id"].unique())
print(f"  Train patients: {sorted(train_ids)}")
print(f"  Test patients:  {sorted(test_ids)}")
assert train_ids.isdisjoint(test_ids), "Patient leakage detected!"
print("  ✓ No patient leakage")

print("\n[split] StratifiedPatientSplitter — preserving outcome rate...")
# Add a synthetic binary outcome: patients 1-3 are "high risk"
chartevents_with_outcome = chartevents.copy()
chartevents_with_outcome["mortality"] = (
    chartevents_with_outcome["subject_id"] <= 3
).astype(int)
strat_splitter = StratifiedPatientSplitter(
    id_col="subject_id",
    outcome_col="mortality",
    test_size=0.3,
    random_state=42,
)
s_result = strat_splitter.split(chartevents_with_outcome)
print(s_result.summary())


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅  v0.1 pipeline complete")
print("=" * 60)
print(f"  Raw events:      {len(chartevents):,} rows")
print(f"  Windows:         {len(windows):,}")
print(f"  Enriched shape:  {enriched.shape}")
print(f"  Aligned events:  {len(aligned):,}  (0–36h post-ICU-admission)")
print()
print("Next steps:")
print("  • clinops.ingest.MimicLoader  — swap in real MIMIC-IV data")
print("  • clinops.ingest.FHIRLoader   — load FHIR R4 observations")
print("  • clinops v0.2                — drift detection + GCS/S3 orchestration")
print(f"  Outlier report:  {len(report)} columns checked")
print(f"  ICD codes mapped: {len(dx_harmonized)} dx records harmonized")
print(f"  Temporal split:  train={t_result.train_size:,} / test={t_result.test_size:,} rows")
print(f"  Patient split:   train={p_result.metadata['n_train_patients']} / "
      f"test={p_result.metadata['n_test_patients']} patients")
