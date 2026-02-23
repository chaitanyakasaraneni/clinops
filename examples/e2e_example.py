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
