"""
clinops v0.1 — End-to-End Example
==================================

Demonstrates the full v0.1 pipeline:
  ingest → preprocess → temporal windowing → lag features → cohort alignment → split

Uses synthetic ICU data — no MIMIC-IV access required.

Run: python examples/e2e_example.py
"""

import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from clinops.ingest import ClinicalSchema, ColumnSpec, FlatFileLoader, MimicTableLoader
from clinops.preprocess import ClinicalOutlierClipper, ICDMapper, UnitNormalizer
from clinops.preprocess.units import UNIT_CONVERSIONS
from clinops.split import PatientSplitter, StratifiedPatientSplitter, TemporalSplitter
from clinops.temporal import (
    CohortAligner,
    ImputationStrategy,
    Imputer,
    LagFeatureBuilder,
    TemporalWindower,
)

print("=" * 60)
print("clinops v0.1 — End-to-End Example")
print("=" * 60)

# ── Synthetic MIMIC-IV-style data ───────────────────────────────────────────
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

# ── MimicTableLoader (demo with tmp directory) ───────────────────────────────
# In production, point this at your MIMIC-IV root:
#   tbl = MimicTableLoader("/data/mimic-iv-2.2")
#
# Here we write the synthetic data to a tmp directory to show the API.
print("\n[ingest] MimicTableLoader — pre-built schemas, no manual ColumnSpec required...")

with tempfile.TemporaryDirectory() as tmp_root:
    import pathlib
    root = pathlib.Path(tmp_root)
    (root / "icu").mkdir()
    (root / "hosp").mkdir()

    # Write minimal CSVs that satisfy MimicTableLoader schemas
    chartevents.rename(columns={
        "heart_rate": "valuenum", "spo2": "valueuom"
    }).assign(itemid=220045, valueuom="bpm")[
        ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum", "valueuom"]
    ].to_csv(root / "icu" / "chartevents.csv", index=False)

    pd.DataFrame({
        "subject_id":     range(1, N_PATIENTS + 1),
        "hadm_id":        range(1001, 1001 + N_PATIENTS),
        "admittime":      [BASE_TIME] * N_PATIENTS,
        "dischtime":      [BASE_TIME + timedelta(days=5)] * N_PATIENTS,
        "deathtime":      [None] * N_PATIENTS,
        "admission_type": ["EMERGENCY"] * N_PATIENTS,
        "admission_location":  ["EMERGENCY ROOM"] * N_PATIENTS,
        "discharge_location":  ["HOME"] * N_PATIENTS,
        "insurance":           ["Medicare"] * N_PATIENTS,
        "hospital_expire_flag": [0] * N_PATIENTS,
    }).to_csv(root / "hosp" / "admissions.csv", index=False)

    dx_rows = []
    for pid in range(1, N_PATIENTS + 1):
        dx_rows.append({"subject_id": pid, "hadm_id": 1000 + pid,
                        "seq_num": 1, "icd_code": "I509", "icd_version": 10})
        dx_rows.append({"subject_id": pid, "hadm_id": 1000 + pid,
                        "seq_num": 2, "icd_code": "E119", "icd_version": 10})
    pd.DataFrame(dx_rows).to_csv(root / "hosp" / "diagnoses_icd.csv", index=False)

    pd.DataFrame({
        "subject_id":   range(1, N_PATIENTS + 1),
        "hadm_id":      range(1001, 1001 + N_PATIENTS),
        "stay_id":      range(2001, 2001 + N_PATIENTS),
        "first_careunit": ["MICU"] * N_PATIENTS,
        "last_careunit":  ["MICU"] * N_PATIENTS,
        "intime":   [BASE_TIME] * N_PATIENTS,
        "outtime":  [BASE_TIME + timedelta(days=4)] * N_PATIENTS,
        "los":      [float(rng.integers(1, 8))] * N_PATIENTS,
    }).to_csv(root / "icu" / "icustays.csv", index=False)

    pd.DataFrame({
        "subject_id": range(1, N_PATIENTS + 1),
        "hadm_id":    range(1001, 1001 + N_PATIENTS),
        "itemid":     [50912] * N_PATIENTS,
        "charttime":  [BASE_TIME] * N_PATIENTS,
        "valuenum":   rng.uniform(0.6, 1.4, N_PATIENTS).round(2),
        "valueuom":   ["mg/dL"] * N_PATIENTS,
        "ref_range_lower": [0.5] * N_PATIENTS,
        "ref_range_upper": [1.5] * N_PATIENTS,
    }).to_csv(root / "hosp" / "labevents.csv", index=False)

    tbl = MimicTableLoader(root)

    adm = tbl.admissions()
    print(f"[ingest] admissions:     {len(adm):,} rows — "
          f"columns: {list(adm.columns)}")

    dx = tbl.diagnoses_icd(primary_only=True)
    print(f"[ingest] diagnoses_icd:  {len(dx):,} rows (primary only) — "
          f"ICD versions present: {sorted(dx['icd_version'].unique())}")

    stays = tbl.icustays(with_los_band=True)
    print(f"[ingest] icustays:       {len(stays):,} rows — "
          f"los_band counts:\n{stays['los_band'].value_counts().to_string()}")

    labs = tbl.labevents(with_ref_range=False)
    print(f"[ingest] labevents:      {len(labs):,} rows "
          f"(ref_range columns dropped by default)")

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

# ── ClinicalOutlierClipper ───────────────────────────────────────────────────
print("\n[preprocess] Clipping physiologically impossible values...")
clipper = ClinicalOutlierClipper(action="clip")
clipped = clipper.fit_transform(chartevents)
report = clipper.report()
print(f"[preprocess] Outlier report ({len(report)} columns checked):")
print(report.to_string(index=False))

# ── UnitNormalizer ───────────────────────────────────────────────────────────
print("\n[preprocess] Normalizing mixed glucose units (mg/dL + mmol/L → mg/dL)...")
glucose_rows = []
for pid in range(1, N_PATIENTS + 1):
    # half patients have glucose in mg/dL, half in mmol/L
    if pid <= N_PATIENTS // 2:
        glucose_rows.append({
            "subject_id": pid,
            "glucose":      float(rng.uniform(80, 140)),
            "glucose_unit": "mg/dL",
        })
    else:
        glucose_rows.append({
            "subject_id": pid,
            "glucose":      float(rng.uniform(4.4, 7.8)),   # mmol/L range
            "glucose_unit": "mmol/L",
        })
glucose_df = pd.DataFrame(glucose_rows)

print("[preprocess] Before normalization:")
print(glucose_df.to_string(index=False))

normalizer = UnitNormalizer(column_unit_map={"glucose": "glucose_unit"})
normalized = normalizer.transform(glucose_df)

print("[preprocess] After normalization (all mg/dL):")
print(normalized.to_string(index=False))
norm_report = normalizer.report()
print(f"[preprocess] Conversions applied:\n{norm_report.to_string(index=False)}")

# ── ICDMapper ────────────────────────────────────────────────────────────────
print("\n[preprocess] Harmonizing mixed ICD-9 / ICD-10 codes to ICD-10...")
icd_df = pd.DataFrame({
    "subject_id":   [1, 2, 3, 4, 5],
    "icd_code":     ["I509", "4280",  "E119", "25000", "N179"],
    "icd_version":  [10,     9,       10,     9,       10],
})
print("[preprocess] Before harmonization:")
print(icd_df.to_string(index=False))

mapper = ICDMapper()
harmonized = mapper.harmonize(icd_df, code_col="icd_code", version_col="icd_version")
harmonized["chapter"] = mapper.chapter_series(harmonized["icd_code"])
print("[preprocess] After harmonization:")
print(harmonized.to_string(index=False))
print(f"[preprocess] ICDMapper loaded {mapper.n_mappings:,} curated mappings")

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
split   = int(len(windows) * 0.7)
train_w = windows.iloc[:split].copy()
test_w  = windows.iloc[split:].copy()

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

# ── Cohort alignment ─────────────────────────────────────────────────────────
admissions_df = pd.DataFrame({
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
).align(events_df=chartevents, anchor_df=admissions_df)

print(f"[temporal] {len(aligned):,} events retained | "
      f"hours_from_anchor: [{aligned['hours_from_anchor'].min():.1f}, "
      f"{aligned['hours_from_anchor'].max():.1f}]")

# ── TemporalSplitter ─────────────────────────────────────────────────────────
print("\n[split] Temporal split — no future data leakage...")
cutoff = BASE_TIME + timedelta(hours=int(N_HOURS * 0.8))
t_result = TemporalSplitter(
    cutoff=cutoff.isoformat(),
    time_col="charttime",
).split(chartevents)
print(t_result.summary())

# ── PatientSplitter ──────────────────────────────────────────────────────────
print("\n[split] Patient-level split — no patient appears in both sets...")
p_result = PatientSplitter(
    id_col="subject_id",
    test_size=0.2,
    random_state=42,
).split(chartevents)
print(p_result.summary())
train_pids = set(p_result.train["subject_id"].unique())
test_pids  = set(p_result.test["subject_id"].unique())
assert not train_pids & test_pids, "Patient leakage detected!"
print(f"[split] Train patients: {sorted(train_pids)}")
print(f"[split] Test  patients: {sorted(test_pids)}")
print("[split] ✓ No patient leakage between train and test")

# ── StratifiedPatientSplitter ────────────────────────────────────────────────
print("\n[split] Stratified patient split — preserves outcome rate...")
# Add a synthetic binary outcome: patients 1–3 are "high risk"
outcome_map = {pid: int(pid <= 3) for pid in range(1, N_PATIENTS + 1)}
chartevents_with_outcome = chartevents.copy()
chartevents_with_outcome["hospital_expire_flag"] = (
    chartevents_with_outcome["subject_id"].map(outcome_map)
)

s_result = StratifiedPatientSplitter(
    id_col="subject_id",
    outcome_col="hospital_expire_flag",
    test_size=0.3,
    random_state=42,
).split(chartevents_with_outcome)
print(s_result.summary())

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅  v0.1 pipeline complete")
print("=" * 60)
print(f"  Raw events:           {len(chartevents):,} rows")
print(f"  Clipped events:       {len(clipped):,} rows")
print(f"  Outlier report cols:  {len(report)}")
print(f"  ICD codes mapped:     {len(harmonized)}")
print(f"  Windows:              {len(windows):,}")
print(f"  Enriched shape:       {enriched.shape}")
print(f"  Aligned events:       {len(aligned):,}  (0–36h post-ICU-admission)")
print(f"  Temporal split  →  train: {len(t_result.train):,}  "
      f"test: {len(t_result.test):,}")
print(f"  Patient split   →  train: {len(p_result.train):,}  "
      f"test: {len(p_result.test):,}")
print(f"  Stratified split→  train: {len(s_result.train):,}  "
      f"test: {len(s_result.test):,}")
print()
print("Next steps:")
print("  • MimicTableLoader('/data/mimic-iv-2.2') — swap in real MIMIC-IV data")
print("  • clinops.ingest.FHIRLoader              — load FHIR R4 observations")
print("  • clinops v0.2                           — drift detection + GCS/S3 orchestration")
