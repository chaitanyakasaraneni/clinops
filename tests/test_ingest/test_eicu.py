"""
Tests for the eICU-CRD adapter (clinops.ingest.eicu).

All fixtures are SYNTHETIC — eICU is credentialed PhysioNet data and must never
be committed. The synthetic tables reproduce the real eICU schema (column names,
offset-based time, free-text labs, string age) so the loader logic is exercised
end-to-end without any real data.

Edge cases covered (per the adapter spec):
- ``age = '> 89'``  → parsed to 90, patient retained as adult
- a patient with two stays (``uniquepid`` repeats) → grouped split keeps them together
- a stay with missing GCS → graceful NaN, no crash
- an unmapped ``labname`` → logged warning, row ignored (not an error)
- a stay above the missingness threshold → excluded from the cohort
- the leakage guard → label-only variables never reach the feature matrix
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinops.ingest import EicuCohortConfig, EicuLoader, EicuTableLoader
from clinops.ingest.eicu import (
    FEATURE_COLUMNS,
    LEAKAGE_FORBIDDEN,
    ORGAN_LABELS,
    parse_eicu_age,
)
from clinops.ingest.mappings.eicu_lab_map import canonical_lab_features, map_labname
from clinops.ingest.schema import LeakageError
from clinops.split import GroupedPatientSplitter

# ---------------------------------------------------------------------------
# Small cohort config so sequences are exercised without large fixtures.
# ---------------------------------------------------------------------------

TEST_CFG = EicuCohortConfig(
    min_age=18,
    min_los_hours=4,
    max_icu_hours=8,
    observation_hours=2,
    prediction_hours=1,
    baseline_hours=3,
    missingness_threshold=0.50,
    first_stay_only=True,
)


def _vital_rows(stay: int, *, low_map: bool = False) -> pd.DataFrame:
    """Dense ~30-min vitalPeriodic rows over 8h for one stay."""
    offsets = list(range(0, 8 * 60, 30))
    n = len(offsets)
    rng = np.random.default_rng(stay)
    return pd.DataFrame(
        {
            "vitalperiodicid": [stay * 1000 + i for i in range(n)],
            "patientunitstayid": stay,
            "observationoffset": offsets,
            "temperature": rng.normal(37, 0.3, n).round(1),
            "sao2": (rng.normal(85, 2, n) if low_map else rng.normal(97, 1, n)).round(),
            "heartrate": rng.normal(85, 5, n).round(),
            "respiration": rng.normal(18, 2, n).round(),
            "cvp": np.nan,
            "etco2": np.nan,
            "systemicsystolic": rng.normal(120, 8, n).round(),
            "systemicdiastolic": rng.normal(70, 5, n).round(),
            "systemicmean": (rng.normal(60, 3, n) if low_map else rng.normal(85, 5, n)).round(),
            "pasystolic": np.nan,
            "padiastolic": np.nan,
            "pamean": np.nan,
            "st1": np.nan,
            "st2": np.nan,
            "st3": np.nan,
            "icp": np.nan,
        }
    )


def _lab_rows(stay: int, *, include_unmapped: bool = False) -> pd.DataFrame:
    """Long-format lab rows mapping to canonical features."""
    base = [
        ("creatinine", 1.0),
        ("total bilirubin", 0.8),
        ("lactate", 1.2),
        ("pH", 7.38),
        ("paO2", 95.0),
        ("paCO2", 40.0),
        ("FiO2", 40.0),
        ("platelets x 1000", 220.0),
        ("WBC x 1000", 8.0),
    ]
    if include_unmapped:
        base.append(("troponin - I", 0.01))  # not in EICU_LAB_MAP
    recs = []
    for h in range(0, 8):  # one set per hour
        for name, val in base:
            recs.append(
                {
                    "labid": stay * 10000 + h * 100 + len(recs),
                    "patientunitstayid": stay,
                    "labresultoffset": h * 60 + 5,
                    "labtypeid": 1,
                    "labname": name,
                    "labresult": val,
                    "labresulttext": str(val),
                    "labmeasurenamesystem": "",
                    "labmeasurenameinterface": "",
                    "labresultrevisedoffset": h * 60 + 5,
                }
            )
    return pd.DataFrame(recs)


def _gcs_rows(stay: int, value: int = 15) -> pd.DataFrame:
    recs = []
    for h in range(0, 8):
        recs.append(
            {
                "nursingchartid": stay * 100 + h,
                "patientunitstayid": stay,
                "nursingchartoffset": h * 60 + 10,
                "nursingchartentryoffset": h * 60 + 10,
                "nursingchartcelltypecat": "Scores",
                "nursingchartcelltypevallabel": "Glasgow coma score",
                "nursingchartcelltypevalname": "GCS Total",
                "nursingchartvalue": str(value),
            }
        )
    return pd.DataFrame(recs)


@pytest.fixture
def eicu_dir(tmp_path: Path) -> Path:
    """Write a synthetic eICU-CRD directory and return its path."""
    d = tmp_path / "eicu-crd"
    d.mkdir()

    # patient.csv ----------------------------------------------------------
    patient = pd.DataFrame(
        [
            # P1 normal adult, single stay
            _pt(101, "P1", "67", "Male", "MICU", los_min=480, visit=1),
            # P2 two stays — first_stay_only must keep 201, drop 202
            _pt(201, "P2", "55", "Female", "SICU", los_min=480, visit=1),
            _pt(202, "P2", "55", "Female", "SICU", los_min=480, visit=2),
            # P3 age '> 89' → 90, adult, retained
            _pt(301, "P3", "> 89", "Male", "Cardiac ICU", los_min=480, visit=1),
            # P4 missing GCS, otherwise fine
            _pt(401, "P4", "72", "Female", "Neuro ICU", los_min=480, visit=1),
            # P5 high-missingness stay → excluded by missingness filter
            _pt(501, "P5", "60", "Male", "MICU", los_min=480, visit=1),
            # P6 pediatric → excluded by adult filter
            _pt(601, "P6", "10", "Female", "MICU", los_min=480, visit=1),
            # P7 short LOS → excluded by min_los filter
            _pt(701, "P7", "45", "Male", "MICU", los_min=120, visit=1),
            # P8 DNR within 6h → excluded
            _pt(801, "P8", "80", "Male", "MICU", los_min=480, visit=1),
        ]
    )
    patient.to_csv(d / "patient.csv", index=False)

    # vitalPeriodic.csv ----------------------------------------------------
    vitals = pd.concat(
        [_vital_rows(s) for s in [101, 201, 301, 401, 601, 701, 801]]
        # P5 (501): only 2 of 8 hours have vitals → high missingness
        + [_vital_rows(501).iloc[:4]],
        ignore_index=True,
    )
    vitals.to_csv(d / "vitalPeriodic.csv", index=False)

    # vitalAperiodic.csv (NIBP fallback) -----------------------------------
    aperiodic = pd.DataFrame(
        {
            "vitalaperiodicid": [1, 2],
            "patientunitstayid": [101, 301],
            "observationoffset": [35, 35],
            "noninvasivesystolic": [118, 119],
            "noninvasivediastolic": [68, 69],
            "noninvasivemean": [84, 85],
            "paop": [np.nan, np.nan],
            "cardiacoutput": [np.nan, np.nan],
            "cardiacinput": [np.nan, np.nan],
            "svr": [np.nan, np.nan],
            "svri": [np.nan, np.nan],
            "pvr": [np.nan, np.nan],
            "pvri": [np.nan, np.nan],
        }
    )
    aperiodic.to_csv(d / "vitalAperiodic.csv", index=False)

    # lab.csv (P1 carries an unmapped labname) -----------------------------
    labs = pd.concat(
        [_lab_rows(101, include_unmapped=True)]
        + [_lab_rows(s) for s in [201, 301, 401, 501, 601, 701, 801]],
        ignore_index=True,
    )
    labs.to_csv(d / "lab.csv", index=False)

    # nurseCharting.csv (P4=401 deliberately has NO GCS rows) --------------
    gcs = pd.concat(
        [_gcs_rows(s) for s in [101, 201, 301, 501, 601, 701, 801]],
        ignore_index=True,
    )
    gcs.to_csv(d / "nurseCharting.csv", index=False)

    # medication.csv (vasopressor for 101) ---------------------------------
    med = pd.DataFrame(
        {
            "medicationid": [1],
            "patientunitstayid": [101],
            "drugorderoffset": [60],
            "drugstartoffset": [60],
            "drugivadmixture": [""],
            "drugordercancelled": ["No"],
            "drugname": ["NOREPINEPHRINE 4 MG/250 ML NS INFUSION"],
            "drughiclseqno": [1],
            "dosage": ["5 mcg/min"],
            "routeadmin": ["IV"],
            "frequency": [""],
            "loadingdose": [""],
            "prn": ["No"],
            "drugstopoffset": [300],
            "gtc": [0],
        }
    )
    med.to_csv(d / "medication.csv", index=False)

    # respiratoryCare.csv (mech vent for 301) ------------------------------
    resp = pd.DataFrame(
        {
            "respcareid": [1],
            "patientunitstayid": [301],
            "respcarestatusoffset": [30],
            "currenthistoryseqnum": [1],
            "airwaytype": ["ETT"],
            "ventstartoffset": [30],
            "ventendoffset": [240],
        }
    )
    resp.to_csv(d / "respiratoryCare.csv", index=False)

    # treatment.csv (RRT for 401) ------------------------------------------
    treat = pd.DataFrame(
        {
            "treatmentid": [1],
            "patientunitstayid": [401],
            "treatmentoffset": [120],
            "treatmentstring": ["renal|dialysis|hemodialysis"],
            "activeupondischarge": ["true"],
        }
    )
    treat.to_csv(d / "treatment.csv", index=False)

    # diagnosis.csv (normal diagnoses; eICU has no DNR here) ---------------
    diag = pd.DataFrame(
        {
            "diagnosisid": [1, 2],
            "patientunitstayid": [101, 301],
            "activeupondischarge": ["true", "true"],
            "diagnosisoffset": [200, 200],
            "diagnosisstring": ["sepsis", "acute respiratory failure"],
            "icd9code": ["995.91", "518.81"],
            "diagnosispriority": ["Primary", "Primary"],
        }
    )
    diag.to_csv(d / "diagnosis.csv", index=False)

    # carePlanGeneral.csv (DNR within 6h for 801 — eICU's real DNR location)
    cpg = pd.DataFrame(
        {
            "cplgeneralid": [1, 2],
            "patientunitstayid": [801, 101],
            "activeupondischarge": ["true", "true"],
            "cplitemoffset": [120, 300],
            "cplgroup": ["Care Limitation", "Care Limitation"],
            "cplitemvalue": ["Do not resuscitate", "Full therapy"],
        }
    )
    cpg.to_csv(d / "carePlanGeneral.csv", index=False)

    # apachePatientResult.csv ----------------------------------------------
    apache = pd.DataFrame(
        {
            "apachepatientresultsid": [1, 2],
            "patientunitstayid": [101, 301],
            "physicianspeciality": ["", ""],
            "physicianinterventioncategory": ["", ""],
            "acutephysiologyscore": [40, 55],
            "apachescore": [62, 78],
            "apacheversion": ["IVa", "IVa"],
            "predictedicumortality": [0.1, 0.2],
            "actualicumortality": ["ALIVE", "ALIVE"],
            "predictediculos": [3.0, 4.0],
            "actualiculos": [3.5, 4.5],
            "predictedhospitalmortality": [0.1, 0.2],
            "actualhospitalmortality": ["ALIVE", "ALIVE"],
            "predictedhospitallos": [5.0, 6.0],
            "actualhospitallos": [5.5, 6.5],
            "preopmi": [0, 0],
            "preopcardiaccath": [0, 0],
            "ptcawithin24h": [0, 0],
            "unabridgedunitlos": [3.5, 4.5],
            "unabridgedhosplos": [5.5, 6.5],
            "actualventdays": [1, 2],
            "predventdays": [1, 2],
            "unabridgedactualventdays": [1, 2],
        }
    )
    apache.to_csv(d / "apachePatientResult.csv", index=False)

    return d


def _pt(
    stay: int,
    uniquepid: str,
    age: str,
    gender: str,
    unittype: str,
    *,
    los_min: int,
    visit: int,
) -> dict:
    """One synthetic patient.csv row (only the columns the loader reads matter)."""
    return {
        "patientunitstayid": stay,
        "patienthealthsystemstayid": int(uniquepid[1:]) * 10,
        "gender": gender,
        "age": age,
        "ethnicity": "Caucasian",
        "hospitalid": 100 + (stay % 3),
        "wardid": 1,
        "apacheadmissiondx": "Sepsis",
        "admissionheight": 170.0,
        "hospitaladmittime24": "00:00:00",
        "hospitaladmitoffset": -60,
        "hospitaladmitsource": "ED",
        "hospitaldischargeyear": 2015,
        "hospitaldischargetime24": "00:00:00",
        "hospitaldischargeoffset": los_min + 600,
        "hospitaldischargelocation": "Home",
        "hospitaldischargestatus": "Alive",
        "unittype": unittype,
        "unitadmittime24": "12:00:00",
        "unitadmitsource": "ED",
        "unitvisitnumber": visit,
        "unitstaytype": "admit",
        "admissionweight": 80.0,
        "dischargeweight": 80.0,
        "unitdischargetime24": "20:00:00",
        "unitdischargeoffset": los_min,
        "unitdischargelocation": "Floor",
        "unitdischargestatus": "Alive",
        "uniquepid": uniquepid,
    }


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_parse_eicu_age_over_89():
    assert parse_eicu_age("> 89") == 90.0
    assert parse_eicu_age("67") == 67.0
    assert np.isnan(parse_eicu_age(""))
    assert np.isnan(parse_eicu_age(None))


def test_lab_map_canonical_and_unmapped():
    assert map_labname("WBC x 1000") == "wbc"
    assert map_labname("  Total Bilirubin ") == "bilirubin"
    assert map_labname("paO2") == "pao2"
    assert map_labname("troponin - I") is None
    assert "gcs_total" not in canonical_lab_features()  # GCS comes from nurseCharting


def test_feature_schema_invariants():
    assert len(FEATURE_COLUMNS) == 22
    assert not (LEAKAGE_FORBIDDEN & set(FEATURE_COLUMNS))
    assert len(ORGAN_LABELS) == 5


# ---------------------------------------------------------------------------
# EicuLoader (low-level)
# ---------------------------------------------------------------------------


def test_loader_reads_tables(eicu_dir: Path):
    loader = EicuLoader(eicu_dir)
    pt = loader.load_patient()
    assert len(pt) == 9
    assert {"uniquepid", "patientunitstayid", "age"} <= set(pt.columns)

    vp = loader.load_vital_periodic(patient_unit_stay_ids=[101])
    assert (vp["patientunitstayid"] == 101).all()

    lab = loader.load_lab(patient_unit_stay_ids=[101], labnames=["creatinine"])
    assert set(lab["labname"].str.lower()) == {"creatinine"}


def test_loader_chunked_filter_pushdown(eicu_dir: Path):
    # Tiny chunk_size forces the chunked path; stay filter must still work.
    loader = EicuLoader(eicu_dir, chunk_size=7)
    vp = loader.load_vital_periodic(patient_unit_stay_ids=[301])
    assert len(vp) > 0
    assert set(vp["patientunitstayid"].unique()) == {301}


# ---------------------------------------------------------------------------
# EicuTableLoader (high-level)
# ---------------------------------------------------------------------------


def test_build_cohort_applies_filters(eicu_dir: Path):
    loader = EicuTableLoader(eicu_dir, config=TEST_CFG)
    cohort = loader.build_cohort()
    stays = set(cohort["patientunitstayid"])

    # Retained: P1, P2(first stay 201), P3(>89), P4, P5(missingness handled later)
    assert {101, 201, 301, 401} <= stays
    # P3 age '> 89' parsed to 90
    assert cohort.loc[cohort["patientunitstayid"] == 301, "age"].iloc[0] == 90.0
    # Excluded
    assert 202 not in stays  # second stay of P2
    assert 601 not in stays  # pediatric
    assert 701 not in stays  # short LOS
    assert 801 not in stays  # DNR within 6h
    # APACHE joined
    assert cohort.loc[cohort["patientunitstayid"] == 101, "apache_ii"].iloc[0] == 62


def test_build_feature_matrix_shape_and_no_leakage(eicu_dir: Path):
    loader = EicuTableLoader(eicu_dir, config=TEST_CFG)
    fm = loader.build_feature_matrix()
    assert list(fm.columns) == ["patientunitstayid", "hour", *FEATURE_COLUMNS]
    # Leakage-forbidden columns are absent
    assert not (LEAKAGE_FORBIDDEN & set(fm.columns))
    # Treatment flags are present and 0/1
    for col in ["vasopressor_flag", "mechanical_ventilation_flag", "rrt_flag"]:
        assert set(fm[col].unique()) <= {0, 1}
    # Vasopressor flag fired for stay 101 (norepinephrine, offsets 60–300 min)
    s101 = fm[fm["patientunitstayid"] == 101]
    assert s101["vasopressor_flag"].max() == 1


def test_leakage_guard_raises(eicu_dir: Path, monkeypatch):
    # Force a label-only variable into the feature set; the guard must refuse.
    import clinops.ingest.eicu as eicu_mod

    monkeypatch.setattr(
        eicu_mod, "FEATURE_COLUMNS", [*eicu_mod.FEATURE_COLUMNS, "baseline_creatinine"]
    )
    loader = EicuTableLoader(eicu_dir, config=TEST_CFG)
    with pytest.raises(LeakageError, match="baseline_creatinine"):
        loader.build_feature_matrix()


def test_missing_gcs_is_graceful(eicu_dir: Path):
    loader = EicuTableLoader(eicu_dir, config=TEST_CFG)
    fm = loader.build_feature_matrix()
    s401 = fm[fm["patientunitstayid"] == 401]
    # P4 has no nurseCharting GCS rows → gcs_total all NaN, no crash
    assert s401["gcs_total"].isna().all()


def test_high_missingness_stay_excluded(eicu_dir: Path):
    loader = EicuTableLoader(eicu_dir, config=TEST_CFG)
    fm = loader.build_feature_matrix()
    # P5 (501) only had vitals for the first 2h of an 8h stay → excluded
    assert 501 not in set(fm["patientunitstayid"])


def test_unmapped_labname_warns_not_errors(eicu_dir: Path, caplog):
    loader = EicuTableLoader(eicu_dir, config=TEST_CFG)
    with caplog.at_level("WARNING"):
        loader.build_feature_matrix()
    assert any("unmapped labname" in r.message for r in caplog.records)


def test_build_labels(eicu_dir: Path):
    loader = EicuTableLoader(eicu_dir, config=TEST_CFG)
    labels = loader.build_labels()
    assert list(labels.columns) == ["patientunitstayid", "hour", *ORGAN_LABELS]
    for organ in ORGAN_LABELS:
        assert set(labels[organ].unique()) <= {0, 1}


def test_build_sequences_shapes(eicu_dir: Path):
    loader = EicuTableLoader(eicu_dir, config=TEST_CFG)
    x_arr, y_arr = loader.build_sequences()
    assert x_arr.ndim == 3
    assert x_arr.shape[1] == TEST_CFG.observation_hours
    assert x_arr.shape[2] == len(FEATURE_COLUMNS)
    assert y_arr.shape[1] == len(ORGAN_LABELS)
    assert x_arr.shape[0] == y_arr.shape[0] > 0


# ---------------------------------------------------------------------------
# GroupedPatientSplitter — uniquepid grouping
# ---------------------------------------------------------------------------


def test_grouped_split_no_patient_leak():
    # Two stays per patient; split must keep each patient wholly in one fold.
    rows = []
    for p in range(30):
        for stay in range(2):
            rows.append({"uniquepid": f"P{p}", "patientunitstayid": p * 10 + stay})
    df = pd.DataFrame(rows)

    splitter = GroupedPatientSplitter(
        group_col="uniquepid",
        stay_col="patientunitstayid",
        val_size=0.2,
        test_size=0.2,
        random_state=42,
    )
    res = splitter.split(df)
    g = "uniquepid"
    assert not (set(res.train[g]) & set(res.val[g]))
    assert not (set(res.train[g]) & set(res.test[g]))
    assert not (set(res.val[g]) & set(res.test[g]))
    # every stay accounted for exactly once
    total = len(res.train) + len(res.val) + len(res.test)
    assert total == len(df)


def test_grouped_split_is_deterministic():
    df = pd.DataFrame({"uniquepid": [f"P{i // 2}" for i in range(40)]})
    a = GroupedPatientSplitter(random_state=7).split(df)
    b = GroupedPatientSplitter(random_state=7).split(df)
    assert set(a.test["uniquepid"]) == set(b.test["uniquepid"])


@pytest.mark.parametrize("n_groups", [1, 2])
def test_grouped_split_raises_for_tiny_cohort(n_groups):
    # A three-way split is impossible with <3 groups; must raise, not return an
    # empty train fold.
    df = pd.DataFrame({"uniquepid": [f"P{i}" for i in range(n_groups)]})
    with pytest.raises(ValueError, match="at least 3 groups"):
        GroupedPatientSplitter().split(df)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_window_larger_than_horizon():
    # observation_hours + prediction_hours must fit within max_icu_hours.
    with pytest.raises(ValueError, match="max_icu_hours"):
        EicuCohortConfig(max_icu_hours=8, observation_hours=24, prediction_hours=6)


def test_config_rejects_nonpositive_and_out_of_range():
    with pytest.raises(ValueError, match="must be positive"):
        EicuCohortConfig(min_los_hours=0)
    with pytest.raises(ValueError, match="missingness_threshold"):
        EicuCohortConfig(missingness_threshold=1.5)


# ---------------------------------------------------------------------------
# Regression tests for review fixes
# ---------------------------------------------------------------------------


def test_dnr_filter_respects_care_limitation_group(tmp_path: Path):
    # Only DNR rows in the 'Care Limitation' group, within the window, exclude.
    d = tmp_path / "eicu-crd"
    d.mkdir()
    pd.DataFrame(
        {
            "cplgeneralid": [1, 2, 3],
            "patientunitstayid": [1, 2, 3],
            "cplitemoffset": [60, 60, 5000],
            "cplgroup": ["Care Limitation", "Care Plan General", "Care Limitation"],
            "cplitemvalue": ["Do not resuscitate", "Do not resuscitate", "Do not resuscitate"],
        }
    ).to_csv(d / "carePlanGeneral.csv", index=False)
    loader = EicuTableLoader(d, config=TEST_CFG)
    dnr = loader._dnr_stays(within_minutes=360)
    assert dnr == {1}  # stay 2: wrong group; stay 3: outside the 6h window


def test_load_large_columns_on_parquet(tmp_path: Path):
    # `columns=` must work for Parquet-backed tables (no CSV reader on .parquet).
    pytest.importorskip("pyarrow")
    d = tmp_path / "eicu-crd"
    d.mkdir()
    _vital_rows(101).to_parquet(d / "vitalPeriodic.parquet", index=False)
    loader = EicuLoader(d)
    out = loader.load_vital_periodic(patient_unit_stay_ids=[101], columns=["heartrate", "sao2"])
    assert {"patientunitstayid", "heartrate", "sao2"} <= set(out.columns)
    assert (out["patientunitstayid"] == 101).all()


def test_build_sequences_does_not_mutate_config(eicu_dir: Path):
    cfg = EicuCohortConfig(
        min_age=18,
        min_los_hours=4,
        max_icu_hours=8,
        observation_hours=2,
        prediction_hours=1,
        baseline_hours=3,
        missingness_threshold=0.50,
    )
    loader = EicuTableLoader(eicu_dir, config=cfg)
    before = (cfg.max_icu_hours, cfg.observation_hours, cfg.prediction_hours)
    loader.build_sequences(observation_hours=3, prediction_hours=2, max_icu_hours=6)
    after = (cfg.max_icu_hours, cfg.observation_hours, cfg.prediction_hours)
    assert before == after == (8, 2, 1)
