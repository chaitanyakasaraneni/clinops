"""
Tests for MimicIIILoader.

Uses minimal uppercase CSV fixtures in a flat tmp directory — no real
MIMIC-III data needed. Fixtures mirror the actual MIMIC-III column names
(uppercase) to verify the normalisation logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from clinops.ingest.mimic_iii import MimicIIILoader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_csvs(tmp_path):
    """Write minimal MIMIC-III-style CSV files (uppercase columns, flat dir)."""

    # CHARTEVENTS — 3 rows, 2 patients, uppercase columns
    (tmp_path / "CHARTEVENTS.csv").write_text(
        "SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME,VALUENUM,VALUEUOM,ERROR\n"
        "1,100,1000,211,2150-01-01 06:00:00,72.0,bpm,\n"
        "1,100,1000,211,2150-01-01 12:00:00,75.0,bpm,\n"
        "2,200,2000,211,2150-01-02 06:00:00,80.0,bpm,\n"
    )

    # LABEVENTS — 2 rows with ref range columns
    (tmp_path / "LABEVENTS.csv").write_text(
        "SUBJECT_ID,HADM_ID,ITEMID,CHARTTIME,VALUE,VALUENUM,VALUEUOM,"
        "REF_RANGE_LOWER,REF_RANGE_UPPER,FLAG\n"
        "1,100,50912,2150-01-01 08:00:00,1.1,1.1,mg/dL,0.5,1.5,\n"
        "2,200,50912,2150-01-02 08:00:00,0.9,0.9,mg/dL,0.5,1.5,\n"
    )

    # ADMISSIONS
    (tmp_path / "ADMISSIONS.csv").write_text(
        "SUBJECT_ID,HADM_ID,ADMITTIME,DISCHTIME,DEATHTIME,ADMISSION_TYPE,"
        "ADMISSION_LOCATION,DISCHARGE_LOCATION,INSURANCE,HOSPITAL_EXPIRE_FLAG\n"
        "1,100,2150-01-01 00:00:00,2150-01-05 00:00:00,,EMERGENCY,"
        "EMERGENCY ROOM,HOME,Medicare,0\n"
        "2,200,2150-01-02 00:00:00,2150-01-07 00:00:00,,ELECTIVE,"
        "PHYSICIAN REFERRAL,HOME,Private,0\n"
    )

    # DIAGNOSES_ICD — ICD-9 only, mix of numeric and alphanumeric codes
    (tmp_path / "DIAGNOSES_ICD.csv").write_text(
        "SUBJECT_ID,HADM_ID,SEQ_NUM,ICD9_CODE\n"
        "1,100,1,4280\n"
        "1,100,2,V053\n"
        "2,200,1,41401\n"
        "2,200,2,5849\n"
    )

    # ICUSTAYS
    (tmp_path / "ICUSTAYS.csv").write_text(
        "SUBJECT_ID,HADM_ID,ICUSTAY_ID,FIRST_CAREUNIT,LAST_CAREUNIT,"
        "INTIME,OUTTIME,LOS\n"
        "1,100,1000,MICU,MICU,2150-01-01 08:00:00,2150-01-05 08:00:00,4.0\n"
        "2,200,2000,SICU,SICU,2150-01-02 08:00:00,2150-01-09 08:00:00,7.0\n"
    )

    # PRESCRIPTIONS
    (tmp_path / "PRESCRIPTIONS.csv").write_text(
        "SUBJECT_ID,HADM_ID,STARTDATE,ENDDATE,DRUG,DRUG_TYPE,"
        "DOSE_VAL_RX,DOSE_UNIT_RX\n"
        "1,100,2150-01-01,2150-01-03,Heparin,MAIN,5000,UNIT\n"
        "1,100,2150-01-01,2150-01-05,Metoprolol,MAIN,25,mg\n"
        "2,200,2150-01-02,2150-01-04,Aspirin,MAIN,81,mg\n"
    )

    # INPUTEVENTS_MV
    (tmp_path / "INPUTEVENTS_MV.csv").write_text(
        "SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,STARTTIME,ENDTIME,"
        "AMOUNT,AMOUNTUOM,RATE,RATEUOM\n"
        "1,100,1000,225158,2150-01-01 08:00:00,2150-01-01 10:00:00,"
        "500,mL,250,mL/hr\n"
        "2,200,2000,225158,2150-01-02 08:00:00,2150-01-02 10:00:00,"
        "250,mL,125,mL/hr\n"
    )

    # INPUTEVENTS_CV
    (tmp_path / "INPUTEVENTS_CV.csv").write_text(
        "SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME,AMOUNT,AMOUNTUOM\n"
        "1,100,1000,30008,2150-01-01 09:00:00,100,mL\n"
    )

    # D_ITEMS
    (tmp_path / "D_ITEMS.csv").write_text(
        "ITEMID,LABEL,ABBREVIATION,DBSOURCE,LINKSTO,CATEGORY,UNITNAME\n"
        "211,Heart Rate,HR,metavision,chartevents,Routine Vital Signs,bpm\n"
        "220277,O2 Saturation,SpO2,metavision,chartevents,Routine Vital Signs,%\n"
    )

    # D_LABITEMS
    (tmp_path / "D_LABITEMS.csv").write_text(
        "ITEMID,LABEL,FLUID,CATEGORY,LOINC_CODE\n"
        "50912,Creatinine,Blood,Chemistry,2160-0\n"
        "50820,pH,Blood,Blood Gas,11558-4\n"
    )

    # PATIENTS
    (tmp_path / "PATIENTS.csv").write_text(
        "SUBJECT_ID,GENDER,DOB,DOD,DOD_HOSP,DOD_SSN,EXPIRE_FLAG\n"
        "1,M,2080-01-01 00:00:00,,,, 0\n"
        "2,F,2070-06-15 00:00:00,2150-01-09 00:00:00,2150-01-09,,1\n"
    )

    return tmp_path


@pytest.fixture()
def mimic3_dir(tmp_path):
    return _write_csvs(tmp_path)


@pytest.fixture()
def loader(mimic3_dir):
    return MimicIIILoader(mimic3_dir)


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------


class TestColumnNormalisation:
    def test_chartevents_columns_lowercase(self, loader):
        df = loader.chartevents()
        assert all(c == c.lower() for c in df.columns)

    def test_admissions_columns_lowercase(self, loader):
        df = loader.admissions()
        assert all(c == c.lower() for c in df.columns)

    def test_diagnoses_columns_lowercase(self, loader):
        df = loader.diagnoses_icd()
        assert all(c == c.lower() for c in df.columns)


# ---------------------------------------------------------------------------
# chartevents
# ---------------------------------------------------------------------------


class TestChartevents:
    def test_loads_all_rows(self, loader):
        assert len(loader.chartevents()) == 3

    def test_filter_by_subject(self, loader):
        df = loader.chartevents(subject_ids=[1])
        assert set(df["subject_id"]) == {1}
        assert len(df) == 2

    def test_filter_by_icustay(self, loader):
        df = loader.chartevents(icustay_ids=[2000])
        assert len(df) == 1
        assert df.iloc[0]["subject_id"] == 2

    def test_filter_by_item(self, loader):
        df = loader.chartevents(item_ids=[211])
        assert len(df) == 3

    def test_filter_by_item_no_match(self, loader):
        assert len(loader.chartevents(item_ids=[99999])) == 0

    def test_filter_by_start_time(self, loader):
        df = loader.chartevents(start_time="2150-01-01 12:00:00")
        assert len(df) == 2

    def test_filter_by_end_time(self, loader):
        df = loader.chartevents(end_time="2150-01-01 06:00:00")
        assert len(df) == 1

    def test_charttime_is_datetime(self, loader):
        df = loader.chartevents()
        assert pd.api.types.is_datetime64_any_dtype(df["charttime"])

    def test_icustay_id_column_present(self, loader):
        # MIMIC-III uses icustay_id not stay_id
        assert "icustay_id" in loader.chartevents().columns

    def test_required_columns_present(self, loader):
        df = loader.chartevents()
        for col in ["subject_id", "hadm_id", "icustay_id", "itemid", "charttime", "valuenum"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# labevents
# ---------------------------------------------------------------------------


class TestLabevents:
    def test_loads_all_rows(self, loader):
        assert len(loader.labevents()) == 2

    def test_ref_range_dropped_by_default(self, loader):
        df = loader.labevents()
        assert "ref_range_lower" not in df.columns
        assert "ref_range_upper" not in df.columns

    def test_ref_range_retained_when_requested(self, loader):
        df = loader.labevents(with_ref_range=True)
        assert "ref_range_lower" in df.columns
        assert "ref_range_upper" in df.columns

    def test_filter_by_subject(self, loader):
        df = loader.labevents(subject_ids=[1])
        assert len(df) == 1

    def test_charttime_is_datetime(self, loader):
        assert pd.api.types.is_datetime64_any_dtype(loader.labevents()["charttime"])


# ---------------------------------------------------------------------------
# admissions
# ---------------------------------------------------------------------------


class TestAdmissions:
    def test_loads_all_rows(self, loader):
        assert len(loader.admissions()) == 2

    def test_filter_by_subject(self, loader):
        df = loader.admissions(subject_ids=[2])
        assert len(df) == 1

    def test_admittime_is_datetime(self, loader):
        assert pd.api.types.is_datetime64_any_dtype(loader.admissions()["admittime"])


# ---------------------------------------------------------------------------
# diagnoses_icd
# ---------------------------------------------------------------------------


class TestDiagnosesIcd:
    def test_loads_all_rows(self, loader):
        assert len(loader.diagnoses_icd()) == 4

    def test_icd9_code_column_present(self, loader):
        assert "icd9_code" in loader.diagnoses_icd().columns

    def test_synthetic_icd_version_added(self, loader):
        df = loader.diagnoses_icd()
        assert "icd_version" in df.columns
        assert (df["icd_version"] == 9).all()

    def test_primary_only(self, loader):
        df = loader.diagnoses_icd(primary_only=True)
        assert len(df) == 2
        assert set(df["seq_num"]) == {1}

    def test_filter_by_subject(self, loader):
        df = loader.diagnoses_icd(subject_ids=[1])
        assert len(df) == 2

    def test_filter_by_icd9_code(self, loader):
        df = loader.diagnoses_icd(icd9_codes=["4280"])
        assert len(df) == 1

    def test_filter_by_icd9_code_case_insensitive(self, loader):
        df = loader.diagnoses_icd(icd9_codes=["v053"])
        assert len(df) == 1
        assert df.iloc[0]["icd9_code"] == "V053"

    def test_filter_by_alphanumeric_icd9_code(self, loader):
        # V053 would be read as int by pandas if not cast — this catches regression
        df = loader.diagnoses_icd(icd9_codes=["V053"])
        assert len(df) == 1
        assert df.iloc[0]["icd9_code"] == "V053"

    def test_combined_subject_and_primary_only(self, loader):
        df = loader.diagnoses_icd(subject_ids=[1], primary_only=True)
        assert len(df) == 1
        assert df.iloc[0]["icd9_code"] == "4280"


# ---------------------------------------------------------------------------
# icustays
# ---------------------------------------------------------------------------


class TestIcustays:
    def test_loads_all_rows(self, loader):
        assert len(loader.icustays()) == 2

    def test_filter_by_subject(self, loader):
        assert len(loader.icustays(subject_ids=[1])) == 1

    def test_filter_by_icustay(self, loader):
        df = loader.icustays(icustay_ids=[2000])
        assert df.iloc[0]["icustay_id"] == 2000

    def test_intime_is_datetime(self, loader):
        assert pd.api.types.is_datetime64_any_dtype(loader.icustays()["intime"])

    def test_los_is_numeric(self, loader):
        assert pd.api.types.is_float_dtype(loader.icustays()["los"])

    def test_icustay_id_not_stay_id(self, loader):
        df = loader.icustays()
        assert "icustay_id" in df.columns
        assert "stay_id" not in df.columns


# ---------------------------------------------------------------------------
# prescriptions
# ---------------------------------------------------------------------------


class TestPrescriptions:
    def test_loads_all_rows(self, loader):
        assert len(loader.prescriptions()) == 3

    def test_filter_by_subject(self, loader):
        assert len(loader.prescriptions(subject_ids=[2])) == 1

    def test_filter_by_drug_name(self, loader):
        df = loader.prescriptions(drugs=["Heparin"])
        assert len(df) == 1
        assert df.iloc[0]["drug"] == "Heparin"

    def test_filter_by_drug_case_insensitive(self, loader):
        assert len(loader.prescriptions(drugs=["heparin"])) == 1

    def test_startdate_is_datetime(self, loader):
        assert pd.api.types.is_datetime64_any_dtype(loader.prescriptions()["startdate"])


# ---------------------------------------------------------------------------
# inputevents
# ---------------------------------------------------------------------------


class TestInputevents:
    def test_mv_loads(self, loader):
        df = loader.inputevents(source="mv")
        assert len(df) == 2

    def test_cv_loads(self, loader):
        df = loader.inputevents(source="cv")
        assert len(df) == 1

    def test_both_merged(self, loader):
        df = loader.inputevents(source="both")
        assert len(df) == 3

    def test_filter_by_subject_mv(self, loader):
        df = loader.inputevents(subject_ids=[1], source="mv")
        assert len(df) == 1

    def test_invalid_source_raises(self, loader):
        with pytest.raises(ValueError, match="source must be"):
            loader.inputevents(source="invalid")

    def test_both_adds_event_time_column(self, loader):
        # source="both" must add a unified event_time column so callers
        # can sort/window across MV and CV rows without knowing which
        # native time column to use.
        df = loader.inputevents(source="both")
        assert "event_time" in df.columns

    def test_both_adds_source_column(self, loader):
        # source="both" must tag each row with its origin system so callers
        # can filter or stratify by MetaVision vs CareVue after merging.
        df = loader.inputevents(source="both")
        assert "source" in df.columns
        assert set(df["source"].unique()) == {"mv", "cv"}

    def test_both_mv_event_time_equals_starttime(self, loader):
        # For MetaVision rows, event_time must mirror starttime.
        df = loader.inputevents(source="both")
        mv_rows = df[df["source"] == "mv"]
        assert (mv_rows["event_time"] == mv_rows["starttime"]).all()

    def test_both_cv_event_time_equals_charttime(self, loader):
        # For CareVue rows, event_time must mirror charttime.
        df = loader.inputevents(source="both")
        cv_rows = df[df["source"] == "cv"]
        assert (cv_rows["event_time"] == cv_rows["charttime"]).all()

    def test_single_source_no_event_time_column(self, loader):
        # event_time/source normalisation only applies when source="both".
        # Single-table loads should not add extra columns.
        df_mv = loader.inputevents(source="mv")
        assert "event_time" not in df_mv.columns
        assert "source" not in df_mv.columns
        df_cv = loader.inputevents(source="cv")
        assert "event_time" not in df_cv.columns
        assert "source" not in df_cv.columns


# ---------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------


class TestDictionaries:
    def test_d_items_loads(self, loader):
        df = loader.d_items()
        assert len(df) == 2
        assert "label" in df.columns

    def test_d_labitems_loads(self, loader):
        df = loader.d_labitems()
        assert len(df) == 2
        assert "loinc_code" in df.columns

    def test_patients_loads(self, loader):
        df = loader.patients()
        assert len(df) == 2
        assert "subject_id" in df.columns


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_path_raises(self, tmp_path):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="does not exist"):
            MimicIIILoader(tmp_path / "nonexistent")

    def test_missing_table_raises(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        loader = MimicIIILoader(tmp_path)
        with pytest.raises(FileNotFoundError, match="CHARTEVENTS"):
            loader.chartevents()

    def test_strict_validation_raises_on_missing_cols(self, tmp_path):
        # Write a chartevents CSV missing required 'valuenum'
        (tmp_path / "CHARTEVENTS.csv").write_text(
            "SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME\n1,100,1000,211,2150-01-01 06:00:00\n"
        )
        loader = MimicIIILoader(tmp_path, strict_validation=True)
        from clinops.ingest.schema import SchemaValidationError

        with pytest.raises(SchemaValidationError):
            loader.chartevents()

    def test_non_strict_warns_on_missing_cols(self, tmp_path):
        (tmp_path / "CHARTEVENTS.csv").write_text(
            "SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME\n1,100,1000,211,2150-01-01 06:00:00\n"
        )
        loader = MimicIIILoader(tmp_path, strict_validation=False)
        df = loader.chartevents()  # should not raise
        assert len(df) == 1
