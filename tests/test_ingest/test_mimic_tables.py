"""
Tests for MimicTableLoader.

Uses the same tmp-directory fixtures as test_ingest.py so no real MIMIC data
is needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from clinops.ingest.mimic_tables import MimicTableLoader

# ---------------------------------------------------------------------------
# Fixtures — minimal CSV files that satisfy each schema
# ---------------------------------------------------------------------------


def _write_csvs(tmp_path):
    hosp = tmp_path / "hosp"
    icu = tmp_path / "icu"
    hosp.mkdir()
    icu.mkdir()

    # chartevents
    (icu / "chartevents.csv").write_text(
        "subject_id,hadm_id,stay_id,itemid,charttime,valuenum,valueuom\n"
        "1,100,1000,220045,2150-01-01 06:00:00,72.0,bpm\n"
        "1,100,1000,220045,2150-01-01 12:00:00,75.0,bpm\n"
        "2,200,2000,220045,2150-01-02 06:00:00,80.0,bpm\n"
    )

    # labevents
    (hosp / "labevents.csv").write_text(
        "subject_id,hadm_id,itemid,charttime,valuenum,valueuom,"
        "ref_range_lower,ref_range_upper\n"
        "1,100,50912,2150-01-01 08:00:00,1.1,mg/dL,0.5,1.5\n"
        "2,200,50912,2150-01-02 08:00:00,0.9,mg/dL,0.5,1.5\n"
    )

    # admissions
    (hosp / "admissions.csv").write_text(
        "subject_id,hadm_id,admittime,dischtime,deathtime,"
        "admission_type,admission_location,discharge_location,"
        "insurance,hospital_expire_flag\n"
        "1,100,2150-01-01 00:00:00,2150-01-05 00:00:00,,EMERGENCY,"
        "EMERGENCY ROOM,HOME,Medicare,0\n"
        "2,200,2150-01-02 00:00:00,2150-01-07 00:00:00,,ELECTIVE,"
        "PHYSICIAN REFERRAL,HOME,Private,0\n"
    )

    # diagnoses_icd
    (hosp / "diagnoses_icd.csv").write_text(
        "subject_id,hadm_id,seq_num,icd_code,icd_version\n"
        "1,100,1,I509,10\n"
        "1,100,2,E119,10\n"
        "2,200,1,4280,9\n"
        "2,200,2,25000,9\n"
    )

    # icustays
    (icu / "icustays.csv").write_text(
        "subject_id,hadm_id,stay_id,first_careunit,last_careunit,"
        "intime,outtime,los\n"
        "1,100,1000,MICU,MICU,2150-01-01 08:00:00,2150-01-05 08:00:00,4.0\n"
        "2,200,2000,SICU,SICU,2150-01-02 08:00:00,2150-01-09 08:00:00,7.0\n"
    )

    return tmp_path


@pytest.fixture()
def mimic_dir(tmp_path):
    return _write_csvs(tmp_path)


@pytest.fixture()
def loader(mimic_dir):
    return MimicTableLoader(mimic_dir)


# ---------------------------------------------------------------------------
# chartevents
# ---------------------------------------------------------------------------


class TestChartevents:
    def test_loads_all_rows(self, loader):
        df = loader.chartevents()
        assert len(df) == 3

    def test_filter_by_subject(self, loader):
        df = loader.chartevents(subject_ids=[1])
        assert set(df["subject_id"]) == {1}
        assert len(df) == 2

    def test_required_columns_present(self, loader):
        df = loader.chartevents()
        for col in ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"]:
            assert col in df.columns

    def test_charttime_parsed_as_datetime(self, loader):
        df = loader.chartevents()
        assert pd.api.types.is_datetime64_any_dtype(df["charttime"])


# ---------------------------------------------------------------------------
# labevents
# ---------------------------------------------------------------------------


class TestLabevents:
    def test_loads_rows(self, loader):
        df = loader.labevents()
        assert len(df) == 2

    def test_ref_range_dropped_by_default(self, loader):
        df = loader.labevents()
        assert "ref_range_lower" not in df.columns
        assert "ref_range_upper" not in df.columns

    def test_ref_range_retained_when_requested(self, loader):
        df = loader.labevents(with_ref_range=True)
        assert "ref_range_lower" in df.columns
        assert "ref_range_upper" in df.columns

    def test_filter_by_subject(self, loader):
        df = loader.labevents(subject_ids=[2])
        assert len(df) == 1
        assert df.iloc[0]["subject_id"] == 2


# ---------------------------------------------------------------------------
# admissions
# ---------------------------------------------------------------------------


class TestAdmissions:
    def test_loads_rows(self, loader):
        df = loader.admissions()
        assert len(df) == 2

    def test_required_columns(self, loader):
        df = loader.admissions()
        for col in ["subject_id", "hadm_id", "admittime", "admission_type"]:
            assert col in df.columns

    def test_filter_by_hadm(self, loader):
        df = loader.admissions(hadm_ids=[100])
        assert len(df) == 1
        assert df.iloc[0]["hadm_id"] == 100


# ---------------------------------------------------------------------------
# diagnoses_icd
# ---------------------------------------------------------------------------


class TestDiagnosesIcd:
    def test_loads_all_rows(self, loader):
        df = loader.diagnoses_icd()
        assert len(df) == 4

    def test_primary_only(self, loader):
        df = loader.diagnoses_icd(primary_only=True)
        assert len(df) == 2
        assert set(df["seq_num"]) == {1}

    def test_filter_by_subject(self, loader):
        df = loader.diagnoses_icd(subject_ids=[1])
        assert set(df["subject_id"]) == {1}
        assert len(df) == 2

    def test_icd_version_column_present(self, loader):
        df = loader.diagnoses_icd()
        assert "icd_version" in df.columns
        assert set(df["icd_version"]).issubset({9, 10, "9", "10"})

    def test_mixed_icd_versions(self, loader):
        df = loader.diagnoses_icd()
        versions = set(df["icd_version"].astype(str))
        assert "9" in versions and "10" in versions

    # --- line 317: hadm_ids filter ---

    def test_filter_by_hadm(self, loader):
        df = loader.diagnoses_icd(hadm_ids=[100])
        assert len(df) == 2
        assert set(df["hadm_id"]) == {100}

    def test_filter_by_hadm_no_match(self, loader):
        df = loader.diagnoses_icd(hadm_ids=[999])
        assert len(df) == 0

    # --- line 407: parquet branch in _load_extra_table ---

    def test_loads_from_parquet(self, mimic_dir):
        import pandas as pd
        csv_path = mimic_dir / "hosp" / "diagnoses_icd.csv"
        df_orig = pd.read_csv(csv_path)
        csv_path.unlink()
        df_orig.to_parquet(mimic_dir / "hosp" / "diagnoses_icd.parquet", index=False)
        ldr = MimicTableLoader(mimic_dir)
        df = ldr.diagnoses_icd()
        assert len(df) == 4
        assert "icd_code" in df.columns

    # --- line 421: FileNotFoundError in _resolve_extra_path ---

    def test_missing_file_raises(self, mimic_dir):
        (mimic_dir / "hosp" / "diagnoses_icd.csv").unlink()
        ldr = MimicTableLoader(mimic_dir)
        with pytest.raises(FileNotFoundError, match="diagnoses_icd"):
            ldr.diagnoses_icd()


# ---------------------------------------------------------------------------
# icustays
# ---------------------------------------------------------------------------


class TestIcustays:
    def test_loads_rows(self, loader):
        df = loader.icustays()
        assert len(df) == 2

    def test_los_band_not_added_by_default(self, loader):
        df = loader.icustays()
        assert "los_band" not in df.columns

    def test_los_band_added_when_requested(self, loader):
        df = loader.icustays(with_los_band=True)
        assert "los_band" in df.columns

    def test_los_band_values(self, loader):
        df = loader.icustays(with_los_band=True)
        # patient 1: los=4.0 → "3-7d", patient 2: los=7.0 → "3-7d"
        assert set(df["los_band"].astype(str)).issubset({"<1d", "1-3d", "3-7d", ">7d"})

    def test_filter_by_stay_id(self, loader):
        df = loader.icustays(stay_ids=[1000])
        assert len(df) == 1
        assert df.iloc[0]["stay_id"] == 1000


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_returns_dataframe(self, loader):
        result = loader.summary()
        assert isinstance(result, pd.DataFrame)
        assert "table" in result.columns
        assert "rows_sampled" in result.columns
        assert "null_rate_pct" in result.columns

    def test_summary_covers_all_tables(self, loader):
        result = loader.summary()
        assert set(result["table"]) == {
            "chartevents",
            "labevents",
            "admissions",
            "diagnoses_icd",
            "icustays",
        }

    # --- lines 385-386: FileNotFoundError branch in summary() ---

    def test_missing_tables_appear_with_zero_rows(self, tmp_path):
        # Only chartevents present; all other tables absent → hits except branch
        (tmp_path / "hosp").mkdir()
        (tmp_path / "icu").mkdir()
        (tmp_path / "icu" / "chartevents.csv").write_text(
            "subject_id,hadm_id,stay_id,itemid,charttime,valuenum,valueuom\n"
            "1,100,1000,220045,2150-01-01 06:00:00,72.0,bpm\n"
        )
        ldr = MimicTableLoader(tmp_path)
        result = ldr.summary()
        missing = result[result["rows_sampled"] == 0]
        assert len(missing) >= 1
        assert missing["null_rate_pct"].isna().all()
