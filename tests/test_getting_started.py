"""Run the documented end-to-end workflows against synthetic MIMIC-IV data.

The Python blocks of the getting-started walkthrough (docs/getting-started.md)
and the JOSS paper's example (paper/paper.md) are executed as written against a
synthetic MIMIC-IV 2.2 directory in which some patients have several ICU stays
and admissions, so the documented workflows cannot silently drift from the API.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "getting-started.md"
PAPER = ROOT / "paper" / "paper.md"
DOC_PATH = "/data/mimic-iv-2.2"
VITALS = {220045: (80, 15), 220277: (96, 3), 220210: (18, 5), 220052: (75, 12), 220181: (78, 12)}


def python_blocks(path: Path, start: str, end: str) -> str:
    text = path.read_text()
    section = text[text.index(start) : text.index(end)]
    return "\n".join(re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL))


def walkthrough_code() -> str:
    return python_blocks(DOC, "## End-to-end example", "## Next steps")


def run(code: str, path: Path, mimic_root: Path) -> dict:
    assert DOC_PATH in code
    ns: dict = {}
    exec(compile(code.replace(DOC_PATH, str(mimic_root)), str(path), "exec"), ns)
    return ns


def write_mimic(root: Path) -> None:
    """Synthetic MIMIC-IV: 30 patients, every third one readmitted to the ICU."""
    rng = np.random.default_rng(0)
    (root / "hosp").mkdir(parents=True)
    (root / "icu").mkdir()
    stays, admissions, charts = [], [], []
    for i in range(30):
        subject = 10000032 + i
        n_stays = 2 if i % 3 == 0 else 1
        for k in range(n_stays):
            hadm, stay = subject * 10 + k, subject * 100 + k
            intime = pd.Timestamp("2150-01-01 08:00") + pd.Timedelta(days=200 * k + i)
            died = int(i % 5 == 0 and k == n_stays - 1)
            admissions.append(
                {
                    "subject_id": subject,
                    "hadm_id": hadm,
                    "admittime": intime - pd.Timedelta(hours=6),
                    "dischtime": intime + pd.Timedelta(days=5),
                    "deathtime": intime + pd.Timedelta(days=5) if died else None,
                    "admission_type": "EMERGENCY",
                    "admission_location": "EMERGENCY ROOM",
                    "discharge_location": "DIED" if died else "HOME",
                    "insurance": "Medicare",
                    "hospital_expire_flag": died,
                }
            )
            stays.append(
                {
                    "subject_id": subject,
                    "hadm_id": hadm,
                    "stay_id": stay,
                    "first_careunit": "MICU",
                    "last_careunit": "MICU",
                    "intime": intime,
                    "outtime": intime + pd.Timedelta(days=4),
                    "los": 4.0,
                }
            )
            # Hourly vitals from 2h before to 60h after admission; some fall outside
            # the 48h window and one heart rate is implausible (clipped).
            for h in range(-2, 60):
                for itemid, (mean, sd) in VITALS.items():
                    if rng.random() < 0.3:
                        continue
                    charts.append(
                        {
                            "subject_id": subject,
                            "hadm_id": hadm,
                            "stay_id": stay,
                            "itemid": itemid,
                            "charttime": intime + pd.Timedelta(hours=h),
                            "valuenum": round(rng.normal(mean, sd), 1),
                            "valueuom": "unit",
                        }
                    )
    charts[0]["valuenum"] = 999.0
    pd.DataFrame(admissions).to_csv(root / "hosp" / "admissions.csv", index=False)
    pd.DataFrame(stays).to_csv(root / "icu" / "icustays.csv", index=False)
    pd.DataFrame(charts).to_csv(root / "icu" / "chartevents.csv", index=False)
    # Primary diagnoses in both ICD versions, as in MIMIC-IV.
    pd.DataFrame(
        [
            {
                "subject_id": a["subject_id"],
                "hadm_id": a["hadm_id"],
                "seq_num": 1,
                # Mapped ICD-9, ICD-10, and an ICD-9 code with no built-in mapping.
                "icd_code": ["4280", "I509", "99999"][n % 3],
                "icd_version": [9, 10, 9][n % 3],
            }
            for n, a in enumerate(admissions)
        ]
    ).to_csv(root / "hosp" / "diagnoses_icd.csv", index=False)


@pytest.fixture()
def mimic_root(tmp_path: Path) -> Path:
    root = tmp_path / "mimic-iv-2.2"
    write_mimic(root)
    return root


def test_walkthrough_runs_with_readmitted_patients(mimic_root: Path) -> None:
    ns = run(walkthrough_code(), DOC, mimic_root)

    stays, aligned, windows = ns["stays"], ns["aligned"], ns["windows"]
    result = ns["result"]

    # Readmitted patients are present and each stay is aligned to its own intime.
    assert stays.groupby("subject_id").size().max() == 2
    assert aligned["hours_from_anchor"].between(0, 48).all()
    assert set(aligned["stay_id"]) == set(stays["stay_id"])
    assert ns["vitals"]["heart_rate"].max() <= 300

    # Windows are per stay and carry the admission's outcome.
    assert set(windows["stay_id"]) == set(stays["stay_id"])
    assert windows["hospital_expire_flag"].notna().all()

    # The split is by patient: a readmitted patient's stays stay together.
    train_ids, test_ids = set(result.train["subject_id"]), set(result.test["subject_id"])
    assert not train_ids & test_ids
    assert result.train["hospital_expire_flag"].any()
    assert result.test["hospital_expire_flag"].any()
    for split in (result.train, result.test):
        per_patient = split.groupby("subject_id")["stay_id"].nunique()
        expected = stays.groupby("subject_id")["stay_id"].nunique()
        assert (per_patient == expected[per_patient.index]).all()


def test_paper_example_runs(mimic_root: Path) -> None:
    code = python_blocks(PAPER, "# Example Usage", "# Availability")
    ns = run(code, PAPER, mimic_root)

    # Mapped ICD-9 and ICD-10 codes get their chapter; the unmapped ICD-9 code
    # is kept in icd_code, is NaN in icd10_code, and is labelled "Unknown".
    dx = ns["dx"].set_index("icd_code")
    heart = "Diseases of the circulatory system"
    assert set(dx.loc[["4280", "I509"], "chapter"]) == {heart}
    assert dx.loc[["4280"], "icd10_code"].eq("I509").all()
    assert dx.loc[["99999"], "icd10_code"].isna().all()
    assert set(dx.loc[["99999"], "chapter"]) == {"Unknown"}
    windows, result = ns["windows"], ns["result"]
    assert set(windows["stay_id"]) == set(ns["stays"]["stay_id"])
    assert not set(result.train["subject_id"]) & set(result.test["subject_id"])
    assert result.test["hospital_expire_flag"].any()
