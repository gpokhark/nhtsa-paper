from __future__ import annotations

import zipfile
from pathlib import Path

from nhtsa_pipeline.data import make_gold_label_sample, parse_complaint_flat_file, parse_recall_flat_file


def _write_zip(path: Path, member: str, text: str) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, text)
    return path


def test_parse_official_complaint_flat_file_filters_2018_and_preserves_ids(tmp_path: Path) -> None:
    rows = [
        [
            "100000001", "110000001", "FORD MOTOR COMPANY", "FORD", "F-150", "2018",
            "N", "20180105", "N", "0", "0", "SERVICE BRAKES", "AUSTIN", "TX", "1FTXXXXXXX1",
            "20180110", "20180112", "12000", "1", "Brake pedal went to floor while stopping at low speed.",
        ],
        [
            "100000002", "110000002", "FORD MOTOR COMPANY", "FORD", "F-150", "2018",
            "N", "20190105", "N", "0", "0", "ENGINE", "AUSTIN", "TX", "1FTXXXXXXX2",
            "20190110", "20190112", "12000", "1", "Engine stalled while driving.",
        ],
    ]
    text = "\n".join("\t".join(row) for row in rows)
    zip_path = _write_zip(tmp_path / "COMPLAINTS_2015-2019.zip", "COMPLAINTS_RECEIVED_2015-2019.txt", text)

    df = parse_complaint_flat_file(zip_path, received_year=2018)

    assert len(df) == 1
    assert df.loc[0, "Complaint ID"] == "100000001"
    assert df.loc[0, "ODI Number"] == "110000001"
    assert df.loc[0, "Make"] == "FORD"
    assert df.loc[0, "Component"] == "SERVICE BRAKES"
    assert df.loc[0, "Date Received"].year == 2018
    assert "Brake pedal" in df.loc[0, "Narrative"]


def test_parse_official_recall_flat_file_maps_core_columns(tmp_path: Path) -> None:
    row = [
        "1", "18V001000", "FORD", "F-150", "2018", "18S01", "SERVICE BRAKES",
        "FORD MOTOR COMPANY", "20170101", "20181231", "V", "1000", "20180201", "ODI",
        "FORD", "20180115", "20180116", "", "", "Brake defect summary", "Crash risk consequence",
        "Dealer remedy", "Notes", "RCLCOMP", "Brake", "Brake assembly", "PART1", "NO", "NO",
    ]
    zip_path = _write_zip(tmp_path / "RECALLS_post-2010.zip", "FLAT_RCL_POST_2010.txt", "\t".join(row))

    df = parse_recall_flat_file(zip_path, reported_year_min=2010)

    assert len(df) == 1
    assert df.loc[0, "Recall Number"] == "18V001000"
    assert df.loc[0, "Make"] == "FORD"
    assert df.loc[0, "Model Year"] == 2018
    assert df.loc[0, "Component"] == "SERVICE BRAKES"
    assert df.loc[0, "Reported Date"].year == 2018
    assert "Brake defect" in df.loc[0, "Defect Summary"]


def test_make_gold_label_sample_adds_blank_annotation_columns(tmp_path: Path) -> None:
    rows = []
    for idx in range(12):
        rows.append({
            "Complaint ID": f"C{idx}",
            "ODI Number": f"O{idx}",
            "Date Received": "2018-01-01",
            "Make": "FORD" if idx % 2 == 0 else "TOYOTA",
            "Model": "MODEL",
            "Model Year": 2018,
            "Component": "AIR BAGS" if idx % 3 == 0 else "ENGINE",
            "CRASH": "Y" if idx == 0 else "N",
            "FIRE": "N",
            "INJURED": 1 if idx == 1 else 0,
            "DEATHS": 0,
            "Mileage": 1000 + idx,
            "Narrative": f"Complaint narrative {idx} with enough detail for sampling.",
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    sample = make_gold_label_sample(df, n=6, random_state=7, min_per_priority_group=2)

    assert len(sample) == 6
    assert sample["sample_id"].is_unique
    assert "gold_component" in sample.columns
    assert "gold_failure_mode" in sample.columns
    assert "reviewer_id" in sample.columns
    assert set(sample["gold_component"].unique()) == {""}
    assert "safety_flag" in set(sample["_sample_stratum"])
