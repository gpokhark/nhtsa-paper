"""
Data download and preprocessing for official NHTSA flat-file datasets.

The north-star implementation uses NHTSA's official ODI flat files rather
than legacy CSV-style complaint downloads:
- complaints: https://static.nhtsa.gov/odi/ffdd/cmpl/
- recalls:    https://static.nhtsa.gov/odi/ffdd/rcl/

Complaint records are tab-delimited TXT files inside ZIP archives. The parser
keeps the original ODI identifiers (`CMPLID`, `ODINO`) and adds friendly column
aliases used by the rest of this prototype.
"""

from __future__ import annotations

import csv
import logging
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

NHTSA_STATIC_BASE = "https://static.nhtsa.gov/odi/ffdd"

# Official NHTSA complaint bulk files, split by received-date blocks.
COMPLAINT_URLS = {
    "1995-1999": f"{NHTSA_STATIC_BASE}/cmpl/COMPLAINTS_RECEIVED_1995-1999.zip",
    "2000-2004": f"{NHTSA_STATIC_BASE}/cmpl/COMPLAINTS_RECEIVED_2000-2004.zip",
    "2005-2009": f"{NHTSA_STATIC_BASE}/cmpl/COMPLAINTS_RECEIVED_2005-2009.zip",
    "2010-2014": f"{NHTSA_STATIC_BASE}/cmpl/COMPLAINTS_RECEIVED_2010-2014.zip",
    "2015-2019": f"{NHTSA_STATIC_BASE}/cmpl/COMPLAINTS_RECEIVED_2015-2019.zip",
    "2020-2024": f"{NHTSA_STATIC_BASE}/cmpl/COMPLAINTS_RECEIVED_2020-2024.zip",
    "2025-2026": f"{NHTSA_STATIC_BASE}/cmpl/COMPLAINTS_RECEIVED_2025-2026.zip",
}

RECALL_URLS = {
    "post-2010": f"{NHTSA_STATIC_BASE}/rcl/FLAT_RCL_POST_2010.zip",
    "pre-2010": f"{NHTSA_STATIC_BASE}/rcl/FLAT_RCL_PRE_2010.zip",
    "all": f"{NHTSA_STATIC_BASE}/rcl/FLAT_RCL.zip",
}

COMPLAINT_FIELDS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT",
    "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC",
    "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES",
    "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN",
    "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN",
    "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED", "DOT",
    "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN",
    "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME",
    "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP",
    "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN",
    "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]

RECALL_FIELDS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO",
    "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF",
    "ODATE", "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO",
    "FMVSS", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION",
    "NOTES", "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC",
    "MFR_COMP_PTNO", "DO_NOT_DRIVE", "PARK_OUTSIDE",
]

COMPLAINT_RENAME = {
    "CMPLID": "Complaint ID",
    "ODINO": "ODI Number",
    "MAKETXT": "Make",
    "MODELTXT": "Model",
    "YEARTXT": "Model Year",
    "LDATE": "Date Received",
    "FAILDATE": "Failure Date",
    "MILES": "Mileage",
    "COMPDESC": "Component",
    "CDESCR": "Narrative",
}

RECALL_RENAME = {
    "CAMPNO": "Recall Number",
    "MAKETXT": "Make",
    "MODELTXT": "Model",
    "YEARTXT": "Model Year",
    "COMPNAME": "Component",
    "DESC_DEFECT": "Defect Summary",
    "CONEQUENCE_DEFECT": "Consequence Summary",
    "CORRECTIVE_ACTION": "Corrective Action",
    "RCDATE": "Reported Date",
    "POTAFF": "Potential Units Affected",
}


def _download_urls(urls: dict[str, str], out_dir: str | Path, prefix: str, selected: str | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if selected:
        if selected not in urls:
            raise KeyError(f"Unknown {prefix} selection {selected!r}; expected one of {sorted(urls)}")
        urls = {selected: urls[selected]}

    downloaded: list[Path] = []
    for key, url in tqdm(urls.items(), desc=f"Downloading {prefix}"):
        dest = out_dir / f"{prefix}_{key}.zip"
        if dest.exists():
            logger.info("Already exists: %s", dest)
            downloaded.append(dest)
            continue

        from urllib.request import urlopen
        with urlopen(url, timeout=300) as resp, open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        downloaded.append(dest)
        logger.info("Downloaded %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return downloaded


def download_complaints(out_dir: str | Path, year_range: str | None = None) -> list[Path]:
    """Download official NHTSA complaint ZIP files.

    For the 2018 pilot, use `year_range="2015-2019"` and filter to
    `--received-year 2018` during preprocessing.
    """
    return _download_urls(COMPLAINT_URLS, out_dir, "COMPLAINTS", selected=year_range)


def download_recalls(out_dir: str | Path, recall_range: str = "post-2010") -> list[Path]:
    """Download official NHTSA recall ZIP files."""
    return _download_urls(RECALL_URLS, out_dir, "RECALLS", selected=recall_range)


def _read_zip_flat_file(zip_path: str | Path, field_names: list[str]) -> pd.DataFrame:
    zip_path = Path(zip_path)
    logger.info("Parsing %s", zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        flat_files = [name for name in zf.namelist() if name.lower().endswith((".txt", ".lst", ".csv"))]
        if not flat_files:
            raise ValueError(f"No flat data file found in {zip_path}")
        member = flat_files[0]
        logger.info("Found flat file: %s", member)
        with zf.open(member) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=field_names,
                dtype=str,
                encoding="latin-1",
                engine="python",
                quoting=csv.QUOTE_NONE,
                on_bad_lines="warn",
            )
    return df


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().replace({"": pd.NA, "0": pd.NA, "00000000": pd.NA})
    return pd.to_datetime(values, format="%Y%m%d", errors="coerce")


def standardize_complaints(df: pd.DataFrame, received_year: int | None = None) -> pd.DataFrame:
    """Add friendly complaint columns while preserving official columns."""
    df = df.copy()
    for col in ["LDATE", "FAILDATE", "DATEA", "PURCH_DT", "MANUF_DT"]:
        if col in df.columns:
            df[col] = _parse_yyyymmdd(df[col])

    for col in ["YEARTXT", "MILES", "INJURED", "DEATHS", "OCCURENCES", "VEH_SPEED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if received_year is not None:
        before = len(df)
        df = df[df["LDATE"].dt.year == received_year]
        logger.info("Filtered received year %s: %s -> %s", received_year, before, len(df))

    if "CDESCR" in df.columns:
        df["CDESCR"] = df["CDESCR"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        before = len(df)
        df = df[df["CDESCR"].str.len() >= 10]
        logger.info("Filtered short/empty narratives: %s -> %s", before, len(df))

    df = df.rename(columns=COMPLAINT_RENAME)
    logger.info("Final complaints: %s rows, %s columns", len(df), len(df.columns))
    return df


def parse_complaint_flat_file(zip_path: str | Path, received_year: int | None = None) -> pd.DataFrame:
    """Parse an official NHTSA complaint ZIP into a standardized DataFrame."""
    return standardize_complaints(_read_zip_flat_file(zip_path, COMPLAINT_FIELDS), received_year=received_year)


def parse_complaint_csv(zip_path: Path) -> pd.DataFrame:
    """Backward-compatible alias for older code; parses official flat files."""
    return parse_complaint_flat_file(zip_path)


def standardize_recalls(df: pd.DataFrame, reported_year_min: int | None = None) -> pd.DataFrame:
    """Add friendly recall columns while preserving official fields."""
    df = df.copy()
    for col in ["BGMAN", "ENDMAN", "ODATE", "RCDATE", "DATEA"]:
        if col in df.columns:
            df[col] = _parse_yyyymmdd(df[col])
    for col in ["RECORD_ID", "YEARTXT", "POTAFF"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if reported_year_min is not None:
        before = len(df)
        df = df[df["RCDATE"].dt.year >= reported_year_min]
        logger.info("Filtered recalls reported >= %s: %s -> %s", reported_year_min, before, len(df))

    df = df.rename(columns=RECALL_RENAME)
    logger.info("Final recalls: %s rows, %s columns", len(df), len(df.columns))
    return df


def parse_recall_flat_file(zip_path: str | Path, reported_year_min: int | None = 2010) -> pd.DataFrame:
    """Parse an official NHTSA recall ZIP into a standardized DataFrame."""
    return standardize_recalls(_read_zip_flat_file(zip_path, RECALL_FIELDS), reported_year_min=reported_year_min)


def load_all_complaints(
    data_dir: str | Path,
    year_ranges: list[str] | None = None,
    received_year: int | None = None,
) -> pd.DataFrame:
    """Load and combine downloaded complaint ZIP files."""
    data_dir = Path(data_dir)
    zip_files = sorted(data_dir.glob("COMPLAINTS_*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No complaint ZIP files found in {data_dir}")
    if year_ranges:
        zip_files = [f for f in zip_files if any(yr in f.name for yr in year_ranges)]

    dfs = [parse_complaint_flat_file(zf, received_year=received_year) for zf in zip_files]
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    logger.info("Combined complaints: %s rows from %s files", len(combined), len(zip_files))
    return combined


def load_all_recalls(data_dir: str | Path, recall_range: str = "post-2010") -> pd.DataFrame:
    """Load downloaded recall ZIP files."""
    data_dir = Path(data_dir)
    zip_files = sorted(data_dir.glob(f"RECALLS_{recall_range}.zip")) or sorted(data_dir.glob("RECALLS_*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No recall ZIP files found in {data_dir}")
    dfs = [parse_recall_flat_file(zf, reported_year_min=2010 if recall_range == "post-2010" else None) for zf in zip_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def export_to_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """Export a DataFrame to Parquet for downstream processing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Exported %s rows to %s", len(df), output_path)


def load_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load data from Parquet."""
    df = pd.read_parquet(path)
    logger.info("Loaded %s rows from %s", len(df), path)
    return df


def get_narrative_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic complaint narrative statistics."""
    narrative_col = "Narrative" if "Narrative" in df.columns else "CDESCR"
    stats: dict[str, object] = {
        "total_complaints": len(df),
        "with_narrative": df[narrative_col].fillna("").astype(str).str.len().gt(0).sum(),
        "avg_narrative_length": df[narrative_col].fillna("").astype(str).str.len().mean(),
        "median_narrative_length": df[narrative_col].fillna("").astype(str).str.len().median(),
    }
    if "Date Received" in df.columns:
        stats["year_distribution"] = df["Date Received"].dt.year.value_counts().sort_index().to_dict()
    if "Make" in df.columns:
        stats["top_makes"] = df["Make"].value_counts().head(20).to_dict()
    if "Component" in df.columns:
        stats["top_components"] = df["Component"].value_counts().head(20).to_dict()
    return pd.DataFrame([stats])


GOLD_LABEL_COLUMNS = [
    "gold_component",
    "gold_failure_mode",
    "gold_severity_1_5",
    "gold_safety_outcome",
    "gold_crash_fire_injury_death",
    "gold_notes",
    "reviewer_id",
]


def make_gold_label_sample(
    df: pd.DataFrame,
    n: int = 500,
    random_state: int = 42,
    min_per_priority_group: int = 25,
) -> pd.DataFrame:
    """Create a stratified complaint sample for manual gold-label annotation.

    The sample intentionally over-represents safety-salient records because the
    first research question is extraction validity for defect triage, not a
    population estimate. Strata:
    - any crash/fire/injury/death flag
    - top NHTSA components by volume
    - remaining long-tail components

    Returns a CSV-friendly DataFrame with blank `gold_*` columns.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if df.empty:
        raise ValueError("cannot sample from an empty DataFrame")

    work = df.copy()
    for col in ["CRASH", "FIRE"]:
        if col not in work.columns:
            work[col] = "N"
    for col in ["INJURED", "DEATHS"]:
        if col not in work.columns:
            work[col] = 0

    safety_mask = (
        work["CRASH"].fillna("N").astype(str).str.upper().eq("Y")
        | work["FIRE"].fillna("N").astype(str).str.upper().eq("Y")
        | pd.to_numeric(work["INJURED"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(work["DEATHS"], errors="coerce").fillna(0).gt(0)
    )
    work["_sample_stratum"] = "long_tail_component"
    work.loc[safety_mask, "_sample_stratum"] = "safety_flag"

    if "Component" in work.columns:
        top_components = work.loc[~safety_mask, "Component"].value_counts().head(10).index
        top_mask = work["Component"].isin(top_components) & ~safety_mask
        work.loc[top_mask, "_sample_stratum"] = "top_component"

    pieces: list[pd.DataFrame] = []
    safety_take = min(min_per_priority_group, n // 3, int((work["_sample_stratum"] == "safety_flag").sum()))
    top_take = min(min_per_priority_group, n // 3, int((work["_sample_stratum"] == "top_component").sum()))

    for stratum, take in [("safety_flag", safety_take), ("top_component", top_take)]:
        if take > 0:
            pieces.append(
                work[work["_sample_stratum"] == stratum].sample(
                    n=take, random_state=random_state, replace=False
                )
            )

    selected = pd.concat(pieces, ignore_index=False) if pieces else work.iloc[0:0]
    remaining_n = min(n - len(selected), len(work) - len(selected))
    if remaining_n > 0:
        remaining = work.drop(index=selected.index, errors="ignore")
        # Component-stratified fill for broader coverage when possible.
        if "Component" in remaining.columns and remaining["Component"].nunique() > 1:
            fill = (
                remaining.groupby("Component", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), max(1, remaining_n // max(1, remaining["Component"].nunique()))), random_state=random_state))
                .head(remaining_n)
            )
            if len(fill) < remaining_n:
                extra = remaining.drop(index=fill.index, errors="ignore").sample(
                    n=remaining_n - len(fill), random_state=random_state, replace=False
                )
                fill = pd.concat([fill, extra])
        else:
            fill = remaining.sample(n=remaining_n, random_state=random_state, replace=False)
        selected = pd.concat([selected, fill], ignore_index=False)

    preferred_cols = [
        "Complaint ID", "ODI Number", "Date Received", "Failure Date", "Make", "Model",
        "Model Year", "Component", "CRASH", "FIRE", "INJURED", "DEATHS", "Mileage",
        "Narrative", "_sample_stratum",
    ]
    keep_cols = [c for c in preferred_cols if c in selected.columns]
    sample = selected[keep_cols].copy().sample(frac=1, random_state=random_state).reset_index(drop=True)
    sample.insert(0, "sample_id", [f"GL{idx + 1:04d}" for idx in range(len(sample))])
    for col in GOLD_LABEL_COLUMNS:
        sample[col] = ""
    return sample
