"""
Data download and preprocessing for NHTSA complaint data.

Handles:
- Downloading complaint ZIP files from NHTSA
- Parsing CSV files (with known encoding issues)
- Deduplication and filtering
- Export to Parquet for efficient downstream processing
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# NHTSA complaint data download URLs
COMPLAINT_URLS = {
    "1995-1999": "https://www.nhtsa.gov/downloads/folders/Complaints/COMPLAINTS_RECEIVED_1995-1999.zip",
    "2000-2004": "https://www.nhtsa.gov/downloads/folders/Complaints/COMPLAINTS_RECEIVED_2000-2004.zip",
    "2005-2009": "https://www.nhtsa.gov/downloads/folders/Complaints/COMPLAINTS_RECEIVED_2005-2009.zip",
    "2010-2014": "https://www.nhtsa.gov/downloads/folders/Complaints/COMPLAINTS_RECEIVED_2010-2014.zip",
    "2015-2019": "https://www.nhtsa.gov/downloads/folders/Complaints/COMPLAINTS_RECEIVED_2015-2019.zip",
    "2020-2024": "https://www.nhtsa.gov/downloads/folders/Complaints/COMPLAINTS_RECEIVED_2020-2024.zip",
    "2025-2026": "https://www.nhtsa.gov/downloads/folders/Complaints/COMPLAINTS_RECEIVED_2025-2026.zip",
}

# Expected CSV columns (may vary slightly by year block)
EXPECTED_COLUMNS = [
    "Complaint ID",
    "Date Received",
    "Make",
    "Model",
    "Model Year",
    "Purchase Date",
    "Mileage",
    "Failure Date",
    "Failure Mileage",
    "Component",
    "Defect Characteristic",
    "Narrative",
    "Owner City",
    "Owner State",
    "PMI",
]


def download_complaints(out_dir: str | Path, year_range: str | None = None) -> list[Path]:
    """Download complaint ZIP files from NHTSA.

    Args:
        out_dir: Output directory for downloaded files
        year_range: Specific year range to download (e.g., "2020-2024").
                   If None, downloads all available ranges.

    Returns:
        List of downloaded file paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if year_range:
        urls = {year_range: COMPLAINT_URLS[year_range]}
    else:
        urls = COMPLAINT_URLS

    downloaded = []
    for yr, url in tqdm(urls.items(), desc="Downloading complaints"):
        dest = out_dir / f"COMPLAINTS_{yr}.zip"
        if dest.exists():
            logger.info(f"Already exists: {dest}")
            downloaded.append(dest)
            continue

        try:
            import requests
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(dest)
            logger.info(f"Downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")

    return downloaded


def parse_complaint_csv(zip_path: Path) -> pd.DataFrame:
    """Parse a NHTSA complaint ZIP file into a DataFrame.

    NHTSA complaint CSVs have quirks:
    - Encoded in Windows-1252 or Latin-1 (not UTF-8)
    - Some fields contain embedded newlines
    - Narrative field is the richest source of information
    - Complaint ID is the unique identifier

    Args:
        zip_path: Path to the ZIP file

    Returns:
        DataFrame with standardized columns
    """
    logger.info(f"Parsing {zip_path.name}...")

    with zipfile.ZipFile(zip_path) as zf:
        # Find the CSV file in the ZIP
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV file found in {zip_path}")

        csv_name = csv_files[0]
        logger.info(f"Found CSV: {csv_name}")

        # Try multiple encodings
        for encoding in ["windows-1252", "latin-1", "utf-8"]:
            try:
                with zf.open(csv_name) as f:
                    raw = f.read().decode(encoding)

                # Parse CSV with proper quoting
                reader = csv.DictReader(io.StringIO(raw))
                rows = list(reader)
                logger.info(f"  Parsed {len(rows)} rows with encoding={encoding}")
                break
            except (UnicodeDecodeError, Exception) as e:
                if encoding == "utf-8":
                    raise ValueError(f"Failed to parse {csv_name} with all encodings: {e}")
                logger.debug(f"  Encoding {encoding} failed: {e}, trying next...")
                continue

    df = pd.DataFrame(rows)

    # Standardize column names (handle variations)
    col_map = {}
    for expected in EXPECTED_COLUMNS:
        for col in df.columns:
            if expected.lower() in col.lower() or col.lower() in expected.lower():
                col_map[col] = expected
                break

    df = df.rename(columns=col_map)

    # Keep only expected columns that exist
    keep_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    df = df[keep_cols]

    # Clean narrative: replace newlines with spaces, strip whitespace
    if "Narrative" in df.columns:
        df["Narrative"] = df["Narrative"].astype(str).str.replace(r"\n+", " ", regex=True)
        df["Narrative"] = df["Narrative"].str.strip()
        # Filter out very short or empty narratives
        df = df[df["Narrative"].str.len() >= 10]

    # Clean date_received
    if "Date Received" in df.columns:
        df["Date Received"] = pd.to_datetime(df["Date Received"], errors="coerce")

    # Clean model_year
    if "Model Year" in df.columns:
        df["Model Year"] = pd.to_numeric(df["Model Year"], errors="coerce")

    # Clean mileage
    if "Mileage" in df.columns:
        df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

    # Drop rows with no complaint ID
    if "Complaint ID" in df.columns:
        df = df[df["Complaint ID"].notna() & (df["Complaint ID"].astype(str).str.len() > 0)]

    # Deduplicate by complaint ID
    before = len(df)
    df = df.drop_duplicates(subset=["Complaint ID"], keep="first")
    after = len(df)
    if before != after:
        logger.info(f"  Deduplicated: {before} -> {after} ({before - after} duplicates removed)")

    # Filter to complaints with narratives
    if "Narrative" in df.columns:
        before = len(df)
        df = df[df["Narrative"].str.len() >= 10]
        after = len(df)
        if before != after:
            logger.info(f"  Filtered narratives: {before} -> {after} ({before - after} removed)")

    logger.info(f"  Final: {len(df)} complaints, {len(df.columns)} columns")
    logger.info(f"  Columns: {list(df.columns)}")

    return df


def load_all_complaints(data_dir: str | Path, year_ranges: list[str] | None = None) -> pd.DataFrame:
    """Load and combine all complaint datasets.

    Args:
        data_dir: Directory containing complaint ZIP files
        year_ranges: List of year ranges to load. If None, loads all available.

    Returns:
        Combined DataFrame with all complaints
    """
    data_dir = Path(data_dir)
    zip_files = sorted(data_dir.glob("COMPLAINTS_*.zip"))

    if not zip_files:
        raise FileNotFoundError(f"No complaint ZIP files found in {data_dir}")

    if year_ranges:
        zip_files = [f for f in zip_files if any(yr in f.name for yr in year_ranges)]

    dfs = []
    for zf in zip_files:
        try:
            df = parse_complaint_csv(zf)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to parse {zf}: {e}")

    if not dfs:
        raise ValueError("No complaint data could be loaded")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined: {len(combined):,} total complaints across {len(zip_files)} files")

    return combined


def export_to_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """Export complaint DataFrame to Parquet for efficient downstream processing.

    Args:
        df: DataFrame to export
        output_path: Output Parquet file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure consistent dtypes
    export_df = df.copy()

    # Convert Date Received to ISO format string
    if "Date Received" in export_df.columns:
        export_df["Date Received"] = export_df["Date Received"].dt.strftime("%Y-%m-%d")

    # Fill NaN narratives with empty string
    if "Narrative" in export_df.columns:
        export_df["Narrative"] = export_df["Narrative"].fillna("")

    export_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Exported {len(export_df):,} complaints to {output_path}")


def load_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load complaint data from Parquet.

    Args:
        path: Path to Parquet file

    Returns:
        DataFrame
    """
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} complaints from {path}")
    return df


def get_narrative_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics about complaint narratives.

    Args:
        df: DataFrame with Narrative column

    Returns:
        DataFrame with narrative statistics
    """
    stats = {
        "total_complaints": len(df),
        "with_narrative": df["Narrative"].str.len().gt(0).sum(),
        "avg_narrative_length": df["Narrative"].str.len().mean(),
        "median_narrative_length": df["Narrative"].str.len().median(),
        "min_narrative_length": df["Narrative"].str.len().min(),
        "max_narrative_length": df["Narrative"].str.len().max(),
        "short_narratives_lt_50": (df["Narrative"].str.len() < 50).sum(),
        "medium_narratives_50_200": ((df["Narrative"].str.len() >= 50) & (df["Narrative"].str.len() <= 200)).sum(),
        "long_narratives_gt_200": (df["Narrative"].str.len() > 200).sum(),
    }

    # Year distribution
    if "Date Received" in df.columns:
        stats["year_distribution"] = df["Date Received"].dt.year.value_counts().sort_index().to_dict()

    # Make distribution (top 20)
    if "Make" in df.columns:
        stats["top_makes"] = df["Make"].value_counts().head(20).to_dict()

    return pd.DataFrame([stats])
