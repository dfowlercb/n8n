#!/usr/bin/env python3
"""
Downloads catalog_skus.json from OneDrive and converts it to CSV for use in R.
"""

import subprocess
import sys
import json
import pandas as pd
from pathlib import Path

REMOTE = "onedrive:n8n"
INPUT_FILE = "catalog_skus.json"
OUTPUT_FILE = "catalog_skus.csv"


def download_file():
    """Download file from OneDrive using rclone."""
    source = f"{REMOTE}/{INPUT_FILE}"
    print(f"Downloading {source}...")
    result = subprocess.run(
        ["rclone", "copy", "--ignore-times", source, "."],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error downloading {INPUT_FILE}: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  Downloaded {INPUT_FILE}")


def convert_to_csv():
    """Convert JSON to CSV for R."""
    path = Path(INPUT_FILE)
    if not path.exists():
        print(f"Error: {INPUT_FILE} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT_FILE}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} records")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Ensure SKU is string to preserve leading zeros
    df["SKU"] = df["SKU"].astype(str)

    # Convert boolean columns to R-friendly format (TRUE/FALSE)
    bool_columns = df.select_dtypes(include=["bool"]).columns
    for col in bool_columns:
        df[col] = df[col].map({True: "TRUE", False: "FALSE"})

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE} ({len(df)} rows, {len(df.columns)} columns)")


def main():
    print("=== Preparing catalog_skus for R ===\n")

    print("Step 1: Downloading file from OneDrive...")
    download_file()

    print("\nStep 2: Converting to CSV...")
    convert_to_csv()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
