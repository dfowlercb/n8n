#!/usr/bin/env python3
"""
Downloads hxcatalogseasonal.csv and hxrecentseasonal.csv from OneDrive,
combines them, filters by Disc Flag (empty) and Promo Net Sales (>=425),
and outputs hx.csv for use by stream_to_n8n.py.
"""

import subprocess
import sys
import pandas as pd
from pathlib import Path

REMOTE = "onedrive:n8n"
FILES = ["hxcatalogseasonal.csv", "hxrecentseasonal.csv"]
OUTPUT_FILE = "hx.csv"


def download_files():
    """Download files from OneDrive using rclone."""
    for filename in FILES:
        source = f"{REMOTE}/{filename}"
        print(f"Downloading {source}...")
        result = subprocess.run(
            ["rclone", "copy", "--ignore-times", source, "."],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error downloading {filename}: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(f"  Downloaded {filename}")


def combine_and_filter():
    """Combine CSVs and filter by Disc Flag and Promo Net Sales."""
    dfs = []
    for filename in FILES:
        path = Path(filename)
        if not path.exists():
            print(f"Error: {filename} not found", file=sys.stderr)
            sys.exit(1)
        # Read with SKU as string to preserve leading zeros
        # Use encoding='utf-8-sig' to handle BOM
        df = pd.read_csv(
            path,
            dtype={"SKU": str, "Disc Flag": str, "Sale Code": str},
            encoding="utf-8-sig",
            thousands=","  # Handle commas in numbers like "7,238.32"
        )
        print(f"  Loaded {filename}: {len(df)} rows")
        dfs.append(df)

    # Combine the dataframes
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined total: {len(combined)} rows")

    # Print dtypes for debugging
    print("\nColumn dtypes:")
    print(combined.dtypes.to_string())

    # Ensure Promo Net Sales is numeric (in case thousands separator didn't fully work)
    combined["Promo Net Sales"] = pd.to_numeric(
        combined["Promo Net Sales"].astype(str).str.replace(",", ""),
        errors="coerce"
    )

    # Filter: Disc Flag has no number AND Promo Net Sales >= 425
    # Convert to numeric - if it fails (NaN), that means it's not a number, so include it
    disc_flag_numeric = pd.to_numeric(combined["Disc Flag"], errors="coerce")
    disc_flag_not_numeric = disc_flag_numeric.isna()
    promo_net_sales_ok = combined["Promo Net Sales"] >= 425

    filtered = combined[disc_flag_not_numeric & promo_net_sales_ok].copy()
    print(f"After filtering (Disc Flag non-numeric, Promo Net Sales >= 425): {len(filtered)} rows")

    # Convert float columns that should be integers to nullable Int64
    int_columns = [
        "Family Link", "Promo Quantity",
        "Regular Baseline Days", "Regular Baseline Quantity",
        "Promo Lift Quantity", "Halo-Cannibal Quantity",
        "Steal Vs Gain Quantity", "ON Hand", "ON Order"
    ]
    for col in int_columns:
        if col in filtered.columns:
            filtered[col] = filtered[col].astype("Int64")

    # Save to output file, keeping SKU as string
    filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


def main():
    print("=== Preparing HX data ===\n")

    print("Step 1: Downloading files from OneDrive...")
    download_files()

    print("\nStep 2: Combining and filtering...")
    combine_and_filter()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
