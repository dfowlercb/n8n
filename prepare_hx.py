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
PROMO_FILES = ["hxcatalogseasonal.csv", "hxrecentseasonal.csv"]
MERCHANT_FILE = "merchantpicks.csv"
ALL_FILES = PROMO_FILES + [MERCHANT_FILE]
OUTPUT_FILE = "hx.csv"


def download_files():
    """Download files from OneDrive using rclone."""
    for filename in ALL_FILES:
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

    # Load promo files with is_merchant_pick = False
    for filename in PROMO_FILES:
        path = Path(filename)
        if not path.exists():
            print(f"Error: {filename} not found", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(
            path,
            dtype={"SKU": str, "Disc Flag": str, "Sale Code": str},
            encoding="utf-8-sig",
            thousands=","
        )
        df["is_merchant_pick"] = False
        print(f"  Loaded {filename}: {len(df)} rows")
        dfs.append(df)

    # Combine promo dataframes
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined promo total: {len(combined)} rows")

    # Ensure Promo Net Sales is numeric
    combined["Promo Net Sales"] = pd.to_numeric(
        combined["Promo Net Sales"].astype(str).str.replace(",", ""),
        errors="coerce"
    )

    # Filter promo data: Disc Flag has no number AND Promo Net Sales >= 425
    disc_flag_numeric = pd.to_numeric(combined["Disc Flag"], errors="coerce")
    disc_flag_not_numeric = disc_flag_numeric.isna()
    promo_net_sales_ok = combined["Promo Net Sales"] >= 425

    filtered = combined[disc_flag_not_numeric & promo_net_sales_ok].copy()
    print(f"After filtering (Disc Flag non-numeric, Promo Net Sales >= 425): {len(filtered)} rows")

    # Load merchant picks with is_merchant_pick = True
    merchant_path = Path(MERCHANT_FILE)
    if not merchant_path.exists():
        print(f"Error: {MERCHANT_FILE} not found", file=sys.stderr)
        sys.exit(1)
    merchant_df = pd.read_csv(
        merchant_path,
        dtype={"SKU": str},
        encoding="utf-8-sig"
    )
    # Normalize column name
    if "OFFICERS_CATEGORY" in merchant_df.columns:
        merchant_df = merchant_df.rename(columns={"OFFICERS_CATEGORY": "Officers Category"})
    merchant_df["is_merchant_pick"] = True
    print(f"  Loaded {MERCHANT_FILE}: {len(merchant_df)} rows")

    # Get set of merchant SKUs for later use
    merchant_skus = set(merchant_df["SKU"].unique())

    # Combine filtered promo data with merchant picks (missing columns become NaN)
    filtered = pd.concat([filtered, merchant_df], ignore_index=True)
    print(f"Final combined total: {len(filtered)} rows")

    # Set is_merchant_pick=True for ALL rows with SKUs from merchantpicks.csv
    # This handles duplicates from promo files that should be marked as merchant picks
    filtered.loc[filtered["SKU"].isin(merchant_skus), "is_merchant_pick"] = True
    merchant_pick_count = filtered["is_merchant_pick"].sum()
    print(f"Marked {merchant_pick_count} rows as merchant picks")

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
