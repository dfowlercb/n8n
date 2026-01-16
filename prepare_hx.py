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


def normalize_column(col):
    """Normalize column names to UPPERCASE_WITH_UNDERSCORES for n8n compatibility."""
    return (col
        .strip()
        .upper()
        .replace(" ", "_")
        .replace("%", "_PERC")
        .replace("-", "_")
    )


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
        # Normalize column names IMMEDIATELY after loading
        df.columns = [normalize_column(c) for c in df.columns]
        df["IS_MERCHANT_PICK"] = False
        print(f"  Loaded {filename}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
        dfs.append(df)

    # Combine promo dataframes
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined promo total: {len(combined)} rows")

    # Ensure PROMO_NET_SALES is numeric (using normalized column name)
    combined["PROMO_NET_SALES"] = pd.to_numeric(
        combined["PROMO_NET_SALES"].astype(str).str.replace(",", ""),
        errors="coerce"
    )

    # Filter promo data: DISC_FLAG has no number AND PROMO_NET_SALES >= 425
    disc_flag_numeric = pd.to_numeric(combined["DISC_FLAG"], errors="coerce")
    disc_flag_not_numeric = disc_flag_numeric.isna()
    promo_net_sales_ok = combined["PROMO_NET_SALES"] >= 425

    filtered = combined[disc_flag_not_numeric & promo_net_sales_ok].copy()
    print(f"After filtering (DISC_FLAG non-numeric, PROMO_NET_SALES >= 425): {len(filtered)} rows")
    
    # Debug: Check OFFICERS_CATEGORY in filtered promo data
    if "OFFICERS_CATEGORY" in filtered.columns:
        cat_counts = filtered["OFFICERS_CATEGORY"].value_counts()
        print(f"  OFFICERS_CATEGORY distribution (top 5): {cat_counts.head().to_dict()}")

    # Load merchant picks with IS_MERCHANT_PICK = True
    merchant_path = Path(MERCHANT_FILE)
    if not merchant_path.exists():
        print(f"Error: {MERCHANT_FILE} not found", file=sys.stderr)
        sys.exit(1)
    merchant_df = pd.read_csv(
        merchant_path,
        dtype={"SKU": str},
        encoding="utf-8-sig"
    )
    # Normalize column names IMMEDIATELY after loading
    merchant_df.columns = [normalize_column(c) for c in merchant_df.columns]
    merchant_df["IS_MERCHANT_PICK"] = True
    print(f"  Loaded {MERCHANT_FILE}: {len(merchant_df)} rows, columns: {list(merchant_df.columns)}")
    
    # Debug: Check if merchant picks have OFFICERS_CATEGORY
    if "OFFICERS_CATEGORY" in merchant_df.columns:
        print(f"  Merchant OFFICERS_CATEGORY values: {merchant_df['OFFICERS_CATEGORY'].value_counts().to_dict()}")
    else:
        print(f"  ⚠ WARNING: Merchant picks missing OFFICERS_CATEGORY column!")

    # Get set of merchant SKUs for later use
    merchant_skus = set(merchant_df["SKU"].unique())

    # Combine filtered promo data with merchant picks (missing columns become NaN)
    filtered = pd.concat([filtered, merchant_df], ignore_index=True)
    print(f"Final combined total: {len(filtered)} rows")
    
    # Debug: Check OFFICERS_CATEGORY after concat
    if "OFFICERS_CATEGORY" in filtered.columns:
        non_empty = filtered["OFFICERS_CATEGORY"].notna() & (filtered["OFFICERS_CATEGORY"] != "")
        print(f"  Rows with OFFICERS_CATEGORY after concat: {non_empty.sum()} / {len(filtered)}")

    # Set IS_MERCHANT_PICK=True for ALL rows with SKUs from merchantpicks.csv
    # This handles duplicates from promo files that should be marked as merchant picks
    filtered.loc[filtered["SKU"].isin(merchant_skus), "IS_MERCHANT_PICK"] = True
    merchant_pick_count = filtered["IS_MERCHANT_PICK"].sum()
    print(f"Marked {merchant_pick_count} rows as merchant picks")

    # Convert float columns that should be integers to nullable Int64
    int_columns = [
        "FAMILY_LINK", "PROMO_QUANTITY",
        "REGULAR_BASELINE_DAYS", "REGULAR_BASELINE_QUANTITY",
        "PROMO_LIFT_QUANTITY", "HALO_CANNIBAL_QUANTITY",
        "STEAL_VS_GAIN_QUANTITY", "ON_HAND", "ON_ORDER"
    ]
    for col in int_columns:
        if col in filtered.columns:
            filtered[col] = filtered[col].astype("Int64")

    # Final column list
    print(f"Final columns: {list(filtered.columns)}")

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
