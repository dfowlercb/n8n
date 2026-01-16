#!/usr/bin/env python3
"""
Catalog SKU Picker - Python Version
Reads hx.csv (from prepare_hx.py), selects SKUs for catalog, outputs catalog_skus.json
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'ITEMS_PER_PAGE': 10,
    
    # Bestsellers section (pulls top performers from ALL categories)
    'BESTSELLER_PAGES': 4,
    
    # Category page allocation
    'CATEGORY_PAGES': {
        'BOOKS':           25,
        'BIBLES':          25,
        'CHURCH SUPPLIES': 2,
        'GIFTS':           1,
        'HOMESCHOOL':      1,
        'CLOSEOUTS':       2,
        'CHRISTIAN LIVING': 1,
    },
    
    # Scoring weights (must sum to 1.0)
    'WEIGHTS': {
        'sales': 0.40,
        'margin': 0.30,
        'profit': 0.20,
        'lift': 0.10,
    },
    
    # Filtering thresholds
    'MIN_SALES': 0,
    'MIN_MARGIN': 0,
    
    # SKU deduplication strategy: 'highest_score', 'first', 'sum_sales'
    'DEDUP_STRATEGY': 'highest_score',
}

# File paths
INPUT_FILE = 'hx.csv'
OUTPUT_FILE = 'catalog_skus.json'
ONEDRIVE_PATH = 'onedrive:n8n'
GDRIVE_PATH = 'gdrive:dcatalog'

# Category aliases for mapping
CATEGORY_ALIASES = {
    'BOOK': 'BOOKS',
    'PAPERBACK': 'BOOKS',
    'HARDCOVER': 'BOOKS',
    'BIBLE': 'BIBLES',
    'CHURCH': 'CHURCH SUPPLIES',
    'SUPPLIES': 'CHURCH SUPPLIES',
    'COMMUNION': 'CHURCH SUPPLIES',
    'CHURCH SUP': 'CHURCH SUPPLIES',
    'GIFT': 'GIFTS',
    'GIFTS & ACCESSORIES': 'GIFTS',
    'ACCESSORIES': 'GIFTS',
    'HOMESCHOOLING': 'HOMESCHOOL',
    'HOME SCHOOL': 'HOMESCHOOL',
    'CURRICULUM': 'HOMESCHOOL',
    'EDUCATION': 'HOMESCHOOL',
    'CLOSEOUT': 'CLOSEOUTS',
    'CLEARANCE': 'CLOSEOUTS',
    'SALE': 'CLOSEOUTS',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_numeric(value) -> float:
    """Clean and parse currency/numeric values."""
    if isinstance(value, (int, float)):
        return float(value)
    if not value or value == '':
        return 0.0
    
    s = str(value).strip()
    # Remove currency symbols, commas, percent signs
    cleaned = re.sub(r'[$,€£¥%]', '', s)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def extract_family_key(description: str) -> str:
    """Extract a family key from product description to group related products."""
    if not description:
        return 'UNKNOWN'
    
    key = str(description).upper()
    
    # Remove common quantity patterns
    key = re.sub(r',?\s*BOX OF \d+', '', key, flags=re.IGNORECASE)
    key = re.sub(r',?\s*PACK OF \d+', '', key, flags=re.IGNORECASE)
    key = re.sub(r',?\s*SET OF \d+', '', key, flags=re.IGNORECASE)
    key = re.sub(r',?\s*\d+[\s-]*(PACK|COUNT|CT|PC|PCS|PIECES?)', '', key, flags=re.IGNORECASE)
    key = re.sub(r',?\s*\(\d+\)', '', key)
    
    # Remove slash-number patterns
    key = re.sub(r'/\d+$', '', key)
    key = re.sub(r'/\d+\s', ' ', key)
    
    # Remove trailing numbers
    key = re.sub(r'\s+\d+$', '', key)
    
    # Remove size variations
    key = re.sub(r',?\s*(SMALL|MEDIUM|LARGE|XL|XXL|XXXL|SM|MD|LG)', '', key, flags=re.IGNORECASE)
    key = re.sub(r',?\s*\d+\s*(OZ|OUNCE|ML|LITER|L|GAL|GALLON)', '', key, flags=re.IGNORECASE)
    key = re.sub(r',?\s*(LARGE PRINT|GIANT PRINT|COMPACT|PERSONAL SIZE)', '', key, flags=re.IGNORECASE)
    key = re.sub(r',?\s*(LGPT|LGPRT|PT)\b', '', key, flags=re.IGNORECASE)
    
    # Remove color variations
    key = re.sub(r',?\s*(BLACK|BROWN|BLUE|RED|GREEN|PINK|PURPLE|BURGUNDY|TAN|NAVY|WHITE|GRAY|GREY|BK|BRN|PK)', '', key, flags=re.IGNORECASE)
    
    # Remove binding/material variations
    key = re.sub(r',?\s*(HARDCOVER|SOFTCOVER|PAPERBACK|LEATHER|BONDED LEATHER|GENUINE LEATHER|IMITATION LEATHER)', '', key, flags=re.IGNORECASE)
    key = re.sub(r',?\s*(WITH THUMB INDEX|THUMB[- ]?INDEXED|CUST|GL IN|BON)', '', key, flags=re.IGNORECASE)
    
    # Remove year patterns
    key = re.sub(r'\s*\d{4}[-/]\d{2,4}', '', key)
    key = re.sub(r'\s*\d{2}-\d{2}', '', key)
    key = re.sub(r'\s+\d{4}\b', '', key)
    
    # Clean up
    key = re.sub(r'[,\-/]+\s*$', '', key)
    key = re.sub(r'\s+', ' ', key).strip()
    
    return key or 'UNKNOWN'


def map_to_configured_category(raw_category: str) -> Optional[str]:
    """Map source category to configured category."""
    category = str(raw_category or '').strip().upper()
    
    # Direct match
    if category in CONFIG['CATEGORY_PAGES']:
        return category
    
    # Check aliases
    if category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[category]
    
    # Partial match
    for config_cat in CONFIG['CATEGORY_PAGES']:
        if category in config_cat or config_cat in category:
            return config_cat
    
    return None


def infer_category_from_description(description: str) -> Optional[str]:
    """Infer category from product description if not provided."""
    desc = str(description or '').upper()
    
    if any(x in desc for x in ['BIBLE', 'NIV', 'ESV', 'KJV', 'NKJV', 'NLT']):
        return 'BIBLES'
    
    if any(x in desc for x in ['HOMESCHOOL', 'CURRICULUM', 'WORKBOOK', 'STUDENT BOOK', 'TEACHER', 'GRADE ']):
        return 'HOMESCHOOL'
    
    if any(x in desc for x in ['CHURCH', 'COMMUNION', 'OFFERING', 'BULLETIN']):
        return 'CHURCH SUPPLIES'
    
    if any(x in desc for x in ['GIFT', 'MUG', 'JOURNAL', 'ORNAMENT', 'DECOR']):
        return 'GIFTS'
    
    if any(x in desc for x in ['BOOK', ' HC ', ' PB ']):
        return 'BOOKS'
    
    return None


def determine_category(row: dict) -> Optional[str]:
    """Determine the configured category for an item."""
    raw_category = row.get('OFFICERS_CATEGORY', '')
    
    mapped = map_to_configured_category(raw_category)
    if mapped:
        return mapped
    
    return infer_category_from_description(row.get('ITEM_DESCRIPTION', ''))


def is_merchant_pick(value) -> bool:
    """Check if value indicates a merchant pick."""
    if value is True:
        return True
    if isinstance(value, str):
        lower = value.lower()
        return lower in ('true', '1', 'yes')
    if isinstance(value, (int, float)):
        return value == 1
    return False


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def load_data(filepath: str) -> list[dict]:
    """Load CSV data."""
    rows = []
    with open(filepath, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def process_items(raw_items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Process raw items into merchant picks and regular items."""
    merchant_picks = []
    regular_items = []
    
    for idx, row in enumerate(raw_items):
        description = row.get('ITEM_DESCRIPTION', '')
        
        processed = {
            **row,
            'original_index': idx,
            'clean_sales': parse_numeric(row.get('PROMO_NET_SALES')),
            'clean_margin': parse_numeric(row.get('PROMO_MARGIN__PERC')),
            'clean_profit': parse_numeric(row.get('PROMO_PROFIT')),
            'clean_lift': parse_numeric(row.get('PROMO_LIFT_SALES')),
            'mapped_category': determine_category(row),
            'raw_category': row.get('OFFICERS_CATEGORY', ''),
            'family_key': extract_family_key(description),
        }
        
        if is_merchant_pick(row.get('IS_MERCHANT_PICK')):
            processed['is_merchant_pick'] = True
            processed['catalog_score'] = 9999  # High score for guaranteed placement
            merchant_picks.append(processed)
        else:
            processed['is_merchant_pick'] = False
            regular_items.append(processed)
    
    return merchant_picks, regular_items


def calculate_scores(items: list[dict]) -> None:
    """Calculate catalog scores for items (modifies in place)."""
    if not items:
        return
    
    # Filter valid items
    valid = [i for i in items if i['clean_sales'] > CONFIG['MIN_SALES'] 
             and i['clean_margin'] >= CONFIG['MIN_MARGIN']]
    
    if not valid:
        return
    
    # Calculate stats
    stats = {
        'min': {
            'sales': min(i['clean_sales'] for i in valid),
            'margin': min(i['clean_margin'] for i in valid),
            'profit': min(i['clean_profit'] for i in valid),
            'lift': min(i['clean_lift'] for i in valid),
        },
        'max': {
            'sales': max(i['clean_sales'] for i in valid),
            'margin': max(i['clean_margin'] for i in valid),
            'profit': max(i['clean_profit'] for i in valid),
            'lift': max(i['clean_lift'] for i in valid),
        },
    }
    
    # Calculate scores
    weights = CONFIG['WEIGHTS']
    for item in valid:
        sales_norm = normalize(item['clean_sales'], stats['min']['sales'], stats['max']['sales'])
        margin_norm = normalize(item['clean_margin'], stats['min']['margin'], stats['max']['margin'])
        profit_norm = normalize(item['clean_profit'], stats['min']['profit'], stats['max']['profit'])
        lift_norm = normalize(item['clean_lift'], stats['min']['lift'], stats['max']['lift'])
        
        score = (
            sales_norm * weights['sales'] +
            margin_norm * weights['margin'] +
            profit_norm * weights['profit'] +
            lift_norm * weights['lift']
        ) * 1000
        
        item['catalog_score'] = score


def deduplicate_skus(items: list[dict]) -> list[dict]:
    """Deduplicate by SKU, preferring merchant picks and highest scores."""
    sku_map = {}
    
    for item in items:
        sku = item.get('SKU')
        if sku not in sku_map:
            sku_map[sku] = []
        sku_map[sku].append(item)
    
    deduped = []
    for sku, dupes in sku_map.items():
        if len(dupes) == 1:
            deduped.append(dupes[0])
        else:
            # Prefer merchant picks
            merchant = next((d for d in dupes if d.get('is_merchant_pick')), None)
            if merchant:
                deduped.append(merchant)
            elif CONFIG['DEDUP_STRATEGY'] == 'highest_score':
                best = max(dupes, key=lambda x: x.get('catalog_score', 0))
                deduped.append(best)
            elif CONFIG['DEDUP_STRATEGY'] == 'sum_sales':
                combined = dupes[0].copy()
                combined['clean_sales'] = sum(d['clean_sales'] for d in dupes)
                combined['clean_profit'] = sum(d['clean_profit'] for d in dupes)
                combined['catalog_score'] = sum(d.get('catalog_score', 0) for d in dupes) / len(dupes)
                deduped.append(combined)
            else:  # 'first'
                deduped.append(dupes[0])
    
    return deduped


def select_with_family_dedup(available: list[dict], count: int, featured_families: set) -> list[dict]:
    """Select items while avoiding duplicate families."""
    selected = []
    
    for item in available:
        if len(selected) >= count:
            break
        
        family_key = item.get('family_key', 'UNKNOWN')
        if family_key in featured_families:
            continue
        
        selected.append(item)
        featured_families.add(family_key)
    
    return selected


def build_catalog(merchant_picks: list[dict], regular_items: list[dict]) -> list[dict]:
    """Build the final catalog with page assignments."""
    items_per_page = CONFIG['ITEMS_PER_PAGE']
    bestseller_pages = CONFIG['BESTSELLER_PAGES']
    bestseller_count = bestseller_pages * items_per_page
    
    # Combine and deduplicate
    all_items = merchant_picks + regular_items
    deduped = deduplicate_skus(all_items)
    
    print(f"Items after SKU deduplication: {len(deduped)}")
    
    # Separate after dedup
    deduped_merchants = [i for i in deduped if i.get('is_merchant_pick')]
    deduped_regular = [i for i in deduped if not i.get('is_merchant_pick')]
    
    # Sort regular items by score
    deduped_regular.sort(key=lambda x: x.get('catalog_score', 0), reverse=True)
    
    print(f"Merchant picks after dedup: {len(deduped_merchants)}")
    
    # Category distribution
    print("\nCategory distribution in source data:")
    category_count = {}
    for item in deduped:
        cat = item.get('mapped_category') or 'UNCATEGORIZED'
        category_count[cat] = category_count.get(cat, 0) + 1
    
    for cat, count in sorted(category_count.items(), key=lambda x: -x[1]):
        needed = CONFIG['CATEGORY_PAGES'].get(cat, 0) * items_per_page
        if needed > 0:
            status = '✓' if count >= needed else f'⚠ need {needed}'
        else:
            status = '(not configured)'
        print(f"  {cat}: {count} available {status}")
    
    # Track featured families and selected SKUs
    featured_families = set()
    selected_skus = set()
    
    # Place merchant picks by category
    merchant_by_category = {}
    for item in deduped_merchants:
        cat = item.get('mapped_category') or 'UNCATEGORIZED'
        if cat not in merchant_by_category:
            merchant_by_category[cat] = []
        merchant_by_category[cat].append(item)
        featured_families.add(item.get('family_key', 'UNKNOWN'))
        selected_skus.add(item.get('SKU'))
    
    print("\nMerchant picks by category:")
    for cat, items in merchant_by_category.items():
        print(f"  {cat}: {len(items)} items")
    
    # Select bestsellers
    available_for_bestsellers = [i for i in deduped_regular if i.get('SKU') not in selected_skus]
    bestsellers = select_with_family_dedup(available_for_bestsellers, bestseller_count, featured_families)
    
    for item in bestsellers:
        selected_skus.add(item.get('SKU'))
    
    # Assign bestseller pages
    for idx, item in enumerate(bestsellers):
        page = (idx // items_per_page) + 1
        position = (idx % items_per_page) + 1
        item['catalog_page'] = page
        item['page_position'] = position
        item['layout_section'] = 'BESTSELLERS'
    
    print(f"\nBestsellers selected: {len(bestsellers)} items")
    
    # Select category items
    category_items = []
    current_page = bestseller_pages + 1
    
    for category, page_count in CONFIG['CATEGORY_PAGES'].items():
        items_needed = page_count * items_per_page
        
        # Start with merchant picks
        merchants_for_cat = merchant_by_category.get(category, [])
        selected_for_cat = list(merchants_for_cat)
        
        for item in merchants_for_cat:
            featured_families.add(item.get('family_key', 'UNKNOWN'))
        
        # Fill remaining slots
        remaining = items_needed - len(selected_for_cat)
        if remaining > 0:
            available = [i for i in deduped_regular 
                        if i.get('mapped_category') == category 
                        and i.get('SKU') not in selected_skus]
            available.sort(key=lambda x: x.get('catalog_score', 0), reverse=True)
            
            additional = select_with_family_dedup(available, remaining, featured_families)
            for item in additional:
                selected_skus.add(item.get('SKU'))
            
            selected_for_cat.extend(additional)
        
        # Assign page positions
        for idx, item in enumerate(selected_for_cat):
            page_offset = idx // items_per_page
            position = (idx % items_per_page) + 1
            item['catalog_page'] = current_page + page_offset
            item['page_position'] = position
            item['layout_section'] = category
            category_items.append(item)
        
        merchant_count = len(merchants_for_cat)
        regular_count = len(selected_for_cat) - merchant_count
        pages_used = max(1, (len(selected_for_cat) + items_per_page - 1) // items_per_page)
        
        print(f"{category}: {len(selected_for_cat)}/{items_needed} items "
              f"({merchant_count} merchant picks + {regular_count} regular), "
              f"pages {current_page}-{current_page + pages_used - 1}")
        
        if len(selected_for_cat) < items_needed:
            print(f"  ⚠ WARNING: Only {len(selected_for_cat)} items available, need {items_needed}")
        
        current_page += page_count
    
    # Combine and finalize
    final_catalog = bestsellers + category_items
    
    # Sort by page and position
    final_catalog.sort(key=lambda x: (x.get('catalog_page', 0), x.get('page_position', 0)))
    
    # Add sequential catalog ID
    for idx, item in enumerate(final_catalog):
        item['catalog_sequence'] = idx + 1
    
    return final_catalog, featured_families


def upload_to_drive(filepath: str):
    """Upload file to OneDrive and Google Drive using rclone."""
    # Upload to OneDrive
    print(f"\nUploading {filepath} to {ONEDRIVE_PATH}...")
    result = subprocess.run(
        ['rclone', 'copy', '--ignore-times', filepath, ONEDRIVE_PATH],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"OneDrive upload failed: {result.stderr}", file=sys.stderr)
    else:
        print(f"  Uploaded to OneDrive successfully!")

    # Upload to Google Drive
    print(f"Uploading {filepath} to {GDRIVE_PATH}...")
    result = subprocess.run(
        ['rclone', 'copy', '--ignore-times', filepath, GDRIVE_PATH],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Google Drive upload failed: {result.stderr}", file=sys.stderr)
    else:
        print(f"  Uploaded to Google Drive successfully!")


def main():
    print("=" * 60)
    print("CATALOG SKU PICKER")
    print("=" * 60)
    
    # Calculate totals
    bestseller_count = CONFIG['BESTSELLER_PAGES'] * CONFIG['ITEMS_PER_PAGE']
    total_category_pages = sum(CONFIG['CATEGORY_PAGES'].values())
    total_pages = CONFIG['BESTSELLER_PAGES'] + total_category_pages
    total_items = total_pages * CONFIG['ITEMS_PER_PAGE']
    
    print(f"Bestseller pages: {CONFIG['BESTSELLER_PAGES']} ({bestseller_count} items)")
    print("\nCategory pages:")
    for cat, pages in CONFIG['CATEGORY_PAGES'].items():
        print(f"  {cat}: {pages} pages ({pages * CONFIG['ITEMS_PER_PAGE']} items)")
    print(f"\nTotal: {total_pages} pages, {total_items} items")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    raw_items = load_data(INPUT_FILE)
    print(f"Loaded {len(raw_items)} rows")
    
    # Process items
    print("\nProcessing items...")
    merchant_picks, regular_items = process_items(raw_items)
    print(f"Merchant picks: {len(merchant_picks)}")
    print(f"Regular items: {len(regular_items)}")
    
    # Calculate scores
    print("\nCalculating scores...")
    calculate_scores(regular_items)
    
    # Build catalog
    print("\nBuilding catalog...")
    final_catalog, featured_families = build_catalog(merchant_picks, regular_items)
    
    # Summary
    print("\n" + "=" * 60)
    print("CATALOG SUMMARY")
    print("=" * 60)
    print(f"Total items: {len(final_catalog)}")
    print(f"Total pages: {(len(final_catalog) + CONFIG['ITEMS_PER_PAGE'] - 1) // CONFIG['ITEMS_PER_PAGE']}")
    print(f"Unique families featured: {len(featured_families)}")
    print("=" * 60)
    
    # Save output
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_catalog, f, indent=2)
    print(f"Saved {len(final_catalog)} items")
    
    # Upload
    upload_to_drive(OUTPUT_FILE)
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
