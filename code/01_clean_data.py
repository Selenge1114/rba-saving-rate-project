"""
01_clean_data.py
================
Cleans and merges raw datasets into analysis-ready CSV.

Inputs (place in data/raw/):
  a2-data.csv       RBA Table A2 cash rate decisions (download from rba.gov.au)
  5206034_q.csv     ABS quarterly saving ratio [PREFERRED - see README]
  Table_34.csv      ABS annual saving ratio    [FALLBACK - already included]

Output:
  data/clean/final_dataset.csv

Run from project root:
    python code/01_clean_data.py
"""

import pandas as pd
import numpy as np
import os, sys

RAW   = os.path.join('data', 'raw')
CLEAN = os.path.join('data', 'clean')
os.makedirs(CLEAN, exist_ok=True)

# ── 1. RBA A2 Cash Rate → quarterly averages ──────────────────────
print("=" * 55)
print("STEP 1: RBA A2 cash rate")
print("=" * 55)

rba_path = os.path.join(RAW, 'a2-data.csv')
if not os.path.exists(rba_path):
    sys.exit(f"ERROR: {rba_path} not found.\nDownload Table A2 CSV from rba.gov.au/statistics/tables/")

rba = pd.read_csv(rba_path, skiprows=11, header=0,
                  usecols=[0, 2], names=['date', 'cash_rate'])
rba = rba.dropna(subset=['date'])
rba['date'] = pd.to_datetime(rba['date'], dayfirst=True, errors='coerce')
rba['cash_rate'] = pd.to_numeric(rba['cash_rate'], errors='coerce')
rba = rba.dropna().sort_values('date').reset_index(drop=True)
print(f"  {len(rba)} decision dates: {rba['date'].min().date()} → {rba['date'].max().date()}")

# Build daily series (forward-fill announced rate to every day)
daily = pd.DataFrame({'date': pd.date_range('2000-01-01', '2025-09-30', freq='D')})
daily = daily.merge(rba, on='date', how='left')
seed = float(rba[rba['date'] < '2000-01-01']['cash_rate'].iloc[-1])
if pd.isna(daily['cash_rate'].iloc[0]):
    daily.loc[daily.index[0], 'cash_rate'] = seed
daily['cash_rate'] = daily['cash_rate'].ffill()
daily['quarter'] = daily['date'].dt.to_period('Q')
cash_q = daily.groupby('quarter')['cash_rate'].mean().round(4).reset_index()
cash_q.columns = ['quarter', 'cash_rate']
print(f"  → {len(cash_q)} quarterly obs ({cash_q['quarter'].min()} to {cash_q['quarter'].max()})")

# ── 2. ABS Saving Ratio ───────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: ABS household saving ratio")
print("=" * 55)

q_path = os.path.join(RAW, '5206034_q.csv')   # quarterly (preferred)
a_path = os.path.join(RAW, 'Table_34.csv')     # annual (fallback)

if os.path.exists(q_path):
    # ── Quarterly (preferred) ──
    print("  Loading QUARTERLY saving ratio.")
    abs_raw = pd.read_csv(q_path, skiprows=9, header=0)
    # Find saving ratio column
    saving_col = next((c for c in abs_raw.columns
                       if 'saving ratio' in str(c).lower()), None)
    if saving_col is None:
        print("  Columns found:"); [print(f"    {c}") for c in abs_raw.columns[:15]]
        sys.exit("Cannot find saving ratio column. Check column name above and update script.")
    abs_raw = abs_raw[[abs_raw.columns[0], saving_col]].copy()
    abs_raw.columns = ['date', 'saving_rate']
    abs_raw['date'] = pd.to_datetime(abs_raw['date'], errors='coerce')
    frequency = 'quarterly'

elif os.path.exists(a_path):
    # ── Annual fallback ──
    print("  QUARTERLY file not found. Using ANNUAL Table 34 (fallback).")
    print("  → Only ~26 annual obs. For full analysis, obtain quarterly data (see README).")
    abs_raw = pd.read_csv(a_path, skiprows=9, header=0, usecols=[0, 35])
    abs_raw.columns = ['date', 'saving_rate']
    abs_raw['date'] = pd.to_datetime(abs_raw['date'], format='%b-%Y', errors='coerce')
    frequency = 'annual'

else:
    sys.exit(f"No ABS file found in {RAW}/\nExpected: 5206034_q.csv or Table_34.csv")

abs_raw['saving_rate'] = pd.to_numeric(abs_raw['saving_rate'], errors='coerce')
abs_raw = abs_raw.dropna().copy()
abs_raw['quarter'] = abs_raw['date'].dt.to_period('Q')
abs_clean = abs_raw[abs_raw['quarter'] >= '2000Q1'][['quarter', 'saving_rate']].reset_index(drop=True)
print(f"  → {len(abs_clean)} {frequency} obs ({abs_clean['quarter'].min()} to {abs_clean['quarter'].max()})")

# ── 3. Merge ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Merging")
print("=" * 55)

merged = pd.merge(cash_q, abs_clean, on='quarter', how='inner')
merged['date'] = merged['quarter'].dt.to_timestamp()
merged['quarter'] = merged['quarter'].astype(str)
merged = merged[['date','quarter','cash_rate','saving_rate']].sort_values('date').reset_index(drop=True)
print(f"  Final dataset: {len(merged)} obs, {merged['quarter'].iloc[0]} to {merged['quarter'].iloc[-1]}")

# ── 4. Validation ─────────────────────────────────────────────────
assert merged.isnull().sum().sum() == 0, "Missing values found in merged dataset!"
assert merged['cash_rate'].between(0, 20).all(), "Cash rate value out of range!"
assert merged['saving_rate'].between(-20, 40).all(), "Saving rate value out of range!"
print("  Validation: PASSED")
print(f"  Cash rate:    {merged['cash_rate'].min():.2f}% – {merged['cash_rate'].max():.2f}%")
print(f"  Saving rate:  {merged['saving_rate'].min():.1f}% – {merged['saving_rate'].max():.1f}%")

# ── 5. Save ───────────────────────────────────────────────────────
out = os.path.join(CLEAN, 'final_dataset.csv')
merged.to_csv(out, index=False)
print(f"\n✓ Saved: {out}")
print("\nPreview:")
print(merged.to_string(index=False))
