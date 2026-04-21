"""
01_clean_data.py
================


Inputs (place in data/raw/):
  a2-data.csv       RBA Table A2 cash rate decisions
  5206034_q.csv     ABS quarterly household income account [PREFERRED]
  Table_34.csv      ABS annual saving ratio               [FALLBACK]

Output:
  data/clean/final_dataset.csv

Run from project root:
    python3 code/01_clean_data.py
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
    sys.exit(f"ERROR: {rba_path} not found.")

rba = pd.read_csv(rba_path, skiprows=11, header=0,
                  usecols=[0, 2], names=['date', 'cash_rate'])
rba = rba.dropna(subset=['date'])
rba['date'] = pd.to_datetime(rba['date'], dayfirst=True, errors='coerce')
rba['cash_rate'] = pd.to_numeric(rba['cash_rate'], errors='coerce')
rba = rba.dropna().sort_values('date').reset_index(drop=True)
print(f"  {len(rba)} decision dates: {rba['date'].min().date()} to {rba['date'].max().date()}")

daily = pd.DataFrame({'date': pd.date_range('2000-01-01', '2025-12-31', freq='D')})
daily = daily.merge(rba, on='date', how='left')
seed = float(rba[rba['date'] < '2000-01-01']['cash_rate'].iloc[-1])
if pd.isna(daily['cash_rate'].iloc[0]):
    daily.loc[daily.index[0], 'cash_rate'] = seed
daily['cash_rate'] = daily['cash_rate'].ffill()
daily['quarter'] = daily['date'].dt.to_period('Q')
cash_q = daily.groupby('quarter')['cash_rate'].mean().round(4).reset_index()
cash_q.columns = ['quarter', 'cash_rate']
print(f"  Quarterly: {len(cash_q)} obs ({cash_q['quarter'].min()} to {cash_q['quarter'].max()})")

# ── 2. ABS Saving Ratio ───────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: ABS household saving ratio")
print("=" * 55)

q_path = os.path.join(RAW, '5206034_q.csv')
a_path = os.path.join(RAW, 'Table_34.csv')

if os.path.exists(q_path):
    print("  Loading QUARTERLY saving ratio.")

    # This file has Net saving and Gross Disposable Income (seasonally adjusted)
    # Saving ratio = Net saving / Gross Disposable Income * 100
    # Seasonally adjusted columns are at index 74 (disposable income) and 76 (net saving)
    df = pd.read_csv(q_path, skiprows=9, header=0)

    date_col  = df.columns[0]
    disp_col  = df.columns[74]
    save_col  = df.columns[76]

    abs_raw = df[[date_col, disp_col, save_col]].copy()
    abs_raw.columns = ['date', 'gross_disposable_income', 'net_saving']
    abs_raw['gross_disposable_income'] = pd.to_numeric(abs_raw['gross_disposable_income'], errors='coerce')
    abs_raw['net_saving'] = pd.to_numeric(abs_raw['net_saving'], errors='coerce')
    abs_raw = abs_raw.dropna()
    abs_raw['saving_rate'] = (abs_raw['net_saving'] / abs_raw['gross_disposable_income'] * 100).round(2)
    abs_raw['date'] = pd.to_datetime(abs_raw['date'], errors='coerce')
    abs_raw = abs_raw.dropna(subset=['date'])
    abs_raw['quarter'] = abs_raw['date'].dt.to_period('Q')
    abs_clean = abs_raw[abs_raw['quarter'] >= '2000Q1'][['quarter', 'saving_rate']].reset_index(drop=True)
    frequency = 'quarterly'

elif os.path.exists(a_path):
    print("  QUARTERLY file not found. Using ANNUAL Table 34 (fallback).")
    abs_raw = pd.read_csv(a_path, skiprows=9, header=0, usecols=[0, 35])
    abs_raw.columns = ['date', 'saving_rate']
    abs_raw['date'] = pd.to_datetime(abs_raw['date'], format='%b-%Y', errors='coerce')
    abs_raw['saving_rate'] = pd.to_numeric(abs_raw['saving_rate'], errors='coerce')
    abs_raw = abs_raw.dropna()
    abs_raw['quarter'] = abs_raw['date'].dt.to_period('Q')
    abs_clean = abs_raw[abs_raw['quarter'] >= '2000Q1'][['quarter', 'saving_rate']].reset_index(drop=True)
    frequency = 'annual'

else:
    sys.exit(f"No ABS file found in {RAW}/")

print(f"  {frequency.capitalize()}: {len(abs_clean)} obs ({abs_clean['quarter'].min()} to {abs_clean['quarter'].max()})")

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
assert merged.isnull().sum().sum() == 0, "Missing values found!"
assert merged['cash_rate'].between(0, 20).all(), "Cash rate out of range!"
assert merged['saving_rate'].between(-20, 40).all(), "Saving rate out of range!"
print(f"  Cash rate:   {merged['cash_rate'].min():.2f}% to {merged['cash_rate'].max():.2f}%")
print(f"  Saving rate: {merged['saving_rate'].min():.1f}% to {merged['saving_rate'].max():.1f}%")
print("  Validation: PASSED")

# ── 5. Save ───────────────────────────────────────────────────────
out = os.path.join(CLEAN, 'final_dataset.csv')
merged.to_csv(out, index=False)
print(f"\nSaved: {out}")
print("\nPreview:")
print(merged.head(8).to_string(index=False))
print("...")
print(merged.tail(4).to_string(index=False))
