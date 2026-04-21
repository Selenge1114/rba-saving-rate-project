"""
02_analysis.py
==============
Primary regression analysis: effect of RBA cash rate on household saving rate.

Runs after 01_clean_data.py has produced data/clean/final_dataset.csv.

Run from project root:
    python code/02_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import os

os.makedirs('output', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/clean/final_dataset.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df['quarter'] = df['date'].dt.to_period('Q')

print(f"Dataset: {len(df)} quarters, {df['quarter'].min()} to {df['quarter'].max()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
# First differences
df['d_saving_rate'] = df['saving_rate'].diff()
df['d_cash_rate']   = df['cash_rate'].diff()

# Lagged cash rate (1 and 2 quarters)
df['cash_rate_lag1'] = df['cash_rate'].shift(1)
df['cash_rate_lag2'] = df['cash_rate'].shift(2)
df['d_cash_rate_lag1'] = df['d_cash_rate'].shift(1)

# COVID dummy: 1 for 2020Q2–2021Q4
df['covid'] = ((df['quarter'] >= '2020Q2') & (df['quarter'] <= '2021Q4')).astype(int)

# GFC dummy: 1 for 2008Q3–2009Q2
df['gfc'] = ((df['quarter'] >= '2008Q3') & (df['quarter'] <= '2009Q2')).astype(int)

# Lagged dependent variable
df['d_saving_lag1'] = df['d_saving_rate'].shift(1)

# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Simple OLS — levels (baseline, likely spurious if I(1))
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 1: OLS Levels (no controls)")
print("saving_rate = α + β·cash_rate + ε")
print("=" * 60)

df1 = df[['saving_rate', 'cash_rate']].dropna()
X1 = sm.add_constant(df1['cash_rate'])
m1 = sm.OLS(df1['saving_rate'], X1).fit(cov_type='HC3')
print(m1.summary())

# ─────────────────────────────────────────────────────────────────────────────
# Model 2: Levels with COVID dummy
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2: OLS Levels + COVID dummy")
print("saving_rate = α + β·cash_rate + γ·COVID + ε")
print("=" * 60)

df2 = df[['saving_rate', 'cash_rate', 'covid']].dropna()
X2 = sm.add_constant(df2[['cash_rate', 'covid']])
m2 = sm.OLS(df2['saving_rate'], X2).fit(cov_type='HC3')
print(m2.summary())

# ─────────────────────────────────────────────────────────────────────────────
# Model 3: First differences (addresses non-stationarity)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 3: First Differences")
print("Δsaving_rate = α + β·Δcash_rate + ε")
print("=" * 60)

df3 = df[['d_saving_rate', 'd_cash_rate']].dropna()
X3 = sm.add_constant(df3['d_cash_rate'])
m3 = sm.OLS(df3['d_saving_rate'], X3).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(m3.summary())
print(f"Durbin-Watson: {durbin_watson(m3.resid):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Model 4: First differences with COVID dummy and lagged DV (preferred model)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 4: First Differences + COVID dummy + Lagged DV (Preferred)")
print("Δsaving_rate = α + β·Δcash_rate(t-1) + γ·Δsaving(t-1) + δ·COVID + ε")
print("=" * 60)

df4 = df[['d_saving_rate', 'd_cash_rate_lag1', 'd_saving_lag1', 'covid']].dropna()
X4 = sm.add_constant(df4[['d_cash_rate_lag1', 'd_saving_lag1', 'covid']])
m4 = sm.OLS(df4['d_saving_rate'], X4).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(m4.summary())
print(f"Durbin-Watson: {durbin_watson(m4.resid):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Results summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<45} {'β (cash rate)':<15} {'R²':<8} {'N'}")
print("-" * 75)
print(f"{'1. Levels (no controls)':<45} {m1.params['cash_rate']:>+.3f}{'*' if m1.pvalues['cash_rate']<0.05 else '':<5} {m1.rsquared:.3f}    {int(m1.nobs)}")
print(f"{'2. Levels + COVID dummy':<45} {m2.params['cash_rate']:>+.3f}{'*' if m2.pvalues['cash_rate']<0.05 else '':<5} {m2.rsquared:.3f}    {int(m2.nobs)}")
print(f"{'3. First differences':<45} {m3.params['d_cash_rate']:>+.3f}{'*' if m3.pvalues['d_cash_rate']<0.05 else '':<5} {m3.rsquared:.3f}    {int(m3.nobs)}")
print(f"{'4. First diff + COVID + Lag DV':<45} {m4.params['d_cash_rate_lag1']:>+.3f}{'*' if m4.pvalues['d_cash_rate_lag1']<0.05 else '':<5} {m4.rsquared:.3f}    {int(m4.nobs)}")
print("Note: * p<0.05. HAC standard errors used for first-difference models.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: actual vs fitted for preferred model (Model 4)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
idx4 = df4.index
ax.plot(df.loc[idx4, 'date'], df4['d_saving_rate'], label='Actual Δsaving_rate',
        color='steelblue', linewidth=1.5)
ax.plot(df.loc[idx4, 'date'], m4.fittedvalues, label='Fitted (Model 4)',
        color='darkorange', linewidth=1.5, linestyle='--')
ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
ax.set_title('Model 4: Actual vs Fitted — Quarterly Change in Saving Rate',
             fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Δ Saving Rate (pp)')
ax.legend()
plt.tight_layout()
plt.savefig('output/fig_actual_vs_fitted.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved to output/fig_actual_vs_fitted.png")
