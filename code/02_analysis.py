"""
02_analysis.py
==============
RBA Cash Rate and Australian Household Saving Rate
Descriptive analysis — no causal claim is made.

Runs after 01_clean_data.py has produced data/clean/final_dataset.csv.

Run from project root:
    python3 code/02_analysis.py

Produces:
    output/fig_actual_vs_fitted.png   – Model 4 actual vs fitted
    output/table2_regression.txt      – Regression table (text)
    output/results.txt                – Summary statistics
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# 0. DECLARATION
print(
    "Declaration: This analysis is DESCRIPTIVE. The coefficients capture "
    "conditional correlations between changes in the RBA cash rate and changes "
    "in the household saving rate. No causal claim is made. The cash rate is set "
    "by the RBA in response to macroeconomic conditions, so reverse causality and "
    "omitted variable bias are both plausible."
)

# 1. LOAD DATA
df = pd.read_csv('data/clean/final_dataset.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df['quarter'] = df['date'].dt.to_period('Q')

print(f"\nDataset: {len(df)} quarters, {df['quarter'].min()} to {df['quarter'].max()}\n")

# 2. FEATURE ENGINEERING
df['d_saving_rate'] = df['saving_rate'].diff()
df['d_cash_rate']   = df['cash_rate'].diff()
df['d_cash_rate_lag1'] = df['d_cash_rate'].shift(1)  # Δcash(t-1) for M4
df['d_saving_lag1'] = df['d_saving_rate'].shift(1)   # Δsave(t-1) for M4

# COVID dummy: 2020Q2–2021Q4 (quarters where saving spiked due to non-monetary
# factors: lockdowns, restricted spending, fiscal transfers)
df['covid'] = ((df['quarter'] >= '2020Q2') & (df['quarter'] <= '2021Q4')).astype(int)

# 3. SUMMARY STATISTICS
print("=" * 60)
print("TABLE 1: Summary Statistics")
print("=" * 60)
cols = ['saving_rate', 'cash_rate', 'd_saving_rate', 'd_cash_rate']
desc = df[cols].describe().T
desc.columns = ['N', 'Mean', 'Std', 'Min', 'p25', 'p50', 'p75', 'Max']
print(desc.round(3).to_string())
print()

with open('output/results.txt', 'w') as f:
    f.write("Summary Statistics\n")
    f.write(desc.round(3).to_string())
    f.write("\n")

# 4. ECONOMETRIC SPECIFICATION
print(
    "Models 1–4 estimated with OLS and HAC standard errors (Newey-West, 4 lags).\n"
    "M1: Levels bivariate\n"
    "M2: Levels + COVID dummy\n"
    "M3: First differences (Δsaving = alpha + beta*Δcash + eps)\n"
    "M4 (PREFERRED): Δsaving_t = alpha + beta*Δcash_(t-1) + gamma*Δsaving_(t-1)"
    " + delta*COVID + eps\n"
    "Declaration: DESCRIPTIVE — no causal claim is made.\n"
)

def fit_hac(y, X_df, lags=4):
    X = sm.add_constant(X_df, has_constant='add')
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

# 5. ESTIMATE MODELS
# M1 & M2: levels
df1 = df[['saving_rate', 'cash_rate', 'covid']].dropna()
m1 = fit_hac(df1['saving_rate'], df1[['cash_rate']])
m2 = fit_hac(df1['saving_rate'], df1[['cash_rate', 'covid']])

# M3: first differences (contemp)
df3 = df[['d_saving_rate', 'd_cash_rate']].dropna()
m3 = fit_hac(df3['d_saving_rate'], df3[['d_cash_rate']])

# M4 (preferred): first diff + lagged Δcash + lagged DV + COVID
df4 = df[['d_saving_rate', 'd_cash_rate_lag1', 'd_saving_lag1', 'covid']].dropna()
m4 = fit_hac(df4['d_saving_rate'], df4[['d_cash_rate_lag1', 'd_saving_lag1', 'covid']])

print(f"M1 sample: N={int(m1.nobs)}")
print(f"M2 sample: N={int(m2.nobs)}")
print(f"M3 sample: N={int(m3.nobs)}")
print(f"M4 sample: N={int(m4.nobs)}, COVID quarters: {df4['covid'].sum()}\n")

# 6. REGRESSION TABLE
def fmt(model, var):
    try:
        c  = model.params[var]
        se = model.bse[var]
        p  = model.pvalues[var]
        st = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
        return f"{c:+.3f}{st}", f"({se:.3f})"
    except KeyError:
        return "", ""

MODELS = [m1, m2, m3, m4]
LABELS = ['(1) Levels', '(2)+COVID', '(3) FD', '(4) FD Preferred']
VARLIST = [
    ('cash_rate',        'Cash rate (%)'),
    ('covid',            'COVID dummy'),
    ('d_cash_rate',      'dCash rate'),
    ('d_cash_rate_lag1', 'dCash rate, t-1'),
    ('d_saving_lag1',    'dSaving rate, t-1'),
    ('const',            'Constant'),
]

W = 18
hdr = f"{'Variable':<28}" + "".join(f"{l:>{W}}" for l in LABELS)
sep = "-" * (28 + W * 4)

lines_out = []
lines_out.append("")
lines_out.append("=" * (28 + W * 4))
lines_out.append("TABLE 2: OLS Estimates")
lines_out.append("HAC robust SEs (Newey-West, 4 lags) in parentheses.")
lines_out.append("Declaration: DESCRIPTIVE. No causal claim is made.")
lines_out.append("=" * (28 + W * 4))
lines_out.append(hdr)
lines_out.append(sep)

for code, label in VARLIST:
    crow = f"{label:<28}"
    srow = f"{'':28}"
    for m in MODELS:
        c, s = fmt(m, code)
        crow += f"{c:>{W}}"
        srow += f"{s:>{W}}"
    lines_out.append(crow)
    lines_out.append(srow)

lines_out.append(sep)
lines_out.append(f"{'N':<28}" + "".join(f"{int(m.nobs):>{W}}" for m in MODELS))
lines_out.append(f"{'R2':<28}" + "".join(("{:" + f">{W}" + "}").format(f"{m.rsquared:.3f}") for m in MODELS))
lines_out.append(f"{'Adj. R2':<28}" + "".join(("{:" + f">{W}" + "}").format(f"{m.rsquared_adj:.3f}") for m in MODELS))
lines_out.append(f"{'Durbin-Watson':<28}" + "".join(("{:" + f">{W}" + "}").format(f"{durbin_watson(m.resid):.3f}") for m in MODELS))
lines_out.append(sep)
lines_out.append("* p<0.10  ** p<0.05  *** p<0.01")
lines_out.append("M1-M2: dependent variable = saving_rate (levels).")
lines_out.append("M3-M4: dependent variable = first difference of saving_rate.")
lines_out.append("M4 is the preferred specification.")
lines_out.append("")

table_str = "\n".join(lines_out)
print(table_str)

with open('output/table2_regression.txt', 'w') as f:
    f.write(table_str)
print("Saved: output/table2_regression.txt")

# 7. DIAGNOSTICS
print("\nDIAGNOSTICS — Preferred Specification (M4)")
print("-" * 50)
lb = acorr_ljungbox(m4.resid, lags=[4, 8], return_df=True)
print("Ljung-Box test (residual serial correlation):")
print(lb[['lb_stat', 'lb_pvalue']].round(4))

X4 = sm.add_constant(df4[['d_cash_rate_lag1', 'd_saving_lag1', 'covid']])
bp_stat, bp_p, _, _ = het_breuschpagan(m4.resid, X4)
print(f"\nBreusch-Pagan: stat={bp_stat:.3f}, p={bp_p:.4f}")
print(f"Durbin-Watson: {durbin_watson(m4.resid):.3f}")

# 8. FIGURE: ACTUAL VS FITTED (M4)
fig, ax = plt.subplots(figsize=(12, 5))
dates4 = df.loc[df4.index, 'date']
ax.plot(dates4, df4['d_saving_rate'], color='steelblue', linewidth=1.5,
        label='Actual delta saving rate')
ax.plot(dates4, m4.fittedvalues, color='darkorange', linewidth=1.5,
        linestyle='--', label='Fitted (M4 preferred)')
ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
ax.set_title('Model 4: Actual vs Fitted — Quarterly Change in Saving Rate',
             fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Change in Saving Rate (pp)')
ax.legend()
plt.tight_layout()
plt.savefig('output/fig_actual_vs_fitted.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: output/fig_actual_vs_fitted.png")

# 9. SESSION INFO
print("\nSESSION INFO")
print("-" * 40)
print(f"Python:       {sys.version.split()[0]}")
print(f"pandas:       {pd.__version__}")
print(f"numpy:        {np.__version__}")
import statsmodels
print(f"statsmodels:  {statsmodels.__version__}")
print("\nAnalysis complete. Outputs saved to output/")
print("\nFigure saved to output/fig_actual_vs_fitted.png")
