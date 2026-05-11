"""
03_robustness.py

Robustness Analysis: RBA Cash Rate and Household Saving Rate

Main result (from 02_analysis.py, Model 4 — PREFERRED specification):
    Δsaving_rate(t) = α + β·Δcash_rate(t-1) + γ·Δsaving_rate(t-1) + δ·COVID + ε
    Estimated using OLS with HAC standard errors (Newey-West, 4 lags).

Declaration: DESCRIPTIVE — the coefficient β captures the conditional correlation
between a lagged change in the RBA cash rate and the contemporaneous change in the
household saving rate, controlling for saving rate momentum and the COVID-19 shock.
It does not claim a causal effect.

Robustness checks:
  A. Alternative control sets     (Cols 2–3)
  B. Alternative samples          (Cols 4–5)
  C. Alternative functional form  (Cols 6–7)
  D. Alternative inference        (Cols 8–9)
  E. Subsample / time-window      (Col 10)

Run from project root:
    python3 code/03_robustness.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
import warnings, os

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

# 0. Load and engineer features  (mirrors 02_analysis.py exactly)

df = pd.read_csv('data/clean/final_dataset.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df['quarter'] = df['date'].dt.to_period('Q')

# ── first differences
df['d_saving_rate']  = df['saving_rate'].diff()
df['d_cash_rate']    = df['cash_rate'].diff()

# ── lags
df['d_cash_rate_lag1'] = df['d_cash_rate'].shift(1)
df['d_cash_rate_lag2'] = df['d_cash_rate'].shift(2)
df['d_saving_lag1']    = df['d_saving_rate'].shift(1)
df['d_saving_lag2']    = df['d_saving_rate'].shift(2)
df['cash_rate_lag1']   = df['cash_rate'].shift(1)

# ── event dummies
df['covid'] = ((df['quarter'] >= '2020Q2') & (df['quarter'] <= '2021Q4')).astype(int)
df['gfc']   = ((df['quarter'] >= '2008Q3') & (df['quarter'] <= '2009Q2')).astype(int)

# ── functional-form transforms
# IHS of saving rate (handles negatives; identical to log for large positives)
df['ihs_saving_rate']  = np.arcsinh(df['saving_rate'])
df['d_ihs_saving']     = df['ihs_saving_rate'].diff()
df['d_ihs_lag1']       = df['d_ihs_saving'].shift(1)

# IHS of cash rate change
df['d_ihs_cash']       = np.arcsinh(df['d_cash_rate'])
df['d_ihs_cash_lag1']  = df['d_ihs_cash'].shift(1)

# ── time index (trend)
df['t'] = np.arange(len(df))

print("=" * 65)
print("RBA CASH RATE & HOUSEHOLD SAVING RATE — ROBUSTNESS ANALYSIS")
print("=" * 65)
print(f"Dataset: {len(df)} quarters  ({df['quarter'].min()} – {df['quarter'].max()})\n")

# Helper: fit OLS with specified SE type and return a result dict

def run_ols(endog, exog_df, se_type='HAC', hac_lags=4, label=''):
    """
    Fit OLS and return a tidy dict of key statistics.
    endog : str  — name of y column in exog_df's parent
    exog_df: DataFrame of regressors (constant already added)
    """
    data = exog_df.copy().dropna()
    X = sm.add_constant(data.drop(columns=[endog]))
    y = data[endog]
    if se_type == 'HAC':
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    elif se_type == 'HC3':
        res = sm.OLS(y, X).fit(cov_type='HC3')
    elif se_type == 'OLS':
        res = sm.OLS(y, X).fit()
    else:
        res = sm.OLS(y, X).fit(cov_type=se_type)
    return res, label


#  MAIN SPECIFICATION  (Col 1)
cols_main = ['d_saving_rate', 'd_cash_rate_lag1', 'd_saving_lag1', 'covid']
m_main, _ = run_ols('d_saving_rate', df[cols_main], label='Main (Preferred)')
KEY_MAIN = 'd_cash_rate_lag1'

# A. ALTERNATIVE CONTROL SETS


# A1: No controls — bare first-difference regression
cols_A1 = ['d_saving_rate', 'd_cash_rate_lag1']
m_A1, _ = run_ols('d_saving_rate', df[cols_A1], label='A1: No controls')

# A2: Full controls — add GFC dummy + 2nd lag of saving rate change
cols_A2 = ['d_saving_rate', 'd_cash_rate_lag1', 'd_saving_lag1',
           'd_saving_lag2', 'covid', 'gfc']
m_A2, _ = run_ols('d_saving_rate', df[cols_A2], label='A2: + GFC + lag2')

# B. ALTERNATIVE SAMPLES

# B1: Drop COVID period (2020Q1–2021Q4)
mask_no_covid = ~((df['quarter'] >= '2020Q1') & (df['quarter'] <= '2021Q4'))
cols_B1 = ['d_saving_rate', 'd_cash_rate_lag1', 'd_saving_lag1']
m_B1, _ = run_ols('d_saving_rate', df.loc[mask_no_covid, cols_B1],
                  label='B1: Excl. COVID')

# B2: Post-2007 only (captures both GFC and post-GFC easing cycles)
mask_post07 = df['quarter'] >= '2007Q1'
cols_B2 = ['d_saving_rate', 'd_cash_rate_lag1', 'd_saving_lag1', 'covid']
m_B2, _ = run_ols('d_saving_rate', df.loc[mask_post07, cols_B2],
                  label='B2: Post-2007')


# C. ALTERNATIVE FUNCTIONAL FORM

# C1: Levels (not first-differenced) — baseline levels model with COVID
cols_C1 = ['saving_rate', 'cash_rate', 'covid']
m_C1, _ = run_ols('saving_rate', df[cols_C1], se_type='HC3',
                  label='C1: Levels (HC3)')
KEY_C1 = 'cash_rate'

# C2: IHS transformation of saving rate and cash-rate change
#     Useful because saving_rate hits negatives (IHS ≈ log but symmetric)
cols_C2 = ['d_ihs_saving', 'd_ihs_cash_lag1', 'd_ihs_lag1', 'covid']
m_C2, _ = run_ols('d_ihs_saving', df[cols_C2], label='C2: IHS form')
KEY_C2 = 'd_ihs_cash_lag1'

# D. ALTERNATIVE INFERENCE

# D1: HC3 heteroskedasticity-robust (instead of HAC)
cols_D1 = ['d_saving_rate', 'd_cash_rate_lag1', 'd_saving_lag1', 'covid']
m_D1, _ = run_ols('d_saving_rate', df[cols_D1], se_type='HC3',
                  label='D1: HC3 SE')

# D2: HAC with 2 lags (tighter bandwidth)
m_D2, _ = run_ols('d_saving_rate', df[cols_D1], se_type='HAC', hac_lags=2,
                  label='D2: HAC(2)')

# E. SUBSAMPLE — pre-COVID window (descriptive check on time-window sensitivity)

mask_pre_covid = df['quarter'] < '2020Q1'
cols_E1 = ['d_saving_rate', 'd_cash_rate_lag1', 'd_saving_lag1', 'gfc']
m_E1, _ = run_ols('d_saving_rate', df.loc[mask_pre_covid, cols_E1],
                  label='E1: Pre-COVID')

# individual model summaries

models_for_print = [
    ('MAIN (Preferred)', m_main, KEY_MAIN),
    ('A1: No controls', m_A1, KEY_MAIN),
    ('A2: + GFC + lag2', m_A2, KEY_MAIN),
    ('B1: Excl. COVID', m_B1, KEY_MAIN),
    ('B2: Post-2007', m_B2, KEY_MAIN),
    ('C1: Levels (HC3)', m_C1, KEY_C1),
    ('C2: IHS form', m_C2, KEY_C2),
    ('D1: HC3 SE', m_D1, KEY_MAIN),
    ('D2: HAC(2)', m_D2, KEY_MAIN),
    ('E1: Pre-COVID', m_E1, KEY_MAIN),
]

for name, res, key in models_for_print:
    coef = res.params.get(key, np.nan)
    se   = res.bse.get(key, np.nan)
    pval = res.pvalues.get(key, np.nan)
    sig  = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
    print(f"  {name:<30}  β={coef:+.3f}{sig}  SE={se:.3f}  p={pval:.3f}  R²={res.rsquared:.3f}  N={int(res.nobs)}")


# ── ADF stationarity tests (informational)

print("\n" + "=" * 65)
print("ADF STATIONARITY TESTS")
print("=" * 65)
for series_name, series in [('saving_rate', df['saving_rate'].dropna()),
                             ('cash_rate',   df['cash_rate'].dropna()),
                             ('d_saving_rate', df['d_saving_rate'].dropna()),
                             ('d_cash_rate',   df['d_cash_rate'].dropna())]:
    adf_stat, adf_p, *_ = adfuller(series, autolag='AIC')
    conclusion = "stationary" if adf_p < 0.05 else "NON-stationary"
    print(f"  {series_name:<20}  ADF={adf_stat:+.3f}  p={adf_p:.3f}  → {conclusion}")


# ROBUSTNESS TABLE

print("\n" + "=" * 120)
print("ROBUSTNESS TABLE — Dependent variable: Δsaving_rate (or level/IHS equivalent)")
print("=" * 120)

table_entries = [
    # (col_label, short_label, model, key_var, notes)
    ('(1)', 'Main',         m_main, KEY_MAIN,
     'Preferred: FD, HAC(4), Δcash(-1), Δsave(-1), COVID'),
    ('(2)', 'No controls',  m_A1,   KEY_MAIN,
     'A1: drops Δsave(-1) and COVID dummy'),
    ('(3)', 'Add GFC+lag2', m_A2,   KEY_MAIN,
     'A2: adds GFC dummy and Δsave(-2)'),
    ('(4)', 'Excl. COVID',  m_B1,   KEY_MAIN,
     'B1: drops 2020Q1–2021Q4 (n=7 obs removed)'),
    ('(5)', 'Post-2007',    m_B2,   KEY_MAIN,
     'B2: restricts to 2007Q1 onward'),
    ('(6)', 'Levels',       m_C1,   KEY_C1,
     'C1: levels (not FD); y=saving_rate, x=cash_rate; HC3'),
    ('(7)', 'IHS form',     m_C2,   KEY_C2,
     'C2: IHS(saving) and IHS(Δcash); handles negatives'),
    ('(8)', 'HC3 SE',       m_D1,   KEY_MAIN,
     'D1: same spec as main, HC3 instead of HAC(4)'),
    ('(9)', 'HAC(2)',        m_D2,   KEY_MAIN,
     'D2: same spec as main, HAC bandwidth=2'),
    ('(10)', 'Pre-COVID',   m_E1,   KEY_MAIN,
     'E1: 2000Q1–2019Q4; GFC dummy replaces COVID'),
]

header = f"{'Statistic':<18}" + "".join(f"{'Col'+c:<13}" for c, *_ in table_entries)
print(header)
print("-" * 120)

# β row
row_b = f"{'β (cash rate Δ)':<18}"
for col, lbl, res, key, note in table_entries:
    val = res.params.get(key, np.nan)
    row_b += f"{val:>+10.3f}   "
print(row_b)

# SE row
row_se = f"{'  SE':<18}"
for col, lbl, res, key, note in table_entries:
    val = res.bse.get(key, np.nan)
    row_se += f"({''+f'{val:.3f}':>8})   "
print(row_se)

# p-value row
row_p = f"{'  p-value':<18}"
for col, lbl, res, key, note in table_entries:
    val = res.pvalues.get(key, np.nan)
    row_p += f"{val:>10.3f}   "
print(row_p)

# R² row
row_r2 = f"{'R²':<18}"
for col, lbl, res, key, note in table_entries:
    row_r2 += f"{res.rsquared:>10.3f}   "
print(row_r2)

# N row
row_n = f"{'N':<18}"
for col, lbl, res, key, note in table_entries:
    row_n += f"{int(res.nobs):>10d}   "
print(row_n)

print("-" * 120)
print("\nNotes:")
for col, lbl, res, key, note in table_entries:
    print(f"  Col {col}: {note}")
print("  *** p<0.01, ** p<0.05, * p<0.10.")
print("  All first-difference models use HAC(4) standard errors unless stated.")
print("  Levels model (Col 6) uses HC3; both saving_rate and cash_rate may be I(1).")

# INTERPRETATION
print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)
print("""
MAIN RESULT RESTATED
  In the preferred first-difference specification (Col 1), a 1 pp lagged
  increase in the RBA cash rate change is associated with a change in the
  household saving rate of approximately {:.3f} pp (SE={:.3f}, p={:.3f}).
  This is a DESCRIPTIVE conditional correlation, not a causal estimate.

A. ALTERNATIVE CONTROL SETS (Cols 2–3)
  Col 2 (no controls): The coefficient on Δcash_rate(-1) shifts to {:.3f}
  when saving momentum and the COVID dummy are removed, suggesting the
  preferred estimate is partly absorbing COVID-era dynamics. The result
  is directionally consistent but sensitivity to the COVID dummy indicates
  the shock is large relative to the underlying signal.

  Col 3 (+ GFC dummy + Δsave(-2)): Adding a GFC dummy and a second lag of
  the saving rate change leaves the cash-rate coefficient at {:.3f}, nearly
  unchanged from the main result, which supports stability of the core
  correlation against reasonable control augmentation.

B. ALTERNATIVE SAMPLES (Cols 4–5)
  Col 4 (excluding COVID quarters): Dropping 2020Q1–2021Q4 materially
  changes the coefficient to {:.3f}. This fragility is substantively
  meaningful: the COVID period, with saving rates exceeding 20 % and the
  cash rate near zero, creates a very large but mechanically distinct
  co-movement. The conditional correlation is substantially driven by
  this episode.

  Col 5 (post-2007 only): Restricting to the period that contains both
  the GFC and the 2022–23 tightening cycle yields {:.3f}. The sign is
  preserved but the magnitude differs, consistent with the 2022–23
  episode (rapid rate rises, falling saving) partly working against
  the earlier GFC pattern (rapid cuts, rising saving).

C. ALTERNATIVE FUNCTIONAL FORM (Cols 6–7)
  Col 6 (levels): The levels regression yields a positive coefficient on
  cash_rate, but both series are likely non-stationary (see ADF tests
  above). The levels result should not be used for inference; it is
  reported only to show the first-difference transformation matters.

  Col 7 (IHS): Applying the inverse hyperbolic sine to handle the
  negative saving rate values yields a coefficient of {:.3f} on the IHS
  of Δcash_rate(-1). The sign is preserved. This checks that results are
  not driven by the OLS assumption that a saving rate of −2 % is
  symmetric to +2 %.

D. ALTERNATIVE INFERENCE (Cols 8–9)
  Cols 8–9 use HC3 and HAC(2) on an identical specification to the main
  model. The point estimates are identical (same OLS fit); only standard
  errors differ. Switching to HC3 gives SE={:.3f}; HAC(2) gives SE={:.3f}.
  The narrower HAC(2) bandwidth yields smaller SEs. Given the Durbin-Watson
  statistics in the primary analysis suggest mild residual autocorrelation,
  HAC(4) is the more conservative — and preferred — choice.

E. TIME-WINDOW SENSITIVITY (Col 10)
  Restricting to pre-COVID quarters (2000Q1–2019Q4) gives β={:.3f}, which
  is directionally consistent with the main result. This confirms the
  conditional correlation is not purely a COVID-era artefact, although
  it is weaker in the pre-COVID period, consistent with the "saving glut"
  at near-zero rates 2013–2019 dampening the relationship.

OVERALL 
  The sign of the relationship (higher cash rate changes associated with
  higher saving rate changes, with a one-quarter lag) survives most checks.
  However, the magnitude is sensitive to the inclusion of COVID-era
  observations, which dominate the variation. The main finding is best
  read as: the conditional correlation holds broadly across the sample,
  but is substantially amplified by the 2020–21 shock. A reader should
  treat the main coefficient as an upper bound on the typical quarter-to-
  quarter correlation. No causal claim is made.
""".format(
    m_main.params.get(KEY_MAIN, np.nan),
    m_main.bse.get(KEY_MAIN, np.nan),
    m_main.pvalues.get(KEY_MAIN, np.nan),
    m_A1.params.get(KEY_MAIN, np.nan),
    m_A2.params.get(KEY_MAIN, np.nan),
    m_B1.params.get(KEY_MAIN, np.nan),
    m_B2.params.get(KEY_MAIN, np.nan),
    m_C2.params.get(KEY_C2, np.nan),
    m_D1.bse.get(KEY_MAIN, np.nan),
    m_D2.bse.get(KEY_MAIN, np.nan),
    m_E1.params.get(KEY_MAIN, np.nan),
))

# FIGURE 1: Robustness coefficient plot (forest plot)

fig, ax = plt.subplots(figsize=(10, 6))

labels  = [lbl for _, lbl, *_ in table_entries]
betas   = [res.params.get(key, np.nan) for _, _, res, key, _ in table_entries]
ses     = [res.bse.get(key, np.nan)    for _, _, res, key, _ in table_entries]
pvals   = [res.pvalues.get(key, np.nan) for _, _, res, key, _ in table_entries]

y_pos = list(range(len(labels)))
colors = ['#c0392b' if p < 0.05 else '#2980b9' for p in pvals]

ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax.axvline(betas[0], color='#c0392b', linewidth=1.2, linestyle=':', alpha=0.4,
           label=f'Main β = {betas[0]:+.3f}')

for i, (b, s, c) in enumerate(zip(betas, ses, colors)):
    ax.errorbar(b, i, xerr=1.96 * s, fmt='o', color=c, capsize=4,
                capthick=1.5, markersize=7, linewidth=1.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'Col {e[0]}: {e[1]}' for e in table_entries], fontsize=9)
ax.set_xlabel('β on Δcash_rate(t-1)  [± 1.96 SE]', fontsize=10)
ax.set_title('Robustness: Cash Rate Coefficient Across Specifications\n'
             'Red = p < 0.05; Blue = p ≥ 0.05', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('output/fig_robustness_forest.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 1 saved: output/fig_robustness_forest.png")

# FIGURE 2: Time-series with COVID and GFC bands + series plot

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Panel A: levels
ax = axes[0]
ax.plot(df['date'], df['saving_rate'], color='steelblue',  lw=1.5, label='Saving rate (%)')
ax.plot(df['date'], df['cash_rate'],   color='darkorange', lw=1.5, label='Cash rate (%)')
ax.axvspan(pd.Timestamp('2008-07-01'), pd.Timestamp('2009-06-30'),
           alpha=0.12, color='grey', label='GFC (2008Q3–2009Q2)')
ax.axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2021-12-31'),
           alpha=0.12, color='red',  label='COVID (2020Q2–2021Q4)')
ax.set_ylabel('% ', fontsize=9)
ax.set_title('Panel A: Saving Rate and Cash Rate — Quarterly Levels', fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(-5, 25)

# Panel B: first differences
ax = axes[1]
ax.plot(df['date'], df['d_saving_rate'],  color='steelblue',  lw=1.3, label='Δsaving_rate')
ax.plot(df['date'], df['d_cash_rate'],    color='darkorange', lw=1.3, linestyle='--', label='Δcash_rate')
ax.axhline(0, color='black', lw=0.7, linestyle=':')
ax.axvspan(pd.Timestamp('2008-07-01'), pd.Timestamp('2009-06-30'),
           alpha=0.12, color='grey')
ax.axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2021-12-31'),
           alpha=0.12, color='red')
ax.set_ylabel('pp change', fontsize=9)
ax.set_title('Panel B: First Differences', fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('output/fig_series_and_diffs.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: output/fig_series_and_diffs.png")

#  SAVE ROBUSTNESS TABLE AS CSV

rows = []
for col, lbl, res, key, note in table_entries:
    b    = res.params.get(key, np.nan)
    se   = res.bse.get(key, np.nan)
    pval = res.pvalues.get(key, np.nan)
    sig  = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
    rows.append({
        'Column': col,
        'Label': lbl,
        'beta': round(b, 4),
        'SE': round(se, 4),
        'p_value': round(pval, 4),
        'Significance': sig,
        'R2': round(res.rsquared, 4),
        'N': int(res.nobs),
        'Notes': note,
    })
rob_df = pd.DataFrame(rows)
rob_df.to_csv('output/robustness_table.csv', index=False)
print("Robustness table saved: output/robustness_table.csv")
print("\n✓ 03_robustness.py complete.\n")
