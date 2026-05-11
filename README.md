# RBA Cash Rate and Household Saving Rate

Analysis of the conditional correlation between the RBA cash rate and
Australian household saving behaviour using quarterly data (2000Q1–2025Q4).

---

## Repository Structure

```
data/
  raw/          – original datasets (not tracked; see Step 3 below)
  clean/        – merged quarterly dataset (final_dataset.csv)

code/
  01_clean_data.py   – data cleaning and merging
  02_analysis.py     – primary econometric analysis (4 OLS models)
  03_robustness.py   – robustness analysis (10 specifications)

output/
  fig_actual_vs_fitted.png      – Model 4 actual vs fitted (from 02)
  fig_robustness_forest.png     – coefficient forest plot (from 03)
  fig_series_and_diffs.png      – levels and first-difference series (from 03)
  robustness_table.csv          – machine-readable robustness table (from 03)

README.md
```

---

## Main Result

**Preferred specification (Model 4 / Column 1 of robustness table):**

```
Δsaving_rate(t) = α + β·Δcash_rate(t-1) + γ·Δsaving_rate(t-1) + δ·COVID + ε
```

Estimated by OLS with HAC standard errors (Newey-West, 4 lags).

Declaration: DESCRIPTIVE. The coefficient β captures the conditional
correlation between a one-quarter lagged change in the RBA cash rate and the
contemporaneous quarterly change in the household saving rate.  No causal
claim is made.

---

## How to Reproduce the Full Pipeline

### 1. Clone the repository

```bash
git clone https://github.com/Selenge1114/rba-saving-rate-project.git
cd rba-saving-rate-project
```

### 2. Install dependencies

```bash
pip install pandas numpy statsmodels matplotlib
```

Python 3.8+ required.

### 3. Prepare raw data

Download the following files and place them in `data/raw/`:

| File | Source | Notes |
|------|--------|-------|
| `a2-data.csv` | [RBA Table A2](https://www.rba.gov.au/statistics/tables/) | Cash rate decisions |
| `5206034_q.csv` | [ABS 5206.0](https://www.abs.gov.au/statistics/economy/national-accounts/australian-national-accounts-national-income-expenditure-and-product/latest-release) | Quarterly household income account |

Raw data is not tracked in the repository due to ABS/RBA licensing.
Remove metadata rows and save as plain CSV before running Step 4.

### 4. Run scripts in order

```bash
python3 code/01_clean_data.py    # produces data/clean/final_dataset.csv
python3 code/02_analysis.py      # primary analysis + fig_actual_vs_fitted.png
python3 code/03_robustness.py    # robustness checks + forest plot + table CSV
```

## Data

### data/raw/

Manually cleaned source files from RBA and ABS (not tracked).

### data/clean/final_dataset.csv

Final merged quarterly dataset used for all analysis.

| Column | Description |
|--------|-------------|
| `date` | Quarter start date (YYYY-MM-DD) |
| `quarter` | Period label (e.g. 2008Q3) |
| `cash_rate` | RBA target cash rate, quarterly average (%) |
| `saving_rate` | Household net saving / gross disposable income × 100 (%) |

---

## Code

| Script | Purpose |
|--------|---------|
| `01_clean_data.py` | Merges RBA A2 daily decisions (forward-filled to quarterly averages) with ABS quarterly saving ratios |
| `02_analysis.py` | Runs Models 1–4: levels (no controls), levels + COVID dummy, first differences, first differences + COVID dummy + lagged DV (preferred) |
| `03_robustness.py` | Ten-column robustness table: alternative controls, samples, functional forms, inference methods, and time windows. Saves forest plot and CSV table. |

---

## Robustness Summary

Ten specifications are compared against the main result (see `code/03_robustness.py` and `output/robustness_table.csv`):

| Col | Check | Category |
|-----|-------|----------|
| (1) | Main specification | — |
| (2) | No controls (drop Δsave lag, COVID) | Alternative controls |
| (3) | Add GFC dummy + Δsave lag 2 | Alternative controls |
| (4) | Exclude COVID quarters (2020Q1–2021Q4) | Alternative sample |
| (5) | Post-2007 only | Alternative sample |
| (6) | Levels (not first-differenced), HC3 | Alternative functional form |
| (7) | IHS of saving rate and cash rate change | Alternative functional form |
| (8) | HC3 standard errors | Alternative inference |
| (9) | HAC(2) standard errors | Alternative inference |
| (10) | Pre-COVID window (2000Q1–2019Q4) | Time-window sensitivity |

Key finding: the sign of the cash-rate association is broadly preserved,
but the magnitude is sensitive to COVID-era observations, which dominate
the variation in the full-sample estimates.

---

## About

Analysis by Selenge Bayarkhuu.
Data: Reserve Bank of Australia (RBA) and Australian Bureau of Statistics (ABS).
