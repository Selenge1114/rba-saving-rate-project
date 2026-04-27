# RBA Cash Rate and Household Saving Rate

**Research Question:** Does a change in the RBA cash rate affect the household saving rate in Australia between 2011 and 2025?

**Analytical Ambition:** Causal — we estimate an ADL(1,1) model using the RBA's institutional independence as the basis for conditional exogeneity.

---

## Repository Structure

- README.md: Project overview and instructions
- analysis.py: Primary analysis file — runs end-to-end on clean data
- data/raw: Contains a2-data.csv, Table_34.csv, and 5206034_q.csv
- data/clean: Contains final_dataset.csv and codebook.md
- code: Contains 01_clean_data.py, 02_analysis.py, and 03_eda.ipynb
- output: Folder for generated figures and regression table

---

## How to Run the Project

### Step 1 — Manual data downloads

**RBA Cash Rate — already in repo:**
The file data/raw/a2-data.csv is included. To refresh it, download Table A2 as CSV from https://www.rba.gov.au/statistics/tables/ and save as data/raw/a2-data.csv

**ABS Quarterly Saving Ratio:**
1. Go to https://www.abs.gov.au/statistics/economy/national-accounts/australian-national-accounts-national-income-expenditure-and-product/latest-release
2. Scroll to Data downloads
3. Download the quarterly household income account table
4. Save as data/raw/5206034_q.csv

**ABS Annual Saving Ratio — already in repo:**
data/raw/Table_34.csv is included as a fallback. The cleaning script uses this automatically if 5206034_q.csv is not present.

---

### Step 2 — Install required packages

```bash
pip3 install pandas numpy matplotlib seaborn scipy statsmodels jupyter
```

### Step 3 — Run scripts in order

```bash
python3 code/01_clean_data.py
python3 analysis.py
jupyter notebook code/03_eda.ipynb
```

---

## Econometric Specification

Model: ADL(1,1) — SavingRate_t = a + b1*CashRate_t + b2*CashRate_(t-1) + b3*SavingRate_(t-1) + b4*COVID_t + b5*t + e_t

Error structure: Newey-West HAC standard errors (4 lags)

Identification: The RBA sets the cash rate via its inflation and employment mandate. Conditional on lagged saving, the COVID dummy, and a time trend, cash rate movements are treated as approximately exogenous.

---

## Data Sources

The cash rate data comes from the RBA Table A2, which records each policy decision from 1990 to the present. The household saving ratio comes from the ABS catalogue 5206.0, which provides quarterly estimates from 1960 to the present.

---

## Software

Python 3.10 or higher is required. The following packages are used: pandas for data manipulation, numpy for numerical operations, matplotlib and seaborn for visualisation, scipy for correlation tests, statsmodels for OLS regression and ADF tests, and jupyter for the EDA notebook.