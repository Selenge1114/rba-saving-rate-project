# RBA Cash Rate and Household Saving Rate in Australia

**Research Question:** Did reductions in the RBA cash rate reduce the household saving rate in Australia between 2000 and 2025?

---

## Repository Structure

```
rba-saving-rate-project/
├── README.md
├── data/
│   ├── raw/
│   │   ├── a2-data.csv          ← RBA cash rate (decision dates)
│   │   ├── Table_34.csv         ← ABS annual saving ratio (fallback)
│   │   └── 5206034_q.csv        ← ABS quarterly saving ratio [PREFERRED - obtain separately]
│   └── clean/
│       ├── final_dataset.csv    ← Merged analysis-ready dataset
│       └── codebook.md          ← Variable descriptions
├── code/
│   ├── 01_clean_data.py         ← Cleans and merges raw data
│   ├── 02_analysis.py           ← Regression analysis
│   └── 03_eda.ipynb             ← Exploratory Data Analysis notebook
└── output/                      ← Generated figures
```

---

## How to Run the Project From Scratch

### Step 1 — Manual data downloads

**RBA Cash Rate (Table A2) — already in repo:**
The file `data/raw/a2-data.csv` is included. If you need to refresh it:
1. Go to: https://www.rba.gov.au/statistics/tables/
2. Find "A2 – Changes in Monetary Policy and Administered Rates"
3. Click CSV download → save as `data/raw/a2-data.csv`

**ABS Quarterly Saving Ratio (PREFERRED — gives ~102 quarterly obs):**
1. Go to: https://www.abs.gov.au/statistics/economy/national-accounts/australian-national-accounts-national-income-expenditure-and-product/latest-release
2. Scroll to **"Data downloads"**
3. Find the table containing **"Household saving ratio"** at quarterly frequency
   (look for "Table 20 – Household Income Account" or search for "saving ratio")
4. Download CSV → save as `data/raw/5206034_q.csv`

> **Why the quarterly file is not in the repo:** It is large, updated each quarter by the ABS,
> and freely available from the official source above.

**ABS Annual Saving Ratio (FALLBACK — already in repo):**
`data/raw/Table_34.csv` is included. This gives 26 annual observations (one per financial year).
The cleaning script automatically uses this if `5206034_q.csv` is not present.

---

### Step 2 — Install required packages

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels jupyter
```

Python 3.8+ required.

### Step 3 — Run scripts in order

```bash
# 1. Produce data/clean/final_dataset.csv
python code/01_clean_data.py

# 2. Run regression analysis
python code/02_analysis.py

# 3. Open EDA notebook
jupyter notebook code/03_eda.ipynb
```

---

## Data Sources

| Dataset | Source | Frequency | Coverage |
|---------|--------|-----------|----------|
| Cash Rate Target | RBA Table A2 | As announced (daily forward-fill → quarterly avg) | 1990–present |
| Household Saving Ratio | ABS Cat. 5206.0 | Quarterly (preferred) / Annual (fallback) | 1960–present |

---

## Software

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Visualisation |
| scipy | Correlation tests |
| statsmodels | OLS regression, ADF tests |
| jupyter | EDA notebook |
