# RBA Cash Rate and Household Saving Rate

**Research Question:** Did reductions in the RBA cash rate reduce the household saving rate in Australia between 2000 and 2025?

---

## Repository Structure

- README.md: Project overview and instructions
- data/raw: Contains a2-data.csv, Table_34.csv, and 5206034_q.csv
- data/clean: Contains final_dataset.csv and codebook.md
- code: Contains 01_clean_data.py, 02_analysis.py, and 03_eda.ipynb
- output: Folder for generated figures

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
python3 code/02_analysis.py
jupyter notebook code/03_eda.ipynb
```

---

## Data Sources

The cash rate data comes from the RBA Table A2, which records each policy decision from 1990 to the present. The household saving ratio comes from the ABS catalogue 5206.0, which provides quarterly estimates from 1960 to the present.

---

## Software

Python 3.10 or higher is required. The following packages are used: pandas for data manipulation, numpy for numerical operations, matplotlib and seaborn for visualisation, scipy for correlation tests, statsmodels for OLS regression and ADF tests, and jupyter for the EDA notebook.