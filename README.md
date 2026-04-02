##   Research Question

Did reductions in the RBA cash rate reduce the household saving rate in Australia?

---

##   Repository Structure

data/
  raw/ – original datasets (cash rate, saving rate)
  clean/ – final merged dataset

code/
  01_clean_data.py – data cleaning and merging
  02_analysis.py – regression analysis

output/ – optional outputs

README.md

---

##  How to Run This Project 

### Step 1: Clone repository

git clone <your-repo-link>
cd rba-saving-rate-project

---

### Step 2: Install required packages

python3 -m pip install pandas statsmodels matplotlib

---

### Step 3: Prepare raw data 

Download data from:

* Reserve Bank of Australia (cash rate)
* Australian Bureau of Statistics (saving rate)

Then:

* Keep only date + variable columns
* Remove extra headers and notes
* Save as CSV files:

  * Cash_rate(2011-2025).csv
  * Saving_rate(2011-2025).csv

Place them into:
data/raw/

---

### Step 4: Run scripts in order

python3 code/01_clean_data.py
python3 code/02_analysis.py

---

##  Output

Final dataset:
data/clean/final_dataset.csv

Regression results are displayed in the terminal.

---

##  Data Description (Codebook)

* date: Time period
* saving_rate: Household saving ratio (%)
* cash_rate: RBA interest rate (%)
* quarter: Quarterly time period

---

##  Software Used

* Python 3
* pandas
* statsmodels
* matplotlib

---

##  notes

* Raw data requires manual cleaning due to formatting issues
* Cash rate is converted from daily to quarterly averages
* Saving rate is already quarterly

---
