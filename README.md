# RBA Cash Rate and Household Saving Rate

##  Repository Structure

data/
  raw/ – original datasets
  clean/ – cleaned and merged dataset

code/
  01_clean_data.py – data cleaning and merging
  02_analysis.py – regression analysis

output/ – optional outputs

README.md

---

## ▶ How to Run the Project 

### 1. Clone repository

git clone <https://github.com/Selenge1114/rba-saving-rate-project.git>
cd rba-saving-rate-project

---

### 2. Install required software

Install required packages:

python3 -m pip install pandas statsmodels matplotlib

---

### 3. Prepare raw data 

Download data from:

* Reserve Bank of Australia (cash rate)
* Australian Bureau of Statistics (saving rate)

Clean the data manually in Excel:

* Remove titles, notes, and metadata
* Keep only relevant columns: date and variable
* Save as CSV files

  * Cash_rate(2011-2025).csv
  * Saving_rate(2011-2025).csv

Place the files in:
data/raw/

Raw data is not included due to formatting issues and manual preprocessing requirements.

---

### 4. Run scripts in order

python3 code/01_clean_data.py
python3 code/02_analysis.py

---

##  Data

### data/raw/

Contains manually cleaned raw datasets obtained from external sources.

### data/clean/

Contains the final merged dataset used for analysis:

data/clean/final_dataset.csv

---

##  Codebook

* date: Time period
* saving_rate: Household saving ratio (%)
* cash_rate: RBA policy interest rate (%)
* quarter: Quarterly time index

---

##  Code

All scripts required to reproduce the dataset are located in the `code/` folder.

Running the scripts in the specified order will generate the cleaned dataset and analysis results.

---
