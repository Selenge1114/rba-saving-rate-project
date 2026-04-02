import pandas as pd

# -----------------------------
# 1. Load data
# -----------------------------
saving = pd.read_csv("data/raw/Saving_rate(2011-2025).csv")
cash = pd.read_csv("data/raw/Cash_rate(2011-2025).csv")

# -----------------------------
# 2. Convert column names
# -----------------------------
saving.columns = ["date", "saving_rate"]
cash.columns = ["date", "cash_rate"]

# -----------------------------
# 3. Convert date formats
# -----------------------------
# Saving is like "Mar-11"
saving["date"] = pd.to_datetime(saving["date"], format="%b-%y")

# Cash is daily → normal conversion
cash["date"] = pd.to_datetime(cash["date"])

# -----------------------------
# 4. Convert to quarterly
# -----------------------------
saving["quarter"] = saving["date"].dt.to_period("Q")

cash["quarter"] = cash["date"].dt.to_period("Q")
cash_q = cash.groupby("quarter")["cash_rate"].mean().reset_index()

# -----------------------------
# 5. Merge datasets
# -----------------------------
data = pd.merge(saving, cash_q, on="quarter", how="inner")

# -----------------------------
# 6. Save final dataset
# -----------------------------
data.to_csv("data/clean/final_dataset.csv", index=False)

# -----------------------------
# 7. Preview
# -----------------------------
print("\nFinal dataset preview:")
print(data.head())

print("\nColumns:")
print(data.columns)





