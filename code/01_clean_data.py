import pandas as pd

saving = pd.read_csv("data/raw/Saving_rate(2011-2025).csv")
cash = pd.read_csv("data/raw/Cash_rate(2011-2025).csv")

saving.columns = ["date", "saving_rate"]
cash.columns = ["date", "cash_rate"]

# Saving is like "Mar-11"
saving["date"] = pd.to_datetime(saving["date"], format="%b-%y")

cash["date"] = pd.to_datetime(cash["date"])

# 4. Convert to quarterly

saving["quarter"] = saving["date"].dt.to_period("Q")

cash["quarter"] = cash["date"].dt.to_period("Q")
cash_q = cash.groupby("quarter")["cash_rate"].mean().reset_index()

data = pd.merge(saving, cash_q, on="quarter", how="inner")


data.to_csv("data/clean/final_dataset.csv", index=False)


print("\nFinal dataset preview:")
print(data.head())

print("\nColumns:")
print(data.columns)





