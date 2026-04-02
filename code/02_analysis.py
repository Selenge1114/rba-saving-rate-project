import pandas as pd
import statsmodels.api as sm

# -----------------------------
# 1. Load final dataset
# -----------------------------
data = pd.read_csv("data/clean/final_dataset.csv")

# -----------------------------
# 2. Define variables
# -----------------------------
X = data["cash_rate"]   # independent variable
y = data["saving_rate"] # dependent variable

# Add constant (intercept)
X = sm.add_constant(X)

# -----------------------------
# 3. Run regression
# -----------------------------
model = sm.OLS(y, X).fit()

# -----------------------------
# 4. Print results
# -----------------------------
print(model.summary())
import matplotlib.pyplot as plt

plt.scatter(data["cash_rate"], data["saving_rate"])
plt.xlabel("Cash Rate")
plt.ylabel("Saving Rate")
plt.title("Cash Rate vs Saving Rate")
plt.show()
