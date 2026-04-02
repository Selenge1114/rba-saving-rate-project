import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("data/clean/final_dataset.csv")

X = data["cash_rate"]   # independent variable
y = data["saving_rate"] # dependent variable

# constant intercept
X = sm.add_constant(X)

#regression model
model = sm.OLS(y, X).fit()

print(model.summary())
import matplotlib.pyplot as plt

plt.scatter(data["cash_rate"], data["saving_rate"])
plt.xlabel("Cash Rate")
plt.ylabel("Saving Rate")
plt.title("Cash Rate vs Saving Rate")
plt.show()
