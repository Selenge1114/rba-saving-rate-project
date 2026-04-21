# Data Codebook — `final_dataset.csv`

## Overview

This dataset is the analysis-ready merged file used for all econometric analysis.
It was produced by `code/01_clean_data.py` from two raw data sources.

- **Unit of observation:** Australian economy, one row per calendar quarter
- **Frequency:** Quarterly
- **Coverage:** 2000 Q1 to 2025 Q1
- **Number of observations:** ~101 quarters

---

## Variables

| Variable | Type | Unit | Description | Source |
|----------|------|------|-------------|--------|
| `date` | Date | YYYY-MM-DD | First day of the quarter (e.g., 2000-01-01 = Q1 2000) | Derived |
| `quarter` | Period | e.g., 2000Q1 | Calendar quarter identifier | Derived |
| `cash_rate` | Numeric | % per annum | RBA cash rate target, quarterly average of daily forward-filled values from A2 decision dates | RBA Table A2 |
| `saving_rate` | Numeric | % of household disposable income | Household saving ratio, seasonally adjusted | ABS Cat. 5206.0, Table 34 |

---

## Construction Notes

### `cash_rate`
The RBA's A2 table records the cash rate target only on dates when it changes ("as announced").
To obtain a quarterly series, the daily rate was reconstructed by:
1. Forward-filling the most recently announced rate to every calendar day
2. Averaging all daily values within each calendar quarter

This means `cash_rate` for a given quarter reflects the average policy rate that was in
effect throughout that quarter, weighted by calendar days.

### `saving_rate`
The ABS household saving ratio is defined as:
> Gross saving as a percentage of gross household disposable income (plus adjustment for
> change in net equity of households in pension funds).

The series is seasonally adjusted. It is taken directly from ABS Table 34 with no further
transformation. Values below zero indicate that households were on average dissaving
(consuming more than their disposable income, financed by drawing down assets or borrowing).

---

## Key Data Features to Note

- **Pre-GFC (2000–2007):** Saving ratio was negative for much of this period — households were
  net dissavers. This is important context for interpreting regression results.
- **GFC (2008–2009):** Saving ratio spiked sharply as households became more cautious, coinciding
  with aggressive RBA rate cuts — a counter-example to simple theory.
- **COVID-19 (2020 Q2–2021 Q4):** Saving ratio reached its highest recorded values (~20%) while
  the cash rate was at its effective lower bound (0.10%). This period requires a dummy variable
  in regression analysis to avoid severely biasing estimates.
- **Rate hiking cycle (2022 Q2 onwards):** The RBA raised rates from 0.10% to 4.35% over
  approximately 12 months — providing useful variation for testing the reverse direction.
