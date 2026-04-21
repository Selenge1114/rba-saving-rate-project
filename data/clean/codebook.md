# Data Codebook — `final_dataset.csv`

## Overview

- **Unit of observation:** Australian economy, one row per calendar quarter
- **Frequency:** Quarterly
- **Coverage:** 2000 Q1 to 2025 Q1
- **Number of observations:** ~101 quarters

---

## Variables

The analysis utilizes two primary data series, both of which have been processed into a quarterly format for consistency.

The cash rate is measured as a percentage per annum. It represents the RBA cash rate target, calculated as a quarterly average of daily forward-filled values based on A2 decision dates, with data sourced from RBA Table A2.

The saving rate is measured as a percentage of household disposable income. This reflects the household saving ratio and is seasonally adjusted, with data sourced from ABS Cat. 5206.0, Table 34.

Two time identifiers are used: date, representing the first day of the quarter (for example, 2000-01-01 for Q1 2000), and quarter, which serves as the calendar quarter identifier. Both are derived from the primary sources to align the datasets.

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
