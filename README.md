# Avocado (Cost of living)
Avocado — AI-powered cost of living predictor for any city.  Avocado uses machine learning to compare cities and instantly estimate the cost of living, helping users understand expenses like housing, food, transport, and lifestyle before moving.
Data analysis:

Understanding the dataset
Transpose to move rows to columns and columns to rows
Created a “median” column to calculate the median of the values in each row
Renaming the column values to one word features:
EDA:
![Cost Distribution](https://github.com/Vidhikalal/Avocado/blob/main/Cost%20Distribution.png)
![Outlier Analysis for Cost of living Score](https://github.com/Vidhikalal/Avocado/blob/main/Oulier%20Analysis.png)
Most cities fall between ~4 and ~8
The whiskers span roughly from ~3 to ~12
This means typical cities range from:
Cheap cities (~3–5)
Moderate cities (~5–7)
High-cost cities (~7–12)
We have one true outlier (can be New York, San Francisco) due to the high median accounted by the high cost of housing, rent, and utilities
The ML model should be able to work well with extreme outlier data since it can overfit or over weigh other features
The median (center line) is around ~6
This shows the global “middle cost” city is near a score of 6.
Hence, we can say the data is right-skewed
We can use log-transform may help stabilize variance
X_scaled = (X - median) / IQR (Inter Quartile range) instead of z = (x - mean) / std
IQR = Q3 − Q1 (middle 50% of data)






