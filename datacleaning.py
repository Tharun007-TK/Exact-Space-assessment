# task1_analysis.ipynb

# ------------------------
# 1. Imports & Setup
# ------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plot style
plt.style.use("seaborn-v0_8")
sns.set(rc={'figure.figsize':(12,6)})

# ------------------------
# 2. Load Dataset
# ------------------------
df = pd.read_csv("data.csv")

print("Original Shape:", df.shape)
print("\nNull counts:\n", df.isnull().sum())

# ------------------------
# 3. Data Cleaning
# ------------------------

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Convert all sensor columns to float
sensor_cols = [c for c in df.columns if c != 'time']
for c in sensor_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Handle missing values (forward fill, then backfill if still missing)
df[sensor_cols] = df[sensor_cols].ffill().bfill()

# Set time as index and handle duplicates
df = df.set_index("time").sort_index()

# Remove duplicate timestamps by keeping the first occurrence
df = df[~df.index.duplicated(keep='first')]

# Reindex to strict 5-min interval
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
df = df.reindex(full_index)
df.index.name = "time"

# Fill any new gaps (linear interpolation)
df = df.interpolate(method="linear")

print("Cleaned Shape:", df.shape)

# ------------------------
# 4. Summary Statistics
# ------------------------
print("\nSummary Stats:\n", df.describe().T)

# Correlation matrix
corr = df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Cyclone Variables")
plt.show()

# ------------------------
# 5. Representative Plots
# ------------------------

# Plot one week
sample_week = df.loc["2017-01-01":"2017-01-07"]

fig, ax = plt.subplots()
sample_week[['Cyclone_Inlet_Gas_Temp','Cyclone_Gas_Outlet_Temp']].plot(ax=ax)
ax.set_title("Inlet vs Outlet Gas Temp (1 Week Sample)")
plt.show()

# Plot one year (downsampled to daily mean for clarity)
sample_year = df.loc["2017"].resample("1D").mean()

fig, ax = plt.subplots()
sample_year[['Cyclone_Inlet_Gas_Temp','Cyclone_Gas_Outlet_Temp']].plot(ax=ax)
ax.set_title("Inlet vs Outlet Gas Temp (2017 - Daily Avg)")
plt.show()
