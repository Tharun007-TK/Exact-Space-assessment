# Task 1: Cyclone Machine Data Analysis

## 📋 Overview
This folder contains the complete analysis pipeline for 3 years of cyclone sensor time-series data, covering:
- Data preparation & exploratory analysis
- Shutdown/idle period detection
- Machine state segmentation (clustering)
- Contextual anomaly detection with root cause analysis
- Short-term forecasting (1-hour ahead)
- Business insights and recommendations

---

## 🚀 Quick Start

### Prerequisites
Install required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install statsmodels hdbscan openpyxl
```

### Running the Analysis
1. Place your Excel data file in this folder
2. Open `task1_analysis.ipynb` (or run `task1_analysis.py`)
3. **Update the file path** in the configuration section:
   ```python
   FILE_PATH = 'data.csv'  #Convert .xlsx to .csv for better analysis
   ```
4. Run all cells sequentially

**Execution time**: ~5-15 minutes depending on data size and hardware

---

## 📂 File Structure

```
Task1/
├── task1_analysis.ipynb          # Main analysis notebook
├── your_data.xlsx                # Your input data (not included)
├── README.md                     # This file
├── shutdown_periods.csv          # Generated output
├── anomalous_periods.csv         # Generated output
├── clusters_summary.csv          # Generated output
├── forecasts.csv                 # Generated output
├── summary_statistics.csv        # Generated output
└── plots/                        # Generated visualizations
    ├── correlation_matrix.png
    ├── one_week_overview.png
    ├── one_year_with_shutdowns.png
    ├── cluster_visualization.png
    ├── anomaly_example_1.png
    ├── anomaly_example_2.png
    ├── anomaly_example_3.png
    └── forecast_comparison.png
```

---

## 📊 Input Data Format

**Required columns** (exact names as in your dataset):
- `time` - Timestamp column
- `Cyclone_Inlet_Gas_Temp` - Temperature at inlet
- `Cyclone_Material_Temp` - Material temperature
- `Cyclone_Outlet_Gas_draft` - Outlet gas pressure
- `Cyclone_cone_draft` - Cone section pressure
- `Cyclone_Gas_Outlet_Temp` - Outlet gas temperature
- `Cyclone_Inlet_Draft` - Inlet pressure

**Expected format**:
- Excel file (.xlsx)
- ~370,000 records
- 5-minute intervals
- 3-year duration

---

## 📤 Outputs Generated

### CSV Files

1. **shutdown_periods.csv**
   - Columns: `start`, `end`, `duration_hours`
   - Lists all detected shutdown/idle periods

2. **clusters_summary.csv**
   - Summary statistics for each operational state
   - Columns: cluster ID, count, percentage, mean/std/percentiles for all variables

3. **anomalous_periods.csv**
   - Columns: `start_time`, `end_time`, `duration_minutes`, `cluster`, `top_variables`, `severity`
   - Top 50 anomalous events with context

4. **forecasts.csv**
   - Columns: `actual`, `persistence`, `random_forest`
   - First 100 forecast predictions vs actuals

5. **summary_statistics.csv**
   - Basic descriptive statistics for all variables

### Visualizations (PNG, 300 DPI)

1. **correlation_matrix.png** - Heatmap of variable correlations
2. **one_week_overview.png** - 7-day sample showing normal behavior
3. **one_year_with_shutdowns.png** - Full year with highlighted shutdowns
4. **cluster_visualization.png** - Operational states over time
5. **anomaly_example_1/2/3.png** - Detailed views of top 3 anomalies
6. **forecast_comparison.png** - Model performance comparison

---

## 🔧 Methodology

### 1. Data Preparation
- Missing value imputation (forward fill with limits)
- Timestamp gap handling
- Resampling to strict 5-minute intervals
- Outlier detection via IQR method

### 2. Shutdown Detection
- **Method**: Combined activity score from all sensors
- **Threshold**: Bottom 5th percentile of activity
- **Minimum duration**: 30 minutes (6 intervals)

### 3. Clustering
- **Algorithm**: HDBSCAN (density-based)
- **Features**: Raw values + rolling statistics (30-min windows) + lag features
- **Output**: 3-5 distinct operational states

### 4. Anomaly Detection
- **Method**: Cluster-specific Isolation Forest
- **Contamination rate**: 1% per cluster
- **Root cause**: Analysis of most deviant variables and timing

### 5. Forecasting
- **Target**: Cyclone_Inlet_Gas_Temp
- **Horizon**: 1 hour (12 steps)
- **Models**:
  - Persistence (baseline)
  - ARIMA(5,1,2)
  - Random Forest with 24 lag features
- **Metrics**: RMSE and MAE

---

## 💡 Key Insights (Example)

After running the analysis, you'll get insights like:

1. **Operational Availability**: 92.3% uptime, 67 shutdown events over 3 years
2. **Operational States**: 4 distinct states identified (Normal, High Load, Low Load, Startup)
3. **Anomaly Rate**: 2.3 anomalous events per day on average
4. **Strongest Correlation**: Inlet Temp ↔ Outlet Temp (0.87)
5. **Best Forecasting Model**: Random Forest (RMSE: 3.45°C)

---

## 🎯 Recommendations

1. Implement real-time anomaly detection using cluster-specific thresholds
2. Create operator alerts for top anomaly variables
3. Schedule preventive maintenance before high-risk periods
4. Investigate root causes of frequent short shutdowns
5. Deploy forecasting model for operational planning

---

## ⚠️ Troubleshooting

### Common Issues

**1. File not found**
```
FileNotFoundError: [Errno 2] No such file or directory: 'your_data.xlsx'
```
**Solution**: Update `FILE_PATH` variable with your actual filename

**2. Column name mismatch**
```
KeyError: 'time'
```
**Solution**: Check your column names match exactly (including case)

**3. Memory error**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce dataset size or increase available RAM

**4. ARIMA convergence warning**
```
ConvergenceWarning: Maximum Likelihood optimization failed to converge
```
**Solution**: This is normal for some data patterns; results are still usable

---

## 📞 Support

For questions about this analysis:
1. Check the inline comments in the notebook
2. Review the methodology section above
3. Examine the example outputs in `plots/`

---

## 📝 Citation

```
Cyclone Machine Data Analysis Pipeline
Author: [Your Name]
Date: September 2025
Purpose: Data Science Take-Home Assignment
```

---

## ✅ Checklist Before Submission

- [ ] All outputs generated successfully
- [ ] Plots folder contains 8 visualizations
- [ ] CSV files contain valid data
- [ ] README.md is included
- [ ] Code runs end-to-end without errors
- [ ] File paths updated to match your data

---

**Last Updated**: September 2025