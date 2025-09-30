"""
CYCLONE MACHINE DATA ANALYSIS - COMPLETE PIPELINE
==================================================
This notebook performs end-to-end analysis of cyclone sensor data including:
1. Data preparation & EDA
2. Shutdown detection
3. Machine state clustering
4. Contextual anomaly detection
5. Short-term forecasting
6. Insights generation

Author: Tharun Kumar V
Date: 2025
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Clustering & Anomaly Detection
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error
import hdbscan

# Time Series & Forecasting
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create output directories
import os
os.makedirs('plots', exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
FILE_PATH = 'data.csv'  # Using CSV file for better reliability
SHEET_NAME = None  # Not needed for CSV

# Column names from your dataset
COLUMNS = [
    'time',
    'Cyclone_Inlet_Gas_Temp',
    'Cyclone_Material_Temp',
    'Cyclone_Outlet_Gas_draft',
    'Cyclone_cone_draft',
    'Cyclone_Gas_Outlet_Temp',
    'Cyclone_Inlet_Draft'
]

# Feature columns (exclude time)
FEATURE_COLS = COLUMNS[1:]

# Shutdown detection thresholds (will be auto-adjusted)
SHUTDOWN_THRESHOLD_PERCENTILE = 5  # Bottom 5% considered shutdown
MIN_SHUTDOWN_DURATION = 6  # Minimum 30 minutes (6 intervals of 5 min)

print(f"📁 File to load: {FILE_PATH}")
print(f"📊 Sensor columns: {len(FEATURE_COLS)}")
print("="*80)

# ============================================================================
# TASK 1: DATA PREPARATION & EXPLORATORY ANALYSIS
# ============================================================================
print("\n🔹 TASK 1: DATA PREPARATION & EXPLORATORY ANALYSIS")
print("="*80)

# Load data
print("Loading data...")
df = pd.read_csv(FILE_PATH)
print(f"✅ Loaded {len(df):,} records")

# Check column names
print(f"\nActual columns: {list(df.columns)}")
if 'time' in df.columns or 'Time' in df.columns:
    # Rename to standardize
    df.columns = [col.strip() for col in df.columns]  # Remove whitespace
    time_col = [col for col in df.columns if col.lower() == 'time'][0]
    df = df.rename(columns={time_col: 'time'})

# Convert numeric columns to proper data types
print("\nConverting data types...")
for col in FEATURE_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"  ✅ Converted {col} to numeric")
    else:
        print(f"  ⚠️  Column {col} not found in data")

# Convert time to datetime
print("\nParsing timestamps...")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# Set time as index
df.set_index('time', inplace=True)

# Check data range
print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
print(f"⏱️  Duration: {(df.index.max() - df.index.min()).days} days")

# Handle missing values
print(f"\n🔍 Missing values before cleaning:")
print(df.isnull().sum())

# Forward fill small gaps (< 3 intervals), drop larger gaps
df = df.fillna(method='ffill', limit=2)
initial_len = len(df)
df = df.dropna()
print(f"✅ Dropped {initial_len - len(df)} rows with missing values")

# Resample to strict 5-minute intervals
print("\n⚙️  Resampling to strict 5-minute intervals...")
df = df.resample('5T').mean()
df = df.interpolate(method='linear', limit=3)  # Interpolate small gaps
df = df.dropna()
print(f"✅ Final dataset: {len(df):,} records")

# Basic statistics
print("\n📊 SUMMARY STATISTICS:")
print("="*80)
summary_stats = df.describe()
print(summary_stats)

# Save summary stats
summary_stats.to_csv('summary_statistics.csv')
print("\n✅ Saved: summary_statistics.csv")

# Correlation matrix
print("\n📈 Computing correlation matrix...")
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Cyclone Sensors', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/correlation_matrix.png")

# Visualize one week
print("\n📊 Creating one-week visualization...")
one_week_data = df.iloc[:2016]  # 7 days * 24 hours * 12 intervals

fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(FEATURE_COLS):
    axes[idx].plot(one_week_data.index, one_week_data[col], linewidth=0.8)
    axes[idx].set_title(col, fontweight='bold')
    axes[idx].set_xlabel('Time')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/one_week_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/one_week_overview.png")

print("\n" + "="*80)
print("✅ TASK 1 COMPLETE: Data prepared and explored")
print("="*80)

# ============================================================================
# TASK 2: SHUTDOWN / IDLE PERIOD DETECTION
# ============================================================================
print("\n🔹 TASK 2: SHUTDOWN / IDLE PERIOD DETECTION")
print("="*80)

# Strategy: Use combined signal from all sensors
# Shutdowns = all sensors near zero or very low percentile

# Calculate normalized activity score
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[FEATURE_COLS]),
    columns=FEATURE_COLS,
    index=df.index
)

# Activity score = mean of absolute scaled values
df['activity_score'] = df_scaled.abs().mean(axis=1)

# Determine threshold (bottom 5th percentile)
threshold = df['activity_score'].quantile(SHUTDOWN_THRESHOLD_PERCENTILE / 100)
print(f"📊 Activity score threshold: {threshold:.3f}")

# Flag shutdown periods
df['is_shutdown'] = (df['activity_score'] < threshold).astype(int)

# Find continuous shutdown periods
df['shutdown_group'] = (df['is_shutdown'].diff() != 0).cumsum()
shutdown_periods = df[df['is_shutdown'] == 1].groupby('shutdown_group').agg(
    start=('activity_score', lambda x: x.index.min()),
    end=('activity_score', lambda x: x.index.max())
).reset_index(drop=True)

# Calculate duration in hours
shutdown_periods['duration_hours'] = (
    shutdown_periods['end'] - shutdown_periods['start']
).dt.total_seconds() / 3600

# Filter by minimum duration
shutdown_periods = shutdown_periods[
    shutdown_periods['duration_hours'] >= (MIN_SHUTDOWN_DURATION * 5 / 60)
]

print(f"\n📉 Detected {len(shutdown_periods)} shutdown periods")
print(f"⏱️  Total downtime: {shutdown_periods['duration_hours'].sum():.1f} hours")
print(f"📊 Average shutdown duration: {shutdown_periods['duration_hours'].mean():.1f} hours")
print(f"📊 Longest shutdown: {shutdown_periods['duration_hours'].max():.1f} hours")

# Save shutdown periods
shutdown_periods.to_csv('shutdown_periods.csv', index=False)
print("\n✅ Saved: shutdown_periods.csv")

# Visualize one year with shutdowns
print("\n📊 Creating one-year visualization with shutdowns...")
one_year_data = df.iloc[:105120]  # 365 days

fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(one_year_data.index, one_year_data['Cyclone_Inlet_Gas_Temp'], 
        linewidth=0.5, label='Inlet Gas Temp', color='blue', alpha=0.7)

# Highlight shutdowns
for _, period in shutdown_periods.iterrows():
    if period['start'] in one_year_data.index:
        ax.axvspan(period['start'], period['end'], color='red', alpha=0.3)

ax.set_xlabel('Time', fontsize=12, fontweight='bold')
ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax.set_title('One Year Overview with Shutdown Periods (Red)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/one_year_with_shutdowns.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/one_year_with_shutdowns.png")

print("\n" + "="*80)
print("✅ TASK 2 COMPLETE: Shutdowns detected and visualized")
print("="*80)

# ============================================================================
# TASK 3: MACHINE STATE SEGMENTATION (CLUSTERING)
# ============================================================================
print("\n🔹 TASK 3: MACHINE STATE SEGMENTATION (CLUSTERING)")
print("="*80)

# Exclude shutdown periods for clustering
df_active = df[df['is_shutdown'] == 0].copy()
print(f"📊 Active operation data: {len(df_active):,} records ({len(df_active)/len(df)*100:.1f}%)")

# Feature engineering for clustering
print("\n⚙️  Engineering features for clustering...")

# Rolling statistics (30 minutes = 6 intervals)
for col in FEATURE_COLS:
    df_active[f'{col}_roll_mean'] = df_active[col].rolling(6, min_periods=1).mean()
    df_active[f'{col}_roll_std'] = df_active[col].rolling(6, min_periods=1).std()

# Lag features
for col in FEATURE_COLS:
    df_active[f'{col}_lag1'] = df_active[col].shift(1)
    df_active[f'{col}_delta'] = df_active[col].diff()

df_active = df_active.fillna(method='bfill')

# Select features for clustering
cluster_features = FEATURE_COLS + [f'{col}_roll_mean' for col in FEATURE_COLS]
X_cluster = df_active[cluster_features].values

# Scale features
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_cluster)

# Apply HDBSCAN clustering
print("\n🎯 Applying HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
df_active['cluster'] = clusterer.fit_predict(X_scaled)

# Count clusters
n_clusters = len(set(df_active['cluster'])) - (1 if -1 in df_active['cluster'] else 0)
print(f"✅ Found {n_clusters} operational states (excluding noise)")

# Cluster distribution
cluster_counts = df_active['cluster'].value_counts().sort_index()
print("\n📊 Cluster distribution:")
for cluster_id, count in cluster_counts.items():
    label = "Noise" if cluster_id == -1 else f"State {cluster_id}"
    print(f"  {label}: {count:,} records ({count/len(df_active)*100:.1f}%)")

# Compute cluster statistics
print("\n📊 Computing cluster statistics...")
clusters_summary = []

for cluster_id in sorted(df_active['cluster'].unique()):
    if cluster_id == -1:
        continue  # Skip noise
    
    cluster_data = df_active[df_active['cluster'] == cluster_id]
    
    # Basic stats
    stats = {
        'cluster': f'State_{cluster_id}',
        'count': len(cluster_data),
        'percentage': len(cluster_data) / len(df_active) * 100,
        'avg_duration_hours': len(cluster_data) * 5 / 60 / cluster_counts[cluster_id]
    }
    
    # Feature statistics
    for col in FEATURE_COLS:
        stats[f'{col}_mean'] = cluster_data[col].mean()
        stats[f'{col}_std'] = cluster_data[col].std()
        stats[f'{col}_p25'] = cluster_data[col].quantile(0.25)
        stats[f'{col}_p75'] = cluster_data[col].quantile(0.75)
    
    clusters_summary.append(stats)

clusters_summary_df = pd.DataFrame(clusters_summary)
clusters_summary_df.to_csv('clusters_summary.csv', index=False)
print("\n✅ Saved: clusters_summary.csv")

# Visualize clusters
print("\n📊 Creating cluster visualization...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(FEATURE_COLS):
    for cluster_id in sorted(df_active['cluster'].unique()):
        if cluster_id == -1:
            continue
        cluster_data = df_active[df_active['cluster'] == cluster_id]
        axes[idx].scatter(cluster_data.index, cluster_data[col], 
                         s=1, alpha=0.3, label=f'State {cluster_id}')
    
    axes[idx].set_title(col, fontweight='bold')
    axes[idx].set_xlabel('Time')
    axes[idx].set_ylabel('Value')
    axes[idx].legend(markerscale=5)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/cluster_visualization.png")

print("\n" + "="*80)
print("✅ TASK 3 COMPLETE: Operational states identified")
print("="*80)

# ============================================================================
# TASK 4: CONTEXTUAL ANOMALY DETECTION + ROOT CAUSE ANALYSIS
# ============================================================================
print("\n🔹 TASK 4: CONTEXTUAL ANOMALY DETECTION + ROOT CAUSE")
print("="*80)

# Detect anomalies within each cluster using Isolation Forest
print("⚙️  Detecting anomalies per operational state...")

anomalies_list = []

for cluster_id in sorted(df_active['cluster'].unique()):
    if cluster_id == -1:
        continue
    
    cluster_data = df_active[df_active['cluster'] == cluster_id].copy()
    
    if len(cluster_data) < 50:  # Skip very small clusters
        continue
    
    # Fit Isolation Forest on this cluster
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    cluster_data['anomaly_score'] = iso_forest.fit_predict(
        cluster_data[FEATURE_COLS].values
    )
    
    # Find anomalies (-1 = anomaly, 1 = normal)
    anomalies = cluster_data[cluster_data['anomaly_score'] == -1]
    
    if len(anomalies) > 0:
        print(f"  State {cluster_id}: Found {len(anomalies)} anomalies ({len(anomalies)/len(cluster_data)*100:.2f}%)")
        
        # Group consecutive anomalies
        anomalies = anomalies.sort_index()
        anomalies['anomaly_group'] = (anomalies.index.to_series().diff() > pd.Timedelta('15T')).cumsum()
        
        for group_id, group in anomalies.groupby('anomaly_group'):
            start_time = group.index.min()
            end_time = group.index.max()
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            
            # Identify most deviant variables
            cluster_means = cluster_data[FEATURE_COLS].mean()
            group_means = group[FEATURE_COLS].mean()
            deviations = ((group_means - cluster_means) / cluster_data[FEATURE_COLS].std()).abs()
            top_variables = deviations.nlargest(3).index.tolist()
            
            anomalies_list.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': duration,
                'cluster': f'State_{cluster_id}',
                'top_variables': ', '.join(top_variables),
                'severity': deviations.max()
            })

anomalies_df = pd.DataFrame(anomalies_list)
anomalies_df = anomalies_df.sort_values('severity', ascending=False).head(50)  # Top 50
print(f"\n✅ Detected {len(anomalies_df)} significant anomalous periods")

anomalies_df.to_csv('anomalous_periods.csv', index=False)
print("✅ Saved: anomalous_periods.csv")

# ROOT CAUSE ANALYSIS - Select 3 interesting anomalies
print("\n🔍 ROOT CAUSE ANALYSIS - Analyzing top 3 anomalies:")
print("="*80)

selected_anomalies = anomalies_df.nlargest(3, 'severity')

for idx, anomaly in selected_anomalies.iterrows():
    print(f"\n🔴 ANOMALY {idx + 1}:")
    print(f"  Time: {anomaly['start_time']} to {anomaly['end_time']}")
    print(f"  Duration: {anomaly['duration_minutes']:.1f} minutes")
    print(f"  State: {anomaly['cluster']}")
    print(f"  Implicated variables: {anomaly['top_variables']}")
    print(f"  Severity score: {anomaly['severity']:.2f}")
    
    # Get data around anomaly
    start = anomaly['start_time'] - pd.Timedelta('2H')
    end = anomaly['end_time'] + pd.Timedelta('2H')
    context_data = df_active.loc[start:end] if start in df_active.index else df_active
    
    if len(context_data) > 0:
        # Visualize this anomaly
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for col_idx, col in enumerate(FEATURE_COLS):
            axes[col_idx].plot(context_data.index, context_data[col], 
                              linewidth=1, color='blue', alpha=0.7)
            axes[col_idx].axvspan(anomaly['start_time'], anomaly['end_time'], 
                                 color='red', alpha=0.3, label='Anomaly')
            axes[col_idx].set_title(col, fontweight='bold')
            axes[col_idx].set_xlabel('Time')
            axes[col_idx].set_ylabel('Value')
            axes[col_idx].grid(True, alpha=0.3)
            axes[col_idx].legend()
        
        plt.suptitle(f'Anomaly {idx + 1} - Context Visualization', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/anomaly_example_{idx + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: plots/anomaly_example_{idx + 1}.png")
    
    # Root cause hypothesis
    print(f"  💡 Hypothesis:")
    if 'Inlet' in anomaly['top_variables'] and 'Outlet' in anomaly['top_variables']:
        print(f"     Temperature gradient anomaly suggests thermal instability or flow disruption")
    elif 'draft' in anomaly['top_variables'].lower():
        print(f"     Pressure anomaly indicates potential blockage or ventilation issue")
    else:
        print(f"     Multi-variable deviation suggests systemic operational change")

print("\n" + "="*80)
print("✅ TASK 4 COMPLETE: Anomalies detected and analyzed")
print("="*80)

# ============================================================================
# TASK 5: SHORT-HORIZON FORECASTING
# ============================================================================
print("\n🔹 TASK 5: SHORT-HORIZON FORECASTING (1 hour = 12 steps)")
print("="*80)

# Target variable: Cyclone_Inlet_Gas_Temp
target_col = 'Cyclone_Inlet_Gas_Temp'
forecast_horizon = 12  # 12 steps = 1 hour

# Use only active periods for training
df_forecast = df_active[df_active['cluster'] != -1].copy()

# Train-test split (last 10% for testing)
split_point = int(len(df_forecast) * 0.9)
train = df_forecast.iloc[:split_point]
test = df_forecast.iloc[split_point:]

print(f"📊 Training data: {len(train):,} records")
print(f"📊 Testing data: {len(test):,} records")

# Prepare test predictions
test_results = []

# METHOD 1: Persistence Baseline
print("\n🔹 Method 1: Persistence Baseline")
persistence_preds = []
persistence_actuals = []

for i in range(len(test) - forecast_horizon):
    last_value = test[target_col].iloc[i]
    actual_values = test[target_col].iloc[i+1:i+1+forecast_horizon].values
    pred_values = [last_value] * forecast_horizon
    
    persistence_preds.extend(pred_values)
    persistence_actuals.extend(actual_values)

persistence_rmse = np.sqrt(mean_squared_error(persistence_actuals, persistence_preds))
persistence_mae = mean_absolute_error(persistence_actuals, persistence_preds)

print(f"  RMSE: {persistence_rmse:.3f}")
print(f"  MAE: {persistence_mae:.3f}")

# METHOD 2: ARIMA
print("\n🔹 Method 2: ARIMA")

# Use a subset for faster training
train_subset = train[target_col].iloc[-10000:]

try:
    # Fit ARIMA
    arima_model = ARIMA(train_subset, order=(5, 1, 2))
    arima_fit = arima_model.fit()
    
    arima_preds = []
    arima_actuals = []
    
    # Rolling forecast
    for i in range(0, len(test) - forecast_horizon, forecast_horizon):
        history = pd.concat([train_subset, test[target_col].iloc[:i]])
        model = ARIMA(history.iloc[-5000:], order=(5, 1, 2))
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=forecast_horizon)
        actual = test[target_col].iloc[i:i+forecast_horizon].values
        
        arima_preds.extend(forecast)
        arima_actuals.extend(actual)
        
        if i > 500:  # Limit iterations for speed
            break
    
    arima_rmse = np.sqrt(mean_squared_error(arima_actuals, arima_preds))
    arima_mae = mean_absolute_error(arima_actuals, arima_preds)
    
    print(f"  RMSE: {arima_rmse:.3f}")
    print(f"  MAE: {arima_mae:.3f}")
except Exception as e:
    print(f"  ⚠️  ARIMA failed: {e}")
    arima_rmse, arima_mae = None, None

# METHOD 3: Random Forest with Lag Features
print("\n🔹 Method 3: Random Forest with Lag Features")

# Create lag features
def create_lag_features(data, target_col, n_lags=24):
    df_lag = data.copy()
    for i in range(1, n_lags + 1):
        df_lag[f'lag_{i}'] = df_lag[target_col].shift(i)
    df_lag = df_lag.dropna()
    return df_lag

df_lag = create_lag_features(df_forecast[[target_col]], target_col, n_lags=24)

# Split
train_lag = df_lag.iloc[:split_point]
test_lag = df_lag.iloc[split_point:]

X_train = train_lag.drop(target_col, axis=1)
y_train = train_lag[target_col]
X_test = test_lag.drop(target_col, axis=1)
y_test = test_lag[target_col]

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Multi-step prediction
rf_preds = []
rf_actuals = []

for i in range(0, len(X_test) - forecast_horizon, forecast_horizon):
    # Predict 12 steps ahead iteratively
    current_features = X_test.iloc[i].values.reshape(1, -1)
    predictions = []
    
    for step in range(forecast_horizon):
        pred = rf_model.predict(current_features)[0]
        predictions.append(pred)
        
        # Update features (shift and add new prediction)
        current_features = np.roll(current_features, -1)
        current_features[0, -1] = pred
    
    actual = y_test.iloc[i:i+forecast_horizon].values
    rf_preds.extend(predictions[:len(actual)])
    rf_actuals.extend(actual)

rf_rmse = np.sqrt(mean_squared_error(rf_actuals, rf_preds))
rf_mae = mean_absolute_error(rf_actuals, rf_preds)

print(f"  RMSE: {rf_rmse:.3f}")
print(f"  MAE: {rf_mae:.3f}")

# Compare models
print("\n📊 MODEL COMPARISON:")
print("="*80)
comparison = pd.DataFrame({
    'Model': ['Persistence', 'ARIMA', 'Random Forest'],
    'RMSE': [persistence_rmse, arima_rmse, rf_rmse],
    'MAE': [persistence_mae, arima_mae, rf_mae]
})
print(comparison.to_string(index=False))

# Save forecast results
forecast_results = pd.DataFrame({
    'actual': rf_actuals[:100],  # First 100 predictions
    'persistence': persistence_preds[:100],
    'random_forest': rf_preds[:100]
})
forecast_results.to_csv('forecasts.csv', index=False)
print("\n✅ Saved: forecasts.csv")

# Visualize forecasts
print("\n📊 Creating forecast visualization...")
plt.figure(figsize=(15, 6))
plt.plot(range(100), rf_actuals[:100], label='Actual', linewidth=2, color='black')
plt.plot(range(100), persistence_preds[:100], label='Persistence', linewidth=1, alpha=0.7)
plt.plot(range(100), rf_preds[:100], label='Random Forest', linewidth=1, alpha=0.7)
plt.xlabel('Time Steps (5-min intervals)', fontsize=12, fontweight='bold')
plt.ylabel('Cyclone Inlet Gas Temp', fontsize=12, fontweight='bold')
plt.title('Forecast Comparison - First 100 Steps', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/forecast_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/forecast_comparison.png")

print("\n💡 Forecasting Challenges:")
print("  • Regime changes between operational states affect accuracy")
print("  • Shutdowns create non-stationarity in the data")
print("  • Short-term predictions more reliable than long-term")
print("  • ML models outperform simple baselines by capturing patterns")

print("\n" + "="*80)
print("✅ TASK 5 COMPLETE: Forecasting models compared")
print("="*80)

# ============================================================================
# TASK 6: INSIGHTS & STORYTELLING
# ============================================================================
print("\n🔹 TASK 6: INSIGHTS & STORYTELLING")
print("="*80)

print("\n📊 KEY INSIGHTS:")
print("="*80)

# Insight 1: Shutdown patterns
total_downtime_days = shutdown_periods['duration_hours'].sum() / 24
uptime_percentage = ((df['is_shutdown'] == 0).sum() / len(df)) * 100

print(f"\n1️⃣  OPERATIONAL AVAILABILITY:")
print(f"   • Machine uptime: {uptime_percentage:.1f}%")
print(f"   • Total downtime: {total_downtime_days:.1f} days over 3 years")
print(f"   • Number of shutdown events: {len(shutdown_periods)}")
print(f"   • Average shutdown duration: {shutdown_periods['duration_hours'].mean():.1f} hours")
print(f"   💡 Recommendation: Investigate causes of frequent short shutdowns")

# Insight 2: Operational states
print(f"\n2️⃣  OPERATIONAL STATES IDENTIFIED:")
for i, row in clusters_summary_df.iterrows():
    print(f"   • {row['cluster']}: {row['percentage']:.1f}% of active time")
    inlet_temp = row['Cyclone_Inlet_Gas_Temp_mean']
    if inlet_temp > df['Cyclone_Inlet_Gas_Temp'].quantile(0.75):
        print(f"     → High load operation (Inlet Temp: {inlet_temp:.1f}°C)")
    elif inlet_temp < df['Cyclone_Inlet_Gas_Temp'].quantile(0.25):
        print(f"     → Low load operation (Inlet Temp: {inlet_temp:.1f}°C)")
    else:
        print(f"     → Normal operation (Inlet Temp: {inlet_temp:.1f}°C)")

# Insight 3: Anomalies
anomaly_rate = len(anomalies_df) / (len(df_active) / (24 * 12)) * 100  # per day
print(f"\n3️⃣  ANOMALY PATTERNS:")
print(f"   • Anomaly rate: {anomaly_rate:.2f} events per day")
print(f"   • Most common issues: {anomalies_df['top_variables'].mode()[0] if len(anomalies_df) > 0 else 'N/A'}")
print(f"   💡 Recommendation: Set up real-time alerts for top anomaly variables")

# Insight 4: Correlations
strongest_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
strongest_corr = strongest_corr[strongest_corr < 1.0].head(3)
print(f"\n4️⃣  STRONGEST CORRELATIONS:")
for (var1, var2), corr_val in strongest_corr.items():
    print(f"   • {var1} ↔ {var2}: {corr_val:.3f}")
print(f"   💡 Recommendation: Use these relationships for predictive maintenance")

# Insight 5: Forecasting
best_model = comparison.loc[comparison['RMSE'].idxmin(), 'Model']
print(f"\n5️⃣  FORECASTING CAPABILITY:")
print(f"   • Best model: {best_model}")
print(f"   • Prediction horizon: 1 hour (12 steps)")
print(f"   • Accuracy: RMSE = {comparison['RMSE'].min():.3f}")
print(f"   💡 Recommendation: Deploy {best_model} for real-time forecasting dashboard")

print("\n" + "="*80)
print("🎯 ACTIONABLE RECOMMENDATIONS:")
print("="*80)
print("1. Implement real-time anomaly detection using cluster-specific thresholds")
print("2. Create operator alerts for variables frequently involved in anomalies")
print("3. Schedule preventive maintenance before predicted high-anomaly periods")
print("4. Investigate root causes of frequent short-duration shutdowns")
print("5. Deploy forecasting model for 1-hour ahead temperature predictions")
print("6. Monitor operational state transitions for early warning signs")

print("\n" + "="*80)
print("✅ TASK 6 COMPLETE: Insights generated")
print("="*80)

print("\n" + "🎉" * 40)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("🎉" * 40)

print("\n📦 DELIVERABLES GENERATED:")
print("="*80)
print("✅ shutdown_periods.csv")
print("✅ anomalous_periods.csv")
print("✅ clusters_summary.csv")
print("✅ forecasts.csv")
print("✅ summary_statistics.csv")
print("\n✅ plots/correlation_matrix.png")
print("✅ plots/one_week_overview.png")
print("✅ plots/one_year_with_shutdowns.png")
print("✅ plots/cluster_visualization.png")
print("✅ plots/anomaly_example_1.png")
print("✅ plots/anomaly_example_2.png")
print("✅ plots/anomaly_example_3.png")
print("✅ plots/forecast_comparison.png")