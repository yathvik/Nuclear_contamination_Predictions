import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid _tkinter.TclError
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime

# Set working directory
working_dir = 'F:\\project details'
os.chdir(working_dir)
print("Current Working Directory:", os.getcwd())

# Find latest GNN predictions and test data files
prediction_files = glob(os.path.join(working_dir, 'gcn_predictions_*.csv'))
test_files = glob(os.path.join(working_dir, 'X_test_*.csv'))
if not (prediction_files and test_files):
    print("Error: Missing gcn_predictions_*.csv or X_test_*.csv files.")
    exit()
latest_pred_file = max(prediction_files, key=os.path.getctime)
latest_test_file = max(test_files, key=os.path.getctime)
print(f"Using predictions: {latest_pred_file}")
print(f"Using test data: {latest_test_file}")

# Load test data and GNN predictions
try:
    X_test = pd.read_csv(latest_test_file)
    gnn_pred = pd.read_csv(latest_pred_file)
    print("Test data and GNN predictions loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Verify required columns
required_cols = ['Latitude_(deg)', 'Longitude_(deg)', 'Distance_from_FDNPP', 'Height_Elevation', 'Total_biomass', 'Soil_type_encoded']
if not all(col in X_test.columns for col in required_cols):
    print(f"Error: Missing required columns {required_cols} in X_test_*.csv.")
    exit()

# Combine data, keeping only necessary columns
data = pd.DataFrame({
    'Latitude_(deg)': X_test['Latitude_(deg)'],
    'Longitude_(deg)': X_test['Longitude_(deg)'],
    'Distance_from_FDNPP': X_test['Distance_from_FDNPP'],
    'Height_Elevation': X_test['Height_Elevation'],
    'Total_biomass': X_test['Total_biomass'],
    'Soil_type_encoded': X_test['Soil_type_encoded'],
    'Actual': gnn_pred['Actual'],
    'GCN_Predicted': gnn_pred['GCN_Predicted']
})

# Filter out rows with NaN/null values
data_complete = data.dropna()
print(f"Filtered out NaN/null rows. Original rows: {len(data)}, Complete rows: {len(data_complete)}")

# Extract features
lat = data_complete['Latitude_(deg)']
lon = data_complete['Longitude_(deg)']
y_pred = data_complete['GCN_Predicted']
y_actual = data_complete['Actual']
dist = data_complete['Distance_from_FDNPP']
height_elev = data_complete['Height_Elevation']
biomass = data_complete['Total_biomass']
soil_type = data_complete['Soil_type_encoded']

# Normalize additional features for weighting (scale to [0, 1])
dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
height_elev_norm = (height_elev - height_elev.min()) / (height_elev.max() - height_elev.min() + 1e-6)
biomass_norm = (biomass - biomass.min()) / (biomass.max() - biomass.min() + 1e-6)
soil_type_norm = (soil_type - soil_type.min()) / (soil_type.max() - soil_type.min() + 1e-6)

# Weight predictions (increase risk for closer distance, higher elevation, more biomass, clayey soils)
weights = 1.0 - 0.4 * dist_norm + 0.2 * height_elev_norm + 0.2 * biomass_norm + 0.2 * soil_type_norm
weighted_pred = y_pred * weights

# Define risk zone thresholds using terciles of weighted predictions
low_threshold = np.percentile(weighted_pred, 33)
high_threshold = np.percentile(weighted_pred, 66)
print(f"Low risk threshold (33rd percentile): {low_threshold:.4f} Bq/kg")
print(f"High risk threshold (66th percentile): {high_threshold:.4f} Bq/kg")

# Classify risk zones
risk_zones = []
for pred in weighted_pred:
    if pred > high_threshold:
        risk_zones.append('High')
    elif pred > low_threshold:
        risk_zones.append('Medium')
    else:
        risk_zones.append('Low')

# Create DataFrame with classifications
output_df = pd.DataFrame({
    'Latitude_(deg)': lat,
    'Longitude_(deg)': lon,
    'Distance_from_FDNPP': dist,
    'Height_Elevation': height_elev,
    'Total_biomass': biomass,
    'Soil_type_encoded': soil_type,
    'Actual': y_actual,
    'GCN_Predicted': y_pred,
    'Weighted_Predicted': weighted_pred,
    'Risk_Zone': risk_zones
})

# Save classified data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_df.to_csv(f'risk_zones_{timestamp}.csv', index=False)
print(f"\nClassified risk zones saved to risk_zones_{timestamp}.csv")

# Create risk zone map
plt.figure(figsize=(10, 8))
colors = {'High': 'red', 'Medium': 'yellow', 'Low': 'blue'}
sizes = 50 + 100 * height_elev_norm  # Adjust point size by Height_Elevation
for zone in colors:
    mask = output_df['Risk_Zone'] == zone
    plt.scatter(
        output_df['Longitude_(deg)'][mask],
        output_df['Latitude_(deg)'][mask],
        c=colors[zone],
        s=sizes[mask],
        alpha=0.6,
        label=f"{zone} (Dist: {output_df['Distance_from_FDNPP'][mask].mean():.1f} km, Biomass: {output_df['Total_biomass'][mask].mean():.1f})"
    )

plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Fukushima Cesium Contamination Risk Zones (Weighted GCN Predictions)')
plt.legend(title='Risk Zone (Avg. Distance, Biomass)')
plt.tight_layout()

# Save risk zone map
plt.savefig(f'risk_zones_map_{timestamp}.png')
plt.close()
print(f"Risk zones map saved to risk_zones_map_ujj{timestamp}.png")