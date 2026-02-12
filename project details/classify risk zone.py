import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime

# Set working directory
working_dir = 'F:\\project details'
os.chdir(working_dir)
print("Current Working Directory:", os.getcwd())

# Find latest GNN predictions file
prediction_files = glob(os.path.join(working_dir, 'gcn_predictions_*.csv'))
if not prediction_files:
    print("Error: No GNN prediction files found (gcn_predictions_*.csv).")
    exit()
latest_pred_file = max(prediction_files, key=os.path.getctime)
print(f"Using latest predictions: {latest_pred_file}")

# Load test data and GNN predictions
try:
    X_test = pd.read_csv('X_test.csv')
    gnn_pred = pd.read_csv(latest_pred_file)
    print("Test data and GNN predictions loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure X_test.csv and {latest_pred_file} exist.")
    exit()

# Verify spatial columns
spatial_cols = ['Latitude_(deg)', 'Longitude_(deg)']
if not all(col in X_test.columns for col in spatial_cols):
    print(f"Error: Missing spatial columns {spatial_cols} in X_test.csv.")
    exit()

# Extract spatial coordinates and predictions
lat = X_test['Latitude_(deg)']
lon = X_test['Longitude_(deg)']
y_pred = gnn_pred['GCN_Predicted']
y_actual = gnn_pred['Actual']

# Define risk zone thresholds using terciles
low_threshold = np.percentile(y_pred, 33)
high_threshold = np.percentile(y_pred, 66)
print(f"Low risk threshold (33rd percentile): {low_threshold:.4f} Bq/kg")
print(f"High risk threshold (66th percentile): {high_threshold:.4f} Bq/kg")

# Classify risk zones
risk_zones = []
for pred in y_pred:
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
    'Actual': y_actual,
    'GCN_Predicted': y_pred,
    'Risk_Zone': risk_zones
})

# Save classified data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_df.to_csv(f'risk_zones_{timestamp}.csv', index=False)
print(f"\nClassified risk zones saved to risk_zones_{timestamp}.csv")

# Create risk zone map
plt.figure(figsize=(10, 8))

# Define colors for risk zones
colors = {'High': 'red', 'Medium': 'yellow', 'Low': 'blue'}
for zone in colors:
    mask = output_df['Risk_Zone'] == zone
    plt.scatter(
        output_df['Longitude_(deg)'][mask],
        output_df['Latitude_(deg)'][mask],
        c=colors[zone],
        label=zone,
        s=50,
        alpha=0.6
    )

plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Fukushima Cesium Contamination Risk Zones (GCN Predictions)')
plt.legend(title='Risk Zone')
plt.tight_layout()

# Save risk zone map
plt.savefig(f'risk_zones_map_{timestamp}.png')
plt.close()
print(f"Risk zones map saved to risk_zones_map_{timestamp}.png")