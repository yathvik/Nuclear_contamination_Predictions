import pandas as pd
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

# Define threshold for high-risk zones (median of predictions)
threshold = y_pred.median()
print(f"Classification threshold (median GCN_Predicted): {threshold:.4f}")

# Determine data bounds for specific area
lat_min, lat_max = lat.min(), lat.max()
lon_min, lon_max = lon.min(), lon.max()
print(f"Data bounds: Lat {lat_min:.4f} to {lat_max:.4f}, Lon {lon_min:.4f} to {lon_max:.4f}")

# Create scatter plot with colorbar, focused on data area
plt.figure(figsize=(10, 8))
scatter = plt.scatter(lon, lat, c=y_pred, cmap='RdBu_r', s=50, alpha=0.6)
plt.colorbar(scatter, label='Predicted Activity Concentration (Bq/kg)')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title(f'Fukushima Cesium Risk Map (GCN) - Data Area [{lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°E to {lon_max:.2f}°E]')
plt.xlim(lon_min - 0.1, lon_max + 0.1)  # Add small buffer
plt.ylim(lat_min - 0.1, lat_max + 0.1)  # Add small buffer
plt.axhline(y=threshold, color='r', linestyle='--', label=f'High Risk Threshold ({threshold:.2f} Bq/kg)')
plt.legend()
plt.tight_layout()

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'risk_map_{timestamp}.png')
plt.close()
print(f"\nRisk map saved to risk_map_{timestamp}.png")