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

# Timestamp for unique file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20250828_0110

# Load prediction files
try:
    xgb_pred = pd.read_csv('xgb_predictions.csv')
    rf_pred = pd.read_csv('rf_predictions.csv')
    pytorch_pred = pd.read_csv('pytorch_predictions.csv')
    gnn_pred = pd.read_csv('gnn predictions.csv')
    print("Prediction files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure all model prediction files exist.")
    exit()

# Extract predicted values with length check and column fallback
if not (len(xgb_pred) == len(rf_pred) == len(pytorch_pred) == len(gnn_pred)):
    print("Error: Mismatched lengths in prediction files.")
    exit()

xgb_pred_test = xgb_pred['XGB_Predicted']
rf_pred_test = rf_pred['RF_Predicted']
pytorch_pred_test = pytorch_pred['PyTorch_Predicted']
gnn_pred_test = gnn_pred.get('GNN_Predicted', gnn_pred.get('GCN_Predicted'))
if gnn_pred_test is None:
    print("Error: Neither 'GNN_Predicted' nor 'GCN_Predicted' found in gnn_predictions.csv.")
    exit()

# Compute risk thresholds based on all predictions with debugging
all_predictions = np.concatenate([xgb_pred_test, rf_pred_test, pytorch_pred_test, gnn_pred_test])
print(f"All predictions range: min={all_predictions.min():.4f}, max={all_predictions.max():.4f}")
q25, q75 = np.percentile(all_predictions, [25, 75])
if q25 == q75:  # Handle case where all values are identical
    q25 = all_predictions.min()
    q75 = all_predictions.max() if all_predictions.max() > all_predictions.min() else all_predictions.min() + 1e-6
print(f"Computed thresholds: Low={q25:.4f}, High={q75:.4f}")
risk_thresholds = {'low': q25, 'high': q75}

# Classify predictions into activity categories with debugging
def classify_activity(pred_values):
    activity_classes = pred_values.copy()  # Start with float values
    print(f"Classifying {len(pred_values)} values with thresholds: Low={risk_thresholds['low']:.4f}, High={risk_thresholds['high']:.4f}")
    activity_classes[pred_values <= risk_thresholds['low']] = 'Low Activity'
    mask_avg = (pred_values > risk_thresholds['low']) & (pred_values <= risk_thresholds['high'])
    activity_classes[mask_avg] = 'Average Activity'
    activity_classes[pred_values > risk_thresholds['high']] = 'High Activity'
    print(f"Classification counts: {np.unique(activity_classes, return_counts=True)}")
    return activity_classes

xgb_activity = classify_activity(xgb_pred_test)
rf_activity = classify_activity(rf_pred_test)
pytorch_activity = classify_activity(pytorch_pred_test)
gnn_activity = classify_activity(gnn_pred_test)

# Calculate activity distribution percentages
models = ['XGBoost', 'Random Forest', 'PyTorch', 'GNN']
activity_distributions = {}
for pred_activity, model_name in zip([xgb_activity, rf_activity, pytorch_activity, gnn_activity], models):
    activity_counts = np.unique(pred_activity, return_counts=True)
    activity_dict = dict(zip(activity_counts[0], activity_counts[1] / len(pred_activity) * 100))
    activity_distributions[model_name] = {
        'Low Activity %': activity_dict.get('Low Activity', 0),
        'Average Activity %': activity_dict.get('Average Activity', 0),
        'High Activity %': activity_dict.get('High Activity', 0)
    }
    print(f"\n{model_name} Activity Distribution:")
    print(f"Low Activity: {activity_dict.get('Low Activity', 0):.1f}%, Average Activity: {activity_dict.get('Average Activity', 0):.1f}%, "
          f"High Activity: {activity_dict.get('High Activity', 0):.1f}%")

# Visualize activity distribution
plt.figure(figsize=(12, 8))
for i, (model_name, dist) in enumerate(activity_distributions.items(), 1):
    plt.subplot(2, 2, i)
    plt.bar(dist.keys(), dist.values(), color=['#96CEB4', '#FFEEAD', '#FF6B6B'])
    plt.title(f'{model_name} Activity Distribution')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)  # Ensure y-axis fits all values
    for j, v in enumerate(dist.values()):
        plt.text(j, v + 1, f'{v:.1f}', ha='center')

plt.tight_layout(pad=2.0)  # Increase padding to avoid warning
plt.savefig(f'activity_distribution_plot_{timestamp}.png')
plt.close()
print(f"\nActivity distribution plot saved to activity_distribution_plot_{timestamp}.png")