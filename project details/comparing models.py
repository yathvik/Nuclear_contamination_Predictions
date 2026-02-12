import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set working directory
working_dir = 'F:\\project details'
os.chdir(working_dir)
print("Current Working Directory:", os.getcwd())

# Timestamp for unique file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20250825_1601XX

# Load prediction files
try:
    xgb_pred = pd.read_csv('xgb_predictions.csv')
    rf_pred = pd.read_csv('rf_predictions.csv')
    pytorch_pred = pd.read_csv('pytorch_predictions.csv')
    gnn_pred = pd.read_csv('gnn prediction.csv')
    print("Prediction files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure all model prediction files exist.")
    exit()

# Load feature importance files
try:
    xgb_importance = pd.read_csv('xgb_feature_importance.csv')
    rf_importance = pd.read_csv('rf_feature_importance.csv')
    pytorch_importance = pd.read_csv('pytorch_feature_importance.csv')
    gnn_importance = pd.read_csv('gnn_feature_importance.csv')
    print("Feature importance files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure all feature importance files exist.")
    exit()

# Extract actual and predicted values with length check and column fallback
y_test = xgb_pred['Actual']
if not (len(xgb_pred) == len(rf_pred) == len(pytorch_pred) == len(gnn_pred)):
    print("Error: Mismatched lengths in prediction files.")
    exit()

# Handle potential column name mismatch for GNN
predicted_cols = {'XGB_Predicted', 'RF_Predicted', 'PyTorch_Predicted', 'GNN_Predicted', 'GCN_Predicted'}
xgb_pred_test = xgb_pred['XGB_Predicted']
rf_pred_test = rf_pred['RF_Predicted']
pytorch_pred_test = pytorch_pred['PyTorch_Predicted']
gnn_pred_test = gnn_pred.get('GNN_Predicted', gnn_pred.get('GCN_Predicted'))
if gnn_pred_test is None:
    print("Error: Neither 'GNN_Predicted' nor 'GCN_Predicted' found in gnn_predictions.csv.")
    exit()

# Compute MSE and R² for each model
models = ['XGBoost', 'Random Forest', 'PyTorch', 'GNN']
metrics = {
    'Model': models,
    'MSE': [],
    'R²': []
}
for pred, model_name in zip([xgb_pred_test, rf_pred_test, pytorch_pred_test, gnn_pred_test], models):
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    metrics['MSE'].append(mse)
    metrics['R²'].append(r2)
    print(f"\n{model_name} Test Metrics:")
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

# Create comparison table
comparison_df = pd.DataFrame(metrics)
comparison_df.to_csv(f'model_comparison_{timestamp}.csv', index=False)
print(f"\nModel comparison table saved to model_comparison_{timestamp}.csv")
print("\nModel Comparison Table:")
print(comparison_df)

# Statistical significance (paired t-test with best model)
best_model_idx = comparison_df['R²'].idxmax()
best_pred = [xgb_pred_test, rf_pred_test, pytorch_pred_test, gnn_pred_test][best_model_idx]
for i, pred in enumerate([xgb_pred_test, rf_pred_test, pytorch_pred_test, gnn_pred_test]):
    if i != best_model_idx:
        t_stat, p_val = ttest_rel(best_pred, pred)
        model_name = models[i]
        print(f"\nT-test (Best vs {model_name}): t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
        if p_val < 0.05:
            print(f"Significant difference (p < 0.05) between Best and {model_name}")

# Visualize MSE and R²
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(models, comparison_df['MSE'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('Model Comparison: MSE')
plt.xticks(rotation=45)
for i, v in enumerate(comparison_df['MSE']):
    plt.text(i, v + 0.1, f'{v:.2f}', ha='center')

plt.subplot(1, 2, 2)
plt.bar(models, comparison_df['R²'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.xlabel('Model')
plt.ylabel('R²')
plt.title('Model Comparison: R²')
plt.xticks(rotation=45)
for i, v in enumerate(comparison_df['R²']):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.savefig(f'model_metrics_plot_{timestamp}.png')
plt.close()
print(f"\nModel metrics plot saved to model_metrics_plot_{timestamp}.png")

# Visualize top 5 feature importances
plt.figure(figsize=(10, 6))
top_features = {}
for df, model_name in zip([xgb_importance, rf_importance, pytorch_importance, gnn_importance], models):
    top_features[model_name] = df.head(5)

bottom = np.zeros(len(top_features['XGBoost']['Feature']))
for model_name in models:
    plt.barh(top_features['XGBoost']['Feature'], top_features[model_name]['Importance'], 
             label=model_name, left=bottom, color={'XGBoost': '#FF6B6B', 'Random Forest': '#4ECDC4', 
                                                  'PyTorch': '#45B7D1', 'GNN': '#96CEB4'}[model_name])
    bottom += top_features[model_name]['Importance']

plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 5 Feature Importances Across Models')
plt.legend()
plt.tight_layout()
plt.savefig(f'feature_importance_plot_{timestamp}.png')
plt.close()
print(f"\nFeature importance plot saved to feature_importance_plot_{timestamp}.png")

# Identify best model based on R²
best_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
print(f"\nBest Model (based on R²): {best_model}")