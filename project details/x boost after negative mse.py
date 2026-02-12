import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime

# Set working directory
working_dir = 'F:/project details'
os.chdir(working_dir)
print("Current Working Directory:", os.getcwd())

# Timestamp for unique file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load data
try:
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')['Activity_concentration']
    y_test = pd.read_csv('y_test.csv')['Activity_concentration']
    print("Data loaded successfully.")
    print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure Step 6 outputs exist.")
    exit()

# Verify key columns
expected_cols = ['Latitude_(deg)', 'Longitude_(deg)', 'Distance_from_FDNPP', 'Height_Elevation']
missing_cols = [col for col in expected_cols if col not in X_train.columns]
if missing_cols:
    print(f"Warning: Missing columns in X_train: {missing_cols}")
else:
    print("Key columns present:", expected_cols)

# Feature selection: Remove low-variance features
variance = X_train.var()
low_variance_cols = variance[variance < 1e-6].index
print(f"\nRemoving low-variance columns: {list(low_variance_cols)}")
X_train = X_train.drop(columns=low_variance_cols)
X_test = X_test.drop(columns=low_variance_cols)

# Initialize XGBoost with hyperparameter tuning
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
print("\nPerforming GridSearchCV for hyperparameter tuning...")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Train the best model
print("\nTraining XGBoost with best parameters...")
best_model.fit(X_train, y_train)
xgb_pred_train = best_model.predict(X_train)
xgb_pred_test = best_model.predict(X_test)

# Evaluate XGBoost
xgb_mse_train = mean_squared_error(y_train, xgb_pred_train)
xgb_rmse_train = np.sqrt(xgb_mse_train)
xgb_r2_train = r2_score(y_train, xgb_pred_train)
xgb_mse_test = mean_squared_error(y_test, xgb_pred_test)
xgb_rmse_test = np.sqrt(xgb_mse_test)
xgb_r2_test = r2_score(y_test, xgb_pred_test)
print("XGBoost Results:")
print(f"Train MSE: {xgb_mse_train:.4f}, RMSE: {xgb_rmse_train:.4f}, R²: {xgb_r2_train:.4f}")
print(f"Test MSE: {xgb_mse_test:.4f}, RMSE: {xgb_rmse_test:.4f}, R²: {xgb_r2_test:.4f}")

# Generate accuracy plot (cross-validation MSE converted to positive)
cv_scores = -cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores  # Convert to positive MSE
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), cv_mse_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD'])
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title('XGBoost Cross-Validation MSE Across Folds')
plt.xticks(range(1, 6))
plt.tight_layout()
plt.savefig(f'accuracy_plot_{timestamp}_xgb.png')
plt.close()
print(f"\nAccuracy plot saved to accuracy_plot_{timestamp}_xgb.png")

# Generate risk map
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    X_test['Longitude_(deg)'], 
    X_test['Latitude_(deg)'], 
    c=xgb_pred_test, 
    cmap='viridis', 
    s=50, 
    alpha=0.7
)
plt.colorbar(sc, label='Predicted Activity Concentration')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Risk Map: XGBoost Predicted Activity Concentration')
plt.tight_layout()
plt.savefig('risk_map_xgb.png')
plt.close()
print("\nRisk map saved to risk_map_xgb.png")

# Save predictions
pd.DataFrame({
    'Actual': y_test,
    'XGB_Predicted': xgb_pred_test
}).to_csv('xgb_predictions.csv', index=False)
print("Predictions saved to xgb_predictions.csv")

# Save model
joblib.dump(best_model, 'xgb_model.pkl')
print("Model saved to xgb_model.pkl")

# Save feature importances
xgb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
xgb_importance.to_csv('xgb_feature_importance.csv_n', index=False)
print("Feature importances saved to xgb_feature_importance_n.csv")