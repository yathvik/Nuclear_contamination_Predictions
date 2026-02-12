import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

# Set working directory
working_dir = 'F:/project details'
os.chdir(working_dir)
print("Step 7 - Current Working Directory:", os.getcwd())

# Timestamp for unique file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load data from Step 6
try:
    for file in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
        file_path = os.path.join(working_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found! Run Step 6 first.")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')['Activity_concentration']
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')['Activity_concentration']
    print("Step 7 - Data loaded successfully.")
    print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Feature selection: Remove low-variance features
variance = X_train.var()
low_variance_cols = variance[variance < 1e-6].index
print(f"\nStep 7 - Removing low-variance columns: {list(low_variance_cols)}")
X_train = X_train.drop(columns=low_variance_cols)
X_test = X_test.drop(columns=low_variance_cols)

# Verify key columns
expected_cols = ['Latitude_(deg)', 'Longitude_(deg)', 'Distance_from_FDNPP', 'Height_Elevation']
missing_cols = [col for col in expected_cols if col not in X_train.columns]
if missing_cols:
    print(f"Step 7 - Warning: Missing columns in X_train: {missing_cols}")
    exit()
else:
    print("Step 7 - Key columns present:", expected_cols)

# Check y_test variance
y_test_var = y_test.var()
print("\nStep 7 - y_test variance:", y_test_var)
if y_test_var < 1e-6:
    print("Step 7 - Warning: y_test has near-zero variance, R² may be undefined")

# Initialize and train Random Forest with multiple n_estimators for tracking
n_estimators_list = [10, 50, 100, 200]
train_rmse_list = []
test_rmse_list = []
train_r2_list = []
test_r2_list = []

for n_estimators in n_estimators_list:
    print(f"\nStep 7 - Training Random Forest with n_estimators={n_estimators}...")
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    rf_pred_train = rf_model.predict(X_train)
    rf_pred_test = rf_model.predict(X_test)
    
    # Evaluate
    rf_rmse_train = np.sqrt(mean_squared_error(y_train, rf_pred_train))
    rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_pred_test))
    rf_r2_train = r2_score(y_train, rf_pred_train)
    rf_r2_test = r2_score(y_test, rf_pred_test)
    
    train_rmse_list.append(rf_rmse_train)
    test_rmse_list.append(rf_rmse_test)
    train_r2_list.append(rf_r2_train)
    test_r2_list.append(rf_r2_test)
    
    print(f"Step 7 - n_estimators={n_estimators}, Train RMSE: {rf_rmse_train:.4f}, Test RMSE: {rf_rmse_test:.4f}")
    print(f"Step 7 - n_estimators={n_estimators}, Train R²: {rf_r2_train:.4f}, Test R²: {rf_r2_test:.4f}")

# Use the best model (n_estimators=100 for consistency with original)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

# Save the final model
model_path = os.path.join(working_dir, f'cesium_model_rf_{timestamp}.pkl')
joblib.dump(rf_model, model_path)
print(f"Step 7 - Model saved to {model_path}")

# Final evaluation
rf_rmse_train = np.sqrt(mean_squared_error(y_train, rf_pred_train))
rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_pred_test))
rf_r2_train = r2_score(y_train, rf_pred_train)
rf_r2_test = r2_score(y_test, rf_pred_test)
print("\nStep 7 - Final Random Forest Results (n_estimators=100):")
print(f"Train RMSE: {rf_rmse_train:.4f}, R²: {rf_r2_train:.4f}")
print(f"Test RMSE: {rf_rmse_test:.4f}, R²: {rf_r2_test:.4f}")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'RF_Predicted': rf_pred_test
})
predictions_df.to_csv(f'rf_predictions_{timestamp}.csv', index=False)
print(f"\nStep 7 - Predictions saved to rf_predictions_{timestamp}.csv")

# Save feature importances
rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
rf_importance.to_csv(f'rf_feature_importance_{timestamp}.csv', index=False)
print(f"Step 7 - Feature importances saved to rf_feature_importance_{timestamp}.csv")

# Plot train vs test metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(n_estimators_list, train_rmse_list, label='Train RMSE', color='blue', marker='o')
plt.plot(n_estimators_list, test_rmse_list, label='Test RMSE', color='orange', marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('RMSE')
plt.title('Train vs Test RMSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_estimators_list, train_r2_list, label='Train R²', color='blue', marker='o')
plt.plot(n_estimators_list, test_r2_list, label='Test R²', color='orange', marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('R² Score')
plt.title('Train vs Test R²')
plt.legend()

plt.tight_layout()
plt.savefig(f'accuracy_plot_{timestamp}_rf.png')
plt.close()
print(f"Step 7 - Accuracy plot saved to accuracy_plot_{timestamp}_rf.png")

# Generate risk map
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    X_test['Longitude_(deg)'], 
    X_test['Latitude_(deg)'], 
    c=rf_pred_test, 
    cmap='viridis', 
    s=50, 
    alpha=0.7
)
plt.colorbar(sc, label='Predicted Activity Concentration')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Risk Map: Predicted Activity Concentration')
plt.tight_layout()
plt.savefig(f'risk_map_{timestamp}_rf.png')
plt.close()
print(f"Step 7 - Risk map saved to risk_map_{timestamp}_rf.png")