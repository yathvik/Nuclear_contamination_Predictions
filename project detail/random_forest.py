import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import os
from datetime import datetime
import joblib  # For saving the model

working_dir = 'F:\\project detail'
os.chdir(working_dir)

# Load latest train/test files
X_train_file = max(glob('X_train_*.csv'), key=os.path.getctime)
X_test_file = max(glob('X_test_*.csv'), key=os.path.getctime)
y_train_file = max(glob('y_train_*.csv'), key=os.path.getctime)
y_test_file = max(glob('y_test_*.csv'), key=os.path.getctime)

X_train = pd.read_csv(X_train_file).fillna(-999)
X_test = pd.read_csv(X_test_file).fillna(-999)
y_train = pd.read_csv(y_train_file)['Activity_concentration'].fillna(0)
y_test = pd.read_csv(y_test_file)['Activity_concentration'].fillna(0)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Metrics
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

# Save predictions
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rf_pred_df = pd.DataFrame({'Actual': y_test, 'RF_Predicted': rf_pred})
rf_pred_df.to_csv(f'rf_predictions_{timestamp}.csv', index=False)

# Save the trained model
joblib.dump(rf_model, f'rf_model_{timestamp}.joblib')

# Plot Actual vs Predicted
plt.figure()
plt.scatter(y_test, rf_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest - Actual vs Predicted')
plt.savefig(f'rf_accuracy_{timestamp}.png')
plt.close()

# Plot MSE and R²
plt.figure()
metrics = {'MSE': rf_mse, 'R²': rf_r2}
plt.bar(metrics.keys(), metrics.values(), color=['orange', 'green'])
plt.title('Random Forest Model Metrics')
plt.ylabel('Value')
plt.savefig(f'rf_metrics_{timestamp}.png')
plt.close()
