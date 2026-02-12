import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import os
from datetime import datetime
import joblib

# --- Config ---
working_dir = 'F:\\project detail'
os.chdir(working_dir)

# --- Find latest splits ---
X_train_file = max(glob('X_train_*.csv'), key=os.path.getctime)
X_test_file  = max(glob('X_test_*.csv'),  key=os.path.getctime)
y_train_file = max(glob('y_train_*.csv'), key=os.path.getctime)
y_test_file  = max(glob('y_test_*.csv'),  key=os.path.getctime)

# --- Load (no filling NaNs; XGBoost handles them) ---
X_train = pd.read_csv(X_train_file)
X_test  = pd.read_csv(X_test_file)
y_train = pd.read_csv(y_train_file)['Activity_concentration']
y_test  = pd.read_csv(y_test_file)['Activity_concentration']

# --- Train XGBoost ---
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# --- Predictions ---
xgb_pred = xgb_model.predict(X_test)

# --- Metrics ---
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"XGBoost - MSE: {xgb_mse:.4f}, R²: {xgb_r2:.4f}")

# --- Save results ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pred_df = pd.DataFrame({'Actual': y_test, 'XGB_Predicted': xgb_pred})
pred_df.to_csv(f'xgb_predictions_{timestamp}.csv', index=False)

# Save model
joblib.dump(xgb_model, f'xgb_model_{timestamp}.joblib')
xgb_model.save_model(f'xgb_model_{timestamp}.json')

# --- Graphs ---
# 1. Actual vs Predicted
plt.figure()
plt.scatter(y_test, xgb_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('XGBoost - Actual vs Predicted')
plt.savefig(f'xgb_accuracy_{timestamp}.png')
plt.close()

# 2. Metrics Bar Plot
plt.figure()
metrics = {'MSE': xgb_mse, 'R²': xgb_r2}
plt.bar(metrics.keys(), metrics.values(), color=['orange', 'green'])
plt.title('XGBoost Model Metrics')
plt.ylabel('Value')
plt.savefig(f'xgb_metrics_{timestamp}.png')
plt.close()

# 3. Feature Importance
plt.figure(figsize=(8, 6))
importances = xgb_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1][:20]  # top 20
plt.barh(np.array(X_train.columns)[sorted_idx][::-1],
         importances[sorted_idx][::-1], color='skyblue')
plt.xlabel('Importance')
plt.title('XGBoost - Top 20 Feature Importances')
plt.tight_layout()
plt.savefig(f'xgb_feature_importance_{timestamp}.png')
plt.close()
