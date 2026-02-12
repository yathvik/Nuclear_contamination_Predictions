# step6_prepare_data.py
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os

# Set working directory
working_dir = 'F:\project details'
print("Step 6 - Current Working Directory:", os.getcwd())
input_csv = os.path.join(working_dir, 'CsDB_ver1.1_cleaned.csv')
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"{input_csv} not found! Run Step 5 first.")

# Load CSV
csdb = pd.read_csv(input_csv, low_memory=False)

# Create interaction feature
if 'Elevation' in csdb.columns and 'Tree_height' in csdb.columns:
    csdb['Height_Elevation'] = csdb['Tree_height'] * csdb['Elevation']

# Select features and target
encoded_cols = [col for col in csdb.columns if col.endswith('_encoded')]
numeric_cols = csdb.select_dtypes(include=[np.number]).columns
feature_cols = list(set(list(numeric_cols) + encoded_cols) - {'Activity_concentration'})
X = csdb[feature_cols]
y = csdb['Activity_concentration']

# Check for NaN/Inf
print("\nStep 6 - NaN or Inf in X and y:")
print("X NaN count:", X.isna().sum().sum())
print("X Inf count:", np.isinf(X).sum().sum())
print("y NaN count:", y.isna().sum())
print("y Inf count:", np.isinf(y).sum())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float64)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float64)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float64)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float64)

# Scale features
mean = X_train_tensor.mean(dim=0)
std = X_train_tensor.std(dim=0)
std[std < 1e-6] = 1  # Prevent division by zero
X_train_tensor = (X_train_tensor - mean) / std
X_test_tensor = (X_test_tensor - mean) / std

# Check for NaN/Inf
print("\nStep 6 - NaN or Inf after scaling:")
print("X_train NaN count:", torch.isnan(X_train_tensor).sum().item())
print("X_train Inf count:", torch.isinf(X_train_tensor).sum().item())
print("X_test NaN count:", torch.isnan(X_test_tensor).sum().item())
print("X_test Inf count:", torch.isinf(X_test_tensor).sum().item())

# Save processed data
X_train_df = pd.DataFrame(X_train_tensor.numpy(), columns=X.columns)
X_test_df = pd.DataFrame(X_test_tensor.numpy(), columns=X.columns)
output_x_train = os.path.join(working_dir, 'X_train.csv')
output_x_test = os.path.join(working_dir, 'X_test.csv')
output_y_train = os.path.join(working_dir, 'y_train.csv')
output_y_test = os.path.join(working_dir, 'y_test.csv')
X_train_df.to_csv(output_x_train, index=False)
X_test_df.to_csv(output_x_test, index=False)
pd.Series(y_train_tensor.numpy(), name='Activity_concentration').to_csv(output_y_train, index=False)
pd.Series(y_test_tensor.numpy(), name='Activity_concentration').to_csv(output_y_test, index=False)

# Verify
print("\nStep 6 - Training set shape:", X_train_df.shape)
print("Step 6 - Test set shape:", X_test_df.shape)
print("\nStep 6 - X_train Info:")
print(X_train_df.info())
print(f"\nStep 6 - Output files created: {output_x_train}, {output_x_test}, {output_y_train}, {output_y_test}")