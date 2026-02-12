# step5_clean_impute.py
import pandas as pd
import numpy as np
import torch
import os

# Set working directory
working_dir = 'F:\project details'
print("Step 5 - Current Working Directory:", os.getcwd())
input_csv = os.path.join(working_dir, 'CsDB_ver1.1_encoded.csv')
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"{input_csv} not found! Run Step 4 first.")

# Load CSV
csdb = pd.read_csv(input_csv, low_memory=False)

# Inspect non-numeric columns
print("\nStep 5 - Non-numeric columns before cleaning:")
non_numeric_cols = csdb.select_dtypes(include=['object']).columns
for col in non_numeric_cols:
    print(f"\nColumn: {col}")
    print(csdb[col].value_counts(dropna=False).head(10))
    # Handle special case for Total_biomass (e.g., '254/68')
    if col == 'Total_biomass':
        csdb[col] = csdb[col].apply(lambda x: float(x.split('/')[0]) if isinstance(x, str) and '/' in x else x)
    csdb[col] = pd.to_numeric(csdb[col], errors='coerce')

# Impute all numeric columns
numeric_cols = csdb.select_dtypes(include=[np.number]).columns
print("\nStep 5 - Numeric columns:", list(numeric_cols))
data_tensor = torch.tensor(csdb[numeric_cols].values, dtype=torch.float64)
print("\nStep 5 - NaN count before imputation:", torch.isnan(data_tensor).sum().item())
median_values = torch.nanmedian(data_tensor, dim=0).values
median_values[torch.isnan(median_values)] = 0.0  # Fallback for all-NaN columns
data_tensor = torch.where(torch.isnan(data_tensor), median_values, data_tensor)
if torch.isinf(data_tensor).any():
    print("\nStep 5 - Warning: Infinite values detected")
    data_tensor[torch.isinf(data_tensor)] = median_values.repeat(data_tensor.shape[0], 1)[torch.isinf(data_tensor)]
csdb[numeric_cols] = data_tensor.numpy()

# Verify
print("\nStep 5 - NaN or Inf in numeric columns:")
print("NaN count:", csdb[numeric_cols].isna().sum().sum())
print("Inf count:", np.isinf(csdb[numeric_cols]).sum().sum())
print("\nStep 5 - Dataset Info:")
print(csdb.info())

# Save result
output_csv = os.path.join(working_dir, 'CsDB_ver1.1_cleaned.csv')
csdb.to_csv(output_csv, index=False)
print(f"\nStep 5 - {output_csv} created successfully.")