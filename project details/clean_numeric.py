# step3_clean_numeric.py
import pandas as pd
import numpy as np
import torch
import os

# Set working directory
working_dir = 'F:\project details'
print("Step 3 - Current Working Directory:", os.getcwd())
input_csv = os.path.join(working_dir, 'CsDB_ver1.1_dropped.csv')
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"{input_csv} not found! Run Step 2 first.")

# Load CSV
csdb = pd.read_csv(input_csv, low_memory=False)

# Define numeric columns (exclude categorical columns from Step 4 output)
categorical_cols = [
    'Variance_type_inv', 'Wild_Grown_mushroom', 'Leaf/ring_age', 'Slope_aspect',
    'Position_on_slope', 'Thickness_type_litter', 'Leaf_habit',
    'Mushroom_animal_name', 'Parts_animal', 'Mushroom_animal_scientific_name',
    'Parts', 'Species', 'Soil_type'
]
numeric_cols = [
    'Activity_concentration', 'Tree_height', 'Sampling_month', 'Elevation', 'Stand_age',
    'Variance_inv', 'Minimum_inv', 'Upper_sampling_soil_depth', 'Variance_act',
    'Minimum_act', 'Sampling_year', 'Inventory_Soil_Tag', 'Diameter_at_breast_height',
    'Lower_sampling_height', 'N_sample_act', 'LAI', 'Fresh_weight_basis',
    'Upper_sampling_height', 'N_inv'
]
# Include mixed-type columns from DtypeWarning, excluding categorical
mixed_cols_indices = [21, 24, 25, 26, 27, 29, 30, 31, 44, 45, 46, 48, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 65, 66, 67, 68, 71]
mixed_cols = [csdb.columns[idx] for idx in mixed_cols_indices if idx < len(csdb.columns) and csdb.columns[idx] not in categorical_cols]
numeric_cols.extend(mixed_cols)
numeric_cols = list(set(numeric_cols))  # Remove duplicates
non_numeric_values = ['ND', 'N/A', '', 'r28-4996', '<5', '<5.0', '<10', 'unknown', 'na', 'null', 'NaN', '-', '<15']

# Clean and impute numeric columns
for col in numeric_cols:
    if col in csdb.columns:
        print(f"\nStep 3 - Unique values in {col} before cleaning:")
        print(csdb[col].value_counts(dropna=False).head(10))
        csdb[col] = pd.to_numeric(csdb[col].replace(non_numeric_values, np.nan), errors='coerce')
        print(f"Step 3 - NaN count in {col} before imputation:", csdb[col].isna().sum())
        col_tensor = torch.tensor(csdb[col].values, dtype=torch.float64)
        col_median = torch.nanmedian(col_tensor)
        if torch.isnan(col_median):
            col_median = torch.tensor(0.0, dtype=torch.float64)
        csdb[col] = torch.where(torch.isnan(col_tensor), col_median, col_tensor).numpy()
        print(f"Step 3 - NaN or Inf in {col}:")
        print("NaN count:", csdb[col].isna().sum())
        print("Inf count:", np.isinf(csdb[col]).sum())

# Log-transform Activity_concentration
if 'Activity_concentration' in csdb.columns:
    activity_tensor = torch.tensor(csdb['Activity_concentration'].values, dtype=torch.float64)
    activity_tensor = torch.where(activity_tensor < 0, torch.tensor(0.0, dtype=torch.float64), activity_tensor)
    csdb['Activity_concentration'] = torch.log1p(activity_tensor + 1).numpy()
    print("\nStep 3 - NaN or Inf in Activity_concentration after log-transform:")
    print("NaN count:", csdb['Activity_concentration'].isna().sum())
    print("Inf count:", np.isinf(csdb['Activity_concentration']).sum())

# Verify
print("\nStep 3 - Missing Values in Numeric Columns:")
print(csdb[numeric_cols].isnull().sum())

# Save result
output_csv = os.path.join(working_dir, 'CsDB_ver1.1_numeric_cleaned.csv')
csdb.to_csv(output_csv, index=False)
print(f"\nStep 3 - {output_csv} created successfully.")