# step4_encode_categorical.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Set working directory
working_dir = 'F:/project details'
print("Step 4 - Current Working Directory:", os.getcwd())
input_csv = os.path.join(working_dir, 'CsDB_ver1.1_numeric_cleaned.csv')
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"{input_csv} not found! Run Step 3 first.")

# Load CSV
csdb = pd.read_csv(input_csv, low_memory=False)

# Define categorical columns
categorical_cols = [
    'Variance_type_inv', 'Wild_Grown_mushroom', 'Leaf/ring_age', 'Slope_aspect',
    'Position_on_slope', 'Thickness_type_litter', 'Leaf_habit',
    'Mushroom_animal_name', 'Parts_animal', 'Mushroom_animal_scientific_name',
    'Parts', 'Species', 'Soil_type'
]
print("\nStep 4 - Categorical columns:")
for col in categorical_cols:
    if col in csdb.columns:
        print(f"\nUnique values in {col}:")
        print(csdb[col].value_counts(dropna=False).head(10))

# Encode categorical columns
for col in categorical_cols:
    if col in csdb.columns:
        le = LabelEncoder()
        csdb[f'{col}_encoded'] = le.fit_transform(csdb[col].astype(str).fillna(csdb[col].mode()[0]))
        print(f"\nStep 4 - Encoded {col} to {col}_encoded")

# Verify
print("\nStep 4 - First 5 Rows (Selected Categorical and Encoded Columns):")
for col in categorical_cols:
    if col in csdb.columns and f'{col}_encoded' in csdb.columns:
        print(f"\n{col} and {col}_encoded:")
        print(csdb[[col, f'{col}_encoded']].head())
print("\nStep 4 - Dataset Info:")
print(csdb.info())

# Save result
output_csv = os.path.join(working_dir, 'CsDB_ver1.1_encoded.csv')
csdb.to_csv(output_csv, index=False)
print(f"\nStep 4 - {output_csv} created successfully.")