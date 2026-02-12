
import pandas as pd
import os

# Set working directory
working_dir = 'F:\project details'
print("Step 2 - Current Working Directory:", os.getcwd())
input_csv = ('CsDB_ver1.1_dataset.csv')
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"{input_csv} not found! Run Step 1 first.")

# Load CSV
csdb = pd.read_csv(input_csv, low_memory=False)

# Drop non-useful columns (update based on Step 1 mixed-type columns)
drop_cols = ['Sample_ID', 'Record_number']  # Add others from Step 1 output
print("\nStep 2 - Columns to drop:", drop_cols)
csdb = csdb.drop(columns=[col for col in drop_cols if col in csdb.columns], errors='ignore')

# Verify
print("\nStep 2 - Dataset Info after dropping columns:")
print(csdb.info())
print(f"\nStep 2 - Dataset Shape: {csdb.shape}")

# Save result
output_csv = os.path.join(working_dir, 'CsDB_ver1.1_dropped.csv')
csdb.to_csv(output_csv, index=False)
print(f"\nStep 2 - {output_csv} created successfully.")