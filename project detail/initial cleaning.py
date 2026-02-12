import pandas as pd
import os
from glob import glob
from datetime import datetime

working_dir = 'F:\\project detail'
os.chdir(working_dir)
print("Step 2 - Current Working Directory:", os.getcwd())

dataset_files = glob(os.path.join(working_dir, 'CsDB_ver1.1_dataset_*.csv'))
if not dataset_files:
    print("Error: No dataset found. Run Step 1.")
    exit()
latest_file = max(dataset_files, key=os.path.getctime)
print(f"Using dataset: {latest_file}")

try:
    data = pd.read_csv(latest_file)
    print("Step 2 - Dataset loaded:", data.shape)
except FileNotFoundError:
    print(f"Error: {latest_file} not found.")
    exit()

def clean_numeric(value):
    if isinstance(value, str):
        try:
            if '/' in value:
                return float(value.split('/')[0])
            if value in ['ND', '<5', '']:
                return pd.NA
            return float(value)
        except ValueError:
            return pd.NA
    return value

numeric_cols = ['Distance_from_FDNPP', 'Total_biomass', 'Tree_height', 'Elevation', 'Activity_concentration']
for col in numeric_cols:
    if col in data.columns:
        data[col] = data[col].apply(clean_numeric)
        print(f"Step 2 - Cleaned {col}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data.to_csv(f'CsDB_ver1.1_initial_cleaned_{timestamp}.csv', index=False)
print(f"Step 2 - Saved to CsDB_ver1.1_initial_cleaned_{timestamp}.csv")