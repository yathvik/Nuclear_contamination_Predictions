import pandas as pd
import os
from datetime import datetime

# Set working directory
working_dir = 'F:\\project detail'
os.chdir(working_dir)
print("Step 1 - Current Working Directory:", os.getcwd())

# Load Excel file
try:
    data = pd.read_excel('CsDB-ver1.1.xlsx', engine='openpyxl')
    print("Step 1 - Excel file loaded successfully:", data.shape)
except FileNotFoundError:
    print("Error: CsDB-ver1.1.xlsx not found in F:\\project detail")
    exit()

# Save to CSV, preserving NaN/null
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data.to_csv(f'CsDB_ver1.1_dataset_{timestamp}.csv', index=False)
print(f"Step 1 - Dataset saved to CsDB_ver1.1_dataset_csv_file{timestamp}.csv")