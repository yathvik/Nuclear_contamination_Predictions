import pandas as pd
import numpy as np
import os
from glob import glob
from datetime import datetime

working_dir = 'F:\\project detail'
os.chdir(working_dir)
print("Step 3 - Current Working Directory:", os.getcwd())

cleaned_files = glob(os.path.join(working_dir, 'CsDB_ver1.1_initial_cleaned_*.csv'))
if not cleaned_files:
    print("Error: No cleaned dataset found. Run Step 2.")
    exit()
latest_file = max(cleaned_files, key=os.path.getctime)
print(f"Using dataset: {latest_file}")

try:
    data = pd.read_csv(latest_file, low_memory=False)
    print("Step 3 - Dataset loaded:", data.shape)
except FileNotFoundError:
    print(f"Error: {latest_file} not found.")
    exit()

important_cols = [
    'Latitude_(deg)', 'Longitude_(deg)', 'Activity_concentration', 'Distance_from_FDNPP',
    'Total_biomass', 'Tree_height', 'Elevation', 'Soil_type', 'Tree_scientific_name',
    'Mushroom_animal_name', 'Variance_type_inv', 'Wild_Grown_mushroom', 'Leaf/ring_age',
    'Slope_aspect', 'Position_on_slope', 'Thickness_type_litter', 'Leaf_habit',
    'Parts_animal', 'Mushroom_animal_scientific_name', 'Parts', 'Species'
]
available_cols = [col for col in important_cols if col in data.columns]
data = data[available_cols]
print(f"Step 3 - Selected columns: {available_cols}")

if 'Activity_concentration' in data.columns:
    data['Activity_concentration'] = np.log1p(data['Activity_concentration'].where(data['Activity_concentration'].notna(), 0))
    print("Step 3 - Log-transformed Activity_concentration")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data.to_csv(f'CsDB_ver1.1_selected_{timestamp}.csv', index=False)
print(f"Step 3 - Saved to CsDB_ver1.1_selected_{timestamp}.csv")