import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from glob import glob
from datetime import datetime

working_dir = 'F:\\project detail'
os.chdir(working_dir)
print("Step 4 - Current Working Directory:", os.getcwd())

selected_files = glob(os.path.join(working_dir, 'CsDB_ver1.1_selected_*.csv'))
if not selected_files:
    print("Error: No selected dataset found. Run Step 3.")
    exit()
latest_file = max(selected_files, key=os.path.getctime)
print(f"Using dataset: {latest_file}")

try:
    data = pd.read_csv(latest_file, low_memory=False)
    print("Step 4 - Dataset loaded:", data.shape)
except FileNotFoundError:
    print(f"Error: {latest_file} not found.")
    exit()

categorical_cols = [
    'Soil_type', 'Tree_scientific_name', 'Mushroom_animal_name', 'Variance_type_inv',
    'Wild_Grown_mushroom', 'Leaf/ring_age', 'Slope_aspect', 'Position_on_slope',
    'Thickness_type_litter', 'Leaf_habit', 'Parts_animal', 'Mushroom_animal_scientific_name',
    'Parts', 'Species'
]
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
        data = data.drop(columns=[col])
        print(f"Step 4 - Encoded {col} to {col}_encoded")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data.to_csv(f'CsDB_ver1.1_encoded_{timestamp}.csv', index=False)
print(f"Step 4 - Saved to CsDB_ver1.1_encoded_{timestamp}.csv")