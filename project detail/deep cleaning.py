import pandas as pd
import os
from glob import glob
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Set working directory
working_dir = r'F:\project detail'
os.chdir(working_dir)
print("Step 5 - Current Working Directory:", os.getcwd())

# Find the latest encoded file
encoded_files = glob(os.path.join(working_dir, 'CsDB_ver1.1_encoded_*.csv'))
if not encoded_files:
    print("Error: No encoded dataset found. Run Step 4.")
    exit()

latest_file = max(encoded_files, key=os.path.getctime)
print(f"Using dataset: {latest_file}")

# Load dataset
try:
    data = pd.read_csv(latest_file, low_memory=False)
    print("Step 5 - Dataset loaded:", data.shape)
except FileNotFoundError:
    print(f"Error: {latest_file} not found.")
    exit()

# Reintroduce Soil_type from initial_cleaned dataset if missing
initial_cleaned_files = glob(os.path.join(working_dir, 'CsDB_ver1.1_initial_cleaned_*.csv'))
if initial_cleaned_files:
    initial_cleaned_file = max(initial_cleaned_files, key=os.path.getctime)
    initial_data = pd.read_csv(initial_cleaned_file, low_memory=False)
    if 'Soil_type' in initial_data.columns and 'Soil_type_encoded' not in data.columns:
        data['Soil_type'] = initial_data['Soil_type']
        le = LabelEncoder()
        data['Soil_type_encoded'] = le.fit_transform(data['Soil_type'].fillna('Unknown'))
        data = data.drop(columns=['Soil_type'])
        print("Step 5 - Reintroduced and encoded Soil_type to Soil_type_encoded")
    else:
        print("Step 5 - Soil_type already present or not found in initial_cleaned dataset")

# Clean Total_biomass column
def clean_total_biomass(value):
    if isinstance(value, str):
        try:
            if '/' in value:
                return float(value.split('/')[0])
            return float(value)
        except ValueError:
            return pd.NA
    return value

if 'Total_biomass' in data.columns:
    data['Total_biomass'] = data['Total_biomass'].apply(clean_total_biomass)
    print("Step 5 - Cleaned Total_biomass")

# Drop low-variance columns
variance = data.var(numeric_only=True)
low_variance_cols = variance[variance < 1e-6].index
low_variance_cols = [col for col in low_variance_cols if col != 'Soil_type_encoded']
if low_variance_cols:
    print(f"Step 5 - Dropping low-variance columns: {low_variance_cols}")
    data = data.drop(columns=low_variance_cols)

# Check required columns
required_cols = ['Latitude_(deg)', 'Longitude_(deg)', 'Activity_concentration', 
                 'Distance_from_FDNPP', 'Total_biomass', 'Soil_type_encoded']

missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    print(f"Warning: Missing required columns: {missing_cols}. You may need to fix Step 4 or initial_cleaned dataset.")
else:
    print("Step 5 - All required columns present.")

# Save cleaned dataset
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'CsDB_ver1.1_cleaned_{timestamp}.csv'
data.to_csv(output_file, index=False)
print(f"Step 5 - Saved to {output_file}")
