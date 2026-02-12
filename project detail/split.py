import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from glob import glob
from datetime import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import seaborn as sns


working_dir = 'F:\\project detail'
os.chdir(working_dir)
print("Step 6 - Current Working Directory:", os.getcwd())

# Load latest cleaned dataset
cleaned_files = glob(os.path.join(working_dir, 'CsDB_ver1.1_cleaned_*.csv'))
if not cleaned_files:
    print("Error: No cleaned dataset found. Run Step 5.")
    exit()

latest_file = max(cleaned_files, key=os.path.getctime)
print(f"Using dataset: {latest_file}")

data = pd.read_csv(latest_file, low_memory=False)
print("Step 6 - Dataset loaded:", data.shape)

# Ensure all required columns exist
required_cols = ['Latitude_(deg)', 'Longitude_(deg)', 'Activity_concentration', 
                 'Distance_from_FDNPP', 'Total_biomass', 'Soil_type_encoded']
for col in required_cols:
    if col not in data.columns:
        print(f"Warning: {col} missing, creating empty column.")
        data[col] = pd.NA

# Optional feature: Tree_height * Elevation
if 'Tree_height' in data.columns and 'Elevation' in data.columns:
    data['Height_Elevation'] = data['Tree_height'] * data['Elevation']
    print("Step 6 - Added Height_Elevation")

# Add missing-indicator features
for col in data.columns:
    if col != 'Activity_concentration':
        data[f'{col}_is_missing'] = data[col].isna().astype(int)
print("Step 6 - Added missing-indicator features")

# Prepare features and target
X = data.drop(columns=['Activity_concentration'])
y = data['Activity_concentration']

# Train/test split
train_size = int(0.8 * len(data))
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]
print(f"Step 6 - Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.fillna(0))
X_test_scaled = scaler.transform(X_test.fillna(0))
X_train_scaled = np.where(X_train.isna(), np.nan, X_train_scaled)
X_test_scaled = np.where(X_test.isna(), np.nan, X_test_scaled)

# Save scaled datasets
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f'X_train_{timestamp}.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f'X_test_{timestamp}.csv', index=False)
pd.DataFrame(y_train, columns=['Activity_concentration']).to_csv(f'y_train_{timestamp}.csv', index=False)
pd.DataFrame(y_test, columns=['Activity_concentration']).to_csv(f'y_test_{timestamp}.csv', index=False)
print(f"Step 6 - Saved: X_train_{timestamp}.csv, X_test_{timestamp}.csv, y_train_{timestamp}.csv, y_test_{timestamp}.csv")

# Plot histogram of Activity_concentration
plt.figure(figsize=(10,6))
sns.histplot(data['Activity_concentration'].dropna(), bins=50, kde=True, color='skyblue')
plt.title('Distribution of Activity_concentration')
plt.xlabel('Activity_concentration')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('activity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()


# Optional: scatter plot by location
if 'Latitude_(deg)' in data.columns and 'Longitude_(deg)' in data.columns:
    plt.figure(figsize=(10,6))
    sc = plt.scatter(data['Longitude_(deg)'], data['Latitude_(deg)'], 
                     c=data['Activity_concentration'], cmap='viridis', s=20)
    plt.colorbar(sc, label='Activity_concentration')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Distribution of Activity_concentration')
    plt.savefig('activity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

