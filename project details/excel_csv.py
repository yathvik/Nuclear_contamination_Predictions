import pandas as pd

# Converting Excel to CSV
# Specify the path to your Excel file
excel_file = 'CsDB-ver1.1.xlsx'
output_csv = 'CsDB_ver1.1_dataset.csv'

# Read the Excel file (assuming data is in the 'CsDB' sheet)
csdb = pd.read_excel(excel_file, sheet_name='CsDB')

# Save to CSV
csdb.to_csv(output_csv, index=False)

# Verifying the conversion
# Read the CSV to check its contents
csdb_csv = pd.read_csv(output_csv)

# Print basic information to confirm
print("CSV File Info:")
print(csdb_csv.info())
print("\nFirst 5 Rows of CSV:")
print(csdb_csv.head())
print(f"\nDataset Shape: {csdb_csv.shape}")
print("\nMissing Values in CSV:")
print(csdb_csv.isnull().sum())