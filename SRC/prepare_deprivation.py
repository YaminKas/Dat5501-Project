import pandas as pd

# Load deprivation data from Excel
deprivation = pd.read_excel(
    '/Users/yaminkashim/Desktop/Project_data/Data/Raw/populationbyimdenglandandwales2020.xlsx',
    sheet_name=0  # Adjust if your data is on a different sheet
)

# Quick look at the data
print("=== Deprivation Data Head ===")
print(deprivation.head())

print("\n=== Deprivation Data Info ===")
print(deprivation.info())

# Optional: clean up columns, rename, or filter if needed
# Example: keep only columns you want
# deprivation = deprivation[['Region', 'IMD_Rank', 'Population']]  # adjust as needed

# Save cleaned version for your model
deprivation.to_csv(
    '/Users/yaminkashim/Desktop/Project_data/Data/Processed/populationbyimd_clean.csv',
    index=False
)
print("Deprivation data cleaned and saved.")