import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ================================
# Load Student Maths Dataset
# ================================
students_path = '/Users/yaminkashim/Desktop/Project_data/Data/Raw/student-mat.csv'
students = pd.read_csv(students_path, sep=';')

print("=== Student Data Head ===")
print(students.head(), "\n")

print("=== Student Data Info ===")
print(students.info(), "\n")

print("=== Student Data Description ===")
print(students.describe(include='all'), "\n")

# Check missing values
print("=== Missing Values per Column ===")
print(students.isnull().sum(), "\n")

# Check unique values per column
print("=== Unique Values per Column ===")
print(students.nunique(), "\n")

# ================================
# Load UK Deprivation Dataset (ONS IMD)
# ================================
# Updated to read Excel directly
deprivation_path = '/Users/yaminkashim/Desktop/Project_data/Data/Raw/populationbyimdenglandandwales2020.xlsx'

deprivation = pd.read_excel(
    deprivation_path,
    sheet_name=0,       # first sheet
    engine='openpyxl'   # Excel engine
)

print("=== Deprivation Data Head ===")
print(deprivation.head(), "\n")

print("=== Deprivation Data Info ===")
print(deprivation.info(), "\n")

# ================================
# Basic Preprocessing for Students
# ================================

# Encode categorical columns for modelling
categorical_cols = students.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    students[col] = le.fit_transform(students[col])
    label_encoders[col] = le

print("=== Encoded Student Data Head ===")
print(students.head(), "\n")

# ================================
# Optional: Merge with Deprivation Data
# ================================
# Example merge if deprivation has a column like 'Local Authority' and students have 'school'
# You may need to rename columns to match
# merged_data = pd.merge(students, deprivation, left_on='school', right_on='LocalAuthority', how='left')

# ================================
# Save cleaned student data for modelling
# ================================
students.to_csv('/Users/yaminkashim/Desktop/Project_data/Data/Processed/students_cleaned.csv', index=False)
print("Cleaned student data saved!")

# ================================
# Ready for predictive modelling
# ================================
# Example target: G3 (final grade)
X = students.drop('G3', axis=1)
y = students['G3']

print("Feature shape:", X.shape)
print("Target shape:", y.shape)