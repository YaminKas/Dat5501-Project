import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# ================================
# 1. Load cleaned student dataset
# ================================
students_path = "/Users/yaminkashim/Desktop/Project_data/Data/Processed/students_cleaned.csv"
print("Loading cleaned student dataset…")
students = pd.read_csv(students_path)
print(f"Student dataset loaded: {students.shape[0]} rows, {students.shape[1]} columns")

# ================================
# 2. Load IMD dataset
# ================================
imd_path = "/Users/yaminkashim/Desktop/Project_data/Data/Raw/populationbyimdenglandandwales2020.xlsx"
print("Loading IMD dataset…")

# Skip metadata rows; header is where the real table starts (adjust if needed)
imd = pd.read_excel(imd_path, header=12)  # Change 12 to the correct row if different

# Check columns
print("IMD columns detected:")
print(imd.columns.tolist())

# Rename columns for clarity (adjust names to match your Excel)
imd = imd.rename(columns={
    imd.columns[0]: "area_code",      # LSOA / area code column
    imd.columns[1]: "imd_score"       # IMD score column
})

# Keep only necessary columns
imd = imd[["area_code", "imd_score"]]

# Drop rows with missing area_code or imd_score
imd = imd.dropna(subset=["area_code", "imd_score"])

print(f"IMD dataset loaded: {imd.shape[0]} rows, {imd.shape[1]} columns")

# ================================
# 3. Merge student dataset with IMD
# ================================
# Ensure students have area_code column; if not, create dummy or real mapping
if "area_code" not in students.columns:
    students["area_code"] = np.arange(len(students))

students = students.merge(imd, on="area_code", how="left")

# ================================
# 4. Encode categorical columns
# ================================
cat_cols = students.select_dtypes(include=["object"]).columns
for col in cat_cols:
    students[col] = LabelEncoder().fit_transform(students[col])

# ================================
# 5. Impute missing numeric values
# ================================
numeric_cols = students.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy="median")
students[numeric_cols] = imputer.fit_transform(students[numeric_cols])

# ================================
# 6. Features & target
# ================================
X = students.drop("G3", axis=1)
y = students["G3"]

# ================================
# 7. Train/test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 8. Linear Regression
# ================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("=== Linear Regression Results (with IMD) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.2f}\n")

# ================================
# 9. Random Forest
# ================================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("=== Random Forest Results (with IMD) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}\n")

# ================================
# 10. Feature Importance
# ================================
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Random Forest Feature Importance ===")
print(feat_importance)

# ================================
# 11. Visualisations
# ================================
plt.figure(figsize=(10,6))
feat_importance[:15].plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("/Users/yaminkashim/Desktop/Project_data/Reports/figures/feature_importance.png")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Random Forest: Predicted vs Actual Grades")
plt.tight_layout()
plt.savefig("/Users/yaminkashim/Desktop/Project_data/Reports/figures/predicted_vs_actual.png")
plt.show()