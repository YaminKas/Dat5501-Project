import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load student data ---
students = pd.read_csv('/Users/yaminkashim/Desktop/Project_data/Data/Raw/student-mat.csv', sep=';')

# --- 2. Load ONS deprivation data ---
# Use 'ISO-8859-1' to avoid Unicode errors
deprivation = pd.read_csv(
    '/Users/yaminkashim/Desktop/Project_data/Data/Raw/populationbyimdenglandandwales2020.csv',
    encoding='ISO-8859-1'
)

# --- 3. Prepare deprivation data ---
# Example: select 'Area code' and 'IMD score' (update column names as needed)
deprivation = deprivation[['Area code (2020)', 'IMD score (where 1=most deprived)']]
deprivation.columns = ['area_code', 'imd_score']

# --- 4. Merge datasets ---
# Since student data does not have area codes, we'll create a random mapping for demonstration
np.random.seed(42)
students['area_code'] = np.random.choice(deprivation['area_code'], size=len(students))

# Merge IMD score into students
students = students.merge(deprivation, on='area_code', how='left')

# --- 5. Encode categorical columns ---
cat_cols = students.select_dtypes(include=['object']).columns
for col in cat_cols:
    students[col] = LabelEncoder().fit_transform(students[col])

# --- 6. Features & target ---
X = students.drop('G3', axis=1)
y = students['G3']

# --- 7. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 8. Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("=== Linear Regression Results (with IMD) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.2f}\n")

# --- 9. Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("=== Random Forest Results (with IMD) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}\n")

# --- 10. Feature Importance ---
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Random Forest Feature Importance ===")
print(feat_importance)

# --- 11. Visualisations ---
plt.figure(figsize=(10,6))
feat_importance[:15].plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig('/Users/yaminkashim/Desktop/Project_data/Reports/figures/feature_importance.png')
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Random Forest: Predicted vs Actual Grades")
plt.tight_layout()
plt.savefig('/Users/yaminkashim/Desktop/Project_data/Reports/figures/predicted_vs_actual.png')
plt.show()