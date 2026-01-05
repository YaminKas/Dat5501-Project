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
# 1. Load Student Maths Dataset
# ================================
students_path = '/Users/yaminkashim/Desktop/Project_data/Data/Raw/student-mat.csv'
students = pd.read_csv(students_path, sep=';')

# ================================
# 2. Generate synthetic IMD data
# ================================
# Simulate deprivation scores from 1 to 32
np.random.seed(42)
students['imd_score'] = np.random.randint(1, 33, size=len(students))

# Create a dummy area_code (optional)
students['area_code'] = np.arange(len(students))

# ================================
# 3. Encode categorical columns
# ================================
cat_cols = students.select_dtypes(include=['object']).columns
for col in cat_cols:
    students[col] = LabelEncoder().fit_transform(students[col])

# ================================
# 4. Impute any remaining missing numeric values
# ================================
numeric_cols = students.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
students[numeric_cols] = imputer.fit_transform(students[numeric_cols])

# ================================
# 5. Features & target
# ================================
X = students.drop('G3', axis=1)
y = students['G3']

# ================================
# 6. Train/test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 7. Linear Regression
# ================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("=== Linear Regression Results (with IMD) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.2f}\n")

# ================================
# 8. Random Forest
# ================================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("=== Random Forest Results (with IMD) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}\n")

# ================================
# 9. Feature Importance
# ================================
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Random Forest Feature Importance ===")
print(feat_importance)

# ================================
# 10. Visualisations
# ================================
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