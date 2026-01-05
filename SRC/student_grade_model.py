import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load cleaned student data
students = pd.read_csv('/Users/yaminkashim/Desktop/Project_data/Data/Raw/student-mat.csv', sep=';')

# Encode categorical columns
cat_cols = students.select_dtypes(include=['object']).columns
for col in cat_cols:
    students[col] = LabelEncoder().fit_transform(students[col])

# Features & target
X = students.drop('G3', axis=1)
y = students['G3']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL 1: Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("=== Linear Regression Results ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.2f}\n")

# --- MODEL 2: Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("=== Random Forest Results ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}\n")

# Optional: Feature importance
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Random Forest Feature Importance ===")
print(feat_importance)