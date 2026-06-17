"""
AI Analyst Exercise — Exercise 1
Task: Loan Interest Rate Prediction
------------------------------------
Use this starter script as your foundation.
Extend, improve, and complete the analysis as described in the README.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("exercise_1\data\loan_applications.csv")
print("Shape:", df.shape)
print(df.dtypes)
print(df.head())


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic statistics:\n", df.describe())

# Distribution of target variable
plt.figure(figsize=(8, 4))
sns.histplot(df['interest_rate'], bins=40, kde=True)
plt.title("Interest Rate Distribution")
plt.xlabel("Interest Rate (%)")
plt.tight_layout()
plt.savefig("interest_rate_dist.png")
plt.show()

# Loan type imbalance
loan_counts = df['loan_type'].value_counts()
print("\nLoan Type Distribution:\n", loan_counts)

# BUG 1: Using df.groupby().mean() to compute average interest rate per loan type,
# but the result is never reset_index()'d — the bar plot receives a Series with a
# MultiIndex, causing the x-axis labels to display incorrectly / silently fail.
avg_rate_by_type = df.groupby('loan_type')['interest_rate'].mean()
plt.figure(figsize=(8, 4))
avg_rate_by_type.plot(kind='bar', color='coral')
plt.title("Avg Interest Rate by Loan Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("avg_rate_by_type.png")
plt.show()


# ─────────────────────────────────────────────
# 3. OUTLIER DETECTION & CLEANING
# ─────────────────────────────────────────────
# Remove physically impossible values first
df = df[df['years_employed'] >= 0]
df = df[df['annual_income'] > 0]

# IQR on interest_rate
Q1 = df['interest_rate'].quantile(0.25)
Q3 = df['interest_rate'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"\nOutlier bounds: [{lower:.2f}, {upper:.2f}]")
print(f"Outliers found: {((df['interest_rate'] < lower) | (df['interest_rate'] > upper)).sum()}")

df_clean = df[(df['interest_rate'] >= lower) & (df['interest_rate'] <= upper)].copy()
print(f"Rows after cleaning: {len(df_clean)}")

# Also cap credit score to valid range
# BUG 2: The condition clips credit_score values that are OUT of range — but the
# comparison direction is wrong. Values below 300 or above 850 should be dropped
# or capped, but this line keeps only the invalid ones (inverted filter).
df_clean = df_clean[(df_clean['credit_score'] <= 300) | (df_clean['credit_score'] >= 850)]
print(f"After credit score filter: {len(df_clean)}")


# ─────────────────────────────────────────────
# 4. MISSING VALUE TREATMENT
# ─────────────────────────────────────────────
# Impute with median (robust to any remaining skew)
for col in ['credit_score', 'annual_income', 'loan_term_months']:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

print("\nMissing after imputation:", df_clean.isnull().sum().sum())


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING & ENCODING
# ─────────────────────────────────────────────
le = LabelEncoder()
df_clean['loan_type_enc'] = le.fit_transform(df_clean['loan_type'])

features = [
    'applicant_age', 'years_employed', 'loan_type_enc',
    'credit_score', 'annual_income', 'loan_amount', 'loan_term_months'
]
target = 'interest_rate'

X = df_clean[features].copy()
y = df_clean[target].copy()

# Derived feature
X['income_to_loan_ratio'] = X['annual_income'] / (X['loan_amount'] + 1)


# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────
# 7. MODELLING
# ─────────────────────────────────────────────
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"\n{name}  →  MAE: {mae:.3f}  |  RMSE: {rmse:.3f}  |  R²: {r2:.4f}")


# ─────────────────────────────────────────────
# 8. CROSS-VALIDATION
# ─────────────────────────────────────────────
# BUG 3: cross_val_score is called on X_train_sc (scaled array) but y_train
# still has its original index from the DataFrame. When sklearn internally
# shuffles folds, the numpy array and the pandas Series go out of alignment.
# y_train should be reset_index(drop=True) or converted to a numpy array.
gb_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
cv_scores = cross_val_score(gb_model, X_train_sc, y_train, cv=5, scoring='r2')
print(f"\nCV R² scores: {cv_scores.round(4)}")
print(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
gb_fitted = models["Gradient Boosting"]
feature_names = features + ['income_to_loan_ratio']
importances = pd.Series(gb_fitted.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue')
plt.title("Feature Importances — Gradient Boosting")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nAnalysis complete.")
