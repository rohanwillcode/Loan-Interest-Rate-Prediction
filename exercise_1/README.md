# Loan Interest Rate Prediction

## Project Overview

This project develops a machine learning pipeline to predict loan interest rates using applicant and loan-related information. The objective is to identify the factors influencing loan pricing and evaluate the effectiveness of various regression algorithms in estimating interest rates.

The project covers the complete machine learning lifecycle, including:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Transformation
- Model Development
- Model Evaluation
- Technical Summary

---

## Problem Statement

Predict the interest rate offered to a borrower using applicant and loan-related attributes.

### Target Variable

```text
interest_rate
```

### Available Features

```text
applicant_age
years_employed
loan_type
credit_score
annual_income
loan_amount
loan_term_months
```

---

## Exploratory Data Analysis

Several exploratory analyses were performed to understand the dataset and identify potential issues before model development.

### Analysis Performed

- Missing Value Analysis
- Distribution Analysis
- Correlation Analysis
- Multicollinearity Detection
- Target Relationship Analysis
- Loan Type Segmentation Analysis

### Key Findings

- Credit Score showed the strongest relationship with interest rate.
- Annual Income, Applicant Age, and Years Employed showed weak relationships with the target.
- Loan Amount and Loan Term exhibited minimal influence on interest rates.
- Strong multicollinearity was identified among:
  - Applicant Age
  - Years Employed
  - Annual Income

These variables demonstrated correlation values ranging from approximately **0.92 to 0.99**, indicating substantial redundancy.

---

## Feature Engineering

No additional features were created during this project. Instead, the focus was placed on understanding the existing variables and preparing them appropriately for modeling.

The following steps were performed:

### Feature and Target Separation

The target variable was separated from the independent variables:

```python
X = df.drop('interest_rate', axis=1)
y = df['interest_rate']
```

### Correlation Analysis

Correlation analysis was performed to understand the relationship between features and the target variable, as well as to identify potential multicollinearity among independent variables.

Key observations included:

- Credit Score showed the strongest relationship with interest rate.
- Annual Income, Applicant Age, and Years Employed exhibited relatively weak relationships with the target.
- Strong multicollinearity was observed among Applicant Age, Years Employed, and Annual Income, with correlation values ranging between **0.92 and 0.99**.

Although multicollinearity was identified, no features were removed at this stage. All available variables were retained to evaluate their impact on model performance before making any feature elimination decisions.

### Train-Test Split

The dataset was divided into training and testing subsets to evaluate model performance on unseen data.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This ensured that model evaluation was performed on data that was not used during training.

### Preprocessing Pipeline

A preprocessing pipeline was implemented using a `ColumnTransformer` to ensure consistent treatment of both numerical and categorical features.

The pipeline included:

- Median imputation for numerical features
- Most-frequent imputation for categorical features
- Standard scaling for numerical variables
- One-Hot Encoding for categorical variables

This approach ensured that all preprocessing steps were applied consistently to both the training and testing datasets while preventing data leakage.
---

## Data Preprocessing

### Missing Value Treatment

Numerical features were imputed using:

```python
SimpleImputer(strategy="median")
```

Categorical features were imputed using:

```python
SimpleImputer(strategy="most_frequent")
```

### Feature Scaling

Numerical variables were standardized using:

```python
StandardScaler()
```

### Categorical Encoding

Categorical features were transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

### Column Transformer

A `ColumnTransformer` pipeline was implemented to ensure consistent preprocessing for both training and testing datasets.

```python
ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)
```

---

## Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## Models Evaluated

The following regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Support Vector Regressor (SVR)
4. Multi-Layer Perceptron (MLP Regressor)
5. LightGBM Regressor
6. XGBoost Regressor

---

## Evaluation Metrics

Model performance was assessed using:

### R² Score

Measures the proportion of variance explained by the model.

### Mean Absolute Error (MAE)

Measures the average absolute prediction error.

### Root Mean Squared Error (RMSE)

Measures prediction error while penalizing larger deviations.

---

## Model Performance

| Model | R² Score | MAE | RMSE |
|---------|---------:|---------:|---------:|
| **Linear Regression** | **0.4244** | **1.3058** | **2.0749** |
| Random Forest | 0.3749 | 1.4490 | 2.1622 |
| SVR | 0.3709 | 1.3630 | 2.1691 |
| MLP Regressor | 0.3561 | 1.4365 | 2.1945 |
| LightGBM | 0.2210 | 1.6174 | 2.4138 |
| XGBoost | 0.1442 | 1.7139 | 2.5300 |

---

## Best Model

### Linear Regression

Linear Regression achieved the highest predictive performance:

```text
R² Score : 0.4244
MAE      : 1.3058
RMSE     : 2.0749
```

Interestingly, more advanced models such as XGBoost and LightGBM failed to outperform the baseline Linear Regression model.

This indicates that the available features exhibit predominantly linear relationships with the target variable and that the current dataset lacks strong nonlinear predictive signals.

---

## Technical Findings

### Strongest Predictor

| Feature | Correlation with Interest Rate |
|----------|----------:|
| Credit Score | -0.49 |

Credit Score emerged as the only feature with a meaningful relationship to interest rate.

### Weak Predictors

- Annual Income
- Applicant Age
- Years Employed
- Loan Amount
- Loan Term

These variables demonstrated limited predictive power.

### Multicollinearity

Significant multicollinearity was observed among:

- Applicant Age
- Years Employed
- Annual Income

This suggests that these features provide overlapping information and may not contribute independent predictive value.

---

## Limitations

The current dataset lacks several variables commonly used in real-world loan pricing and underwriting systems, including:

- Debt-to-Income Ratio (DTI)
- Existing Debt Obligations
- Credit Utilization
- Previous Loan Defaults
- Delinquency History
- Bankruptcy Records
- Loan-to-Value Ratio (LTV)
- Collateral Information
- Down Payment Information
- Employment Type
- Income Verification Status
- Market Interest Rates

The absence of these variables significantly limits predictive performance.

---

## Future Improvements

Potential improvements include:

- Incorporating richer borrower risk indicators.
- Adding credit behavior and repayment history features.
- Including macroeconomic variables.
- Introducing collateral and loan security information.
- Collecting additional historical lending data.
- Expanding feature engineering efforts.

---

## Disclaimer

This project demonstrates a complete machine learning workflow for loan interest rate prediction. However, the current model should **not be used as an automated loan pricing engine**.

The best-performing model explains approximately **42.44%** of interest-rate variation, leaving a substantial portion of pricing behavior unexplained. This limitation stems primarily from the lack of critical underwriting and risk-assessment features rather than shortcomings in the modeling process.

At its current stage, the model should be considered a:

- Decision-support tool
- Exploratory analytics tool
- Educational machine learning project

rather than a production-ready loan pricing system.

With richer borrower, credit-risk, and market-related data, the predictive performance could improve substantially and potentially support more reliable decision-making workflows.