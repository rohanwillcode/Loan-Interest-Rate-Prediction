# Business Interpretation: Loan Interest Rate Prediction

## Executive Summary

The objective of this analysis was to determine whether loan interest rates can be reliably predicted using the applicant and loan information currently available.

After evaluating multiple machine learning approaches, the best-performing model achieved an **R² score of 42.44%**, meaning that more than half of the variation in interest rates remains unexplained. More importantly, advanced machine learning models consistently underperformed compared to a simple Linear Regression model.

This finding leads to a critical business conclusion:

> The primary limitation is not the modeling approach. The primary limitation is the lack of business-relevant data required to explain how interest rates are actually determined.

As a result, the current solution should not be used as an automated loan pricing engine. It may serve as a decision-support or validation tool, but not as a standalone system for setting interest rates.

---

# Model Performance Summary

| Model | R² Score | MAE | RMSE |
|---------|---------:|---------:|---------:|
| **Linear Regression** | **0.4244** | **1.3058** | **2.0749** |
| Random Forest | 0.3749 | 1.4490 | 2.1622 |
| SVR | 0.3709 | 1.3630 | 2.1691 |
| MLP Regressor | 0.3561 | 1.4365 | 2.1945 |
| LightGBM | 0.2210 | 1.6174 | 2.4138 |
| XGBoost | 0.1442 | 1.7139 | 2.5300 |

### Key Observation

The strongest model explains only **42.44%** of the variation in loan interest rates.

The fact that advanced models such as XGBoost and LightGBM performed significantly worse than Linear Regression indicates that there are no strong hidden nonlinear relationships within the current dataset. If such relationships existed, these models would have captured them.

This suggests that the missing predictive power is not hidden within the existing data. It is missing entirely.

---

# What Currently Drives Interest Rates?

The analysis identified **Credit Score** as the only variable with a meaningful relationship to interest rates.

| Variable | Correlation with Interest Rate | Business Interpretation |
|-----------|-----------:|-----------|
| Credit Score | -0.49 | Strongest predictor. Lower scores consistently receive higher rates. |
| Annual Income | -0.13 | Weak relationship. Limited pricing value on its own. |
| Applicant Age | -0.11 | Minimal impact. |
| Years Employed | -0.11 | Minimal impact. |
| Loan Amount | -0.03 | Negligible impact. |
| Loan Term | +0.02 | Negligible impact. |

### Business Interpretation

If the organization had to price loans using only one variable from the current dataset, **Credit Score would be the only reliable choice**.

The remaining variables contribute little explanatory power and do not appear to meaningfully influence interest rate decisions by themselves.

---

# Data Redundancy Identified

A strong correlation was observed between:

- Applicant Age
- Years Employed
- Annual Income

These variables showed correlations ranging between **0.92 and 0.99**, indicating that they largely describe the same borrower profile.

From a business perspective, this means:

- Older applicants generally have longer employment histories.
- Longer employment histories generally correspond to higher incomes.
- Collecting all three variables may not provide three independent signals.

This redundancy is unlikely to improve pricing decisions significantly.

---

# Loan Type Segmentation Insights

Average interest rates varied across loan categories.

### Higher Average Interest Rates

- Business Loans (~13.3%)
- Education Loans (~13.3%)

### Moderate Average Interest Rates

- Auto Loans (~12.8%)

### Lower Average Interest Rates

- Personal Loans (~11.8%)
- Home Loans (~11.6%)

### Business Interpretation

Loan purpose appears to influence pricing more than most applicant attributes included in the dataset.

However, Business Loans represent a relatively small portion of the data, so these findings should be treated as directional rather than conclusive.

---

# Why The Current Solution Is Not Reliable Enough

A lending model should ideally explain a substantial portion of pricing decisions before being trusted in production.

The current model leaves approximately:

```text
100% - 42.44% = 57.56%
```

of interest-rate variation unexplained.

This means that a majority of the factors influencing pricing decisions are absent from the dataset.

Consequently:

- The model cannot consistently justify why one borrower receives a significantly different rate than another.
- The model may produce reasonable estimates in some cases but unreliable estimates in others.
- Automated pricing based on this model would introduce unnecessary business risk.

---

# What Data Is Missing?

The current dataset focuses primarily on demographic and basic loan information.

Modern lending decisions rely heavily on additional risk and affordability indicators that are not present here.

### Borrower Debt Profile

- Total Existing Debt
- Debt-to-Income Ratio (DTI)
- Monthly Debt Obligations
- Number of Active Loans

### Credit Behaviour

- Delinquency History
- Missed Payment History
- Previous Defaults
- Bankruptcy Records
- Credit Utilization Ratio

### Loan Security Information

- Collateral Value
- Loan-to-Value Ratio (LTV)
- Down Payment Amount

### Employment & Financial Stability

- Employment Type
- Industry Sector
- Job Stability
- Income Verification Status

### Market Conditions

- Central Bank Interest Rate
- Inflation Rate
- Lending Environment at Loan Origination
- Economic Conditions at Approval Time

These variables are commonly used in real-world underwriting and would likely contribute significantly more predictive power than additional model tuning.

---

# How The Model Should Be Used Today

### Appropriate Uses

- Decision-support tool
- Secondary validation mechanism
- Rate consistency checks
- Portfolio-level trend analysis
- Risk review assistance

### Inappropriate Uses

- Automated loan pricing
- Independent underwriting decisions
- Final interest-rate determination
- Approval or rejection decisions

---

# Monitoring Recommendations

If deployed as a support tool, the model should be monitored regularly.

### Performance Monitoring

Track:

- R² Score
- MAE
- RMSE

A significant decline may indicate changes in borrower behaviour or lending practices.

### Data Quality Monitoring

Track:

- Missing values
- Distribution shifts in credit scores
- Distribution shifts in income levels
- Changes in loan-type composition

### Business Monitoring

Track:

- Average predicted rate vs actual rate
- Prediction error by loan type
- Prediction error by credit-score band
- Prediction error by income segment

---

# Final Recommendation

The analysis demonstrates that the current dataset does not contain enough information to support a reliable interest-rate prediction system.

The strongest model explains only **42.44%** of pricing variation, while more advanced algorithms consistently failed to improve performance. This strongly suggests that the primary challenge is **insufficient business information rather than insufficient modeling sophistication**.

Before investing additional effort into algorithm selection or hyperparameter tuning, priority should be given to collecting richer underwriting, credit-risk, and borrower-behaviour data. These variables are far more likely to improve predictive accuracy than further experimentation with machine learning techniques.

**Recommendation:** Use the current model only as a supplementary decision-support tool and prioritize data enrichment before considering any production-grade automation of loan pricing decisions.
