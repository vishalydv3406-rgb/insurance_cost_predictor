# Model Performance Report

## Main Insurance Dataset Evaluation (Claims)

### Regression Metrics

| Model | R2 Score | MAE |
| --- | --- | --- |
| Linear Regression | 0.9287 | 508.48 |
| Decision Tree | 0.8641 | 684.06 |
| Random Forest | 0.9288 | 500.80 |
| Gradient Boosting | 0.9362 | 473.12 |
| K-Nearest Neighbors | 0.9012 | 558.95 |
| Support Vector Regressor | 0.8421 | 632.41 |
| XGBoost | 0.9360 | 472.40 |

### Sample Predictions (Gradient Boosting)

| Actual | Predicted |
| --- | --- |
| 3709.61 | 3690.2 |
| 5806.6 | 5186.87 |
| 8765.63 | 8667.41 |
| 6931.04 | 6059.94 |
| 2491.89 | 2695.81 |

### Classification Accuracy

- **Risk Classifier (Random Forest):** 0.9971 accuracy

## Domain-Specific Models Evaluation

### Domain R2 Scores Summary

| Domain | Gradient Boosting | Linear Regression | Random Forest |
| --- | --- | --- | --- |
| Business | 0.5967 | 0.5374 | 0.5701 |
| Claim (Domain) | 0.9351 | 0.8847 | 0.9270 |
| Health | 0.6576 | 0.5899 | 0.6201 |
| Life | 0.6828 | 0.6430 | 0.6486 |
| Motor | 0.6301 | 0.5725 | 0.5963 |
| Property | 0.6198 | 0.6253 | 0.5823 |
| Specialty | 0.5927 | 0.5283 | 0.5412 |
| Travel | 0.7209 | 0.6505 | 0.6959 |

