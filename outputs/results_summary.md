# CLV & Churn Prediction - Results Summary

## CLV Regression Results

| Model             | RMSE    | MAE     |     R² |
|:------------------|:--------|:--------|-------:|
| Linear Regression | $339.17 | $151.63 | 0.8915 |
| Ridge Regression  | $340.30 | $151.61 | 0.8908 |
| Random Forest     | $187.55 | $89.13  | 0.9668 |
| Gradient Boosting | $176.49 | $83.11  | 0.9706 |

## Churn Classification Results

| Model               |   AUC-ROC |   Precision |   Recall |   F1-Score |
|:--------------------|----------:|------------:|---------:|-----------:|
| Logistic Regression |    0.9996 |      0.9412 |   0.9846 |     0.9624 |
| Random Forest       |    0.9998 |      0.9348 |   0.9923 |     0.9627 |
| Gradient Boosting   |    0.9998 |      0.9844 |   0.9692 |     0.9767 |

## Segment Summary

|   segment |   Count |   Avg_CLV |   Median_CLV |   Churn_Rate |
|----------:|--------:|----------:|-------------:|-------------:|
|         0 |     349 |  3502.85  |     3454.61  |     0        |
|         1 |    1800 |   866.067 |      700.385 |     0        |
|         2 |    1267 |   396.911 |      294.73  |     0.173639 |
|         3 |    1584 |   341.659 |      225.53  |     0.272727 |

## Key Findings

1. **CLV Drivers**: Tenure, monetary value, plan type (Premium/Enterprise)
2. **Churn Drivers**: Low engagement, high recency, low satisfaction, Basic plan
3. **High-Value Segments**: Enterprise/Premium customers with high engagement
4. **High-Risk Segments**: Basic plan, low tenure, recent inactivity
5. **Support Impact**: Low satisfaction and long resolution times correlate with churn
