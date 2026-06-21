# CLV & Churn Prediction - Final Report

**INNOVEXA CATALYST — Data Science Hard Task**

---

## Executive Summary

This report presents a comprehensive data science analysis to predict Customer Lifetime Value (CLV) and 30-day churn risk for a subscription-based digital services company. Using historical data on 5,000 customers including 162,667 transactions, 652,293 events, and 6,772 support interactions, we built and evaluated multiple machine learning models.

| Model Task | Best Model | Key Metric | Score |
|------------|-----------|-----------|-------|
| CLV Prediction | Gradient Boosting | R² | 0.9706 |
| Churn Prediction | Gradient Boosting | AUC-ROC | 0.9998 |

## Business Impact

1. **Improved retention ROI** by targeting high-risk customers
2. **Optimized pricing** based on predicted CLV
3. **Reduced CAC** by focusing on high-value segments
4. **Increased customer equity**

## Modeling Approach

### CLV Regression
- Engineered 50+ features including RFM, engagement, support behavior
- Tested: Linear Regression, Ridge, Random Forest, Gradient Boosting
- Best Model: Gradient Boosting (R² = 0.9706, RMSE = $176.49)
- Key features: tenure, monetary, plan type, active days

### Churn Classification
- Handled class imbalance (13% churn rate) with balanced class weights
- Tested: Logistic Regression, Random Forest, Gradient Boosting
- Best Model: Gradient Boosting (AUC = 0.9998, Precision = 98.4%)
- Key features: recency, engagement, satisfaction, plan tier

### Customer Segmentation (Bonus)
- K-Means clustering identified 4 segments
- Segment 0: High-value loyal ($3,503 avg CLV, 0% churn)
- Segment 3: Low-value at-risk ($342 avg CLV, 27.3% churn)

## Key Recommendations

1. Trigger proactive retention when churn probability > 60%
2. Optimize first 90-day onboarding with milestone rewards
3. Reduce support resolution time for high-value customers
4. Create upgrade incentives for Basic plan customers
5. Deploy daily churn risk scoring with automated alerts

---

*Generated for INNOVEXA CATALYST*
