# Customer Lifetime Value & Churn Prediction

**INNOVEXA CATALYST — Data Science Hard Task**

A complete end-to-end data science project where I predicted **Customer Lifetime Value (CLV)** and **30-day churn risk** for a subscription-based digital services company.

---

## What I Did

I was given a business problem: a subscription digital services company wanted to understand customer behavior better — specifically, they needed to **predict future CLV** and **identify customers likely to churn in the next 30 days**.

Since no real data was available, I first **generated synthetic datasets** (6 tables, ~830K rows total) with realistic patterns for 5,000 customers — their demographics, transactions, platform events, support tickets, and ground-truth CLV/churn labels.

Then I built the full pipeline:

1. **Exploratory Data Analysis** — uncovered key drivers of CLV and churn through visualizations and correlation analysis
2. **Feature Engineering** — created 50+ features spanning RFM metrics, engagement patterns, support behavior, and interaction terms
3. **CLV Regression** — trained and compared 4–5 models; **Gradient Boosting** gave the best result with **R² = 0.9706**
4. **Churn Classification** — handled 13% class imbalance using balanced weights and SMOTE; **Gradient Boosting** achieved **AUC-ROC = 0.9998** with 98.4% precision
5. **Customer Segmentation** (bonus) — used K-Means to identify 4 distinct segments with very different CLV and churn profiles
6. **Deliverables** — compiled everything into a Jupyter Notebook, an HTML report with 13 embedded figures, a markdown summary, and a Streamlit dashboard

## Key Findings

| Insight | Detail |
|---------|--------|
| **CLV is driven by** | Tenure, total monetary spend, plan tier (Premium/Enterprise), and active engagement days |
| **Churn is driven by** | High recency (inactivity > 30 days), low satisfaction scores, Basic plan, and multiple recent support tickets |
| **Highest-value segment** | 349 customers averaging **$3,503 CLV**, 0% churn — long-tenured Enterprise/Premium users |
| **Highest-risk segment** | 1,584 customers averaging **$342 CLV**, 27.3% churn — Basic plan, low engagement |

## Model Performance

| Task | Best Model | Metric | Score |
|------|-----------|--------|-------|
| CLV Regression | Gradient Boosting | **R²** | **0.9706** |
| CLV Regression | Gradient Boosting | RMSE | $176.49 |
| Churn Classification | Gradient Boosting | **AUC-ROC** | **0.9998** |
| Churn Classification | Gradient Boosting | Precision | 98.4% |
| Churn Classification | Gradient Boosting | Recall | 96.9% |

## Recommendations I Made

- Trigger proactive retention outreach when churn probability exceeds 60%
- Focus onboarding improvements on the first 90 days (highest churn risk period)
- Reduce support resolution time for high-value customers
- Create upgrade incentives for Basic plan subscribers
- Deploy the churn model for daily risk scoring with automated alerts

---

## Repository Structure

```
.
├── data/                           # Raw synthetic datasets (6 CSVs)
├── scripts/                        # All source code
│   ├── generate_data.py            # Data generation
│   ├── run_analysis.py             # Full analysis pipeline
│   ├── build_notebook.py           # Notebook builder
│   ├── generate_report.py          # Report generator
│   └── dashboard.py                # Streamlit dashboard (bonus)
├── outputs/                        # 13 visualizations + results summary
├── deliverables/                   # Final submission
│   ├── clv_churn_analysis.ipynb    # Jupyter Notebook (48 cells)
│   ├── report.html                 # Interactive HTML report
│   └── report.md                   # Markdown report
└── README.md
```

## How to Reproduce

```bash
# 1. Generate data (optional — data/ is already populated)
python scripts/generate_data.py

# 2. Run full analysis
python scripts/run_analysis.py

# 3. Launch dashboard (bonus)
streamlit run scripts/dashboard.py

# 4. Open notebook
jupyter notebook deliverables/clv_churn_analysis.ipynb
```

---

*Built for INNOVEXA CATALYST · Learn · Solve · Innovate · Grow*
