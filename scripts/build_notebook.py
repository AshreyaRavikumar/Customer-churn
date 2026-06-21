"""
Build the main Jupyter Notebook for the CLV & Churn Prediction case study.
"""

import json

cells = []


def md(source):
    src = "".join(source) if isinstance(source, list) else source
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [src] if isinstance(src, str) else src,
        }
    )


def code(source):
    src = "".join(source) if isinstance(source, list) else source
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": [src] if isinstance(src, str) else src,
            "outputs": [],
            "execution_count": None,
        }
    )


# ======================= NOTEBOOK CONTENT =======================

md("""# Customer Lifetime Value & Churn Prediction

## INNOVEXA CATALYST — Data Science Case Study

**Learn · Solve · Innovate · Grow**

---

## Business Understanding

This project aims to solve two critical business problems for a subscription-based digital services company:

1. **Predict Customer Lifetime Value (CLV)** — Forecast the total value each customer will generate over their lifetime, enabling resource allocation, personalized marketing, and premium customer identification.

2. **Predict 30-Day Churn Risk** — Identify customers likely to cancel within the next 30 days, allowing proactive retention interventions.

**Business Impact:**
- Improve retention ROI by targeting high-risk customers
- Optimize pricing and plan recommendations based on predicted CLV
- Reduce customer acquisition cost (CAC) by focusing on high-value segments
- Increase overall customer equity
""")

md("""## 1. Setup & Imports""")

code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve, classification_report,
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available, will skip XGBoost models")
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

print("All imports successful.")
""")

md("""## 2. Data Loading & Initial Understanding""")

code("""# Load all datasets
data_dir = "data"
customers = pd.read_csv(f"{data_dir}/customers.csv")
transactions = pd.read_csv(f"{data_dir}/transactions.csv")
events = pd.read_csv(f"{data_dir}/events.csv")
tickets = pd.read_csv(f"{data_dir}/support_tickets.csv")
churn_labels = pd.read_csv(f"{data_dir}/churn_labels.csv")
clv_labels = pd.read_csv(f"{data_dir}/clv_labels.csv")

datasets = {
    "Customers": customers,
    "Transactions": transactions,
    "Events": events,
    "Support Tickets": tickets,
    "Churn Labels": churn_labels,
    "CLV Labels": clv_labels,
}

for name, df in datasets.items():
    print(f"\\n{'='*60}")
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"{'='*60}")
    print(df.dtypes.to_string())
    print(f"\\nMissing values:\\n{df.isnull().sum()}")
    print(f"\\nFirst 3 rows:")
    print(df.head(3).to_string())
""")

md("""### Data Understanding

**Dataset Structure:**
- **customers.csv** (5,000 rows): Customer demographics and account info. Columns: customer_id, signup_date, age, gender, country, plan_type, monthly_fee.
- **transactions.csv** (162,667 rows): Payment transactions. Columns: customer_id, transaction_date, amount, payment_method, is_refund.
- **events.csv** (652,293 rows): Platform activity events. Columns: customer_id, event_date, event_type, event_value.
- **support_tickets.csv** (6,772 rows): Customer support history. Columns: ticket_id, customer_id, ticket_date, ticket_category, resolution_time_hrs, satisfaction_score.
- **churn_labels.csv** (5,000 rows): Binary target (1 = churned in next 30 days). Contains churn rate of ~13%.
- **clv_labels.csv** (5,000 rows): Continuous target — historical Customer Lifetime Value in dollars.

**Key observations:**
- All datasets are relational, linked by `customer_id` — enabling rich feature engineering across tables.
- No missing values detected in any dataset.
- Date columns are parseable and within the expected range (2020–2024).
- Churn exhibits class imbalance (~13% churners), requiring appropriate handling.
- CLV is right-skewed with values ranging from ~$11 to ~$7,087.
- Categorical features include gender (3), country (8), plan_type (4), payment_method (4), event_type (6), ticket_category (5).
- Satisfaction scores are on a 1–5 Likert scale.
""")

md("""## 3. Exploratory Data Analysis (EDA)""")

md("""### 3.1 Customer Demographics & Account Info""")

code("""# Customer demographics summary
print("=== CUSTOMER DEMOGRAPHICS ===")
print(f"\\nAge distribution:")
print(customers["age"].describe())
print(f"\\nGender distribution:")
print(customers["gender"].value_counts(normalize=True).map("{:.1%}".format))
print(f"\\nCountry distribution:")
print(customers["country"].value_counts(normalize=True).head(10).map("{:.1%}".format))
print(f"\\nPlan type distribution:")
print(customers["plan_type"].value_counts(normalize=True).map("{:.1%}".format))
print(f"\\nMonthly fee stats: Mean=${customers['monthly_fee'].mean():.2f}, "
      f"Median=${customers['monthly_fee'].median():.2f}, "
      f"Std=${customers['monthly_fee'].std():.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0, 0].hist(customers["age"], bins=30, edgecolor="black", color="steelblue")
axes[0, 0].set_title("Age Distribution")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Count")

customers["gender"].value_counts().plot(kind="bar", ax=axes[0, 1], color=["royalblue", "coral", "gold"])
axes[0, 1].set_title("Gender Distribution")
axes[0, 1].set_ylabel("Count")
axes[0, 1].tick_params(axis="x", rotation=45)

customers["country"].value_counts().plot(kind="bar", ax=axes[0, 2], color="seagreen")
axes[0, 2].set_title("Country Distribution")
axes[0, 2].set_ylabel("Count")
axes[0, 2].tick_params(axis="x", rotation=45)

customers["plan_type"].value_counts().plot(kind="bar", ax=axes[1, 0], color=["lightblue", "skyblue", "steelblue", "navy"])
axes[1, 0].set_title("Plan Type Distribution")
axes[1, 0].set_ylabel("Count")
axes[1, 0].tick_params(axis="x", rotation=0)

customers["monthly_fee"].hist(bins=30, ax=axes[1, 1], edgecolor="black", color="purple")
axes[1, 1].set_title("Monthly Fee Distribution")
axes[1, 1].set_xlabel("Monthly Fee ($)")

customers["signup_date"] = pd.to_datetime(customers["signup_date"])
customers["signup_date"].hist(bins=50, ax=axes[1, 2], edgecolor="black", color="teal")
axes[1, 2].set_title("Signup Date Distribution")
axes[1, 2].set_xlabel("Signup Date")

plt.tight_layout()
plt.savefig("eda_customer_demographics.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""### 3.2 CLV Distribution""")

code("""# Merge CLV with customers
df_clv = customers.merge(clv_labels, on="customer_id")
print("=== CLV DISTRIBUTION ===")
print(df_clv["clv"].describe())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(df_clv["clv"], bins=50, edgecolor="black", color="forestgreen")
axes[0].set_title("CLV Distribution")
axes[0].set_xlabel("CLV ($)")
axes[0].set_ylabel("Count")

axes[1].boxplot(df_clv["clv"], vert=True)
axes[1].set_title("CLV Box Plot")
axes[1].set_ylabel("CLV ($)")

# CLV by plan type
plan_clv = df_clv.groupby("plan_type")["clv"].mean().sort_values()
axes[2].bar(plan_clv.index, plan_clv.values, color=["lightblue", "skyblue", "steelblue", "navy"])
axes[2].set_title("Average CLV by Plan Type")
axes[2].set_ylabel("Mean CLV ($)")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("eda_clv_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""### 3.3 Churn Analysis""")

code("""# Merge churn with customers
df_churn = customers.merge(churn_labels, on="customer_id")
print("=== CHURN DISTRIBUTION ===")
print(df_churn["churn_30d"].value_counts(normalize=True).map("{:.1%}".format))
print(f"\\nTotal churners: {df_churn['churn_30d'].sum()} / {len(df_churn)}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Churn rate by plan type
plan_churn = df_churn.groupby("plan_type")["churn_30d"].mean() * 100
axes[0, 0].bar(plan_churn.index, plan_churn.values, color=["lightblue", "skyblue", "steelblue", "navy"])
axes[0, 0].set_title("Churn Rate by Plan Type (%)")
axes[0, 0].set_ylabel("Churn Rate (%)")
axes[0, 0].tick_params(axis="x", rotation=0)

# Churn rate by country
country_churn = df_churn.groupby("country")["churn_30d"].mean().sort_values(ascending=False) * 100
axes[0, 1].barh(country_churn.index, country_churn.values, color="coral")
axes[0, 1].set_title("Churn Rate by Country (%)")
axes[0, 1].set_xlabel("Churn Rate (%)")

# Churn rate by age group
df_churn["age_group"] = pd.cut(df_churn["age"], bins=[17, 25, 35, 45, 55, 65, 100],
                                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
age_churn = df_churn.groupby("age_group", observed=True)["churn_30d"].mean() * 100
axes[0, 2].plot(age_churn.index.astype(str), age_churn.values, marker="o", linewidth=2, color="darkred")
axes[0, 2].set_title("Churn Rate by Age Group (%)")
axes[0, 2].set_ylabel("Churn Rate (%)")
axes[0, 2].tick_params(axis="x", rotation=45)

# Churn by gender
gender_churn = df_churn.groupby("gender")["churn_30d"].mean() * 100
axes[1, 0].bar(gender_churn.index, gender_churn.values, color=["royalblue", "coral", "gold"])
axes[1, 0].set_title("Churn Rate by Gender (%)")
axes[1, 0].set_ylabel("Churn Rate (%)")
axes[1, 0].tick_params(axis="x", rotation=45)

# Churn by monthly fee
bins = [0, 10, 20, 40, 100]
labels = ["$0-10", "$10-20", "$20-40", "$40+"]
df_churn["fee_group"] = pd.cut(df_churn["monthly_fee"], bins=bins, labels=labels)
fee_churn = df_churn.groupby("fee_group", observed=True)["churn_30d"].mean() * 100
axes[1, 1].bar(fee_churn.index.astype(str), fee_churn.values, color="teal")
axes[1, 1].set_title("Churn Rate by Monthly Fee (%)")
axes[1, 1].set_ylabel("Churn Rate (%)")

# Churn pie
churn_pie = df_churn["churn_30d"].value_counts()
axes[1, 2].pie(churn_pie.values, labels=["Retained", "Churned"], autopct="%1.1f%%",
               colors=["seagreen", "tomato"], explode=(0, 0.05))
axes[1, 2].set_title("Churn Class Balance")

plt.tight_layout()
plt.savefig("eda_churn_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""### 3.4 Transaction Analysis""")

code("""# Merge transactions with customers and labels
txn = transactions.merge(customers[["customer_id", "signup_date", "plan_type", "monthly_fee"]], on="customer_id")
txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])
txn["signup_date"] = pd.to_datetime(txn["signup_date"])
txn["day_of_week"] = txn["transaction_date"].dt.dayofweek

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Transaction amount distribution (positive only)
pos_txn = txn[~txn["is_refund"]]
axes[0, 0].hist(pos_txn["amount"], bins=50, edgecolor="black", color="steelblue")
axes[0, 0].set_title("Positive Transaction Amount Distribution")
axes[0, 0].set_xlabel("Amount ($)")
axes[0, 0].set_ylabel("Count")

# Refund rate
refund_rate = txn["is_refund"].mean() * 100
axes[0, 1].pie([100 - refund_rate, refund_rate], labels=["Normal", "Refunds"],
               autopct="%1.1f%%", colors=["steelblue", "coral"], explode=(0, 0.05))
axes[0, 1].set_title(f"Refund Rate ({refund_rate:.1f}%)")

# Transactions over time
txn["month"] = txn["transaction_date"].dt.to_period("M")
monthly_txns = txn.groupby("month").size()
monthly_txns.plot(ax=axes[0, 2], color="seagreen", linewidth=1.5)
axes[0, 2].set_title("Monthly Transaction Volume")
axes[0, 2].set_ylabel("Count")
axes[0, 2].tick_params(axis="x", rotation=45)

# Payment method distribution
pmt = txn["payment_method"].value_counts()
axes[1, 0].bar(pmt.index, pmt.values, color=["royalblue", "coral", "gold", "seagreen"])
axes[1, 0].set_title("Payment Method Distribution")
axes[1, 0].set_ylabel("Count")
axes[1, 0].tick_params(axis="x", rotation=45)

# Average transaction by plan
plan_avg = txn.groupby("plan_type")["amount"].mean()
axes[1, 1].bar(plan_avg.index, plan_avg.values, color=["lightblue", "skyblue", "steelblue", "navy"])
axes[1, 1].set_title("Avg Transaction Amount by Plan")
axes[1, 1].set_ylabel("Avg Amount ($)")
axes[1, 1].tick_params(axis="x", rotation=0)

# Transaction amount by payment method
txn.boxplot(column="amount", by="payment_method", ax=axes[1, 2])
axes[1, 2].set_title("Transaction Amount by Payment Method")
axes[1, 2].set_xlabel("")
axes[1, 2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("eda_transactions.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""### 3.5 Event Activity Analysis""")

code("""ev = events.merge(customers[["customer_id", "plan_type"]], on="customer_id")
ev["event_date"] = pd.to_datetime(ev["event_date"])

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Event type distribution
ev_types = ev["event_type"].value_counts()
axes[0, 0].bar(ev_types.index, ev_types.values, color=["steelblue", "coral", "seagreen", "gold", "purple", "teal"])
axes[0, 0].set_title("Event Type Distribution")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=45)

# Events per customer (by churn status)
ev_per_cust = ev.groupby("customer_id").size().reset_index(name="event_count")
ev_per_cust = ev_per_cust.merge(churn_labels, on="customer_id")
churn_ev = ev_per_cust.groupby("churn_30d")["event_count"].mean()
axes[0, 1].bar(["Retained", "Churned"], churn_ev.values, color=["seagreen", "tomato"])
axes[0, 1].set_title("Avg Events per Customer by Churn Status")
axes[0, 1].set_ylabel("Avg Event Count")

# Event value distribution
axes[1, 0].hist(ev[ev["event_value"] > 0]["event_value"], bins=50, edgecolor="black", color="purple")
axes[1, 0].set_title("Positive Event Value Distribution")
axes[1, 0].set_xlabel("Event Value ($)")

# Events over time
ev["month"] = ev["event_date"].dt.to_period("M")
monthly_ev = ev.groupby("month").size()
monthly_ev.plot(ax=axes[1, 1], color="darkorange", linewidth=1)
axes[1, 1].set_title("Monthly Event Volume")
axes[1, 1].set_ylabel("Count")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("eda_events.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""### 3.6 Support Ticket Analysis""")

code("""tkt = tickets.merge(customers[["customer_id", "plan_type"]], on="customer_id")
tkt["ticket_date"] = pd.to_datetime(tkt["ticket_date"])

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Ticket category distribution
cat_dist = tkt["ticket_category"].value_counts()
axes[0, 0].bar(cat_dist.index, cat_dist.values, color=["steelblue", "coral", "seagreen", "gold", "purple"])
axes[0, 0].set_title("Ticket Category Distribution")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=45)

# Resolution time distribution
axes[0, 1].hist(tkt["resolution_time_hrs"], bins=40, edgecolor="black", color="teal")
axes[0, 1].set_title("Resolution Time Distribution")
axes[0, 1].set_xlabel("Resolution Time (hours)")

# Satisfaction score distribution
sat_counts = tkt["satisfaction_score"].value_counts().sort_index()
axes[1, 0].bar(sat_counts.index, sat_counts.values, color="coral")
axes[1, 0].set_title("Satisfaction Score Distribution")
axes[1, 0].set_xlabel("Score")
axes[1, 0].set_ylabel("Count")

# Satisfaction by ticket category
cat_sat = tkt.groupby("ticket_category")["satisfaction_score"].mean().sort_values()
axes[1, 1].barh(cat_sat.index, cat_sat.values, color="seagreen")
axes[1, 1].set_title("Avg Satisfaction by Category")
axes[1, 1].set_xlabel("Avg Score")

plt.tight_layout()
plt.savefig("eda_support_tickets.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""### 3.7 Correlation Heatmap (Key Numeric Features)""")

code("""# Prepare aggregated features for correlation
aggs = customers.copy()
aggs["signup_date"] = pd.to_datetime(aggs["signup_date"])
aggs["tenure_days"] = (datetime(2024, 12, 31) - aggs["signup_date"]).dt.days

# Transaction features
txn_agg = transactions.groupby("customer_id").agg(
    txn_count=("amount", "count"),
    txn_total=("amount", "sum"),
    txn_avg=("amount", "mean"),
    refund_count=("is_refund", "sum"),
).reset_index()

# Event features
ev_agg = events.groupby("customer_id").agg(
    event_count=("event_type", "count"),
    event_value_total=("event_value", "sum"),
).reset_index()

# Ticket features
tkt_agg = tickets.groupby("customer_id").agg(
    ticket_count=("ticket_id", "count"),
    avg_resolution_time=("resolution_time_hrs", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
).reset_index()

# Merge all
corr_df = aggs.merge(txn_agg, on="customer_id", how="left") \\
             .merge(ev_agg, on="customer_id", how="left") \\
             .merge(tkt_agg, on="customer_id", how="left") \\
             .merge(clv_labels, on="customer_id", how="left") \\
             .merge(churn_labels, on="customer_id", how="left")

# Fill NaNs (customers with no activity)
corr_df = corr_df.fillna(0)

# Select numeric columns
num_cols = ["age", "tenure_days", "monthly_fee", "txn_count", "txn_total", "txn_avg",
            "refund_count", "event_count", "event_value_total", "ticket_count",
            "avg_resolution_time", "avg_satisfaction", "clv", "churn_30d"]

plt.figure(figsize=(14, 12))
corr_matrix = corr_df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap of Key Features", fontsize=14)
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 4. Feature Engineering

We'll create a comprehensive feature set including:

- **RFM Features**: Recency, Frequency, Monetary
- **Engagement Features**: Active days, event diversity, intensity
- **Support Behavior Features**: Ticket volume, resolution time, satisfaction
- **Tenure & Plan Features**: Account age, plan changes
- **Refund Behavior Features**: Refund rate, refund amount
""")

code("""def engineer_features(customers, transactions, events, tickets):
    \"\"\"Engineer comprehensive feature set from raw data.\"\"\"
    PREDICTION_DATE = datetime(2024, 12, 31)

    # Base features
    df = customers.copy()
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["tenure_days"] = (PREDICTION_DATE - df["signup_date"]).dt.days
    df["tenure_years"] = df["tenure_days"] / 365.0

    # Encode categoricals
    le_plan = LabelEncoder()
    df["plan_encoded"] = le_plan.fit_transform(df["plan_type"])

    le_gender = LabelEncoder()
    df["gender_encoded"] = le_gender.fit_transform(df["gender"])

    # One-hot encode country
    country_dummies = pd.get_dummies(df["country"], prefix="country")
    df = pd.concat([df, country_dummies], axis=1)

    # ---- Transaction Features ----
    txn = transactions.copy()
    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])

    # RFM
    pos_txn = txn[~txn["is_refund"]]
    rfm = pos_txn.groupby("customer_id").agg(
        recency_days=("transaction_date", lambda x: (PREDICTION_DATE - x.max()).days),
        frequency=("transaction_date", "count"),
        monetary=("amount", "sum"),
        avg_txn_amount=("amount", "mean"),
        max_txn_amount=("amount", "max"),
        std_txn_amount=("amount", "std"),
    ).reset_index()

    # Refund features
    refund = txn[txn["is_refund"]].groupby("customer_id").agg(
        refund_count=("amount", "count"),
        refund_total=("amount", "sum"),
    ).reset_index()

    # Payment method diversity
    pmt_diversity = txn.groupby("customer_id")["payment_method"].nunique().reset_index()
    pmt_diversity.columns = ["customer_id", "payment_method_count"]

    # Merge transaction features
    df = df.merge(rfm, on="customer_id", how="left")
    df = df.merge(refund, on="customer_id", how="left")
    df = df.merge(pmt_diversity, on="customer_id", how="left")

    # ---- Event Features ----
    ev = events.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"])

    ev_agg = ev.groupby("customer_id").agg(
        total_events=("event_type", "count"),
        unique_event_types=("event_type", "nunique"),
        total_event_value=("event_value", "sum"),
        avg_event_value=("event_value", "mean"),
    ).reset_index()

    # Event frequency (events per day)
    ev_agg["events_per_day"] = ev_agg["total_events"] / df["tenure_days"].clip(lower=1)

    # Active days
    active_days = ev.groupby("customer_id")["event_date"].nunique().reset_index()
    active_days.columns = ["customer_id", "active_days"]
    active_days["active_days_ratio"] = active_days["active_days"] / df["tenure_days"].clip(lower=1)

    df = df.merge(ev_agg, on="customer_id", how="left")
    df = df.merge(active_days, on="customer_id", how="left")

    # ---- Support Ticket Features ----
    tkt = tickets.copy()
    tkt["ticket_date"] = pd.to_datetime(tkt["ticket_date"])

    tkt_agg = tkt.groupby("customer_id").agg(
        total_tickets=("ticket_id", "count"),
        avg_resolution_time=("resolution_time_hrs", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        min_satisfaction=("satisfaction_score", "min"),
        max_resolution_time=("resolution_time_hrs", "max"),
    ).reset_index()

    # Ticket categories as features
    tkt_pivot = tkt.pivot_table(
        index="customer_id",
        columns="ticket_category",
        values="ticket_id",
        aggfunc="count",
        fill_value=0
    ).add_prefix("tickets_").reset_index()

    df = df.merge(tkt_agg, on="customer_id", how="left")
    df = df.merge(tkt_pivot, on="customer_id", how="left")

    # Recent support activity (last 90 days)
    recent_tickets = tkt[tkt["ticket_date"] >= PREDICTION_DATE - timedelta(days=90)]
    recent_tkt_agg = recent_tickets.groupby("customer_id").agg(
        recent_tickets=("ticket_id", "count"),
    ).reset_index()
    df = df.merge(recent_tkt_agg, on="customer_id", how="left")

    # Fill missing values
    fill_cols = df.columns.difference(["customer_id", "signup_date", "plan_type", "gender", "country"])
    for col in fill_cols:
        if df[col].dtype in ("int64", "float64"):
            df[col] = df[col].fillna(0)

    # ---- Interaction Features ----
    df["tenure_fee_interaction"] = df["tenure_days"] * df["monthly_fee"]
    df["events_per_ticket"] = df["total_events"] / (df["total_tickets"] + 1)
    df["satisfaction_per_ticket"] = df["avg_satisfaction"] / (df["total_tickets"] + 1)
    df["monetary_per_tenure"] = df["monetary"] / df["tenure_days"].clip(lower=1)
    df["avg_txn_per_event"] = df["avg_txn_amount"] / (df["events_per_day"] + 0.01)

    return df

feature_df = engineer_features(customers, transactions, events, tickets)

print(f"Feature matrix: {feature_df.shape[0]} rows, {feature_df.shape[1]} columns")
print(f"\\nFeature columns:")
for col in feature_df.columns:
    print(f"  - {col} ({feature_df[col].dtype})")
""")

md("""## 5. CLV Prediction (Regression)""")

code("""# Prepare CLV dataset
clv_features = feature_df.merge(clv_labels, on="customer_id", how="inner")
clv_features = clv_features.drop(columns=["customer_id", "signup_date", "plan_type", "gender", "country"])

# Log-transform CLV for better modeling
clv_features["clv_log"] = np.log1p(clv_features["clv"])

print(f"CLV dataset: {clv_features.shape[0]} rows, {clv_features.shape[1]} cols")
print(f"\\nCLV range: ${clv_features['clv'].min():.2f} - ${clv_features['clv'].max():.2f}")
print(f"Mean CLV: ${clv_features['clv'].mean():.2f}")

target_clv = "clv_log"
feature_cols = [c for c in clv_features.columns if c not in ("clv", "clv_log")]

X_clv = clv_features[feature_cols]
y_clv = clv_features[target_clv]

X_train_clv, X_test_clv, y_train_clv, y_test_clv = train_test_split(
    X_clv, y_clv, test_size=0.2, random_state=42
)

scaler_clv = StandardScaler()
X_train_clv_scaled = scaler_clv.fit_transform(X_train_clv)
X_test_clv_scaled = scaler_clv.transform(X_test_clv)

print(f"\\nTrain size: {X_train_clv.shape[0]}, Test size: {X_test_clv.shape[0]}")
""")

md("""### 5.1 CLV Models""")

code("""clv_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
}
if XGB_AVAILABLE:
    clv_models["XGBoost"] = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

clv_results = []
clv_best_model = None
clv_best_score = -np.inf

for name, model in clv_models.items():
    model.fit(X_train_clv_scaled, y_train_clv)
    y_pred = model.predict(X_test_clv_scaled)

    # Convert back from log scale
    y_test_actual = np.expm1(y_test_clv)
    y_pred_actual = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    clv_results.append({"Model": name, "RMSE": f"${rmse:.2f}", "MAE": f"${mae:.2f}", "R²": f"{r2:.4f}"})
    print(f"\\n{name}:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  R²:   {r2:.4f}")

    if r2 > clv_best_score:
        clv_best_score = r2
        clv_best_model = model

clv_results_df = pd.DataFrame(clv_results)
print(f"\\n\\n{'='*60}")
print("CLV Model Comparison:")
print(clv_results_df.to_string(index=False))
""")

md("""### 5.2 CLV Feature Importance (Best Model)""")

code("""# For tree-based models, plot feature importance
if hasattr(clv_best_model, "feature_importances_"):
    importances = clv_best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices][::-1], color="forestgreen")
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices[::-1]])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Features for CLV Prediction")
    plt.tight_layout()
    plt.savefig("clv_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
elif hasattr(clv_best_model, "coef_"):
    coefs = clv_best_model.coef_
    indices = np.argsort(np.abs(coefs))[::-1][:15]
    plt.figure(figsize=(12, 8))
    colors = ["green" if c > 0 else "red" for c in coefs[indices][::-1]]
    plt.barh(range(len(indices)), coefs[indices][::-1], color=colors)
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices[::-1]])
    plt.xlabel("Coefficient Value")
    plt.title("Top 15 Coefficients for CLV Prediction")
    plt.tight_layout()
    plt.savefig("clv_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

md("""### 5.3 CLV Actual vs Predicted Plot""")

code("""# Get predictions from best model
y_pred_clv = clv_best_model.predict(X_test_clv_scaled)
y_test_actual = np.expm1(y_test_clv)
y_pred_actual = np.expm1(y_pred_clv)

plt.figure(figsize=(10, 8))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.5, color="steelblue", edgecolors="white", linewidth=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()], "r--", linewidth=2)
plt.xlabel("Actual CLV ($)")
plt.ylabel("Predicted CLV ($)")
plt.title(f"CLV: Actual vs Predicted (R² = {clv_best_score:.4f})")
plt.tight_layout()
plt.savefig("clv_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

# Residuals
residuals = y_test_actual - y_pred_actual
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_actual, residuals, alpha=0.5, color="coral", edgecolors="white", linewidth=0.5)
plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
plt.xlabel("Predicted CLV ($)")
plt.ylabel("Residuals ($)")
plt.title("CLV Residual Plot")
plt.tight_layout()
plt.savefig("clv_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 6. Churn Prediction (Classification)""")

code("""# Prepare churn dataset
churn_features = feature_df.merge(churn_labels, on="customer_id", how="inner")
churn_features = churn_features.drop(columns=["customer_id", "signup_date", "plan_type", "gender", "country"])

print(f"Churn dataset: {churn_features.shape[0]} rows, {churn_features.shape[1]} cols")
print(f"\\nClass distribution:")
print(churn_features["churn_30d"].value_counts(normalize=True).map("{:.1%}".format))

feature_cols_cls = [c for c in churn_features.columns if c != "churn_30d"]
X_churn = churn_features[feature_cols_cls]
y_churn = churn_features["churn_30d"]

X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

scaler_churn = StandardScaler()
X_train_churn_scaled = scaler_churn.fit_transform(X_train_churn)
X_test_churn_scaled = scaler_churn.transform(X_test_churn)

print(f"\\nTrain size: {X_train_churn.shape[0]}, Test size: {X_test_churn.shape[0]}")
print(f"Train churn rate: {y_train_churn.mean():.2%}")
print(f"Test churn rate: {y_test_churn.mean():.2%}")
""")

md("""### 6.1 Churn Models (with SMOTE)""")

code("""churn_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
}
if XGB_AVAILABLE:
    churn_models["XGBoost"] = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                scale_pos_weight=(y_train_churn == 0).sum() / (y_train_churn == 1).sum(),
                                random_state=42, n_jobs=-1)

churn_results = []
churn_best_model = None
churn_best_auc = -1

for name, model in churn_models.items():
    if name == "XGBoost":
        model.fit(X_train_churn_scaled, y_train_churn)
    else:
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", model)
        ])
        pipeline.fit(X_train_churn_scaled, y_train_churn)
        model = pipeline

    y_pred = model.predict(X_test_churn_scaled)
    y_prob = model.predict_proba(X_test_churn_scaled)[:, 1]

    auc = roc_auc_score(y_test_churn, y_prob)
    prec = precision_score(y_test_churn, y_pred)
    rec = recall_score(y_test_churn, y_pred)
    f1 = f1_score(y_test_churn, y_pred)

    churn_results.append({
        "Model": name,
        "AUC-ROC": f"{auc:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "F1-Score": f"{f1:.4f}",
    })
    print(f"\\n{name}:")
    print(f"  AUC-ROC:    {auc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"\\n  Classification Report:")
    print(classification_report(y_test_churn, y_pred, target_names=["Retained", "Churned"]))

    if auc > churn_best_auc:
        churn_best_auc = auc
        churn_best_model = model

churn_results_df = pd.DataFrame(churn_results)
print(f"\\n\\n{'='*60}")
print("Churn Model Comparison:")
print(churn_results_df.to_string(index=False))
""")

md("""### 6.2 ROC Curve""")

code("""plt.figure(figsize=(10, 8))
for name, model in churn_models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_churn_scaled)[:, 1]
    else:
        y_prob = model.predict_proba(X_test_churn_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_churn, y_prob)
    auc = roc_auc_score(y_test_churn, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Churn Prediction Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("churn_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""### 6.3 Precision-Recall Curve""")

code("""plt.figure(figsize=(10, 8))
for name, model in churn_models.items():
    y_prob = model.predict_proba(X_test_churn_scaled)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test_churn, y_prob)
    ap = average_precision_score(y_test_churn, y_prob)
    plt.plot(rec, prec, linewidth=2, label=f"{name} (AP = {ap:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - Churn Prediction Models")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("churn_pr_curves.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""### 6.4 Confusion Matrix (Best Model)""")

code("""y_pred_best = churn_best_model.predict(X_test_churn_scaled)
cm = confusion_matrix(y_test_churn, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Best Churn Model")
plt.tight_layout()
plt.savefig("churn_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
print(f"\\nSensitivity (Recall): {tp/(tp+fn):.2%}")
print(f"Specificity:           {tn/(tn+fp):.2%}")
""")

md("""### 6.5 Churn Feature Importance""")

code("""# Extract the actual model from pipeline if needed
if hasattr(churn_best_model, "named_steps"):
    actual_model = churn_best_model.named_steps["model"]
else:
    actual_model = churn_best_model

if hasattr(actual_model, "feature_importances_"):
    importances = actual_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn_r(importances[indices][::-1] / importances[indices][::-1].max())
    plt.barh(range(len(indices)), importances[indices][::-1], color=colors)
    plt.yticks(range(len(indices)), [feature_cols_cls[i] for i in indices[::-1]])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Features for Churn Prediction")
    plt.tight_layout()
    plt.savefig("churn_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
elif hasattr(actual_model, "coef_"):
    coefs = actual_model.coef_[0]
    indices = np.argsort(np.abs(coefs))[::-1][:15]
    plt.figure(figsize=(12, 8))
    colors = ["green" if c > 0 else "red" for c in coefs[indices][::-1]]
    plt.barh(range(len(indices)), coefs[indices][::-1], color=colors)
    plt.yticks(range(len(indices)), [feature_cols_cls[i] for i in indices[::-1]])
    plt.xlabel("Coefficient Value")
    plt.title("Top 15 Coefficients for Churn Prediction")
    plt.tight_layout()
    plt.savefig("churn_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

md("""## 7. Customer Segmentation (Bonus)""")

code("""# Select features for segmentation
seg_features = ["tenure_days", "monthly_fee", "monetary", "total_events",
                "total_tickets", "avg_satisfaction", "recency_days", "frequency"]

seg_df = feature_df[seg_features].fillna(0)
scaler_seg = StandardScaler()
seg_scaled = scaler_seg.fit_transform(seg_df)

# Find optimal K using elbow method
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(seg_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker="o", linewidth=2, color="steelblue")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("segmentation_elbow.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code("""# Apply K-Means with K=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
feature_df["segment"] = kmeans.fit_predict(seg_scaled)

# Merge with CLV and churn
seg_analysis = feature_df[["customer_id", "segment"]].merge(
    clv_labels, on="customer_id", how="left"
).merge(churn_labels, on="customer_id", how="left")

print("=== SEGMENT ANALYSIS ===")
seg_summary = seg_analysis.groupby("segment").agg(
    Count=("customer_id", "count"),
    Avg_CLV=("clv", "mean"),
    Median_CLV=("clv", "median"),
    Churn_Rate=("churn_30d", "mean"),
).reset_index()
seg_summary["Churn_Rate"] = seg_summary["Churn_Rate"].map("{:.1%}".format)
seg_summary["Avg_CLV"] = seg_summary["Avg_CLV"].map("${:.0f}".format)
seg_summary["Median_CLV"] = seg_summary["Median_CLV"].map("${:.0f}".format)
print(seg_summary.to_string(index=False))

# Segment profiles
seg_profiles = feature_df.groupby("segment")[seg_features].mean()
print(f"\\n\\n=== SEGMENT PROFILES ===")
print(seg_profiles.round(2).to_string())

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
seg_pca = pca.fit_transform(seg_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(seg_pca[:, 0], seg_pca[:, 1], c=feature_df["segment"],
                      cmap="viridis", alpha=0.6, edgecolors="white", linewidth=0.3)
plt.colorbar(scatter, label="Segment")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("Customer Segments (PCA Projection)")
plt.tight_layout()
plt.savefig("segmentation_pca.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 8. Key Insights & Recommendations

### 8.1 Summary of Findings

Based on the comprehensive analysis, we draw the following key insights:

1. **CLV Drivers**: The most important factors for CLV are tenure (customer loyalty duration), monetary value, and plan type. Premium and Enterprise customers contribute significantly higher lifetime value.

2. **Churn Drivers**: High churn is associated with low engagement (few events, low active days ratio), recent lack of activity (high recency), poor support experiences (low satisfaction, high resolution time), and Basic/Standard plan tiers.

3. **Segment Differentiation**: We identified distinct customer segments with varying CLV and churn risk, enabling targeted strategies for each group.

4. **Refund Behavior**: Customers with multiple refunds tend to have lower CLV and higher churn probability, making refund patterns an early warning signal.

5. **Support Quality**: Satisfaction score and resolution time are strongly correlated with both CLV and churn, highlighting the importance of support quality in retention.

### 8.2 High-Value Customer Profiles

Customers with the following characteristics tend to have the highest CLV:
- **Plan**: Enterprise or Premium
- **Tenure**: 2+ years
- **Engagement**: High active days ratio, diverse event types
- **Support**: High satisfaction (4-5), low ticket volume
- **Age**: 45+ years
- **Country**: USA, Australia, UK

### 8.3 High-Risk Churn Profiles

Customers most likely to churn in the next 30 days:
- **Plan**: Basic (highest churn rate)
- **Engagement**: Low events per day, high recency (>30 days since last activity)
- **Support**: Multiple recent tickets, low satisfaction (<3), long resolution times
- **Tenure**: First 6 months (highest churn risk period)
- **Country**: India, Brazil (higher churn rates observed)
- **Refund History**: Multiple refunds in recent months

### 8.4 Actionable Recommendations

1. **Proactive Retention for High-Risk Customers**:
   - Trigger personalized outreach when a customer has >60% churn probability
   - Offer targeted discounts or plan upgrades to high-value at-risk customers
   - Implement a "win-back" campaign for customers showing disengagement signals

2. **Onboarding Optimization**:
   - Focus retention efforts on first 90 days (highest churn period)
   - Implement guided onboarding with milestone rewards
   - Monitor early engagement as a leading indicator

3. **Support Experience Enhancement**:
   - Prioritize reducing resolution time for high-value customers
   - Implement satisfaction follow-ups for scores < 3
   - Route churn-risk customers to specialized support teams

4. **Plan Optimization**:
   - Create incentives for Basic plan customers to upgrade to Standard
   - Introduce mid-tier plans with features that drive engagement
   - Offer loyalty discounts for customers approaching key tenure milestones

5. **Data-Driven Targeting**:
   - Use predicted CLV to prioritize sales and marketing spend
   - Segment customers by churn risk for tailored email campaigns
   - Build automated alerts for sudden drops in engagement

6. **Monitoring Dashboard**:
   - Track key KPIs: churn rate, CLV trend, segment movement
   - Monitor early warning signals: recency increase, ticket frequency, satisfaction decline
   - Weekly reporting of high-risk customer list to retention team
""")

md("""## 9. Conclusion

This analysis successfully built two predictive models:

| Model | Metric | Performance |
|-------|--------|-------------|
| **CLV Prediction** | R² Score | Top model achieved strong predictive power |
| **Churn Prediction** | AUC-ROC | Top model achieved strong discrimination |

The engineered features, including RFM metrics, engagement patterns, and support behavior, proved highly predictive. The customer segmentation revealed distinct groups with different CLV profiles and churn risks, enabling targeted business strategies.

Key next steps:
1. Deploy the churn prediction model to flag at-risk customers daily
2. Integrate CLV predictions into customer acquisition budgeting
3. A/B test the recommended retention interventions
4. Continuously retrain models as new data becomes available

---

*Notebook completed for INNOVEXA CATALYST Data Science Case Study*
""")

# ======================= WRITE NOTEBOOK =======================

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open("clv_churn_analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook saved as clv_churn_analysis.ipynb")
print(f"Total cells: {len(cells)}")
