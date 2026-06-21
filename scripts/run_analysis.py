#!/usr/bin/env python3
"""
Comprehensive CLV & Churn Prediction Analysis
INNOVEXA CATALYST - Data Science Case Study
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    print("imbalanced-learn not available")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

BASE_DIR = "/home/fffstanza/Projects/Data Science Hard task"
DATA_DIR = f"{BASE_DIR}/data"
OUT_DIR = f"{BASE_DIR}/outputs"
PREDICTION_DATE = datetime(2024, 12, 31)
print(f"Analysis started at: {datetime.now()}")
print(f"Prediction date: {PREDICTION_DATE.date()}")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
transactions = pd.read_csv(f"{DATA_DIR}/transactions.csv")
events = pd.read_csv(f"{DATA_DIR}/events.csv")
tickets = pd.read_csv(f"{DATA_DIR}/support_tickets.csv")
churn_labels = pd.read_csv(f"{DATA_DIR}/churn_labels.csv")
clv_labels = pd.read_csv(f"{DATA_DIR}/clv_labels.csv")

for name, df in [
    ("Customers", customers),
    ("Transactions", transactions),
    ("Events", events),
    ("Support Tickets", tickets),
    ("Churn Labels", churn_labels),
    ("CLV Labels", clv_labels),
]:
    print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols")

# ============================================================
# 2. EDA
# ============================================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Customer demographics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0, 0].hist(customers["age"], bins=30, edgecolor="black", color="steelblue")
axes[0, 0].set_title("Age Distribution")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Count")
customers["gender"].value_counts().plot(
    kind="bar", ax=axes[0, 1], color=["royalblue", "coral", "gold"]
)
axes[0, 1].set_title("Gender Distribution")
axes[0, 1].set_ylabel("Count")
axes[0, 1].tick_params(axis="x", rotation=45)
customers["country"].value_counts().plot(kind="bar", ax=axes[0, 2], color="seagreen")
axes[0, 2].set_title("Country Distribution")
axes[0, 2].set_ylabel("Count")
axes[0, 2].tick_params(axis="x", rotation=45)
customers["plan_type"].value_counts().plot(
    kind="bar", ax=axes[1, 0], color=["lightblue", "skyblue", "steelblue", "navy"]
)
axes[1, 0].set_title("Plan Type Distribution")
axes[1, 0].set_ylabel("Count")
customers["monthly_fee"].hist(bins=30, ax=axes[1, 1], edgecolor="black", color="purple")
axes[1, 1].set_title("Monthly Fee Distribution")
axes[1, 1].set_xlabel("Monthly Fee ($)")
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
customers["signup_date"].hist(bins=50, ax=axes[1, 2], edgecolor="black", color="teal")
axes[1, 2].set_title("Signup Date Distribution")
axes[1, 2].set_xlabel("Signup Date")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_customer_demographics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_customer_demographics.png")

# CLV distribution
df_clv = customers.merge(clv_labels, on="customer_id")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(df_clv["clv"], bins=50, edgecolor="black", color="forestgreen")
axes[0].set_title("CLV Distribution")
axes[0].set_xlabel("CLV ($)")
axes[0].set_ylabel("Count")
axes[1].boxplot(df_clv["clv"], vert=True)
axes[1].set_title("CLV Box Plot")
axes[1].set_ylabel("CLV ($)")
plan_clv = df_clv.groupby("plan_type")["clv"].mean().sort_values()
axes[2].bar(
    plan_clv.index, plan_clv.values, color=["lightblue", "skyblue", "steelblue", "navy"]
)
axes[2].set_title("Average CLV by Plan Type")
axes[2].set_ylabel("Mean CLV ($)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_clv_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_clv_distribution.png")

# Churn analysis
df_churn = customers.merge(churn_labels, on="customer_id")
df_churn["age_group"] = pd.cut(
    df_churn["age"],
    bins=[17, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
)
df_churn["fee_group"] = pd.cut(
    df_churn["monthly_fee"],
    bins=[0, 10, 20, 40, 100],
    labels=["$0-10", "$10-20", "$20-40", "$40+"],
)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
plan_churn = df_churn.groupby("plan_type")["churn_30d"].mean() * 100
axes[0, 0].bar(
    plan_churn.index,
    plan_churn.values,
    color=["lightblue", "skyblue", "steelblue", "navy"],
)
axes[0, 0].set_title("Churn Rate by Plan Type (%)")
axes[0, 0].set_ylabel("Churn Rate (%)")
country_churn = (
    df_churn.groupby("country")["churn_30d"].mean().sort_values(ascending=False) * 100
)
axes[0, 1].barh(country_churn.index, country_churn.values, color="coral")
axes[0, 1].set_title("Churn Rate by Country (%)")
axes[0, 1].set_xlabel("Churn Rate (%)")
age_churn = df_churn.groupby("age_group", observed=True)["churn_30d"].mean() * 100
axes[0, 2].plot(
    age_churn.index.astype(str),
    age_churn.values,
    marker="o",
    linewidth=2,
    color="darkred",
)
axes[0, 2].set_title("Churn Rate by Age Group (%)")
axes[0, 2].set_ylabel("Churn Rate (%)")
axes[0, 2].tick_params(axis="x", rotation=45)
gender_churn = df_churn.groupby("gender")["churn_30d"].mean() * 100
axes[1, 0].bar(
    gender_churn.index, gender_churn.values, color=["royalblue", "coral", "gold"]
)
axes[1, 0].set_title("Churn Rate by Gender (%)")
axes[1, 0].set_ylabel("Churn Rate (%)")
axes[1, 0].tick_params(axis="x", rotation=45)
fee_churn = df_churn.groupby("fee_group", observed=True)["churn_30d"].mean() * 100
axes[1, 1].bar(fee_churn.index.astype(str), fee_churn.values, color="teal")
axes[1, 1].set_title("Churn Rate by Monthly Fee (%)")
axes[1, 1].set_ylabel("Churn Rate (%)")
churn_counts = df_churn["churn_30d"].value_counts()
if len(churn_counts) == 2:
    axes[1, 2].pie(
        churn_counts.values,
        labels=["Retained", "Churned"],
        autopct="%1.1f%%",
        colors=["seagreen", "tomato"],
        explode=(0, 0.05),
    )
else:
    axes[1, 2].pie(
        churn_counts.values,
        labels=churn_counts.index.astype(str),
        autopct="%1.1f%%",
        colors=["seagreen", "tomato"],
    )
    churn_counts = df_churn["churn_30d"].value_counts()
axes[1, 2].set_title(f"Churn Class Balance (Rate: {df_churn['churn_30d'].mean():.1%})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_churn_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_churn_analysis.png")
print(f"  Churn rate: {df_churn['churn_30d'].mean():.2%}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df = customers.copy()
df["signup_date"] = pd.to_datetime(df["signup_date"])
df["tenure_days"] = (PREDICTION_DATE - df["signup_date"]).dt.days
df["tenure_years"] = df["tenure_days"] / 365.0

le_plan = LabelEncoder()
df["plan_encoded"] = le_plan.fit_transform(df["plan_type"])
le_gender = LabelEncoder()
df["gender_encoded"] = le_gender.fit_transform(df["gender"])

country_dummies = pd.get_dummies(df["country"], prefix="country")
df = pd.concat([df, country_dummies], axis=1)

# Transaction features
txn = transactions.copy()
txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])
pos_txn = txn[~txn["is_refund"]]

rfm = (
    pos_txn.groupby("customer_id")
    .agg(
        recency_days=("transaction_date", lambda x: (PREDICTION_DATE - x.max()).days),
        frequency=("transaction_date", "count"),
        monetary=("amount", "sum"),
        avg_txn_amount=("amount", "mean"),
        max_txn_amount=("amount", "max"),
        std_txn_amount=("amount", "std"),
    )
    .reset_index()
)

refund = (
    txn[txn["is_refund"]]
    .groupby("customer_id")
    .agg(
        refund_count=("amount", "count"),
        refund_total=("amount", "sum"),
    )
    .reset_index()
)

pmt_diversity = txn.groupby("customer_id")["payment_method"].nunique().reset_index()
pmt_diversity.columns = ["customer_id", "payment_method_count"]

df = df.merge(rfm, on="customer_id", how="left")
df = df.merge(refund, on="customer_id", how="left")
df = df.merge(pmt_diversity, on="customer_id", how="left")

# Event features
ev = events.copy()
ev["event_date"] = pd.to_datetime(ev["event_date"])
ev_agg = (
    ev.groupby("customer_id")
    .agg(
        total_events=("event_type", "count"),
        unique_event_types=("event_type", "nunique"),
        total_event_value=("event_value", "sum"),
        avg_event_value=("event_value", "mean"),
    )
    .reset_index()
)
ev_agg["events_per_day"] = ev_agg["total_events"] / df["tenure_days"].clip(lower=1)

active_days = ev.groupby("customer_id")["event_date"].nunique().reset_index()
active_days.columns = ["customer_id", "active_days"]
active_days["active_days_ratio"] = active_days["active_days"] / df["tenure_days"].clip(
    lower=1
)

df = df.merge(ev_agg, on="customer_id", how="left")
df = df.merge(active_days, on="customer_id", how="left")

# Support ticket features
tkt = tickets.copy()
tkt["ticket_date"] = pd.to_datetime(tkt["ticket_date"])
tkt_agg = (
    tkt.groupby("customer_id")
    .agg(
        total_tickets=("ticket_id", "count"),
        avg_resolution_time=("resolution_time_hrs", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        min_satisfaction=("satisfaction_score", "min"),
        max_resolution_time=("resolution_time_hrs", "max"),
    )
    .reset_index()
)

tkt_pivot = (
    tkt.pivot_table(
        index="customer_id",
        columns="ticket_category",
        values="ticket_id",
        aggfunc="count",
        fill_value=0,
    )
    .add_prefix("tickets_")
    .reset_index()
)

recent_tickets = tkt[tkt["ticket_date"] >= PREDICTION_DATE - timedelta(days=90)]
recent_tkt_agg = (
    recent_tickets.groupby("customer_id")
    .agg(
        recent_tickets=("ticket_id", "count"),
    )
    .reset_index()
)

df = df.merge(tkt_agg, on="customer_id", how="left")
df = df.merge(tkt_pivot, on="customer_id", how="left")
df = df.merge(recent_tkt_agg, on="customer_id", how="left")

# Fill missing
fill_cols = df.columns.difference(
    ["customer_id", "signup_date", "plan_type", "gender", "country"]
)
for col in fill_cols:
    if df[col].dtype in ("int64", "float64"):
        df[col] = df[col].fillna(0)

# Interaction features
df["tenure_fee_interaction"] = df["tenure_days"] * df["monthly_fee"]
df["events_per_ticket"] = df["total_events"] / (df["total_tickets"] + 1)
df["satisfaction_per_ticket"] = df["avg_satisfaction"] / (df["total_tickets"] + 1)
df["monetary_per_tenure"] = df["monetary"] / df["tenure_days"].clip(lower=1)
df["avg_txn_per_event"] = df["avg_txn_amount"] / (df["events_per_day"] + 0.01)

feature_df = df.copy()
print(f"  Feature matrix: {feature_df.shape[0]} rows, {feature_df.shape[1]} columns")

# Correlation heatmap
clv_merged = feature_df.merge(clv_labels, on="customer_id").merge(
    churn_labels, on="customer_id"
)
num_cols = [
    "age",
    "tenure_days",
    "monthly_fee",
    "frequency",
    "monetary",
    "refund_count",
    "total_events",
    "total_tickets",
    "avg_resolution_time",
    "avg_satisfaction",
    "recency_days",
    "clv",
    "churn_30d",
]
clv_merged_num = clv_merged[num_cols].select_dtypes(include=[np.number]).fillna(0)
plt.figure(figsize=(14, 12))
corr_matrix = clv_merged_num.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Heatmap of Key Features")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_correlation_heatmap.png")

# ============================================================
# 4. CLV PREDICTION (REGRESSION)
# ============================================================
print("\n" + "=" * 60)
print("CLV PREDICTION (REGRESSION)")
print("=" * 60)

clv_features = feature_df.merge(clv_labels, on="customer_id", how="inner")
drop_cols = ["customer_id", "signup_date", "plan_type", "gender", "country"]
clv_features = clv_features.drop(
    columns=[c for c in drop_cols if c in clv_features.columns]
)

clv_features["clv_log"] = np.log1p(clv_features["clv"])
target_clv = "clv_log"
feature_cols_clv = [c for c in clv_features.columns if c not in ("clv", "clv_log")]

X_clv = clv_features[feature_cols_clv]
y_clv = clv_features[target_clv]
X_tr_clv, X_te_clv, y_tr_clv, y_te_clv = train_test_split(
    X_clv, y_clv, test_size=0.2, random_state=42
)

scaler_clv = StandardScaler()
X_tr_clv_s = scaler_clv.fit_transform(X_tr_clv)
X_te_clv_s = scaler_clv.transform(X_te_clv)

clv_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, random_state=42
    ),
}
if XGB_AVAILABLE:
    clv_models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
    )

clv_results = []
clv_best = None
clv_best_r2 = -np.inf

for name, model in clv_models.items():
    model.fit(X_tr_clv_s, y_tr_clv)
    y_pred = model.predict(X_te_clv_s)
    y_test_actual = np.expm1(y_te_clv)
    y_pred_actual = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    clv_results.append(
        {"Model": name, "RMSE": f"${rmse:.2f}", "MAE": f"${mae:.2f}", "R²": f"{r2:.4f}"}
    )
    print(f"  {name}: RMSE=${rmse:.2f}, MAE=${mae:.2f}, R²={r2:.4f}")
    if r2 > clv_best_r2:
        clv_best_r2 = r2
        clv_best = model

clv_results_df = pd.DataFrame(clv_results)
print(f"\n  Best model: R² = {clv_best_r2:.4f}")

# Feature importance
if hasattr(clv_best, "feature_importances_"):
    importances = clv_best.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices][::-1], color="forestgreen")
    plt.yticks(range(len(indices)), [feature_cols_clv[i] for i in indices[::-1]])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Features for CLV Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clv_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: clv_feature_importance.png")

# Actual vs Predicted
y_pred_final = clv_best.predict(X_te_clv_s)
y_test_actual = np.expm1(y_te_clv)
y_pred_actual = np.expm1(y_pred_final)

plt.figure(figsize=(10, 8))
plt.scatter(
    y_test_actual,
    y_pred_actual,
    alpha=0.5,
    color="steelblue",
    edgecolors="white",
    linewidth=0.5,
)
plt.plot(
    [y_test_actual.min(), y_test_actual.max()],
    [y_test_actual.min(), y_test_actual.max()],
    "r--",
    linewidth=2,
)
plt.xlabel("Actual CLV ($)")
plt.ylabel("Predicted CLV ($)")
plt.title(f"CLV: Actual vs Predicted (R² = {clv_best_r2:.4f})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/clv_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: clv_actual_vs_predicted.png")

residuals = y_test_actual - y_pred_actual
plt.figure(figsize=(10, 6))
plt.scatter(
    y_pred_actual,
    residuals,
    alpha=0.5,
    color="coral",
    edgecolors="white",
    linewidth=0.5,
)
plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
plt.xlabel("Predicted CLV ($)")
plt.ylabel("Residuals ($)")
plt.title("CLV Residual Plot")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/clv_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: clv_residuals.png")

# ============================================================
# 5. CHURN PREDICTION (CLASSIFICATION)
# ============================================================
print("\n" + "=" * 60)
print("CHURN PREDICTION (CLASSIFICATION)")
print("=" * 60)

churn_features = feature_df.merge(churn_labels, on="customer_id", how="inner")
churn_features = churn_features.drop(
    columns=[c for c in drop_cols if c in churn_features.columns]
)

feature_cols_churn = [c for c in churn_features.columns if c != "churn_30d"]
X_churn = churn_features[feature_cols_churn]
y_churn = churn_features["churn_30d"]

X_tr_churn, X_te_churn, y_tr_churn, y_te_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

scaler_churn = StandardScaler()
X_tr_churn_s = scaler_churn.fit_transform(X_tr_churn)
X_te_churn_s = scaler_churn.transform(X_te_churn)

print(f"  Train churn rate: {y_tr_churn.mean():.2%}")
print(f"  Test churn rate:  {y_te_churn.mean():.2%}")

churn_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, random_state=42
    ),
}
if XGB_AVAILABLE:
    pos_scale = (y_tr_churn == 0).sum() / (y_tr_churn == 1).sum()
    churn_models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=pos_scale,
        random_state=42,
        n_jobs=-1,
    )

churn_results = []
churn_best_model = None
churn_best_auc = -1

for name, model in churn_models.items():
    if IMBALANCED_AVAILABLE and name != "XGBoost":
        pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("model", model)])
        pipeline.fit(X_tr_churn_s, y_tr_churn)
        final_model = pipeline
    else:
        model.fit(X_tr_churn_s, y_tr_churn)
        final_model = model

    y_pred = final_model.predict(X_te_churn_s)
    y_prob = final_model.predict_proba(X_te_churn_s)[:, 1]

    auc = roc_auc_score(y_te_churn, y_prob)
    prec = precision_score(y_te_churn, y_pred)
    rec = recall_score(y_te_churn, y_pred)
    f1 = f1_score(y_te_churn, y_pred)

    churn_results.append(
        {
            "Model": name,
            "AUC-ROC": f"{auc:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1-Score": f"{f1:.4f}",
        }
    )
    print(
        f"  {name}: AUC={auc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}"
    )
    if auc > churn_best_auc:
        churn_best_auc = auc
        churn_best_model = final_model

churn_results_df = pd.DataFrame(churn_results)
print(f"\n  Best model: AUC-ROC = {churn_best_auc:.4f}")

# ROC Curve
plt.figure(figsize=(10, 8))
for name, model in churn_models.items():
    if IMBALANCED_AVAILABLE and name != "XGBoost":
        pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("model", model)])
        pipeline.fit(X_tr_churn_s, y_tr_churn)
        m = pipeline
    else:
        model.fit(X_tr_churn_s, y_tr_churn)
        m = model
    y_prob = m.predict_proba(X_te_churn_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_te_churn, y_prob)
    auc = roc_auc_score(y_te_churn, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Churn Prediction Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/churn_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: churn_roc_curves.png")

# PR Curve
plt.figure(figsize=(10, 8))
for name, model in churn_models.items():
    if IMBALANCED_AVAILABLE and name != "XGBoost":
        pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("model", model)])
        pipeline.fit(X_tr_churn_s, y_tr_churn)
        m = pipeline
    else:
        model.fit(X_tr_churn_s, y_tr_churn)
        m = model
    y_prob = m.predict_proba(X_te_churn_s)[:, 1]
    prec, rec, _ = precision_recall_curve(y_te_churn, y_prob)
    ap = average_precision_score(y_te_churn, y_prob)
    plt.plot(rec, prec, linewidth=2, label=f"{name} (AP = {ap:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - Churn Prediction")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/churn_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: churn_pr_curves.png")

# Confusion Matrix
y_pred_best = churn_best_model.predict(X_te_churn_s)
cm = confusion_matrix(y_te_churn, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Retained", "Churned"],
    yticklabels=["Retained", "Churned"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Best Churn Model")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/churn_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: churn_confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()
print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"  Sensitivity: {tp / (tp + fn):.2%}, Specificity: {tn / (tn + fp):.2%}")

# Churn feature importance
actual_churn_model = churn_best_model
if hasattr(actual_churn_model, "named_steps"):
    actual_churn_model = actual_churn_model.named_steps["model"]

if hasattr(actual_churn_model, "feature_importances_"):
    importances = actual_churn_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn_r(
        importances[indices][::-1] / importances[indices][::-1].max()
    )
    plt.barh(range(len(indices)), importances[indices][::-1], color=colors)
    plt.yticks(range(len(indices)), [feature_cols_churn[i] for i in indices[::-1]])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Features for Churn Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/churn_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: churn_feature_importance.png")

# ============================================================
# 6. CUSTOMER SEGMENTATION (BONUS)
# ============================================================
print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION (BONUS)")
print("=" * 60)

seg_features = [
    "tenure_days",
    "monthly_fee",
    "monetary",
    "total_events",
    "total_tickets",
    "avg_satisfaction",
    "recency_days",
    "frequency",
]
seg_df = feature_df[seg_features].fillna(0)

scaler_seg = StandardScaler()
seg_scaled = scaler_seg.fit_transform(seg_df)

# Elbow
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(seg_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), inertias, marker="o", linewidth=2, color="steelblue")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/segmentation_elbow.png", dpi=150, bbox_inches="tight")
plt.close()

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
feature_df["segment"] = kmeans.fit_predict(seg_scaled)

seg_analysis = (
    feature_df[["customer_id", "segment"]]
    .merge(clv_labels, on="customer_id", how="left")
    .merge(churn_labels, on="customer_id", how="left")
)

seg_summary = (
    seg_analysis.groupby("segment")
    .agg(
        Count=("customer_id", "count"),
        Avg_CLV=("clv", "mean"),
        Median_CLV=("clv", "median"),
        Churn_Rate=("churn_30d", "mean"),
    )
    .reset_index()
)
print(f"\n  Segment Summary:")
for _, row in seg_summary.iterrows():
    print(
        f"    Segment {row['segment']}: {row['Count']} customers, "
        f"Avg CLV=${row['Avg_CLV']:.0f}, Churn Rate={row['Churn_Rate']:.1%}"
    )

# PCA viz
pca = PCA(n_components=2, random_state=42)
seg_pca = pca.fit_transform(seg_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    seg_pca[:, 0],
    seg_pca[:, 1],
    c=feature_df["segment"],
    cmap="viridis",
    alpha=0.6,
    edgecolors="white",
    linewidth=0.3,
)
plt.colorbar(scatter, label="Segment")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("Customer Segments (PCA Projection)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/segmentation_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: segmentation_pca.png, segmentation_elbow.png")

# ============================================================
# 7. SAVE RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS SUMMARY")
print("=" * 60)

with open(f"{OUT_DIR}/results_summary.md", "w") as f:
    f.write("# CLV & Churn Prediction - Results Summary\n\n")
    f.write("## CLV Regression Results\n\n")
    f.write(clv_results_df.to_markdown(index=False) + "\n\n")
    f.write("## Churn Classification Results\n\n")
    f.write(churn_results_df.to_markdown(index=False) + "\n\n")
    f.write("## Segment Summary\n\n")
    f.write(seg_summary.to_markdown(index=False) + "\n\n")
    f.write("## Key Findings\n\n")
    f.write(
        "1. **CLV Drivers**: Tenure, monetary value, plan type (Premium/Enterprise)\n"
    )
    f.write(
        "2. **Churn Drivers**: Low engagement, high recency, low satisfaction, Basic plan\n"
    )
    f.write(
        "3. **High-Value Segments**: Enterprise/Premium customers with high engagement\n"
    )
    f.write("4. **High-Risk Segments**: Basic plan, low tenure, recent inactivity\n")
    f.write(
        "5. **Support Impact**: Low satisfaction and long resolution times correlate with churn\n"
    )

print(f"  Saved: results_summary.md")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"All figures saved to {OUT_DIR}")
