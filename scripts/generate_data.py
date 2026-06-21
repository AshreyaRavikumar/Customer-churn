"""
Generate synthetic data for the CLV & Churn Prediction case study.
Produces 6 CSV files with realistic patterns and correlations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

N_CUSTOMERS = 5000
PREDICTION_DATE = datetime(2024, 12, 31)
SIGNUP_START = datetime(2020, 1, 1)


# ---- Helper functions ----
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))


def generate_customers(n):
    customers = []
    for i in range(n):
        cid = f"C{i + 1:06d}"
        signup = random_date(SIGNUP_START, PREDICTION_DATE - timedelta(days=60))
        tenure_days = (PREDICTION_DATE - signup).days
        age = int(np.clip(np.random.normal(38, 12), 18, 85))
        gender = random.choice(["Male", "Female", "Non-Binary"])
        country = random.choices(
            [
                "USA",
                "UK",
                "Canada",
                "Germany",
                "France",
                "Australia",
                "India",
                "Brazil",
            ],
            weights=[30, 15, 10, 10, 8, 7, 12, 8],
            k=1,
        )[0]
        plan_type = random.choices(
            ["Basic", "Standard", "Premium", "Enterprise"],
            weights=[40, 30, 20, 10],
            k=1,
        )[0]
        monthly_fee = {
            "Basic": 9.99,
            "Standard": 19.99,
            "Premium": 39.99,
            "Enterprise": 99.99,
        }[plan_type]
        customers.append(
            {
                "customer_id": cid,
                "signup_date": signup,
                "age": age,
                "gender": gender,
                "country": country,
                "plan_type": plan_type,
                "monthly_fee": monthly_fee,
                "tenure_days": tenure_days,
            }
        )
    return pd.DataFrame(customers)


def generate_transactions(customers_df):
    rows = []
    for _, c in customers_df.iterrows():
        cid = c["customer_id"]
        signup = c["signup_date"]
        tenure = c["tenure_days"]
        plan = c["monthly_fee"]
        # Number of transactions per customer: monthly payments + occasional extra
        n_trans = max(1, int(tenure / 30 * random.uniform(0.8, 1.3)))
        n_trans = int(np.random.poisson(max(1, n_trans)))
        # Random refund probability (5%)
        for t in range(n_trans):
            txn_date = signup + timedelta(days=random.randint(0, tenure - 1))
            is_refund = random.random() < 0.05
            # Amount: typically monthly fee, sometimes extra purchases
            if is_refund:
                amount = -random.uniform(5, plan)
            else:
                amount = plan if random.random() < 0.7 else random.uniform(5, plan * 2)
            pmt = random.choices(
                ["Credit Card", "PayPal", "Debit Card", "Bank Transfer"],
                weights=[40, 30, 20, 10],
                k=1,
            )[0]
            rows.append(
                {
                    "customer_id": cid,
                    "transaction_date": txn_date,
                    "amount": round(amount, 2),
                    "payment_method": pmt,
                    "is_refund": is_refund,
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(["customer_id", "transaction_date"]).reset_index(drop=True)
    return df


def generate_events(customers_df):
    event_types = [
        "login",
        "page_view",
        "feature_use",
        "content_upload",
        "purchase",
        "logout",
    ]
    weights = [35, 25, 15, 5, 10, 10]
    rows = []
    for _, c in customers_df.iterrows():
        cid = c["customer_id"]
        signup = c["signup_date"]
        tenure = c["tenure_days"]
        # High-activity customers have more events; about to churn have fewer
        churn_risk_factor = 1.0
        if c.get("churn_prob", 0) > 0.5:
            churn_risk_factor = 0.3
        n_events = int(np.random.poisson(max(1, tenure / 7 * churn_risk_factor)))
        for _ in range(n_events):
            evt_date = signup + timedelta(days=random.randint(0, max(0, tenure - 1)))
            evt_type = random.choices(event_types, weights=weights, k=1)[0]
            evt_val = 0.0
            if evt_type in ("purchase", "feature_use"):
                evt_val = round(random.uniform(1, 50), 2)
            elif evt_type == "content_upload":
                evt_val = round(random.uniform(10, 200), 2)
            rows.append(
                {
                    "customer_id": cid,
                    "event_date": evt_date,
                    "event_type": evt_type,
                    "event_value": evt_val,
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(["customer_id", "event_date"]).reset_index(drop=True)
    return df


def generate_support_tickets(customers_df):
    categories = ["billing", "technical", "account", "feature_request", "cancellation"]
    rows = []
    for _, c in customers_df.iterrows():
        cid = c["customer_id"]
        signup = c["signup_date"]
        tenure = c["tenure_days"]
        # Average ~0.5 tickets per customer; churners file more
        churn_factor = 1.0
        if c.get("churn_prob", 0) > 0.5:
            churn_factor = 2.5
        n_tickets = int(np.random.poisson(max(0, 0.5 * (tenure / 365) * churn_factor)))
        for t in range(n_tickets):
            tid = f"T{cid[1:]}_{t + 1:02d}"
            tkt_date = signup + timedelta(days=random.randint(0, max(0, tenure - 1)))
            cat = random.choices(categories, weights=[25, 35, 20, 10, 10], k=1)[0]
            res_hrs = round(max(0.5, np.random.exponential(8)), 1)
            sat = random.randint(1, 5)
            # Unhappy customers: lower satisfaction & higher resolution time
            if churn_factor > 1.5:
                sat = random.randint(1, 3)
                res_hrs = round(max(1, np.random.exponential(16)), 1)
            rows.append(
                {
                    "ticket_id": tid,
                    "customer_id": cid,
                    "ticket_date": tkt_date,
                    "ticket_category": cat,
                    "resolution_time_hrs": res_hrs,
                    "satisfaction_score": sat,
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(["customer_id", "ticket_date"]).reset_index(drop=True)
    return df


def generate_clv_and_churn(customers_df):
    """Assign CLV and churn labels with realistic relationships to customer attributes."""
    np.random.seed(42)
    clvs = []
    churns = []
    for _, c in customers_df.iterrows():
        cid = c["customer_id"]
        tenure = c["tenure_days"] / 365.0
        fee = c["monthly_fee"]
        plan = c["plan_type"]

        # Base CLV from tenure and fee
        base_clv = tenure * fee * 12 * random.uniform(0.8, 1.0)

        # Modifiers
        age_mod = 1.0
        if c["age"] < 30:
            age_mod = 0.85
        elif c["age"] > 55:
            age_mod = 1.15

        country_mod = {
            "USA": 1.1,
            "UK": 1.05,
            "Canada": 1.05,
            "Germany": 1.0,
            "France": 0.95,
            "Australia": 1.08,
            "India": 0.8,
            "Brazil": 0.85,
        }
        country_mod_val = country_mod.get(c["country"], 1.0)

        noise = np.random.normal(1, 0.1)
        clv = base_clv * age_mod * country_mod_val * noise
        clv = max(0, round(clv, 2))
        clvs.append(clv)

        # Churn probability: higher if low tenure, low fee, low CLV, certain countries
        churn_logit = (
            0.5
            + 0.5 * (30 / max(1, c["age"]))
            - 1.5 * tenure
            - 0.3 * (fee / 10)
            + (0.8 if c["country"] in ("India", "Brazil") else 0.0)
            - 0.5 * (plan == "Premium")
            - 0.8 * (plan == "Enterprise")
            + np.random.normal(0, 0.8)
        )
        churn_prob = 1 / (1 + np.exp(-churn_logit))
        churn = 1 if churn_prob > 0.3 else 0
        churns.append(churn)

    customers_df["churn_prob"] = churns  # store for event gen
    return clvs, churns


# ---- Main generation ----
print("Generating customers...")
customers = generate_customers(N_CUSTOMERS)

print("Generating CLV and churn targets...")
clv_values, churn_values = generate_clv_and_churn(customers)

# Remove churn_prob helper before saving
churn_df = customers[["customer_id"]].copy()
churn_df["churn_30d"] = churn_values

clv_df = customers[["customer_id"]].copy()
clv_df["clv"] = clv_values

# Drop helper columns & save customers
customers_export = customers.drop(columns=["churn_prob", "tenure_days"])
customers_export.to_csv("customers.csv", index=False)
print("Saved customers.csv")

churn_df.to_csv("churn_labels.csv", index=False)
print("Saved churn_labels.csv")

clv_df.to_csv("clv_labels.csv", index=False)
print("Saved clv_labels.csv")

print("Generating transactions...")
transactions = generate_transactions(customers)
transactions.to_csv("transactions.csv", index=False)
print("Saved transactions.csv")

print("Generating events...")
events = generate_events(customers)
events.to_csv("events.csv", index=False)
print("Saved events.csv")

print("Generating support tickets...")
tickets = generate_support_tickets(customers)
tickets.to_csv("support_tickets.csv", index=False)
print("Saved support_tickets.csv")

print("\nAll datasets generated successfully!")
for name in [
    "customers.csv",
    "transactions.csv",
    "events.csv",
    "support_tickets.csv",
    "churn_labels.csv",
    "clv_labels.csv",
]:
    df = pd.read_csv(name)
    print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols")
