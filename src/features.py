import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"]
RFM_COLS = ["Recency", "Frequency", "Monetary"]
CHURN_FEATURES = ["Recency", "Frequency", "Monetary", "Tenure",
                  "UniqueProducts", "Orders_Last90", "AvgOrderValue"]
# Ye mapping saved K-Means model ke saath bandhi hui hai
SEGMENT_NAMES = {3: "Champions", 1: "Loyal", 0: "At Risk", 2: "Lost"}


def check_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))


def clean_transactions(df):
    check_columns(df)
    df = df.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df.drop_duplicates().copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    q_limit = df["Quantity"].quantile(0.999)
    p_limit = df["Price"].quantile(0.999)
    return df[(df["Quantity"] <= q_limit) & (df["Price"] <= p_limit)]


def build_features(df, snapshot=None):
    if snapshot is None:
        snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    feat = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        UniqueProducts=("StockCode", "nunique"),
    ).reset_index()
    feat["Tenure"] = (snapshot - feat["FirstPurchase"]).dt.days

    recent = df[df["InvoiceDate"] > snapshot - pd.Timedelta(days=91)]
    orders_90 = (recent.groupby("Customer ID")["Invoice"].nunique()
                 .rename("Orders_Last90").reset_index())
    feat = feat.merge(orders_90, on="Customer ID", how="left")
    feat["Orders_Last90"] = feat["Orders_Last90"].fillna(0)
    feat["AvgOrderValue"] = feat["Monetary"] / feat["Frequency"]
    return feat.drop(columns="FirstPurchase")


def assign_segments(feat, scaler, kmeans):
    x = scaler.transform(np.log1p(feat[RFM_COLS]))
    out = feat.copy()
    out["Segment"] = pd.Series(kmeans.predict(x), index=out.index).map(SEGMENT_NAMES)
    return out


def predict_churn(feat, model):
    out = feat.copy()
    out["ChurnRisk"] = model.predict_proba(out[CHURN_FEATURES])[:, 1]
    return out
