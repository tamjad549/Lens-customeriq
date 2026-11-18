import numpy as np
import xgboost as xgb

from src.features import CHURN_FEATURES

REASON_TEXT = {
    "Recency": lambda v: f"No purchase for {v:.0f} days",
    "Frequency": lambda v: f"Only {v:.0f} order(s) in total",
    "Monetary": lambda v: f"Low total spending ({v:,.0f})",
    "Tenure": lambda v: f"Long time since first purchase ({v:.0f} days)",
    "UniqueProducts": lambda v: f"Few different products bought ({v:.0f})",
    "Orders_Last90": lambda v: f"Only {v:.0f} order(s) in the last 90 days",
    "AvgOrderValue": lambda v: f"Average order value of {v:,.0f}",
}

# Reason tabhi dikhate hain jab value risk wali taraf ho
HIGH_IS_RISKY = {"Recency", "Tenure"}
LOW_IS_RISKY = {"Frequency", "Monetary", "UniqueProducts", "Orders_Last90"}


def top_reasons(model, feat, reference, n=2):
    X = feat[CHURN_FEATURES]
    median = reference[CHURN_FEATURES].median()
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    reasons = []
    for i in range(len(X)):
        parts = []
        for j in np.argsort(-contrib[i]):
            if contrib[i, j] <= 0 or len(parts) == n:
                break
            name = CHURN_FEATURES[j]
            value = X.iloc[i, j]
            if name in HIGH_IS_RISKY and value < median[name]:
                continue
            if name in LOW_IS_RISKY and value > median[name]:
                continue
            parts.append(REASON_TEXT[name](value))
        reasons.append("; ".join(parts) if parts else "Mixed signals")
    return reasons