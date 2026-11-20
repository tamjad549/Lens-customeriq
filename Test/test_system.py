import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src import features as F
from src.explain import top_reasons


def load_models():
    scaler = joblib.load(ROOT / "models" / "rfm_scaler.joblib")
    km = joblib.load(ROOT / "models" / "kmeans_rfm.joblib")
    model = joblib.load(ROOT / "models" / "churn_xgb.joblib")
    return scaler, km, model


def two_customers():
    # order: Recency, Frequency, Monetary, Tenure, UniqueProducts, Orders_Last90, AvgOrderValue
    return pd.DataFrame(
        [
            [5, 30, 9000, 600, 150, 5, 300],   # active, big customer
            [600, 1, 120, 600, 3, 0, 120],     # one small order, long ago
        ],
        columns=F.CHURN_FEATURES,
    )


def scored():
    scaler, km, model = load_models()
    return F.predict_churn(F.assign_segments(two_customers(), scaler, km), model)


def test_segments_and_risk_values_are_valid():
    out = scored()
    assert set(out["Segment"]) <= {"Champions", "Loyal", "At Risk", "Lost"}
    assert out["ChurnRisk"].between(0, 1).all()


def test_segments_make_sense():
    out = scored()
    assert out.loc[0, "Segment"] == "Champions"
    assert out.loc[1, "Segment"] == "Lost"


def test_inactive_customer_is_riskier_than_active_one():
    out = scored()
    assert out.loc[1, "ChurnRisk"] > out.loc[0, "ChurnRisk"]
    assert out.loc[1, "ChurnRisk"] > 0.6
    assert out.loc[0, "ChurnRisk"] < 0.4


def test_reasons_are_text_and_honest():
    _, _, model = load_models()
    feat = two_customers()
    reasons = top_reasons(model, feat, feat)
    assert len(reasons) == 2
    assert all(isinstance(r, str) for r in reasons)
    assert "No purchase for 600 days" in reasons[1]
    assert "Only 30" not in reasons[0]   # a customer with 30 orders is never "only 30"