import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src import features as F
from src.explain import top_reasons

app = FastAPI(title="Lens CustomerIQ API", version="1.0")

scaler = joblib.load(ROOT / "models" / "rfm_scaler.joblib")
kmeans = joblib.load(ROOT / "models" / "kmeans_rfm.joblib")
model = joblib.load(ROOT / "models" / "churn_xgb.joblib")
with open(ROOT / "models" / "feature_medians.json") as f:
    reference = pd.DataFrame([json.load(f)])


class Customer(BaseModel):
    Recency: float = Field(..., ge=0, description="Days since last order")
    Frequency: float = Field(..., ge=1, description="Number of orders")
    Monetary: float = Field(..., ge=0, description="Total spent")
    Tenure: float = Field(..., ge=0, description="Days since first order")
    UniqueProducts: float = Field(..., ge=1, description="Different products bought")
    Orders_Last90: float = Field(..., ge=0, description="Orders in the last 90 days")
    AvgOrderValue: float = Field(..., ge=0, description="Monetary divided by Frequency")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    feat = pd.DataFrame([customer.model_dump()])
    feat = F.assign_segments(feat, scaler, kmeans)
    feat = F.predict_churn(feat, model)
    return {
        "churn_risk": round(float(feat.loc[0, "ChurnRisk"]), 4),
        "segment": feat.loc[0, "Segment"],
        "main_reasons": top_reasons(model, feat, reference)[0],
    }
