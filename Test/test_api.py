import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

GOOD = {"Recency": 443, "Frequency": 1, "Monetary": 2803.2, "Tenure": 443,
        "UniqueProducts": 1, "Orders_Last90": 0, "AvgOrderValue": 2803.2}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_risk_segment_and_reasons():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["churn_risk"] <= 1
    assert data["segment"] in {"Champions", "Loyal", "At Risk", "Lost"}
    assert isinstance(data["main_reasons"], str)


def test_known_customer_matches_the_dashboard():
    data = client.post("/predict", json=GOOD).json()   # customer 13205
    assert data["segment"] == "At Risk"
    assert round(data["churn_risk"], 2) == 0.96


@pytest.mark.parametrize("field,value", [
    ("Recency", -1), ("Frequency", 0), ("Monetary", -50), ("UniqueProducts", 0),
])
def test_bad_values_are_rejected(field, value):
    body = dict(GOOD)
    body[field] = value
    assert client.post("/predict", json=body).status_code == 422


def test_missing_field_is_rejected():
    body = dict(GOOD)
    del body["Tenure"]
    assert client.post("/predict", json=body).status_code == 422


def test_text_instead_of_number_is_rejected():
    body = dict(GOOD)
    body["Recency"] = "abc"
    assert client.post("/predict", json=body).status_code == 422