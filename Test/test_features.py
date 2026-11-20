import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import features as F

COLUMNS = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"]


def raw_sales():
    rows = [
        ("1001", "A", 2, "2011-01-10 10:00", 5.0, 1.0),    # good
        ("1002", "B", 1, "2011-02-10 10:00", 10.0, 1.0),   # good
        ("1003", "A", 3, "2011-03-01 10:00", 5.0, 2.0),    # good
        ("C1004", "A", 1, "2011-03-02 10:00", 5.0, 2.0),   # cancelled order
        ("1005", "A", -1, "2011-03-03 10:00", 5.0, 2.0),   # negative quantity
        ("1006", "A", 1, "2011-03-04 10:00", 0.0, 2.0),    # zero price
        ("1007", "A", 1, "2011-03-05 10:00", 5.0, np.nan),  # missing customer id
        ("1001", "A", 2, "2011-01-10 10:00", 5.0, 1.0),    # exact duplicate
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_missing_columns_gives_clear_error():
    with pytest.raises(ValueError):
        F.clean_transactions(raw_sales().drop(columns=["Price"]))


def test_cleaning_removes_bad_rows():
    clean = F.clean_transactions(raw_sales())
    assert len(clean) == 3
    assert set(clean["Customer ID"]) == {1, 2}
    assert not clean["Invoice"].astype(str).str.startswith("C").any()
    assert (clean["Quantity"] > 0).all() and (clean["Price"] > 0).all()
    assert list(clean["TotalPrice"]) == [10.0, 10.0, 15.0]


def test_small_file_keeps_all_valid_rows():
    assert len(F.clean_transactions(raw_sales())) == 3


def test_features_are_calculated_correctly():
    clean = F.clean_transactions(raw_sales())
    feat = F.build_features(clean).set_index("Customer ID")

    a = feat.loc[1]   # snapshot = 2011-03-02 10:00
    assert a["Recency"] == 20
    assert a["Frequency"] == 2
    assert a["Monetary"] == 20.0
    assert a["Tenure"] == 51
    assert a["UniqueProducts"] == 2
    assert a["Orders_Last90"] == 2
    assert a["AvgOrderValue"] == 10.0

    b = feat.loc[2]
    assert b["Recency"] == 1
    assert b["Frequency"] == 1
    assert b["Monetary"] == 15.0
    assert b["UniqueProducts"] == 1
    assert b["AvgOrderValue"] == 15.0


def test_old_orders_are_not_counted_in_last_90_days():
    df = raw_sales().iloc[[0, 2]].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df["Customer ID"] = df["Customer ID"].astype(int)
    feat = F.build_features(df, snapshot=pd.Timestamp("2011-09-01")).set_index("Customer ID")
    assert feat.loc[1, "Orders_Last90"] == 0
    assert feat.loc[2, "Orders_Last90"] == 0


def test_feature_columns_match_what_the_model_expects():
    feat = F.build_features(F.clean_transactions(raw_sales()))
    for col in F.CHURN_FEATURES:
        assert col in feat.columns
    assert not feat[F.CHURN_FEATURES].isna().any().any()