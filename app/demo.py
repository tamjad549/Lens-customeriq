import io
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src import features as F

from src.explain import top_reasons
MAX_MB = 150

st.set_page_config(page_title="Lens CustomerIQ", layout="wide")
st.title("Lens CustomerIQ")
st.caption("Customer segments and churn risk for shops")


@st.cache_resource
def load_models():
    scaler = joblib.load(ROOT / "models" / "rfm_scaler.joblib")
    km = joblib.load(ROOT / "models" / "kmeans_rfm.joblib")
    model = joblib.load(ROOT / "models" / "churn_xgb.joblib")
    return scaler, km, model


@st.cache_data(show_spinner="Analysing customers...")
def analyse(file_bytes):
    raw = pd.read_csv(io.BytesIO(file_bytes))
    clean = F.clean_transactions(raw)
    feat = F.build_features(clean)
    scaler, km, model = load_models()
    result = F.predict_churn(F.assign_segments(feat, scaler, km), model)
    return result, len(raw), len(clean)


uploaded = st.file_uploader("Upload your sales history (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV with these columns: " + ", ".join(F.REQUIRED_COLUMNS))
    st.stop()

if uploaded.size > MAX_MB * 1024 * 1024:
    st.error(f"File is too large. Please upload a file under {MAX_MB} MB.")
    st.stop()

try:
    result, n_raw, n_clean = analyse(uploaded.getvalue())
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception:
    st.error("Could not read this file. Please check that it is a valid sales CSV.")
    st.stop()

# ---- Summary ----
c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{len(result):,}")
c2.metric("Rows used after cleaning", f"{n_clean:,} of {n_raw:,}")
c3.metric("High risk (70%+)", f"{(result['ChurnRisk'] >= 0.7).sum():,}")

# ---- Segments ----
st.subheader("Customer segments")
order = ["Champions", "Loyal", "At Risk", "Lost"]
left, right = st.columns(2)
left.bar_chart(result["Segment"].value_counts().reindex(order))
profile = (result.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
           .median().round(1).reindex(order))
profile.insert(0, "Customers", result["Segment"].value_counts().reindex(order))
right.dataframe(profile, use_container_width=True)

# ---- Call list ----
st.subheader("Call list: customers likely to stop buying")
segs = st.multiselect("Segments to include", order, default=["Champions", "Loyal", "At Risk"])
threshold = st.slider("Minimum churn risk", 0.0, 1.0, 0.7, 0.05)

calls = result[result["Segment"].isin(segs) & (result["ChurnRisk"] >= threshold)]
calls = calls.sort_values("ChurnRisk", ascending=False).copy()
calls["Churn risk %"] = (calls["ChurnRisk"] * 100).round(1)
if len(calls) > 0:
    _, _, model = load_models()
    calls["Main reasons"] = top_reasons(model, calls,result)
else:
    calls["Main reasons"] = []
calls = calls[["Customer ID", "Segment", "Churn risk %", "Main reasons",
               "Recency", "Frequency", "Monetary"]]

st.write(f"{len(calls):,} customers match")
st.dataframe(calls, use_container_width=True, hide_index=True)
st.download_button("Download call list (CSV)", calls.to_csv(index=False),
                   file_name="call_list.csv", mime="text/csv")