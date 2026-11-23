import io
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src import features as F
from src.explain import top_reasons

MAX_MB = 150
ORDER = ["Champions", "Loyal", "At Risk", "Lost"]
COLORS = {"Champions": "#2E9E6B", "Loyal": "#3B82F6", "At Risk": "#F59E0B", "Lost": "#E5534B"}
MEANING = {
    "Champions": "Buy often, recently and spend the most. Reward them and keep them happy.",
    "Loyal": "Bought recently and steadily. Keep them engaged with small offers.",
    "At Risk": "Used to buy but have been quiet for months. The best group for win-back offers.",
    "Lost": "Bought rarely and have been gone for a long time. Low priority.",
}

st.set_page_config(page_title="Lens CustomerIQ", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .card {border-left: 6px solid var(--c); background: #FFFFFF; color: #1B2A41;
           padding: 14px 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.10);}
    .card h4 {margin: 0 0 6px 0; color: #1B2A41;}
    .card p {margin: 0; color: #4B5563; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    scaler = joblib.load(ROOT / "models" / "rfm_scaler.joblib")
    km = joblib.load(ROOT / "models" / "kmeans_rfm.joblib")
    model = joblib.load(ROOT / "models" / "churn_xgb.joblib")
    return scaler, km, model


@st.cache_data(ttl=1800, max_entries=3, show_spinner="Analysing customers...")
def analyse(file_bytes):
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        raise ValueError("Could not read this file. Please upload a valid CSV file.")
    clean = F.clean_transactions(raw)
    if len(clean) == 0:
        raise ValueError("No usable rows were found after cleaning. Please check the file.")
    feat = F.build_features(clean)
    scaler, km, model = load_models()
    result = F.predict_churn(F.assign_segments(feat, scaler, km), model)
    return result, len(raw), len(clean)


st.title("Lens CustomerIQ")
st.caption("Find your best customers, and the ones who are about to stop buying.")

with st.expander("How does this work?"):
    st.markdown(
        "1. Upload your sales history (a CSV file).\n"
        "2. The app cleans the data and describes every customer by how recently, "
        "how often and how much they buy.\n"
        "3. Customers are grouped into four segments, and a model estimates the chance "
        "that each customer buys nothing in the next 90 days.\n"
        "4. You get a call list of customers worth contacting, with the reasons."
    )

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

if len(result) == 0:
    st.error("No usable rows were found after cleaning. Please check the file.")
    st.stop()

result = result.copy()
result["Risk %"] = (result["ChurnRisk"] * 100).round(1)
_, _, model = load_models()

# ---- Key numbers ----
high = result[result["ChurnRisk"] >= 0.7]
worth = high[high["Segment"] != "Lost"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(result):,}")
k2.metric("High risk (70%+)", f"{len(high):,}")
k3.metric("Worth calling (high risk, not Lost)", f"{len(worth):,}")
k4.metric("Past spend of those customers", f"{worth['Monetary'].sum():,.0f}")
st.caption(f"{n_clean:,} of {n_raw:,} rows were used after cleaning.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segments", "Call list", "Customer lookup"])

# ---- Overview ----
with tab1:
    left, right = st.columns(2)
    counts = pd.DataFrame({
        "Segment": ORDER,
        "Customers": [int((result["Segment"] == s).sum()) for s in ORDER],
    })
    fig1 = px.pie(counts, names="Segment", values="Customers", hole=0.55,
                  color="Segment", color_discrete_map=COLORS,
                  category_orders={"Segment": ORDER})
    fig1.update_layout(title="Customers by segment", margin=dict(t=50, b=10, l=10, r=10))
    left.plotly_chart(fig1)

    avg = pd.DataFrame({
        "Segment": ORDER,
        "Average churn risk %": [
            round(float(result.loc[result["Segment"] == s, "Risk %"].mean()), 1)
            if (result["Segment"] == s).any() else 0.0
            for s in ORDER
        ],
    })
    fig2 = px.bar(avg, x="Segment", y="Average churn risk %", color="Segment",
                  color_discrete_map=COLORS, category_orders={"Segment": ORDER})
    fig2.update_layout(title="Average churn risk by segment", showlegend=False,
                       yaxis_range=[0, 100], margin=dict(t=50, b=10, l=10, r=10))
    right.plotly_chart(fig2)

    fig3 = px.histogram(result, x="Risk %", nbins=20, color_discrete_sequence=["#1F6FEB"])
    fig3.update_layout(title="How risky are my customers?", xaxis_title="Churn risk %",
                       yaxis_title="Customers", bargap=0.05,
                       margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig3)

# ---- Segments ----
with tab2:
    cols = st.columns(4)
    for col, s in zip(cols, ORDER):
        col.markdown(
            f'<div class="card" style="--c:{COLORS[s]}"><h4>{s}</h4><p>{MEANING[s]}</p></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    profile = result.groupby("Segment").agg(
        Customers=("Customer ID", "count"),
        Recency=("Recency", "median"),
        Frequency=("Frequency", "median"),
        Monetary=("Monetary", "median"),
        Risk=("Risk %", "mean"),
    ).reindex(ORDER).round(1)
    profile.insert(1, "Share %", (profile["Customers"] / len(result) * 100).round(1))
    profile = profile.rename(columns={
        "Recency": "Median recency (days)",
        "Frequency": "Median orders",
        "Monetary": "Median spend",
        "Risk": "Average churn risk %",
    })
    profile.index.name = "Segment"
    st.dataframe(profile)

# ---- Call list ----
with tab3:
    c1, c2, c3 = st.columns(3)
    segs = c1.multiselect("Segments", ORDER, default=["Champions", "Loyal", "At Risk"])
    threshold = c2.slider("Minimum churn risk %", 0, 100, 70, 5)
    sort_by = c3.selectbox("Sort by", ["Churn risk", "Past spend (money at stake)"])

    calls = result[result["Segment"].isin(segs) & (result["ChurnRisk"] >= threshold / 100)]
    sort_col = "Risk %" if sort_by == "Churn risk" else "Monetary"
    calls = calls.sort_values(sort_col, ascending=False).copy()
    if len(calls) > 0:
        calls["Main reasons"] = top_reasons(model, calls, result)
    else:
        calls["Main reasons"] = []

    show = calls[["Customer ID", "Segment", "Risk %", "Main reasons",
                  "Recency", "Frequency", "Monetary"]].rename(columns={"Risk %": "Churn risk %"})
    st.write(f"{len(show):,} customers match")
    st.dataframe(
        show,
        hide_index=True,
        column_config={
            "Churn risk %": st.column_config.ProgressColumn(
                "Churn risk %", min_value=0, max_value=100, format="%.1f"),
        },
    )
    st.download_button("Download call list (CSV)", show.to_csv(index=False),
                       file_name="call_list.csv", mime="text/csv")

# ---- Customer lookup ----
with tab4:
    cid = st.selectbox("Choose a Customer ID", sorted(result["Customer ID"].tolist()))
    row = result[result["Customer ID"] == cid].iloc[[0]]
    reasons = top_reasons(model, row, result)[0]
    r = row.iloc[0]
    a, b, c = st.columns(3)
    a.metric("Churn risk", f"{r['Risk %']:.1f}%")
    b.metric("Segment", str(r["Segment"]))
    c.metric("Past spend", f"{r['Monetary']:,.0f}")
    st.write("Main reasons: " + reasons)
    st.dataframe(row[F.CHURN_FEATURES], hide_index=True)

st.caption("A churn risk is an estimate, not a certainty. The model was trained on the public "
           "Online Retail II data.")