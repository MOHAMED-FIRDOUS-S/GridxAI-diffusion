
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.preprocessing import preprocess_data
from src.gridxai_diffusion import run_gridxai_diffusion
from src.explainability import explain_anomalies

st.set_page_config(
    page_title="Explainability | GridXAI",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Anomaly Explainability")
st.write(
    "Inspect flagged sensor readings and identify the measurement "
    "with the strongest standardized deviation."
)

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "smart_grid_data.csv"
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Run: python data/generate_data.py")
    st.stop()

raw_data = pd.read_csv(DATA_PATH)
data = preprocess_data(raw_data)
results, threshold = run_gridxai_diffusion(data)

top_n = st.slider(
    "Number of flagged readings to display",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
)

explanations = explain_anomalies(results, top_n=top_n)

c1, c2 = st.columns(2)
c1.metric("Detection Threshold", f"{threshold:.4f}")
c2.metric("Flagged Readings", int(results["predicted_anomaly"].sum()))

st.subheader("Highest-Risk Readings")

if explanations.empty:
    st.info("No readings were flagged.")
    st.stop()

st.dataframe(
    explanations[
        [
            "timestamp",
            "node_id",
            "voltage",
            "current",
            "frequency",
            "power",
            "anomaly_score",
            "explanation",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Anomaly Score Comparison")

fig = px.bar(
    explanations,
    x="node_id",
    y="anomaly_score",
    color="node_id",
    hover_data=["timestamp", "explanation"],
    title="Highest-Risk Readings by Grid Node",
)
fig.add_hline(
    y=threshold,
    line_dash="dash",
    annotation_text="Detection threshold",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Selected Reading Details")

selected_index = st.selectbox(
    "Choose a reading to inspect",
    options=list(range(len(explanations))),
    format_func=lambda i: (
        f"{explanations.iloc[i]['node_id']} — "
        f"Score {explanations.iloc[i]['anomaly_score']:.3f}"
    ),
)

selected = explanations.iloc[selected_index]

st.write(f"**Grid node:** {selected['node_id']}")
st.write(f"**Timestamp:** {selected['timestamp']}")
st.write(f"**Anomaly score:** {selected['anomaly_score']:.4f}")
st.write(f"**Explanation:** {selected['explanation']}")

st.info(
    "This explanation identifies the strongest individual sensor deviation. "
    "It is not a causal explanation and does not prove a cyberattack occurred."
)
