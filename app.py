
import streamlit as st
import pandas as pd
from pathlib import Path

from src.preprocessing import preprocess_data
from src.gridxai_diffusion import run_gridxai_diffusion
from src.explainability import explain_anomalies
from src.evaluation import evaluate_model


st.set_page_config(
    page_title="GridXAI-Diffusion",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ GridXAI-Diffusion")
st.subheader("Interpretable Smart Grid Cybersecurity Analysis")

st.markdown(
    "An adaptive graph intelligence prototype for detecting "
    "unusual smart-grid sensor readings and explaining potential anomalies."
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "smart_grid_data.csv"


@st.cache_data
def load_results():
    raw_data = pd.read_csv(DATA_PATH)
    processed_data = preprocess_data(raw_data)
    results, threshold = run_gridxai_diffusion(processed_data)
    return raw_data, results, threshold


if not DATA_PATH.exists():
    st.error("Dataset not found. Run: python data/generate_data.py")
    st.stop()

try:
    raw_data, results, threshold = load_results()
except Exception as error:
    st.error(f"Unable to process the dataset: {error}")
    st.stop()


total_readings = len(results)
total_nodes = results["node_id"].nunique()
flagged_count = int(results["predicted_anomaly"].sum())
actual_anomalies = int(raw_data["is_anomaly"].sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Sensor Readings", f"{total_readings:,}")
col2.metric("Grid Nodes", total_nodes)
col3.metric("Flagged Readings", flagged_count)
col4.metric("Injected Anomalies", actual_anomalies)

st.divider()

st.subheader("Anomaly Detection Overview")

left, right = st.columns(2)

with left:
    anomaly_counts = (
        results["predicted_anomaly"]
        .map({0: "Normal", 1: "Flagged"})
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Readings")
    )

    st.plotly_chart(
        __import__("plotly.express", fromlist=["express"]).bar(
            anomaly_counts,
            x="Status",
            y="Readings",
            color="Status",
            title="Normal vs Flagged Readings",
        ),
        use_container_width=True,
    )

with right:
    node_summary = (
        results.groupby("node_id", as_index=False)
        .agg(
            Mean_Anomaly_Score=("anomaly_score", "mean"),
            Flagged_Readings=("predicted_anomaly", "sum"),
        )
    )

    import plotly.express as px

    st.plotly_chart(
        px.bar(
            node_summary,
            x="node_id",
            y="Flagged_Readings",
            title="Flagged Readings by Grid Node",
        ),
        use_container_width=True,
    )

st.subheader("Highest-Risk Readings")

risk_data = results.sort_values(
    "anomaly_score", ascending=False
).head(10)

st.dataframe(
    risk_data[
        [
            "timestamp",
            "node_id",
            "voltage",
            "current",
            "frequency",
            "power",
            "anomaly_score",
            "predicted_anomaly",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Explainability")

explanations = explain_anomalies(results, top_n=10)

if explanations.empty:
    st.info("No readings were flagged at the current threshold.")
else:
    st.dataframe(
        explanations[
            [
                "timestamp",
                "node_id",
                "anomaly_score",
                "explanation",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    f"Anomaly threshold: {threshold:.4f} | "
    "Research prototype using synthetic data. "
    "Not validated for operational grid security."
)
