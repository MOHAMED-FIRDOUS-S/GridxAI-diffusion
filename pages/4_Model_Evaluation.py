import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from src.preprocessing import preprocess_data
from src.gridxai_diffusion import run_gridxai_diffusion
from src.evaluation import evaluate_model

st.set_page_config(
    page_title="Model Evaluation | GridXAI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Model Evaluation")
st.write(
    "Evaluate anomaly detection performance against the injected "
    "anomaly labels in the synthetic dataset."
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

metrics, matrix, report = evaluate_model(results)

st.subheader("Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
c2.metric("Precision", f"{metrics['Precision']:.2%}")
c3.metric("Recall", f"{metrics['Recall']:.2%}")
c4.metric("F1-score", f"{metrics['F1-score']:.2%}")

st.caption(f"Detection threshold: {threshold:.4f}")

st.divider()

st.subheader("Confusion Matrix")

st.write(
    "Rows represent actual labels; columns represent predicted labels."
)


fig = go.Figure(
    data=go.Heatmap(
        z=matrix.tolist(),
        x=["Predicted Normal", "Predicted Anomaly"],
        y=["Actual Normal", "Actual Anomaly"],
        colorscale="Blues",
        showscale=True,
        text=matrix.tolist(),
        texttemplate="%{text}",
        hovertemplate=(
            "Actual: %{y}<br>"
            "Prediction: %{x}<br>"
            "Count: %{z}<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="Actual vs Predicted Readings",
    xaxis_title="Prediction",
    yaxis_title="Actual Label",
    height=450,
)

st.plotly_chart(fig, use_container_width=True)




st.subheader("Classification Report")
st.code(report, language="text")

st.subheader("Prediction Summary")

actual_anomalies = int(raw_data["is_anomaly"].sum())
predicted_anomalies = int(results["predicted_anomaly"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Total Readings", f"{len(results):,}")
c2.metric("Actual Injected Anomalies", actual_anomalies)
c3.metric("Predicted Anomalies", predicted_anomalies)

st.warning(
    "These metrics are based on synthetic data. The current prototype "
    "detects all injected anomalies but also produces false positives. "
    "It has not been validated for real-world smart-grid cybersecurity."
)
