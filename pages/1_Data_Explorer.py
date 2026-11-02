
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.preprocessing import preprocess_data

st.set_page_config(
    page_title="Data Explorer | GridXAI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Smart Grid Data Explorer")
st.write("Explore sensor readings, measurements, and injected anomalies.")

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "smart_grid_data.csv"
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Run: python data/generate_data.py")
    st.stop()

df = pd.read_csv(DATA_PATH)
df = preprocess_data(df)

st.sidebar.header("Data Filters")

nodes = sorted(df["node_id"].unique())
selected_nodes = st.sidebar.multiselect(
    "Select grid nodes",
    nodes,
    default=nodes,
)

status = st.sidebar.selectbox(
    "Reading type",
    ["All readings", "Normal", "Injected anomalies"],
)

filtered = df[df["node_id"].isin(selected_nodes)].copy()

if status == "Normal":
    filtered = filtered[filtered["is_anomaly"] == 0]
elif status == "Injected anomalies":
    filtered = filtered[filtered["is_anomaly"] == 1]

c1, c2, c3 = st.columns(3)
c1.metric("Filtered Readings", f"{len(filtered):,}")
c2.metric("Grid Nodes", filtered["node_id"].nunique())
c3.metric("Injected Anomalies", int(filtered["is_anomaly"].sum()))

st.subheader("Sensor Measurements")

if filtered.empty:
    st.warning("No readings match the selected filters.")
else:
    feature = st.selectbox(
        "Choose a measurement",
        ["voltage", "current", "frequency", "power"],
    )

    chart = px.line(
        filtered.sort_values("timestamp"),
        x="timestamp",
        y=feature,
        color="node_id",
        title=f"{feature.title()} over Time",
        labels={feature: feature.title(), "timestamp": "Time"},
    )
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Dataset Records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)",
        data=csv,
        file_name="gridxai_filtered_data.csv",
        mime="text/csv",
    )
