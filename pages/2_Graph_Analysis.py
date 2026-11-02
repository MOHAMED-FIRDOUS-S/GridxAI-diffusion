
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path

from src.preprocessing import preprocess_data
from src.graph_builder import build_grid_graph
from src.gridxai_diffusion import run_gridxai_diffusion

st.set_page_config(
    page_title="Graph Analysis | GridXAI",
    page_icon="🕸️",
    layout="wide",
)

st.title("🕸️ Smart Grid Graph Analysis")
st.write(
    "Visualize grid-node connectivity and inspect anomaly scores "
    "calculated by the GridXAI-Diffusion prototype."
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

graph = build_grid_graph(data["node_id"].unique())

node_summary = (
    results.groupby("node_id", as_index=False)
    .agg(
        Mean_Anomaly_Score=("anomaly_score", "mean"),
        Flagged_Readings=("predicted_anomaly", "sum"),
    )
)

st.subheader("Grid Network Topology")

positions = nx.spring_layout(graph, seed=42)

edge_x = []
edge_y = []

for source, target in graph.edges():
    x0, y0 = positions[source]
    x1, y1 = positions[target]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(width=1.5, color="#94a3b8"),
    hoverinfo="none",
    name="Grid connections",
)

node_x = [positions[node][0] for node in graph.nodes()]
node_y = [positions[node][1] for node in graph.nodes()]

summary_map = node_summary.set_index("node_id")

node_scores = [
    float(summary_map.loc[node, "Mean_Anomaly_Score"])
    for node in graph.nodes()
]

node_flags = [
    int(summary_map.loc[node, "Flagged_Readings"])
    for node in graph.nodes()
]

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=list(graph.nodes()),
    textposition="top center",
    customdata=list(zip(node_scores, node_flags)),
    marker=dict(
        size=22,
        color=node_scores,
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="Mean score"),
        line=dict(width=1, color="#334155"),
    ),
    hovertemplate=(
        "<b>%{text}</b>"
        "<br>Mean anomaly score: %{customdata[0]:.3f}"
        "<br>Flagged readings: %{customdata[1]}"
        "<extra></extra>"
    ),
    name="Grid nodes",
)

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    height=600,
    title="Baseline Grid Connectivity",
    showlegend=False,
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)

st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Grid Nodes", graph.number_of_nodes())
col2.metric("Connections", graph.number_of_edges())
col3.metric("Flagged Readings", int(results["predicted_anomaly"].sum()))

st.caption(
    f"Detection threshold: {threshold:.4f}. "
    "Node colors represent mean anomaly scores."
)

st.subheader("Node-Level Analysis")
st.dataframe(
    node_summary.sort_values(
        "Mean_Anomaly_Score", ascending=False
    ),
    use_container_width=True,
    hide_index=True,
)
