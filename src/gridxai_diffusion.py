
import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS
from src.graph_builder import (
    build_grid_graph,
    get_normalized_adjacency,
)


def run_gridxai_diffusion(
    df,
    diffusion_alpha=0.5,
    threshold_percentile=95,
):
    if not 0 <= diffusion_alpha < 1:
        raise ValueError("diffusion_alpha must be in [0, 1).")

    if not 0 < threshold_percentile < 100:
        raise ValueError("threshold_percentile must be between 0 and 100.")

    if df.empty:
        raise ValueError("Input data cannot be empty.")

    scaled_columns = [f"{col}_scaled" for col in FEATURE_COLUMNS]
    missing = [col for col in scaled_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing scaled features: {missing}")

    data = df.copy()

    # Measure unusual sensor readings without using ground-truth labels.
    sensor_scores = np.sqrt(
        np.mean(np.square(data[scaled_columns].to_numpy()), axis=1)
    )

    # Build a graph representing the grid's baseline connectivity.
    graph = build_grid_graph(data["node_id"].unique())
    nodes, adjacency = get_normalized_adjacency(graph)

    node_to_index = {node: index for index, node in enumerate(nodes)}

    # Aggregate sensor scores for each grid node.
    node_scores = (
        pd.DataFrame({
            "node_id": data["node_id"].to_numpy(),
            "sensor_score": sensor_scores,
        })
        .groupby("node_id")["sensor_score"]
        .mean()
        .reindex(nodes)
        .fillna(0.0)
        .to_numpy()
    )

    # Propagate anomaly evidence through the graph.
    diffused_scores = node_scores.copy()

    for _ in range(10):
        diffused_scores = (
            (1 - diffusion_alpha) * node_scores
            + diffusion_alpha * adjacency @ diffused_scores
        )

    data["sensor_score"] = sensor_scores
    data["diffusion_score"] = data["node_id"].map(
        {
            node: diffused_scores[node_to_index[node]]
            for node in nodes
        }
    )

    # Combine local sensor evidence and graph-level evidence.
    data["anomaly_score"] = (
        0.7 * data["sensor_score"]
        + 0.3 * data["diffusion_score"]
    )

    threshold = float(
        np.percentile(data["anomaly_score"], threshold_percentile)
    )

    data["predicted_anomaly"] = (
        data["anomaly_score"] >= threshold
    ).astype(int)

    return data, threshold
