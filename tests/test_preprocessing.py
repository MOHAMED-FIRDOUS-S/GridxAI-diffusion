
import pandas as pd

from src.preprocessing import preprocess_data
from src.graph_builder import build_grid_graph, get_normalized_adjacency


def test_preprocessing():
    df = pd.read_csv("data/smart_grid_data.csv")
    result = preprocess_data(df)

    assert len(result) == 1000
    assert "voltage_scaled" in result.columns
    assert result["is_anomaly"].sum() == 15


def test_graph_builder():
    nodes = ["Node_00", "Node_01", "Node_02"]

    graph = build_grid_graph(nodes)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3

    ordered_nodes, matrix = get_normalized_adjacency(graph)

    assert len(ordered_nodes) == 3
    assert matrix.shape == (3, 3)
    assert abs(matrix[0].sum() - 1.0) < 1e-9
