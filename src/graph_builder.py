
import networkx as nx


def build_grid_graph(node_ids):
    nodes = sorted(set(node_ids))

    if not nodes:
        raise ValueError("At least one grid node is required.")

    graph = nx.Graph()

    graph.add_nodes_from(nodes)

    # Baseline topology: connect neighboring node IDs.
    for index in range(len(nodes) - 1):
        graph.add_edge(
            nodes[index],
            nodes[index + 1],
            weight=1.0
        )

    # Close the ring when there are at least three nodes.
    if len(nodes) >= 3:
        graph.add_edge(
            nodes[-1],
            nodes[0],
            weight=1.0
        )

    return graph


def get_normalized_adjacency(graph):
    nodes = list(graph.nodes())

    adjacency = nx.to_numpy_array(
        graph,
        nodelist=nodes,
        weight="weight",
        dtype=float
    )

    # Add self-connections.
    import numpy as np

    adjacency += np.eye(len(nodes))

    row_sums = adjacency.sum(axis=1, keepdims=True)

    normalized_adjacency = adjacency / row_sums

    return nodes, normalized_adjacency
