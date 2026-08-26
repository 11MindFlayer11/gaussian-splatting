
import torch
import numpy as np


def knn_to_forward_star(
    knn_indices,
    knn_distances,
    symmetric=True,
    weight_mode="inverse_distance",
    eps=1e-8,
):
    """
    Convert a KNN neighborhood representation into the
    forward-star graph representation required by Cut Pursuit.

    Args:
        knn_indices:
            [N, K] integer tensor containing neighbor indices.

        knn_distances:
            [N, K] tensor containing Euclidean distances.

        symmetric:
            If True, symmetrize the directed KNN graph.

        weight_mode:
            How to compute graph edge weights.

            "uniform":
                w_ij = 1

            "inverse_distance":
                w_ij = 1 / (d_ij + eps)

            "gaussian":
                w_ij = exp(-d_ij^2 / (2 sigma^2))

        eps:
            Numerical stability constant.

    Returns:
        first_edge:
            [N + 1] uint32 NumPy array.

        adj_vertices:
            [E] uint32 NumPy array.

        edge_weights:
            [E] float32 NumPy array.
    """

    knn_indices = knn_indices.detach().cpu()
    knn_distances = knn_distances.detach().cpu()

    N, K = knn_indices.shape

    edges = []

    for i in range(N):

        for j_idx in range(K):

            j = int(knn_indices[i, j_idx])
            d = float(knn_distances[i, j_idx])

            edges.append((i, j, d))

            if symmetric:
                edges.append((j, i, d))

    # --------------------------------------------------
    # Remove duplicate edges
    # --------------------------------------------------

    unique_edges = {}

    for i, j, d in edges:

        key = (i, j)

        if key not in unique_edges:
            unique_edges[key] = d
        else:
            unique_edges[key] = min(unique_edges[key], d)

    edges = sorted(unique_edges.items())

    # --------------------------------------------------
    # Build forward-star representation
    # --------------------------------------------------

    first_edge = np.zeros(N + 1, dtype=np.uint32)

    adj_vertices = []
    edge_distances = []

    current_vertex = 0
    edge_count = 0

    for (i, j), d in edges:

        while current_vertex <= i:
            first_edge[current_vertex] = edge_count
            current_vertex += 1

        adj_vertices.append(j)
        edge_distances.append(d)

        edge_count += 1

    while current_vertex <= N:
        first_edge[current_vertex] = edge_count
        current_vertex += 1

    adj_vertices = np.asarray(
        adj_vertices,
        dtype=np.uint32
    )

    edge_distances = np.asarray(
        edge_distances,
        dtype=np.float32
    )

    # --------------------------------------------------
    # Edge weights
    # --------------------------------------------------

    if weight_mode == "uniform":

        edge_weights = np.ones_like(
            edge_distances,
            dtype=np.float32
        )

    elif weight_mode == "inverse_distance":

        edge_weights = 1.0 / (
            edge_distances + eps
        )

        edge_weights = edge_weights.astype(
            np.float32
        )

    else:

        raise ValueError(
            f"Unknown weight_mode: {weight_mode}"
        )

    return (
        first_edge,
        adj_vertices,
        edge_weights
    )
