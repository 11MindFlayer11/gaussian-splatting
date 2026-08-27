
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
    Convert KNN representation into the forward-star graph
    representation required by Cut Pursuit.

    The Cut Pursuit boundary is CPU/NumPy, so tensors are moved
    to CPU once and all graph construction is vectorized.
    """

    # ------------------------------------------------------------
    # Move to CPU once
    # ------------------------------------------------------------

    knn_indices = (
        knn_indices
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    knn_distances = (
        knn_distances
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    N, K = knn_indices.shape

    # ------------------------------------------------------------
    # Construct directed KNN edges
    #
    # src = [0,0,...,1,1,...]
    # dst = corresponding KNN indices
    # ------------------------------------------------------------

    src = np.repeat(
        np.arange(N, dtype=np.int64),
        K
    )

    dst = knn_indices.reshape(-1)

    distances = knn_distances.reshape(-1)

    # ------------------------------------------------------------
    # Symmetrize
    #
    # Add:
    #
    # i -> j
    #
    # and:
    #
    # j -> i
    # ------------------------------------------------------------

    if symmetric:

        src = np.concatenate(
            [src, dst]
        )

        dst = np.concatenate(
            [dst, src[:N * K]]
        )

        distances = np.concatenate(
            [distances, distances.copy()]
        )

    # ------------------------------------------------------------
    # Remove duplicate directed edges
    #
    # We keep the minimum distance when both
    # directions / KNN lists produce the same edge.
    # ------------------------------------------------------------

    edge_keys = (
        src.astype(np.int64) * N
        + dst.astype(np.int64)
    )

    order = np.argsort(
        edge_keys,
        kind="stable"
    )

    edge_keys_sorted = edge_keys[order]

    src_sorted = src[order]
    dst_sorted = dst[order]
    distances_sorted = distances[order]

    # First occurrence of every unique (src, dst)
    unique_mask = np.empty(
        edge_keys_sorted.shape,
        dtype=bool
    )

    unique_mask[0] = True

    unique_mask[1:] = (
        edge_keys_sorted[1:]
        != edge_keys_sorted[:-1]
    )

    # Since duplicate edges are adjacent after sorting,
    # we need the minimum distance for each group.
    #
    # Find group boundaries.
    group_starts = np.flatnonzero(
        unique_mask
    )

    group_ends = np.concatenate(
        [
            group_starts[1:],
            np.array(
                [len(distances_sorted)],
                dtype=np.int64
            )
        ]
    )

    min_distances = np.minimum.reduceat(
        distances_sorted,
        group_starts
    )

    src_unique = src_sorted[
        group_starts
    ]

    dst_unique = dst_sorted[
        group_starts
    ]

    # ------------------------------------------------------------
    # Sort edges by source vertex
    #
    # Forward-star requires all outgoing edges from
    # vertex i to be contiguous.
    # ------------------------------------------------------------

    source_order = np.argsort(
        src_unique,
        kind="stable"
    )

    src_unique = src_unique[
        source_order
    ]

    dst_unique = dst_unique[
        source_order
    ]

    edge_distances = min_distances[
        source_order
    ].astype(
        np.float32,
        copy=False
    )

    # ------------------------------------------------------------
    # Forward-star offsets
    # ------------------------------------------------------------

    edge_count = len(src_unique)

    first_edge = np.zeros(
        N + 1,
        dtype=np.uint32
    )

    counts = np.bincount(
        src_unique,
        minlength=N
    )

    first_edge[1:] = np.cumsum(
        counts,
        dtype=np.uint64
    ).astype(
        np.uint32
    )

    # ------------------------------------------------------------
    # Adjacency
    # ------------------------------------------------------------

    adj_vertices = dst_unique.astype(
        np.uint32,
        copy=False
    )

    # ------------------------------------------------------------
    # Edge weights
    # ------------------------------------------------------------

    if weight_mode == "uniform":

        edge_weights = np.ones(
            edge_count,
            dtype=np.float32
        )

    elif weight_mode == "inverse_distance":

        edge_weights = (
            1.0
            / (edge_distances + eps)
        ).astype(
            np.float32
        )

    elif weight_mode == "gaussian":

        raise NotImplementedError(
            "Gaussian edge weights require a sigma parameter."
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