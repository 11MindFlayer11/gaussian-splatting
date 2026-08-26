import torch


@torch.no_grad()
def knn_indices(points, k=16, chunk_size=1024):
    """
    Compute K nearest neighbors for a point cloud.

    Args:
        points: [N, 3]
        k: number of neighbors
        chunk_size: number of query points processed at once

    Returns:
        indices: [N, k]
        distances: [N, k]
    """

    N = points.shape[0]

    if k >= N:
        raise ValueError(
            f"k={k} must be smaller than number of points N={N}"
        )

    all_indices = []
    all_distances = []

    for start in range(0, N, chunk_size):

        end = min(start + chunk_size, N)

        query = points[start:end]

        # [chunk, N]
        distances = torch.cdist(query, points)

        # Remove self-neighbor
        local_rows = torch.arange(
            end - start,
            device=points.device
        )

        global_rows = torch.arange(
            start,
            end,
            device=points.device
        )

        distances[local_rows, global_rows] = float("inf")

        knn_distances, knn_idx = torch.topk(
            distances,
            k=k,
            dim=1,
            largest=False,
            sorted=True
        )

        all_indices.append(knn_idx)
        all_distances.append(knn_distances)

    indices = torch.cat(all_indices, dim=0)
    distances = torch.cat(all_distances, dim=0)

    return indices, distances