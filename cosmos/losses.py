import torch
import torch.nn.functional as F

def compute_D_avg(
    xyz,
    first_edge,
    adj_vertices,
    eps=1e-8
):
    """
    Vectorized computation of average edge distance.

    Eq. (10):
        D_avg = (1/N) sum_i [
            (1/|N_i|) sum_{j in N_i} ||x_i - x_j||
        ]
    """

    N = xyz.shape[0]

    # ------------------------------------------------------------
    # Number of neighbors for every vertex
    # ------------------------------------------------------------

    degrees = first_edge[1:] - first_edge[:-1]

    # ------------------------------------------------------------
    # Build source vertex index for every edge
    #
    # Example:
    # degrees = [2, 3, 1]
    #
    # source =
    # [0, 0, 1, 1, 1, 2]
    # ------------------------------------------------------------

    source = torch.repeat_interleave(
        torch.arange(
            N,
            device=xyz.device,
            dtype=torch.long
        ),
        degrees
    )

    # ------------------------------------------------------------
    # Gather edge endpoints
    # ------------------------------------------------------------

    neighbors = adj_vertices

    # ------------------------------------------------------------
    # Compute all edge distances simultaneously
    # ------------------------------------------------------------

    diff = xyz[source] - xyz[neighbors]

    distances = torch.linalg.norm(
        diff,
        dim=1
    )

    # ------------------------------------------------------------
    # Sum distances belonging to each source vertex
    # ------------------------------------------------------------

    distance_sum = torch.zeros(
        N,
        device=xyz.device,
        dtype=xyz.dtype
    )

    distance_sum.scatter_add_(
        0,
        source,
        distances
    )

    # ------------------------------------------------------------
    # Average over neighbors
    # ------------------------------------------------------------

    avg_distance = distance_sum / (
        degrees.to(xyz.dtype) + eps
    )

    # ------------------------------------------------------------
    # Average over all Gaussians
    # ------------------------------------------------------------

    return avg_distance.mean()


def compute_D_ctr(
    xyz,
    supergaussian_ids,
    eps=1e-8
):
    """
    Vectorized computation of intra-SuperGaussian
    positional compactness.

        D_ctr = (1/N) sum_g sum_{i in g}
                ||x_i - c_g||^2

    where c_g is the centroid of SuperGaussian g.
    """

    N = xyz.shape[0]

    # ------------------------------------------------------------
    # Number of groups
    # ------------------------------------------------------------

    num_groups = (
        torch.max(supergaussian_ids) + 1
    )

    # ------------------------------------------------------------
    # Compute group sizes
    # ------------------------------------------------------------

    group_counts = torch.bincount(
        supergaussian_ids,
        minlength=num_groups
    )

    # ------------------------------------------------------------
    # Compute group coordinate sums
    #
    # [N, 3] -> [G, 3]
    # ------------------------------------------------------------

    group_sums = torch.zeros(
        num_groups,
        3,
        device=xyz.device,
        dtype=xyz.dtype
    )

    group_sums.scatter_add_(
        0,
        supergaussian_ids.unsqueeze(1).expand(-1, 3),
        xyz
    )

    # ------------------------------------------------------------
    # Group centroids
    # ------------------------------------------------------------

    group_centers = group_sums / (
        group_counts.to(xyz.dtype).unsqueeze(1) + eps
    )

    # ------------------------------------------------------------
    # Difference from corresponding group centroid
    # ------------------------------------------------------------

    diff = (
        xyz
        - group_centers[supergaussian_ids]
    )

    # ------------------------------------------------------------
    # Squared distance to centroid
    # ------------------------------------------------------------

    squared_dist = torch.sum(
        diff * diff,
        dim=1
    )

    # ------------------------------------------------------------
    # Eq. normalization
    # ------------------------------------------------------------

    return squared_dist.sum() / N