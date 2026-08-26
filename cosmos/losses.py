import torch
import torch.nn.functional as F

def compute_D_avg(
    xyz,
    first_edge,
    adj_vertices,
    eps=1e-8
):
    N = xyz.shape[0]

    total = torch.zeros(
        (),
        device=xyz.device,
        dtype=xyz.dtype
    )

    for i in range(N):

        start = int(first_edge[i])
        end = int(first_edge[i + 1])

        neighbors = adj_vertices[start:end]

        if len(neighbors) == 0:
            continue

        diff = xyz[i] - xyz[neighbors]

        # Eq. (10): ||xi - xj||_2, NOT squared
        distances = torch.linalg.norm(
            diff,
            dim=1
        )

        total += distances.sum() / (
            len(neighbors) + eps
        )

    return total / N

def compute_D_ctr(
    xyz,
    supergaussian_ids,
    eps=1e-8
):

    N = xyz.shape[0]

    unique_groups = torch.unique(
        supergaussian_ids
    )

    total = torch.zeros(
        (),
        device=xyz.device,
        dtype=xyz.dtype
    )

    for g in unique_groups:

        mask = supergaussian_ids == g

        group_xyz = xyz[mask]

        center = group_xyz.sum(dim=0) / (
            group_xyz.shape[0] + eps
        )

        diff = group_xyz - center

        squared_dist = torch.sum(
            diff * diff,
            dim=1
        )

        total += squared_dist.sum()

    return total / N