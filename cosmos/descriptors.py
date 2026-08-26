import torch


@torch.no_grad()
def compute_geometric_descriptors(
    xyz,
    knn_indices,
    eps=1e-8
):
    """
    Compute local geometric descriptors for every point.

    Args:
        xyz:
            [N, 3] point coordinates

        knn_indices:
            [N, K] neighboring point indices

    Returns:
        descriptors:
            [N, 4]

            columns:
                0 -> linearity
                1 -> scattering
                2 -> verticality
                3 -> planarity

        eigenvalues:
            [N, 3], sorted descending

        eigenvectors:
            [N, 3, 3]
    """

    # [N, K, 3]
    neighbors = xyz[knn_indices]

    # Local centroid
    mean = neighbors.mean(dim=1, keepdim=True)

    # Center neighborhoods
    centered = neighbors - mean

    # Covariance:
    #
    # [N, K, 3] -> [N, 3, 3]
    covariance = (
        centered.transpose(1, 2) @ centered
    ) / max(knn_indices.shape[1], 1)

    # Symmetric eigendecomposition.
    #
    # torch.linalg.eigh returns ascending eigenvalues.
    eigenvalues, eigenvectors = torch.linalg.eigh(
        covariance
    )

    # Reverse to:
    # lambda1 >= lambda2 >= lambda3
    eigenvalues = torch.flip(
        eigenvalues,
        dims=[1]
    )

    eigenvectors = torch.flip(
        eigenvectors,
        dims=[2]
    )

    lambda1 = eigenvalues[:, 0].clamp_min(eps)
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]

    linearity = (
        (lambda1 - lambda2) / lambda1
    ).clamp(0.0, 1.0)

    planarity = (
        (lambda2 - lambda3) / lambda1
    ).clamp(0.0, 1.0)

    scattering = (
        lambda3 / lambda1
    ).clamp(0.0, 1.0)

    # Eigenvector corresponding to smallest
    # eigenvalue = local surface-normal direction.
    #
    # After reversing, this is column 2.
    normal = eigenvectors[:, :, 2]

    # Vertical direction = +Z.
    verticality = (
        1.0 - normal[:, 2].abs()
    ).clamp(0.0, 1.0)

    descriptors = torch.stack(
        [
            linearity,
            scattering,
            verticality,
            planarity
            
        ],
        dim=1
    )

    return (
        descriptors,
        eigenvalues,
        eigenvectors
    )