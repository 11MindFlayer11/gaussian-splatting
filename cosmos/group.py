import numpy as np
import torch

from cosmos.graph import knn_to_forward_star
from pycut_pursuit.cp_d0_dist import cp_d0_dist

from utils.sh_utils import SH2RGB
from cosmos.knn import knn_indices
from cosmos.descriptors import compute_geometric_descriptors


@torch.no_grad()
def compute_group_statistics(
    xyz,
    supergaussian_ids,
    eps=1e-8
):
    """
    Vectorized computation of SuperGaussian statistics.

    Returns:
        group_centers:      [G, 3]
        group_covariance:   [G, 3, 3]
        group_sizes:        [G]
        group_descriptors:  [G, 4]
    """

    # --------------------------------------------------------
    # Unique groups
    # --------------------------------------------------------

    unique_groups, inverse, group_sizes = torch.unique(
        supergaussian_ids,
        sorted=True,
        return_inverse=True,
        return_counts=True
    )

    G = unique_groups.shape[0]
    N = xyz.shape[0]

    group_sizes = group_sizes.to(
        device=xyz.device,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Group centers
    #
    # scatter_add:
    #   sum of xyz belonging to each group
    # --------------------------------------------------------

    group_sums = torch.zeros(
        G,
        3,
        device=xyz.device,
        dtype=xyz.dtype
    )

    group_sums.scatter_add_(
        0,
        inverse.unsqueeze(1).expand(-1, 3),
        xyz
    )

    group_centers = (
        group_sums /
        group_sizes.unsqueeze(1).to(xyz.dtype)
    )

    # --------------------------------------------------------
    # Center every Gaussian
    # --------------------------------------------------------

    centered = (
        xyz -
        group_centers[inverse]
    )

    # --------------------------------------------------------
    # Group covariance
    #
    # covariance[g] =
    #   1 / |g| sum_i (x_i-c_g)(x_i-c_g)^T
    # --------------------------------------------------------

    outer_products = (
        centered.unsqueeze(2) *
        centered.unsqueeze(1)
    )

    group_covariance = torch.zeros(
        G,
        3,
        3,
        device=xyz.device,
        dtype=xyz.dtype
    )

    group_covariance.scatter_add_(
        0,
        inverse[:, None, None].expand(-1, 3, 3),
        outer_products
    )

    group_covariance = (
        group_covariance /
        group_sizes.to(xyz.dtype)[:, None, None]
    )

    # --------------------------------------------------------
    # Eigen decomposition
    # --------------------------------------------------------

    eigenvalues, eigenvectors = torch.linalg.eigh(
        group_covariance
    )

    # eigh returns ascending eigenvalues:
    #
    # lambda3 <= lambda2 <= lambda1
    #
    # Reverse to:
    #
    # lambda1 >= lambda2 >= lambda3
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Geometric descriptors
    # --------------------------------------------------------

    linearity = (
        (lambda1 - lambda2) / lambda1
    ).clamp(0.0, 1.0)

    planarity = (
        (lambda2 - lambda3) / lambda1
    ).clamp(0.0, 1.0)

    scattering = (
        lambda3 / lambda1
    ).clamp(0.0, 1.0)

    # Smallest eigenvalue eigenvector = normal
    normal = eigenvectors[:, :, 2]

    verticality = (
        1.0 - normal[:, 2].abs()
    ).clamp(0.0, 1.0)

    group_descriptors = torch.stack(
        [
            linearity,
            planarity,
            scattering,
            verticality
        ],
        dim=1
    )

    return (
        group_centers,
        group_covariance,
        group_sizes,
        group_descriptors
    )

def supergaussian_grouping(
    xyz,
    color,
    scale,
    descriptors,
    knn_indices,
    knn_distances,
    *,
    symmetric=True,
    weight_mode="uniform",
    cp_dif_tol=1e-3,
    cp_it_max=10,
    K=2,
    split_iter_num=2,
    split_damp_ratio=1.0,
    kmpp_init_num=3,
    kmpp_iter_num=3,
    min_comp_weight=0.0,
    verbose=True
):
    """
    COSMOS Section 3.2:

        Gaussian attributes
            +
        local geometric descriptors
            ↓
        feature vectors
            ↓
        KNN graph
            ↓
        L0-cut pursuit
            ↓
        SuperGaussian groups

    Args:
        xyz:
            [N, 3]

        color:
            [N, 3]

        scale:
            [N, 3]

        descriptors:
            [N, 4]
            [linearity, planarity, scattering, verticality]

        knn_indices:
            [N, K]

        knn_distances:
            [N, K]

    Returns:
        dictionary containing:

            supergaussian_ids:
                [N]

            group_centers:
                [G, 3]

            group_covariance:
                [G, 3, 3]

            group_sizes:
                [G]

            group_descriptors:
                [G, 4]

            features:
                [N, 13]

            cut_pursuit_result:
                raw Cut Pursuit result
    """

    # ========================================================
    # 1. Build COSMOS feature vector
    #
    # f_i = [x_i, c_i, sigma_i, l_i, s_i, v_i, p_i]
    #
    # In our implementation:
    #
    # xyz          -> 3
    # color        -> 3
    # scale        -> 3
    # descriptors  -> 4
    #
    # total        -> 13
    # ========================================================

    features = torch.cat(
        [
            xyz,
            color,
            scale,
            descriptors
        ],
        dim=1
    )

    # ========================================================
    # 2. Convert KNN graph to Cut Pursuit graph
    # ========================================================

    first_edge, adj_vertices, edge_weights = (
        knn_to_forward_star(
            knn_indices,
            knn_distances,
            symmetric=symmetric,
            weight_mode=weight_mode
        )
    )

    # ========================================================
    # 3. Convert features to Cut Pursuit format
    #
    # Cut Pursuit expects:
    #
    # Y = [feature_dimension, number_of_points]
    #
    # and Fortran-contiguous memory.
    # ========================================================

    Y = np.asfortranarray(
        features.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .T
    )

    # ========================================================
    # 4. L0-cut pursuit
    # ========================================================

    result = cp_d0_dist(
        loss=13,
        Y=Y,

        first_edge=first_edge,
        adj_vertices=adj_vertices,
        edge_weights=edge_weights,

        cp_dif_tol=cp_dif_tol,
        cp_it_max=cp_it_max,

        K=K,
        split_iter_num=split_iter_num,
        split_damp_ratio=split_damp_ratio,

        kmpp_init_num=kmpp_init_num,
        kmpp_iter_num=kmpp_iter_num,

        min_comp_weight=min_comp_weight,

        verbose=verbose,
        max_num_threads=0,

        compute_List=True,
        compute_Graph=True,
        compute_Obj=True,
        compute_Time=True,
        compute_Dif=True
    )

    # ========================================================
    # 5. Extract component assignments
    # ========================================================

    (
        component_ids,
        reduced_solution,
        List,
        Graph,
        Obj,
        Time,
        Dif
    ) = result

    supergaussian_ids = torch.as_tensor(
        component_ids,
        device=xyz.device,
        dtype=torch.long
    )

    # ========================================================
    # 6. Compute group-level statistics
    # ========================================================

    (
        group_centers,
        group_covariance,
        group_sizes,
        group_descriptors
    ) = compute_group_statistics(
        xyz,
        supergaussian_ids
    )

    return {
        "supergaussian_ids": supergaussian_ids,

        "group_centers": group_centers,

        "group_covariance": group_covariance,

        "group_sizes": group_sizes,

        "group_descriptors": group_descriptors,

        "features": features,

        "cut_pursuit_result": result,

        "component_ids": component_ids,

        "reduced_solution": reduced_solution,

        "List": List,

        "Graph": Graph,

        "Obj": Obj,

        "Time": Time,

        "Dif": Dif,
        # --------------------------------------------------------
        # COSMOS positional regularization graph
        # --------------------------------------------------------

        "first_edge": first_edge,
        "adj_vertices": adj_vertices,
        "edge_weights": edge_weights,
    }

def build_supergaussians(
    gaussians,
    k=16,
    chunk_size=1024,
):
    """
    Complete COSMOS SuperGaussian grouping pipeline.

    Converts a 3DGS GaussianModel into COSMOS grouping features,
    computes local geometric descriptors, and performs cut-pursuit
    grouping.

    Returns:
        groups: dictionary containing SuperGaussian IDs,
                group statistics, and cut-pursuit results.
    """

    # ============================================================
    # 1. Extract Gaussian attributes
    # ============================================================

    xyz = gaussians.get_xyz

    sh_dc = gaussians.get_features[:, 0, :]
    color = SH2RGB(sh_dc)

    scale = gaussians.get_scaling

    # ============================================================
    # 2. KNN
    # ============================================================

    knn_idx, knn_dist = knn_indices(
        xyz,
        k=k,
        chunk_size=chunk_size,
    )

    # ============================================================
    # 3. Local geometric descriptors
    # ============================================================

    descriptors, eigenvalues, eigenvectors = \
        compute_geometric_descriptors(
            xyz,
            knn_idx,
        )

    # ============================================================
    # 4. SuperGaussian grouping
    # ============================================================

    groups = supergaussian_grouping(
        xyz=xyz,
        color=color,
        scale=scale,
        descriptors=descriptors,
        knn_indices=knn_idx,
        knn_distances=knn_dist,
    )

    # Keep these available for validation/debugging.
    groups["knn_indices"] = knn_idx
    groups["knn_distances"] = knn_dist
    groups["eigenvalues"] = eigenvalues
    groups["eigenvectors"] = eigenvectors

    return groups