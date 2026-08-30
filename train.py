#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from cosmos.group import build_supergaussians
from utils.sh_utils import SH2RGB
import torch.nn.functional as F

from cosmos.attention import (
    PositionalEncoding3D,
    SuperGaussianSelfAttention,
    SparseLocalAttention,
)
from cosmos.losses import (
    compute_D_avg,
    compute_D_ctr,
)

from cosmos.heads import ResidualMLP
from cosmos.knn import knn_indices
from cosmos.graph import knn_to_forward_star
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def rebuild_cosmos_local_graph(
    gaussians,
    k=16,
    chunk_size=1024,
):
    """
    Rebuild the local Gaussian KNN graph after
    densification/pruning changes the Gaussian count.

    Returns:
        knn_indices
        first_edge
        adj_vertices
        edge_weights
    """

    xyz = gaussians.get_xyz

    # --------------------------------------------------------
    # GPU KNN
    # --------------------------------------------------------

    knn_idx, knn_dist = knn_indices(
        xyz,
        k=k,
        chunk_size=chunk_size
    )

    # --------------------------------------------------------
    # Cut-Pursuit / loss graph representation
    # --------------------------------------------------------

    first_edge, adj_vertices, edge_weights = (
        knn_to_forward_star(
            knn_idx,
            knn_dist,
            symmetric=True,
            weight_mode="uniform"
        )
    )

    return (
        knn_idx,
        first_edge,
        adj_vertices,
        edge_weights
    )

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    torch.backends.cudnn.benchmark = True

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    # ============================================================
    # COSMOS — SUPERGAUSSIAN GROUPING
    # ============================================================

    # print("=" * 70)
    # print("COSMOS SUPERGAUSSIAN GROUPING")
    # print("=" * 70)

    # groups = build_supergaussians(gaussians)

    # gaussians.supergaussian_ids = groups["supergaussian_ids"]

    # print("Gaussians      :", gaussians.get_xyz.shape[0])
    # print(
    #     "SuperGaussians :",
    #     torch.unique(gaussians.supergaussian_ids).numel()
    # )
    # print(
    #     "Feature dim    :",
    #     groups["features"].shape[1]
    # )

    # assert gaussians.supergaussian_ids.shape[0] == \
    #     gaussians.get_xyz.shape[0]

    # print("=" * 70)


    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

        # ============================================================
    # COSMOS — ITERATION 100 INITIALIZATION
    # ============================================================


    global_attention = None
    local_attention = None

    cosmos_first_edge = None
    cosmos_adj_vertices = None
    cosmos_edge_weights = None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # ============================================================
    # COSMOS STATE
    # ============================================================

    cosmos_initialized = False

    groups = None

    global_model = None
    local_model = None

    position_head = None
    orientation_head = None
    scale_head = None
    color_head = None
    opacity_head = None
    cosmos_optimizer = None

    neighbor_indices = None
    group_centers = None
    local_pos_encoder = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
                # ========================================================
        # COSMOS — INITIALIZE AT ITERATION 100
        # ========================================================

        if iteration == 100 and not cosmos_initialized:

            print("=" * 70)
            print("COSMOS INITIALIZATION — ITERATION 100")
            print("=" * 70)

            # ----------------------------------------------------
            # Build SuperGaussian grouping
            # ----------------------------------------------------

            with torch.no_grad():
                groups = build_supergaussians(gaussians)

            print("\n========== COSMOS GROUP AUTOGRAD ==========")

            for k, v in groups.items():
                if torch.is_tensor(v):
                    print(
                        f"{k:20s} "
                        f"dtype={v.dtype} "
                        f"requires_grad={v.requires_grad} "
                        f"grad_fn={v.grad_fn}"
                    )

            print("===========================================\n")

            # COSMOS topology is fixed after initialization.
            groups["features"] = groups["features"].detach()

            gaussians.supergaussian_ids = (
                groups["supergaussian_ids"]
            ).detach()

            gaussians.cosmos_descriptors = (
                groups["features"][:, 9:13]
            ).detach().clone()

            cosmos_first_edge = torch.as_tensor(
                groups["first_edge"], device="cuda", dtype=torch.long
            )
            
            cosmos_adj_vertices = torch.as_tensor(
                groups["adj_vertices"], device="cuda", dtype=torch.long
            )
            
            cosmos_edge_weights = torch.as_tensor(
                groups["edge_weights"], device="cuda", dtype=torch.float32
            ).detach()
            
            cosmos_knn_indices = torch.as_tensor(
                groups["knn_indices"], device="cuda", dtype=torch.long
            )

            # group_centers = groups["group_centers"].detach()

            # ----------------------------------------------------
            # Feature dimensions
            # ----------------------------------------------------

            remaining_dim = (
                groups["features"].shape[1] - 3
            )

            pos_dim = 3 + 2 * 6 * 3
            # xyz + sin/cos positional encoding
            # with 6 frequencies

            # ----------------------------------------------------
            # Global SuperGaussian attention
            # ----------------------------------------------------

            global_attention = SuperGaussianSelfAttention(
                remaining_dim=remaining_dim,
                pos_dim=pos_dim,
                d_model=128,
                num_heads=4,
                num_freqs=6
            ).cuda()

            # ----------------------------------------------------
            # Local Gaussian attention
            # ----------------------------------------------------

            local_attention = SparseLocalAttention(
                input_dim=49,
                embed_dim=128,
                num_heads=4
            ).cuda()

            local_pos_encoder = PositionalEncoding3D(
                num_freqs=6
            ).cuda()
            cosmos_initialized = True

            print(
                "Gaussians:",
                gaussians.get_xyz.shape[0]
            )

            print(
                "SuperGaussians:",
                torch.unique(
                    gaussians.supergaussian_ids
                ).numel()
            )

            print(
                "Global attention initialized"
            )

            print(
                "Local attention initialized"
            )

            print("=" * 70)

        

           

            # ----------------------------------------------------
            # COSMOS residual prediction heads initialization
            # ----------------------------------------------------

            unified_dim = 128 + 128  # global + local

            position_head = ResidualMLP(
                input_dim=unified_dim,
                output_dim=3
            ).cuda()

            orientation_head = ResidualMLP(
                input_dim=unified_dim,
                output_dim=4
            ).cuda()

            scale_head = ResidualMLP(
                input_dim=unified_dim,
                output_dim=3
            ).cuda()

            color_head = ResidualMLP(
                input_dim=unified_dim,
                output_dim=3
            ).cuda()

            opacity_head = ResidualMLP(
                input_dim=unified_dim,
                output_dim=1
            ).cuda()

            # ----------------------------------------------------
            # COSMOS optimizer initialization
            # ----------------------------------------------------

            cosmos_optimizer = torch.optim.Adam(
                [
                    {"params": position_head.parameters(), "lr": 1e-5},
                    {"params": (
                        list(global_attention.parameters())
                        + list(local_attention.parameters())
                        + list(orientation_head.parameters())
                        + list(scale_head.parameters())
                        + list(color_head.parameters())
                        + list(opacity_head.parameters())
                    ), "lr": 1e-4},
                ]
            )

            print("Residual heads initialized.")
            print("Unified feature dimension:", unified_dim)

        if cosmos_initialized:

             # ----------------------------------------------------
            # Build COSMOS features for current Gaussian state
            # ----------------------------------------------------
            xyz = gaussians.get_xyz

            sh_dc = gaussians.get_features[:, 0, :]
            color = SH2RGB(sh_dc)

            scale = gaussians.get_scaling

            descriptors = gaussians.cosmos_descriptors
            assert xyz.shape[0] == gaussians.supergaussian_ids.shape[0]
            assert xyz.shape[0] == gaussians.cosmos_descriptors.shape[0]
            gaussian_features = torch.cat(
                [
                    xyz,
                    color,
                    scale,
                    descriptors
                ],
                dim=1
            )

            # ----------------------------------------------------
            # Group-wise max pooling
            # ----------------------------------------------------

            supergaussian_ids = (
                gaussians.supergaussian_ids
            )

            unique_ids, inverse_ids = torch.unique(
                supergaussian_ids,
                sorted=True,
                return_inverse=True
            )

            # IMPORTANT:
            # Compute the number of currently surviving SuperGaussians
            # before allocating any group-level tensors.
            num_groups = unique_ids.shape[0]

            group_centers = torch.zeros(
                (num_groups, 3),
                device=xyz.device,
                dtype=xyz.dtype
            )

            group_centers.scatter_add_(
                0,
                inverse_ids.unsqueeze(1).expand(-1, 3),
                xyz
            )

            group_counts = torch.bincount(
                inverse_ids,
                minlength=num_groups
            ).to(xyz.dtype).unsqueeze(1)

            group_centers = group_centers / group_counts.clamp_min(1.0)

            if iteration in testing_iterations:
                print(
                    f"[COSMOS] "
                    f"Gaussians={xyz.shape[0]} "
                    f"SuperGaussians={num_groups}"
                )

            
            feature_dim = gaussian_features.shape[1]
            group_features = torch.full(
                (num_groups, feature_dim),
                -torch.inf,
                device=gaussian_features.device,
                dtype=gaussian_features.dtype
            )

            group_features.scatter_reduce_(
                0,
                inverse_ids.unsqueeze(1).expand(-1, feature_dim),
                gaussian_features,
                reduce="amax",
                include_self=True
            )

            # ----------------------------------------------------
            # Separate position from remaining attributes
            # ----------------------------------------------------

            group_remaining = group_features[:, 3:]

            # ----------------------------------------------------
            # Global SuperGaussian attention
            # ----------------------------------------------------

            global_features, global_attention_weights = (
                global_attention(
                    group_centers,
                    group_remaining
                )
            )
          
            # ----------------------------------------------------
            # Broadcast global features to Gaussians
            # ----------------------------------------------------

            global_gaussian_features = (
                global_features[
                    inverse_ids
                ]
            )

            # ----------------------------------------------------
            # Local Gaussian features
            # ----------------------------------------------------

            local_pos = local_pos_encoder(xyz)

            local_features = torch.cat(
                [
                    local_pos,
                    gaussian_features[:, 3:]
                ],
                dim=1
            )

            # ----------------------------------------------------
            # Local sparse attention
            # ----------------------------------------------------

            local_features_out, local_attention_weights = (
                local_attention(
                    local_features,
                    cosmos_knn_indices
                )
            )

            # ----------------------------------------------------
            # Unified COSMOS representation
            # ----------------------------------------------------

            unified_features = torch.cat(
                [
                    global_gaussian_features,
                    local_features_out
                ],
                dim=1
            )

            # ============================================================
            # COSMOS — PREDICT GAUSSIAN RESIDUALS
            # ============================================================

            delta_position = position_head(
                unified_features
            )

            delta_rotation = orientation_head(
                unified_features
            )

            delta_scale = scale_head(
                unified_features
            )

            delta_color = color_head(
                unified_features
            )

            delta_opacity = opacity_head(
                unified_features
            )

            # print("Residual shapes:")
            # print("  Δposition :", delta_position.shape)
            # print("  Δrotation :", delta_rotation.shape)
            # print("  Δscale    :", delta_scale.shape)
            # print("  Δcolor    :", delta_color.shape)
            # print("  Δopacity  :", delta_opacity.shape)

            # print(
            #     "Global features:",
            #     global_gaussian_features.shape
            # )

            # print(
            #     "Local features:",
            #     local_features_out.shape
            # )

            # print(
            #     "Unified features:",
            #     unified_features.shape
            # )

            # ----------------------------------------------------
            # COSMOS refined Gaussian attributes
            # ----------------------------------------------------
            max_disp = 2.0 * gaussians.get_scaling.max(dim=1, keepdim=True).values
            disp_norm = delta_position.norm(dim=1, keepdim=True)
            scale_factor = (max_disp / disp_norm.clamp_min(1e-8)).clamp(max=1.0)
            delta_position = delta_position * scale_factor

            warmup_iters = 500
            ramp = min(1.0, max(0.0, (iteration - 100) / warmup_iters))
            delta_position = ramp * delta_position

            refined_position = (
                gaussians.get_xyz + delta_position
            )

            # Rotation: residual quaternion update
            refined_rotation = (
                gaussians.get_rotation + delta_rotation
            )

            refined_rotation = F.normalize(
                refined_rotation,
                dim=1
            )

            # Scale: activated scale space
            refined_scale = (
                gaussians.get_scaling + delta_scale
            )

            refined_scale = refined_scale.clamp_min(1e-8)

            # Color: RGB space
            refined_color = (
                SH2RGB(gaussians.get_features[:, 0, :])
                + delta_color
            )

            refined_color = refined_color.clamp(0.0, 1.0)

            # Opacity: activated opacity space
            refined_opacity = (
                gaussians.get_opacity + delta_opacity
            )

            refined_opacity = refined_opacity.clamp(
                1e-6,
                1.0 - 1e-6
            )

            # ----------------------------------------------------
            # Keep the model's notion of "current opacity" in sync
            # with what's actually rendered, so opacity-based pruning
            # (densify_and_prune) judges Gaussians by their real
            # contribution instead of the decoupled base parameter.
            # ----------------------------------------------------
            # gaussians.cosmos_effective_opacity = refined_opacity.detach()

            # print("=" * 70)
            # print("COSMOS REFINED ATTRIBUTES")
            # print("=" * 70)

            # print("Position :", refined_position.shape)
            # print("Rotation :", refined_rotation.shape)
            # print("Scale    :", refined_scale.shape)
            # print("Color    :", refined_color.shape)
            # print("Opacity  :", refined_opacity.shape)

           

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE,
                                override_xyz=refined_position if cosmos_initialized else None,
                                override_opacity=refined_opacity if cosmos_initialized else None,
                                override_scale=refined_scale if cosmos_initialized else None,
                                override_rotation=refined_rotation if cosmos_initialized else None,
                                override_color=refined_color if cosmos_initialized else None
                                )["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, override_xyz=refined_position if cosmos_initialized else None,
    override_opacity=refined_opacity if cosmos_initialized else None,
    override_scale=refined_scale if cosmos_initialized else None,
    override_rotation=refined_rotation if cosmos_initialized else None ,
    override_color = refined_color if cosmos_initialized else None)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
         # ============================================================
        # COSMOS — POSITION REGULARIZATION
        # ============================================================

        if cosmos_initialized:

            D_avg = compute_D_avg(
                refined_position,
                cosmos_first_edge,
                cosmos_adj_vertices
            )

            D_ctr = compute_D_ctr(
                refined_position,
                gaussians.supergaussian_ids
            )
            D_anchor = delta_position.pow(2).sum(dim=1).mean()
            L_pos = D_avg + D_ctr + 10.0 * D_anchor

            D_opacity_anchor = delta_opacity.pow(2).mean()
            loss = loss + 0.1 * L_pos + 1.0  * D_opacity_anchor 

           

        # ============================================================
        # BACKWARD
        # ============================================================

        loss.backward()
        # if cosmos_initialized and iteration == 100:
        #     print("=" * 70)
        #     print("COSMOS GRADIENT CHECK")
        #     print("=" * 70)

        #     print(
        #         "Global attention grad:",
        #         global_attention.input_proj.weight.grad is not None
        #     )

        #     print(
        #         "Local attention grad:",
        #         local_attention.input_proj.weight.grad is not None
        #     )

        #     print(
        #         "Position head grad:",
        #         position_head.fc1.weight.grad is not None
        #     )

        #     print("=" * 70)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # ===========================================================
            # Densification / Pruning
            # ============================================================

            if iteration < opt.densify_until_iter:

                # --------------------------------------------------------
                # Track max image-space radii
                # --------------------------------------------------------

                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )

                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter
                )

                # --------------------------------------------------------
                # Densify / prune
                # --------------------------------------------------------

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):

                    size_threshold = (
                        20
                        if iteration > opt.opacity_reset_interval
                        else 40
                    )

                    old_num_gaussians = gaussians.get_xyz.shape[0]

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        max_num_gaussians=350000
                    )
                    if cosmos_initialized:

                        print(
                            "[COSMOS DEBUG]",
                            "xyz =", gaussians.get_xyz.shape,
                            "ids =", gaussians.supergaussian_ids.shape,
                            "desc =", gaussians.cosmos_descriptors.shape,
                            "knn =", cosmos_knn_indices.shape
                        )
                    new_num_gaussians = gaussians.get_xyz.shape[0]

                    # ====================================================
                    # COSMOS LOCAL GRAPH UPDATE
                    # ====================================================

                    if cosmos_initialized:

                        if new_num_gaussians != old_num_gaussians:

                            print(
                                f"\n[COSMOS] Gaussian count changed: "
                                f"{old_num_gaussians} -> {new_num_gaussians}"
                            )

                            (
                                cosmos_knn_indices,
                                new_first_edge,
                                new_adj_vertices,
                                new_edge_weights
                            ) = rebuild_cosmos_local_graph(
                                gaussians,
                                k=16,
                                chunk_size=1024
                            )

                            # --------------------------------------------------------
                            # Update local graph used by COSMOS losses
                            # --------------------------------------------------------

                            cosmos_first_edge = torch.as_tensor(
                                new_first_edge,
                                device=gaussians.get_xyz.device,
                                dtype=torch.long
                            )

                            cosmos_adj_vertices = torch.as_tensor(
                                new_adj_vertices,
                                device=gaussians.get_xyz.device,
                                dtype=torch.long
                            )

                            cosmos_edge_weights = torch.as_tensor(
                                new_edge_weights,
                                device=gaussians.get_xyz.device,
                                dtype=torch.float32
                            )

                            print(
                                "[COSMOS] Local KNN graph rebuilt:",
                                cosmos_knn_indices.shape
                            )

                            print(
                                "[COSMOS] Forward-star graph rebuilt:",
                                "vertices =", cosmos_adj_vertices.shape,
                                "edges =", cosmos_edge_weights.shape
                            )
                # --------------------------------------------------------
                # Opacity reset
                # --------------------------------------------------------
                            # ----------------------------------------------------
                            # COSMOS topology consistency checks
                            # ----------------------------------------------------

                            N = gaussians.get_xyz.shape[0]

                            assert gaussians.supergaussian_ids.shape[0] == N, (
                                f"SuperGaussian ID mismatch: "
                                f"{gaussians.supergaussian_ids.shape[0]} vs {N}"
                            )

                            assert gaussians.cosmos_descriptors.shape[0] == N, (
                                f"COSMOS descriptor mismatch: "
                                f"{gaussians.cosmos_descriptors.shape[0]} vs {N}"
                            )

                            assert cosmos_knn_indices.shape[0] == N, (
                                f"KNN row mismatch: "
                                f"{cosmos_knn_indices.shape[0]} vs {N}"
                            )

                            assert cosmos_knn_indices.max() < N, (
                                f"KNN contains invalid index: "
                                f"max={cosmos_knn_indices.max()} vs N={N}"
                            )

                            assert cosmos_knn_indices.min() >= 0, (
                                f"KNN contains negative index: "
                                f"min={cosmos_knn_indices.min()}"
                            )

                            print(
                                f"[COSMOS] Topology OK: "
                                f"N={N}, "
                                f"IDs={gaussians.supergaussian_ids.shape[0]}, "
                                f"desc={gaussians.cosmos_descriptors.shape[0]}, "
                                f"KNN={cosmos_knn_indices.shape}"
                            )
                if (
                    iteration % opt.opacity_reset_interval == 0
                    or (
                        dataset.white_background
                        and iteration == opt.densify_from_iter
                    )
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if cosmos_initialized:
                    cosmos_optimizer.step()
                    cosmos_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")