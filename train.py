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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

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
    cosmos_initialized = False
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

            groups = build_supergaussians(gaussians)

            gaussians.supergaussian_ids = (
                groups["supergaussian_ids"]
            )

            cosmos_first_edge = groups["first_edge"]
            cosmos_adj_vertices = groups["adj_vertices"]
            cosmos_edge_weights = groups["edge_weights"]
            cosmos_knn_indices = groups["knn_indices"]

            group_centers = groups["group_centers"]

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
            # Build COSMOS features for current Gaussian state
            # ----------------------------------------------------

            # ----------------------------------------------------
            # COSMOS residual prediction heads
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
            # COSMOS optimizer
            # ----------------------------------------------------

            cosmos_optimizer = torch.optim.Adam(
                list(global_attention.parameters())
                + list(local_attention.parameters())
                + list(position_head.parameters())
                + list(orientation_head.parameters())
                + list(scale_head.parameters())
                + list(color_head.parameters())
                + list(opacity_head.parameters()),
                lr=1e-4
            )

            print("Residual heads initialized.")
            print("Unified feature dimension:", unified_dim)

            xyz = gaussians.get_xyz

            sh_dc = gaussians.get_features[:, 0, :]
            color = SH2RGB(sh_dc)

            scale = gaussians.get_scaling

            descriptors = groups["features"][:, 9:13]

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

            unique_ids = torch.unique(
                supergaussian_ids
            )

            group_features = []

            for group_id in unique_ids:

                mask = (
                    supergaussian_ids == group_id
                )

                group_features.append(
                    gaussian_features[mask].max(
                        dim=0
                    ).values
                )

            group_features = torch.stack(
                group_features,
                dim=0
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
            # ============================================================
            # COSMOS — GLOBAL SELF-ATTENTION VALIDATION
            # ============================================================

            print("=" * 70)
            print("GLOBAL SELF-ATTENTION VALIDATION")
            print("=" * 70)

            print("Input group centers :", group_centers.shape)
            print("Input group features:", group_remaining.shape)
            print("Output features     :", global_features.shape)
            print("Attention shape     :", global_attention_weights.shape)

            print(
                "Output finite:",
                torch.isfinite(global_features).all().item()
            )

            print(
                "Attention finite:",
                torch.isfinite(global_attention_weights).all().item()
            )

            # Each attention distribution should sum to 1
            attention_row_sums = (
                global_attention_weights.sum(dim=-1)
            )

            print(
                "Attention row sum min/max:",
                attention_row_sums.min().item(),
                attention_row_sums.max().item()
            )

            print(
                "Global feature mean/std:",
                global_features.mean().item(),
                global_features.std().item()
            )

            print("=" * 70)
            # ----------------------------------------------------
            # Broadcast global features to Gaussians
            # ----------------------------------------------------

            global_gaussian_features = (
                global_features[
                    supergaussian_ids
                ]
            )

            # ----------------------------------------------------
            # Local Gaussian features
            # ----------------------------------------------------

            local_pos = PositionalEncoding3D(
                num_freqs=6
            ).cuda()(xyz)

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

            print("Residual shapes:")
            print("  Δposition :", delta_position.shape)
            print("  Δrotation :", delta_rotation.shape)
            print("  Δscale    :", delta_scale.shape)
            print("  Δcolor    :", delta_color.shape)
            print("  Δopacity  :", delta_opacity.shape)

            print(
                "Global features:",
                global_gaussian_features.shape
            )

            print(
                "Local features:",
                local_features_out.shape
            )

            print(
                "Unified features:",
                unified_features.shape
            )

            # ----------------------------------------------------
            # COSMOS refined Gaussian attributes
            # ----------------------------------------------------

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

            print("=" * 70)
            print("COSMOS REFINED ATTRIBUTES")
            print("=" * 70)

            print("Position :", refined_position.shape)
            print("Rotation :", refined_rotation.shape)
            print("Scale    :", refined_scale.shape)
            print("Color    :", refined_color.shape)
            print("Opacity  :", refined_opacity.shape)

            # ============================================================
            # COSMOS — POSITION REGULARIZATION
            # ============================================================

            refined_xyz = (
                gaussians.get_xyz
                + delta_position
            )

            D_avg = compute_D_avg(
                refined_xyz,
                cosmos_first_edge,
                cosmos_adj_vertices
            )

            D_ctr = compute_D_ctr(
                refined_xyz,
                gaussians.supergaussian_ids
            )

            # ============================================================
            # COSMOS POSITION REGULARIZATION LOSS
            # ============================================================

            lambda1_pos = 1.0
            lambda2_pos = 1.0

            L_pos = (
                lambda1_pos * D_avg
                + lambda2_pos * D_ctr
)

            print("COSMOS position regularization:")
            print("  D_avg :", D_avg.item())
            print("  D_ctr :", D_ctr.item())
            print("  L_pos :", L_pos.item())


        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
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

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
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

        loss.backward()

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

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
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
