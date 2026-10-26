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
from pathlib import Path
from math import sqrt
from plyfile import PlyData
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
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

def resolve_gs3i_ply(path):
    """Accept a Gaussian PLY or an iteration directory, without altering it."""
    ply_path = Path(path).expanduser().resolve()
    if ply_path.is_dir():
        ply_path = ply_path / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Gaussian PLY not found: {ply_path}")
    return ply_path


def ply_sh_degree(path):
    vertex = PlyData.read(str(path))["vertex"]
    names = set(vertex.data.dtype.names)
    required = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
    required.update(f"scale_{i}" for i in range(3))
    required.update(f"rot_{i}" for i in range(4))
    if not len(vertex.data) or not required.issubset(names):
        raise ValueError("Expected a nonempty Gaussian PLY with XYZ, opacity, SH, scale and rotation.")
    rest = {n for n in names if n.startswith("f_rest_")}
    bands = int(sqrt(len(rest) // 3 + 1))
    if len(rest) != 3 * (bands * bands - 1) or rest != {f"f_rest_{i}" for i in range(len(rest))}:
        raise ValueError("Invalid SH coefficient layout in Gaussian PLY.")
    return bands - 1


class FixedGeometryGaussianModel(GaussianModel):
    """Preserve imported geometry and optimize only opacity and SH colors."""

    def __init__(self, sh_degree, optimizer_type, ply_path):
        super().__init__(sh_degree, optimizer_type)
        self.initial_ply = str(ply_path)

    def create_from_pcd(self, pcd, cam_infos, spatial_lr_scale):
        # Scene supplies cameras, but initialization comes entirely from the PLY.
        self.load_ply(self.initial_ply)
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device="cuda")
        self.exposure_mapping = {cam.image_name: i for i, cam in enumerate(cam_infos)}
        self.pretrained_exposures = None
        self._exposure = torch.nn.Parameter(
            torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1),
            requires_grad=False,
        )
        print(f"Loaded {len(self._xyz)} fixed Gaussians from {self.initial_ply}")

    def training_setup(self, training_args):
        super().training_setup(training_args)
        for parameter in (self._xyz, self._scaling, self._rotation, self._exposure):
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in (self._features_dc, self._features_rest, self._opacity):
            parameter.requires_grad_(True)
        self.optimizer.param_groups[:] = [
            group for group in self.optimizer.param_groups
            if group["name"] in {"f_dc", "f_rest", "opacity"}
        ]
        self.active_sh_degree = self.max_sh_degree

    def update_learning_rate(self, iteration):
        # Only fixed-rate color and opacity groups remain.
        return None


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, gs3i_path):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = FixedGeometryGaussianModel(dataset.sh_degree, opt.optimizer_type, gs3i_path)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

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

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=(SPARSE_ADAM_AVAILABLE and not pipe.convert_SHs_python))["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # All imported SH bands stay active throughout fine-tuning.

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

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=(SPARSE_ADAM_AVAILABLE and not pipe.convert_SHs_python))
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

            # Update before saving, including the final requested iteration.
            if use_sparse_adam:
                gaussians.optimizer.step(radii > 0, radii.shape[0])
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE and not pipe.convert_SHs_python, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

from train import prepare_output_and_logger, training_report

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Fine-tune GS-3I opacity and SH colors with fixed geometry")
    parser.add_argument("--gs3i_path", required=True, help="Gaussian PLY file or directory containing point_cloud.ply")
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
    args.gs3i_path = resolve_gs3i_ply(args.gs3i_path)
    if args.train_test_exp:
        parser.error("--train_test_exp is disabled: this script trains only opacity and SH colors.")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.model_path and Path(args.model_path).resolve() in args.gs3i_path.parents:
        parser.error("Use a separate output directory to preserve the source GS-3I model.")
    args.sh_degree = ply_sh_degree(args.gs3i_path)
    print(f"PLY SH degree: {args.sh_degree}; trainable: opacity, DC color, remaining SH")
    print("Fixed: XYZ, scale, rotation, exposure, Gaussian count. No densification/pruning/reset.")
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.gs3i_path)

    # All done
    print("\nTraining complete.")
