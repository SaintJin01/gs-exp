#!/usr/bin/env python3
"""Train 16D Gaussian features with single-membership SAM correspondences."""

import json
import math
import random
import struct
import sys
from functools import lru_cache
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from saga_supervision import sample_single_supervision, contrastive_loss
from cosine_history import CosineHistory
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
import depth_boxplot_in_segment as base  # noqa: E402
from diff_gaussian_rasterization import (  # noqa: E402
    GaussianRasterizationSettings, GaussianRasterizer,
)
from gaussian_renderer import GaussianModel  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def feature_rasterizer(camera, gaussians, pipeline, white_background=False):
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=torch.full((3,), float(white_background), dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=pipeline.debug,
        antialiasing=pipeline.antialiasing,
    )
    return GaussianRasterizer(raster_settings=settings)


def render_feature_map(camera, gaussians, pipeline, gaussian_features, white_background=False):
    """Render 16D/32D once; the explicit RGB backend is a correctness reference."""
    normalized_features = gaussian_features / (gaussian_features.norm(dim=1, keepdim=True) + 1e-9)
    backend = getattr(pipeline, "feature_backend", "feature")
    if backend == "feature":
        if normalized_features.shape[1] not in (16, 32):
            raise ValueError("Feature backend requires 16D or 32D; use --feature_backend rgb for other dimensions")
        from feature_blend import render_features
        settings = feature_rasterizer(camera, gaussians, pipeline, white_background).raster_settings
        return render_features(camera, gaussians, pipeline, normalized_features,
                               white_background, settings)
    means3d = gaussians.get_xyz.detach()
    means2d = torch.zeros_like(means3d, device="cuda")
    opacity = gaussians.get_opacity.detach()
    if pipeline.compute_cov3D_python:
        scales, rotations = None, None
        covariance = gaussians.get_covariance(1.0).detach()
    else:
        scales = gaussians.get_scaling.detach()
        rotations = gaussians.get_rotation.detach()
        covariance = None

    rasterizer = feature_rasterizer(camera, gaussians, pipeline, white_background)
    rendered_chunks = []
    for start in range(0, normalized_features.shape[1], 3):
        chunk = normalized_features[:, start:start + 3]
        channel_count = chunk.shape[1]
        if channel_count < 3:
            chunk = torch.cat((
                chunk,
                torch.zeros(
                    (chunk.shape[0], 3 - channel_count),
                    dtype=chunk.dtype, device=chunk.device,
                ),
            ), dim=1)
        rendered, _, _ = rasterizer(
            means3D=means3d,
            means2D=means2d,
            shs=None,
            colors_precomp=chunk,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=covariance,
        )
        rendered_chunks.append(rendered[:channel_count])
    return torch.cat(rendered_chunks, dim=0)


def load_or_build_xyz_knn(xyz, k, cache_path, query_batch_size, rebuild=False):
    """Build exact Euclidean xyz kNN without allocating an NxN distance matrix."""
    count = int(xyz.shape[0])
    if k > count:
        raise ValueError("--smooth_k {} exceeds {} Gaussians".format(k, count))
    if cache_path.is_file() and not rebuild:
        cached = np.load(str(cache_path), allow_pickle=False)
        if cached.shape == (count, k):
            print("Loaded xyz kNN cache {}".format(cache_path))
            return torch.from_numpy(cached.astype(np.int64, copy=False)).to("cuda")
        print("Ignoring incompatible xyz kNN cache {} with shape {}".format(
            cache_path, cached.shape
        ))

    xyz_cpu = xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    print("Building exact xyz {}-NN for {:,} Gaussians...".format(k, count))
    tree = cKDTree(xyz_cpu)
    indices = np.empty((count, k), dtype=np.int64)
    for start in range(0, count, query_batch_size):
        stop = min(start + query_batch_size, count)
        batch_indices = tree.query(
            xyz_cpu[start:stop], k=k, workers=-1
        )[1]
        if k == 1:
            batch_indices = batch_indices[:, np.newaxis]
        indices[start:stop] = batch_indices
        print("  xyz kNN {:,}/{:,}".format(stop, count))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), indices)
    print("Saved xyz kNN cache {}".format(cache_path))
    return torch.from_numpy(indices).to("cuda")


class _NeighborMean(torch.autograd.Function):
    """Chunked kNN mean with one shared gradient buffer instead of N per chunk."""
    @staticmethod
    def forward(ctx, normalized, neighbor_indices, ranks, chunk_size):
        ctx.save_for_backward(neighbor_indices, ranks)
        ctx.chunk_size, ctx.shape = chunk_size, normalized.shape
        output = torch.empty_like(normalized)
        for start in range(0, len(normalized), chunk_size):
            stop = min(start + chunk_size, len(normalized))
            indices = neighbor_indices[start:stop][:, ranks]
            output[start:stop] = normalized[indices].mean(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        neighbors, ranks = ctx.saved_tensors
        gradient = torch.zeros(ctx.shape, device=grad_output.device, dtype=grad_output.dtype)
        k = len(ranks)
        for start in range(0, len(grad_output), ctx.chunk_size):
            stop = min(start + ctx.chunk_size, len(grad_output))
            indices = neighbors[start:stop][:, ranks].reshape(-1)
            values = (grad_output[start:stop, None, :].expand(-1, k, -1) / k).reshape(-1, ctx.shape[1])
            gradient.index_add_(0, indices, values)
        return gradient, None, None, None


def smooth_gaussian_features(features, neighbor_indices, dropout, chunk_size):
    """SAGA-style mean of L2-normalized xyz-neighbor features."""
    k = int(neighbor_indices.shape[1])
    if k <= 1:
        return features
    normalized = F.normalize(features, dim=1)
    if 0.0 < dropout < 1.0:
        keep_count = max(1, int(k * dropout))
        selected_ranks = torch.randperm(k, device=features.device)[:keep_count]
    else:
        selected_ranks = torch.arange(k, device=features.device)
    return _NeighborMean.apply(normalized, neighbor_indices, selected_ranks, chunk_size)


def build_camera_entries(scene, model_path, data_dir, manifest):
    entries = []
    for record in manifest["cameras"]:
        camera_id = int(record["camera_id"])
        _, camera, _ = base.find_camera_by_json_id(scene, model_path, camera_id)
        label_path = data_dir / record["file"]
        if not label_path.is_file():
            raise FileNotFoundError("Missing SAM label file: {}".format(label_path))
        entries.append({
            "camera_id": camera_id,
            "camera": camera,
            "label_path": label_path,
        })
    return entries


@lru_cache(maxsize=4)
def load_packed_masks(label_path_string):
    label_path = Path(label_path_string)
    with np.load(str(label_path), allow_pickle=False) as data:
        required = {"masks_packed", "coverage_packed", "height", "width"}
        if not required.issubset(data.files):
            raise ValueError(
                "{} uses the old disjoint-label format. Rerun "
                "prepare_sam_pairs.py to create overlapping SAGA-style masks.".format(
                    label_path
                )
            )
        masks_packed = data["masks_packed"].astype(np.uint8, copy=True)
        coverage_packed = data["coverage_packed"].astype(np.uint8, copy=True)
        height = int(data["height"])
        width = int(data["width"])
    return masks_packed, coverage_packed, height, width


def save_checkpoint(path, features, iteration, args, gaussians, optimizer=None,
                    export_features=None, gate=None, quantile=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": features.detach().cpu(),
        "training_format": "sam_single_mask_v3",
        "scale_gate_enabled": False,
        "iteration": int(iteration),
        "feature_dim": int(features.shape[1]),
        "num_gaussians": int(features.shape[0]),
        "source_iteration": int(args.source_iteration),
        "model_path": str(Path(args.model_path).resolve()),
        "training": {
            "learning_rate": args.learning_rate,
            "num_sampled_rays": args.num_sampled_rays,
            "hard_positive_threshold": args.hard_positive_threshold,
            "hard_negative_threshold": args.hard_negative_threshold,
            "rfn": args.rfn,
            "scale_gate_enabled": False,
            "smooth_k": args.smooth_k,
            "knn_dropout": args.knn_dropout,
        },
    }
    if args.save_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, str(path))
    if path.name == "latest.pt":
        exported = features if export_features is None else export_features
        normalized = exported.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        feature_path = path.with_suffix(".gsfeat")
        with open(str(feature_path), "wb") as file:
            file.write(struct.pack(
                "<8sIIII", b"GSFEAT1", 1,
                int(normalized.shape[0]), int(normalized.shape[1]), 0,
            ))
            normalized.tofile(file)


def run(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("Feature rendering requires a CUDA GPU")
    pipeline.feature_backend = args.feature_backend
    if args.feature_backend == "feature":
        from feature_blend import load_extension
        print("Preparing 16D/32D feature CUDA renderer (first run compiles once)...", flush=True)
        load_extension()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    custom_data_dir = getattr(args, "data_dir", None)
    data_dir = (
        Path(custom_data_dir) if custom_data_dir else
        Path(dataset.model_path) / "sam_feature_data"
    )
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "SAM training manifest not found: {}. Run prepare_sam_pairs.py first.".format(
                manifest_path
            )
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mask_format") != "single_membership_packbits_v3":
        raise ValueError("Single-membership masks are required. Rerun prepare_sam_pairs.py before training.")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset, gaussians, load_iteration=args.source_iteration, shuffle=False
    )
    args.source_iteration = int(scene.loaded_iter)
    args.model_path = dataset.model_path
    entries = build_camera_entries(scene, dataset.model_path, data_dir, manifest)
    if not entries:
        raise RuntimeError("No prepared cameras found in {}".format(manifest_path))
    if int(manifest["iteration"]) != args.source_iteration:
        raise ValueError("SAM mask scales were prepared from a different geometry iteration")
    gate, quantile = None, None

    num_gaussians = int(gaussians.get_xyz.shape[0])
    print("Single-mask training (no gate): {:,} Gaussians, {}D, backend={}, K={}, keep={}, iterations={}, rays={}".format(
        num_gaussians, args.feature_dim, args.feature_backend, args.smooth_k,
        max(1, int(args.smooth_k * args.knn_dropout)), args.iterations, args.num_sampled_rays))
    start_iteration = 0
    resume_path = getattr(args, "resume", None)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        if checkpoint.get("training_format") != "sam_single_mask_v3":
            raise ValueError("Checkpoint format differs; start a fresh 16D single-mask training run")
        loaded = checkpoint["features"]
        if loaded.shape != (num_gaussians, args.feature_dim):
            raise ValueError(
                "Resume feature shape {} does not match expected {}".format(
                    tuple(loaded.shape), (num_gaussians, args.feature_dim)
                )
            )
        initial = loaded.to(device="cuda", dtype=torch.float32)
        start_iteration = int(checkpoint.get("iteration", 0))
    else:
        initial = torch.randn(
            (num_gaussians, args.feature_dim), dtype=torch.float32, device="cuda"
        ) * 1e-2
    features = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([features], lr=args.learning_rate, eps=1e-15)
    if resume_path and args.save_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    custom_output = getattr(args, "output_dir", None)
    output_dir = (
        Path(custom_output) if custom_output else
        Path(dataset.model_path) / "sam_features"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cosine_history = CosineHistory(output_dir, start_iteration)
    knn_cache_path = output_dir / "xyz_knn_iteration_{:06d}_k{}.npy".format(
        args.source_iteration, args.smooth_k
    )
    neighbor_indices = load_or_build_xyz_knn(
        gaussians.get_xyz, args.smooth_k, knn_cache_path,
        args.knn_query_batch_size, args.rebuild_knn_cache,
    )
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with open(str(output_dir / "training_config.json"), "w") as file:
        json.dump(config, file, indent=2, default=str)

    ema_loss = None
    camera_stack = []
    progress = tqdm(
        range(start_iteration + 1, args.iterations + 1),
        desc="Training SAM Gaussian features",
    )
    for iteration in progress:
        if not camera_stack:
            camera_stack = entries.copy()
        entry = camera_stack.pop(random.randrange(len(camera_stack)))
        camera = entry["camera"]
        packed, coverage, height, width = load_packed_masks(str(entry["label_path"]))
        with np.load(str(entry["label_path"]), allow_pickle=False) as data:
            sampled = sample_single_supervision(packed, coverage, data["mask_areas"],
                                                height, width, args.num_sampled_rays,
                                                features.device, args.ray_sample_rate)
        if sampled is None:
            continue

        smoothed_features = smooth_gaussian_features(
            features, neighbor_indices, args.knn_dropout, args.knn_chunk_size
        )
        feature_map = render_feature_map(
            camera, gaussians, pipeline, smoothed_features, dataset.white_background
        )
        log_cosine = (iteration == start_iteration + 1 or iteration == args.iterations
                      or iteration % args.cosine_log_every == 0)
        loss, stats = contrastive_loss(feature_map, sampled, gate, quantile,
                                       (height, width), args.rfn,
                                       args.hard_positive_threshold,
                                       args.hard_negative_threshold, -1,
                                       collect_cosine=log_cosine)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite SAGA loss at camera {}".format(entry["camera_id"]))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if log_cosine:
            cosine_history.record(iteration, entry["camera_id"],
                                  stats["positive_cosine"], stats["negative_cosine"])
        if (iteration == start_iteration + 1 or iteration == args.iterations
                or iteration % args.cosine_plot_every == 0):
            cosine_history.plot()

        value = float(loss.detach())
        ema_loss = value if ema_loss is None else 0.1 * value + 0.9 * ema_loss
        progress.set_postfix(
            loss="{:.4f}".format(ema_loss),
            norm="{:.3f}".format(stats["rendered_norm"]),
            hard="{}/{}".format(
                stats["positive_pairs"], stats["negative_pairs"]
            ),
            camera=entry["camera_id"],
        )
        if iteration % args.save_every == 0 or iteration == args.iterations:
            with torch.no_grad():
                export_features = smooth_gaussian_features(
                    features, neighbor_indices, -1.0, args.knn_chunk_size
                )
            save_checkpoint(
                output_dir / "iteration_{:06d}.pt".format(iteration),
                features, iteration, args, gaussians, optimizer,
                gate=gate, quantile=quantile,
            )
            save_checkpoint(
                output_dir / "latest.pt",
                features, iteration, args, gaussians, optimizer,
                export_features=export_features,
                gate=gate, quantile=quantile,
            )
            del export_features
        # Do not carry the previous render graph into the next native CUDA pass.
        del feature_map, loss, smoothed_features, sampled
    cosine_history.plot()
    print("Cosine training chart: {}".format(cosine_history.plot_path))
    print("Saved trained Gaussian features to {}".format(output_dir / "latest.pt"))
    return output_dir / "latest.pt"


def build_parser():
    parser = ArgumentParser(
        description="Train 16D alpha-blended Gaussian features without scale gates"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--source_iteration", "--iteration", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--feature_backend", choices=("feature", "rgb"), default="feature",
                        help="feature: single geometry pass and feature-only backward; rgb: slow reference")
    parser.add_argument("--rfn", type=float, default=1.0)
    parser.add_argument("--ray_sample_rate", type=float, default=0.0)
    parser.add_argument(
        "--smooth_k", type=int, default=16,
        help="xyz-neighbor count for SAGA-style feature smoothing",
    )
    parser.add_argument(
        "--knn_dropout", type=float, default=0.5,
        help="fraction of the K neighbor ranks retained during training",
    )
    parser.add_argument(
        "--knn_chunk_size", type=int, default=65536,
        help="Gaussian batch size used while averaging neighbor features",
    )
    parser.add_argument(
        "--knn_query_batch_size", type=int, default=200000,
        help="CPU batch size used to construct the exact xyz kNN cache",
    )
    parser.add_argument("--rebuild_knn_cache", action="store_true")
    parser.add_argument(
        "--num_sampled_rays", "--num_sampled_pairs", dest="num_sampled_rays",
        type=int, default=1000,
        help="expected Bernoulli ray count over the whole image, before masking",
    )
    parser.add_argument("--learning_rate", type=float, default=0.0025)
    parser.add_argument(
        "--hard_positive_threshold", type=float, default=0.75,
        help="SAGA hard-positive cosine threshold",
    )
    parser.add_argument(
        "--hard_negative_threshold", type=float, default=0.5,
        help="SAGA hard-negative cosine threshold",
    )
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--cosine_log_every", type=int, default=10)
    parser.add_argument("--cosine_plot_every", type=int, default=100)
    parser.add_argument("--save_optimizer", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


if __name__ == "__main__":
    arg_parser, model_params, pipeline_params = build_parser()
    parsed = get_combined_args(arg_parser)
    if not getattr(parsed, "model_path", None):
        arg_parser.error("--model_path/-m is required")
    if parsed.feature_dim <= 0:
        arg_parser.error("--feature_dim must be positive")
    if parsed.feature_backend == "feature" and parsed.feature_dim not in (16, 32):
        arg_parser.error("--feature_backend feature requires --feature_dim 16 or 32")
    if parsed.smooth_k <= 0:
        arg_parser.error("--smooth_k must be positive")
    if not 0.0 < parsed.knn_dropout <= 1.0:
        arg_parser.error("--knn_dropout must be in (0, 1]")
    if parsed.knn_chunk_size <= 0 or parsed.knn_query_batch_size <= 0:
        arg_parser.error("kNN batch sizes must be positive")
    if parsed.iterations <= 0 or parsed.num_sampled_rays <= 0:
        arg_parser.error("--iterations and --num_sampled_rays must be positive")
    if parsed.save_every <= 0:
        arg_parser.error("--save_every must be positive")
    if parsed.cosine_log_every <= 0 or parsed.cosine_plot_every <= 0:
        arg_parser.error("cosine logging and plotting intervals must be positive")
    safe_state(parsed.quiet)
    run(parsed, model_params.extract(parsed), pipeline_params.extract(parsed))
