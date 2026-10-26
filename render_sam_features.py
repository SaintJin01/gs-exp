#!/usr/bin/env python3
"""Render trained per-Gaussian SAM affinity features."""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
import depth_boxplot_in_segment as base  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
import train_sam_features as feature_training  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def pca_rgb(feature_map, max_samples=100000):
    channels, height, width = feature_map.shape
    flat_raw = feature_map.reshape(channels, -1).T
    visible = torch.linalg.vector_norm(flat_raw, dim=1) > 1e-8
    flat = F.normalize(flat_raw, dim=1, eps=1e-8)
    indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
    if len(indices) < 3:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if len(indices) > max_samples:
        step = max(1, len(indices) // max_samples)
        indices = indices[::step][:max_samples]
    samples = flat[indices]
    center = samples.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(samples - center, q=3, center=False, niter=5)
    projected = (flat - center).matmul(components[:, :3])
    visible_values = projected[visible]
    low = torch.quantile(visible_values, 0.01, dim=0)
    high = torch.quantile(visible_values, 0.99, dim=0)
    projected = (projected - low) / (high - low).clamp_min(1e-8)
    projected = projected.clamp(0, 1)
    projected[~visible] = 0
    rgb = projected.reshape(height, width, 3).detach().cpu().numpy()
    return np.rint(rgb * 255.0).astype(np.uint8)


def save_similarity(rendered_rgb, similarity, threshold, prompt_x, prompt_y,
                    output_dir):
    similarity_np = similarity.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rendered_rgb)
    axes[0].scatter([prompt_x], [prompt_y], c="red", s=45, marker="x")
    axes[0].set_title("Prompt ({}, {})".format(prompt_x, prompt_y))
    axes[0].axis("off")
    shown = axes[1].imshow(similarity_np, cmap="turbo", vmin=-1, vmax=1)
    axes[1].set_title("Cosine similarity")
    axes[1].axis("off")
    fig.colorbar(shown, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(output_dir / "similarity.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    mask = similarity_np >= threshold
    segmented = rendered_rgb.copy()
    segmented[~mask] = 0
    Image.fromarray(segmented).save(str(output_dir / "similarity_mask.png"))
    np.save(str(output_dir / "similarity.npy"), similarity_np.astype(np.float32))


def run(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("Feature rendering requires a CUDA GPU")
    custom_checkpoint = getattr(args, "feature_checkpoint", None)
    checkpoint_path = (
        Path(custom_checkpoint) if custom_checkpoint else
        Path(dataset.model_path) / "sam_features" / "latest.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Feature checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source_iteration = (
        args.source_iteration if args.source_iteration != -1 else
        int(checkpoint.get("source_iteration", -1))
    )

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=source_iteration, shuffle=False)
    split, camera, camera_record = base.find_camera_by_json_id(
        scene, dataset.model_path, args.camera_id
    )
    features = checkpoint["features"].to(device="cuda", dtype=torch.float32)
    pipeline.feature_backend = (args.feature_backend if args.feature_backend is not None
                                else ("feature" if features.shape[1] in (16, 32) else "rgb"))
    if features.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError(
            "Feature/Gaussian count mismatch: {} versus {}".format(
                features.shape[0], gaussians.get_xyz.shape[0]
            )
        )

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    with torch.no_grad():
        smooth_k = int(checkpoint.get("training", {}).get("smooth_k", 1))
        if smooth_k > 1:
            cache = checkpoint_path.parent / "xyz_knn_iteration_{:06d}_k{}.npy".format(
                scene.loaded_iter, smooth_k)
            neighbors = feature_training.load_or_build_xyz_knn(
                gaussians.get_xyz, smooth_k, cache, 200000)
            features = feature_training.smooth_gaussian_features(features, neighbors, -1, 65536)
        rendered = render(
            camera, gaussians, pipeline, background,
            separate_sh=base.SEPARATE_SH,
            use_trained_exp=dataset.train_test_exp,
        )["render"]
        feature_map = feature_training.render_feature_map(
            camera, gaussians, pipeline, features, dataset.white_background
        )
        if args.scale is not None:
            print("--scale ignored: scale gate disabled")
    rendered_rgb = base.tensor_to_rgb_uint8(rendered)

    default_folder = "camera_{:05d}_{}".format(
        args.camera_id, Path(str(camera.image_name)).stem
    )
    custom_output = getattr(args, "output_dir", None)
    output_dir = (
        Path(custom_output) if custom_output else
        Path(dataset.model_path) / "sam_features" / "renders" / default_folder
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered_rgb).save(str(output_dir / "render.png"))
    Image.fromarray(pca_rgb(feature_map, args.pca_samples)).save(
        str(output_dir / "feature_pca.png")
    )

    prompt_x = getattr(args, "prompt_x", None)
    prompt_y = getattr(args, "prompt_y", None)
    if (prompt_x is None) != (prompt_y is None):
        raise ValueError("Pass both --prompt_x and --prompt_y, or neither")
    if prompt_x is not None:
        x, y = int(prompt_x), int(prompt_y)
        height, width = int(camera.image_height), int(camera.image_width)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Prompt ({}, {}) is outside {}x{} image".format(
                x, y, width, height
            ))
        normalized = F.normalize(feature_map, dim=0, eps=1e-8)
        prompt = normalized[:, y, x]
        similarity = torch.sum(normalized * prompt[:, None, None], dim=0)
        save_similarity(
            rendered_rgb, similarity, args.similarity_threshold,
            x, y, output_dir,
        )

    metadata = {
        "model_path": str(Path(dataset.model_path).resolve()),
        "source_iteration": int(scene.loaded_iter),
        "feature_checkpoint": str(checkpoint_path.resolve()),
        "feature_training_iteration": int(checkpoint.get("iteration", -1)),
        "feature_dim": int(features.shape[1]),
        "scale_gate_applied": False,
        "camera_id": int(args.camera_id),
        "camera_split": split,
        "image_name": str(camera.image_name),
        "cameras_json_record": camera_record,
        "prompt": None if prompt_x is None else {
            "x": int(prompt_x), "y": int(prompt_y),
            "similarity_threshold": args.similarity_threshold,
        },
    }
    with open(str(output_dir / "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)
    if args.save_feature_map:
        np.save(
            str(output_dir / "feature_map.npy"),
            feature_map.detach().cpu().numpy().astype(np.float16),
        )
    print("Saved rendered SAM features to {}".format(output_dir))
    return output_dir


def build_parser():
    parser = ArgumentParser(description="Render trained SAM Gaussian affinity features")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--camera_id", "-c", required=True, type=int)
    parser.add_argument("--source_iteration", "--iteration", type=int, default=-1)
    parser.add_argument("--feature_checkpoint", type=str, default=None)
    parser.add_argument("--feature_backend", choices=("feature", "rgb"), default=None,
                        help="default: fast feature backend for 16D/32D, RGB reference otherwise")
    parser.add_argument("--prompt_x", type=int, default=None)
    parser.add_argument("--prompt_y", type=int, default=None)
    parser.add_argument("--similarity_threshold", type=float, default=0.7)
    parser.add_argument("--pca_samples", type=int, default=100000)
    parser.add_argument("--scale", type=float, default=None,
                        help="Deprecated; ignored (no scale gate)")
    parser.add_argument("--save_feature_map", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


if __name__ == "__main__":
    arg_parser, model_params, pipeline_params = build_parser()
    parsed = get_combined_args(arg_parser)
    if not getattr(parsed, "model_path", None):
        arg_parser.error("--model_path/-m is required")
    if parsed.pca_samples < 3:
        arg_parser.error("--pca_samples must be at least 3")
    if parsed.scale is not None and not 0 <= parsed.scale <= 1:
        arg_parser.error("--scale must be in [0,1]")
    safe_state(parsed.quiet)
    run(parsed, model_params.extract(parsed), pipeline_params.extract(parsed))
