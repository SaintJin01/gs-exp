#!/usr/bin/env python3
"""Run SAM on a 3DGS render and plot contributor depths for every segment."""

import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SEPARATE_SH = True
except ImportError:
    SEPARATE_SH = False


def camera_aliases(camera):
    name = str(camera.image_name)
    path = Path(name)
    return {name.casefold(), path.name.casefold(), path.stem.casefold()}


def find_camera_by_json_id(scene, model_path, camera_id):
    """Resolve a loaded camera from the ID recorded in cameras.json.

    Scene writes cameras.json in test-then-train order. We reproduce that order
    first and verify the recorded image name. A name-based fallback supports
    externally produced cameras.json files whose records were reordered.
    """
    cameras_path = Path(model_path) / "cameras.json"
    if not cameras_path.is_file():
        raise FileNotFoundError("cameras.json not found: {}".format(cameras_path))
    records = json.loads(cameras_path.read_text())
    indexed_records = [(index, record) for index, record in enumerate(records)
                       if int(record["id"]) == camera_id]
    if len(indexed_records) != 1:
        available = sorted(int(record["id"]) for record in records)
        raise ValueError(
            "Camera ID {} occurs {} times in cameras.json. Available IDs: {}".format(
                camera_id, len(indexed_records), available
            )
        )
    record_index, record = indexed_records[0]
    recorded_name = str(record["img_name"])
    ordered = ([('test', camera) for camera in scene.getTestCameras()] +
               [('train', camera) for camera in scene.getTrainCameras()])
    if record_index < len(ordered):
        split, camera = ordered[record_index]
        if camera_aliases(camera).intersection(camera_aliases_from_name(recorded_name)):
            return split, camera, record

    matches = []
    recorded_aliases = camera_aliases_from_name(recorded_name)
    for split, cameras in (("test", scene.getTestCameras()), ("train", scene.getTrainCameras())):
        for camera in cameras:
            if camera_aliases(camera).intersection(recorded_aliases):
                matches.append((split, camera))
    if len(matches) != 1:
        names = ["{}:{}".format(split, camera.image_name) for split, camera in matches]
        raise ValueError(
            "Camera ID {} maps to {!r}, but loaded-camera matching is ambiguous: {}".format(
                camera_id, recorded_name, names
            )
        )
    return matches[0][0], matches[0][1], record


def camera_aliases_from_name(name):
    path = Path(str(name))
    return {str(name).casefold(), path.name.casefold(), path.stem.casefold()}


def tensor_to_rgb_uint8(image):
    image = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(image * 255.0).astype(np.uint8)


def resolve_sam_checkpoint(explicit_path, model_path, model_type):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if os.environ.get("SAM_CHECKPOINT"):
        candidates.append(Path(os.environ["SAM_CHECKPOINT"]).expanduser())
    names = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }
    search_names = list(names.values()) if model_type == "auto" else [names[model_type]]
    roots = [Path(model_path), Path(model_path) / "checkpoints", REPO_ROOT / "checkpoints", REPO_ROOT / "assets", Path.cwd()]
    for root in roots:
        candidates.extend(root / name for name in search_names)
    seen = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "SAM checkpoint not found. Pass --sam_checkpoint, set SAM_CHECKPOINT, or place one "
        "in ./checkpoints. Checked:\n  " + "\n  ".join(str(path) for path in seen)
    )


def infer_sam_model_type(checkpoint, requested_type):
    if requested_type != "auto":
        return requested_type
    for model_type in ("vit_h", "vit_l", "vit_b"):
        if model_type in checkpoint.name.casefold():
            return model_type
    raise ValueError("Cannot infer SAM model type from {}; pass --sam_model_type".format(checkpoint))


def generate_sam_masks(image, checkpoint, model_type, args):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment-anything is not installed. Run: "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=args.sam_device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_area,
        output_mode="binary_mask",
    )
    masks = generator.generate(image)
    masks = [mask for mask in masks if int(mask["area"]) >= args.min_mask_area]
    masks.sort(key=lambda mask: int(mask["area"]), reverse=True)
    if args.max_segments > 0:
        masks = masks[:args.max_segments]
    del generator, sam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return masks


def render_contributor_alpha(camera, gaussians, pipeline):
    """Render accumulated alpha while retaining gradients to input opacities."""
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=pipeline.debug,
        antialiasing=pipeline.antialiasing,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    means3d = gaussians.get_xyz
    means2d = torch.zeros_like(means3d, requires_grad=True, device="cuda")
    opacity = gaussians.get_opacity.detach().requires_grad_(True)
    colors = torch.ones((means3d.shape[0], 3), dtype=means3d.dtype, device="cuda")
    if pipeline.compute_cov3D_python:
        scales, rotations, covariance = None, None, gaussians.get_covariance(1.0)
    else:
        scales, rotations, covariance = gaussians.get_scaling, gaussians.get_rotation, None
    alpha_rgb, _, _ = rasterizer(
        means3D=means3d, means2D=means2d, shs=None, colors_precomp=colors,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=covariance,
    )
    return alpha_rgb[0], opacity


def gaussian_camera_depths(camera, gaussians):
    xyz = gaussians.get_xyz.detach()
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    return torch.cat((xyz, ones), dim=1).matmul(camera.world_view_transform)[:, 2].cpu().numpy()


def collect_segment_depths(alpha_image, opacity, masks, camera_depths):
    """Collect each unique Gaussian that the rasterizer blended in a mask."""
    results = []
    for index, annotation in enumerate(masks):
        mask = torch.from_numpy(annotation["segmentation"]).to(alpha_image.device, alpha_image.dtype)
        gradient = torch.autograd.grad(
            (alpha_image * mask).sum(), opacity,
            retain_graph=index + 1 < len(masks), create_graph=False, allow_unused=False,
        )[0]
        contributor_ids = torch.nonzero(
            torch.isfinite(gradient[:, 0]) & gradient[:, 0].ne(0), as_tuple=False
        ).squeeze(1)
        ids = contributor_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        valid = np.isfinite(camera_depths[ids]) & (camera_depths[ids] > 0)
        ids = ids[valid]
        depths = camera_depths[ids]
        results.append({"ids": ids, "depths": depths})
        print("  segment {:03d}: {:7d} pixels, {:7d} Gaussians".format(index, int(annotation["area"]), len(depths)))
    return results


def mask_color(index):
    return np.random.RandomState(1729 + index).randint(35, 240, size=3).astype(np.uint8)


def save_overlay(image, masks, output_path):
    overlay = image.astype(np.float32).copy()
    fig, ax = plt.subplots(figsize=(max(8, image.shape[1] / 140), max(5, image.shape[0] / 140)))
    for index, annotation in enumerate(masks):
        mask = annotation["segmentation"]
        overlay[mask] = 0.62 * overlay[mask] + 0.38 * mask_color(index)
    ax.imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    for index, annotation in enumerate(masks):
        ys, xs = np.nonzero(annotation["segmentation"])
        if len(xs):
            ax.text(float(np.median(xs)), float(np.median(ys)), str(index), ha="center", va="center",
                    fontsize=7, color="white",
                    bbox={"facecolor": "black", "alpha": 0.65, "pad": 1, "edgecolor": "none"})
    ax.set_title("SAM segments on 3DGS render")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def masked_crop(image, annotation, color, margin=8):
    x, y, width, height = [int(round(value)) for value in annotation["bbox"]]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(image.shape[1], x + width + margin), min(image.shape[0], y + height + margin)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    mask = annotation["segmentation"][y0:y1, x0:x1]
    shown = crop * 0.18
    shown[mask] = 0.78 * crop[mask] + 0.22 * color.astype(np.float32)
    return np.clip(shown, 0, 255).astype(np.uint8)


def depth_summary(depths):
    if len(depths) == 0:
        return {"count": 0, "min": None, "q1": None, "median": None, "q3": None,
                "max": None, "lower_fence": None, "lower_whisker": None,
                "upper_fence": None, "upper_whisker": None, "near_outlier_count": 0}
    values = np.percentile(depths, [0, 25, 50, 75, 100])
    iqr = float(values[3] - values[1])
    lower_fence = float(values[1] - 1.5 * iqr)
    upper_fence = float(values[3] + 1.5 * iqr)
    lower_whisker = float(depths[depths >= lower_fence].min())
    upper_whisker = float(depths[depths <= upper_fence].max())
    return {"count": int(len(depths)), "min": float(values[0]), "q1": float(values[1]),
            "median": float(values[2]), "q3": float(values[3]), "max": float(values[4]),
            "lower_fence": lower_fence, "lower_whisker": lower_whisker,
            "upper_fence": upper_fence, "upper_whisker": upper_whisker,
            "near_outlier_count": int((depths < lower_whisker).sum())}


def save_boxplot_pages(image, masks, segment_data, output_dir, rows_per_page):
    for start in range(0, len(masks), rows_per_page):
        end = min(start + rows_per_page, len(masks))
        fig = plt.figure(figsize=(13, max(3.0, (end - start) * 2.5)))
        grid = fig.add_gridspec(end - start, 2, width_ratios=[1.05, 2.6], hspace=0.55, wspace=0.18)
        for row, index in enumerate(range(start, end)):
            annotation, depths = masks[index], segment_data[index]["depths"]
            summary = depth_summary(depths)
            image_ax = fig.add_subplot(grid[row, 0])
            image_ax.imshow(masked_crop(image, annotation, mask_color(index)))
            image_ax.set_title("Segment {:03d} | {:,} px".format(index, int(annotation["area"])), fontsize=10)
            image_ax.axis("off")
            plot_ax = fig.add_subplot(grid[row, 1])
            if len(depths):
                plot_ax.boxplot(depths, vert=False, widths=0.55, patch_artist=True,
                                showfliers=False,
                                boxprops={"facecolor": "#4C9BE8", "alpha": 0.7},
                                medianprops={"color": "#B00020", "linewidth": 2},
                                flierprops={"markersize": 2, "alpha": 0.35})
                lower_whisker = summary["lower_whisker"]
                near_depths = depths[depths < lower_whisker]
                plot_ax.axvspan(0, lower_whisker, color="#D62728", alpha=0.08)
                plot_ax.axvline(lower_whisker, color="#D62728", linestyle="--",
                                linewidth=1.5, label="lower whisker")
                if len(near_depths):
                    stride = max(1, len(near_depths) // 200)
                    shown_near = near_depths[::stride][:200]
                    jitter = np.linspace(0.78, 1.22, len(shown_near))
                    plot_ax.scatter(shown_near, jitter, s=9, color="#D62728", alpha=0.6,
                                    zorder=4, label="saved near outliers")
                plot_ax.set_xlim(0, summary["q3"] + 10.0)
                plot_ax.set_yticks([])
                plot_ax.grid(axis="x", alpha=0.25)
                plot_ax.set_xlabel("camera-space Gaussian center depth")
                near_ratio = 100.0 * summary["near_outlier_count"] / summary["count"]
                plot_ax.set_title(
                    ("{:,} Gaussians | lower whisker {:.4g} | "
                     "saved {:,} ({:.2f}%) | Q3 {:.4g}").format(
                        summary["count"], lower_whisker,
                        summary["near_outlier_count"], near_ratio, summary["q3"]),
                    fontsize=10)
            else:
                plot_ax.text(0.5, 0.5, "No contributing Gaussians", ha="center", va="center")
                plot_ax.set_axis_off()
        page = start // rows_per_page + 1
        fig.suptitle("SAM segment Gaussian depth distributions", fontsize=15, y=0.997)
        fig.savefig(str(output_dir / "depth_boxplots_{:03d}.png".format(page)), dpi=170, bbox_inches="tight")
        plt.close(fig)


def save_results(masks, segment_data, output_dir, metadata):
    records, arrays = [], {}
    for index, (annotation, data) in enumerate(zip(masks, segment_data)):
        records.append({
            "segment_id": index, "pixel_area": int(annotation["area"]),
            "bbox_xywh": [float(value) for value in annotation["bbox"]],
            "predicted_iou": float(annotation["predicted_iou"]),
            "stability_score": float(annotation["stability_score"]),
            "depth": depth_summary(data["depths"]),
        })
        arrays["segment_{:03d}_depth".format(index)] = data["depths"].astype(np.float32)
        arrays["segment_{:03d}_gaussian_id".format(index)] = data["ids"].astype(np.int64)
    report = dict(metadata)
    report["segments"] = records
    with open(str(output_dir / "segments.json"), "w") as file:
        json.dump(report, file, indent=2)
    np.savez_compressed(str(output_dir / "segment_gaussians.npz"), **arrays)


def run(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("The 3DGS rasterizer requires a CUDA GPU")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    split, camera, camera_record = find_camera_by_json_id(
        scene, dataset.model_path, args.camera_id
    )
    print("Matched cameras.json ID {} to {} camera: {}".format(
        args.camera_id, split, camera.image_name
    ))
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    with torch.no_grad():
        rendered = render(camera, gaussians, pipeline, bg, separate_sh=SEPARATE_SH,
                          use_trained_exp=dataset.train_test_exp)["render"]
    rendered_rgb = tensor_to_rgb_uint8(rendered)

    checkpoint = resolve_sam_checkpoint(getattr(args, "sam_checkpoint", None),
                                        dataset.model_path, args.sam_model_type)
    sam_type = infer_sam_model_type(checkpoint, args.sam_model_type)
    print("Running SAM {} from {}".format(sam_type, checkpoint))
    masks = generate_sam_masks(rendered_rgb, checkpoint, sam_type, args)
    if not masks:
        raise RuntimeError("SAM produced no masks after filtering")
    print("SAM produced {} segments".format(len(masks)))

    print("Tracing rasterizer contributors for each segment")
    alpha_image, opacity = render_contributor_alpha(camera, gaussians, pipeline)
    segment_data = collect_segment_depths(
        alpha_image, opacity, masks, gaussian_camera_depths(camera, gaussians)
    )
    custom_output = getattr(args, "output_dir", None)
    default_folder = "camera_{:05d}_{}".format(
        args.camera_id, Path(str(camera.image_name)).stem
    )
    output_dir = (Path(custom_output) if custom_output else
                  Path(dataset.model_path) / "sam_boxplot" / default_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered_rgb).save(str(output_dir / "render.png"))
    save_overlay(rendered_rgb, masks, output_dir / "segments_overlay.png")
    save_boxplot_pages(rendered_rgb, masks, segment_data, output_dir, args.rows_per_page)
    save_results(masks, segment_data, output_dir, {
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter), "camera_id": int(args.camera_id),
        "cameras_json_record": camera_record, "camera_split": split,
        "image_name": str(camera.image_name), "sam_checkpoint": str(checkpoint),
        "sam_model_type": sam_type,
        "depth_definition": "positive camera-space z of each unique contributing Gaussian center",
        "contributor_definition": "nonzero opacity gradient of masked accumulated-alpha render; follows rasterizer alpha cutoff and transmittance termination",
    })
    print("Saved SAM depth box plots to {}".format(output_dir))
    return output_dir


def build_parser():
    parser = ArgumentParser(description="SAM segment box plots of contributing 3DGS Gaussian depths")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--camera_id", "-c", required=True, type=int,
                        help="Camera ID from <model_path>/cameras.json")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_model_type", choices=("auto", "vit_h", "vit_l", "vit_b"), default="auto")
    parser.add_argument("--sam_device", type=str, default="cuda")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--min_mask_area", type=int, default=0)
    parser.add_argument("--max_segments", type=int, default=0, help="Largest masks to keep; 0 keeps all")
    parser.add_argument("--rows_per_page", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


if __name__ == "__main__":
    arg_parser, model_params, pipeline_params = build_parser()
    parsed = get_combined_args(arg_parser)
    if not getattr(parsed, "model_path", None):
        arg_parser.error("--model_path/-m is required")
    if parsed.rows_per_page <= 0:
        arg_parser.error("--rows_per_page must be positive")
    safe_state(parsed.quiet)
    run(parsed, model_params.extract(parsed), pipeline_params.extract(parsed))
