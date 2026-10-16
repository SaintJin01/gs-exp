#!/usr/bin/env python3
"""Cluster rasterizer-contributing Gaussians inside each SAM segment."""

import json
import math
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
import depth_boxplot_in_segment as base  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def import_hdbscan():
    try:
        import hdbscan
    except ImportError as exc:
        raise RuntimeError(
            "cluster_in_segments.py requires the external hdbscan package. "
            "Install it in the gaussian_splatting environment with: "
            "/home/sj/anaconda3/envs/gaussian_splatting/bin/pip install hdbscan"
        ) from exc
    return hdbscan


def gaussian_camera_xyz(camera, gaussians):
    xyz = gaussians.get_xyz.detach()
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    return torch.cat((xyz, ones), dim=1).matmul(
        camera.world_view_transform
    )[:, :3].cpu().numpy()


def gaussian_pixel_xy(camera, gaussians):
    xyz = gaussians.get_xyz.detach()
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    clip = torch.cat((xyz, ones), dim=1).matmul(camera.full_proj_transform)
    denominator = clip[:, 3:4]
    safe_denominator = torch.where(
        denominator.abs() > 1e-8, denominator, torch.full_like(denominator, float("nan"))
    )
    ndc = clip[:, :2] / safe_denominator
    pixels = torch.empty_like(ndc)
    pixels[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * float(camera.image_width)
    pixels[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * float(camera.image_height)
    return pixels.cpu().numpy()


def robust_camera_features(camera_xyz, depth_weight):
    """Return robust-scaled [x/z, y/z, depth_weight*z] features.

    Scaling z before independent robust scaling would cancel depth_weight, so
    all three base dimensions are scaled first and the z dimension is weighted
    afterward.
    """
    z = camera_xyz[:, 2]
    base_features = np.column_stack((camera_xyz[:, 0] / z,
                                     camera_xyz[:, 1] / z, z))
    median = np.median(base_features, axis=0)
    q1, q3 = np.percentile(base_features, [25, 75], axis=0)
    scale = q3 - q1
    fallback = np.median(np.abs(base_features - median), axis=0) * 1.4826
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    features = (base_features - median) / scale
    features[:, 2] *= depth_weight
    return base_features, features, median, scale


def cluster_segment(camera_xyz, args):
    count = len(camera_xyz)
    empty_float = np.empty((0,), dtype=np.float32)
    if count == 0:
        return {
            "base_features": np.empty((0, 3), dtype=np.float64),
            "features": np.empty((0, 3), dtype=np.float64),
            "labels": np.empty((0,), dtype=np.int32),
            "probabilities": empty_float,
            "outlier_scores": empty_float,
            "median": np.zeros(3), "scale": np.ones(3),
            "effective_min_cluster_size": 0,
        }

    base_features, features, median, scale = robust_camera_features(
        camera_xyz, args.depth_weight
    )
    effective_min_cluster_size = max(
        args.min_cluster_size,
        int(math.ceil(args.min_cluster_fraction * count)),
    )
    if count < effective_min_cluster_size or count < 2:
        labels = np.full(count, -1, dtype=np.int32)
        probabilities = np.zeros(count, dtype=np.float32)
        outlier_scores = np.ones(count, dtype=np.float32)
    else:
        hdbscan = import_hdbscan()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",
            cluster_selection_method=args.cluster_selection_method,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            core_dist_n_jobs=args.hdbscan_jobs,
            prediction_data=False,
        )
        labels = clusterer.fit_predict(features).astype(np.int32, copy=False)
        probabilities = clusterer.probabilities_.astype(np.float32, copy=False)
        outlier_scores = clusterer.outlier_scores_.astype(np.float32, copy=False)
    return {
        "base_features": base_features,
        "features": features,
        "labels": labels,
        "probabilities": probabilities,
        "outlier_scores": outlier_scores,
        "median": median,
        "scale": scale,
        "effective_min_cluster_size": effective_min_cluster_size,
    }


def label_color(label):
    if int(label) < 0:
        return np.array([0.45, 0.45, 0.45, 0.65])
    return np.asarray(plt.get_cmap("tab20")(int(label) % 20))


def stratified_plot_indices(labels, maximum):
    if len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    groups = np.unique(labels)
    per_group = max(1, maximum // len(groups))
    chosen = []
    rng = np.random.RandomState(24601)
    for label in groups:
        indices = np.flatnonzero(labels == label)
        take = min(len(indices), per_group)
        chosen.append(rng.choice(indices, size=take, replace=False))
    result = np.concatenate(chosen)
    if len(result) < maximum:
        remaining = np.setdiff1d(np.arange(len(labels)), result, assume_unique=False)
        take = min(len(remaining), maximum - len(result))
        if take:
            result = np.concatenate((result, rng.choice(remaining, size=take, replace=False)))
    return result[:maximum]


def crop_bounds(annotation, image_shape, margin=8):
    x, y, width, height = [int(round(value)) for value in annotation["bbox"]]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1 = min(image_shape[1], x + width + margin)
    y1 = min(image_shape[0], y + height + margin)
    return x0, y0, x1, y1


def cluster_statistics(camera_xyz, labels):
    records = []
    for label in np.unique(labels):
        selected = labels == label
        depths = camera_xyz[selected, 2]
        records.append({
            "label": int(label),
            "kind": "noise" if label < 0 else "cluster",
            "count": int(selected.sum()),
            "ratio": float(selected.mean()),
            "depth_min": float(depths.min()),
            "depth_median": float(np.median(depths)),
            "depth_max": float(depths.max()),
        })
    return records


def save_cluster_pages(image, masks, segment_data, cluster_data, pixel_xy,
                       output_dir, rows_per_page, max_plot_points):
    for start in range(0, len(masks), rows_per_page):
        end = min(start + rows_per_page, len(masks))
        fig = plt.figure(figsize=(16, max(3.2, (end - start) * 2.8)))
        grid = fig.add_gridspec(
            end - start, 3, width_ratios=[1.0, 1.25, 1.5], hspace=0.62, wspace=0.24
        )
        for row, index in enumerate(range(start, end)):
            annotation = masks[index]
            ids = segment_data[index]["ids"]
            xyz = segment_data[index]["camera_xyz"]
            labels = cluster_data[index]["labels"]
            x0, y0, x1, y1 = crop_bounds(annotation, image.shape)
            crop = base.masked_crop(image, annotation, base.mask_color(index))

            image_ax = fig.add_subplot(grid[row, 0])
            image_ax.imshow(crop)
            image_ax.set_title(
                "Segment {:03d} | {:,} px".format(index, int(annotation["area"])),
                fontsize=10,
            )
            image_ax.axis("off")

            overlay_ax = fig.add_subplot(grid[row, 1])
            overlay_ax.imshow(crop)
            if len(ids):
                shown = stratified_plot_indices(labels, max_plot_points)
                points = pixel_xy[ids[shown]]
                colors = np.stack([label_color(label) for label in labels[shown]])
                overlay_ax.scatter(
                    points[:, 0] - x0, points[:, 1] - y0,
                    c=colors, s=3, linewidths=0,
                )
            overlay_ax.set_xlim(0, max(1, x1 - x0))
            overlay_ax.set_ylim(max(1, y1 - y0), 0)
            overlay_ax.set_title("Projected Gaussian clusters", fontsize=10)
            overlay_ax.axis("off")

            feature_ax = fig.add_subplot(grid[row, 2])
            if len(ids):
                shown = stratified_plot_indices(labels, max_plot_points)
                feature = cluster_data[index]["base_features"][shown]
                colors = np.stack([label_color(label) for label in labels[shown]])
                feature_ax.scatter(feature[:, 2], feature[:, 0], c=colors,
                                   s=4, linewidths=0)
                feature_ax.set_xlabel("camera depth z")
                feature_ax.set_ylabel("x / z")
                feature_ax.grid(alpha=0.2)
                cluster_count = int(np.sum(np.unique(labels) >= 0))
                noise_count = int(np.sum(labels < 0))
                feature_ax.set_title(
                    "{:,} Gaussians | {} clusters | {:,} noise".format(
                        len(ids), cluster_count, noise_count
                    ), fontsize=10,
                )
            else:
                feature_ax.text(0.5, 0.5, "No contributing Gaussians",
                                ha="center", va="center")
                feature_ax.set_axis_off()
        page = start // rows_per_page + 1
        fig.suptitle(
            "SAM-segment Gaussian clusters: robust [x/z, y/z, beta*z] + HDBSCAN",
            fontsize=15, y=0.998,
        )
        fig.savefig(
            str(output_dir / "segment_clusters_{:03d}.png".format(page)),
            dpi=170, bbox_inches="tight",
        )
        plt.close(fig)


def save_results(masks, segment_data, cluster_data, output_dir, metadata):
    records, arrays = [], {}
    for index, (annotation, data, clustered) in enumerate(
            zip(masks, segment_data, cluster_data)):
        labels = clustered["labels"]
        records.append({
            "segment_id": index,
            "pixel_area": int(annotation["area"]),
            "bbox_xywh": [float(value) for value in annotation["bbox"]],
            "num_gaussians": int(len(data["ids"])),
            "num_clusters": int(np.sum(np.unique(labels) >= 0)),
            "num_noise": int(np.sum(labels < 0)),
            "effective_min_cluster_size": int(clustered["effective_min_cluster_size"]),
            "robust_center_x_over_z_y_over_z_z": clustered["median"].tolist(),
            "robust_scale_iqr_x_over_z_y_over_z_z": clustered["scale"].tolist(),
            "clusters": cluster_statistics(data["camera_xyz"], labels) if len(labels) else [],
        })
        prefix = "segment_{:03d}_".format(index)
        arrays[prefix + "gaussian_id"] = data["ids"].astype(np.int64)
        arrays[prefix + "camera_xyz"] = data["camera_xyz"].astype(np.float32)
        arrays[prefix + "base_feature_x_over_z_y_over_z_z"] = clustered[
            "base_features"
        ].astype(np.float32)
        arrays[prefix + "scaled_feature"] = clustered["features"].astype(np.float32)
        arrays[prefix + "cluster_label"] = labels.astype(np.int32)
        arrays[prefix + "cluster_probability"] = clustered["probabilities"]
        arrays[prefix + "outlier_score"] = clustered["outlier_scores"]
    report = dict(metadata)
    report["segments"] = records
    with open(str(output_dir / "clusters.json"), "w") as file:
        json.dump(report, file, indent=2)
    np.savez_compressed(str(output_dir / "segment_clusters.npz"), **arrays)


def run(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("The 3DGS rasterizer requires a CUDA GPU")
    import_hdbscan()
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    split, camera, camera_record = base.find_camera_by_json_id(
        scene, dataset.model_path, args.camera_id
    )
    print("Matched cameras.json ID {} to {} camera: {}".format(
        args.camera_id, split, camera.image_name
    ))
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    with torch.no_grad():
        rendered = render(
            camera, gaussians, pipeline, background,
            separate_sh=base.SEPARATE_SH,
            use_trained_exp=dataset.train_test_exp,
        )["render"]
    rendered_rgb = base.tensor_to_rgb_uint8(rendered)

    checkpoint = base.resolve_sam_checkpoint(
        getattr(args, "sam_checkpoint", None), dataset.model_path, args.sam_model_type
    )
    sam_type = base.infer_sam_model_type(checkpoint, args.sam_model_type)
    print("Running SAM {} from {}".format(sam_type, checkpoint))
    masks = base.generate_sam_masks(rendered_rgb, checkpoint, sam_type, args)
    if not masks:
        raise RuntimeError("SAM produced no masks after filtering")
    print("SAM produced {} segments".format(len(masks)))

    print("Tracing rasterizer contributors for each segment")
    alpha_image, opacity = base.render_contributor_alpha(camera, gaussians, pipeline)
    all_camera_xyz = gaussian_camera_xyz(camera, gaussians)
    segment_data = base.collect_segment_depths(
        alpha_image, opacity, masks, all_camera_xyz[:, 2]
    )
    cluster_data = []
    for index, data in enumerate(segment_data):
        data["camera_xyz"] = all_camera_xyz[data["ids"]]
        clustered = cluster_segment(data["camera_xyz"], args)
        cluster_data.append(clustered)
        labels = clustered["labels"]
        print("  clustered segment {:03d}: {} clusters, {:,} noise".format(
            index, int(np.sum(np.unique(labels) >= 0)), int(np.sum(labels < 0))
        ))

    default_folder = "camera_{:05d}_{}".format(
        args.camera_id, Path(str(camera.image_name)).stem
    )
    custom_output = getattr(args, "output_dir", None)
    output_dir = (
        Path(custom_output) if custom_output else
        Path(dataset.model_path) / "sam_boxplot" / default_folder / "clusters"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered_rgb).save(str(output_dir / "render.png"))
    base.save_overlay(rendered_rgb, masks, output_dir / "segments_overlay.png")
    pixel_xy = gaussian_pixel_xy(camera, gaussians)
    save_cluster_pages(
        rendered_rgb, masks, segment_data, cluster_data, pixel_xy,
        output_dir, args.rows_per_page, args.max_plot_points,
    )
    save_results(masks, segment_data, cluster_data, output_dir, {
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "camera_id": int(args.camera_id),
        "cameras_json_record": camera_record,
        "camera_split": split,
        "image_name": str(camera.image_name),
        "sam_checkpoint": str(checkpoint),
        "sam_model_type": sam_type,
        "contributor_definition": (
            "nonzero opacity gradient of masked accumulated-alpha render; follows "
            "rasterizer alpha cutoff and transmittance termination"
        ),
        "feature_definition": (
            "camera-space [x/z, y/z, z], independently median/IQR scaled, then "
            "the scaled z coordinate multiplied by depth_weight"
        ),
        "depth_weight": args.depth_weight,
        "hdbscan": {
            "min_cluster_size": args.min_cluster_size,
            "min_cluster_fraction": args.min_cluster_fraction,
            "min_samples": args.min_samples,
            "cluster_selection_method": args.cluster_selection_method,
            "cluster_selection_epsilon": args.cluster_selection_epsilon,
        },
    })
    print("Saved segment cluster visualizations to {}".format(output_dir))
    return output_dir


def build_parser():
    parser = ArgumentParser(
        description="HDBSCAN clustering of contributing 3DGS Gaussians per SAM segment"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--camera_id", "-c", required=True, type=int,
                        help="Camera ID from <model_path>/cameras.json")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_model_type", choices=("auto", "vit_h", "vit_l", "vit_b"),
                        default="auto")
    parser.add_argument("--sam_device", type=str, default="cuda")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--min_mask_area", type=int, default=0)
    parser.add_argument("--max_segments", type=int, default=0,
                        help="Largest masks to keep; 0 keeps all")
    parser.add_argument("--depth_weight", "--beta", type=float, default=2.0,
                        help="Weight applied to robust-scaled camera depth")
    parser.add_argument("--min_cluster_size", type=int, default=30)
    parser.add_argument("--min_cluster_fraction", type=float, default=0.005,
                        help="Per-segment minimum cluster fraction")
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--cluster_selection_method", choices=("eom", "leaf"),
                        default="eom")
    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.0)
    parser.add_argument("--hdbscan_jobs", type=int, default=-1)
    parser.add_argument("--rows_per_page", type=int, default=8)
    parser.add_argument("--max_plot_points", type=int, default=5000)
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
    if parsed.max_plot_points <= 0:
        arg_parser.error("--max_plot_points must be positive")
    if parsed.depth_weight <= 0:
        arg_parser.error("--depth_weight/--beta must be positive")
    if parsed.min_cluster_size < 2:
        arg_parser.error("--min_cluster_size must be at least 2")
    if not 0.0 <= parsed.min_cluster_fraction <= 1.0:
        arg_parser.error("--min_cluster_fraction must be in [0, 1]")
    safe_state(parsed.quiet)
    run(parsed, model_params.extract(parsed), pipeline_params.extract(parsed))
