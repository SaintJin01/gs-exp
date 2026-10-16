#!/usr/bin/env python3
"""Save binary near outliers and continuous whisker distances over all views."""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
import depth_boxplot_in_segment as segment_depth  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.graphics_utils import focal2fov, getProjectionMatrix  # noqa: E402
from utils.system_utils import searchForMaxIteration  # noqa: E402


NEAR_OUTLIER_MARGIN = 3.0


def build_sam_generator(checkpoint, model_type, args):
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
    return sam, generator


def generate_masks(generator, image, args):
    masks = generator.generate(image)
    masks = [mask for mask in masks if int(mask["area"]) >= args.min_mask_area]
    masks.sort(key=lambda mask: int(mask["area"]), reverse=True)
    if args.max_segments > 0:
        masks = masks[:args.max_segments]
    return masks


def analyze_lower_whisker_distances(segment_data):
    """Return binary outliers and signed lower-whisker distances.

    The continuous value for one segment observation is
    ``lower_whisker - camera_space_depth``.  It is positive on the near side
    of the lower whisker and non-positive on the box-plot inlier side.
    """
    selected = []
    observed_ids = []
    observed_distances = []
    statistics = []
    for segment_id, data in enumerate(segment_data):
        ids = data["ids"]
        depths = data["depths"]
        if len(depths) == 0:
            statistics.append({
                "segment_id": segment_id,
                "num_contributors": 0,
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_fence": None,
                "lower_whisker": None,
                "num_selected": 0,
            })
            continue
        q1, q3 = [float(value) for value in np.percentile(depths, [25, 75])]
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        inlier_depths = depths[depths >= lower_fence]
        lower_whisker = float(inlier_depths.min())
        whisker_distances = lower_whisker - depths
        # Binary near outliers must be at least NEAR_OUTLIER_MARGIN closer than
        # the segment's lower whisker: depth <= lower_whisker - margin.
        near_ids = ids[whisker_distances >= NEAR_OUTLIER_MARGIN]
        selected.append(near_ids)
        observed_ids.append(ids)
        observed_distances.append(whisker_distances.astype(np.float64, copy=False))
        statistics.append({
            "segment_id": segment_id,
            "num_contributors": int(len(ids)),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "lower_whisker": lower_whisker,
            "num_selected": int(len(near_ids)),
        })
    selected_ids = (
        np.unique(np.concatenate(selected))
        if selected else np.empty(0, dtype=np.int64)
    )
    observation_ids = (
        np.concatenate(observed_ids).astype(np.int64, copy=False)
        if observed_ids else np.empty(0, dtype=np.int64)
    )
    observation_distances = (
        np.concatenate(observed_distances)
        if observed_distances else np.empty(0, dtype=np.float64)
    )
    return selected_ids, statistics, observation_ids, observation_distances


def average_observations_within_view(observation_ids, observation_values):
    """Average repeated segment observations so every camera has equal weight."""
    if len(observation_ids) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    unique_ids, inverse = np.unique(observation_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=observation_values)
    counts = np.bincount(inverse)
    return unique_ids, sums / counts


def source_ply_path(model_path, iteration):
    return Path(model_path) / "point_cloud" / "iteration_{}".format(iteration) / "point_cloud.ply"


def camera_from_json_record(record):
    """Build a render-only 3DGS camera from a cameras.json entry."""
    width, height = int(record["width"]), int(record["height"])
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.asarray(record["rotation"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(record["position"], dtype=np.float64)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    fov_x = focal2fov(float(record["fx"]), width)
    fov_y = focal2fov(float(record["fy"]), height)
    z_near, z_far = 0.01, 100.0
    world_view = torch.from_numpy(w2c).transpose(0, 1).cuda()
    projection = getProjectionMatrix(
        znear=z_near, zfar=z_far, fovX=fov_x, fovY=fov_y
    ).transpose(0, 1).cuda()
    camera = MiniCam(
        width, height, fov_y, fov_x, z_near, z_far,
        world_view, world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
    )
    camera.uid = int(record["id"])
    camera.colmap_id = int(record["id"])
    camera.image_name = str(record["img_name"])
    return camera


def append_vertex_properties(vertices, whisker_distance, view_count):
    """Copy a structured vertex array and append/update continuous properties."""
    property_values = {
        "whisker_distance": np.asarray(whisker_distance, dtype=np.float32),
        "whisker_view_count": np.asarray(view_count, dtype=np.uint32),
    }
    if any(len(values) != len(vertices) for values in property_values.values()):
        raise ValueError("Continuous property length does not match PLY vertex count")

    original_names = list(vertices.dtype.names or ())
    dtype_fields = []
    for name in original_names:
        dtype_fields.append((name, vertices.dtype.fields[name][0]))
    for name, values in property_values.items():
        if name not in original_names:
            dtype_fields.append((name, values.dtype))

    enriched = np.empty(vertices.shape, dtype=np.dtype(dtype_fields))
    for name in original_names:
        enriched[name] = vertices[name]
    for name, values in property_values.items():
        enriched[name] = values
    return enriched


def write_output_ply(source_path, output_path, whisker_distance, view_count,
                     selected_mask=None):
    ply = PlyData.read(str(source_path))
    vertices = ply["vertex"].data
    if len(vertices) != len(whisker_distance):
        raise ValueError(
            "PLY/Gaussian count mismatch: {} vertices versus {} model Gaussians".format(
                len(vertices), len(whisker_distance)
            )
        )
    enriched_vertices = append_vertex_properties(vertices, whisker_distance, view_count)
    if selected_mask is not None:
        enriched_vertices = enriched_vertices[selected_mask]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = PlyData(
        [PlyElement.describe(enriched_vertices, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    output.write(str(output_path))


def run(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("The 3DGS rasterizer requires a CUDA GPU")

    custom_cameras_path = getattr(args, "cameras_json", None)
    gaussians = GaussianModel(dataset.sh_degree)
    if custom_cameras_path:
        loaded_iteration = args.iteration
        if loaded_iteration == -1:
            loaded_iteration = searchForMaxIteration(
                str(Path(dataset.model_path) / "point_cloud")
            )
        gaussians.load_ply(
            str(source_ply_path(dataset.model_path, loaded_iteration)),
            dataset.train_test_exp,
        )
        scene = None
    else:
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        loaded_iteration = scene.loaded_iter
    cameras_path = (
        Path(custom_cameras_path)
        if custom_cameras_path else Path(dataset.model_path) / "cameras.json"
    )
    camera_records = json.loads(cameras_path.read_text())
    if args.max_cameras > 0:
        camera_records = camera_records[:args.max_cameras]

    checkpoint = segment_depth.resolve_sam_checkpoint(
        getattr(args, "sam_checkpoint", None), dataset.model_path, args.sam_model_type
    )
    sam_type = segment_depth.infer_sam_model_type(checkpoint, args.sam_model_type)
    print("Loading SAM {} from {}".format(sam_type, checkpoint))
    sam, mask_generator = build_sam_generator(checkpoint, sam_type, args)

    num_gaussians = int(gaussians.get_xyz.shape[0])
    selected_global = np.zeros(num_gaussians, dtype=bool)
    distance_sum = np.zeros(num_gaussians, dtype=np.float64)
    distance_view_count = np.zeros(num_gaussians, dtype=np.uint32)
    camera_statistics = []
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    for camera_index, record in enumerate(camera_records):
        camera_id = int(record["id"])
        if custom_cameras_path:
            split, camera = "interpolated", camera_from_json_record(record)
        else:
            split, camera, _ = segment_depth.find_camera_by_json_id(
                scene, dataset.model_path, camera_id
            )
        print(
            "\n[{}/{}] camera ID {} | {} | {}".format(
                camera_index + 1, len(camera_records), camera_id, split, camera.image_name
            )
        )
        with torch.no_grad():
            rendered = render(
                camera,
                gaussians,
                pipeline,
                background,
                separate_sh=segment_depth.SEPARATE_SH,
                use_trained_exp=(dataset.train_test_exp and not custom_cameras_path),
            )["render"]
        rendered_rgb = segment_depth.tensor_to_rgb_uint8(rendered)
        del rendered

        masks = generate_masks(mask_generator, rendered_rgb, args)
        print("SAM segments: {}".format(len(masks)))
        if not masks:
            camera_statistics.append({
                "camera_id": camera_id,
                "image_name": str(camera.image_name),
                "split": split,
                "num_segments": 0,
                "num_selected_unique": 0,
                "segments": [],
            })
            continue

        alpha_image, opacity = segment_depth.render_contributor_alpha(
            camera, gaussians, pipeline
        )
        segment_data = segment_depth.collect_segment_depths(
            alpha_image,
            opacity,
            masks,
            segment_depth.gaussian_camera_depths(camera, gaussians),
        )
        (selected_camera, segment_statistics, observation_ids,
         observation_distances) = analyze_lower_whisker_distances(segment_data)
        view_ids, view_distances = average_observations_within_view(
            observation_ids, observation_distances
        )
        distance_sum[view_ids] += view_distances
        distance_view_count[view_ids] += 1
        selected_global[selected_camera] = True
        camera_statistics.append({
            "camera_id": camera_id,
            "image_name": str(camera.image_name),
            "split": split,
            "num_segments": int(len(masks)),
            "num_selected_unique": int(len(selected_camera)),
            "num_observed_unique": int(len(view_ids)),
            "segments": segment_statistics,
        })
        print(
            "camera-selected: {:,}; global union: {:,}/{:,}".format(
                len(selected_camera), int(selected_global.sum()), num_gaussians
            )
        )
        del alpha_image, opacity, segment_data, masks, rendered_rgb
        torch.cuda.empty_cache()

    del mask_generator, sam
    torch.cuda.empty_cache()

    custom_output = getattr(args, "output", None)
    output_path = (
        Path(custom_output)
        if custom_output
        else Path(dataset.model_path) / "sam_boxplot" / "all_cameras_near_outliers" /
        "gaussians_nearer_than_lower_whisker.ply"
    )
    custom_continuous_output = getattr(args, "continuous_output", None)
    continuous_output_path = (
        Path(custom_continuous_output)
        if custom_continuous_output
        else (
            output_path.with_name(output_path.stem + "_continuous.ply")
            if custom_output
            else Path(dataset.model_path) / "sam_boxplot" /
            "all_cameras_near_outliers" / "gaussians_with_whisker_distance.ply"
        )
    )
    mean_whisker_distance = np.zeros(num_gaussians, dtype=np.float32)
    observed = distance_view_count > 0
    mean_whisker_distance[observed] = (
        distance_sum[observed] / distance_view_count[observed]
    ).astype(np.float32)
    source_path = source_ply_path(dataset.model_path, loaded_iteration)
    write_output_ply(
        source_path,
        output_path,
        mean_whisker_distance,
        distance_view_count,
        selected_mask=selected_global,
    )
    write_output_ply(
        source_path,
        continuous_output_path,
        mean_whisker_distance,
        distance_view_count,
    )

    report = {
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(loaded_iteration),
        "source_ply": str(source_path),
        "output_ply": str(output_path),
        "continuous_output_ply": str(continuous_output_path),
        "selection": (
            "union across cameras and independent SAM masks of unique contributor "
            "Gaussians whose positive camera-space center depth is at most that "
            "mask's lower box-plot whisker minus 3.0 (near-side outliers, "
            "whis=1.5 IQR)"
        ),
        "binary_outlier_margin": NEAR_OUTLIER_MARGIN,
        "num_gaussians": num_gaussians,
        "num_selected": int(selected_global.sum()),
        "selected_ratio": float(selected_global.mean()),
        "continuous_value": {
            "property": "whisker_distance",
            "definition": (
                "mean across camera views of (segment lower whisker - positive "
                "camera-space Gaussian center depth); repeated overlapping SAM "
                "segment observations are averaged within each view first"
            ),
            "positive_side": "nearer than the lower whisker on average",
            "view_count_property": "whisker_view_count",
            "num_observed_gaussians": int(observed.sum()),
            "min_observed": (
                float(mean_whisker_distance[observed].min()) if observed.any() else None
            ),
            "max_observed": (
                float(mean_whisker_distance[observed].max()) if observed.any() else None
            ),
            "mean_observed": (
                float(mean_whisker_distance[observed].mean()) if observed.any() else None
            ),
        },
        "num_cameras": len(camera_records),
        "cameras_json": str(cameras_path.resolve()),
        "sam": {
            "checkpoint": str(checkpoint),
            "model_type": sam_type,
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "min_mask_area": args.min_mask_area,
        },
        "cameras": camera_statistics,
    }
    report_path = output_path.with_suffix(".json")
    with open(str(report_path), "w") as file:
        json.dump(report, file, indent=2)
    print("\nSaved {:,}/{:,} Gaussians to {}".format(
        int(selected_global.sum()), num_gaussians, output_path
    ))
    print("Saved all {:,} Gaussians with continuous values to {}".format(
        num_gaussians, continuous_output_path
    ))
    print("Report written to {}".format(report_path))
    return output_path


def build_parser():
    parser = ArgumentParser(
        description=(
            "Save binary near outliers and per-Gaussian mean lower-whisker distances"
        )
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument(
        "--cameras_json", type=str, default=None,
        help="Optional render-only cameras.json, including interpolated cameras",
    )
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument(
        "--sam_model_type", choices=("auto", "vit_h", "vit_l", "vit_b"), default="auto"
    )
    parser.add_argument("--sam_device", type=str, default="cuda")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--min_mask_area", type=int, default=0)
    parser.add_argument("--max_segments", type=int, default=0)
    parser.add_argument(
        "--max_cameras", type=int, default=0,
        help="Process only the first N cameras for a test run; 0 processes all",
    )
    parser.add_argument("--output", type=str, default=None, help="Output PLY path")
    parser.add_argument(
        "--continuous_output", type=str, default=None,
        help="All-Gaussian PLY with whisker_distance and whisker_view_count properties",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


if __name__ == "__main__":
    arg_parser, model_params, pipeline_params = build_parser()
    parsed = get_combined_args(arg_parser)
    if not getattr(parsed, "model_path", None):
        arg_parser.error("--model_path/-m is required")
    safe_state(parsed.quiet)
    run(parsed, model_params.extract(parsed), pipeline_params.extract(parsed))
