#!/usr/bin/env python3
"""Save near-side depth outlier Gaussians over all cameras and SAM segments."""

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
from utils.general_utils import safe_state  # noqa: E402


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


def select_nearer_than_lower_whisker(segment_data):
    """Select depths left of Matplotlib's default lower box-plot whisker."""
    selected = []
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
        near_ids = ids[depths < lower_whisker]
        selected.append(near_ids)
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
    if not selected:
        return np.empty(0, dtype=np.int64), statistics
    return np.unique(np.concatenate(selected)), statistics


def source_ply_path(model_path, iteration):
    return Path(model_path) / "point_cloud" / "iteration_{}".format(iteration) / "point_cloud.ply"


def write_selected_ply(source_path, selected_mask, output_path):
    ply = PlyData.read(str(source_path))
    vertices = ply["vertex"].data
    if len(vertices) != len(selected_mask):
        raise ValueError(
            "PLY/Gaussian count mismatch: {} vertices versus {} model Gaussians".format(
                len(vertices), len(selected_mask)
            )
        )
    selected_vertices = vertices[selected_mask]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = PlyData(
        [PlyElement.describe(selected_vertices, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    output.write(str(output_path))


def run(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("The 3DGS rasterizer requires a CUDA GPU")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras_path = Path(dataset.model_path) / "cameras.json"
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
    camera_statistics = []
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    for camera_index, record in enumerate(camera_records):
        camera_id = int(record["id"])
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
                use_trained_exp=dataset.train_test_exp,
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
        selected_camera, segment_statistics = select_nearer_than_lower_whisker(segment_data)
        selected_global[selected_camera] = True
        camera_statistics.append({
            "camera_id": camera_id,
            "image_name": str(camera.image_name),
            "split": split,
            "num_segments": int(len(masks)),
            "num_selected_unique": int(len(selected_camera)),
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
    source_path = source_ply_path(dataset.model_path, scene.loaded_iter)
    write_selected_ply(source_path, selected_global, output_path)

    report = {
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "source_ply": str(source_path),
        "output_ply": str(output_path),
        "selection": (
            "union across cameras and independent SAM masks of unique contributor "
            "Gaussians whose positive camera-space center depth is below that mask's "
            "lower box-plot whisker (near-side outliers, whis=1.5 IQR)"
        ),
        "num_gaussians": num_gaussians,
        "num_selected": int(selected_global.sum()),
        "selected_ratio": float(selected_global.mean()),
        "num_cameras": len(camera_records),
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
    print("Report written to {}".format(report_path))
    return output_path


def build_parser():
    parser = ArgumentParser(
        description=(
            "Save the union of near-side lower-whisker outlier Gaussians over all cameras"
        )
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
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
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


if __name__ == "__main__":
    arg_parser, model_params, pipeline_params = build_parser()
    parsed = get_combined_args(arg_parser)
    if not getattr(parsed, "model_path", None):
        arg_parser.error("--model_path/-m is required")
    safe_state(parsed.quiet)
    run(parsed, model_params.extract(parsed), pipeline_params.extract(parsed))
