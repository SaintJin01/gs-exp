#!/usr/bin/env python3
"""Plot the largest SAM segment's contributor depths for every 3DGS camera."""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def largest_mask(generator, image, min_mask_area):
    masks = generator.generate(image)
    masks = [mask for mask in masks if int(mask["area"]) >= min_mask_area]
    if not masks:
        return None, 0
    return max(masks, key=lambda mask: int(mask["area"])), len(masks)


def extract_masked_crop(image, annotation, margin=8):
    if annotation is None:
        return None
    x, y, width, height = [int(round(value)) for value in annotation["bbox"]]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1 = min(image.shape[1], x + width + margin)
    y1 = min(image.shape[0], y + height + margin)
    crop = image[y0:y1, x0:x1]
    mask = annotation["segmentation"][y0:y1, x0:x1]
    masked = np.zeros_like(crop)
    masked[mask] = crop[mask]
    return masked


def compact_camera_result(camera_id, image_name, split, annotation, num_masks, depths,
                          rendered_rgb=None):
    summary = segment_depth.depth_summary(depths)
    near_depths = depths[depths < summary["lower_whisker"]] if len(depths) else depths
    if len(near_depths) > 120:
        indices = np.linspace(0, len(near_depths) - 1, 120, dtype=np.int64)
        near_sample = np.sort(near_depths)[indices]
    else:
        near_sample = np.sort(near_depths)
    return {
        "camera_id": int(camera_id),
        "image_name": str(image_name),
        "split": split,
        "num_sam_masks": int(num_masks),
        "segment_area": int(annotation["area"]) if annotation is not None else 0,
        "segment_bbox_xywh": (
            [float(value) for value in annotation["bbox"]] if annotation is not None else None
        ),
        "predicted_iou": (
            float(annotation["predicted_iou"]) if annotation is not None else None
        ),
        "stability_score": (
            float(annotation["stability_score"]) if annotation is not None else None
        ),
        "depth": summary,
        "near_sample": near_sample.astype(np.float32),
        "segment_crop": extract_masked_crop(rendered_rgb, annotation)
                        if rendered_rgb is not None else None,
    }


def draw_camera_boxplot(ax, result):
    summary = result["depth"]
    if summary["count"] == 0:
        ax.text(0.5, 0.5, "No SAM mask or contributors", ha="center", va="center")
        ax.set_title("No SAM mask or contributors", fontsize=10)
        ax.set_axis_off()
        return

    depths_stats = [{
        "label": "",
        "q1": summary["q1"],
        "med": summary["median"],
        "q3": summary["q3"],
        "whislo": summary["lower_whisker"],
        "whishi": summary["upper_whisker"],
        "fliers": [],
    }]
    ax.bxp(
        depths_stats,
        vert=False,
        widths=0.46,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#4C9BE8", "alpha": 0.72},
        medianprops={"color": "#B00020", "linewidth": 2},
        whiskerprops={"color": "#222222", "linewidth": 1.2},
        capprops={"color": "#222222", "linewidth": 1.2},
    )
    lower_whisker = summary["lower_whisker"]
    ax.axvspan(0, lower_whisker, color="#D62728", alpha=0.08)
    ax.axvline(lower_whisker, color="#D62728", linestyle="--", linewidth=1.2)
    near_sample = result["near_sample"]
    if len(near_sample):
        jitter = np.linspace(0.80, 1.20, len(near_sample))
        ax.scatter(near_sample, jitter, s=8, color="#D62728", alpha=0.55, zorder=4)
    ax.set_xlim(0, summary["q3"] + 10.0)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("camera-space depth", fontsize=8)
    selected = summary["near_outlier_count"]
    selected_ratio = 100.0 * selected / summary["count"]
    ax.set_title(
        ("{:,} Gaussians | lower whisker {:.4g} | "
         "saved {:,} ({:.2f}%) | Q3 {:.4g}").format(
            summary["count"],
            lower_whisker,
            selected,
            selected_ratio,
            summary["q3"],
        ),
        fontsize=10,
    )


def draw_segment_image(ax, result):
    crop = result["segment_crop"]
    if crop is None or crop.size == 0:
        ax.text(0.5, 0.5, "No segment image", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.imshow(crop)
    ax.set_title(
        "camera {:03d} | {} | {:,} px".format(
            result["camera_id"], Path(result["image_name"]).name, result["segment_area"]
        ),
        fontsize=10,
    )
    ax.axis("off")


def save_page(results, output_dir, page_number):
    row_count = len(results)
    fig = plt.figure(
        figsize=(15, max(4.0, row_count * 2.8)), constrained_layout=True
    )
    grid = fig.add_gridspec(
        row_count, 2, width_ratios=[1.05, 2.9], hspace=0.70, wspace=0.16
    )
    for row, result in enumerate(results):
        image_ax = fig.add_subplot(grid[row, 0])
        plot_ax = fig.add_subplot(grid[row, 1])
        draw_segment_image(image_ax, result)
        draw_camera_boxplot(plot_ax, result)
    fig.suptitle("Largest SAM segment Gaussian depth distributions", fontsize=15)
    output_path = output_dir / "biggest_segment_boxplots_{:03d}.png".format(page_number)
    fig.savefig(str(output_path), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output_path))


def json_safe_result(result):
    cleaned = dict(result)
    cleaned.pop("near_sample", None)
    cleaned.pop("segment_crop", None)
    return cleaned


def run(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("The 3DGS rasterizer requires a CUDA GPU")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    camera_records = json.loads((Path(dataset.model_path) / "cameras.json").read_text())
    if args.max_cameras > 0:
        camera_records = camera_records[:args.max_cameras]

    checkpoint = segment_depth.resolve_sam_checkpoint(
        getattr(args, "sam_checkpoint", None), dataset.model_path, args.sam_model_type
    )
    sam_type = segment_depth.infer_sam_model_type(checkpoint, args.sam_model_type)
    print("Loading SAM {} from {}".format(sam_type, checkpoint))
    sam, mask_generator = build_sam_generator(checkpoint, sam_type, args)

    custom_output = getattr(args, "output_dir", None)
    output_dir = (
        Path(custom_output)
        if custom_output
        else Path(dataset.model_path) / "sam_boxplot" / "biggest_segment_all_cameras"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    page_results = []
    all_results = []
    page_number = 1

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
        annotation, num_masks = largest_mask(mask_generator, rendered_rgb, args.min_mask_area)

        if annotation is None:
            result = compact_camera_result(
                camera_id, camera.image_name, split, None, 0,
                np.empty(0, dtype=np.float32), rendered_rgb
            )
        else:
            print("SAM masks: {}; largest area: {:,} px".format(
                num_masks, int(annotation["area"])
            ))
            alpha_image, opacity = segment_depth.render_contributor_alpha(
                camera, gaussians, pipeline
            )
            segment_data = segment_depth.collect_segment_depths(
                alpha_image,
                opacity,
                [annotation],
                segment_depth.gaussian_camera_depths(camera, gaussians),
            )[0]
            result = compact_camera_result(
                camera_id,
                camera.image_name,
                split,
                annotation,
                num_masks,
                segment_data["depths"],
                rendered_rgb,
            )
            del alpha_image, opacity, segment_data

        page_results.append(result)
        all_results.append(json_safe_result(result))
        if len(page_results) == 8:
            save_page(page_results, output_dir, page_number)
            page_results = []
            page_number += 1
        del rendered_rgb, annotation
        torch.cuda.empty_cache()

    if page_results:
        save_page(page_results, output_dir, page_number)
    del mask_generator, sam
    torch.cuda.empty_cache()

    report = {
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "num_cameras": len(camera_records),
        "plots_per_image": 8,
        "segment_selection": "largest SAM mask by pixel area for each camera",
        "sam": {
            "checkpoint": str(checkpoint),
            "model_type": sam_type,
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "min_mask_area": args.min_mask_area,
        },
        "cameras": all_results,
    }
    report_path = output_dir / "biggest_segment_boxplots.json"
    with open(str(report_path), "w") as file:
        json.dump(report, file, indent=2)
    print("Report written to {}".format(report_path))
    return output_dir


def build_parser():
    parser = ArgumentParser(
        description="Plot the largest SAM segment's Gaussian depths for every camera"
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
    parser.add_argument(
        "--max_cameras", type=int, default=0,
        help="Process only the first N cameras for a test run; 0 processes all",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


if __name__ == "__main__":
    arg_parser, model_params, pipeline_params = build_parser()
    parsed = get_combined_args(arg_parser)
    if not getattr(parsed, "model_path", None):
        arg_parser.error("--model_path/-m is required")
    safe_state(parsed.quiet)
    run(parsed, model_params.extract(parsed), pipeline_params.extract(parsed))
