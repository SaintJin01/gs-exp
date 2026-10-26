#!/usr/bin/env python3
"""Prepare coarse single-membership SAM masks and automatic previews."""

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

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
import depth_boxplot_in_segment as base  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
import save_q3_gaussians_all_cameras as q3  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from train_sam_features import feature_rasterizer


def retain_overlapping_masks(masks, height, width, min_segment_pixels):
    """Keep SAM's overlapping masks and pack their pixels for compact storage."""
    retained = []
    records = []
    ordered = masks
    for annotation in ordered:
        mask = np.asarray(annotation["segmentation"], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                "SAM mask shape {} does not match image shape {}".format(
                    mask.shape, (height, width)
                )
            )
        area = int(mask.sum())
        if area < min_segment_pixels or area == height * width:
            continue
        retained.append(mask)
        records.append({
            "mask_index": len(retained) - 1,
            "original_area": int(annotation["area"]),
            "pixel_area": area,
            "bbox_xywh": [float(value) for value in annotation["bbox"]],
            "predicted_iou": float(annotation["predicted_iou"]),
            "stability_score": float(annotation["stability_score"]),
        })
    pixel_count = height * width
    if retained:
        mask_stack = np.stack(retained, axis=0)
        masks_packed = np.packbits(mask_stack.reshape(len(retained), pixel_count), axis=1)
        coverage_packed = np.bitwise_or.reduce(masks_packed, axis=0)
    else:
        mask_stack = np.empty((0, height, width), dtype=bool)
        packed_width = (pixel_count + 7) // 8
        masks_packed = np.empty((0, packed_width), dtype=np.uint8)
        coverage_packed = np.zeros(packed_width, dtype=np.uint8)
    return mask_stack, masks_packed, coverage_packed, records


def retain_single_masks(masks, height, width, minimum):
    """Largest mask wins: each covered pixel belongs to exactly one segment."""
    occupied = np.zeros((height, width), dtype=bool)
    annotations = []
    for item in sorted(masks, key=lambda item: int(np.asarray(item["segmentation"]).sum()),
                       reverse=True):
        mask = np.asarray(item["segmentation"], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError("SAM mask/image shape mismatch")
        remaining = mask & ~occupied
        if int(remaining.sum()) < minimum or int(remaining.sum()) == height * width:
            continue
        annotations.append(dict(item, segmentation=remaining))
        occupied |= remaining
    return retain_overlapping_masks(annotations, height, width, minimum)


@torch.no_grad()
def render_camera_depth(camera, gaussians, pipeline):
    """Legacy depth diagnostic; not used by single-mask preparation/training."""
    xyz = gaussians.get_xyz.detach()
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)
    z = (homogeneous @ camera.world_view_transform)[:, 2]
    rasterizer = feature_rasterizer(camera, gaussians, pipeline)
    image, _, _ = rasterizer(
        means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
        colors_precomp=z[:, None].expand(-1, 3).contiguous(),
        opacities=gaussians.get_opacity.detach(),
        scales=gaussians.get_scaling.detach(), rotations=gaussians.get_rotation.detach(),
        cov3D_precomp=None,
    )
    return image[0]


def resolve_original_image(directory, image_name):
    """Prefer the exact filename; COLMAP camera names may include an extension."""
    directory = Path(directory)
    name = Path(str(image_name))
    exact = directory / name
    if exact.is_file():
        return exact
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    matches = [p for p in (directory / name.parent).iterdir()
               if p.is_file() and p.suffix.lower() in image_extensions
               and p.stem == name.stem]
    if len(matches) != 1:
        raise ValueError(
            "Cannot uniquely locate original image {!r} in {}: {} matches ({})".format(
                str(image_name), directory, len(matches),
                ", ".join(str(p) for p in matches)))
    return matches[0]


def mask_preview(image, masks, output_path):
    overlay = image.astype(np.float32).copy()
    for index, mask in enumerate(masks):
        color = base.mask_color(index).astype(np.float32)
        overlay[mask] = 0.70 * overlay[mask] + 0.30 * color
    fig, ax = plt.subplots(figsize=(max(8, image.shape[1] / 140),
                                    max(5, image.shape[0] / 140)))
    ax.imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    ax.set_title("Coarse single-membership SAM segments | {} segments".format(len(masks)))
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def selected_cameras(scene, model_path, split, max_cameras):
    records = json.loads((Path(model_path) / "cameras.json").read_text())
    selected = []
    for record in records:
        camera_id = int(record["id"])
        camera_split, camera, matched_record = base.find_camera_by_json_id(
            scene, model_path, camera_id
        )
        if split != "all" and camera_split != split:
            continue
        selected.append((camera_id, camera_split, camera, matched_record))
        if max_cameras > 0 and len(selected) >= max_cameras:
            break
    return selected


def run(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("The 3DGS rasterizer and SAM require a CUDA GPU")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = selected_cameras(
        scene, dataset.model_path, args.split, args.max_cameras
    )
    if not cameras:
        raise RuntimeError("No cameras matched the requested split")

    checkpoint = base.resolve_sam_checkpoint(
        getattr(args, "sam_checkpoint", None), dataset.model_path, args.sam_model_type
    )
    sam_type = base.infer_sam_model_type(checkpoint, args.sam_model_type)
    print("Loading SAM {} from {}".format(sam_type, checkpoint))
    sam, generator = q3.build_sam_generator(checkpoint, sam_type, args)

    custom_output = getattr(args, "output_dir", None)
    output_dir = (
        Path(custom_output) if custom_output else
        Path(dataset.model_path) / "sam_feature_data"
    )
    labels_dir = output_dir / "labels"
    previews_dir = output_dir / "previews"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.save_previews:
        previews_dir.mkdir(parents=True, exist_ok=True)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    manifest_records = []
    for index, (camera_id, camera_split, camera, record) in enumerate(cameras):
        print("[{}/{}] camera ID {} | {} | {}".format(
            index + 1, len(cameras), camera_id, camera_split, camera.image_name
        ))
        if args.image_source == "original":
            directory = Path(dataset.source_path) / dataset.images
            image_path = resolve_original_image(directory, camera.image_name)
            image = np.asarray(Image.open(str(image_path)).convert("RGB"))
        else:
            with torch.no_grad():
                rendered = render(
                    camera, gaussians, pipeline, background,
                    separate_sh=base.SEPARATE_SH,
                    use_trained_exp=dataset.train_test_exp,
                )["render"]
            image = base.tensor_to_rgb_uint8(rendered)
        # Upstream extraction passes cv2.imread's BGR data directly to SAM.
        sam_image = image[:, :, ::-1].copy() if args.sam_color_order == "bgr" else image
        masks = generator.generate(sam_image)
        if args.max_segments > 0:
            masks = masks[:args.max_segments]
        mask_stack, masks_packed, coverage_packed, segment_records = retain_single_masks(
            masks, image.shape[0], image.shape[1],
            max(args.min_segment_pixels, int(np.ceil(image.shape[0] * image.shape[1] * args.min_segment_fraction)))
        )
        for mask_index, record_item in enumerate(segment_records):
            record_item["mask_index"] = mask_index
        file_name = "camera_{:05d}.npz".format(camera_id)
        np.savez_compressed(
            str(labels_dir / file_name),
            format_version=np.asarray(3, dtype=np.int32),
            masks_packed=masks_packed,
            coverage_packed=coverage_packed,
            height=np.asarray(image.shape[0], dtype=np.int32),
            width=np.asarray(image.shape[1], dtype=np.int32),
            mask_areas=np.asarray(
                [item["pixel_area"] for item in segment_records], dtype=np.int64
            ),
            camera_id=np.asarray(camera_id, dtype=np.int32),
            image_name=np.asarray(str(camera.image_name)),
        )
        if args.save_previews:
            Image.fromarray(image).save(
                str(previews_dir / "camera_{:05d}_render.png".format(camera_id))
            )
            mask_preview(
                image, mask_stack,
                previews_dir / "camera_{:05d}_segments.png".format(camera_id),
            )
        covered_pixels = int(np.count_nonzero(mask_stack.any(axis=0))) if len(mask_stack) else 0
        manifest_records.append({
            "camera_id": camera_id,
            "split": camera_split,
            "image_name": str(camera.image_name),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "file": str(Path("labels") / file_name),
            "num_sam_masks": int(len(masks)),
            "num_retained_masks": int(len(segment_records)),
            "covered_pixels": covered_pixels,
            "masks": segment_records,
            "cameras_json_record": record,
        })
        print("  {} SAM masks -> {} single-membership segments, {:,} covered pixels".format(
            len(masks), len(segment_records), covered_pixels
        ))

    del generator, sam
    torch.cuda.empty_cache()
    manifest = {
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "split": args.split,
        "mask_format": "single_membership_packbits_v3",
        "pair_definition": "Same segment positive; different segments negative; uncovered ignored",
        "image_source": args.image_source,
        "sam_color_order": args.sam_color_order,
        "min_segment_pixels": args.min_segment_pixels,
        "min_segment_fraction": args.min_segment_fraction,
        "overlap_resolution": "largest_mask_first",
        "sam": {
            "checkpoint": str(checkpoint),
            "model_type": sam_type,
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "min_mask_area": args.min_mask_area,
        },
        "cameras": manifest_records,
    }
    with open(str(output_dir / "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)
    print("Saved SAM feature-training data to {}".format(output_dir))
    return output_dir


def build_parser():
    parser = ArgumentParser(
        description="Prepare coarse single-membership SAM masks and previews"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--max_cameras", type=int, default=0)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_model_type", choices=("auto", "vit_h", "vit_l", "vit_b"),
                        default="vit_h")
    parser.add_argument("--image_source", choices=("original", "rendered"), default="original")
    parser.add_argument("--sam_color_order", choices=("bgr", "rgb"), default="bgr",
                        help="BGR reproduces SAGA's cv2.imread extraction behavior")
    parser.add_argument("--sam_device", type=str, default="cuda")
    parser.add_argument("--points_per_side", type=int, default=8)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--min_mask_area", type=int, default=1000)
    parser.add_argument("--max_segments", type=int, default=0)
    parser.add_argument("--min_segment_pixels", type=int, default=1)
    parser.add_argument("--min_segment_fraction", type=float, default=0.01)
    parser.add_argument("--save_previews", dest="save_previews", action="store_true")
    parser.add_argument("--no_previews", dest="save_previews", action="store_false")
    parser.set_defaults(save_previews=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


if __name__ == "__main__":
    arg_parser, model_params, pipeline_params = build_parser()
    parsed = get_combined_args(arg_parser)
    if not getattr(parsed, "model_path", None):
        arg_parser.error("--model_path/-m is required")
    if parsed.min_segment_pixels < 1:
        arg_parser.error("--min_segment_pixels must be positive")
    if not 0 <= parsed.min_segment_fraction < 1:
        arg_parser.error("--min_segment_fraction must be in [0,1)")
    safe_state(parsed.quiet)
    run(parsed, model_params.extract(parsed), pipeline_params.extract(parsed))
