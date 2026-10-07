"""Export background RGB-mean L1 loss maps at fixed checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

LOSSMAP_ITERATIONS = (5000, 10000, 15000, 20000)

def load_labels(path, shape):
    labels = np.load(path)
    if labels.ndim != 2:
        raise ValueError("SLIC labels must be HxW: {}".format(path))
    if labels.shape != shape:
        labels = np.asarray(Image.fromarray(labels.astype(np.int32), "I").resize((shape[1], shape[0]), Image.Resampling.NEAREST), dtype=np.int64)
    return labels.astype(np.int64, copy=False)

@torch.no_grad()
def save_background_lossmaps(model_path, iteration, cameras, render_image, slic_dir, sky_superpixel_dir, update_masks=False, threshold=0.20, superpixel_fraction=0.05, stage_name=None):
    output = Path(model_path) / "point_cloud" / "bgaussians" / "lossmap"
    if stage_name is not None:
        output = output / stage_name
    output = output / str(iteration)
    output.mkdir(parents=True, exist_ok=True)
    preview = output / "preview"; preview.mkdir(exist_ok=True)
    records = []
    for index, camera in enumerate(cameras):
        rendered = render_image(camera)
        target = camera.original_image.to(rendered.device)
        source_valid = (camera.alpha_mask.to(rendered.device) > 0.5).squeeze(0)
        active_before = source_valid & (camera.background_loss_mask.to(rendered.device) > 0.5).squeeze(0)
        error = (rendered - target).abs().mean(dim=0)
        rejected_segments = np.empty(0, dtype=np.int64)
        protected_sky_segments = np.empty(0, dtype=np.int64)
        if update_masks and active_before.any():
            label_path = Path(slic_dir) / (Path(camera.image_name).stem + ".npy")
            if not label_path.is_file():
                raise FileNotFoundError("SLIC labels not found: {}".format(label_path))
            labels = load_labels(label_path, tuple(error.shape))
            labels_cuda = torch.from_numpy(labels).to(error.device)
            active_np = active_before.cpu().numpy(); high_np = (active_before & (error >= threshold)).cpu().numpy()
            active_count = np.bincount(labels[active_np]); high_count = np.bincount(labels[high_np], minlength=len(active_count))
            fraction = np.divide(high_count, active_count, out=np.zeros_like(high_count, dtype=np.float64), where=active_count > 0)
            rejected_segments = np.flatnonzero((active_count > 0) & (fraction >= superpixel_fraction))
            sky_superpixel_path = Path(sky_superpixel_dir) / (Path(camera.image_name).stem + ".png")
            if not sky_superpixel_path.is_file():
                raise FileNotFoundError("Sky superpixel mask not found: {}".format(sky_superpixel_path))
            sky_superpixel_mask = np.asarray(Image.open(sky_superpixel_path).convert("L"))
            if sky_superpixel_mask.shape != labels.shape:
                sky_superpixel_mask = np.asarray(Image.fromarray(sky_superpixel_mask).resize((labels.shape[1], labels.shape[0]), Image.Resampling.NEAREST))
            protected_sky_segments = np.unique(labels[sky_superpixel_mask > 127])
            rejected_segments = np.setdiff1d(rejected_segments, protected_sky_segments, assume_unique=True)
            if len(rejected_segments):
                reject_lookup = torch.zeros(int(labels_cuda.max().item()) + 1, dtype=torch.bool, device=error.device)
                reject_lookup[torch.from_numpy(rejected_segments).to(error.device)] = True
                keep = active_before & (~reject_lookup[labels_cuda])
                camera.background_loss_mask = keep.unsqueeze(0).to(camera.background_loss_mask.device, camera.background_loss_mask.dtype)
        active_after = source_valid & (camera.background_loss_mask.to(rendered.device) > 0.5).squeeze(0)
        values = error.masked_fill(~active_after, float("nan")).cpu().numpy()
        source_mask = source_valid.cpu().numpy()
        active_mask = active_after.cpu().numpy()
        name = "%04d_%s" % (index, Path(camera.image_name).name)
        np.save(output / (name + ".npy"), values)
        gray = np.rint(np.clip(np.nan_to_num(values, nan=0.0), 0, 1) * 255).astype(np.uint8)
        rgba = np.stack((gray, gray, gray, active_mask.astype(np.uint8) * 255), axis=-1)
        Image.fromarray(rgba, "RGBA").save(output / (name + ".png"))
        Image.fromarray(active_mask.astype(np.uint8) * 255, "L").save(output / (name + "_active.png"))
        preview_rgb = np.rint(np.clip(target.permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        preview_rgb[~active_mask] = 0
        Image.fromarray(preview_rgb, "RGB").save(preview / (Path(camera.image_name).stem + ".png"), compress_level=3)
        records.append({"image": camera.image_name, "file": name,
                        "source_valid_pixels": int(source_mask.sum()),
                        "active_pixels_before": int(active_before.sum().item()),
                        "active_pixels_after": int(active_mask.sum()),
                        "rejected_pixels": int(source_mask.sum() - active_mask.sum()),
                        "threshold": threshold if update_masks else None,
                        "superpixel_loss_fraction": superpixel_fraction if update_masks else None,
                        "protected_sky_superpixels": int(len(protected_sky_segments)),
                        "newly_rejected_superpixels": int(len(rejected_segments)),
                        "newly_rejected_superpixel_ids": rejected_segments.tolist(),
                        "mean_l1": float(values[active_mask].mean()) if active_mask.any() else None})
    with (output / "summary.json").open("w", encoding="utf8") as stream:
        json.dump({"stage": stage_name, "iteration": iteration, "metric": "mean(abs(render - GT), RGB)",
                   "mask": "alpha > 0.5 AND cumulative background_loss_mask after this update; excluded NPY pixels are NaN", "mask_updated": update_masks,
                   "absolute_l1_threshold": threshold if update_masks else None,
                   "sky_protection": "SLIC superpixels with >=90% sky pixels are never rejected",
                   "superpixel_rejection_rule": "reject entire SLIC superpixel when >= {:.2%} of its active pixels have loss >= threshold".format(superpixel_fraction),
                   "png_scale": [0.0, 1.0], "images": records}, stream, indent=2)
    print("[Background %d] Loss maps saved to %s" % (iteration, output))
