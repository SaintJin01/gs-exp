"""Export background RGB-mean L1 loss maps at fixed checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

LOSSMAP_ITERATIONS = (10000, 15000, 20000)

@torch.no_grad()
def save_background_lossmaps(model_path, iteration, cameras, render_image, update_masks=False, threshold=0.20):
    output = Path(model_path) / "point_cloud" / "bgaussians" / "lossmap" / str(iteration)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, camera in enumerate(cameras):
        rendered = render_image(camera)
        target = camera.original_image.to(rendered.device)
        source_valid = (camera.alpha_mask.to(rendered.device) > 0.5).squeeze(0)
        active_before = source_valid & (camera.background_loss_mask.to(rendered.device) > 0.5).squeeze(0)
        error = (rendered - target).abs().mean(dim=0)
        if update_masks and active_before.any():
            keep = source_valid & (error < threshold)
            camera.background_loss_mask = keep.unsqueeze(0).to(camera.background_loss_mask.device, camera.background_loss_mask.dtype)
        active_after = source_valid & (camera.background_loss_mask.to(rendered.device) > 0.5).squeeze(0)
        values = error.masked_fill(~source_valid, float("nan")).cpu().numpy()
        source_mask = source_valid.cpu().numpy()
        active_mask = active_after.cpu().numpy()
        name = "%04d_%s" % (index, Path(camera.image_name).name)
        np.save(output / (name + ".npy"), values)
        gray = np.rint(np.clip(np.nan_to_num(values, nan=0.0), 0, 1) * 255).astype(np.uint8)
        rgba = np.stack((gray, gray, gray, source_mask.astype(np.uint8) * 255), axis=-1)
        Image.fromarray(rgba, "RGBA").save(output / (name + ".png"))
        Image.fromarray(active_mask.astype(np.uint8) * 255, "L").save(output / (name + "_active.png"))
        records.append({"image": camera.image_name, "file": name,
                        "source_valid_pixels": int(source_mask.sum()),
                        "active_pixels_before": int(active_before.sum().item()),
                        "active_pixels_after": int(active_mask.sum()),
                        "rejected_pixels": int(source_mask.sum() - active_mask.sum()),
                        "threshold": threshold if update_masks else None,
                        "mean_l1": float(values[source_mask].mean()) if source_mask.any() else None})
    with (output / "summary.json").open("w", encoding="utf8") as stream:
        json.dump({"iteration": iteration, "metric": "mean(abs(render - GT), RGB)",
                   "mask": "alpha > 0.5; invalid NPY pixels are NaN", "mask_updated": update_masks,
                   "absolute_l1_threshold": threshold if update_masks else None,
                   "png_scale": [0.0, 1.0], "images": records}, stream, indent=2)
    print("[Background %d] Loss maps saved to %s" % (iteration, output))
