"""Export background RGB-mean L1 loss maps at fixed checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

LOSSMAP_ITERATIONS = (3000, 5000)

@torch.no_grad()
def save_background_lossmaps(model_path, iteration, cameras, render_image):
    output = Path(model_path) / "point_cloud" / "bgaussians" / "lossmap" / str(iteration)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, camera in enumerate(cameras):
        rendered = render_image(camera)
        target = camera.original_image.to(rendered.device)
        valid = (camera.alpha_mask.to(rendered.device) > 0.5).squeeze(0)
        error = (rendered - target).abs().mean(dim=0)
        values = error.masked_fill(~valid, float("nan")).cpu().numpy()
        mask = valid.cpu().numpy()
        name = "%04d_%s" % (index, Path(camera.image_name).name)
        np.save(output / (name + ".npy"), values)
        gray = np.rint(np.clip(np.nan_to_num(values, nan=0.0), 0, 1) * 255).astype(np.uint8)
        rgba = np.stack((gray, gray, gray, mask.astype(np.uint8) * 255), axis=-1)
        Image.fromarray(rgba, "RGBA").save(output / (name + ".png"))
        records.append({"image": camera.image_name, "file": name,
                        "valid_pixels": int(mask.sum()),
                        "mean_l1": float(values[mask].mean()) if mask.any() else None})
    with (output / "summary.json").open("w", encoding="utf8") as stream:
        json.dump({"iteration": iteration, "metric": "mean(abs(render - GT), RGB)",
                   "mask": "alpha > 0.5; invalid NPY pixels are NaN",
                   "png_scale": [0.0, 1.0], "images": records}, stream, indent=2)
    print("[Background %d] Loss maps saved to %s" % (iteration, output))
