#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    root = args.source.resolve()
    labels_dir = root / "slic" / "labels"
    source_sky_dir = root / "sky"
    output_dir = root / "slic" / "sky"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    label_paths = sorted(labels_dir.glob("*.npy"))
    for index, label_path in enumerate(label_paths):
        sky_path = source_sky_dir / (label_path.stem + ".png")
        if not sky_path.is_file():
            raise FileNotFoundError(sky_path)
        labels = np.load(label_path).astype(np.int64, copy=False)
        sky_image = Image.open(sky_path).convert("RGBA")
        sky = np.asarray(sky_image)[..., 3] > 127
        if sky.shape != labels.shape:
            sky = np.asarray(Image.fromarray(sky.astype(np.uint8) * 255).resize((labels.shape[1], labels.shape[0]), Image.Resampling.NEAREST)) > 127
        count = np.bincount(labels.ravel())
        sky_count = np.bincount(labels[sky], minlength=len(count))
        fraction = np.divide(sky_count, count, out=np.zeros_like(sky_count, dtype=np.float64), where=count > 0)
        sky_labels = np.flatnonzero((count > 0) & (fraction >= args.threshold))
        lookup = np.zeros(len(count), dtype=bool)
        lookup[sky_labels] = True
        output = lookup[labels]
        Image.fromarray(output.astype(np.uint8) * 255, "L").save(output_dir / (label_path.stem + ".png"), compress_level=3)
        records.append({"image": label_path.stem + ".png", "superpixels": int(len(count)), "sky_superpixels": int(len(sky_labels)), "sky_pixels": int(output.sum()), "sky_superpixel_ids": sky_labels.tolist()})
        if (index + 1) % 25 == 0 or index + 1 == len(label_paths):
            print("processed {}/{}".format(index + 1, len(label_paths)), flush=True)
    summary = {"definition": "A SLIC superpixel is sky when at least 90% of all its pixels are sky in the source sky alpha mask.", "threshold": args.threshold, "images": len(records), "total_sky_superpixels": sum(record["sky_superpixels"] for record in records), "records": records}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

if __name__ == "__main__":
    main()
