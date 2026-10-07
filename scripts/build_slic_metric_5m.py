#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Mask SLIC superpixels whose pixels are at least 50% valid metric depth <= 5m.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--depth", type=float, default=5.0)
    parser.add_argument("--ratio", type=float, default=0.50)
    args = parser.parse_args()
    source = args.source.resolve()
    rgb_dir = source / "compensation" / "images"
    metric_dir = source / "depths_metric"
    labels_dir = source / "slic" / "labels"
    output = source / "compensation" / "slic_based" / "5m_masked"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("output must be empty: {}".format(output))
    masks_dir = output / "masks"
    images_dir = output / "images"
    preview_dir = output / "preview"
    masks_dir.mkdir(parents=True)
    images_dir.mkdir()
    preview_dir.mkdir()
    names = sorted(path.name for path in rgb_dir.glob("*.png"))
    rows = []
    for index, name in enumerate(names):
        stem = Path(name).stem
        rgb = np.asarray(Image.open(rgb_dir / name).convert("RGB"))
        metric = np.load(metric_dir / (stem + ".npy"))
        labels = np.load(labels_dir / (stem + ".npy")).astype(np.int64, copy=False)
        if metric.shape != labels.shape or metric.shape != rgb.shape[:2]:
            raise ValueError("shape mismatch: {}".format(name))
        label_count = int(labels.max()) + 1
        segment_size = np.bincount(labels.ravel(), minlength=label_count)
        near = np.isfinite(metric) & (metric > 0) & (metric <= args.depth)
        near_count = np.bincount(labels[near], minlength=label_count)
        reject_segment = (near_count / np.maximum(segment_size, 1)) >= args.ratio
        rejected = reject_segment[labels]
        keep = ~rejected
        Image.fromarray(keep.astype(np.uint8) * 255, "L").save(masks_dir / name, compress_level=3)
        masked = rgb.copy()
        masked[rejected] = 0
        Image.fromarray(masked, "RGB").save(images_dir / name, compress_level=3)
        if index % 10 == 0:
            Image.fromarray(masked, "RGB").save(preview_dir / name, compress_level=3)
        rows.append({"image": name, "metric_near_pixels": int(near.sum()), "removed_pixels": int(rejected.sum()), "kept_pixels": int(keep.sum()), "rejected_superpixels": int(reject_segment.sum()), "total_superpixels": int(np.count_nonzero(segment_size))})
        if (index + 1) % 25 == 0 or index + 1 == len(names):
            print("processed {}/{}".format(index + 1, len(names)), flush=True)
    with (output / "image_statistics.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    total_pixels = sum(row["removed_pixels"] + row["kept_pixels"] for row in rows)
    manifest = {"source": str(source), "rgb_source": str(rgb_dir), "metric_depth_source": str(metric_dir), "slic_source": str(labels_dir), "image_count": len(rows), "definition": "A complete SLIC superpixel is masked when at least 50% of all its pixels have finite positive estimated metric depth <= 5m.", "metric_depth_threshold_m": args.depth, "superpixel_rejection_ratio": args.ratio, "number_based_image_exclusion": False, "saturation_used": False, "relative_depth_used": False, "removed_pixels": sum(row["removed_pixels"] for row in rows), "kept_pixels": sum(row["kept_pixels"] for row in rows), "kept_fraction": sum(row["kept_pixels"] for row in rows) / float(total_pixels), "sum_rejected_superpixels_per_image": sum(row["rejected_superpixels"] for row in rows)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")

if __name__ == "__main__":
    main()
