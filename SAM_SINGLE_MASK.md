# Coarse single-mask feature pipeline

Default: 16D features, no scale gate, xyz kNN smoothing unchanged (K=16).
SAM returns a single non-overlapping segmentation: largest masks claim pixels first;
smaller masks retain only unclaimed pixels. Uncovered pixels are ignored, not background negatives.
Same-segment pairs are positive and different-segment pairs are negative.

Coarse defaults: points_per_side=8, crop_n_layers=0, min_mask_area=1000,
min_segment_fraction=0.01, pred_iou_thresh=0.88, stability_score_thresh=0.95.
The 1% cutoff applies to each retained mask, not each connected component.
Automatic previews are saved to sam_feature_data/previews/camera_XXXXX_segments.png
with the input image saved alongside. Disable with --no_previews.

```bash
python prepare_sam_pairs.py -m /home/sj/work/gaussian-splatting/output/train_eval
python train_sam_features.py -m /home/sj/work/gaussian-splatting/output/train_eval --iterations 10000
```

Input remains the original image by default (previous SAGA setup). Use
--image_source rendered to segment the 3DGS rendering instead.
For a quick preview, prepare with --max_cameras 1 --output_dir /tmp/sam_coarse_preview.

Existing overlapping masks and 32D checkpoints are incompatible with this new training format.
Regenerate masks and start fresh; no old checkpoint, Rex feature, or gate file is automatically deleted.
Cluster entrypoints use normalized exported features directly, including when a stale gate file exists.
Cluster --scale/--scale_gate_file options and gate loading code have been removed.
After training, copy the new latest.gsfeat to the Rex feature_file path only when it matches
the displayed PLY Gaussian count and ordering. Restart Rex to load the updated graph plugin.

## Geometry-aware HDBSCAN

By default, remaining noise is assigned to the nearest stable-cluster centroid
using Euclidean distance in the weighted 19D joint space (not feature-only cosine).
Existing non-noise labels are unchanged. Use --keep_noise to retain -1 at all levels.
Input for 16D learned features is 22D:
[feature_weight * L2(feature),
 xyz_weight * robust(xyz) / sqrt(3)].
Each weight defaults to 1. Geometry is median/IQR-scaled on the fitting sample;
constant or tiny-IQR axes use divisor 1. No normalization after concatenation.
Gaussian size/log-scale and rotation are not included. Scale gate is not used.
Ward hierarchy uses the same joint space.
Fitting and out-of-sample density prediction use the identical representation.

```bash
python cluster_gaussian_features.py -m /home/sj/work/gaussian-splatting/output/train_eval --feature_weight 1 --xyz_weight 1
```

With -m, geometry defaults to training_config.json source_iteration when applicable,
otherwise the latest point_cloud iteration. Override with --ply_file or --iteration.
With --feature_file alone, --ply_file is required. PLY/feature counts are checked;
they must also have identical Gaussian ordering. Set xyz_weight to 0
to disable geometry. Add --keep_noise for noise-preserving prediction.
The gscluster format remains Rex-compatible; no viewer rebuild is needed.

