# GCISP: radiance + exposure controller + CRF

Branch: `floater-free-gcisp`. Enable with `--hdr_camera`; without it the vanilla LDR path remains available. ViT camera code and its architecture options are removed.

## Source and scope

Adapted from [nv-tlabs/ppisp](https://github.com/nv-tlabs/ppisp), pinned at `a247368047651db2e03e6f97a36e4f5bebbe5c5f`. Attribution and the Apache-2.0 license are in `third_party/ppisp/`. `scene/camera_gcisp.py` implements the retained PPISP controller, exposure and CRF in PyTorch autograd, avoiding a new CUDA extension build on the current Python 3.7 / PyTorch 1.12 environment.

Vignetting, color-correction homography, controller color head and their regularization parameters are completely absent. This is not the earlier image-to-image Transformer/CNN: the controller predicts one exposure scalar, and the parametric CRF produces final RGB.

## Forward path

```
Gaussian SH -> max(SH + 0.5, 0) (nonnegative unbounded radiance)
-> 3DGS alpha compositing -> radiance image
-> multiply by 2 ** exposure -> PPISP toe/shoulder/gamma CRF -> RGB
```

Radiance follows the official PPISP 3DGRUT SH path (nv-tlabs/3dgrut commit bb203b1fa990e044412396483d4dbc1f3164ae02): initialize DC SH as (normalized point-cloud RGB - 0.5) / C0, with higher-order coefficients zero. No inverse sRGB or inverse softplus is used. Evaluate max(SH + 0.5, 0), then alpha composite without an upper clamp before ISP. This uses gs-exp's rasterizer and does not imply calibrated physical light units.

As in PPISP, the CRF clamps its input to [0,1]. Before this step radiance and exposure-adjusted RGB are not upper-clamped. CRF parameters per RGB channel are toe=0.3+softplus(raw), shoulder=0.3+softplus(raw), gamma=0.1+softplus(raw), center=sigmoid(raw). Initialization is the identity curve. The original 1e-6 normalized-base gradient dead zone is preserved at black/saturation. Numerical guards avoid sigmoid endpoint division and overflow in unselected branches.

## Two-stage training

For the default 30,000 iterations and `--controller_activation_ratio 0.8`:

1. Iterations 1..23,999: optimize Gaussians, per-training-frame exposure and per-physical-camera CRF with L1+DSSIM plus PPISP exposure-mean and CRF-channel regularization. Exposure offsets are in stops. Physical cameras are grouped by COLMAP intrinsic camera ID, not image ID. Blender scenes use one camera. Frame names are sorted for stable checkpoint indexing.
2. Iterations 24,000..30,000: freeze Gaussians, frame exposure and CRF; stop SH-degree changes, densification, opacity reset and depth supervision. Train only exposure controllers using the same reconstruction loss against ground truth, following upstream PPISP's default controller-distillation routing. No additional teacher loss is introduced. Radiance is detached in this phase.

Each camera's controller is the original 1x1 Conv(3->16), MaxPool(3), ReLU, Conv(16->32), ReLU, Conv(32->64), adaptive 5x5 pool, three 128-wide ReLU MLP layers, and a scalar exposure head. The color head is removed. Radiance feature input is detached as in upstream. Optional exposure prior is supported by the module; current dataset integration supplies zero (no EXIF prior). Images smaller than 3 pixels are enlarged only for the controller's pooling operation.

Inference uses the controller after it has received updates. This readiness is saved per physical camera. Before controller training, inference uses zero exposure rather than random controller weights. Novel-view inference never looks up a training frame exposure. Unknown IDs in a multi-camera model raise an error; a single-camera model shares its one camera model across views.

ISP optimizer defaults: Adam lr=0.002, eps=1e-15; 500-step warmup from 0.01x then exponential decay with factor 0.01 over 30,000 decay steps. Controller Adam lr=0.001 is fixed. Optimizer groups are disjoint. CLI options are `--isp_lr`, `--controller_lr` and `--controller_activation_ratio`.

## Gaussian-DK gradient scaling

During Gaussian training only, screen-space gradients used by split/clone statistics are multiplied by:

```
w = min(1, (camera_space_depth / radius)^2)
weight = strength * w + (1-strength)
radius = gradient_scale_radius * scene.cameras_extent
strength = 1 before iteration 4000, 0.5 before 7000, then 0
```

This matches the intended Gaussian-DK backward schedule. XYZ, SH, opacity, exposure, CRF and controller optimizer gradients are not multiplied by this weight. The default radius multiplier is 1; use `--gradient_scale_radius`, `--gradient_scale_anneal_start`, `--gradient_scale_anneal_end` or `--disable_gradient_scaling` for ablations. No densification statistics are updated during controller-only training.

## Run

```bash
cd /home/sj/work/gs-exp
conda activate gaussian_splatting
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

python train.py -s /home/sj/work/gs-dataset/tandt/train \
  -m output/train_gcisp --hdr_camera --eval -r 4 --disable_viewer \
  --checkpoint_iterations 7000 24000 30000

python render.py -m output/train_gcisp --iteration 30000
python render_spherify_mine.py -m output/train_gcisp -k 0 --both
```

`--both` saves matching `trackball_before.mp4` (radiance visualization) and `trackball_after.mp4` (exposure+CRF RGB), plus PNGs. Before uses a fixed `--radiance-exposure` multiplier and [0,1] display clamp; it is not an HDR file. Other trackball options remain unchanged.

## Save and resume

Each PLY requires the adjacent `camera_gcisp.pth`, format version 4 / camera_type=gcisp / radiance_activation=clamp_sh_plus_half, containing camera/frame mapping, exposure/CRF/controller weights and controller readiness. Rendering restores these automatically. Earlier softplus GCISP v3 sidecars and checkpoints are rejected: their SH coefficients have different semantics and require a fresh training run. Legacy `camera_transformer.pth`/`camera_cnn.pth` and their checkpoints are rejected explicitly. Their SH files must not be rendered as vanilla colors. Rex GSAT_Splatter's Transformer/CNN exporter does not support GCISP yet.

Resume with `--start_checkpoint .../chkpnt24000.pth` and the original model/optimization arguments. Keep `--iterations` and activation ratio unchanged so the freeze boundary cannot silently move. A checkpoint preserves both ISP and controller optimizer states; the learning-rate schedule and phase are derived from the restored iteration. The final GCISP iteration performs an optimizer update before saving, so final controller weights and readiness are included.

## Validation

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Eight tests cover CRF identity/exposure stops, nontrivial CRF formula and numerical gradcheck, finite endpoint gradients, phase gradient isolation, camera routing, sidecar roundtrip/legacy rejection, positive radiance and Gaussian-DK scaling.

The reduced exposure+CRF was also compared with the pinned upstream PyTorch reference for outputs and gradients. The retained controller's exposure output was compared with the upstream controller using matching weights. A real GPU smoke run exercises densification, both training phases, checkpoint resume and before/after video rendering. Checkpoint comparisons verify Gaussian tensors and exposure/CRF remain exactly unchanged in stage 2 while controller weights change. These checks establish execution and numerical consistency, not improved floater suppression or reconstruction quality.
