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

For the default 35,000 iterations and controller activation ratio `6/7` (approximately 0.8571428571428571):

1. Iterations 1..30,000: optimize Gaussians, per-training-frame exposure and per-physical-camera CRF with L1+DSSIM plus PPISP exposure-mean and CRF-channel regularization. Exposure offsets are in stops. Physical cameras are grouped by COLMAP intrinsic camera ID, not image ID. Blender scenes use one camera. Frame names are sorted for stable checkpoint indexing.
2. Iterations 30,001..35,000: freeze Gaussians, frame exposure and CRF; stop SH-degree changes, densification, opacity reset and depth supervision. Train only exposure controllers using the same reconstruction loss against ground truth, following upstream PPISP's default controller-distillation routing. No additional teacher loss is introduced. Radiance is detached in this phase.

Each camera's controller is the original 1x1 Conv(3->16), MaxPool(3), ReLU, Conv(16->32), ReLU, Conv(32->64), adaptive 5x5 pool, three 128-wide ReLU MLP layers, and a scalar exposure head. The color head is removed. Radiance feature input is detached as in upstream. Optional exposure prior is supported by the module; current dataset integration supplies zero (no EXIF prior). Images smaller than 3 pixels are enlarged only for the controller's pooling operation.

Inference uses the controller after it has received updates. This readiness is saved per physical camera. Before controller training, inference uses zero exposure rather than random controller weights. Novel-view inference never looks up a training frame exposure. Unknown IDs in a multi-camera model raise an error; a single-camera model shares its one camera model across views.

ISP optimizer defaults: Adam lr=0.002, eps=1e-15; 500-step warmup from 0.01x then exponential decay with factor 0.01 over 30,000 decay steps. Controller Adam lr=0.001 is fixed. Optimizer groups are disjoint. CLI options are `--isp_lr`, `--controller_lr` and `--controller_activation_ratio`.

## AD-GS regularization

DK gradient scaling and its options are removed. Warm-up ends at iteration 1500.
Afterwards low is iteration % 200 >= 100; high is the other half.
Warm-up/high/low use (densification gradient, opacity prune) thresholds
(0.0005, 0.005), (0.0002, 0.005), (0.0005, 0.1), respectively.
Densify/prune every 100 iterations after 500, strictly before densify_until_iter.
The cutoff is 10000, matching the AD-GS run script (last event: 9900).
Size-based pruning is disabled. Low-phase depth regularization continues through
joint training even after densification stops. Reset opacity to at most 0.05
after optimizer updates at 501 + 3000*k, only outside controller training.

Low phases use GT RGB edge weights and alpha-composited camera z-depth:
0.01 * (edge-aware smoothness - 0.001 * (max_depth - min_depth)).
Weights are --depth_smooth_weight and --depth_range_weight. External estimated
inverse-depth L1 supervision is replaced. The rasterizer submodule is unchanged;
its existing inverse-depth output remains available.

Z-depth is a second rasterizer invocation with z repeated as precomputed RGB,
zero background and identical geometry/settings. It bypasses ISP and clamping.
Both invocations share the auxiliary means2D tensor, so their gradient vectors
sum before densification statistics take the norm, matching a joint RGB/depth
renderer mathematically. CUDA accumulation can still differ in floating-point
roundoff; bitwise identity and real-GPU equivalence have not been established.
No extra depth pass, regularization, pruning or opacity reset in controller stage.

## Run

```bash
cd /home/sj/work/gs-exp
conda activate gaussian_splatting
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

python train.py -s /home/sj/work/gs-dataset/tandt/train \
  -m output/train_gcisp --hdr_camera --eval -r 4 --disable_viewer \
  --checkpoint_iterations 7000 30000 35000

python render.py -m output/train_gcisp --iteration 35000
python render_spherify_mine.py -m output/train_gcisp -k 0 --both
```

`--both` saves matching `trackball_before.mp4` (radiance visualization) and `trackball_after.mp4` (exposure+CRF RGB), plus PNGs. Before uses a fixed `--radiance-exposure` multiplier and [0,1] display clamp; it is not an HDR file. Other trackball options remain unchanged.

## Save and resume

Each PLY requires the adjacent `camera_gcisp.pth`, format version 4 / camera_type=gcisp / radiance_activation=clamp_sh_plus_half, containing camera/frame mapping, exposure/CRF/controller weights and controller readiness. Rendering restores these automatically. Earlier softplus GCISP v3 sidecars and checkpoints are rejected: their SH coefficients have different semantics and require a fresh training run. Legacy `camera_transformer.pth`/`camera_cnn.pth` and their checkpoints are rejected explicitly. Their SH files must not be rendered as vanilla colors. Rex GSAT_Splatter's Transformer/CNN exporter does not support GCISP yet.

Resume with `--start_checkpoint .../chkpnt30000.pth` and the original model/optimization arguments. Keep `--iterations` and activation ratio unchanged so the freeze boundary cannot silently move. A checkpoint preserves both ISP and controller optimizer states; the learning-rate schedule and phase are derived from the restored iteration. The final GCISP iteration performs an optimizer update before saving, so final controller weights and readiness are included.

## Validation

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Existing tests cover CRF identity/exposure stops, nontrivial CRF formula and numerical gradcheck, finite endpoint gradients, phase gradient isolation, camera routing, sidecar roundtrip/legacy rejection, positive radiance. AD-GS CPU tests additionally cover phase boundaries, reset timing, loss arithmetic and two-pass gradient equivalence.

The reduced exposure+CRF was also compared with the pinned upstream PyTorch reference for outputs and gradients. The retained controller's exposure output was compared with the upstream controller using matching weights. A real GPU smoke run exercises densification, both training phases, checkpoint resume and before/after video rendering. Checkpoint comparisons verify Gaussian tensors and exposure/CRF remain exactly unchanged in stage 2 while controller weights change. These checks establish execution and numerical consistency, not improved floater suppression or reconstruction quality.

The activation ratio specifies the fraction of completed joint-training updates.
With one-based iteration IDs, controller-only training starts at floor(total * ratio) + 1.
The default run therefore has exactly 30,000 joint and 5,000 controller updates.
