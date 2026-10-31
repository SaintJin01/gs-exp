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

## FSGS training-view depth loss

Uses frozen MiDaS DPT_Hybrid (official dpt_hybrid_384.pt weights), code pinned
to 454597711a62eabcbf7d1e89f3fb9f569051ac9b. Install requirements-fsgs.txt.
On first training startup torch.hub downloads model code/weights if absent.
Exact FSGS preprocessing: resized training GT RGB [0,1], (rgb-0.5)/0.5,
bicubic resize to 384x512, DPT, bicubic resize back, align_corners=False.
No metric conversion or per-image normalization. Float32 priors are cached
under output/fsgs_depth_cache keyed by actual image content, resolution and
model/preprocessing metadata. The model is released before optimization.

Objective: min(1-corr(z_depth, -prior), 1-corr(z_depth, 1/(prior+200))).
Weight --depth_weight defaults to 0.05 throughout joint training; zero disables.
Finite-pixel and degenerate-variance guards avoid undefined correlations.
This ports training-view depth guidance only, NOT FSGS pseudo views/unpooling
or its 10000-step training schedule. Existing 30000+5000 GCISP stages remain.
Controller-only resumes do not load DPT or calculate depth supervision.

AD-GS smoothness/range losses, high/low alternation, strong pruning and reset
schedule are removed. Standard 3DGS densification ends at 15000, threshold
0.0002, opacity pruning <0.005; size pruning restored. Reset caps opacity
at 0.01 every 3000 iterations during densification (white-background exception
retained). SH starts at 0, increases every 1000 iterations up to 3.
DK gradient scaling remains removed.

The rasterizer submodule is unchanged. Z-depth uses the existing extra pass
with z repeated as precomputed RGB, zero background, no ISP/clamp and shared
means2D gradients. Existing inverse-depth output is not used by this loss.
No extra depth pass, densification, reset or SH updates in controller stage.

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

Existing tests cover CRF identity/exposure stops, nontrivial CRF formula and numerical gradcheck, finite endpoint gradients, phase gradient isolation, camera routing, sidecar roundtrip/legacy rejection, positive radiance. FSGS CPU tests cover upstream loss values/gradients, exact preprocessing,
invalid/flat inputs, cache invalidation and two-pass gradient equivalence.

The reduced exposure+CRF was also compared with the pinned upstream PyTorch reference for outputs and gradients. The retained controller's exposure output was compared with the upstream controller using matching weights. A real GPU smoke run exercises densification, both training phases, checkpoint resume and before/after video rendering. Checkpoint comparisons verify Gaussian tensors and exposure/CRF remain exactly unchanged in stage 2 while controller weights change. These checks establish execution and numerical consistency, not improved floater suppression or reconstruction quality.

The activation ratio specifies the fraction of completed joint-training updates.
With one-based iteration IDs, controller-only training starts at floor(total * ratio) + 1.
The default run therefore has exactly 30,000 joint and 5,000 controller updates.
