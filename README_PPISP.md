# 3DGS + PPISP

Integration lives in this checkout. PPISP upstream: https://github.com/nv-tlabs/ppisp
Pinned commit: a247368047651db2e03e6f97a36e4f5bebbe5c5f (1.3.0).

## Environment (WSL Ubuntu 22.04 / RTX 3090)

Run in a WSL bash terminal, not PowerShell:

```bash
cd /home/sj/work/gs-exp
# If the host C++ compiler is missing:
# sudo apt-get update && sudo apt-get install -y build-essential
bash scripts/setup_ppisp_env.sh
source /home/sj/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs-ppisp
```

The script creates Python 3.10 + PyTorch 2.1.2 / torchvision 0.16.2 cu118,
CUDA toolkit 11.8 (including nvcc), NumPy 1.26.4, and the three local 3DGS
CUDA extensions plus pinned PPISP. It uses --no-build-isolation so the
extensions compile against the installed torch. Architecture 8.6 targets this
machine's RTX 3090. Do not rerun the environment creation step if the environment
already exists; activate it and run the installation lines you need instead.

## Train / render / evaluate

Set DATA to a COLMAP scene root containing images/ and sparse/0/, or a Blender
synthetic scene containing transforms_train.json and transforms_test.json:

```bash
cd /home/sj/work/gs-exp
conda activate 3dgs-ppisp
read -r -p "Dataset absolute path: " DATA
python train.py -s "$DATA" -m output/ppisp --eval \
  --iterations 16000 --disable_viewer --checkpoint_iterations 7000 10000 16000
python render.py -m output/ppisp --iteration 16000
python metrics.py -m output/ppisp
```

The saved cfg_args enables PPISP automatically for render.py. Test images use
frame_idx=-1; no test image colors are used to fit their exposure or white balance.
Before controller training, unseen views use identity per-frame corrections
plus learned camera effects. After controller activation, both train and novel
views use the camera's controller, as in upstream PPISP.

## Resume

Keep the original dataset split, image directory and total iteration count:

```bash
python train.py -s "$DATA" -m output/ppisp --eval \
  --iterations 16000 --disable_viewer \
  --start_checkpoint output/ppisp/chkpnt10000.pth
```

Gaussian parameters, PPISP parameters, PPISP optimizers/schedulers, configuration,
and stable camera/frame mappings are restored. Random sampling state is not
serialized, so a resumed trajectory is not guaranteed bit-for-bit identical.

## Behavior

- PPISP is always enabled; there is no non-PPISP mode in this branch.
- --train_test_exp is unavailable because PPISP is always enabled.
- Raw rasterizer RGB reaches PPISP before any upper clamp; evaluation/output
  clamps to the display range.
- AD-GS scheduling/depth regularization is ported from commit `091f454`.
  Gradient scaling and its CLI options are removed. No monocular depth prior loss.
  Warm-up: iterations 1..1500. Afterwards, `iteration % 200 >= 100` selects Low;
  the other 100 iterations select High. Thresholds (gradient, opacity) are
  warm-up=(0.0005,0.005), High=(0.0002,0.005), Low=(0.0005,0.1).
  Densification/pruning runs every 100 steps after 500 and before 10000 (last 9900).
  Size-based pruning is disabled. Low-phase depth regularization runs only during
  the initial 10000-step joint-training stage: smooth_weight=0.01, range_weight=0.001.
  It uses GT RGB edges and alpha-composited z-depth from a second rasterizer pass;
  the rasterizer submodule is unchanged. Both passes share screen-space gradients.
  Opacity resets to at most 0.05 after the optimizer at 501+3000*k until distillation.
  SH still increments every 1000 steps; the older 5000-step SH experiment is not used.
  This is the earlier single-model AD-GS adaptation, without two-model pseudo-view
  consistency. Full official PPISP (including vignetting/color correction) remains.
- By default, steps 1..10000 jointly train Gaussians and ISP parameters, followed
  by controller-only training at steps 10001..16000 (6000 steps). The activation
  ratio is 0.625; overriding total iterations scales this split proportionally.
  Gaussian position learning-rate decay ends at step 10000. In the controller
  phase, Gaussian updates, SH progression, densification, opacity reset and depth
  loss stop; upstream freezes ISP parameters and detaches radiance input.
- COLMAP intrinsic camera IDs identify physical cameras; IDs are compacted.
  Blender views share one physical camera. Dataset-relative image paths identify
  frames independently of shuffled loading and disambiguate train/r_0.png from
  test/r_0.png.
- Every physical camera used for test rendering must occur in training. A camera
  absent from training raises an explicit error. COLMAP datasets assigning a new
  camera ID to every image should be recalibrated/grouped appropriately first.
- Python network-GUI novel views use camera index 0 because GUI camera packets
  do not contain physical camera IDs. Standalone SIBR viewers do not load PPISP.
- Each saved point_cloud/iteration_N directory contains point_cloud.ply and
  ppisp.pth. Keep them together. Copying only the PLY loses ISP correction.
- PPISP-aware checkpoint files are dictionaries; ordinary 3DGS checkpoints retain
  the original tuple format. Switching modes during resume is rejected.
- Rendering restores scheduler phase, avoiding an untrained controller on early
  snapshots, including a snapshot saved exactly at activation.

## Verification

```bash
python scripts/test_ppisp_integration.py
python scripts/test_ppisp_e2e.py
```

Verified on this RTX 3090 with Python 3.10 / torch 2.1.2 / CUDA 11.8 in an isolated
temporary environment: all four CUDA extensions build, backward gradients,
controller phase, stable mappings, early/final snapshots, training and rendering,
resume before/after freezing, and a baseline run without PPISP.
The full conda installation script has not been executed end-to-end.
Real-dataset convergence and image quality require a full training run.

CPU AD-GS checks: `python -m unittest discover -s tests -v`. GPU training is not run by these checks.
