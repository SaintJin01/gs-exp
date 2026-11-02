# 3DGS + PPISP

Integration lives in this checkout. PPISP upstream: https://github.com/nv-tlabs/ppisp
Pinned commit: a247368047651db2e03e6f97a36e4f5bebbe5c5f (1.3.0).

## Environment (WSL Ubuntu 22.04 / RTX 3090)

Run in a WSL bash terminal, not PowerShell:

```bash
cd /home/sj/work/PPIPS/gaussian-splatting
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
cd /home/sj/work/PPIPS/gaussian-splatting
conda activate 3dgs-ppisp
read -r -p "Dataset absolute path: " DATA
python train.py -s "$DATA" -m output/ppisp --eval --ppisp \
  --iterations 30000 --disable_viewer --checkpoint_iterations 7000 24000 30000
python render.py -m output/ppisp --iteration 30000
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
python train.py -s "$DATA" -m output/ppisp --eval --ppisp \
  --iterations 30000 --disable_viewer \
  --start_checkpoint output/ppisp/chkpnt24000.pth
```

Gaussian parameters, PPISP parameters, PPISP optimizers/schedulers, configuration,
and stable camera/frame mappings are restored. Random sampling state is not
serialized, so a resumed trajectory is not guaranteed bit-for-bit identical.

## Behavior

- --ppisp is opt-in. Without it the original 3DGS path runs.
- --train_test_exp and --ppisp cannot be combined.
- Raw rasterizer RGB reaches PPISP before any upper clamp; evaluation/output
  clamps to the display range.
- The first 80% of steps jointly train Gaussians and ISP parameters. At scheduler
  step 24000 (for 30000 total), the following training step starts the controller
  phase. Gaussian updates, SH progression, densification, opacity reset and depth
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
