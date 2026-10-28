# Shared HDR CNN camera experiment

Enable with `--hdr_camera`. Without it, the existing LDR training path is used.

## Forward path

```
Gaussian SH coefficients -> softplus -> positive unbounded RGB radiance
-> alpha-compositing rasterizer -> log1p
-> convolutional encoder -> pooled tokens -> 2 residual CNN blocks
-> bilinear upsampling + high-resolution encoder skip -> convolutional decoder
-> sigmoid RGB -> existing L1 + DSSIM loss
```

One camera network is shared across all views, without view embeddings or per-view exposure optimization. Gaussian parameters and the camera network train end-to-end. The decoder directly generates RGB; it does not predict exposure or tone-curve coefficients.

COLMAP colors already arrive as RGB in [0,1], converted from the original [0,255]. Initialization assumes sRGB, applies inverse sRGB, then inverse softplus to initialize the DC SH coefficient. Higher SH coefficients start at zero. Radiance has no upper clamp; this is relative learned radiance, not calibrated physical HDR. Ground truth remains ordinary RGB in [0,1]. `log1p` preserves absolute intensity without per-image normalization. White background means radiance 1 before the camera network.

Default architecture: encoder width 32, low-resolution feature width 64, 2 residual CNN blocks, at most 16x16 feature cells. Each block is x + Conv3x3(GELU(Conv3x3(x))) with zero padding. The coordinate embedding is retained; attention and Transformer layer normalization are removed. The full-resolution convolutional skip preserves local information. Feature pooling bounds low-resolution CNN cost; full-resolution features still consume memory proportional to image area.

## Gaussian-DK gradient scaling

Reference: local Gaussian-DK rasterizer Python backward. Scale only auxiliary screen-space gradients used by split/clone statistics, leaving optimizer gradients for position, SH, opacity, and the camera unchanged:

```
w = min(1, (camera_space_depth / radius)^2)
weight = strength * w + (1 - strength)
strength = 1 before iteration 4000, 0.5 until 7000, then 0
radius = gradient_scale_radius * scene.cameras_extent
```

The default radius multiplier is 1. This uses camera-space depth as in DK's CUDA code, not NDC depth. Implementation applies the weight immediately before accumulation of densification statistics, so no CUDA extension rebuild is needed. `--disable_gradient_scaling` disables it. The two boundaries are configurable with `--gradient_scale_anneal_start` and `--gradient_scale_anneal_end`.

## Run in WSL

```bash
cd /home/sj/work/gs-exp
conda activate gaussian_splatting
# Needed in the tested WSL environment for cuDNN to find libcuda:
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

python train.py -s /home/sj/work/gs-dataset/tandt/train \
  -m output/train_hdr_camera --hdr_camera --camera_blocks 2 \
  -r 4 --eval --disable_viewer --checkpoint_iterations 7000 15000 30000

python render.py -m output/train_hdr_camera --iteration 30000
```

`--camera_blocks 4` or `6` increases depth for a new experiment. Additional options: `--camera_width 32`, `--camera_token_grid 16`, and `--camera_lr 0.0001`. Use a fresh output directory for each architecture or ablation. For a baseline, omit `--hdr_camera`; to isolate scaling, add `--disable_gradient_scaling` to the HDR run.

Resume with the original data/model/optimization arguments plus `--start_checkpoint output/train_hdr_camera/chkpnt15000.pth`. Architecture options must match the checkpoint. CNN checkpoints include Gaussian and camera optimizer state, version 2 and camera_type=cnn. Transformer version-1 checkpoints/sidecars are rejected; start a new run on this branch. The Rex Transformer ONNX exporter does not support these CNN weights yet. As with the original training script, supply the original optimization and scaling options again when resuming.

Each saved PLY has an adjacent `camera_cnn.pth` containing camera weights and architecture. `render.py` loads it automatically. Keep both files together: HDR SH coefficients are incompatible with a vanilla LDR viewer. An HDR run fails explicitly when its required camera sidecar is missing. Existing LDR checkpoints cannot be resumed directly in HDR mode. `--train_test_exp` is incompatible with the shared camera.

## Validation

```bash
python -m unittest discover -s tests -p test_hdr_camera.py -v
```

Tests cover radiance initialization and unbounded gradients, arbitrary image sizes, gradient propagation through both residual CNN blocks, DK scaling and gradient isolation, and camera weight serialization. Short real-data runs additionally exercise GPU rasterization, densification, training resume, and saved-model rendering. These are execution checks, not evidence of improved reconstruction quality or reduced floaters.

## Keyframe trackball video

```bash
python render_spherify_mine.py -m output/train_hdr_camera -k 0 --both
```

`-k` is the zero-based index in the unshuffled training camera list. Use `--list-keyframes` to list names/indices, `--split test` for test cameras, or `--keyframe-mode name -k IMAGE_NAME` to select by exact image name.

The default orbit is 240 frames at 30 fps with a 10-degree cone angle about a fixed focus in front of the keyframe. `--angle 0` reproduces the keyframe pose. Focus distance defaults to median in-frustum Gaussian depth; override using `--focus-distance` in scene units.

With `--both`, output contains `trackball_before.mp4`, `trackball_after.mp4`, `renders_before/`, and `renders_after/`. Before means the rasterized radiance before the entire camera network (including its encoder), not intermediate attention features. For 8-bit PNG/MP4 display it uses `clamp(radiance * radiance_exposure, 0, 1)`, with default multiplier 1 and no gamma or per-frame normalization. Adjust the fixed multiplier with `--radiance-exposure`; these preview files do not preserve HDR values. After is the learned final RGB. Both use the same single rasterization and trajectory at each frame.

Without `--both`, only the final image is saved as `trackball.mp4` and `renders/`; this also works for baseline LDR models. `--both` requires an HDR camera model. Default output is `MODEL/trackball/ours_ITER/train_k00000/`; use `--output EMPTY_DIRECTORY` for repeat runs. Existing nonempty directories are rejected to avoid mixing frames. `trajectory.json` records poses, display multiplier, and video filenames. Odd image dimensions are padded by one pixel at the right/bottom for MP4 only.

This branch keeps the encoder, adaptive pooling, coordinates, upsampling, high-resolution skip, decoder, radiance activation, loss and gradient scaling unchanged. Only the context-processing blocks are replaced. Two CNN blocks have a 5x5 receptive field on the pooled grid, unlike global self-attention. `--camera_token_grid` retains its name for CLI compatibility and now controls the pooled feature grid.
