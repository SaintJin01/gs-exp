# Heightfield cubemap in gs-exp

Input (fixed for these entrypoints):

`/home/sj/work/gs-dataset/tandt/train/cubemaps/ablation_01_sky_border_2048_gt25m`

The six colour faces are 2048²; radii are 1024². Finite radii are rendered as
camera-dependent patches. Positive infinity is rotation-only sky. NaN/zero
coverage is black unknown. Bilinear RGB taps are renormalised over valid
coverage, preventing black holes from contaminating adjacent texels.

## Train

```bash
cd /home/sj/work/gs-exp
/home/sj/anaconda3/envs/gaussian_splatting/bin/python train_heightfield_bg.py \
  -s /home/sj/work/gs-dataset/tandt/train \
  -m output/train_heightfield_gt25 \
  -r 4 --eval
```

This reuses the patched per-pixel-background Gaussian rasterizer. While
training, `SIBR_remoteGaussian_app` can connect to the default port 6009 and
shows Gaussians composited over the heightfield cubemap.

## Offline render

```bash
cd /home/sj/work/gs-exp
/home/sj/anaconda3/envs/gaussian_splatting/bin/python render_heightfield_bg.py \
  -m output/train_heightfield_gt25 --skip_train
```

Results appear at `output/train_heightfield_gt25/test/ours_<iteration>/renders`.

## SIBR after training

Start the Python rendering server:

```bash
cd /home/sj/work/gs-exp
/home/sj/anaconda3/envs/gaussian_splatting/bin/python viewer_heightfield_bg.py \
  -m output/train_heightfield_gt25 --port 6009
```

In another terminal, run the remote (not local point-cloud) viewer:

```bash
/home/sj/work/gaussian-splatting/SIBR_viewers/install/bin/SIBR_remoteGaussian_app \
  --path /home/sj/work/gs-dataset/tandt/train --ip 127.0.0.1 --port 6009
```

The remote viewer sends its camera to Python, which renders both the Gaussians
and the envmap and sends back one composited RGB frame. The local
`SIBR_gaussianViewer_app` does **not** execute this Python pipeline and will
not show the envmap. Use a free port if 6009 is occupied.

The sampler uses `/home/sj/work/envmap-train` for its heightfield z-buffer.
Override that location with `ENVMAP_TRAIN_ROOT` if needed.

## Validation done

- A one-iteration training smoke test saved `output/heightfield_smoke_01`.
- Offline render wrote 38 test frames from that checkpoint.
- The standalone viewer answered a SIBR-format 64×36 camera request with
  6,912 RGB bytes and the scene path.
- Actual SIBR GUI display was not confirmed in this execution session; one
  launch returned `GLXBadFBConfig`, a window-context issue outside the Python
  render protocol.
