"""Training-view FSGS depth supervision (VITA-Group/FSGS).
DPT_Hybrid weights and preprocessing match utils/depth_utils.py.
Pseudo-view guidance and Gaussian unpooling are intentionally not included.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

MIDAS_REVISION = "454597711a62eabcbf7d1e89f3fb9f569051ac9b"
PRIOR_SPEC = dict(model="DPT_Hybrid", revision=MIDAS_REVISION,
                  weights="dpt_hybrid_384.pt", input_hw=[384,512],
                  normalization="(rgb-0.5)/0.5", interpolation="bicubic",
                  align_corners=False, version=1)

@torch.no_grad()
def estimate_depth(model, image):
    h, w = image.shape[-2:]
    x = F.interpolate((image[None] - 0.5) / 0.5, size=(384,512),
                      mode="bicubic", align_corners=False)
    prediction = model(x)
    return F.interpolate(prediction.unsqueeze(1), size=(h,w),
                         mode="bicubic", align_corners=False)[0].float()

def load_depth_model(device):
    # Pin code as well as the official v3 DPT_Hybrid checkpoint.
    model = torch.hub.load("isl-org/MiDaS:" + MIDAS_REVISION, "DPT_Hybrid",
                           pretrained=True, trust_repo=True, skip_validation=True)
    return model.to(device).eval().requires_grad_(False)

def prepare_depth_priors(cameras, cache_dir, device="cuda", model_loader=None):
    """Hash actual resized GT pixels: stale images/resolutions cannot reuse priors."""
    from tqdm import tqdm
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = None
    loader = model_loader or load_depth_model
    try:
        for camera in tqdm(cameras, desc="FSGS depth priors"):
            image = camera.original_image.detach().cpu().float().contiguous()
            key = hashlib.sha256(json.dumps(PRIOR_SPEC, sort_keys=True).encode()
                                 + str(tuple(image.shape)).encode()
                                 + image.numpy().tobytes()).hexdigest()
            path = cache_dir / (key + ".npy")
            valid = False
            if path.exists():
                try:
                    prior = np.load(str(path), allow_pickle=False)
                    valid = prior.shape == (1, *image.shape[-2:]) and np.isfinite(prior).all()
                except (ValueError, OSError):
                    pass
            if not valid:
                if model is None:
                    model = loader(device)
                prior = estimate_depth(model, image.to(device)).cpu().numpy()
                if not np.isfinite(prior).all():
                    raise ValueError("Nonfinite DPT prior for " + camera.image_name)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as f:
                    np.save(f, prior, allow_pickle=False)
                temporary.replace(path)
            camera.fsgs_depth_path = str(path)
        (cache_dir / "model.json").write_text(json.dumps(PRIOR_SPEC, indent=2))
    finally:
        if model is not None:
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

def load_prior(camera, device):
    return torch.from_numpy(np.load(camera.fsgs_depth_path, allow_pickle=False)).to(device)

def fsgs_depth_loss(depth, prior, eps=1e-12):
    """Exact two-candidate Pearson objective on finite, nondegenerate inputs."""
    x, p = depth.reshape(-1), prior.detach().reshape(-1)
    if x.shape != p.shape:
        raise ValueError("Rendered depth and prior shapes must match")
    valid = torch.isfinite(x) & torch.isfinite(p) & ((p + 200).abs() > eps)
    x, p = x[valid], p[valid]
    if x.numel() < 2:
        return x.sum() * 0
    x = x - x.mean()
    xx = x.square().sum()
    if xx.detach().item() <= eps:
        return x.sum() * 0
    losses = []
    for target in (-p, 1.0 / (p + 200.0)):
        y = target - target.mean()
        yy = y.square().sum()
        if yy.detach().item() > eps:
            corr = (x*y).sum() / torch.sqrt(xx*yy)
            losses.append(1.0 - corr.clamp(-1.0,1.0))
    return torch.stack(losses).min() if losses else x.sum() * 0

def rasterize_z_depth(rasterizer, world_view_transform, geometry):
    # Same geometry and SAME auxiliary means2D tensor as the RGB pass: autograd
    # adds both screen-space gradients before densification takes their norm.
    xyz = geometry["means3D"]
    z = xyz @ world_view_transform[:3, 2] + world_view_transform[3, 2]
    result = rasterizer(colors_precomp=z[:, None].expand(-1, 3).contiguous(),
                        **geometry)
    # Select one replicated channel (no factor of three). No clamp, ISP,
    # reciprocal, alpha normalization or detach.
    return result[0][:1]
