"""Positive unbounded radiance and Gaussian-DK densification weighting."""
import torch
import torch.nn.functional as F
from utils.sh_utils import C0


def srgb_to_linear(rgb):
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))


def radiance_to_sh_dc(radiance):
    # Stable inverse softplus. A small floor permits initialization from black.
    x = radiance.clamp_min(1e-6)
    return (x + torch.log(-torch.expm1(-x))) / C0


def sh_to_radiance(sh):
    return F.softplus(sh)


@torch.no_grad()
def dk_gradient_weights(xyz, world_view_transform, radius, iteration,
                        anneal_start=4000, anneal_end=7000):
    if radius <= 0 or not 0 <= anneal_start < anneal_end:
        raise ValueError("Require radius > 0 and 0 <= anneal_start < anneal_end")
    # DK's CUDA 'depth' is camera-space p_view.z (not normalized depth).
    depth = xyz @ world_view_transform[:3, 2] + world_view_transform[3, 2]
    near_weight = (depth / radius).square().clamp(max=1.0)
    strength = 1.0 if iteration < anneal_start else (0.5 if iteration < anneal_end else 0.0)
    return strength * near_weight + (1.0 - strength)


@torch.no_grad()
def scale_densification_gradient(viewspace_points, weights):
    # Only the auxiliary screen-space gradient used for split/clone statistics.
    # Do not scale xyz/color/opacity gradients or the camera network gradients.
    if viewspace_points.grad is None:
        raise RuntimeError("Densification scaling requires rasterizer backward first")
    viewspace_points.grad.mul_(weights[:, None])
