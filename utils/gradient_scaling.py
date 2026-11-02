"""Gaussian-DK depth-aware scaling for densification gradients only."""
import torch


@torch.no_grad()
def dk_gradient_weights(xyz, world_view_transform, radius, iteration,
                        anneal_start=4000, anneal_end=7000):
    if radius <= 0 or not 0 <= anneal_start < anneal_end:
        raise ValueError("Require radius > 0 and 0 <= anneal_start < anneal_end")
    # DK's CUDA depth is camera-space p_view.z (not normalized depth).
    depth = xyz @ world_view_transform[:3, 2] + world_view_transform[3, 2]
    near_weight = (depth / radius).square().clamp(max=1.0)
    strength = 1.0 if iteration < anneal_start else (0.5 if iteration < anneal_end else 0.0)
    return strength * near_weight + (1.0 - strength)


@torch.no_grad()
def scale_densification_gradient(viewspace_points, weights):
    # Scale only the auxiliary screen-space gradient consumed by split/clone statistics.
    # Gaussian parameter gradients and PPISP/controller gradients remain unchanged.
    if viewspace_points.grad is None:
        raise RuntimeError("Densification scaling requires rasterizer backward first")
    viewspace_points.grad.mul_(weights[:, None])