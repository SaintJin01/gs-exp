"""AD-GS alternating regularization without rasterizer modifications."""
import torch

def adgs_phase(iteration, controller_stage=False):
    if controller_stage:
        return "controller", False, None, None
    if iteration <= 1500:
        return "warmup", False, 0.0005, 0.005
    low = iteration % 200 >= 100
    return ("low", True, 0.0005, 0.1) if low else ("high", False, 0.0002, 0.005)

def opacity_reset_due(iteration, controller_stage=False, interval=3000):
    return not controller_stage and iteration >= 501 and (iteration - 501) % interval == 0

def edge_aware_depth_loss(depth, image, smooth_weight=0.01, range_weight=0.001):
    # Forward differences, channel mean and spatial mean match AD-GS train.py.
    dx = depth[:, :, 1:] - depth[:, :, :-1]
    dy = depth[:, 1:, :] - depth[:, :-1, :]
    wx = torch.exp(-(image[:, :, 1:] - image[:, :, :-1]).abs().mean(0, keepdim=True))
    wy = torch.exp(-(image[:, 1:, :] - image[:, :-1, :]).abs().mean(0, keepdim=True))
    zero = depth.sum() * 0
    smooth = (wx * dx.abs()).mean() if dx.numel() else zero
    smooth = smooth + ((wy * dy.abs()).mean() if dy.numel() else zero)
    return smooth_weight * (smooth - range_weight * (depth.max() - depth.min()))

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


def scene_depth_geometry(geometry, env_count):
    """Exclude the environment prefix from every differentiable depth input."""
    return {key: (value[env_count:] if value is not None else None)
            for key, value in geometry.items()}
