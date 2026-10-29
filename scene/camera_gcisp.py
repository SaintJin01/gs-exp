# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Modified for gs-exp: exposure-only controller and CRF; see third_party/ppisp/NOTICE.md.
"""Reduced PPISP: radiance -> scalar exposure -> per-camera parametric CRF."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class _PowDeadZone(torch.autograd.Function):
    """``base ** exponent`` whose gradients are zero for ``base <= EPS``.

    Used only for the normalized toe/shoulder bases, matching the CUDA CRF
    backward. The exact forward keeps black and saturation unchanged while
    cutting off the singular slope at the endpoints. Gamma must retain its
    gradients for every positive intermediate value.
    """

    # PPISP_CRF_BASE_GRAD_EPS, exported as ppisp_cuda.CRF_BASE_GRAD_EPS and
    # asserted equal by test_reference_dead_zone_matches_cuda_constant.
    EPS = 1e-6

    @staticmethod
    def forward(ctx, base: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        out = torch.pow(base, exponent)
        ctx.save_for_backward(base, exponent, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        base, exponent, out = ctx.saved_tensors
        active = base > _PowDeadZone.EPS
        # Evaluate the singular factors only where they are used.
        safe_base = torch.where(active, base, torch.ones_like(base))
        zero = torch.zeros_like(base)
        grad_base = torch.where(
            active, grad_out * exponent * torch.pow(safe_base, exponent - 1.0), zero)
        grad_exponent = torch.where(active, grad_out * out * torch.log(safe_base), zero)
        return grad_base, grad_exponent.sum_to_size(exponent.shape)



def apply_crf(rgb, params):
    """PPISP toe/shoulder/gamma curve on CHW RGB, with CUDA-compatible endpoint gradients."""
    toe = (0.3 + F.softplus(params[:, 0]))[:, None, None]
    shoulder = (0.3 + F.softplus(params[:, 1]))[:, None, None]
    gamma = (0.1 + F.softplus(params[:, 2]))[:, None, None]
    # Numerical guard only at floating-point sigmoid saturation.
    center = torch.sigmoid(params[:, 3]).clamp(1e-7, 1-1e-7)[:, None, None]
    x = rgb.clamp(0, 1)
    a = shoulder * center / (toe + center * (shoulder - toe))
    # Clamp unused branches' bases to 1 to prevent overflow before torch.where.
    low = a * _PowDeadZone.apply((x / center).clamp(max=1), toe)
    high = 1 - (1-a) * _PowDeadZone.apply(((1-x)/(1-center)).clamp(max=1), shoulder)
    y = torch.where(x <= center, low, high)
    positive = y > 0
    safe_y = torch.where(positive, y, torch.ones_like(y))
    return torch.where(positive, safe_y.pow(gamma), torch.zeros_like(y))


class ExposureController(nn.Module):
    """PPISP's original 1x1 CNN, 5x5 pool and 3-layer MLP, without its color head."""
    def __init__(self, input_downsampling=3, cnn_feature_dim=64,
                 hidden_dim=128, num_mlp_layers=3, pool_grid_size=(5,5)):
        super().__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.MaxPool2d(kernel_size=input_downsampling,
                         stride=input_downsampling),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, cnn_feature_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(pool_grid_size),
            nn.Flatten(),
        )

        # Input dimension: CNN features + prior exposure
        cnn_output_dim = cnn_feature_dim * \
            pool_grid_size[0] * pool_grid_size[1]
        input_dim = cnn_output_dim + 1  # +1 for prior_exposure

        # MLP trunk (shared hidden layers)
        trunk_layers = [nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(inplace=True)]
        for _ in range(num_mlp_layers - 1):
            trunk_layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        self.mlp_trunk = nn.Sequential(*trunk_layers)

        # Separate output heads (avoids misaligned views from slicing)
        self.exposure_head = nn.Linear(hidden_dim, 1)

    def forward(self, rgb, prior_exposure=None):
        # PPISP detaches features even during joint optimization.
        x = rgb.unsqueeze(0).detach()
        if min(x.shape[-2:]) < 3:
            x = F.interpolate(x, size=(max(3,x.shape[-2]), max(3,x.shape[-1])), mode="nearest")
        features = self.cnn_encoder(x).squeeze(0)
        prior = rgb.new_zeros(1) if prior_exposure is None else prior_exposure.to(rgb).reshape(1)
        return self.exposure_head(self.mlp_trunk(torch.cat((features, prior)))).squeeze(-1)


class GCISP(nn.Module):
    def __init__(self, frame_names, camera_ids, controller_activation_ratio=0.8):
        super().__init__()
        if not frame_names or not camera_ids or not 0 < controller_activation_ratio < 1:
            raise ValueError("GCISP needs training frames/cameras and 0 < activation ratio < 1")
        if len(set(frame_names)) != len(frame_names) or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Duplicate frame names or camera IDs")
        self.config = dict(frame_names=list(frame_names), camera_ids=list(camera_ids),
                           controller_activation_ratio=controller_activation_ratio)
        self.frame_index = {name:i for i,name in enumerate(frame_names)}
        self.camera_index = {key:i for i,key in enumerate(camera_ids)}
        self.exposure_params = nn.Parameter(torch.zeros(len(frame_names)))
        raw = torch.tensor([math.log(math.expm1(.7)), math.log(math.expm1(.7)),
                            math.log(math.expm1(.9)), 0.])
        self.crf_params = nn.Parameter(raw.reshape(1,1,4).repeat(len(camera_ids),3,1))
        self.controllers = nn.ModuleList([ExposureController() for _ in camera_ids])
        self.register_buffer('controller_updates', torch.zeros(len(camera_ids), dtype=torch.long))
        self.stage_active = False
        self.active_camera = None

    def set_stage(self, iteration, total_iterations):
        self.stage_active = iteration >= max(1, int(total_iterations*self.config['controller_activation_ratio']))
        self.exposure_params.requires_grad_(not self.stage_active)
        self.crf_params.requires_grad_(not self.stage_active)
        return self.stage_active

    def forward(self, radiance, camera_id=None, frame_name=None, exposure_prior=None):
        if radiance.ndim != 3 or radiance.shape[0] != 3:
            raise ValueError("GCISP expects CHW RGB radiance")
        if camera_id in self.camera_index:
            camera = self.camera_index[camera_id]
        elif len(self.camera_index)==1:
            camera = 0
        else:
            raise ValueError("Unknown physical camera ID: {}".format(camera_id))
        use_controller = (self.training and self.stage_active) or self.controller_updates[camera].item() > 0
        if use_controller:
            rgb = radiance.detach() if self.training and self.stage_active else radiance
            exposure = self.controllers[camera](rgb, exposure_prior)
        else:
            rgb = radiance
            exposure = (self.exposure_params[self.frame_index[frame_name]] if frame_name is not None
                        else radiance.new_zeros(()))
        if self.training and self.stage_active:
            self.active_camera = camera
        return apply_crf(rgb * torch.pow(2., exposure), self.crf_params[camera])

    def regularization_loss(self):
        if self.stage_active:
            return self.crf_params.new_zeros(())
        centered = self.crf_params - self.crf_params.mean(dim=1, keepdim=True)
        return self.exposure_params.mean().square() + 0.1 * centered.square().mean()

    @torch.no_grad()
    def mark_controller_update(self):
        if self.stage_active and self.active_camera is not None:
            self.controller_updates[self.active_camera] += 1
