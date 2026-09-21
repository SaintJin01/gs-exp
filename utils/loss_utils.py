#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def masked_l1_loss(network_output, gt, mask):
    """Mean absolute RGB error over valid pixels only."""
    valid = mask.to(dtype=network_output.dtype, device=network_output.device)
    if valid.ndim == network_output.ndim - 1:
        valid = valid.unsqueeze(-3)
    numerator = (torch.abs(network_output - gt) * valid).sum()
    denominator = valid.sum().clamp_min(1.0) * network_output.shape[-3]
    return numerator / denominator

def masked_ssim(img1, img2, mask, window_size=11):
    """SSIM whose local moments and final average ignore invalid pixels."""
    if img1.ndim == 3:
        img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    mask = mask.to(dtype=img1.dtype, device=img1.device)
    channel = img1.shape[1]
    window = create_window(window_size, 1).to(device=img1.device, dtype=img1.dtype)
    norm = F.conv2d(mask, window, padding=window_size // 2).clamp_min(1e-8)

    def average(value):
        kernel = window.expand(channel, 1, window_size, window_size)
        return F.conv2d(value * mask, kernel, padding=window_size // 2,
                        groups=channel) / norm

    mu1, mu2 = average(img1), average(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1_sq = (average(img1.square()) - mu1_sq).clamp_min(0.0)
    sigma2_sq = (average(img2.square()) - mu2_sq).clamp_min(0.0)
    sigma12 = average(img1 * img2) - mu1_mu2
    score = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    valid_centres = mask.expand(-1, channel, -1, -1)
    return (score * valid_centres).sum() / valid_centres.sum().clamp_min(1.0)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
