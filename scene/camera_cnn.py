"""Shared HDR-to-RGB image CNN, with no per-view latent parameters."""
import torch
from torch import nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.gelu(self.conv1(x)))


class CameraCNN(nn.Module):
    def __init__(self, blocks=2, width=32, token_grid=16):
        super().__init__()
        if blocks < 1 or width < 1 or token_grid < 1:
            raise ValueError("blocks, width and grid must be positive")
        self.config = dict(blocks=blocks, width=width, token_grid=token_grid)
        self.encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.GELU())
        self.low_encoder = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU())
        self.position = nn.Linear(2, width * 2)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(width * 2) for _ in range(blocks)])
        self.decoder = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, radiance):
        squeeze = radiance.ndim == 3
        if squeeze:
            radiance = radiance.unsqueeze(0)
        if radiance.ndim != 4 or radiance.shape[1] != 3:
            raise ValueError("Expected RGB radiance [3,H,W] or [B,3,H,W]")
        # Preserve absolute intensity: no per-image normalization, no HDR clamp.
        local = self.encoder(torch.log1p(radiance))
        low = self.low_encoder(local)
        gh = min(self.config["token_grid"], low.shape[-2])
        gw = min(self.config["token_grid"], low.shape[-1])
        low = F.adaptive_avg_pool2d(low, (gh, gw))
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, gh, device=low.device, dtype=low.dtype),
            torch.linspace(-1, 1, gw, device=low.device, dtype=low.dtype), indexing="ij")
        pos = torch.stack((xx, yy), dim=-1).reshape(1, gh * gw, 2)
        # Keep the same coordinate features, now on the 2D feature grid.
        context = low + self.position(pos).transpose(1, 2).reshape(1, -1, gh, gw)
        for block in self.blocks:
            context = block(context)
        context = F.interpolate(context, size=local.shape[-2:], mode="bilinear", align_corners=False)
        # Direct RGB generation; high-resolution features preserve spatial detail.
        image = torch.sigmoid(self.decoder(torch.cat((local, context), dim=1)))
        return image.squeeze(0) if squeeze else image
