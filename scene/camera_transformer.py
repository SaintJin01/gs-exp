"""Shared HDR-to-RGB image transformer, with no per-view latent parameters."""
import torch
from torch import nn
import torch.nn.functional as F


class CameraTransformer(nn.Module):
    def __init__(self, blocks=2, width=32, token_grid=16):
        super().__init__()
        if blocks < 1 or width < 4 or width % 4 or token_grid < 1:
            raise ValueError("blocks/grid must be positive; width must be a positive multiple of 4")
        self.config = dict(blocks=blocks, width=width, token_grid=token_grid)
        self.encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.GELU())
        self.low_encoder = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU())
        self.position = nn.Linear(2, width * 2)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=width * 2, nhead=4,
                dim_feedforward=width * 4, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True) for _ in range(blocks)])
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
        tokens = low.flatten(2).transpose(1, 2) + self.position(pos)
        for block in self.blocks:
            tokens = block(tokens)
        context = tokens.transpose(1, 2).reshape(low.shape[0], -1, gh, gw)
        context = F.interpolate(context, size=local.shape[-2:], mode="bilinear", align_corners=False)
        # Direct RGB generation; high-resolution features preserve spatial detail.
        image = torch.sigmoid(self.decoder(torch.cat((local, context), dim=1)))
        return image.squeeze(0) if squeeze else image
