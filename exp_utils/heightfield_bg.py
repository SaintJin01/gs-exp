"""Heightfield cubemap background for gs-exp's per-pixel rasterizer background.

The cubemap is expressed in the upright map coordinates saved in manifest.json.
Finite patches are z-buffered from the camera; +inf is rotation-only sky and
NaN is unknown (black, zero coverage). Colour is sampled with coverage-aware
bilinear filtering so black holes do not bleed into adjacent valid texels.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ENVMAP_ROOT = os.environ.get("ENVMAP_TRAIN_ROOT", "/home/sj/work/envmap-train")
if ENVMAP_ROOT not in sys.path:
    sys.path.insert(0, ENVMAP_ROOT)
from envmap.cubemap import to_face_uv
from envmap.raster import visible_layers
from envmap.render import advance

FACES = ("posx", "negx", "posy", "negy", "posz", "negz")


class CubemapBackground:
    def __init__(self, cubemap_dir, device="cuda"):
        self.root = Path(cubemap_dir)
        self.device = torch.device(device)
        manifest = json.loads((self.root / "manifest.json").read_text())
        parent = Path(manifest.get("source", ""))
        while parent and not (parent / "manifest.json").exists():
            next_parent = Path(json.loads((parent / "manifest.json").read_text()).get("source", "")) if (parent / "manifest.json").exists() else None
            if next_parent is None:
                break
            parent = next_parent
        # The original bake manifest carries the world-to-map transform.
        source = self.root
        geometry = None
        for _ in range(8):
            info = json.loads((source / "manifest.json").read_text())
            if "world_to_map_rotation" in info:
                geometry = info
                break
            ref = info.get("source")
            if not ref:
                candidates = [json.loads(p.read_text()) for p in self.root.parent.glob("*/manifest.json")]
                candidates = [m for m in candidates if "world_to_map_rotation" in m]
                if not candidates:
                    raise ValueError("No world_to_map_rotation in cubemap manifest chain or sibling bake")
                geometry = candidates[0]
                break
            source = Path(ref)
        if geometry is None:
            raise ValueError("Cubemap manifest source chain is too long")
        self.rotation = torch.tensor(geometry["world_to_map_rotation"], dtype=torch.float32, device=self.device)
        self.centre = torch.tensor(geometry["centre_in_world"], dtype=torch.float32, device=self.device)
        self.metres_per_unit = float(geometry["units"]["metres_per_unit"])
        rgb, support, radius = [], [], []
        for name in FACES:
            stem = "layer0_" + name
            rgb.append(torch.from_numpy(np.asarray(Image.open(self.root / "colour" / (stem + ".png")).convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1))
            support.append(torch.from_numpy((np.load(self.root / "coverage" / (stem + ".npy")) > 0).astype(np.float32)))
            radius.append(torch.from_numpy(np.load(self.root / "radius" / (stem + ".npy")).astype(np.float32)))
        self.colour = torch.stack(rgb).to(self.device, dtype=torch.float32) / 255.0
        self.support = torch.stack(support)[:, None].to(self.device)
        self.radius = torch.stack(radius).to(self.device)
        self.height_size = int(self.radius.shape[1])
        self.render_supersample = int(os.environ.get("HEIGHTFIELD_RENDER_SUPERSAMPLE", "1"))
        if self.render_supersample not in (1, 2):
            raise ValueError("HEIGHTFIELD_RENDER_SUPERSAMPLE must be 1 or 2")
        assert self.colour.shape[2] == self.support.shape[2]
        assert self.radius.shape[1] == self.radius.shape[2]
        print(f"[heightfield] {self.root}: RGB {self.colour.shape[-1]}, depth {self.height_size}; finite/sky/unknown = "
              f"{int(torch.isfinite(self.radius).sum())}/{int(torch.isposinf(self.radius).sum())}/{int(torch.isnan(self.radius).sum())}")

    @torch.no_grad()
    def background_for_view(self, view):
        output_h, output_w = int(view.image_height), int(view.image_width)
        scale = self.render_supersample
        h, w = output_h * scale, output_w * scale
        fx = w / (2.0 * math.tan(float(view.FoVx) / 2.0))
        fy = h / (2.0 * math.tan(float(view.FoVy) / 2.0))
        c2w = view.world_view_transform[:3, :3].to(self.device)
        # gs-exp/SIBR camera is +Y down; envmap-train's frame is +Y up.
        flip = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=self.device))
        rotation = self.rotation @ c2w @ flip
        origin = self.rotation @ (view.camera_center.to(self.device) - self.centre)
        focal = torch.tensor([fx, fy], device=self.device)
        principal = torch.tensor([w / 2.0, h / 2.0], device=self.device)
        frame = SimpleNamespace(mask=torch.empty((h, w), dtype=torch.bool, device=self.device),
                                rotation=rotation, origin=origin, focal=focal, principal=principal)
        depth, _, covered, _ = visible_layers(self.radius.reshape(1, -1), self.height_size, frame)
        yy, xx = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
        camera = torch.stack(((xx + .5 - principal[0]) / fx,
                              -(yy + .5 - principal[1]) / fy,
                              torch.ones_like(xx, dtype=torch.float32)), -1).float()
        rays = F.normalize(camera, dim=-1) @ rotation.T
        safe = torch.where(covered, depth, torch.ones_like(depth))
        direction = torch.where(covered[..., None],
                                advance(origin, rays.reshape(-1, 3), safe.reshape(-1)).reshape(h, w, 3),
                                rays)
        background = self._sample(direction)
        if scale > 1:
            # Antialias the projected depth-patch boundary before compositing
            # at the camera's native resolution. The GS rasterizer is unchanged.
            background = F.avg_pool2d(background[None], scale, stride=scale)[0]
        return background

    def _sample(self, directions):
        face, u, v = to_face_uv(directions)
        grid = torch.stack((2*u-1, 2*v-1), -1)[None]
        out = torch.zeros((3, *face.shape), dtype=torch.float32, device=self.device)
        for f in range(6):
            selected = face == f
            if not bool(selected.any()):
                continue
            valid = F.grid_sample(self.support[f:f+1], grid, mode="bilinear",
                                  padding_mode="border", align_corners=False)[0]
            weighted = F.grid_sample((self.colour[f:f+1] * self.support[f:f+1]), grid,
                                     mode="bilinear", padding_mode="border",
                                     align_corners=False)[0]
            value = torch.where(valid > 1e-5, weighted / valid.clamp_min(1e-5), torch.zeros_like(weighted))
            out = torch.where(selected[None], value, out)
        return out.clamp(0, 1)
