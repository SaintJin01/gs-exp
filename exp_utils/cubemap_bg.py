#
# Cubemap environment background for Gaussian Splatting renders.
#
# Loads a cubemap baked by the environment_map_baking project (six face PNGs
# plus meta.json) and evaluates it along per-pixel camera rays so it can be
# composited behind the rasterized Gaussians.
#
# This is a plain (infinite) environment map: each pixel samples the cubemap by
# its world-space ray direction only. The background is treated as infinitely
# far, so there is no parallax / height-field correction -- move the camera and
# the environment stays put, exactly like a skybox.
#
# Conventions (must match envmap_baking):
#   - The bake world equals the COLMAP world used by this codebase.
#   - meta.json stores `world_to_cubemap`; sampling direction is
#     dir_cube = world_to_cubemap @ dir_world.
#   - Face order and UV mapping follow envmap_baking.cubemap.directions_to_faces_uv.
#

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FACE_NAMES = ("posx", "negx", "posy", "negy", "posz", "negz")


class CubemapBackground:
    def __init__(self, cubemap_dir, device="cuda"):
        self.device = device
        faces = []
        for name in FACE_NAMES:
            path = os.path.join(cubemap_dir, name + ".png")
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            faces.append(torch.from_numpy(image).permute(2, 0, 1))
        self.faces = torch.stack(faces).to(device)  # (6, 3, S, S)

        meta_path = os.path.join(cubemap_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as handle:
                meta = json.load(handle)
            rotation = np.asarray(meta["world_to_cubemap"], dtype=np.float32)
        else:
            print(f"[cubemap_bg] WARNING: {meta_path} not found, assuming identity orientation")
            rotation = np.eye(3, dtype=np.float32)
        self.world_to_cubemap = torch.from_numpy(rotation).to(device)
        print("[cubemap_bg] direction-only environment map (no parallax correction)")

    def background_for_view(self, view):
        """Return the cubemap evaluated along the view's pixel rays as (3, H, W) in [0, 1]."""
        height = int(view.image_height)
        width = int(view.image_width)
        tanfovx = float(np.tan(view.FoVx * 0.5))
        tanfovy = float(np.tan(view.FoVy * 0.5))
        fx = width / (2.0 * tanfovx)
        fy = height / (2.0 * tanfovy)

        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device),
            torch.arange(width, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        # COLMAP camera axes: +X right, +Y down, +Z forward.
        directions_cam = torch.stack(
            [
                (xs + 0.5 - width * 0.5) / fx,
                (ys + 0.5 - height * 0.5) / fy,
                torch.ones_like(xs),
            ],
            dim=-1,
        )
        # world_view_transform is the transposed world-to-camera matrix, so its
        # upper-left 3x3 block is already the camera-to-world rotation.
        cam_to_world = view.world_view_transform[:3, :3].to(self.device)
        directions = directions_cam @ cam_to_world.T

        directions = directions @ self.world_to_cubemap.T
        return self._sample(self.faces, directions)

    def _sample(self, faces, directions):
        """Bilinearly sample (6, C, S, S) faces for (H, W, 3) directions -> (C, H, W)."""
        face, u, v = _directions_to_faces_uv(directions)
        height, width = face.shape
        # grid_sample with align_corners=False: gx = 2u - 1 samples the texel
        # grid at continuous position u * S - 0.5, matching baked texel centers.
        grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1)  # (H, W, 2)
        output = torch.zeros(faces.shape[1], height, width, device=self.device)
        for f in range(6):
            mask = face == f
            if not torch.any(mask):
                continue
            sampled = F.grid_sample(
                faces[f : f + 1],
                grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )[0]
            output = torch.where(mask.unsqueeze(0), sampled, output)
        return output


def _directions_to_faces_uv(directions):
    """Torch port of envmap_baking.cubemap.directions_to_faces_uv."""
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    ax, ay, az = x.abs(), y.abs(), z.abs()

    face = torch.zeros_like(x, dtype=torch.long)
    uc = torch.zeros_like(x)
    vc = torch.zeros_like(x)

    is_x = (ax >= ay) & (ax >= az)
    is_y = (ay > ax) & (ay >= az)
    is_z = ~(is_x | is_y)

    sel = is_x & (x >= 0)
    face = torch.where(sel, torch.zeros_like(face), face)
    uc = torch.where(sel, -z / ax.clamp_min(1e-12), uc)
    vc = torch.where(sel, -y / ax.clamp_min(1e-12), vc)

    sel = is_x & (x < 0)
    face = torch.where(sel, torch.ones_like(face), face)
    uc = torch.where(sel, z / ax.clamp_min(1e-12), uc)
    vc = torch.where(sel, -y / ax.clamp_min(1e-12), vc)

    sel = is_y & (y >= 0)
    face = torch.where(sel, torch.full_like(face, 2), face)
    uc = torch.where(sel, x / ay.clamp_min(1e-12), uc)
    vc = torch.where(sel, z / ay.clamp_min(1e-12), vc)

    sel = is_y & (y < 0)
    face = torch.where(sel, torch.full_like(face, 3), face)
    uc = torch.where(sel, x / ay.clamp_min(1e-12), uc)
    vc = torch.where(sel, -z / ay.clamp_min(1e-12), vc)

    sel = is_z & (z >= 0)
    face = torch.where(sel, torch.full_like(face, 4), face)
    uc = torch.where(sel, x / az.clamp_min(1e-12), uc)
    vc = torch.where(sel, -y / az.clamp_min(1e-12), vc)

    sel = is_z & (z < 0)
    face = torch.where(sel, torch.full_like(face, 5), face)
    uc = torch.where(sel, -x / az.clamp_min(1e-12), uc)
    vc = torch.where(sel, -y / az.clamp_min(1e-12), vc)

    u = (uc + 1.0) * 0.5
    v = (vc + 1.0) * 0.5
    return face, u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)
