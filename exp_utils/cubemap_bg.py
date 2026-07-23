#
# Cubemap environment background for Gaussian Splatting renders.
#
# Loads a cubemap baked by the environment_map_baking project (six face PNGs
# plus meta.json) and evaluates it along per-pixel camera rays so it can be
# composited behind the rasterized Gaussians.
#
# Conventions (must match envmap_baking):
#   - The bake world equals the COLMAP world used by this codebase.
#   - meta.json stores `world_to_cubemap`; sampling direction is
#     dir_cube = world_to_cubemap @ dir_world.
#   - Face order and UV mapping follow envmap_baking.cubemap.directions_to_faces_uv.
#
# Parallax: the baked background sits at a finite distance, so direction-only
# lookups misalign by up to ~d/Z radians for a camera d away from the bake
# origin. Two corrections, best first:
#   1. Per-texel distance (`distance.npy`): the cubemap is treated as a radial
#      height field r(u) around the bake origin. Each view ray o + t*d is
#      intersected with that field by fixed-point iteration -- sample r along
#      the current direction, re-intersect the sphere of that radius, repeat.
#      Convergence is fast because the camera offset |o| is much smaller than
#      the background distance r.
#   2. Constant-radius skydome (`background_radius` only): intersect each ray
#      with a single sphere, correcting parallax to first order.
#

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FACE_NAMES = ("posx", "negx", "posy", "negy", "posz", "negz")


class CubemapBackground:
    def __init__(self, cubemap_dir, device="cuda", parallax_steps=8):
        self.device = device
        self.parallax_steps = parallax_steps
        faces = []
        for name in FACE_NAMES:
            path = os.path.join(cubemap_dir, name + ".png")
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            faces.append(torch.from_numpy(image).permute(2, 0, 1))
        self.faces = torch.stack(faces).to(device)  # (6, 3, S, S)

        meta_path = os.path.join(cubemap_dir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as handle:
                meta = json.load(handle)
            rotation = np.asarray(meta["world_to_cubemap"], dtype=np.float32)
        else:
            print(f"[cubemap_bg] WARNING: {meta_path} not found, assuming identity orientation")
            rotation = np.eye(3, dtype=np.float32)
        self.world_to_cubemap = torch.from_numpy(rotation).to(device)

        self.bake_origin = None
        self.background_radius = None
        if meta.get("bake_origin") is not None and meta.get("background_radius"):
            self.bake_origin = torch.tensor(meta["bake_origin"], dtype=torch.float32, device=device)
            self.background_radius = float(meta["background_radius"])

        self.disparity = None
        distance_path = os.path.join(cubemap_dir, meta.get("distance_map", "distance.npy"))
        if self.bake_origin is not None and os.path.exists(distance_path):
            self.disparity = self._load_disparity(distance_path)
            print(f"[cubemap_bg] per-texel distance parallax correction "
                  f"({self.parallax_steps} steps, fallback radius {self.background_radius:.2f})")
        elif self.bake_origin is not None:
            print(f"[cubemap_bg] skydome parallax correction: radius {self.background_radius:.2f}")
        else:
            print("[cubemap_bg] WARNING: no bake_origin/background_radius in meta.json, "
                  "sampling by direction only (background parallax uncorrected)")

    def _load_disparity(self, distance_path):
        """Load the baked distance map as (6, 1, S, S) inverse distances.

        Sky (+inf distance) becomes disparity 0; uncovered texels (NaN) fall
        back to the global background radius so they behave like the skydome.
        """
        distance = np.load(distance_path)
        disparity = np.zeros(distance.shape, dtype=np.float32)
        finite = np.isfinite(distance) & (distance > 0)
        disparity[finite] = 1.0 / distance[finite]
        disparity[np.isnan(distance)] = 1.0 / self.background_radius
        return torch.from_numpy(disparity).unsqueeze(1).to(self.device)

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

        if self.bake_origin is not None:
            directions = directions / directions.norm(dim=-1, keepdim=True)
            camera_center = view.camera_center.to(self.device).flatten()[:3]
            offset = camera_center - self.bake_origin
            if self.disparity is not None:
                directions = self._parallax_directions(directions, offset)
            else:
                # Skydome: intersect each ray with the background sphere around
                # the bake origin and sample toward the intersection point.
                t = _ray_sphere_far_root(directions, offset, self.background_radius)
                directions = offset + t[..., None] * directions

        directions = directions @ self.world_to_cubemap.T
        return self._sample(self.faces, directions)

    def _parallax_directions(self, directions, offset):
        """Intersect view rays with the radial height field r(u) = baked distance.

        Fixed-point iteration: sample the distance map along the current
        guess direction, re-intersect the sphere of that radius, repeat. The
        camera offset is small relative to the background distance, so most
        texels converge in a step or two; sky texels (disparity 0) degenerate
        to direction-only sampling via a huge radius.

        Texels straddling a range discontinuity (a real skyline where a finite
        foliage radius abuts sky at infinity) oscillate between the two branches
        instead of converging. On T&T train these are ~1-1.5% of background
        texels at the largest camera offsets, and 3 iterations leave them off by
        up to ~30px; 8 iterations bring the 99th-percentile self-consistency
        error under 1px. The extra grid_samples are negligible because the
        background is computed once per camera and cached during training.
        """
        # Keep the camera strictly inside every candidate sphere so the far
        # intersection root stays positive and the iteration cannot fold back.
        min_radius = 1.05 * float(offset.norm()) + 1e-6
        toward = directions
        for _ in range(self.parallax_steps):
            disparity = self._sample(self.disparity, toward @ self.world_to_cubemap.T)[0]
            radius = disparity.clamp_min(1e-6).reciprocal().clamp_min(min_radius)
            t = _ray_sphere_far_root(directions, offset, radius)
            toward = offset + t[..., None] * directions
            toward = toward / toward.norm(dim=-1, keepdim=True)
        return toward

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


def _ray_sphere_far_root(directions, offset, radius):
    """Distance along unit rays from `offset` to the sphere |x| = radius."""
    b = directions @ offset
    disc = b * b - (offset.dot(offset) - radius ** 2)
    return -b + torch.sqrt(disc.clamp_min(0.0))


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
