#!/usr/bin/env python3
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
"""
Orbit / turntable renderer for a trained Gaussian Splatting model.

Takes a reference (train) camera, places it on an orbit around the scene
center, rotates that orbit into a starting pose by ``--xoffset`` / ``--yoffset``
/ ``--zoffset`` degrees (about the world X, Y, Z axes), then sweeps
``--rotation`` degrees over ``--frames`` frames, saving one image per frame.
All angles are in degrees on a full 360 circle. The sweep itself is about the
scene's vertical (up) axis. The frames are then assembled into an mp4.

Example (half turn, 180 frames, starting tilted 20 deg about X):
    python exp_utils/rotation_render.py -m output/my_run \
        --xoffset 20 --rotation 180 --frames 180
"""

import glob
import os
import shutil
import subprocess
import sys
from os import makedirs

import numpy as np
import torch
import torchvision
from tqdm import tqdm

# Make the repo root importable when run as a script from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False


def camera_to_world(view):
    """Return the 4x4 camera-to-world matrix of a Camera (numpy float64)."""
    # Camera stores world_view_transform as W2V transposed (row-vector form),
    # so undo the transpose before inverting.
    w2v = view.world_view_transform.transpose(0, 1).cpu().numpy().astype(np.float64)
    return np.linalg.inv(w2v)


def rotation_matrix(axis, angle_rad):
    """Rodrigues rotation matrix for a unit-ish axis and angle in radians."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def estimate_scene_frame(views, axis_override=None):
    """Estimate the orbit center and up axis from the train cameras.

    Center is the mean camera position. The up axis is the normal of the
    best-fit plane through the camera centers (the axis the cameras orbit
    around), sign-aligned with the average camera up so positive angles spin
    counter-clockwise when viewed from above.
    """
    centers = np.stack([camera_to_world(v)[:3, 3] for v in views], axis=0)
    scene_center = centers.mean(axis=0)

    # Average camera up in world space. In this convention the camera's local
    # +Y points down, so world up is the negated mean of the C2W second column.
    cam_up = -np.mean([camera_to_world(v)[:3, 1] for v in views], axis=0)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)

    if axis_override is not None:
        up = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }[axis_override.lower()]
    else:
        # Plane normal = singular vector with smallest singular value.
        centered = centers - scene_center
        if centered.shape[0] >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            up = vh[-1]
        else:
            up = cam_up

    up = up / (np.linalg.norm(up) + 1e-12)
    if np.dot(up, cam_up) < 0:
        up = -up
    return scene_center, up


def make_orbit_view(ref_view, scene_center, Rrot, distance=None):
    """Build a MiniCam by rigidly rotating ref_view by `Rrot` around `scene_center`.

    If `distance` is given, the camera is dollied along its radial direction so
    its distance to the scene center equals `distance` (default: keep the
    reference camera's distance).
    """
    c2w = camera_to_world(ref_view)

    # Rotate camera position around the scene center and rotate orientation.
    new_c2w = np.eye(4)
    new_c2w[:3, :3] = Rrot @ c2w[:3, :3]
    eye = scene_center + Rrot @ (c2w[:3, 3] - scene_center)

    if distance is not None:
        radial = eye - scene_center
        eye = scene_center + radial / (np.linalg.norm(radial) + 1e-12) * distance
    new_c2w[:3, 3] = eye

    w2v = np.linalg.inv(new_c2w)
    world_view = torch.tensor(w2v, dtype=torch.float32).transpose(0, 1).cuda()

    proj = getProjectionMatrix(
        znear=ref_view.znear, zfar=ref_view.zfar,
        fovX=ref_view.FoVx, fovY=ref_view.FoVy,
    ).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    return MiniCam(
        width=ref_view.image_width,
        height=ref_view.image_height,
        fovy=ref_view.FoVy,
        fovx=ref_view.FoVx,
        znear=ref_view.znear,
        zfar=ref_view.zfar,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
    )


def frames_to_video(frame_dir, video_path, fps):
    """Stitch the saved ``%05d.png`` frames in `frame_dir` into an mp4.

    Uses ffmpeg if available (best quality / compatibility), otherwise falls
    back to imageio. Returns True on success.
    """
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frame_paths:
        print("[rotation_render] no frames found, skipping video.")
        return False

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [
            ffmpeg, "-y", "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            # Pad odd dimensions to even so yuv420p / libx264 accept them.
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            video_path,
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode == 0:
            return True
        print(f"[rotation_render] ffmpeg failed:\n{ret.stderr}")
        # Fall through to imageio.

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[rotation_render] ffmpeg not found and imageio unavailable; "
              "cannot write video.")
        return False

    writer = imageio.get_writer(video_path, fps=fps, macro_block_size=1)
    for p in frame_paths:
        writer.append_data(imageio.imread(p))
    writer.close()
    return True


def rotation_render(dataset, iteration, pipeline, xoffset, yoffset, zoffset,
                    rotation, frames, axis, output_name, separate_sh, fps,
                    make_video, distance):
    # A negative distance means "keep the reference camera's distance".
    if distance is not None and distance < 0:
        distance = None
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTrainCameras()
        if not views:
            views = scene.getTestCameras()
        if not views:
            raise RuntimeError("No cameras found in the scene to use as a reference.")
        ref_view = views[0]

        scene_center, up = estimate_scene_frame(views, axis_override=axis)
        print(f"[rotation_render] scene center : {scene_center}")
        print(f"[rotation_render] up axis      : {up}")
        ref_dist = float(np.linalg.norm(camera_to_world(ref_view)[:3, 3] - scene_center))
        print(f"[rotation_render] offset=(x={xoffset}, y={yoffset}, z={zoffset}) deg, "
              f"rotation={rotation} deg, frames={frames}")
        print(f"[rotation_render] distance     : "
              f"{ref_dist:.4f} (ref){'' if distance is None else f' -> {distance}'}")

        # Starting pose: rotate about the world X, then Y, then Z axes.
        R_offset = (
            rotation_matrix([0.0, 0.0, 1.0], np.radians(zoffset))
            @ rotation_matrix([0.0, 1.0, 0.0], np.radians(yoffset))
            @ rotation_matrix([1.0, 0.0, 0.0], np.radians(xoffset))
        )

        out_dir = os.path.join(
            dataset.model_path, output_name, f"ours_{scene.loaded_iter}", "renders"
        )
        makedirs(out_dir, exist_ok=True)

        for i in tqdm(range(frames), desc="Rotation render"):
            # Apply the offset pose, then sweep `rotation` degrees about the up
            # axis over `frames` steps; dividing by frames (not frames-1) keeps a
            # full 360 sweep seamless without a duplicate.
            sweep = rotation_matrix(up, np.radians(rotation * (i / frames)))
            view = make_orbit_view(ref_view, scene_center, sweep @ R_offset, distance)
            image = render(
                view, gaussians, pipeline, background,
                use_trained_exp=dataset.train_test_exp, separate_sh=separate_sh,
            )["render"]
            torchvision.utils.save_image(
                image, os.path.join(out_dir, f"{i:05d}.png")
            )

        print(f"[rotation_render] saved {frames} frames to {out_dir}")

        if make_video:
            video_path = os.path.join(
                os.path.dirname(out_dir), f"{output_name}.mp4"
            )
            if frames_to_video(out_dir, video_path, fps):
                print(f"[rotation_render] saved video to {video_path} ({fps} fps)")


if __name__ == "__main__":
    parser = ArgumentParser(description="Orbit / turntable rendering script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--xoffset", default=0.0, type=float,
                        help="Starting rotation about the world X axis, in degrees.")
    parser.add_argument("--yoffset", default=0.0, type=float,
                        help="Starting rotation about the world Y axis, in degrees.")
    parser.add_argument("--zoffset", default=0.0, type=float,
                        help="Starting rotation about the world Z axis, in degrees.")
    parser.add_argument("--rotation", default=360.0, type=float,
                        help="Total angle swept over all frames, in degrees.")
    parser.add_argument("--frames", default=120, type=int,
                        help="Number of frames to render.")
    parser.add_argument("--distance", default=-1.0, type=float,
                        help="Camera distance from the scene center, in world "
                             "units (default -1: keep the reference camera's distance).")
    parser.add_argument("--axis", default="y", choices=["x", "y", "z"],
                        help="Override the orbit axis (default: auto-estimate).")
    parser.add_argument("--output_name", default="rotation",
                        help="Subfolder name under the model path for the frames.")
    parser.add_argument("--fps", default=30, type=int,
                        help="Frames per second for the output mp4.")
    parser.add_argument("--skip_video", action="store_true",
                        help="Only render frames; do not assemble an mp4.")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering rotation for " + args.model_path)

    safe_state(args.quiet)

    rotation_render(
        model.extract(args), args.iteration, pipeline.extract(args),
        args.xoffset, args.yoffset, args.zoffset,
        args.rotation, args.frames, args.axis,
        args.output_name, SPARSE_ADAM_AVAILABLE,
        args.fps, not args.skip_video, args.distance,
    )
