# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved. Non-commercial research/evaluation use; see LICENSE.md.
"""Render a keyframe-relative trackball circle to PNG and MP4.

-k selects the zero-based index in the chosen, unshuffled camera split.
Use --keyframe-mode name to select an exact image name instead.
--both saves rasterized radiance (clipped for display) and GCISP exposure/CRF RGB.
The camera follows a small circle on a sphere around a fixed focus point.
--angle is the cone half-angle in degrees (0 reproduces the keyframe).
"""
from contextlib import ExitStack
import copy
import json
import math
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

BACKEND = "gs-exp"


def select_keyframe(views, key, mode):
    if mode == "index":
        try:
            index = int(key)
        except ValueError:
            raise ValueError("-k must be an integer with --keyframe-mode index")
        if not 0 <= index < len(views):
            raise ValueError(f"Keyframe index {index} outside [0, {len(views) - 1}]")
        return index, views[index]
    matches = [(i, view) for i, view in enumerate(views) if view.image_name == key]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one image named {key!r}, found {len(matches)}")
    return matches[0]


def trackball_pose(reference_c2w, distance, angle_degrees, phase):
    """OpenCV camera axes: right, down, forward; phase is in radians."""
    rotation = reference_c2w[:3, :3]
    origin = reference_c2w[:3, 3]
    angle = math.radians(angle_degrees)
    focus = origin + distance * rotation[:, 2]
    radial = math.cos(phase) * rotation[:, 0] + math.sin(phase) * rotation[:, 1]
    center = focus + distance * (math.sin(angle) * radial - math.cos(angle) * rotation[:, 2])
    forward = (focus - center) / distance
    right = np.cross(rotation[:, 1], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = center
    return pose


def camera_at_pose(reference, pose):
    # Shallow copy retains image/exposure metadata without duplicating GPU images.
    camera = copy.copy(reference)
    w2c = np.linalg.inv(pose)
    camera.R = w2c[:3, :3].T.copy()
    camera.T = w2c[:3, 3].copy()
    camera.trans = np.zeros(3)
    camera.scale = 1.0
    camera.world_view_transform = torch.as_tensor(
        w2c.T.copy(), dtype=reference.world_view_transform.dtype,
        device=reference.world_view_transform.device)
    camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    return camera


def estimate_focus_distance(reference, gaussians):
    # A bounded sample avoids allocating a full point-cloud transform.
    points = gaussians.get_xyz
    points = points[::max(1, math.ceil(len(points) / 200000))]
    transform = reference.world_view_transform
    camera_points = points @ transform[:3, :3] + transform[3, :3]
    z = camera_points[:, 2]
    visible = (torch.isfinite(camera_points).all(dim=1) & (z > reference.znear)
               & (camera_points[:, 0].abs() < z * math.tan(reference.FoVx / 2))
               & (camera_points[:, 1].abs() < z * math.tan(reference.FoVy / 2)))
    if not visible.any():
        raise ValueError("No points in the keyframe frustum; specify --focus-distance explicitly")
    return z[visible].median().item()


def render_trackball(dataset, pipeline, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if args.list_keyframes:
        for index, view in enumerate(views):
            print(f"{index}\t{view.image_name}")
        return
    if args.both and not gaussians.hdr_camera:
        raise ValueError("--both requires an HDR model with camera_gcisp.pth")
    if gaussians.camera_model is not None:
        gaussians.camera_model.eval()
    index, reference = select_keyframe(views, args.keyframe, args.keyframe_mode)
    reference_c2w = np.linalg.inv(reference.world_view_transform.detach().cpu().numpy().T)
    distance = args.focus_distance
    if distance is None:
        distance = estimate_focus_distance(reference, gaussians)
    output = Path(args.output) if args.output else (
        Path(dataset.model_path) / "trackball" / f"ours_{scene.loaded_iter}" /
        f"{args.split}_k{index:05d}")
    # Refuse to mix frames from separate runs or overwrite previous results.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose another --output.")
    streams = ("before", "after") if args.both else ("after",)
    frame_dirs = {name: output / ("renders_" + name if args.both else "renders")
                  for name in streams}
    video_paths = {name: output / ("trackball_" + name + ".mp4" if args.both else "trackball.mp4")
                   for name in streams}
    for directory in frame_dirs.values():
        directory.mkdir()
    height, width = reference.image_height, reference.image_width
    # MPEG-4 needs even dimensions; pad only the bottom/right edges of video.
    video_width, video_height = width + width % 2, height + height % 2
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    poses = []
    print(f"Keyframe {index}: {reference.image_name}; focus distance={distance:.6g}; output={output}")
    with ExitStack() as stack:
        writers = {}
        for name, path in video_paths.items():
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (video_width, video_height))
            stack.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV could not open the MP4 encoder: {path}")
            writers[name] = writer
        for frame in tqdm(range(args.frames), desc="Trackball rendering"):
            phase = 2 * math.pi * frame / args.frames
            pose = trackball_pose(reference_c2w, distance, args.angle, phase)
            camera = camera_at_pose(reference, pose)
            # One rasterization supplies both outputs at exactly the same pose.
            result = render(camera, gaussians, pipeline, background,
                            use_trained_exp=dataset.train_test_exp)
            images = {"after": result["render"]}
            if args.both:
                images["before"] = result["radiance"] * args.radiance_exposure
            for name in streams:
                rendering = images[name]
                if not torch.isfinite(rendering).all():
                    raise RuntimeError(f"Non-finite {name} image at frame {frame}")
                rgb = (rendering.detach().clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(frame_dirs[name] / f"{frame:05d}.png"), bgr):
                    raise RuntimeError(f"Could not write {name} PNG frame {frame}")
                padded = cv2.copyMakeBorder(bgr, 0, video_height - height, 0, video_width - width,
                                            cv2.BORDER_REPLICATE)
                writers[name].write(padded)
            poses.append(pose.tolist())
    for video_path in video_paths.values():
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError(f"MP4 output is missing or empty: {video_path}")
    metadata = dict(backend=BACKEND, split=args.split, keyframe_index=index,
                    both=args.both, hdr_camera=gaussians.hdr_camera,
                    camera_type="gcisp" if gaussians.hdr_camera else None,
                    radiance_activation="clamp_sh_plus_half" if gaussians.hdr_camera else None,
                    before_display="clamp(radiance * radiance_exposure, 0, 1); no gamma or per-frame normalization",
                    radiance_exposure=args.radiance_exposure,
                    videos={name: path.name for name, path in video_paths.items()},
                    image_name=reference.image_name, iteration=scene.loaded_iter,
                    focus_distance=distance, angle_degrees=args.angle, fps=args.fps,
                    frames=args.frames, image_size=[width, height],
                    video_size=[video_width, video_height],
                    reference_c2w=reference_c2w.tolist(), camera_to_world=poses)
    (output / "trajectory.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.frames} PNGs per stream and: " + ", ".join(map(str, video_paths.values())))


def main():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("-k", "--keyframe", help="Zero-based split index, or exact image name")
    parser.add_argument("--keyframe-mode", choices=("index", "name"), default="index")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--list-keyframes", action="store_true")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--angle", type=float, default=10,
                        help="Trackball cone half-angle in degrees, 0 <= angle < 89")
    parser.add_argument("--focus-distance", type=float,
                        help="Focus distance in scene units; default: median in-frustum Gaussian depth")
    parser.add_argument("--output", help="Empty output directory")
    parser.add_argument("--both", action="store_true",
                        help="Save separate pre-GCISP radiance and post-GCISP exposure/CRF RGB PNGs/MP4s")
    parser.add_argument("--radiance-exposure", type=float, default=1.0,
                        help="Fixed multiplier for before-camera display; clamp to [0,1], default 1")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # The repository merger omits arguments whose default is None.
    for name in ('keyframe', 'focus_distance', 'output'):
        if not hasattr(args, name):
            setattr(args, name, None)
    if not args.model_path:
        parser.error("-m/--model_path is required")
    if args.keyframe is None and not args.list_keyframes:
        parser.error("-k/--keyframe is required (or use --list-keyframes)")
    if args.iteration != -1 and args.iteration <= 0:
        parser.error("--iteration must be -1 (latest) or positive")
    if args.frames < 1 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--frames and --fps must be positive")
    if not math.isfinite(args.angle) or not 0 <= args.angle < 89:
        parser.error("--angle must be in [0, 89)")
    if args.focus_distance is not None and (not math.isfinite(args.focus_distance) or args.focus_distance <= 0):
        parser.error("--focus-distance must be finite and positive")
    if not math.isfinite(args.radiance_exposure) or args.radiance_exposure <= 0:
        parser.error("--radiance-exposure must be finite and positive")
    safe_state(args.quiet)
    with torch.no_grad():
        render_trackball(model.extract(args), pipeline.extract(args), args)


if __name__ == "__main__":
    main()
