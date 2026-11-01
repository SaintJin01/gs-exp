import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False

try:
    from utils.ppisp_utils import PPISPAdapter
except ImportError:
    PPISPAdapter = None


def catmull_rom(values, segment, amount):
    """Uniform Catmull-Rom cubic interpolation through consecutive keyframes."""
    p0 = values[max(segment - 1, 0)]
    p1 = values[segment]
    p2 = values[segment + 1]
    p3 = values[min(segment + 2, len(values) - 1)]
    amount2 = amount * amount
    amount3 = amount2 * amount
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * amount
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * amount2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * amount3
    )


def project_to_rotation(matrix):
    """Project a cubic matrix interpolation back onto SO(3)."""
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


class CameraSpline:
    def __init__(self, cameras):
        if len(cameras) < 2:
            raise ValueError("At least two training cameras are required.")

        resolutions = {(camera.image_width, camera.image_height) for camera in cameras}
        if len(resolutions) != 1:
            raise ValueError("All training cameras must have the same resolution.")

        self.cameras = cameras
        self.width, self.height = next(iter(resolutions))
        self.znear = cameras[0].znear
        self.zfar = cameras[0].zfar
        self.positions = []
        self.rotations = []
        self.fov_x = np.asarray([camera.FoVx for camera in cameras], dtype=np.float64)
        self.fov_y = np.asarray([camera.FoVy for camera in cameras], dtype=np.float64)

        for camera in cameras:
            world_to_camera = camera.world_view_transform.detach().cpu().numpy().T
            camera_to_world = np.linalg.inv(world_to_camera)
            self.positions.append(camera_to_world[:3, 3].copy())
            self.rotations.append(camera_to_world[:3, :3].copy())

        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.rotations = np.asarray(self.rotations, dtype=np.float64)

    def make_camera(self, segment, amount):
        position = catmull_rom(self.positions, segment, amount)
        rotation = project_to_rotation(catmull_rom(self.rotations, segment, amount))
        fov_x = float(np.clip(
            catmull_rom(self.fov_x, segment, amount),
            self.fov_x.min(), self.fov_x.max(),
        ))
        fov_y = float(np.clip(
            catmull_rom(self.fov_y, segment, amount),
            self.fov_y.min(), self.fov_y.max(),
        ))

        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = rotation
        camera_to_world[:3, 3] = position
        world_to_camera = np.linalg.inv(camera_to_world)

        world_view = torch.from_numpy(world_to_camera).float().transpose(0, 1).cuda()
        projection = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=fov_x, fovY=fov_y
        ).transpose(0, 1).cuda()
        full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        camera = MiniCam(
            self.width, self.height, fov_y, fov_x,
            self.znear, self.zfar, world_view, full_projection,
        )
        camera.projection_matrix = projection

        # Exposure/PPISP models require a real dataset identity. Use the nearest
        # keyframe while keeping the interpolated camera pose.
        nearest = self.cameras[segment if amount < 0.5 else segment + 1]
        camera.uid = getattr(nearest, "uid", segment)
        camera.image_name = getattr(nearest, "image_name", str(camera.uid))
        if hasattr(nearest, "ppisp_camera_id"):
            camera.ppisp_camera_id = nearest.ppisp_camera_id
        if hasattr(nearest, "ppisp_frame_name"):
            camera.ppisp_frame_name = nearest.ppisp_frame_name
        return camera


def render_interpolation(dataset, iteration, pipeline, frames_per_interval, separate_sh):
    if frames_per_interval < 1:
        raise ValueError("-fps must be at least 1.")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cameras = scene.getTrainCameras()

        if getattr(dataset, "ppisp", False):
            if PPISPAdapter is None:
                raise RuntimeError("PPISP is enabled, but utils.ppisp_utils is unavailable.")
            if dataset.train_test_exp:
                raise ValueError("--ppisp and --train_test_exp are mutually exclusive.")
            gaussians.ppisp_adapter = PPISPAdapter.load(
                dataset.model_path, scene.loaded_iter, cameras
            )

        spline = CameraSpline(cameras)
        background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
        frame_count = (len(cameras) - 1) * frames_per_interval + 1
        output_path = os.path.join(
            dataset.model_path, f"interpolation_{scene.loaded_iter}.mp4"
        )

        writer = None
        progress = tqdm(total=frame_count, desc="Rendering interpolation")
        try:
            for frame_index in range(frame_count):
                if frame_index == frame_count - 1:
                    segment, amount = len(cameras) - 2, 1.0
                else:
                    segment = frame_index // frames_per_interval
                    amount = (frame_index % frames_per_interval) / frames_per_interval

                camera = spline.make_camera(segment, amount)
                rendering = render(
                    camera, gaussians, pipeline, background,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=separate_sh,
                )["render"]
                if dataset.train_test_exp:
                    rendering = rendering[..., rendering.shape[-1] // 2:]

                frame = (
                    rendering.clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .byte()
                    .cpu()
                    .numpy()
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        frames_per_interval,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open MP4 writer for {output_path}")
                writer.write(frame)
                progress.update(1)
        finally:
            progress.close()
            if writer is not None:
                writer.release()

        print(f"Saved {frame_count} frames to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render cubic camera interpolation to MP4")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument(
        "-fps", type=int, default=30,
        help="Interpolated frames per camera interval; also used as MP4 FPS",
    )
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print(f"Rendering camera interpolation for {args.model_path}")
    safe_state(args.quiet)
    render_interpolation(
        model.extract(args), args.iteration, pipeline.extract(args), args.fps,
        SPARSE_ADAM_AVAILABLE,
    )
