"""Serve a trained gs-exp model plus heightfield cubemap to the SIBR network viewer."""
from __future__ import annotations

import argparse
import time

import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, network_gui, render
from scene import Scene
from utils.general_utils import safe_state
from exp_utils.heightfield_bg import CubemapBackground

DEFAULT_CUBEMAP = "/home/sj/work/gs-dataset/tandt/train/cubemaps/ablation_01_sky_border_4096_gt25m"


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--cubemap", default=DEFAULT_CUBEMAP)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    data = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(data.sh_degree)
    scene = Scene(data, gaussians, load_iteration=args.iteration, shuffle=False)
    cubemap = CubemapBackground(args.cubemap, device="cuda")
    network_gui.init(args.ip, args.port)
    print(f"[heightfield viewer] listening on {args.ip}:{args.port}, iteration {scene.loaded_iter}", flush=True)
    while True:
        if network_gui.conn is None:
            network_gui.try_connect()
            time.sleep(0.01)
            continue
        try:
            camera, _, pipe.convert_SHs_python, pipe.compute_cov3D_python, _, scale = network_gui.receive()
            if camera is None:
                network_gui.send(None, data.source_path)
                continue
            with torch.no_grad():
                bg = cubemap.background_for_view(camera)
                frame = render(camera, gaussians, pipe, bg, scaling_modifier=scale,
                               use_trained_exp=data.train_test_exp)["render"]
                pixels = memoryview((frame.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(pixels, data.source_path)
        except (ConnectionError, OSError, ValueError) as exc:
            print(f"[heightfield viewer] disconnected: {exc}", flush=True)
            network_gui.conn = None


if __name__ == "__main__":
    main()
