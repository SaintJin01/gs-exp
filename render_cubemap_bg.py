#
# Render trained Gaussians with a baked environment cubemap composited behind
# them, inside the rasterizer: the patched diff-gaussian-rasterization accepts
# a per-pixel (3, H, W) background, so each pixel's remaining transmittance is
# filled with the cubemap evaluated along its camera ray in a single pass.
#
# Usage:
#   python render_cubemap_bg.py -m output/train_bs_dbp_eval --skip_train
#   (add --cubemap to override; defaults to <source_path>/cubemaps/cubemap)
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from exp_utils.cubemap_bg import CubemapBackground
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, cubemap, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_cubemap")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        background = cubemap.background_for_view(view)
        composite = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

        if train_test_exp:
            composite = composite[..., composite.shape[-1] // 2:]

        torchvision.utils.save_image(composite, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset, iteration, pipeline, skip_train, skip_test, separate_sh, cubemap_dir):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        if not cubemap_dir:
            cubemap_dir = os.path.join(dataset.source_path, "cubemaps", "cubemap")
        cubemap = CubemapBackground(cubemap_dir, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, cubemap, dataset.train_test_exp, separate_sh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, cubemap, dataset.train_test_exp, separate_sh)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render with a baked cubemap composited as the far background")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # get_combined_args drops None-valued CLI args, so use "" as the unset default.
    parser.add_argument("--cubemap", default="", type=str,
                        help="Cubemap directory with posx..negz.png and meta.json "
                             "(default: <source_path>/cubemaps/cubemap)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.cubemap)
