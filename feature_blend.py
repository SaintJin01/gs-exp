"""32D renderer: one host geometry/sort pass, feature-only CUDA backward.

The host renderer supplies its exact forward contributor cutoffs, so alpha
thresholds and early-stop behavior remain unchanged. Geometry is frozen.
An isolated JIT extension leaves the installed RGB rasterizer untouched.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path
import torch
from diff_gaussian_rasterization import _C


@lru_cache(maxsize=1)
def load_extension():
    import torch.utils.cpp_extension as cpp
    if cpp.CUDA_HOME is None and (Path(sys.prefix)/'bin/nvcc').is_file():
        cpp.CUDA_HOME = sys.prefix
    root = Path(__file__).resolve().parent
    os.environ.setdefault('MAX_JOBS', '2')
    return cpp.load(name='gs_sam_feature_blend_v1',
                    sources=[str(root/'feature_blend.cpp'), str(root/'feature_blend.cu')],
                    extra_cflags=['-O3'], extra_cuda_cflags=['-O3'],
                    extra_ldflags=['-L' + str(Path(sys.prefix)/'lib')], verbose=False)


class _Blend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, geometry, bins, image, rendered, height, width, background):
        extension = load_extension()
        ctx.save_for_backward(geometry, bins, image)
        ctx.count, ctx.rendered = features.shape[0], rendered
        return extension.forward(features.contiguous(), geometry, bins, image,
                                  rendered, height, width, float(background))

    @staticmethod
    def backward(ctx, gradient):
        geometry, bins, image = ctx.saved_tensors
        result = load_extension().backward(gradient.contiguous(), geometry, bins,
                                           image, ctx.count, ctx.rendered)
        return result, None, None, None, None, None, None, None


def render_features(camera, gaussians, pipeline, features, white_background, settings):
    empty = torch.empty(0, device=features.device)
    # Only the 3DGS forward is used here; it cannot create geometry gradients.
    with torch.no_grad():
        covariance = gaussians.get_covariance(1.0).detach() if pipeline.compute_cov3D_python else empty
        scales = empty if pipeline.compute_cov3D_python else gaussians.get_scaling.detach()
        rotations = empty if pipeline.compute_cov3D_python else gaussians.get_rotation.detach()
        result = _C.rasterize_gaussians(
            settings.bg, gaussians.get_xyz.detach(),
            torch.zeros((len(features), 3), device=features.device),
            gaussians.get_opacity.detach(), scales, rotations, settings.scale_modifier,
            covariance, settings.viewmatrix, settings.projmatrix, settings.tanfovx,
            settings.tanfovy, settings.image_height, settings.image_width, empty,
            settings.sh_degree, settings.campos, settings.prefiltered,
            settings.antialiasing, settings.debug)
        rendered, _, _, geometry, bins, image, _ = result
    return _Blend.apply(features, geometry, bins, image, rendered,
                         camera.image_height, camera.image_width, white_background)
