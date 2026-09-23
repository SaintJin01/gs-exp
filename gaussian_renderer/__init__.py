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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp = False, bpc: GaussianModel = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    primary_screenspace_points = torch.zeros_like(pc.get_means3D, dtype=pc.get_means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        primary_screenspace_points.retain_grad()
    except:
        pass

    # Separate bgaussians' gradients from gaussians for densification
    if bpc is not None:
        bscreenspace_points = torch.zeros_like(bpc.get_means3D, dtype=bpc.get_means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            bscreenspace_points.retain_grad()
        except:
            pass
        screenspace_points = torch.cat([primary_screenspace_points, bscreenspace_points], dim=0,)
    else:
        screenspace_points = primary_screenspace_points
        bscreenspace_points = None

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    active_sh_degree = pc.active_sh_degree
    if bpc is not None:
        if pc.max_sh_degree != bpc.max_sh_degree:
            raise ValueError("Foreground and background must use the same max SH degree.")
        active_sh_degree = max(pc.active_sh_degree, bpc.active_sh_degree)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if bpc is None:
        means3D = pc.get_means3D
        opacity = pc.get_opacity
    else:
        means3D = torch.cat([pc.get_means3D, bpc.get_means3D], dim=0)
        opacity = torch.cat([pc.get_opacity, bpc.get_opacity], dim=0)
    means2D = screenspace_points
    foreground_count = pc.get_means3D.shape[0]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        if bpc is None:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            cov3D_precomp = torch.cat([pc.get_covariance(scaling_modifier), bpc.get_covariance(scaling_modifier)], dim=0)
    else:
        if bpc is None:
            scales = pc.get_render_scaling
            rotations = pc.get_rotation
        else:
            scales = torch.cat([pc.get_render_scaling, bpc.get_render_scaling], dim=0)
            rotations = torch.cat([pc.get_rotation, bpc.get_rotation], dim=0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    dc = None
    colors_precomp = None
    if override_color is None:
        if bpc is None:
            features = pc.get_features
            features_dc = pc.get_features_dc
            features_rest = pc.get_features_rest
        else:
            features = torch.cat([pc.get_features, bpc.get_features], dim=0)
            features_dc = torch.cat([pc.get_features_dc, bpc.get_features_dc], dim=0)
            features_rest = torch.cat([pc.get_features_rest, bpc.get_features_rest], dim=0)
            
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.unsqueeze(0))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = features_dc, features_rest
            else:
                shs = features
    else:
        if override_color.shape[0] != means3D.shape[0]:
            raise ValueError(
                "override_color must contain one color per combined Gaussian."
            )
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

    foreground_radii = radii[:foreground_count]
    if bpc is not None:
        bradii = radii[foreground_count:]
    else:
        bradii = None
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": primary_screenspace_points,
        "visibility_filter" : (foreground_radii > 0).nonzero(),
        "radii": foreground_radii,

        # for composite renderer debugging
        "all_viewspace_points": screenspace_points,
        "all_radii": radii,

        # for train background
        "background_viewspace_points": bscreenspace_points,
        "background_radii": bradii,
        "background_visibility_filter": ((bradii > 0).nonzero() if bradii is not None else None),
        
        "depth" : depth_image
        }
    
    return out
