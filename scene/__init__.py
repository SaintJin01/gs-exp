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

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_background_cameras=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.background_train_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.initial_point_cloud = scene_info.point_cloud
        self.initial_train_camera_infos = scene_info.train_cameras
        self.sfm_gaussians_loaded = False

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        background_train_camera_infos = []

        if load_background_cameras:
            background_image_dir = os.path.join(args.source_path, "compensation", "depth_masked")
            if not os.path.isdir(background_image_dir):
                raise FileNotFoundError("Background image directory not found: {}:".format(background_image_dir))
            for camera_info in scene_info.train_cameras:
                image_stem = os.path.splitext(os.path.basename(camera_info.image_name))[0]
                background_image_path = os.path.join(background_image_dir, image_stem + ".png")
                if not os.path.isfile(background_image_path):
                    raise FileNotFoundError("Background image not found for '{}': {}".format(camera_info.image_name, background_image_path))
                background_train_camera_infos.append(camera_info._replace(image_path=background_image_path))

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            if load_background_cameras:
                print("Loading Background Training Cameras")
                self.background_train_cameras[resolution_scale] = cameraList_from_camInfos(background_train_camera_infos, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
            self.sfm_gaussians_loaded = True

    def initialize_sfm_gaussians(self):
        if self.sfm_gaussians_loaded:
            print("SfM gaussians already loaded, Skip initializing SfM gaussians")
            return
            
        self.gaussians.create_from_pcd(self.initial_point_cloud, self.initial_train_camera_infos, self.cameras_extent)
        self.sfm_gaussians_loaded = True

    def save_combined_ply(self, path, bgaussians):
        if self.gaussians.max_sh_degree != bgaussians.max_sh_degree:
            raise RuntimeError("Cannot merge foreground/background with different SH degrees: {} vs {}".format(self.gaussians.max_sh_degree, bgaussians.max_sh_degree))
        combined = GaussianModel(self.gaussians.max_sh_degree)
        combined._xyz = torch.cat([self.gaussians.get_means3D.detach(), bgaussians.get_means3D.detach()], dim=0)
        combined._features_dc = torch.cat([self.gaussians._features_dc.detach(), bgaussians._features_dc.detach()], dim=0)
        combined._features_rest = torch.cat([self.gaussians._features_rest.detach(), bgaussians._features_rest.detach()], dim=0)
        combined._opacity = torch.cat([self.gaussians._opacity.detach(), bgaussians._opacity.detach()], dim=0)
        combined._scaling = torch.cat([self.gaussians._scaling.detach(), torch.log(bgaussians.get_render_scaling.detach())], dim=0)
        combined._rotation = torch.cat([self.gaussians._rotation.detach(), bgaussians._rotation.detach()], dim=0)
        combined.active_sh_degree = max(self.gaussians.active_sh_degree, bgaussians.active_sh_degree)
        combined.save_ply(path)
        foreground_count = self.gaussians.get_xyz.shape[0]
        background_count = bgaussians.get_xyz.shape[0]
        print("Saved combined PLY: {} foreground + {} background = {} Gaussians".format(foreground_count, background_count, foreground_count + background_count))

    def save(self, iteration, bgaussians=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if bgaussians is not None:
            torch.save(bgaussians.capture(), os.path.join(point_cloud_path, "bgaussians.pth"))
            bgaussians.save_ply(os.path.join(point_cloud_path, "bgaussians.ply"))
            self.save_combined_ply(os.path.join(point_cloud_path, "combined.ply"), bgaussians)
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def save_bgaussians(self, bgaussians):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/bgaussians")
        os.makedirs(point_cloud_path, exist_ok=True)
        torch.save(bgaussians.capture(), os.path.join(point_cloud_path, "bgaussians.pth"))
        bgaussians.save_ply(os.path.join(point_cloud_path, "bgaussians.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getBackgroundTrainCameras(self, scale=1.0):
        if scale not in self.background_train_cameras:
            raise RuntimeError("Background cameras were not loaded. \nConstruct Scene with load_background_cameras=True.")
        return self.background_train_cameras[scale]

    def releaseBackgroundTrainCameras(self):
        self.background_train_cameras.clear()