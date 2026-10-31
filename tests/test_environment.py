import unittest
from argparse import Namespace
import numpy as np
import torch
from scene.environment_gaussian import phase_at, EnvironmentGaussianModel, parallax_sphere_radius
from utils.background_masks import make_mixed_masks


class EnvironmentTests(unittest.TestCase):
    def options(self):
        return Namespace(env_train_iterations=10000,env_scaled_iterations=7000,
                         env_joint_iterations=20000,env_controller_iterations=3000)

    def test_parallax_radius_units_and_scaling(self):
        centers=np.array([[-1.,0.,0.],[1.,0.,0.]])
        radius,baseline,focal=parallax_sphere_radius(centers,[1000.,800.],5.)
        self.assertEqual((radius,baseline,focal),(400.,2.,1000.))
        self.assertEqual(parallax_sphere_radius(centers+20,[1000.,800.],5.)[0],400.)
        self.assertEqual(parallax_sphere_radius(centers,[1000.,800.],10.)[0],200.)
        self.assertEqual(parallax_sphere_radius(centers,[500.,400.],5.)[0],200.)
        with self.assertRaises(ValueError):
            parallax_sphere_radius(np.zeros((2,3)),[1000.],5.)

    def test_stage_boundaries(self):
        for step,phase,local in [(1,'sfm_scaled',1),(7000,'sfm_scaled',7000),
                (7001,'joint',7001),(10000,'joint',10000),(10001,'joint',10001),
                (27000,'joint',27000),(27001,'controller',27001),(30000,'controller',30000)]:
            self.assertEqual(phase_at(step,self.options()),(phase,local))

    def test_joint_tensors_enabled_until_controller(self):
        from scene.camera_gcisp import GCISP
        model=EnvironmentGaussianModel(0,'default',self.options())
        model.camera_model=GCISP(['frame'],[0])
        for step in (1,10000,10001,27000):
            self.assertFalse(model.set_camera_stage(step))
            self.assertTrue(model._xyz.requires_grad)
            self.assertTrue(model.camera_model.crf_params.requires_grad)
        self.assertTrue(model.set_camera_stage(27001))
        self.assertFalse(model._xyz.requires_grad)
        self.assertFalse(model.camera_model.crf_params.requires_grad)

    def test_environment_stage_freezing_with_adam_momentum(self):
        model=EnvironmentGaussianModel(0,'default',self.options())
        model.env_count=1
        names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
        for name in names:
            setattr(model,name,torch.nn.Parameter(torch.ones(2,3)))
        params=[getattr(model,name) for name in names]
        model.optimizer=torch.optim.Adam(params,lr=.1)
        # Populate momentum as if resuming an earlier configuration.
        sum(p.sum() for p in params).backward()
        model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
        old=[p.detach().clone() for p in params]
        sum(p.sum() for p in params).backward()
        model.freeze_environment_parameters(1)
        model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
        for name,p,previous in zip(names,params,old):
            self.assertEqual(torch.equal(p[0],previous[0]),name=='_rotation')
        old=[p.detach().clone() for p in params]
        for _ in range(3):
            sum(p.sum() for p in params).backward()
            model.freeze_environment_parameters(10001)
            model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
        for p,previous in zip(params,old):
            self.assertTrue(torch.equal(p[0],previous[0]))
            self.assertFalse(torch.equal(p[1],previous[1]))
        self.assertFalse(hasattr(model,'_w'))

    def test_joint_optimizer_keeps_environment_and_scene_learning_rates(self):
        model=EnvironmentGaussianModel(0,'default',self.options())
        model.env_count=1
        model._scaling=torch.nn.Parameter(torch.ones(2,3,dtype=torch.float64))
        model.optimizer=torch.optim.Adam([{'params':[model._scaling],'name':'scaling','lr':.005}],eps=1e-15)
        env=torch.nn.Parameter(torch.ones(1,3,dtype=torch.float64))
        scene=torch.nn.Parameter(torch.ones(1,3,dtype=torch.float64))
        env_opt=torch.optim.Adam([env],lr=.0001,eps=1e-15)
        scene_opt=torch.optim.Adam([scene],lr=.005,eps=1e-15)
        for step in range(1,5):
            model._scaling.grad=torch.full_like(model._scaling,float(step))
            env.grad=torch.full_like(env,float(step));scene.grad=torch.full_like(scene,float(step))
            model.step_joint_optimizer(step);env_opt.step();scene_opt.step()
        torch.testing.assert_close(model._scaling[:1],env,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(model._scaling[1:],scene,rtol=1e-12,atol=1e-12)

    def test_mixed_mask_invalid_consensus(self):
        relative=np.array([[-1,1,2],[3,10,30]],np.float32)
        metric=np.array([[-1,1,-1],[3,10,30]],np.float32)
        masks=make_mixed_masks(relative,metric)
        self.assertEqual(len(masks),12)
        self.assertTrue(masks['mix_invalid_consensus'][0,0])
        self.assertFalse(masks['mix_invalid_consensus'][0,2])
        for mask in masks.values():
            self.assertEqual(mask.shape,relative.shape)


if __name__=='__main__': unittest.main()
