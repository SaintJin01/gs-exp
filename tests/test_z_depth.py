import unittest
import torch
from utils.fsgs_depth import rasterize_z_depth

class ZDepthTests(unittest.TestCase):
    def test_two_pass_matches_joint_compositing_gradients(self):
        # Differentiable analytic rasterizer with geometry-dependent alpha.
        # Compare outputs and every input gradient against joint RGB/z blending.
        torch.manual_seed(5)
        xyz=torch.randn(4,3,dtype=torch.double,requires_grad=True)
        screen=torch.randn(4,3,dtype=torch.double,requires_grad=True)
        opacity=torch.randn(4,1,dtype=torch.double,requires_grad=True)
        scales=torch.randn(4,3,dtype=torch.double,requires_grad=True)
        rotations=torch.randn(4,4,dtype=torch.double,requires_grad=True)
        colors=torch.randn(4,3,dtype=torch.double,requires_grad=True)
        view=torch.eye(4,dtype=torch.double);view[3,2]=10
        def weights():
            alpha=torch.sigmoid(opacity[:,0]+.1*xyz[:,0]+.2*screen[:,0]+
                                .1*scales.sum(1)+.1*rotations.sum(1))
            return alpha*torch.cat((alpha.new_ones(1),torch.cumprod(1-alpha,0)[:-1]))
        def rasterizer(colors_precomp, **geometry):
            self.assertIs(geometry["means2D"],screen)
            return ((weights()[:,None]*colors_precomp).sum(0)[:,None,None],None,None)
        geometry=dict(means3D=xyz,means2D=screen,opacities=opacity,
                      scales=scales,rotations=rotations,shs=None,cov3D_precomp=None)
        depth=rasterize_z_depth(rasterizer,view,geometry)
        rgb=rasterizer(colors_precomp=colors,**geometry)[0]
        loss=rgb.square().sum()+depth.square().sum()
        params=(xyz,screen,opacity,scales,rotations,colors)
        separate=torch.autograd.grad(loss,params)
        payload=torch.cat((colors,(xyz@view[:3,2]+view[3,2])[:,None]),1)
        joint=(weights()[:,None]*payload).sum(0)
        torch.testing.assert_close(depth.flatten(),joint[3:])
        combined=torch.autograd.grad(joint.square().sum(),params)
        for a,b in zip(separate,combined):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

if __name__=="__main__":
    unittest.main()
