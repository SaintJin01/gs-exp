import unittest
import torch
from utils.adgs_utils import adgs_phase, opacity_reset_due, edge_aware_depth_loss, rasterize_z_depth

class ADGSTests(unittest.TestCase):
    def test_boundaries(self):
        expected = {1500:"warmup",1501:"low",1599:"low",1600:"high",
                    1699:"high",1700:"low",29999:"low",30000:"high"}
        for i, phase in expected.items():
            self.assertEqual(adgs_phase(i)[0], phase)
        self.assertEqual(adgs_phase(30001, True), ("controller", False, None, None))
        self.assertEqual(adgs_phase(1700)[2:], (0.0005, 0.1))
        events = [i for i in range(1,35001) if i > 500 and i < 10000 and i % 100 == 0]
        self.assertEqual(events[-1], 9900)
        self.assertEqual([i for i in events if adgs_phase(i)[0]=="low"][0],1700)
        self.assertEqual([i for i in range(1,10000) if opacity_reset_due(i)],
                         [501,3501,6501,9501])
        self.assertFalse(opacity_reset_due(30501, True))

    def test_loss_and_gradient(self):
        d=torch.tensor([[[1.,2.],[3.,4.]]],requires_grad=True)
        rgb=torch.zeros(3,2,2)
        loss=edge_aware_depth_loss(d,rgb)
        torch.testing.assert_close(loss,torch.tensor(.01*(3.-.001*3.)))
        self.assertTrue(torch.autograd.gradcheck(
            lambda x:edge_aware_depth_loss(x,rgb.double()),(d.detach().double().requires_grad_(),)))
        self.assertLess(edge_aware_depth_loss(d,torch.tensor([[[0.,1.],[0.,1.]]]).expand(3,-1,-1)).item(),loss.item())

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
