import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from utils.fsgs_depth import estimate_depth, prepare_depth_priors, load_prior, fsgs_depth_loss

class FSGSDepthTests(unittest.TestCase):
    def test_loss_matches_reference_values_and_gradients(self):
        p=torch.tensor([20.,33.,100.,180.,250.],dtype=torch.double)
        for values in ([9.,7.,4.,3.,2.],[2.,1.,5.,4.,7.]):
            d=torch.tensor(values,dtype=torch.double,requires_grad=True)
            got=fsgs_depth_loss(d,p)
            reference=torch.minimum(1-torch.corrcoef(torch.stack((d,-p)))[0,1],
                       1-torch.corrcoef(torch.stack((d,1/(p+200))))[0,1])
            torch.testing.assert_close(got,reference)
            a=torch.autograd.grad(got,d,retain_graph=True)[0]
            b=torch.autograd.grad(reference,d)[0]
            torch.testing.assert_close(a,b)
            self.assertTrue(torch.autograd.gradcheck(lambda x:fsgs_depth_loss(x,p),(d,)))

    def test_invalid_and_flat_inputs(self):
        for p in (torch.ones(4),torch.full((4,),float("nan"))):
            d=torch.arange(4.).requires_grad_()
            value=fsgs_depth_loss(d,p)
            self.assertTrue(torch.isfinite(value))
            value.backward()
            self.assertTrue(torch.isfinite(d.grad).all())
        d=torch.ones(4,requires_grad=True)
        value=fsgs_depth_loss(d,torch.arange(4.))
        self.assertEqual(value.item(),0.)
        value.backward()
        self.assertTrue(torch.isfinite(d.grad).all())

    def test_preprocessing_exactly_matches_fsgs(self):
        class Probe(torch.nn.Module):
            def forward(self,x):
                self.seen=x.clone()
                return x[:,0]*2+x[:,1]
        model=Probe()
        image=torch.linspace(0,1,3*7*9).reshape(3,7,9)
        actual=estimate_depth(model,image)
        x=F.interpolate((image[None]-.5)/.5,size=(384,512),mode="bicubic",align_corners=False)
        expected=F.interpolate((x[:,0]*2+x[:,1]).unsqueeze(1),
                               size=(7,9),mode="bicubic",align_corners=False)[0]
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(model.seen,x)
        self.assertFalse(actual.requires_grad)

    def test_cache_reuses_and_invalidates_pixels(self):
        class Fake(torch.nn.Module):
            def forward(self,x):
                return x[:,0]*200+350
        calls=[]
        def loader(device):
            calls.append(device)
            return Fake()
        cam=SimpleNamespace(original_image=torch.rand(3,5,6),image_name="frame")
        with tempfile.TemporaryDirectory() as directory:
            prepare_depth_priors([cam],directory,"cpu",loader)
            first=cam.fsgs_depth_path
            prior=load_prior(cam,"cpu")
            self.assertGreater(prior.max().item(),1.) # Raw prior, no normalization.
            prepare_depth_priors([cam],directory,"cpu",loader)
            self.assertEqual(len(calls),1)
            torch.testing.assert_close(load_prior(cam,"cpu"),prior)
            cam.original_image=cam.original_image*.5
            prepare_depth_priors([cam],directory,"cpu",loader)
            self.assertNotEqual(first,cam.fsgs_depth_path)
            self.assertEqual(len(calls),2)

if __name__=="__main__":
    unittest.main()
