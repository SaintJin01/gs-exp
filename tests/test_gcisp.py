import math
import tempfile
import unittest
from pathlib import Path
import torch
from scene.camera_gcisp import GCISP, apply_crf
from scene.gaussian_model import GaussianModel


class GCISPTests(unittest.TestCase):
    def model(self):
        return GCISP(['a', 'b'], [7])

    def test_identity_and_exposure_stops(self):
        m = self.model()
        x = torch.linspace(0, 2, 60).reshape(3,4,5)
        torch.testing.assert_close(m(x, frame_name='a'), x.clamp(0,1))
        with torch.no_grad(): m.exposure_params[0] = 1
        torch.testing.assert_close(m(x, frame_name='a'), (2*x).clamp(0,1))
        self.assertFalse(any('color' in n or 'vignett' in n or 'attention' in n for n,_ in m.named_parameters()))

    def test_crf_formula_and_gradcheck(self):
        torch.manual_seed(2)
        params = torch.randn(3,4,dtype=torch.double,requires_grad=True)
        x = torch.tensor([.1,.2,.8,.9],dtype=torch.double).repeat(3,1).reshape(3,2,2).requires_grad_()
        expected = torch.zeros_like(x)
        for c in range(3):
            t,s,g = [a+math.log1p(math.exp(float(params[c,i]))) for i,a in enumerate([.3,.3,.1])]
            center = 1/(1+math.exp(-float(params[c,3])))
            a = s*center/(t+center*(s-t))
            for y in range(2):
                for z in range(2):
                    v=float(x[c,y,z])
                    expected[c,y,z]=(a*(v/center)**t if v<=center else 1-(1-a)*((1-v)/(1-center))**s)**g
        torch.testing.assert_close(apply_crf(x,params),expected)
        self.assertTrue(torch.autograd.gradcheck(apply_crf,(x,params)))

    def test_endpoint_gradients_are_finite(self):
        params = torch.full((3,4),-3.,requires_grad=True)
        x = torch.tensor([0.,1e-9,.5,1.,2.]).repeat(3,1).reshape(3,1,5).requires_grad_()
        out=apply_crf(x,params);out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(params.grad).all())
        self.assertTrue((out[...,0]==0).all())
        self.assertTrue((out[...,-1]==1).all())

    def test_two_stage_gradient_routing(self):
        m=self.model()
        x=torch.full((3,9,11),.25,requires_grad=True)
        self.assertFalse(m.set_stage(7,10))
        m(x,frame_name='a').sum().backward()
        self.assertGreater(x.grad.abs().sum(),0)
        self.assertIsNotNone(m.exposure_params.grad)
        self.assertIsNone(m.controllers[0].exposure_head.weight.grad)
        m.zero_grad(set_to_none=True);x.grad=None
        self.assertTrue(m.set_stage(8,10))
        m(x,frame_name='a').sum().backward()
        self.assertIsNone(x.grad)
        self.assertIsNone(m.crf_params.grad)
        self.assertIsNone(m.exposure_params.grad)
        self.assertGreater(m.controllers[0].exposure_head.weight.grad.abs().sum(),0)
        m.mark_controller_update(); self.assertEqual(m.controller_updates.item(),1)
        m.eval()
        torch.testing.assert_close(m(x),m(x,frame_name='b'))

    def test_multicamera_and_untrained_inference(self):
        m=GCISP(['a','b'],[1,2]).eval()
        x=torch.full((3,1,2),.25)
        torch.testing.assert_close(m(x,camera_id=1),x)
        with self.assertRaisesRegex(ValueError,'Unknown'): m(x,camera_id=3)
        m.train();m.set_stage(8,10)
        self.assertTrue(torch.isfinite(m(x,camera_id=2)).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_sidecar_roundtrip_and_legacy_rejection(self):
        source=GaussianModel(3)
        source.configure_camera(True,frame_names=['a','b'],camera_ids=[7])
        source.camera_model.controller_updates.fill_(1)
        x=torch.rand(3,9,11,device='cuda')
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'point_cloud.ply')
            source.save_camera(path)
            target=GaussianModel(3);target.load_camera(path)
            source.camera_model.eval()
            torch.testing.assert_close(source.camera_model(x),target.camera_model(x))
            sidecar = Path(d, 'camera_gcisp.pth')
            payload = torch.load(str(sidecar))
            self.assertEqual(payload['radiance_activation'], 'clamp_sh_plus_half')
            payload.update(version=3, radiance_activation='softplus_sh')
            torch.save(payload, str(sidecar))
            with self.assertRaisesRegex(ValueError, 'old softplus'):
                target.load_camera(path)
            with self.assertRaisesRegex(ValueError, 'old softplus'):
                target.restore(dict(version=3, camera_type='gcisp'), None)
            sidecar.unlink()
            with self.assertRaises(FileNotFoundError): target.load_camera(path)
            Path(d,'camera_transformer.pth').touch()
            with self.assertRaisesRegex(ValueError,'Legacy'): target.load_camera(path)


if __name__=='__main__': unittest.main()
