"""Regression tests for coarse single-mask 16D training."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from prepare_sam_pairs import retain_single_masks, mask_preview, build_parser
from saga_supervision import sample_single_supervision, contrastive_loss
import train_sam_features as training

class SingleMaskTests(unittest.TestCase):
    def test_coarse_defaults_and_preview(self):
        args = build_parser()[0].parse_args([])
        self.assertEqual(args.points_per_side, 8)
        self.assertEqual(args.min_segment_fraction, .01)
        self.assertTrue(args.save_previews)
        self.assertEqual(training.build_parser()[0].parse_args([]).feature_dim, 16)
        big = np.zeros((20,20), bool); big[:15,:15] = True
        small = np.zeros_like(big); small[2:5,2:5] = True
        other = np.zeros_like(big); other[10:,10:] = True
        def annotation(mask):
            return dict(segmentation=mask, area=int(mask.sum()), bbox=[0,0,20,20],
                        predicted_iou=.99, stability_score=.99)
        masks, packed, coverage, records = retain_single_masks(
            [annotation(small),annotation(other),annotation(big)],20,20,4)
        self.assertEqual(len(masks),2)
        self.assertLessEqual(masks.sum(0).max(),1)
        self.assertEqual(int(masks[0].sum()),225)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'preview.png'
            mask_preview(np.full((20,20,3),128,np.uint8),masks,output)
            self.assertGreater(output.stat().st_size,100)
            Image.open(str(output)).verify()

    def test_single_pairs_and_backward(self):
        masks = np.zeros((2,4,4),bool)
        masks[0,:,:2] = True; masks[1,:,2:] = True
        packed = np.packbits(masks.reshape(2,-1),axis=1)
        coverage = np.bitwise_or.reduce(packed,axis=0)
        supervision = sample_single_supervision(packed,coverage,[8,8],4,4,16,'cpu',1.)
        truth = supervision[1][0]
        self.assertEqual(float(truth[0,1]),1.)
        self.assertEqual(float(truth[0,2]),0.)
        feature_map = torch.randn(16,4,4,requires_grad=True)
        loss, stats = contrastive_loss(feature_map,supervision,None,None,(4,4),collect_cosine=True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(feature_map.grad).all())
        self.assertIn('positive_cosine',stats)

    def test_reject_overlapping_supervision(self):
        packed=np.full((2,2),255,np.uint8)
        with self.assertRaises(ValueError):
            sample_single_supervision(packed,packed[0],[16,16],4,4,16,'cpu',1.)

    def test_cuda_16d_renderer_matches_rgb(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from types import SimpleNamespace
        from save_q3_gaussians_all_cameras import camera_from_json_record
        camera = camera_from_json_record(dict(id=0,img_name='test',width=32,height=32,
                        position=[0,0,0],rotation=np.eye(3).tolist(),fx=25.,fy=25.))
        cloud = training.GaussianModel(0)
        cloud._xyz = torch.tensor([[0.,0.,2.],[.1,.1,2.5],[-.2,0.,3.]],device='cuda')
        cloud._scaling = torch.full((3,3),-2.,device='cuda')
        cloud._rotation = torch.tensor([[1.,0.,0.,0.]]*3,device='cuda')
        cloud._opacity = torch.full((3,1),1.,device='cuda')
        pipe = SimpleNamespace(compute_cov3D_python=False,debug=False,antialiasing=False,
                               feature_backend='feature')
        features = torch.randn(3,16,device='cuda',requires_grad=True)
        fast = training.render_feature_map(camera,cloud,pipe,features)
        pipe.feature_backend='rgb'
        rgb = training.render_feature_map(camera,cloud,pipe,features)
        torch.testing.assert_allclose(fast,rgb,rtol=2e-5,atol=2e-6)
        upstream = torch.randn_like(fast)
        a=torch.autograd.grad(fast,features,upstream)[0]
        b=torch.autograd.grad(rgb,features,upstream)[0]
        torch.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-5)

if __name__ == '__main__':
    unittest.main()

