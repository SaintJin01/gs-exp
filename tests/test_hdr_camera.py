import unittest
import tempfile
from pathlib import Path
import torch
from scene.camera_cnn import CameraCNN
from scene.gaussian_model import GaussianModel
from utils.hdr_utils import srgb_to_linear, radiance_to_sh_dc, sh_to_radiance, dk_gradient_weights, scale_densification_gradient
from utils.sh_utils import C0


class HDRCameraTests(unittest.TestCase):
    def test_radiance_initialization_and_unbounded_gradient(self):
        rgb = torch.tensor([0., .04, .5, 1.])
        expected = srgb_to_linear(rgb).clamp_min(1e-6)
        torch.testing.assert_close(sh_to_radiance(radiance_to_sh_dc(expected)*C0), expected)
        raw = torch.tensor([-20., 0., 100.], requires_grad=True)
        values = sh_to_radiance(raw)
        self.assertGreater(values[-1].item(), 99.)
        values.sum().backward()
        self.assertTrue(torch.all(raw.grad > 0))

    def test_camera_shape_and_end_to_end_gradient(self):
        torch.manual_seed(7)
        model = CameraCNN()
        self.assertFalse(any(isinstance(m, (torch.nn.MultiheadAttention, torch.nn.TransformerEncoderLayer)) for m in model.modules()))
        radiance = (torch.rand(3, 17, 23)*8).requires_grad_()
        result = model(radiance)
        self.assertEqual(result.shape, radiance.shape)
        self.assertTrue(torch.all((result > 0) & (result < 1)))
        (result - torch.rand_like(result)).square().mean().backward()
        self.assertGreater(radiance.grad.abs().sum().item(), 0)
        for block in model.blocks:
            self.assertGreater(block.conv1.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model(torch.ones(2, 3, 1, 3)).shape, (2, 3, 1, 3))

    def test_dk_scaling_schedule_and_gradient_isolation(self):
        xyz = torch.tensor([[0., 0., .5], [0., 0., 2.]], requires_grad=True)
        view = torch.eye(4)
        torch.testing.assert_close(dk_gradient_weights(xyz, view, 1., 0), torch.tensor([.25, 1.]))
        torch.testing.assert_close(dk_gradient_weights(xyz, view, 1., 4000), torch.tensor([.625, 1.]))
        torch.testing.assert_close(dk_gradient_weights(xyz, view, 1., 7000), torch.ones(2))
        screen = torch.zeros_like(xyz, requires_grad=True)
        (xyz.sum() + screen.sum()).backward()
        original = xyz.grad.clone()
        scale_densification_gradient(screen, dk_gradient_weights(xyz, view, 1., 0))
        torch.testing.assert_close(xyz.grad, original)
        torch.testing.assert_close(screen.grad[:, 0], torch.tensor([.25, 1.]))
        view[3, 2] = .5
        torch.testing.assert_close(dk_gradient_weights(xyz, view, 1., 0), torch.ones(2))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by GaussianModel')
    def test_camera_sidecar_roundtrip_and_missing_file(self):
        source = GaussianModel(3)
        source.configure_camera(True)
        source.camera_model.eval()
        radiance = torch.rand(3, 13, 19, device='cuda') * 5
        with tempfile.TemporaryDirectory() as directory:
            ply = str(Path(directory) / 'point_cloud.ply')
            source.save_camera(ply)
            target = GaussianModel(3)
            target.load_camera(ply)
            self.assertTrue(target.hdr_camera)
            torch.testing.assert_close(source.camera_model(radiance), target.camera_model(radiance))
            Path(directory, 'camera_cnn.pth').unlink()
            with self.assertRaises(FileNotFoundError):
                target.load_camera(ply)
            torch.save({'version': 1}, Path(directory, 'camera_transformer.pth'))
            with self.assertRaisesRegex(ValueError, 'Transformer'):
                target.load_camera(ply)
            with self.assertRaisesRegex(ValueError, 'Transformer'):
                target.restore({'version': 1}, None)


if __name__ == '__main__':
    unittest.main()
