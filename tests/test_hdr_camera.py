import unittest
import tempfile
from pathlib import Path
import torch
from scene.gaussian_model import GaussianModel
from utils.hdr_utils import sh_to_radiance, dk_gradient_weights, scale_densification_gradient
from utils.sh_utils import C0, RGB2SH


class HDRCameraTests(unittest.TestCase):
    def test_radiance_initialization_and_unbounded_gradient(self):
        rgb = torch.tensor([0., .04, .5, 1.])
        torch.testing.assert_close(sh_to_radiance(RGB2SH(rgb)*C0), rgb)
        raw = torch.tensor([-20., 0., 100.], requires_grad=True)
        values = sh_to_radiance(raw)
        torch.testing.assert_close(values, torch.tensor([0., .5, 100.5]))
        values.sum().backward()
        torch.testing.assert_close(raw.grad, torch.tensor([0., 1., 1.]))

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


if __name__ == '__main__':
    unittest.main()
