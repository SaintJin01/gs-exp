import unittest
import tempfile
from pathlib import Path
import torch
from scene.gaussian_model import GaussianModel
from utils.hdr_utils import sh_to_radiance
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



if __name__ == '__main__':
    unittest.main()
