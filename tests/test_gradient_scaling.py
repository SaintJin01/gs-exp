import unittest

import torch

from utils.gradient_scaling import dk_gradient_weights, scale_densification_gradient


class GradientScalingTests(unittest.TestCase):
    def test_schedule_and_gradient_isolation(self):
        xyz = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 2.0]], requires_grad=True)
        view = torch.eye(4)
        torch.testing.assert_close(dk_gradient_weights(xyz, view, 1.0, 0), torch.tensor([0.25, 1.0]))
        torch.testing.assert_close(dk_gradient_weights(xyz, view, 1.0, 4000), torch.tensor([0.625, 1.0]))
        torch.testing.assert_close(dk_gradient_weights(xyz, view, 1.0, 7000), torch.ones(2))

        screen = torch.zeros_like(xyz, requires_grad=True)
        (xyz.sum() + screen.sum()).backward()
        xyz_gradient = xyz.grad.clone()
        scale_densification_gradient(screen, dk_gradient_weights(xyz, view, 1.0, 0))

        torch.testing.assert_close(xyz.grad, xyz_gradient)
        torch.testing.assert_close(screen.grad[:, 0], torch.tensor([0.25, 1.0]))

    def test_validation(self):
        xyz = torch.zeros(1, 3)
        view = torch.eye(4)
        with self.assertRaises(ValueError):
            dk_gradient_weights(xyz, view, 0.0, 0)
        with self.assertRaises(ValueError):
            dk_gradient_weights(xyz, view, 1.0, 0, 5, 5)
        with self.assertRaises(RuntimeError):
            scale_densification_gradient(torch.zeros(1, 3, requires_grad=True), torch.ones(1))


if __name__ == "__main__":
    unittest.main()