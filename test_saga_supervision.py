"""Small CPU/CUDA checks; no model data or training outputs are modified."""
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from saga_supervision import sample_supervision, contrastive_loss, mask_scales_from_depth


class SagaSupervisionTests(unittest.TestCase):
    def test_original_image_names(self):
        from prepare_sam_pairs import resolve_original_image
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "00002.jpg"
            image.touch()
            self.assertEqual(resolve_original_image(root, "00002.jpg"), image)
            self.assertEqual(resolve_original_image(root, "00002"), image)
            self.assertEqual(resolve_original_image(root, "00002.JPG"), image)
            (root / "00002.png").touch()
            self.assertEqual(resolve_original_image(root, "00002.jpg"), image)
            with self.assertRaises(ValueError):
                resolve_original_image(root, "00002")
            with self.assertRaises(ValueError):
                resolve_original_image(root, "missing.jpg")

    def exercise(self, device):
        torch.manual_seed(12)
        masks = np.zeros((9, 8, 8), dtype=bool)
        masks[0, :, :] = True
        for i in range(1, 9):
            masks[i, i % 4:i % 4 + 4, (i // 4) % 4:(i // 4) % 4 + 4] = True
        packed = np.packbits(masks.reshape(9, -1), axis=1)
        areas = masks.sum((1, 2))
        scales = np.arange(1, 10, dtype=np.float32)
        supervision = sample_supervision(packed, np.bitwise_or.reduce(packed, axis=0),
                                         areas, scales, 8, 8, 1000, 10, device)
        pixels, truth, weights, sampled_scales = supervision
        self.assertEqual(tuple(truth.shape), (10, 64, 64))
        self.assertTrue(torch.equal(truth, truth.transpose(1, 2)))
        self.assertTrue((truth.diagonal(dim1=1, dim2=2) == 1).all())
        self.assertGreaterEqual(float(weights.min()), 1)
        self.assertLessEqual(float(weights.max()), 10.00001)
        feature_map = torch.randn(32, 8, 8, device=device, requires_grad=True)
        gate = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Sigmoid()).to(device)
        quantile = QuantileTransformer(n_quantiles=9, output_distribution="uniform").fit(scales[:, None])
        loss, stats = contrastive_loss(feature_map, supervision, gate, quantile, (8, 8))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(feature_map.grad.abs().sum()), 0)
        self.assertGreater(float(gate[0].weight.grad.abs().sum()), 0)
        self.assertGreater(stats["positive_pairs"], 0)
        # Also exercise the constant-mask-area and all-equal-scale safeguards.
        same = sample_supervision(packed[:1], packed[0], areas[:1], np.ones(1, np.float32),
                                  8, 8, 1000, 1, device)
        constant_loss, _ = contrastive_loss(feature_map, same, gate, quantile, (8, 8))
        self.assertTrue(torch.isfinite(constant_loss))
        empty = sample_supervision(packed[:0], np.zeros_like(packed[0]), areas[:0],
                                   scales[:0], 8, 8, 1000, 10, device)
        self.assertIsNone(empty)

    def test_cpu(self):
        self.exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        self.exercise("cuda")

    def test_depth_scale(self):
        depth = torch.ones(8, 10) * 2
        masks = np.ones((1, 8, 10), dtype=bool)
        scales = mask_scales_from_depth(masks, depth, 1.0, 0.8)
        self.assertTrue(np.isfinite(scales).all())
        self.assertGreater(scales[0], 0)
        doubled = mask_scales_from_depth(masks, depth * 2, 1.0, 0.8)
        np.testing.assert_allclose(doubled, scales * 2)

    def test_loss_matches_pair_equations(self):
        # Every off-diagonal pair changes label across scales, so both losses
        # must include every upper-triangle pair, irrespective of random mining.
        image = torch.randn(32, 1, 3, requires_grad=True)
        gate = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Sigmoid())
        truth = torch.stack((torch.ones(3, 3), torch.eye(3)))
        weights = torch.tensor([[1., 2., 3.], [2., 4., 5.], [3., 5., 6.]])
        scales = torch.tensor([1., 2.])
        q = QuantileTransformer(n_quantiles=2, output_distribution="uniform").fit(scales[:, None])
        result, _ = contrastive_loss(image, (torch.arange(3), truth, weights, scales),
                                     gate, q, (1, 3))
        vectors = image.flatten(1).T[None] * gate(torch.tensor([[0.], [1.]]))[:, None]
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        corr = vectors @ vectors.transpose(1, 2)
        selected = torch.triu(torch.ones(3, 3, dtype=torch.bool), 1)
        expected = (-weights[selected][None] * truth[:, selected] * corr[:, selected]).mean()
        expected += (weights[selected][None] * (1-truth[:, selected]) * corr[:, selected].relu()).mean()
        expected += (1-image.norm(dim=0).mean()).square()
        torch.testing.assert_allclose(result, expected)

    def test_vectorized_prefix_matches_saga(self):
        torch.manual_seed(2)
        masks = (torch.rand(25, 80) > .7).float()
        rows = torch.arange(len(masks))[:, None]
        latest = torch.where(masks.bool(), rows.expand_as(masks),
                              torch.full_like(rows.expand_as(masks), -1)).cummax(0).values
        for index in range(len(masks)):
            reference = torch.zeros_like(masks)
            reference[:index + 1] = masks[:index + 1]
            for j in range(index, -1, -1):
                reference[j] *= ~reference[j+1:].bool().any(0)
            reference[index+1:] = masks[index+1:]
            actual = masks.clone()
            actual[:index+1] = (rows[:index+1] == latest[index][None]).float()
            self.assertTrue(torch.equal(actual, reference))

    def test_neighbor_mean_gradient(self):
        from train_sam_features import _NeighborMean
        torch.manual_seed(10)
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            features = torch.randn(45, 32, device=device, requires_grad=True)
            neighbors = torch.randint(0, 45, (45, 16), device=device)
            ranks = torch.randperm(16, device=device)[:8]
            actual = _NeighborMean.apply(features, neighbors, ranks, 7)
            reference = features[neighbors[:, ranks]].mean(1)
            torch.testing.assert_allclose(actual, reference)
            upstream = torch.randn_like(actual)
            actual_grad = torch.autograd.grad(actual, features, upstream)[0]
            expected_grad = torch.autograd.grad(reference, features, upstream)[0]
            torch.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_native_renderer_backward(self):
        from train_sam_features import render_feature_map, smooth_gaussian_features
        from prepare_sam_pairs import render_camera_depth
        from utils.graphics_utils import getProjectionMatrix
        projection = getProjectionMatrix(0.01, 100.0, 1.0, 1.0).T.cuda()
        camera = SimpleNamespace(image_height=32, image_width=32, FoVx=1.0, FoVy=1.0,
                                  world_view_transform=torch.eye(4, device="cuda"),
                                  full_proj_transform=projection,
                                  camera_center=torch.zeros(3, device="cuda"))
        xyz = torch.randn(32, 3, device="cuda") * 0.1
        xyz[:, 2] += 2
        rotations = torch.zeros(32, 4, device="cuda")
        rotations[:, 0] = 1
        cloud = SimpleNamespace(get_xyz=xyz, active_sh_degree=0,
                                 get_scaling=torch.full((32, 3), 0.1, device="cuda"),
                                 get_rotation=rotations,
                                 get_opacity=torch.full((32, 1), 0.5, device="cuda"))
        pipe = SimpleNamespace(debug=False, antialiasing=False, compute_cov3D_python=False)
        features = torch.nn.Parameter(torch.randn(32, 32, device="cuda") * 0.01)
        neighbors = torch.arange(32, device="cuda")[:, None].expand(-1, 16)
        smoothed = smooth_gaussian_features(features, neighbors, 0.5, 16)
        rendered = render_feature_map(camera, cloud, pipe, smoothed)
        self.assertEqual(tuple(rendered.shape), (32, 32, 32))
        self.assertTrue(torch.isfinite(rendered).all())
        pipe.feature_backend = "rgb"
        reference = render_feature_map(camera, cloud, pipe, smoothed)
        torch.testing.assert_allclose(rendered, reference, rtol=2e-5, atol=2e-6)
        grad = torch.randn_like(rendered)
        fast_gradient = torch.autograd.grad(rendered, features, grad, retain_graph=True)[0]
        rgb_gradient = torch.autograd.grad(reference, features, grad, retain_graph=True)[0]
        torch.testing.assert_allclose(fast_gradient, rgb_gradient, rtol=2e-3, atol=2e-4)
        rendered.square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)
        depth = render_camera_depth(camera, cloud, pipe)
        self.assertTrue(torch.isfinite(depth).all())
        self.assertGreater(float(depth.max()), 0)
        # Deliberately force early termination. Later Gaussians must receive
        # no feature gradient at the center pixel, for either background.
        cloud.get_xyz.zero_()
        cloud.get_xyz[:, 2] = 2 + torch.arange(32, device="cuda") * .001
        cloud.get_scaling.fill_(.5)
        cloud.get_opacity.fill_(.99)
        for white in (False, True):
            pipe.feature_backend = "feature"
            fast = render_feature_map(camera, cloud, pipe, features, white)
            pipe.feature_backend = "rgb"
            slow = render_feature_map(camera, cloud, pipe, features, white)
            torch.testing.assert_allclose(fast, slow, rtol=2e-5, atol=2e-6)
            fast_grad = torch.autograd.grad(fast[:, 16, 16].sum(), features)[0]
            slow_grad = torch.autograd.grad(slow[:, 16, 16].sum(), features)[0]
            torch.testing.assert_allclose(fast_grad, slow_grad, rtol=2e-3, atol=2e-4)
            self.assertEqual(int(torch.count_nonzero(fast_grad[10:])), 0)


if __name__ == "__main__":
    unittest.main()
