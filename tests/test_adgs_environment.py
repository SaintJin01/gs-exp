"""CPU regressions for environment exclusion and AD-GS scheduling."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from arguments import OptimizationParams
from argparse import ArgumentParser
from utils.adgs_utils import scene_depth_geometry, rasterize_z_depth, edge_aware_depth_loss, adgs_phase

ROOT = Path(__file__).resolve().parents[1]

class EnvironmentADGSTest(unittest.TestCase):
    def test_depth_gradients_exclude_environment(self):
        for covariance in (False, True):
            geometry = {k: torch.randn(5, width, requires_grad=True) for k, width in
                        [("means3D", 3), ("means2D", 3), ("opacities", 1),
                         ("scales", 3), ("rotations", 4), ("cov3D_precomp", 6)]}
            geometry["shs"] = None
            active = dict(geometry)
            for key in (["scales", "rotations"] if covariance else ["cov3D_precomp"]):
                active[key] = None
            def rasterizer(colors_precomp, **inputs):
                value = colors_precomp.sum() + sum(t.square().sum() for t in inputs.values() if t is not None)
                pattern = torch.arange(1., 10.).reshape(1, 3, 3)
                return (value * pattern.expand(3, -1, -1), None, None)
            depth = rasterize_z_depth(rasterizer, torch.eye(4), scene_depth_geometry(active, 2))
            edge_aware_depth_loss(depth, torch.zeros(3, 3, 3)).backward()
            for key, tensor in active.items():
                if tensor is None: continue
                self.assertEqual(torch.count_nonzero(tensor.grad[:2]).item(), 0, key)
                self.assertGreater(torch.count_nonzero(tensor.grad[2:]).item(), 0, key)

    def test_environment_clone_and_split_excluded(self):
        # Execute the production overrides against a CPU selection probe.
        tree = ast.parse((ROOT / "scene/environment_gaussian.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                    ("densify_and_clone", "densify_and_split")]
        class Probe:
            def densify_and_clone(self, grads, threshold, extent):
                self.selected = grads.norm(dim=-1) >= threshold
            def densify_and_split(self, grads, threshold, extent, N=2):
                self.selected = grads.flatten() >= threshold
        ns = {"GaussianModel": Probe}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "<env-overrides>", "exec"), ns)
        model = ns["EnvironmentGaussianModel"](); model.env_count = 2
        for method in (model.densify_and_clone, model.densify_and_split):
            for threshold in (0.0002, 0.0005):
                method(torch.ones(5, 1), threshold, 1.)
                self.assertEqual(model.selected.tolist(), [False, False, True, True, True])

    def test_schedule(self):
        parser = ArgumentParser(); OptimizationParams(parser); opt = parser.parse_args([])
        self.assertEqual(opt.env_train_iterations, 24000)
        self.assertEqual(opt.env_prune_iteration, 10000)
        tree = ast.parse((ROOT / "scene/environment_gaussian.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "phase_at")
        ns = {}; exec(compile(ast.Module(body=[fn], type_ignores=[]), "<phase>", "exec"), ns)
        joint = controller = 0; last = None
        for i in range(1, opt.iterations + 1):
            stage, _ = ns["phase_at"](i, opt)
            is_controller = stage == "controller"
            controller += is_controller; joint += not is_controller
            if is_controller: self.assertFalse(adgs_phase(i, True)[1])
            if not is_controller and opt.densify_from_iter < i < opt.densify_until_iter and i % opt.densification_interval == 0:
                last = i
        self.assertEqual((joint, controller, last), (24000, 6000, 23900))

if __name__ == "__main__": unittest.main()
