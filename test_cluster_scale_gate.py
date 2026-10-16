"""Scale-gated feature clustering regression tests (no user data modified)."""
import json
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import cluster_gaussian_features as clustering


class ScaleGateTests(unittest.TestCase):
    def test_saga_order_and_numerics(self):
        torch.manual_seed(12)
        features = torch.randn(25, 32)
        features[0] = 0
        gate = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Sigmoid())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save({"features": features, "scale_gate": gate.state_dict()}, str(root/'latest.pt'))
            for scale in (0.0, .5, 1.0):
                gates, metadata = clustering.load_scale_gates(root/'latest.gsfeat', 32, 25,
                    SimpleNamespace(scale=scale, scale_gate_file=None))
                expected = F.normalize(F.normalize(features, dim=-1) *
                                       gate(torch.tensor([[scale]])), dim=-1)
                actual = clustering.normalized_features(features.numpy(), slice(None), gates)
                np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=1e-5, atol=1e-7)
                self.assertFalse(metadata['scale_quantile_reapplied'])

    def test_gate_changes_full_assignment(self):
        features = np.array([[1, 2]], dtype=np.float32)
        centers = np.eye(2, dtype=np.float32)
        self.assertEqual(int(clustering.predict_cosine_labels(features, centers, 1)[0]), 1)
        self.assertEqual(int(clustering.predict_cosine_labels(features, centers, 1,
                            np.array([1, .1], dtype=np.float32))[0]), 0)

    def test_old_or_mismatched_checkpoint_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(scale=.5, scale_gate_file=None)
            with self.assertRaises(FileNotFoundError):
                clustering.load_scale_gates(root/'latest.gsfeat', 32, 3, args)
            torch.save({'features': torch.zeros(3, 32)}, str(root/'latest.pt'))
            with self.assertRaisesRegex(ValueError, 'no trained scale gate'):
                clustering.load_scale_gates(root/'latest.gsfeat', 32, 3, args)
            gate = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Sigmoid())
            torch.save({'features': torch.zeros(4, 32), 'scale_gate': gate.state_dict()}, str(root/'latest.pt'))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                clustering.load_scale_gates(root/'latest.gsfeat', 32, 3, args)

    def test_saga_defaults(self):
        args = clustering.build_parser().parse_args([])
        self.assertEqual(args.scale, .5)
        self.assertEqual(args.sample_ratio, .02)
        self.assertEqual(args.min_cluster_size, 10)
        self.assertIsNone(args.min_samples)
        self.assertEqual(args.cluster_selection_epsilon, .01)
        self.assertEqual(args.cluster_selection_method, 'eom')

    def test_cli_binary_and_metadata(self):
        rng = np.random.RandomState(2)
        features = rng.normal(0, .01, (600, 32)).astype(np.float32)
        for group in range(3):
            features[group*200:(group+1)*200, group] += 1
        gate = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Sigmoid())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'synthetic.gsfeat'
            with path.open('wb') as stream:
                stream.write(clustering.FEATURE_HEADER.pack(b'GSFEAT1', 1, 600, 32, 0))
                features.tofile(stream)
            torch.save(gate.state_dict(), str(root/'scale_gate.pt'))
            subprocess.check_call([sys.executable, str(Path(clustering.__file__)),
                '--feature_file', str(path), '--sample_size', '0', '--jobs', '1', '--scale', '.25'])
            result = root/'synthetic.gscluster'
            metadata = json.loads((root/'synthetic.gscluster.json').read_text())
            self.assertTrue(metadata['scale_gate_applied'])
            self.assertEqual(metadata['scale'], .25)
            self.assertEqual(metadata['noise_count'], 0)
            with result.open('rb') as stream:
                magic, version, count, levels, leaves = clustering.CLUSTER_HEADER.unpack(stream.read(24))
                self.assertEqual((magic.rstrip(b'\0'), version, count), (b'GSCLST2', 2, 600))
                for level in range(levels):
                    clusters = struct.unpack('<I', stream.read(4))[0]
                    labels = np.frombuffer(stream.read(600*4), dtype='<i4')
                    self.assertTrue(((labels >= 0) & (labels < clusters)).all())
                self.assertEqual(stream.read(), b'')


if __name__ == '__main__':
    unittest.main()
