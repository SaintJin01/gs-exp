"""Ungated feature clustering regression tests (legacy filename retained)."""
import json
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import cluster_gaussian_features as clustering


class UngatedClusteringTests(unittest.TestCase):
    def test_direct_normalization(self):
        features=np.array([[3.,4.],[0.,0.]],dtype=np.float32)
        np.testing.assert_allclose(clustering.normalized_features(features,slice(None)),
                                   [[.6,.8],[0.,0.]],atol=1e-7)

    def test_noise_preserved_in_sample_and_unsampled(self):
        rng=np.random.RandomState(42)
        sample=np.vstack((rng.normal([1.,0.,0.],.01,(80,3)),
                          rng.normal([0.,1.,0.],.01,(80,3)),
                          [[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]] )).astype(np.float32)
        sample=clustering.normalized_features(sample,slice(None))
        args=clustering.build_parser().parse_args([])
        args.jobs=1
        clusterer,centers,sizes,metadata=clustering.fit_density_clusters(sample,args)
        self.assertGreater(metadata['sample_noise_count'],0)
        all_features=np.vstack((sample, [[1.,0.,0.],[0.,0.,-1.]] )).astype(np.float32)
        labels=clustering.predict_density_labels(all_features,clusterer,np.arange(len(sample)),32)
        np.testing.assert_array_equal(labels[:len(sample)],clusterer.labels_)
        self.assertGreaterEqual(int(labels[-2]),0)
        self.assertEqual(int(labels[-1]),-1)
        levels=clustering.build_hierarchy(centers,3)
        with TemporaryDirectory() as temporary:
            output=Path(temporary)/'noise.gscluster'
            clustering.write_clusters(output,labels,levels,len(centers))
            with output.open('rb') as stream:
                stream.read(24)
                for count,mapping in levels:
                    stream.read(4)
                    actual=np.frombuffer(stream.read(len(labels)*4),dtype='<i4')
                    np.testing.assert_array_equal(actual<0,labels<0)

    def test_epsilon_prediction_mapping_preserves_fitted_clusters(self):
        rng=np.random.RandomState(9)
        points=np.vstack([rng.normal([1.,v,0.],.001,(60,3))
                          for v in (0.,.03,.9,.93)]).astype(np.float32)
        points=clustering.normalized_features(points,slice(None))
        args=clustering.build_parser().parse_args([])
        args.jobs=1
        args.cluster_selection_method='leaf'
        args.cluster_selection_epsilon=.1
        clusterer,centers,_,_=clustering.fit_density_clusters(points,args)
        # Predict identical new copies: IDs must match the fitted epsilon merges.
        features=np.vstack((points,points))
        labels=clustering.predict_density_labels(features,clusterer,np.arange(len(points)),64)
        np.testing.assert_array_equal(labels[:len(points)],clusterer.labels_)
        np.testing.assert_array_equal(labels[len(points):],clusterer.labels_)
        self.assertTrue(np.all(labels < len(centers)))

    def test_joint_blocks_and_weights(self):
        features=np.tile(np.array([[3.,4.]],np.float32),(8,1))
        geometry=np.zeros((8,3),np.float32)
        geometry[:,0]=np.arange(8)
        joint=clustering.JointFeatures(features,geometry,np.arange(8))
        self.assertEqual(joint.shape,(8,5))
        values=joint[slice(None)]
        np.testing.assert_allclose(values[:,:2],np.tile([.6,.8],(8,1)),atol=1e-7)
        self.assertGreater(float(np.linalg.norm(values[0]-values[-1])),0.)
        off=clustering.JointFeatures(features,geometry,np.arange(8),xyz_weight=0)
        np.testing.assert_allclose(off[slice(None)][:,2:],0.)
        self.assertTrue(np.isfinite(values).all())
        with self.assertRaises(ValueError):
            clustering.JointFeatures(features,geometry,np.arange(8),feature_weight=-1)

    def test_joint_space_splits_identical_semantic_features(self):
        rng=np.random.RandomState(3)
        features=np.ones((160,16),np.float32)
        geometry=np.zeros((160,3),np.float32)
        geometry[:80,:3]=rng.normal(-10,.01,(80,3))
        geometry[80:,:3]=rng.normal(10,.01,(80,3))
        sample_ids=np.arange(0,160,2)
        inputs=clustering.JointFeatures(features,geometry,sample_ids)
        args=clustering.build_parser().parse_args([])
        args.jobs=1
        fitted,centers,_,_=clustering.fit_density_clusters(inputs[sample_ids],args)
        labels=clustering.predict_density_labels(inputs,fitted,sample_ids,32)
        self.assertEqual(len(np.unique(labels)),2)
        self.assertNotEqual(labels[0],labels[-1])
        self.assertTrue(np.all(labels[:80]==labels[0]))
        self.assertTrue(np.all(labels[80:]==labels[-1]))
        self.assertEqual(centers.shape[1],19)

    def test_noise_assignment_uses_joint_euclidean_not_cosine(self):
        points=np.zeros((3,19),np.float32)
        points[:,0]=1.
        points[:,16]=[99.,1.,99.]
        centers=np.zeros((2,19),np.float32)
        centers[:,0]=1.
        centers[:,16]=[1.,100.]
        labels=np.array([-1,-1,0],np.int32)
        assigned=clustering.assign_noise_joint_space(points,labels,centers,1)
        np.testing.assert_array_equal(assigned,[1,0,0])
        np.testing.assert_array_equal(labels,[-1,-1,0])
        with self.assertRaises(ValueError):
            clustering.assign_noise_joint_space(points,labels,np.empty((0,19)),1)

    def test_gate_api_removed(self):
        self.assertFalse(hasattr(clustering, 'load_scale_gates'))
        args=clustering.build_parser().parse_args([])
        self.assertFalse(hasattr(args,'scale'))
        self.assertFalse(hasattr(args,'scale_gate_file'))

    def test_defaults(self):
        args=clustering.build_parser().parse_args([])
        self.assertEqual(args.sample_ratio,.02)
        self.assertFalse(args.keep_noise)
        self.assertEqual(args.min_cluster_size,10)
        self.assertIsNone(args.min_samples)
        self.assertEqual(args.cluster_selection_epsilon,.01)
        self.assertEqual(args.cluster_selection_method,'eom')

    def test_cli_binary_and_metadata(self):
        rng = np.random.RandomState(2)
        features = rng.normal(0, .01, (600, 16)).astype(np.float32)
        for group in range(3):
            features[group*200:(group+1)*200, group] += 1
        gate = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Sigmoid())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'synthetic.gsfeat'
            with path.open('wb') as stream:
                stream.write(clustering.FEATURE_HEADER.pack(b'GSFEAT1', 1, 600, 16, 0))
                features.tofile(stream)
            torch.save(gate.state_dict(), str(root/'scale_gate.pt'))
            vertices=np.zeros(600,dtype=[(key,'f4') for key in
                              ('x','y','z')])
            vertices['x']=np.repeat([-5.,0.,5.],200)
            ply=root/'source.ply'
            PlyData([PlyElement.describe(vertices,'vertex')]).write(str(ply))
            subprocess.check_call([sys.executable, str(Path(clustering.__file__)),
                '--feature_file', str(path), '--ply_file', str(ply), '--sample_size', '0', '--jobs', '1'])
            result = root/'synthetic.gscluster'
            metadata = json.loads((root/'synthetic.gscluster.json').read_text())
            self.assertFalse(metadata['scale_gate_applied'])
            self.assertNotIn('scale', metadata)
            self.assertEqual(metadata['noise_count'], 0)
            self.assertEqual(metadata['clustering_dimension'],19)
            self.assertEqual(metadata['noise_reassigned_count'],0)
            self.assertFalse(metadata['concatenation_L2_normalized'])
            self.assertFalse(metadata['size_features_applied'])
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
