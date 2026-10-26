import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import cluster_gaussians_feature_overlap as graph
from cluster_gaussians_twice import gaussian_geometry, ellipsoid_contacts
from fast_ellipsoid_overlap import ellipsoid_contacts_fast


def vertices(points, axes=None):
    fields = ['x','y','z'] + ['scale_%d'%i for i in range(3)] + ['rot_%d'%i for i in range(4)]
    result = np.zeros(len(points), dtype=[(field,'f4') for field in fields])
    for i, field in enumerate(('x','y','z')):
        result[field] = np.asarray(points)[:,i]
    axes = np.ones((len(points),3)) if axes is None else np.asarray(axes)
    for i in range(3):
        result['scale_%d'%i] = np.log(axes[:,i])
    result['rot_0'] = 1
    return result


def calculate(vertex, features, gamma=1, threshold=.7, minimum=1):
    xyz=np.stack([vertex[k] for k in ('x','y','z')],axis=1).astype(np.float64)
    features=np.asarray(features,dtype=np.float32)
    features/=np.linalg.norm(features,axis=1,keepdims=True)
    return graph.cluster_overlap_graph(xyz,gaussian_geometry(vertex,gamma),features,
                                        threshold,minimum,batch_size=5)


class OverlapGraphTests(unittest.TestCase):
    def test_fast_contact_matches_reference_for_thin_rotated_gaussians(self):
        rng=np.random.RandomState(61)
        for batch in range(12):
            vertex=vertices(rng.randn(60,3),np.exp(rng.uniform(-9,2,(60,3))))
            for k in range(4):vertex['rot_%d'%k]=rng.randn(60)
            xyz=np.stack([vertex[k] for k in ('x','y','z')],axis=1).astype(np.float64)
            geometry=gaussian_geometry(vertex,3)
            extents=np.linalg.norm(geometry[0],axis=2)
            for i in range(10):
                js=np.arange(i+1,60,dtype=np.int32)
                expected=ellipsoid_contacts(xyz,*geometry,i,js)
                actual=ellipsoid_contacts_fast(xyz,*geometry,i,js,extents)
                np.testing.assert_array_equal(actual,expected)

    def test_overlap_and_feature_are_both_required(self):
        vertex=vertices([[0,0,0],[1.5,0,0],[1.6,0,0],[10,0,0]])
        labels,count,_=calculate(vertex,[[1,0],[1,0],[0,1],[1,0]])
        self.assertEqual(count,3)
        self.assertEqual(labels[0],labels[1])
        self.assertNotEqual(labels[0],labels[2])
        self.assertNotEqual(labels[0],labels[3])

    def test_gamma_and_transitive_components(self):
        vertex=vertices([[0,0,0],[4,0,0],[8,0,0]])
        _,count,_=calculate(vertex,[[1,0]]*3,gamma=1)
        self.assertEqual(count,3)
        _,count,_=calculate(vertex,[[1,0]]*3,gamma=3)
        self.assertEqual(count,1)

    def test_rotation_and_thin_ellipsoids(self):
        vertex=vertices([[0,0,0],[0,3,0]],[[5,.05,.05]]*2)
        _,count,_=calculate(vertex,[[1,0]]*2)
        self.assertEqual(count,2)  # bounding spheres overlap, ellipsoids do not
        vertex['rot_0'][1]=np.sqrt(.5);vertex['rot_3'][1]=np.sqrt(.5)
        _,count,_=calculate(vertex,[[1,0]]*2)
        self.assertEqual(count,1)

    def test_no_fixed_neighbor_cap(self):
        points=[[0,0,0],[19,0,0]]+[[.1+i*.01,1,0] for i in range(60)]
        vertex=vertices(points,[[10,.05,.05]]*2+[[.01]*3]*60)
        labels,_,_=calculate(vertex,[[1,0]]*2+[[0,1]]*60)
        self.assertEqual(labels[0],labels[1])  # far center is beyond 48 nearest

    def test_matches_exhaustive_graph(self):
        rng=np.random.RandomState(8)
        vertex=vertices(rng.randn(25,3),np.exp(rng.randn(25,3)-.8))
        for k in range(4):vertex['rot_%d'%k]=rng.randn(25)
        features=rng.randn(25,4).astype(np.float32)
        features/=np.linalg.norm(features,axis=1,keepdims=True)
        xyz=np.stack([vertex[k] for k in ('x','y','z')],axis=1).astype(np.float64)
        geometry=gaussian_geometry(vertex,2)
        labels,_,_=graph.cluster_overlap_graph(xyz,geometry,features,.4,1,batch_size=3)
        expected=graph.UnionFind(25)
        for i in range(25):
            js=np.arange(i+1,25,dtype=np.int32)
            accepted=ellipsoid_contacts(xyz,*geometry,i,js) & ((features[js]@features[i])>=.4)
            for j in js[accepted]:expected.union(i,int(j))
        roots=expected.roots(np.arange(25))
        self.assertTrue(np.array_equal(labels[:,None]==labels[None],roots[:,None]==roots[None]))

    def test_all_noise_cli_and_legacy_entry(self):
        vertex=vertices([[0,0,0],[20,0,0]])
        with TemporaryDirectory() as temporary:
            root=Path(temporary);ply=root/'source.ply';feat=root/'source.gsfeat';out=root/'output.gscluster'
            PlyData([PlyElement.describe(vertex,'vertex')]).write(str(ply))
            gate=torch.nn.Sequential(torch.nn.Linear(1,2),torch.nn.Sigmoid())
            torch.save(gate.state_dict(),str(root/'scale_gate.pt'))
            with feat.open('wb') as stream:
                stream.write(graph.HEADER.pack(b'GSFEAT1',1,2,2,0))
                np.array([[1,0],[1,0]],dtype='<f4').tofile(stream)
            subprocess.check_call([sys.executable,str(Path(graph.__file__).with_name('cluster_gaussians_xyz_feature.py')),
                '--ply_file',str(ply),'--feature_file',str(feat),'--output',str(out)])
            with out.open('rb') as stream:
                header=graph.HEADER.unpack(stream.read(24))
                self.assertEqual(header[3],0)
                self.assertTrue(np.all(np.frombuffer(stream.read(),dtype='<i4')==-1))
            metadata=json.loads(Path(str(out)+'.json').read_text())
            self.assertEqual(metadata['overlap_gamma'],3)
            self.assertFalse(metadata['fixed_k_candidate_cap'])
            self.assertFalse(metadata['scale_gate_applied'])
            self.assertNotIn('scale', metadata)

    def test_direct_normalized_features(self):
        from cluster_gaussian_features import normalized_features
        raw=np.array([[3,4],[3,-4]],dtype=np.float32)
        features=normalized_features(raw,slice(None))
        np.testing.assert_allclose(features, raw/5, atol=1e-7)
        vertex=vertices([[0,0,0],[1,0,0]])
        _,count,_=calculate(vertex,features)
        self.assertEqual(count,2)


if __name__=='__main__':
    unittest.main()
