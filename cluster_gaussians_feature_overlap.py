#!/usr/bin/env python3
"""Connected components: gamma-ellipsoid overlap AND feature cosine >= threshold.

No xyz radius or fixed-k truncation. Radius-binned bounding spheres provide a
conservative broad phase; actual rotated ellipsoid contacts decide edges.
Streaming union-find keeps graph memory O(N), even for dense overlap graphs.
"""
import argparse
import contextlib
import json
import struct
import time
import traceback
from pathlib import Path
import numpy as np
from plyfile import PlyData
from cluster_gaussians_twice import gaussian_geometry, ellipsoid_contacts, SphereIndex
from cluster_gaussian_features import load_scale_gates, normalized_features
from fast_ellipsoid_overlap import ellipsoid_contacts_fast
from threadpoolctl import threadpool_limits

HEADER = struct.Struct('<8sIIII')


def load_features(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError('Truncated GSFEAT1 header')
    magic, version, count, dimension, _ = HEADER.unpack(raw)
    if magic.rstrip(b'\0') != b'GSFEAT1' or version != 1 or dimension < 1 or count < 1:
        raise ValueError('Invalid GSFEAT1 header')
    if Path(path).stat().st_size != HEADER.size + count*dimension*4:
        raise ValueError('GSFEAT1 payload size mismatch')
    data = np.memmap(str(path), dtype='<f4', mode='r', offset=HEADER.size, shape=(count, dimension))
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite features')
    normalized = data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)
    print('Loaded {:,} x {} normalized features'.format(count, dimension), flush=True)
    return normalized


class UnionFind:
    def __init__(self, count):
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int64)

    def roots(self, indices):
        roots = np.asarray(indices, dtype=np.int32).copy()
        while True:
            parents = self.parent[roots]
            if np.array_equal(parents, roots):
                return roots
            self.parent[roots] = self.parent[parents]
            roots = self.parent[roots]

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = int(self.parent[index])
        return index

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


def cluster_overlap_graph(xyz, geometry, features, feature_threshold=.7,
                          min_cluster_size=20, batch_size=2048, progress_every=10000):
    # Thousands of tiny 32D products / 3x3 SVDs must not fan out to 16 MKL
    # workers per call. Larger job-level parallelism is independent of this.
    with threadpool_limits(limits=1, user_api='blas'):
        return _cluster_overlap_graph(xyz, geometry, features, feature_threshold,
                                       min_cluster_size, batch_size, progress_every)


def _cluster_overlap_graph(xyz, geometry, features, feature_threshold,
                           min_cluster_size, batch_size, progress_every):
    factors, whitening, inner, outer = geometry
    index = SphereIndex(xyz, outer)
    groups = UnionFind(len(xyz))
    extents = np.linalg.norm(factors, axis=2)
    diagnostics = dict(radial_witness_accepts=0, separating_axis_rejects=0, svd_contact_pairs=0)
    checks, unions, tested_overlap = 0, 0, 0
    for i in range(len(xyz)):
        for candidate_ids in index.candidates(xyz[i], outer[i]):
            candidate_ids = candidate_ids[candidate_ids > i]
            for start in range(0, len(candidate_ids), batch_size):
                js = candidate_ids[start:start+batch_size]
                # Already-connected endpoints cannot change components. Skip
                # redundant dense edges without changing the graph partition.
                js = js[groups.roots(js) != groups.find(i)]
                if not len(js):
                    continue
                js = js[np.linalg.norm(xyz[js]-xyz[i], axis=1) <= outer[i]+outer[js]]
                checks += len(js)
                similarity = np.einsum('ij,j->i',features[js],features[i],optimize=False)
                js = js[similarity >= feature_threshold]
                if not len(js):
                    continue
                tested_overlap += len(js)
                contacts = ellipsoid_contacts_fast(xyz, factors, whitening, inner, outer,
                                                    i, js, extents, diagnostics)
                for j in js[contacts]:
                    unions += int(groups.union(i, int(j)))
        if (i+1) % progress_every == 0 or i+1 == len(xyz):
            print('Graph {:,}/{:,}: {:,} feature checks, {:,} overlap tests, {:,} SVD pairs, {:,} unions'.format(
                i+1, len(xyz), checks, tested_overlap, diagnostics['svd_contact_pairs'],unions), flush=True)
        if groups.size[groups.find(i)] == len(xyz):
            break
    roots = groups.roots(np.arange(len(xyz), dtype=np.int32))
    _, components = np.unique(roots, return_inverse=True)
    sizes = np.bincount(components)
    keep = sizes >= min_cluster_size
    mapping = np.full(len(sizes), -1, dtype=np.int32)
    mapping[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    labels = mapping[components]
    return labels, int(keep.sum()), dict(diagnostics, **{
        'feature_checks': checks, 'exact_overlap_tests': tested_overlap,
        'accepted_union_edges': unions, 'connected_components_before_size_filter': len(sizes),
    })


def write_clusters(path, labels, cluster_count):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('wb') as stream:
        stream.write(HEADER.pack(b'GSCLST1', 1, len(labels), cluster_count, 0))
        labels.astype('<i4', copy=False).tofile(stream)


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--ply_file', required=True)
    parser.add_argument('--feature_file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--overlap_gamma', '--gamma', dest='overlap_gamma', type=float, default=3.)
    parser.add_argument('--feature_threshold', type=float, default=.7)
    parser.add_argument('--scale', type=float, default=.5,
                        help='SAGA normalized segmentation scale in [0,1]')
    parser.add_argument('--scale_gate_file', default=None,
                        help='trained gate state_dict or checkpoint; auto-detect beside feature file')
    parser.add_argument('--min_cluster_size', type=int, default=20)
    parser.add_argument('--edge_batch_size', type=int, default=2048)
    parser.add_argument('--log_file', default=None)
    return parser


def run(args):
    if not np.isfinite(args.overlap_gamma) or args.overlap_gamma <= 0:
        raise ValueError('overlap_gamma must be finite and positive')
    if not np.isfinite(args.feature_threshold) or not -1 <= args.feature_threshold <= 1:
        raise ValueError('feature_threshold must be in [-1,1]')
    if not np.isfinite(args.scale) or not 0 <= args.scale <= 1:
        raise ValueError('scale must be in [0,1] (SAGA normalized scale)')
    if args.min_cluster_size < 1 or args.edge_batch_size < 1:
        raise ValueError('min_cluster_size and edge_batch_size must be positive')
    started = time.time()
    vertex = PlyData.read(str(args.ply_file))['vertex'].data
    xyz = np.stack([vertex[key] for key in ('x','y','z')], axis=1).astype(np.float64)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Gaussian centers must be nonempty and finite')
    features = load_features(Path(args.feature_file))
    if len(xyz) != len(features):
        raise ValueError('PLY/feature count mismatch: {} vs {}'.format(len(xyz), len(features)))
    gates, gate_metadata = load_scale_gates(Path(args.feature_file).resolve(),
                                            features.shape[1], len(features), args)
    features = normalized_features(features, slice(None), gates)
    print('Scale gate: scale={}, loaded {}'.format(args.scale, gate_metadata['scale_gate_file']), flush=True)
    geometry = gaussian_geometry(vertex, args.overlap_gamma)
    print('Feature-overlap graph: gamma={}, cosine>={}, min_cluster_size={}; no fixed-k cap'.format(
        args.overlap_gamma, args.feature_threshold, args.min_cluster_size), flush=True)
    labels, count, statistics = cluster_overlap_graph(xyz, geometry, features,
        args.feature_threshold, args.min_cluster_size, args.edge_batch_size)
    write_clusters(args.output, labels, count)
    metadata = dict(statistics, format='GSCLST1', method='ellipsoid_overlap_feature_graph',
        ply_file=str(Path(args.ply_file).resolve()), feature_file=str(Path(args.feature_file).resolve()),
        count=len(labels), cluster_count=count, noise_count=int((labels<0).sum()),
        noise_ratio=float((labels<0).mean()), overlap_gamma=args.overlap_gamma,
        feature_threshold=args.feature_threshold, feature_metric='cosine_on_scale_gated_normalized_features',
        min_cluster_size=args.min_cluster_size, fixed_k_candidate_cap=False,
        overlap_test='rotated_ellipsoid_contact_function', elapsed_seconds=time.time()-started)
    metadata.update(gate_metadata)
    Path(str(args.output)+'.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Saved {:,} clusters, {:,} noise -> {}'.format(count, metadata['noise_count'],args.output), flush=True)


def main():
    args = build_parser().parse_args()
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_file, 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)
    else:
        run(args)


if __name__ == '__main__':
    main()
