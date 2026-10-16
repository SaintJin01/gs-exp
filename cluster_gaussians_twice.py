#!/usr/bin/env python3
"""Per-parent ellipsoid-overlap DBSCAN and lossless Gaussian PLY filtering."""
import argparse
import contextlib
import json
import os
import struct
import sys
import traceback
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

HEADER = struct.Struct('<8sIIII')


def gaussian_geometry(vertex, sigma):
    required = ['scale_' + str(k) for k in range(3)] + ['rot_' + str(k) for k in range(4)]
    if any(name not in vertex.dtype.names for name in required):
        raise ValueError('Overlap DBSCAN requires 3DGS log-scale and quaternion PLY fields')
    axes = np.exp(np.stack([vertex['scale_' + str(k)] for k in range(3)], axis=1).astype(np.float64)) * sigma
    q = np.stack([vertex['rot_' + str(k)] for k in range(4)], axis=1).astype(np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(axes).all() or np.any(axes <= 0) or not np.isfinite(q).all() or np.any(norm < 1e-12):
        raise ValueError('Invalid Gaussian scales or quaternions')
    q /= norm
    w, x, y, z = q.T
    rotation = np.empty((len(vertex), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y*y + z*z)
    rotation[:, 0, 1] = 2 * (x*y - w*z)
    rotation[:, 0, 2] = 2 * (x*z + w*y)
    rotation[:, 1, 0] = 2 * (x*y + w*z)
    rotation[:, 1, 1] = 1 - 2 * (x*x + z*z)
    rotation[:, 1, 2] = 2 * (y*z - w*x)
    rotation[:, 2, 0] = 2 * (x*z - w*y)
    rotation[:, 2, 1] = 2 * (y*z + w*x)
    rotation[:, 2, 2] = 1 - 2 * (x*x + y*y)
    scaled = rotation * axes[:, None, :]
    whitening = rotation.transpose(0, 2, 1) / axes[:, :, None]
    # Keep the covariance square root R*diag(axes), rather than forming
    # R*diag(axes**2)*R.T. Forming that matrix loses thin-axis information.
    return scaled, whitening, axes.min(axis=1), axes.max(axis=1)


def ellipsoid_contacts(points, factors, whitening, inner, outer, i, js):
    """Ellipsoid intersection via the maximum contact function on t in [0,1]."""
    delta = points[js] - points[i]
    distance = np.linalg.norm(delta, axis=1)
    accepted = distance <= inner[i] + inner[js]
    possible = (~accepted) & (distance <= outer[i] + outer[js])
    extent_i = np.linalg.norm(factors[i], axis=1)
    extent_j = np.linalg.norm(factors[js], axis=2)
    possible &= np.all(np.abs(delta) <= extent_i + extent_j, axis=1)
    positions = np.flatnonzero(possible)
    if not len(positions):
        return accepted
    transform = whitening[i]
    # SVD of the factor preserves small axes instead of squaring its
    # condition number and then asking eigh to recover them from roundoff.
    transformed_factor = transform @ factors[js[positions]]
    eigenvectors, singular_values, _ = np.linalg.svd(transformed_factor)
    eigenvalues = singular_values**2
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError('Invalid ellipsoid factors: unrepresentable axis ratio')
    transformed_delta = delta[positions] @ transform.T
    coefficients = np.einsum('bji,bj->bi', eigenvectors, transformed_delta)**2
    low = np.zeros(len(positions))
    high = np.ones(len(positions))
    # The contact function is concave; derivative bisection finds its maximum.
    for _ in range(48):
        t = (low + high) * 0.5
        denominator = (1.0 - t[:, None]) + t[:, None] * eigenvalues
        derivative = np.sum(coefficients * (
            (1.0-t[:, None])**2 - eigenvalues*t[:, None]**2
        ) / denominator**2, axis=1)
        low = np.where(derivative > 0, t, low)
        high = np.where(derivative > 0, high, t)
    t = (low + high) * 0.5
    contact = t*(1-t) * np.sum(
        coefficients / ((1.0-t[:, None]) + t[:, None]*eigenvalues), axis=1)
    accepted[positions] = contact <= 1.0 + 1e-9
    return accepted


class SphereIndex:
    """Radius-binned broad phase, returning one bounded query at a time."""
    def __init__(self, points, radii):
        self.points, self.radii = points, radii
        bins = np.floor(np.log2(np.maximum(radii, np.finfo(float).tiny))).astype(np.int64)
        self.trees = []
        for value in np.unique(bins):
            ids = np.flatnonzero(bins == value)
            self.trees.append((ids, cKDTree(points[ids]), float(radii[ids].max())))

    def candidates(self, point, radius):
        for ids, tree, maximum in self.trees:
            local = tree.query_ball_point(point, radius + maximum)
            if len(local):
                yield ids[np.asarray(local, dtype=np.intp)]


def dbscan_overlap_bounded(points, covariance, whitening, inner, outer,
                           min_samples, batch_size=2048):
    """DBSCAN on symmetric ellipsoid intersection, without storing graph edges."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    labels = np.full(n, -1, dtype=np.int32)
    if n < min_samples:
        return labels
    # Points in the same size/grid cell have intersecting inscribed spheres,
    # hence form a proven overlap clique, irrespective of their rotations.
    bins = np.floor(np.log2(inner)).astype(np.int64)
    width = np.exp2(bins.astype(float)) * (2.0 / np.sqrt(3)) * (1 - 1e-12)
    coords = np.floor((points - points.min(axis=0)) / width[:, None]).astype(np.int64)
    _, inverse = np.unique(np.c_[bins, coords], axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse)
    bounds = np.r_[0, np.cumsum(counts)]
    index = SphereIndex(points, outer)
    core = counts[inverse] >= min_samples
    print('  overlap core detection: {:,} proven dense points, {:,} to test'.format(
        np.count_nonzero(core), np.count_nonzero(~core)), flush=True)
    last = time.monotonic()
    for iteration, i in enumerate(np.flatnonzero(~core)):
        neighbors = int(counts[inverse[i]])
        for candidates in index.candidates(points[i], outer[i]):
            candidates = candidates[inverse[candidates] != inverse[i]]
            for start in range(0, len(candidates), batch_size):
                js = candidates[start:start + batch_size]
                neighbors += int(ellipsoid_contacts(points, covariance, whitening, inner, outer, i, js).sum())
                if neighbors >= min_samples:
                    break
            if neighbors >= min_samples:
                break
        core[i] = neighbors >= min_samples
        if time.monotonic() - last > 10:
            print('  overlap density tested {:,} points'.format(iteration + 1), flush=True)
            last = time.monotonic()
    core_ids = np.flatnonzero(core)
    if not len(core_ids):
        return labels
    core_cell_ids = np.unique(inverse[core_ids])
    members = []
    centers, radii = [], []
    for cell in core_cell_ids:
        ids = order[bounds[cell]:bounds[cell + 1]]
        ids = ids[core[ids]]
        members.append(ids)
        center = (points[ids].min(axis=0) + points[ids].max(axis=0)) * 0.5
        centers.append(center)
        radii.append(np.max(np.linalg.norm(points[ids]-center, axis=1) + outer[ids]))
    centers, radii = np.asarray(centers), np.asarray(radii)
    cell_index = SphereIndex(centers, radii)
    parents = np.arange(len(members), dtype=np.int32)
    sizes = np.ones(len(members), dtype=np.int32)

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i

    print('  connecting {:,} overlap core cliques'.format(len(members)), flush=True)
    last = time.monotonic()
    for a, ids in enumerate(members):
        for candidates in cell_index.candidates(centers[a], radii[a]):
            for b in candidates:
                if b <= a or find(a) == find(b):
                    continue
                if np.linalg.norm(centers[a]-centers[b]) > radii[a] + radii[b]:
                    continue
                left, right = ids, members[b]
                if len(left) > len(right):
                    left, right = right, left
                connected = False
                for i in left:
                    for start in range(0, len(right), batch_size):
                        js = right[start:start + batch_size]
                        if ellipsoid_contacts(points, covariance, whitening, inner, outer, i, js).any():
                            connected = True
                            break
                    if connected:
                        break
                if connected:
                    ra, rb = find(a), find(b)
                    if sizes[ra] < sizes[rb]:
                        ra, rb = rb, ra
                    parents[rb] = ra
                    sizes[ra] += sizes[rb]
        if time.monotonic() - last > 10:
            print('  connected overlap cliques {:,}/{:,}'.format(a+1, len(members)), flush=True)
            last = time.monotonic()
    roots = np.asarray([find(a) for a in range(len(members))])
    _, compact = np.unique(roots, return_inverse=True)
    for ids, label in zip(members, compact):
        labels[ids] = label
    # A border Gaussian belongs to a core component only if its ellipsoid
    # actually intersects a core ellipsoid. No center-distance assignment.
    for i in np.flatnonzero(~core):
        assigned = False
        for candidates in index.candidates(points[i], outer[i]):
            candidates = candidates[core[candidates]]
            for start in range(0, len(candidates), batch_size):
                js = candidates[start:start + batch_size]
                hits = ellipsoid_contacts(points, covariance, whitening, inner, outer, i, js)
                if hits.any():
                    labels[i] = labels[js[np.flatnonzero(hits)[0]]]
                    assigned = True
                    break
            if assigned:
                break
    return labels


def source_signature(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns}


def parent_labels(path, level, count):
    with path.open('rb') as stream:
        magic, version, n, a, b = HEADER.unpack(stream.read(HEADER.size))
        if n != count:
            raise ValueError('PLY/cluster Gaussian count mismatch')
        if magic.rstrip(b'\0') == b'GSCLST1' and version == 1:
            if level != 0:
                raise ValueError('Legacy cluster file only has level 0')
            clusters = a
        elif magic.rstrip(b'\0') == b'GSCLST2' and version == 2:
            if not 0 <= level < a:
                raise ValueError('Invalid parent hierarchy level')
            for index in range(level + 1):
                clusters = struct.unpack('<I', stream.read(4))[0]
                if index < level:
                    stream.seek(n * 4, 1)
        else:
            raise ValueError('Unsupported parent cluster file')
        labels = np.fromfile(stream, dtype='<i4', count=n)
        if len(labels) != n or np.any(labels < -1) or np.any(labels >= clusters):
            raise ValueError('Invalid parent labels')
    return labels


def split_overlap(xyz, parents, geometry, overlap_sigma, min_samples):
    """Return display groups, keeping each parent's noise separately."""
    labels = np.full(len(xyz), -1, dtype=np.int32)
    groups = []
    # Sorting once avoids scanning all Gaussians for every parent.
    order = np.argsort(parents, kind='stable')
    sorted_parents = parents[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_parents)) + 1, len(order)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        ids = order[start:stop]
        if not len(ids):
            continue
        parent = int(parents[ids[0]])
        print('Starting Parent {}: {:,} Gaussians; overlap_sigma={}, min_samples={}'.format(
            parent, len(ids), overlap_sigma, min_samples), flush=True)
        if parent < 0 or len(ids) < min_samples:
            local = np.full(len(ids), -1, dtype=np.int32)
        else:
            local = dbscan_overlap_bounded(
                xyz[ids], *(array[ids] for array in geometry), min_samples=min_samples)
        for subcluster in np.unique(local):
            selected = ids[local == subcluster]
            group = len(groups)
            labels[selected] = group
            groups.append({'group': group, 'parent': parent,
                           'subcluster': int(subcluster),
                           'noise': bool(subcluster < 0), 'count': int(len(selected))})
        print('Parent {}: {:,} Gaussians, {} subclusters, {:,} noise'.format(
            parent, len(ids), len(set(local)) - int(-1 in local),
            np.count_nonzero(local < 0)), flush=True)
    return labels, groups


def read_twice(path, count):
    with path.open('rb') as stream:
        magic, version, n, group_count, level = HEADER.unpack(stream.read(HEADER.size))
        if magic.rstrip(b'\0') != b'GSTWICE' or version != 1 or n != count:
            raise ValueError('Invalid twice file or PLY count mismatch')
        table = np.fromfile(stream, dtype='<i4', count=group_count * 3).reshape(-1, 3)
        labels = np.fromfile(stream, dtype='<i4', count=n)
        if len(labels) != n or np.any(labels < 0) or np.any(labels >= group_count):
            raise ValueError('Invalid twice labels')
    return labels, table, level


def run(args):
    ply_path = Path(args.ply_file).resolve()
    output = Path(args.output).resolve()
    ply = PlyData.read(str(ply_path))
    vertex = ply['vertex'].data
    if args.export_ply:
        metadata = json.loads(Path(str(output) + '.json').read_text())
        if metadata['source_ply'] != source_signature(ply_path):
            raise ValueError('Source PLY has changed since DBSCAN; rerun R before exporting')
        labels, table, _ = read_twice(output, len(vertex))
        keep = table[labels, 2] == 0
        destination = Path(args.export_ply).resolve()
        if destination == ply_path:
            raise ValueError('Export cannot overwrite the source PLY')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(destination) + '.tmp')
        # Keep all original Gaussian fields, values, dtypes and row order.
        filtered = PlyData([PlyElement.describe(vertex[keep].copy(), 'vertex')],
                           text=ply.text, byte_order=ply.byte_order,
                           comments=ply.comments, obj_info=ply.obj_info)
        filtered.write(str(temporary))
        os.replace(str(temporary), str(destination))
        print('Saved {:,}/{:,} Gaussians; removed {:,} noise -> {}'.format(
            np.count_nonzero(keep), len(vertex), np.count_nonzero(~keep), destination))
        return
    original_ply = source_signature(ply_path)
    parent_path = Path(args.cluster_file).resolve()
    original_cluster = source_signature(parent_path)
    parents = parent_labels(parent_path, args.hierarchy_level, len(vertex))
    xyz = np.stack([vertex[axis] for axis in ('x', 'y', 'z')], axis=1)
    if not np.isfinite(xyz).all():
        raise ValueError('Non-finite Gaussian centers')
    geometry = gaussian_geometry(vertex, args.overlap_sigma)
    labels, groups = split_overlap(xyz, parents, geometry, args.overlap_sigma, args.min_samples)
    if original_ply != source_signature(ply_path) or original_cluster != source_signature(parent_path):
        raise ValueError('Source files changed during DBSCAN; rerun')
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'source_ply': original_ply, 'source_clusters': original_cluster,
                'hierarchy_level': args.hierarchy_level, 'overlap_sigma': args.overlap_sigma,
                'dbscan_engine': 'bounded_memory_ellipsoid_overlap',
                'neighbor_definition': 'intersection_of_scale_rotation_sigma_ellipsoids',
                'border_assignment': 'first_intersecting_core_component',
                'min_samples': args.min_samples, 'groups': groups,
                'noise_count': sum(g['count'] for g in groups if g['noise']),
                'cluster_count': sum(not g['noise'] for g in groups)}
    temporary = Path(str(output) + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(HEADER.pack(b'GSTWICE', 1, len(vertex), len(groups), args.hierarchy_level))
        np.asarray([[g['parent'], g['subcluster'], int(g['noise'])] for g in groups],
                   dtype='<i4').tofile(stream)
        labels.astype('<i4').tofile(stream)
    Path(str(output) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    os.replace(str(temporary), str(output))
    print('Saved {} clusters and {} parent-specific noise groups -> {}'.format(
        metadata['cluster_count'], sum(g['noise'] for g in groups), output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply_file', required=True)
    parser.add_argument('--cluster_file')
    parser.add_argument('--hierarchy_level', type=int, default=0)
    parser.add_argument('--overlap_sigma', type=float, default=3.0,
                        help='ellipsoid extent in Gaussian standard deviations; default 3')
    parser.add_argument('--min_samples', type=int, default=10)
    parser.add_argument('--output', required=True)
    parser.add_argument('--export_ply')
    parser.add_argument('--log_file')
    args = parser.parse_args()
    if not args.export_ply and not args.cluster_file:
        parser.error('--cluster_file is required to compute DBSCAN')
    if not np.isfinite(args.overlap_sigma) or args.overlap_sigma <= 0 or args.min_samples < 1:
        parser.error('overlap_sigma and min_samples must be positive')
    if args.log_file:
        log = Path(args.log_file)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w', buffering=1) as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                try:
                    run(args)
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1)
    else:
        run(args)


if __name__ == '__main__':
    main()
