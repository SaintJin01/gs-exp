#!/usr/bin/env python3
"""HDBSCAN on learned features and xyz, with optional noise retention."""

import argparse
import json
import struct
from pathlib import Path

import hdbscan
import numpy as np
from plyfile import PlyData
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree


FEATURE_HEADER = struct.Struct("<8sIIII")
CLUSTER_HEADER = struct.Struct("<8sIIII")
LEVEL_HEADER = struct.Struct("<I")


def resolve_paths(args):
    if args.feature_file:
        feature_path = Path(args.feature_file).expanduser().resolve()
    elif args.model_path:
        feature_path = (
            Path(args.model_path).expanduser().resolve()
            / "sam_features" / "latest.gsfeat"
        )
    else:
        raise ValueError("Provide --model_path or --feature_file")
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output else feature_path.with_suffix(".gscluster")
    )
    return feature_path, output_path


def load_features(path):
    with path.open("rb") as stream:
        raw = stream.read(FEATURE_HEADER.size)
    if len(raw) != FEATURE_HEADER.size:
        raise ValueError("Truncated gsfeat header: {}".format(path))
    magic, version, count, dimension, _ = FEATURE_HEADER.unpack(raw)
    if magic.rstrip(b"\0") != b"GSFEAT1" or version != 1:
        raise ValueError("Unsupported gsfeat file: {}".format(path))
    expected = FEATURE_HEADER.size + count * dimension * 4
    if path.stat().st_size != expected:
        raise ValueError(
            "gsfeat size mismatch: expected {}, found {}".format(
                expected, path.stat().st_size
            )
        )
    return np.memmap(
        str(path), dtype="<f4", mode="r", offset=FEATURE_HEADER.size,
        shape=(count, dimension),
    )


def normalized_features(features, indices):
    """L2-normalize exported features directly, without PCA or channel gating."""
    points = np.asarray(features[indices], dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError("Gaussian features contain non-finite values")
    points = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    return points


def resolve_geometry_path(args, feature_path):
    if args.ply_file:
        source = Path(args.ply_file).expanduser().resolve()
    elif args.model_path:
        model = Path(args.model_path).expanduser().resolve()
        iteration = args.iteration
        config_path = feature_path.parent / "training_config.json"
        if iteration is None and feature_path.parent.parent == model and config_path.is_file():
            iteration = int(json.loads(config_path.read_text())["source_iteration"])
        if iteration is None or iteration < 0:
            candidates = []
            for candidate in (model / "point_cloud").glob("iteration_*"):
                suffix = candidate.name[len("iteration_"):]
                if suffix.isdigit() and (candidate / "point_cloud.ply").is_file():
                    candidates.append(int(suffix))
            if not candidates:
                raise FileNotFoundError("No point_cloud/iteration_*/point_cloud.ply in {}".format(model))
            iteration = max(candidates)
        source = model / "point_cloud" / ("iteration_{}".format(iteration)) / "point_cloud.ply"
    else:
        raise ValueError("--feature_file requires --ply_file for matching Gaussian xyz and sizes")
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def load_geometry_features(path, count):
    vertex = PlyData.read(str(path))["vertex"].data
    required = ("x", "y", "z")
    if len(vertex) != count:
        raise ValueError("PLY/feature count mismatch: {} vs {}".format(len(vertex), count))
    if not all(name in vertex.dtype.names for name in required):
        raise ValueError("PLY requires x, y, z fields")
    geometry = np.stack([vertex[name] for name in required], axis=1).astype(np.float32)
    if not np.isfinite(geometry).all():
        raise ValueError("Gaussian centers must be finite")
    return geometry


class JointFeatures:
    """Lazy weighted blocks; never L2-normalize the concatenation afterward."""
    def __init__(self, features, geometry, sample_indices, feature_weight=1.,
                 xyz_weight=1.):
        if geometry.shape != (len(features), 3):
            raise ValueError("Geometry must contain matching xyz")
        weights = np.asarray([feature_weight, xyz_weight], dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
            raise ValueError("Block weights must be finite, nonnegative, and not all zero")
        fitted = geometry[sample_indices]
        low, median, high = np.percentile(fitted, [25, 50, 75], axis=0)
        spread = high - low
        spread = np.where(spread > 1e-6, spread, 1.)
        self.geometry = ((geometry - median) / spread).astype(np.float32)
        self.geometry *= xyz_weight / np.sqrt(3.)
        self.features, self.feature_weight = features, feature_weight
        self.shape = (len(features), features.shape[1] + 3)
        self.metadata = dict(
            feature_weight=float(feature_weight), xyz_weight=float(xyz_weight),
            geometry_normalizer="sample_median_IQR", size_features_applied=False,
            geometry_median=median.tolist(), geometry_IQR_divisor=spread.tolist(),
            geometry_axis_order=["x","y","z"],
            geometry_block_divisor="sqrt(3)", concatenation_L2_normalized=False)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        semantic = normalized_features(self.features, indices) * self.feature_weight
        return np.concatenate((semantic, self.geometry[indices]), axis=1).astype(np.float32)


def cluster_centers(points, labels, count):
    centers = np.zeros((count, points.shape[1]), dtype=np.float64)
    sizes = np.bincount(labels, minlength=count).astype(np.int64)
    for component in range(points.shape[1]):
        centers[:, component] = np.bincount(
            labels, weights=points[:, component], minlength=count
        )
    centers /= np.maximum(sizes[:, None], 1)
    return centers.astype(np.float32), sizes


def fit_density_clusters(sample, args):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        cluster_selection_method=args.cluster_selection_method,
        metric="euclidean",
        prediction_data=True,
        allow_single_cluster=False,
        core_dist_n_jobs=args.jobs,
    )
    labels = clusterer.fit_predict(sample).astype(np.int32, copy=False)
    leaf_count = int(labels.max()) + 1
    noise_count = int(np.count_nonzero(labels < 0))
    print(
        "HDBSCAN found {} stable clusters; sample noise {:,}/{:,} ({:.2%})".format(
            leaf_count, noise_count, labels.size, noise_count / labels.size
        )
    )
    if leaf_count == 0:
        raise RuntimeError(
            "HDBSCAN found no stable cluster. Reduce --min_cluster_size or "
            "--min_samples, or use --cluster_selection_method leaf."
        )
    # Noise is not a cluster prototype and is excluded from the hierarchy.
    valid = labels >= 0
    centers, sizes = cluster_centers(sample[valid], labels[valid], leaf_count)
    return clusterer, centers, sizes, {
        "sample_stable_clusters": leaf_count,
        "sample_noise_count": noise_count,
        "sample_noise_ratio": noise_count / labels.size,
    }


def prediction_label_map(clusterer, batch_size):
    """Map prediction-tree labels to fitted labels, including epsilon merges.

    Older hdbscan versions construct prediction trees without selection epsilon.
    Calibrate on the fitted data rather than assuming the two label IDs match.
    """
    raw = clusterer.prediction_data_.raw_data
    fitted = clusterer.labels_
    maximum = max(clusterer.prediction_data_.cluster_map.values(), default=-1)
    mapping = np.full(maximum + 1, -1, dtype=np.int32)
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        predicted, _ = hdbscan.approximate_predict(clusterer, raw[start:stop])
        actual = fitted[start:stop]
        valid = (predicted >= 0) & (actual >= 0)
        for source, target in np.unique(np.stack((predicted[valid], actual[valid]), axis=1), axis=0):
            if mapping[source] >= 0 and mapping[source] != target:
                raise RuntimeError("Prediction-tree labels do not match fitted HDBSCAN clusters")
            mapping[source] = target
    return mapping


def predict_density_labels(features, clusterer, sample_indices, batch_size):
    """Keep exact fitted labels; predict unsampled points with rejection as noise."""
    labels = np.full(features.shape[0], -1, dtype=np.int32)
    fitted = np.zeros(features.shape[0], dtype=bool)
    fitted[sample_indices] = True
    if not fitted.all():
        mapping = prediction_label_map(clusterer, batch_size)
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            indices = np.flatnonzero(~fitted[start:stop]) + start
            if not len(indices):
                continue
            points = np.asarray(features[indices], dtype=np.float32)
            predicted, _ = hdbscan.approximate_predict(clusterer, points)
            valid = predicted >= 0
            labels[indices[valid]] = mapping[predicted[valid]]
    # Never re-predict the fitting sample: keep its original noise decisions.
    labels[sample_indices] = clusterer.labels_.astype(np.int32, copy=False)
    noise_count = int(np.count_nonzero(labels < 0))
    print("Density-assigned Gaussians; noise {:,}/{:,} ({:.2%})".format(
        noise_count, labels.size, noise_count / labels.size))
    return labels


def assign_noise_joint_space(features, labels, centers, batch_size):
    """Nearest Euclidean centroid in the same weighted joint space as HDBSCAN."""
    if not len(centers):
        raise ValueError("Cannot assign noise without a stable cluster")
    result = labels.copy()
    tree = cKDTree(np.asarray(centers, dtype=np.float64))
    reassigned = 0
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        indices = np.flatnonzero(result[start:stop] < 0) + start
        if not len(indices):
            continue
        points = np.asarray(features[indices], dtype=np.float64)
        result[indices] = tree.query(points, k=1)[1].astype(np.int32)
        reassigned += len(indices)
    print("Joint-space nearest-centroid assignment: {:,} noise assigned; remaining 0".format(reassigned))
    return result


def hierarchy_targets(leaf_count, requested_levels):
    if leaf_count == 1:
        return [1]
    values = np.geomspace(
        leaf_count, 1, num=min(max(requested_levels, 2), leaf_count)
    )
    targets = []
    for value in values:
        target = max(1, min(leaf_count, int(round(float(value)))))
        if not targets or target != targets[-1]:
            targets.append(target)
    if targets[0] != leaf_count:
        targets.insert(0, leaf_count)
    if targets[-1] != 1:
        targets.append(1)
    return targets


def build_hierarchy(leaf_centers, requested_levels):
    leaf_count = leaf_centers.shape[0]
    targets = hierarchy_targets(leaf_count, requested_levels)
    if leaf_count == 1:
        return [(1, np.zeros(1, dtype=np.int32))]
    tree = linkage(leaf_centers.astype(np.float64), method="ward")
    levels = []
    for target in targets:
        if target == leaf_count:
            mapping = np.arange(leaf_count, dtype=np.int32)
        elif target == 1:
            mapping = np.zeros(leaf_count, dtype=np.int32)
        else:
            raw = fcluster(tree, t=target, criterion="maxclust")
            _, mapping = np.unique(raw, return_inverse=True)
            mapping = mapping.astype(np.int32)
        levels.append((int(mapping.max()) + 1, mapping))
    return levels


def write_clusters(path, leaf_labels, levels, leaf_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(CLUSTER_HEADER.pack(
            b"GSCLST2", 2, leaf_labels.size, len(levels), leaf_count
        ))
        for cluster_count, leaf_to_level in levels:
            stream.write(LEVEL_HEADER.pack(cluster_count))
            labels = np.full(leaf_labels.shape, -1, dtype=np.int32)
            valid = leaf_labels >= 0
            labels[valid] = leaf_to_level[leaf_labels[valid]]
            labels.astype("<i4", copy=False).tofile(stream)


def build_parser():
    parser = argparse.ArgumentParser(
        description="HDBSCAN on learned feature + xyz, with optional noise retention"
    )
    parser.add_argument("-m", "--model_path")
    parser.add_argument("--feature_file")
    parser.add_argument("--ply_file", help="Matching PLY, same Gaussian ordering as gsfeat")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Geometry iteration; default training source iteration, then latest")
    parser.add_argument("--feature_weight", type=float, default=1.)
    parser.add_argument("--xyz_weight", type=float, default=1.)
    parser.add_argument("--keep_noise", action="store_true",
                        help="Keep HDBSCAN -1 labels; default assigns noise to nearest joint-space centroid")
    parser.add_argument("-o", "--output")
    parser.add_argument("--sample_ratio", type=float, default=0.02,
                        help="per-Gaussian random fitting probability; SAGA default 2%%")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="optional count override; 0 uses every Gaussian")
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=None,
                        help="defaults to min_cluster_size, as in SAGA")
    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.01)
    parser.add_argument("--cluster_selection_method", choices=("eom", "leaf"),
                        default="eom")
    parser.add_argument("--hierarchy_levels", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser


def main():
    args = build_parser().parse_args()

    feature_path, output_path = resolve_paths(args)
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if args.min_cluster_size < 2:
        raise ValueError("min cluster size must be >= 2")
    if not 0.0 < args.sample_ratio <= 1.0:
        raise ValueError("sample ratio must be in (0, 1]")
    if args.sample_size is not None and args.sample_size < 0:
        raise ValueError("sample size must be non-negative")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if (args.min_samples is not None and args.min_samples < 1) or args.hierarchy_levels < 1:
        raise ValueError("min samples and hierarchy levels must be positive")

    features = load_features(feature_path)
    count, dimension = features.shape
    print("Loaded {:,} x {} features from {}".format(count, dimension, feature_path))
    gate_metadata = {"scale_gate_applied": False}
    print("Scale gate disabled; using learned feature and xyz only")
    rng = np.random.RandomState(args.seed)
    if args.sample_size is None:
        sample_indices = np.flatnonzero(rng.rand(count) > 1.0 - args.sample_ratio)
    else:
        sample_count = count if args.sample_size == 0 else min(args.sample_size, count)
        sample_indices = (
            np.arange(count) if sample_count == count
            else rng.choice(count, sample_count, replace=False)
        )
    sample_count = sample_indices.size
    if sample_count < args.min_cluster_size:
        raise ValueError("Fitting sample is smaller than min_cluster_size")
    ply_path = resolve_geometry_path(args, feature_path)
    geometry = load_geometry_features(ply_path, count)
    inputs = JointFeatures(features, geometry, sample_indices, args.feature_weight,
                           args.xyz_weight)
    print("HDBSCAN input: {}D = {}D learned feature + 3D xyz; no PCA; fitting {:,}/{:,} ({:.2%})".format(
        inputs.shape[1], dimension, sample_count, count, sample_count / count))
    print("Block weights: feature={}, xyz={}; PLY: {}".format(
        args.feature_weight, args.xyz_weight, ply_path))
    sample = inputs[sample_indices]
    clusterer, leaf_centers, sample_leaf_sizes, sample_metadata = fit_density_clusters(sample, args)
    leaf_labels = predict_density_labels(inputs, clusterer, sample_indices, args.batch_size)
    density_noise_count = int(np.count_nonzero(leaf_labels < 0))
    if not args.keep_noise:
        leaf_labels = assign_noise_joint_space(inputs, leaf_labels, leaf_centers, args.batch_size)
    levels = build_hierarchy(leaf_centers, args.hierarchy_levels)
    write_clusters(output_path, leaf_labels, levels, leaf_centers.shape[0])

    level_counts = [cluster_count for cluster_count, _ in levels]
    leaf_sizes = np.bincount(
        leaf_labels[leaf_labels >= 0], minlength=leaf_centers.shape[0]
    )
    noise_count = int(np.count_nonzero(leaf_labels < 0))
    metadata = {
        "format": "GSCLST2",
        "feature_file": str(feature_path),
        "output_file": str(output_path),
        "num_gaussians": int(count),
        "feature_dimension": int(dimension),
        "input_representation": "weighted_L2_feature_and_robust_xyz",
        "ply_file": str(ply_path),
        "clustering_dimension": int(inputs.shape[1]),
        "hierarchy_representation": "centroids_in_same_weighted_joint_space",
        "pca_applied": False,
        "sample_size": int(sample_count),
        "sample_ratio": sample_count / count,
        "requested_sample_ratio": float(args.sample_ratio),
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples if args.min_samples is not None else args.min_cluster_size),
        "allow_single_cluster": False,
        "assignment": ("hdbscan_approximate_predict_with_noise_rejection" if args.keep_noise else
                       "hdbscan_then_noise_to_nearest_joint_space_centroid"),
        "sample_labels_preserved": bool(args.keep_noise),
        "sample_non_noise_labels_preserved": True,
        "keep_noise": bool(args.keep_noise),
        "noise_before_assignment": density_noise_count,
        "noise_reassigned_count": 0 if args.keep_noise else density_noise_count,
        "noise_label": -1,
        "sample_noise_prototype": False,
        "cluster_selection_method": args.cluster_selection_method,
        "cluster_selection_epsilon": float(args.cluster_selection_epsilon),
        "leaf_clusters": int(leaf_centers.shape[0]),
        "hierarchy_cluster_counts": level_counts,
        "leaf_cluster_sizes": leaf_sizes.tolist(),
        "noise_count": noise_count,
        "noise_ratio": noise_count / count,
        "sample_stable_cluster_sizes": sample_leaf_sizes.tolist(),
        "seed": int(args.seed),
    }
    metadata.update(inputs.metadata)
    metadata.update(sample_metadata)
    metadata.update(gate_metadata)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved {:,} Gaussian labels with hierarchy {} and {:,} noise to {}".format(
        count, " -> ".join(map(str, level_counts)), noise_count, output_path
    ))


if __name__ == "__main__":
    main()
