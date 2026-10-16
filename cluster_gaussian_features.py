#!/usr/bin/env python3
"""SAGA-style HDBSCAN prototypes and cosine assignment, with Rex hierarchy."""

import argparse
import json
import struct
from pathlib import Path

import hdbscan
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


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


def normalized_features(features, indices, gates=None):
    """SAGA GUI: normalize exported features, multiply scale gate, normalize."""
    points = np.asarray(features[indices], dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError("Gaussian features contain non-finite values")
    points = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    if gates is not None:
        points = points * gates[None, :]
        points /= np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    return points


def load_scale_gates(feature_path, dimension, count, args):
    """Read the trained gate, never silently cluster without scale conditioning.

--scale is the SAGA GUI's normalized [0,1] value, not a physical 3D size.
The training quantile transform must NOT be applied again at this point.
"""
    import torch
    explicit = getattr(args, "scale_gate_file", None)
    if explicit:
        gate_path = Path(explicit).expanduser().resolve()
    else:
        checkpoint = feature_path.with_suffix(".pt")
        gate_path = checkpoint if checkpoint.is_file() else feature_path.parent / "scale_gate.pt"
    if not gate_path.is_file():
        raise FileNotFoundError(
            "Trained scale gate not found: {}. Use newly trained SAGA features, "
            "or specify --scale_gate_file path/to/scale_gate.pt".format(gate_path))
    payload = torch.load(str(gate_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Expected scale gate state_dict or feature checkpoint: {}".format(gate_path))
    training = payload.get("training", {})
    if "features" in payload:
        if "scale_gate" not in payload:
            raise ValueError("Checkpoint has no trained scale gate; retrain with the SAGA trainer: {}".format(gate_path))
        if tuple(payload["features"].shape) != (count, dimension):
            raise ValueError("Scale gate checkpoint/gsfeat Gaussian count or dimension mismatch")
    state = payload.get("scale_gate", payload)
    gate = torch.nn.Sequential(torch.nn.Linear(1, dimension), torch.nn.Sigmoid())
    gate.load_state_dict(state, strict=True)
    gate.eval()
    if not all(torch.isfinite(parameter).all() for parameter in gate.parameters()):
        raise ValueError("Scale gate contains nonfinite parameters")
    aware = int(training.get("scale_aware_dim", -1))
    with torch.no_grad():
        if 0 < aware < dimension:
            level = int((1.0 - args.scale) * aware)
            values = (torch.arange(dimension) < dimension - aware + level).float()
        else:
            values = gate(torch.tensor([[args.scale]], dtype=torch.float32))[0]
    values = values.cpu().numpy().astype(np.float32)
    return values, {
        "scale_gate_applied": True,
        "scale_gate_file": str(gate_path),
        "scale": float(args.scale),
        "scale_domain": "normalized_SAGA_GUI_0_to_1",
        "scale_quantile_reapplied": False,
        "scale_aware_dim": aware,
        "scale_gate_values": values.tolist(),
    }


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
    # The official SAGA GUI also averages sample noise (-1) into a prototype.
    # Enumerate actual labels explicitly so the no-noise case remains valid.
    prototype_source_labels, prototype_labels = np.unique(labels, return_inverse=True)
    centers, sizes = cluster_centers(
        sample, prototype_labels, len(prototype_source_labels)
    )
    centers = normalized_features(centers, slice(None))
    return centers, sizes, {
        "sample_stable_clusters": leaf_count,
        "sample_noise_count": noise_count,
        "sample_noise_ratio": noise_count / labels.size,
        "prototype_source_labels": prototype_source_labels.tolist(),
    }


def predict_cosine_labels(features, centers, batch_size, gates=None):
    """Assign every Gaussian to its highest-cosine prototype, as in SAGA."""
    labels = np.empty(features.shape[0], dtype=np.int32)
    # Bound the temporary similarity matrix to about 64 MiB.
    batch_size = min(batch_size, max(1, (16 * 1024 * 1024) // centers.shape[0]))
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        points = normalized_features(features, slice(start, stop), gates)
        labels[start:stop] = np.argmax(points @ centers.T, axis=1).astype(np.int32)
    noise_count = int(np.count_nonzero(labels < 0))
    print("Cosine-assigned all Gaussians; unassigned {:,}/{:,} ({:.2%})".format(
        noise_count, labels.size, noise_count / labels.size
    ))
    return labels


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
        description="SAGA-style HDBSCAN and cosine prototype assignment with Rex hierarchy"
    )
    parser.add_argument("-m", "--model_path")
    parser.add_argument("--feature_file")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="SAGA GUI normalized scale [0,1], default 0.5; no second quantile transform")
    parser.add_argument("--scale_gate_file", type=str, default=None,
                        help="trained scale_gate.pt or feature checkpoint; auto-detect beside gsfeat")
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
    if not np.isfinite(args.scale) or not 0.0 <= args.scale <= 1.0:
        raise ValueError("--scale must be in [0,1] (SAGA normalized GUI scale)")
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
    gates, gate_metadata = load_scale_gates(feature_path, dimension, count, args)
    print("Scale gate: scale={:.3f}, weights {:.4f}..{:.4f}, loaded {}".format(
        args.scale, float(gates.min()), float(gates.max()), gate_metadata["scale_gate_file"]))
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
    print("HDBSCAN input: {}D scale-gated normalized features, no PCA; fitting {:,}/{:,} ({:.2%})".format(
        dimension, sample_count, count, sample_count / count
    ))
    sample = normalized_features(features, sample_indices, gates)
    leaf_centers, sample_leaf_sizes, sample_metadata = fit_density_clusters(sample, args)
    leaf_labels = predict_cosine_labels(
        features, leaf_centers, args.batch_size, gates,
    )
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
        "input_representation": "normalize(normalize(exported_smoothed_features)*scale_gate(scale))",
        "pca_applied": False,
        "sample_size": int(sample_count),
        "sample_ratio": sample_count / count,
        "requested_sample_ratio": float(args.sample_ratio),
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples if args.min_samples is not None else args.min_cluster_size),
        "allow_single_cluster": False,
        "assignment": "maximum_cosine_to_normalized_sample_prototypes",
        "sample_noise_prototype": True,
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
    metadata.update(sample_metadata)
    metadata.update(gate_metadata)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved {:,} Gaussian labels with hierarchy {} and {:,} noise to {}".format(
        count, " -> ".join(map(str, level_counts)), noise_count, output_path
    ))


if __name__ == "__main__":
    main()
