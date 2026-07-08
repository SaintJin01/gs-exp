#
# Floater metric for trained 3DGS point clouds.
#
# Detects floater candidates from a trained point_cloud.ply using a k-NN
# mean-distance / opacity based importance score, then saves:
#   <model_path>/floater/<tag>/floater_candidates.ply   (candidates only)
#   <model_path>/floater/<tag>/point_cloud_cleaned.ply  (original minus candidates)
#   <model_path>/floater/<tag>/stats.json               (counts, ratio, floater score)
#
# Scores:
#   d_norm_i      = clip((d_i - median(d)) / (p99.9(d) - median(d)), 0, 1)
#                   where d_i is the mean distance to the k nearest neighbours
#   importance_i  = w1 * d_norm_i + w2 * opacity_i      -> top n% become candidates
#   v_norm_i      = clip(volume_i / p90(volume), 0, 1)  with volume_i = prod(exp(scale))
#   impact_i      = d_norm_i * opacity_i * v_norm_i     (per-gaussian visual impact)
#   floater_score = sum(impact over candidates) / sum(opacity * v_norm over all)
#                   (fraction of the scene's visual mass attributed to floaters)
#
# Scene-radius filter: only gaussians within radius_mult * camera_extent of the
# camera centroid are eligible as candidates (from cameras.json, mirroring
# getNerfppNorm). Distant background gaussians are isolated in 3D but
# render-critical, so they must not be treated as floaters. radius_mult <= 0
# disables the filter.
#
# Usage:
#   python exp_utils/floater_metric.py -m output/train_bs_dbp_eval \
#       --k 10 --w1 1.0 --w2 1.0 --n 5
#

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def make_tag(k, w1, w2, n, radius_mult=1.0):
    return f"k{k}_w1-{w1:g}_w2-{w2:g}_n-{n:g}_r-{radius_mult:g}"


def resolve_ply_path(model_path, iteration=-1):
    pc_dir = Path(model_path) / "point_cloud"
    if iteration is None or iteration <= 0:
        iters = [int(p.name.split("_")[-1]) for p in pc_dir.glob("iteration_*")]
        if not iters:
            raise FileNotFoundError(f"No iteration_* folder under {pc_dir}")
        iteration = max(iters)
    return pc_dir / f"iteration_{iteration}" / "point_cloud.ply", iteration


def model_root_of(ply_path):
    ply_path = Path(ply_path).resolve()
    if ply_path.parent.name.startswith("iteration_") and ply_path.parent.parent.name == "point_cloud":
        return ply_path.parents[2]
    return ply_path.parent


def default_out_dir(ply_path, k, w1, w2, n, radius_mult=1.0):
    return model_root_of(ply_path) / "floater" / make_tag(k, w1, w2, n, radius_mult)


def load_camera_extent(model_root):
    """Camera bounding sphere from cameras.json, mirroring getNerfppNorm
    (center = mean of camera centers, radius = 1.1 * max distance)."""
    cams_path = Path(model_root) / "cameras.json"
    if not cams_path.exists():
        return None
    cams = json.loads(cams_path.read_text())
    pos = np.array([c["position"] for c in cams], dtype=np.float32)
    center = pos.mean(axis=0)
    radius = 1.1 * float(np.linalg.norm(pos - center, axis=1).max())
    return center, radius


def load_gaussians(ply_path):
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-vertex["opacity"].astype(np.float32)))
    log_volume = (vertex["scale_0"].astype(np.float32)
                  + vertex["scale_1"].astype(np.float32)
                  + vertex["scale_2"].astype(np.float32))
    volume = np.exp(np.clip(log_volume, -50.0, 50.0))
    return {"vertex": vertex, "xyz": xyz, "opacity": opacity, "volume": volume}


def knn_mean_distance(xyz, k=10, device=None, max_chunk_elems=2 ** 27):
    """Mean distance to the k nearest neighbours, chunked to bound GPU memory."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.from_numpy(xyz).to(device)
    n = pts.shape[0]
    chunk = max(1, min(n, max_chunk_elems // max(n, 1)))
    result = torch.empty(n, device=device)
    with torch.no_grad():
        for start in range(0, n, chunk):
            block = pts[start:start + chunk]
            dist = torch.cdist(block, pts)
            # k+1 smallest, drop the self-distance (0) in column 0
            knn = dist.topk(min(k + 1, n), dim=1, largest=False).values[:, 1:]
            result[start:start + chunk] = knn.mean(dim=1)
    return result.cpu().numpy()


def compute_scores(knn_dist, opacity, volume, w1, w2):
    med = float(np.median(knn_dist))
    hi = float(np.percentile(knn_dist, 99.9))
    d_norm = np.clip((knn_dist - med) / max(hi - med, 1e-12), 0.0, 1.0)

    importance = w1 * d_norm + w2 * opacity

    v_norm = np.clip(volume / max(float(np.percentile(volume, 90)), 1e-12), 0.0, 1.0)
    visual_weight = opacity * v_norm
    impact = d_norm * visual_weight
    return d_norm, importance, impact, visual_weight


def select_candidates(importance, n_percent, eligible=None):
    """Top n% (of all gaussians) by importance, restricted to eligible ones."""
    total = len(importance)
    num = int(round(total * n_percent / 100.0))
    mask = np.zeros(total, dtype=bool)
    pool = np.arange(total) if eligible is None else np.flatnonzero(eligible)
    num = min(num, len(pool))
    if num <= 0:
        return mask, float("inf")
    idx = pool[np.argpartition(-importance[pool], num - 1)[:num]]
    mask[idx] = True
    return mask, float(importance[idx].min())


def write_subset_ply(vertex, mask, out_path):
    subset = vertex[mask]
    el = PlyElement.describe(subset, "vertex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(out_path))


def run_metric(ply_path, k, w1, w2, n, radius_mult=1.0, out_dir=None, data=None, knn_dist=None):
    """Full pipeline. `data` / `knn_dist` can be passed in to reuse cached work."""
    ply_path = Path(ply_path)
    if out_dir is None:
        out_dir = default_out_dir(ply_path, k, w1, w2, n, radius_mult)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if data is None:
        print(f"Loading {ply_path}")
        data = load_gaussians(ply_path)
    if knn_dist is None:
        print(f"Computing {k}-NN mean distances for {len(data['opacity'])} gaussians")
        knn_dist = knn_mean_distance(data["xyz"], k)

    eligible = None
    camera_radius = None
    if radius_mult > 0:
        extent = load_camera_extent(model_root_of(ply_path))
        if extent is None:
            print("Warning: cameras.json not found, scene-radius filter disabled")
        else:
            center, camera_radius = extent
            dist_from_center = np.linalg.norm(data["xyz"] - center, axis=1)
            eligible = dist_from_center <= camera_radius * radius_mult

    d_norm, importance, impact, visual_weight = compute_scores(
        knn_dist, data["opacity"], data["volume"], w1, w2)
    mask, threshold = select_candidates(importance, n, eligible)

    num_total = int(len(importance))
    num_floaters = int(mask.sum())
    denom = float(visual_weight.sum())
    floater_score = float(impact[mask].sum() / max(denom, 1e-12))

    stats = {
        "params": {"k": k, "w1": w1, "w2": w2, "n": n, "radius_mult": radius_mult},
        "ply": str(ply_path),
        "num_gaussians": num_total,
        "num_eligible": int(eligible.sum()) if eligible is not None else num_total,
        "camera_radius": camera_radius,
        "num_floaters": num_floaters,
        "floater_ratio": num_floaters / max(num_total, 1),
        "floater_score": floater_score,
        "floater_impact_mean": float(impact[mask].mean()) if num_floaters else 0.0,
        "importance_threshold": threshold,
        "knn_dist": {
            "median": float(np.median(knn_dist)),
            "p99": float(np.percentile(knn_dist, 99)),
            "max": float(knn_dist.max()),
        },
        "candidate_opacity_mean": float(data["opacity"][mask].mean()) if num_floaters else 0.0,
    }

    write_subset_ply(data["vertex"], mask, out_dir / "floater_candidates.ply")
    write_subset_ply(data["vertex"], ~mask, out_dir / "point_cloud_cleaned.ply")
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[{make_tag(k, w1, w2, n, radius_mult)}] floaters: {num_floaters}/{num_total} "
          f"({stats['floater_ratio'] * 100:.2f}%), floater_score: {floater_score:.6f}")
    print(f"Results written to {out_dir}")
    return stats, out_dir


if __name__ == "__main__":
    parser = ArgumentParser(description="Floater metric for trained 3DGS ply")
    parser.add_argument("--model_path", "-m", type=str, default=None,
                        help="Trained model folder (contains point_cloud/iteration_*)")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--ply", type=str, default=None,
                        help="Direct path to a point_cloud.ply (overrides -m)")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbours")
    parser.add_argument("--w1", type=float, default=1.0, help="Weight of normalized k-NN distance")
    parser.add_argument("--w2", type=float, default=1.0, help="Weight of opacity")
    parser.add_argument("--n", type=float, default=5.0, help="Top n%% selected as floater candidates")
    parser.add_argument("--radius_mult", type=float, default=1.0,
                        help="Only gaussians within radius_mult * camera_extent are candidate-"
                             "eligible; <= 0 disables the filter")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.ply:
        ply_path = Path(args.ply)
    elif args.model_path:
        ply_path, _ = resolve_ply_path(args.model_path, args.iteration)
    else:
        parser.error("Either --ply or -m/--model_path is required")

    run_metric(ply_path, args.k, args.w1, args.w2, args.n,
               radius_mult=args.radius_mult, out_dir=args.out_dir)
