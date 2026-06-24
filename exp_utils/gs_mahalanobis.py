#!/usr/bin/env python3
"""
Compute Mahalanobis-distance statistics over all Gaussians of a trained model.

For every Gaussian i, its k nearest neighbours j (by Euclidean distance on the
centers) are found, and the pairwise Mahalanobis distance is computed as

    d_ij  = mu_i - mu_j
    S_ij  = Sigma_i + Sigma_j          (Sigma = R diag(s^2) R^T)
    r_ij  = sqrt(d_ij^T S_ij^-1 d_ij)

The mean / max / min of r_ij over all (i, j) pairs are reported.

Usage:
    python exp_utils/gs_mahalanobis.py -m output/<run> -k 3
    python exp_utils/gs_mahalanobis.py -m output/<run> -k 3 --iteration 7000
"""

import argparse
import os
import sys
from argparse import Namespace

import torch
from tqdm import tqdm

# Make the repo root importable (this file lives in exp_utils/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scene import GaussianModel
from utils.general_utils import build_rotation
from utils.system_utils import searchForMaxIteration


def read_sh_degree(model_path, default=3):
    """Read sh_degree from the model's cfg_args, falling back to a default."""
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        return default
    try:
        with open(cfg_path) as f:
            args = eval(f.read(), {"Namespace": Namespace})
        return int(getattr(args, "sh_degree", default))
    except Exception:
        return default


def resolve_ply(model_path, iteration=None):
    pc_root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_root):
        sys.exit(f"No point_cloud/ directory under {model_path}")
    if iteration is None:
        iteration = searchForMaxIteration(pc_root)
    ply = os.path.join(pc_root, f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.isfile(ply):
        sys.exit(f"point_cloud.ply not found at {ply}")
    return ply, iteration


def inv3x3(M):
    """Analytic inverse of a batch of 3x3 matrices: M is (..., 3, 3)."""
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]

    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I = a * e - b * d

    det = a * A + b * B + c * C
    # adjugate (transpose of cofactor matrix) / det
    inv = torch.stack([
        torch.stack([A, D, G], dim=-1),
        torch.stack([B, E, H], dim=-1),
        torch.stack([C, F, I], dim=-1),
    ], dim=-2)
    return inv / det.unsqueeze(-1).unsqueeze(-1)


@torch.no_grad()
def knn_indices(xyz, q_start, q_end, k, db_chunk):
    """k nearest neighbours (exclude self) for queries xyz[q_start:q_end].

    The reference set is tiled over `db_chunk` columns so that topk only ever
    runs over a small (k + db_chunk) dimension — this avoids the CUDA topk
    illegal-access bug on very large reduction dimensions, and NaN distances
    are mapped to +inf so they are never selected.
    """
    dev = xyz.device
    N = xyz.shape[0]
    q = xyz[q_start:q_end]
    rows = q_end - q_start

    best_d = torch.full((rows, k), float("inf"), device=dev)
    best_idx = torch.zeros((rows, k), dtype=torch.long, device=dev)

    for j in range(0, N, db_chunk):
        jend = min(j + db_chunk, N)
        dblock = torch.cdist(q, xyz[j:jend])           # (rows, db)
        dblock = torch.nan_to_num(dblock, nan=float("inf"),
                                  posinf=float("inf"), neginf=float("inf"))
        # exclude self (global col index == global row index)
        lo, hi = max(q_start, j), min(q_end, jend)
        if lo < hi:
            g = torch.arange(lo, hi, device=dev)
            dblock[g - q_start, g - j] = float("inf")

        idx_block = torch.arange(j, jend, device=dev).expand(rows, -1)
        cat_d = torch.cat([best_d, dblock], dim=1)
        cat_idx = torch.cat([best_idx, idx_block], dim=1)
        kk = min(k, cat_d.shape[1])
        best_d, sel = cat_d.topk(kk, largest=False)    # topk over small dim
        best_idx = torch.gather(cat_idx, 1, sel)

    return best_idx


@torch.no_grad()
def mahalanobis_stats(gaussians, k, chunk_size=2048, db_chunk=16384, eps=1e-6):
    xyz = gaussians.get_xyz                            # (N, 3)
    scales = gaussians.get_scaling                     # (N, 3)
    rots_q = gaussians.get_rotation                    # (N, 4)

    # Drop non-finite Gaussians (a diverged model can contain NaN/Inf)
    finite = torch.isfinite(xyz).all(dim=1) & torch.isfinite(scales).all(dim=1) \
        & torch.isfinite(rots_q).all(dim=1)
    n_bad = int((~finite).sum().item())
    if n_bad:
        print(f"[gs_mahalanobis] WARNING: dropping {n_bad} non-finite Gaussians")
        xyz, scales, rots_q = xyz[finite], scales[finite], rots_q[finite]

    N = xyz.shape[0]
    if N <= 1:
        sys.exit(f"Need at least 2 finite Gaussians, got {N}.")
    k = min(k, N - 1)

    # Sigma_i = R_i diag(s_i^2) R_i^T
    rots = build_rotation(rots_q)                      # (N, 3, 3)
    S2 = scales ** 2
    cov = (rots * S2.unsqueeze(1)) @ rots.transpose(-1, -2)   # (N, 3, 3)

    eye = eps * torch.eye(3, device=xyz.device)

    # Accumulate stats in chunks to keep memory bounded
    r_sum = torch.zeros((), device=xyz.device, dtype=torch.float64)
    r_count = 0
    r_max = torch.tensor(float("-inf"), device=xyz.device)
    r_min = torch.tensor(float("inf"), device=xyz.device)

    n_chunks = (N + chunk_size - 1) // chunk_size
    for i in tqdm(range(0, N, chunk_size), total=n_chunks, desc="Mahalanobis"):
        end = min(i + chunk_size, N)
        nn_idx = knn_indices(xyz, i, end, k, db_chunk)  # (chunk, k)

        d_vec = xyz[i:end].unsqueeze(1) - xyz[nn_idx]  # (chunk, k, 3)
        S_ij = cov[i:end].unsqueeze(1) + cov[nn_idx] + eye  # (chunk, k, 3, 3)

        # r_ij^2 = d_ij^T S_ij^-1 d_ij  (analytic 3x3 inverse; avoids cusolver)
        Sinv_d = (inv3x3(S_ij) @ d_vec.unsqueeze(-1)).squeeze(-1)  # (chunk, k, 3)
        r2 = (d_vec * Sinv_d).sum(-1)                  # (chunk, k)
        r = r2.clamp(min=0).sqrt()

        r_sum += r.double().sum()
        r_count += r.numel()
        r_max = torch.maximum(r_max, r.max())
        r_min = torch.minimum(r_min, r.min())

    mean = (r_sum / r_count).item()
    return {
        "N": N,
        "k": k,
        "pairs": r_count,
        "mean": mean,
        "max": r_max.item(),
        "min": r_min.item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Mahalanobis-distance statistics over all Gaussians of a model."
    )
    parser.add_argument("-m", "--model_path", required=True,
                        help="Trained model folder (containing point_cloud/).")
    parser.add_argument("-k", "--k", type=int, default=3,
                        help="Number of nearest neighbours per Gaussian.")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Specific iteration to load (default: latest).")
    parser.add_argument("--chunk_size", type=int, default=2048,
                        help="Query chunk size for the kNN / distance computation.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required (GaussianModel allocates on cuda).")

    ply, iteration = resolve_ply(args.model_path, args.iteration)
    sh_degree = read_sh_degree(args.model_path)

    print(f"[gs_mahalanobis] model     : {args.model_path}")
    print(f"[gs_mahalanobis] iteration : {iteration}")
    print(f"[gs_mahalanobis] ply       : {ply}")
    print(f"[gs_mahalanobis] sh_degree : {sh_degree}, k = {args.k}")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply)

    stats = mahalanobis_stats(gaussians, args.k, chunk_size=args.chunk_size)

    print("\n=== Mahalanobis distance (r_ij) over all Gaussians ===")
    print(f"  Gaussians (N) : {stats['N']}")
    print(f"  neighbours (k): {stats['k']}")
    print(f"  pairs         : {stats['pairs']}")
    print(f"  mean          : {stats['mean']:.6f}")
    print(f"  max           : {stats['max']:.6f}")
    print(f"  min           : {stats['min']:.6f}")


if __name__ == "__main__":
    main()
