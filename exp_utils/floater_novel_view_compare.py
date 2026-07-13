#
# Novel-view comparison for floater metric validation.
#
# Test-view metrics barely register floaters: they were optimized to look
# correct from the training trajectory. This script quantifies the visual
# impact of the removed candidates OFF the trajectory, where floaters actually
# hurt. There is no GT for novel views, so we compare the original render
# against the cleaned render from identical cameras:
#
#   - trajectory views: cameras straight from cameras.json
#   - novel views: interpolated between neighbouring cameras + random offset
#                  (offset_scale * camera_extent)
#
# If the removed gaussians are floaters, orig-vs-cleaned difference should be
# much larger on novel views than on trajectory views (visibility ratio >> 1).
#
# Outputs under <floater_dir>/novel_view/:
#   report.json, and per-view renders orig_/clean_/diff_ (diff amplified x5)
#
# Usage:
#   python exp_utils/floater_novel_view_compare.py -m <model> \
#       --k 5 --w1 1.0 --w2 4.0 --n 0.5 --radius_mult 0.25 --num_views 20
#

import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import floater_metric as fm  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402
import torchvision  # noqa: E402


def rotmat_to_quat(m):
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-6:
        x = (m[2, 1] - m[1, 2]) / (4 * w)
        y = (m[0, 2] - m[2, 0]) / (4 * w)
        z = (m[1, 0] - m[0, 1]) / (4 * w)
    else:  # fall back for 180-degree rotations
        x = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0
        y = (m[0, 1] + m[1, 0]) / (4 * x + 1e-12)
        z = (m[0, 2] + m[2, 0]) / (4 * x + 1e-12)
        w = (m[2, 1] - m[1, 2]) / (4 * x + 1e-12)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp(qa, qb, t):
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = qa + t * (qb - qa)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        q = (np.sin((1 - t) * theta) * qa + np.sin(t * theta) * qb) / np.sin(theta)
    return q / np.linalg.norm(q)


def load_cameras_json(model_path):
    cams = json.loads((Path(model_path) / "cameras.json").read_text())
    for c in cams:
        c["position"] = np.array(c["position"], dtype=np.float64)
        c["rotation"] = np.array(c["rotation"], dtype=np.float64)  # C2W rotation
    return cams


def build_minicam(rot_c2w, position, fx, fy, width, height, znear=0.01, zfar=100.0):
    # Same convention as scene.cameras.Camera: R is C2W rotation, T is W2C translation
    R = rot_c2w
    T = -rot_c2w.T @ position
    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))
    world_view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)


def sample_trajectory_views(cams, num_views, rng):
    idx = rng.choice(len(cams), size=min(num_views, len(cams)), replace=False)
    views = []
    for i in idx:
        c = cams[i]
        views.append(build_minicam(c["rotation"], c["position"],
                                   c["fx"], c["fy"], c["width"], c["height"]))
    return views


def sample_novel_views(cams, num_views, offset_scale, extent_radius, rng):
    pos = np.stack([c["position"] for c in cams])
    dists = np.linalg.norm(pos[None, :] - pos[:, None], axis=-1)
    np.fill_diagonal(dists, np.inf)
    nearest = dists.argmin(axis=1)

    views = []
    for _ in range(num_views):
        a = int(rng.integers(len(cams)))
        b = int(nearest[a])
        t = float(rng.uniform(0.25, 0.75))
        p = (1 - t) * cams[a]["position"] + t * cams[b]["position"]
        q = slerp(rotmat_to_quat(cams[a]["rotation"]), rotmat_to_quat(cams[b]["rotation"]), t)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        p = p + direction * offset_scale * extent_radius
        c = cams[a]
        views.append(build_minicam(quat_to_rotmat(q), p,
                                   c["fx"], c["fy"], c["width"], c["height"]))
    return views


def load_model(ply_path, sh_degree):
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path))
    return gaussians


def compare_views(views, orig, cleaned, pipe, background, out_dir, prefix, save_images):
    out_dir.mkdir(parents=True, exist_ok=True)
    l1s, ssims = [], []
    with torch.no_grad():
        for i, view in enumerate(views):
            img_o = render(view, orig, pipe, background)["render"]
            img_c = render(view, cleaned, pipe, background)["render"]
            l1s.append(float((img_o - img_c).abs().mean()))
            ssims.append(float(ssim(img_o.unsqueeze(0), img_c.unsqueeze(0))))
            if save_images:
                torchvision.utils.save_image(img_o, out_dir / f"{prefix}_{i:03d}_orig.png")
                torchvision.utils.save_image(img_c, out_dir / f"{prefix}_{i:03d}_clean.png")
                diff = (img_o - img_c).abs().mean(0, keepdim=True)
                torchvision.utils.save_image(diff.clamp(0, 1 / 5) * 5,
                                             out_dir / f"{prefix}_{i:03d}_diff.png")
    return {"l1_mean": float(np.mean(l1s)), "l1_max": float(np.max(l1s)),
            "ssim_mean": float(np.mean(ssims)), "per_view_l1": l1s, "per_view_ssim": ssims}


def run_compare(model_path, k, w1, w2, n, radius_mult, iteration=-1, num_views=20,
                offset_scale=0.15, seed=0, save_images=True):
    ply_path, iteration = fm.resolve_ply_path(model_path, iteration)
    floater_dir = fm.default_out_dir(ply_path, k, w1, w2, n, radius_mult)
    cleaned_ply = floater_dir / "point_cloud_cleaned.ply"
    if not cleaned_ply.exists():
        fm.run_metric(ply_path, k, w1, w2, n, radius_mult=radius_mult, out_dir=floater_dir)

    cfg = eval((Path(model_path) / "cfg_args").read_text())
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     debug=False, antialiasing=False)
    bg = [1, 1, 1] if getattr(cfg, "white_background", False) else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    cams = load_cameras_json(model_path)
    center, radius = fm.load_camera_extent(model_path)
    rng = np.random.default_rng(seed)
    trajectory_views = sample_trajectory_views(cams, num_views, rng)
    novel_views = sample_novel_views(cams, num_views, offset_scale, radius, rng)

    print(f"Loading original ({ply_path}) and cleaned ({cleaned_ply})")
    orig = load_model(ply_path, cfg.sh_degree)
    cleaned = load_model(cleaned_ply, cfg.sh_degree)

    out_dir = floater_dir / "novel_view"
    print(f"Comparing {num_views} trajectory views")
    traj = compare_views(trajectory_views, orig, cleaned, pipe, background,
                         out_dir / "trajectory", "traj", save_images)
    print(f"Comparing {num_views} novel views (offset {offset_scale} x extent)")
    novel = compare_views(novel_views, orig, cleaned, pipe, background,
                          out_dir / "novel", "novel", save_images)

    ratio = novel["l1_mean"] / max(traj["l1_mean"], 1e-12)
    report = {
        "params": {"k": k, "w1": w1, "w2": w2, "n": n, "radius_mult": radius_mult},
        "iteration": iteration,
        "num_views": num_views, "offset_scale": offset_scale, "seed": seed,
        "trajectory": traj, "novel": novel,
        "novel_visibility_ratio": ratio,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {fm.make_tag(k, w1, w2, n, radius_mult)} | novel-view comparison ===")
    print(f"{'':14s}{'L1 diff':>12s}{'SSIM':>10s}")
    print(f"{'trajectory':14s}{traj['l1_mean']:12.6f}{traj['ssim_mean']:10.5f}")
    print(f"{'novel':14s}{novel['l1_mean']:12.6f}{novel['ssim_mean']:10.5f}")
    print(f"novel visibility ratio (novel L1 / trajectory L1): {ratio:.2f}")
    print(f"Report written to {out_dir / 'report.json'}")
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare original vs cleaned renders on novel views")
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--n", type=float, default=5.0)
    parser.add_argument("--radius_mult", type=float, default=1.0)
    parser.add_argument("--num_views", type=int, default=20)
    parser.add_argument("--offset_scale", type=float, default=0.15,
                        help="Novel-view position offset as a fraction of camera extent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_images", action="store_true", help="Skip saving per-view images")
    args = parser.parse_args()

    run_compare(args.model_path, args.k, args.w1, args.w2, args.n, args.radius_mult,
                iteration=args.iteration, num_views=args.num_views,
                offset_scale=args.offset_scale, seed=args.seed,
                save_images=not args.no_images)
