#
# Validates the floater metric by comparing render quality (PSNR/SSIM/LPIPS)
# of the original point cloud vs. the floater-removed point cloud.
#
# Reuses render.py / metrics.py by building a minimal eval model folder
# (cfg_args + point_cloud/iteration_X/point_cloud.ply) under the floater
# result directory:
#   <model_path>/floater/<tag>/eval/
# and writes the comparison to:
#   <model_path>/floater/<tag>/comparison.json
#
# Usage:
#   python exp_utils/floater_metric_validation.py -m output/train_bs_dbp_eval \
#       --k 10 --w1 1.0 --w2 1.0 --n 5
#

import json
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import floater_metric as fm

REPO_ROOT = Path(__file__).resolve().parent.parent

METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")


def run_cmd(cmd):
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT), check=True)


def render_and_evaluate(model_dir, iteration, source_path=None, force=False):
    """Render the test set and run metrics.py; returns {PSNR, SSIM, LPIPS}.

    Skips work already present in <model_dir>/results.json unless force=True.
    """
    model_dir = Path(model_dir).resolve()
    method = f"ours_{iteration}"
    results_path = model_dir / "results.json"

    if not force and results_path.exists():
        results = json.loads(results_path.read_text())
        if method in results:
            return results[method]

    renders_dir = model_dir / "test" / method / "renders"
    if force or not renders_dir.exists() or not any(renders_dir.iterdir()):
        cmd = [sys.executable, "render.py", "-m", model_dir,
               "--iteration", iteration, "--skip_train", "--quiet"]
        if source_path:
            cmd += ["-s", source_path]
        run_cmd(cmd)

    run_cmd([sys.executable, "metrics.py", "-m", model_dir])

    results = json.loads(results_path.read_text())
    if method not in results:
        raise RuntimeError(f"metrics.py did not produce '{method}' in {results_path}")
    return results[method]


def build_eval_dir(model_path, floater_dir, iteration):
    """Minimal model folder so render.py can load the cleaned ply."""
    floater_dir = Path(floater_dir)
    eval_dir = floater_dir / "eval"
    dst_ply = eval_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(floater_dir / "point_cloud_cleaned.ply", dst_ply)
    for fname in ("cfg_args", "exposure.json"):
        src = Path(model_path) / fname
        if src.exists():
            shutil.copyfile(src, eval_dir / fname)
    return eval_dir


def compare_metrics(original, cleaned):
    return {key: {"original": original[key],
                  "cleaned": cleaned[key],
                  "delta": cleaned[key] - original[key]}
            for key in METRIC_KEYS}


def validate(model_path, k, w1, w2, n, radius_mult=1.0, iteration=-1, source_path=None, force=False):
    ply_path, iteration = fm.resolve_ply_path(model_path, iteration)
    floater_dir = fm.default_out_dir(ply_path, k, w1, w2, n, radius_mult)

    stats_path = floater_dir / "stats.json"
    cleaned_ply = floater_dir / "point_cloud_cleaned.ply"
    if force or not stats_path.exists() or not cleaned_ply.exists():
        stats, _ = fm.run_metric(ply_path, k, w1, w2, n, radius_mult=radius_mult, out_dir=floater_dir)
    else:
        stats = json.loads(stats_path.read_text())

    print("Evaluating original point cloud")
    original = render_and_evaluate(model_path, iteration, source_path)

    print("Evaluating floater-removed point cloud")
    eval_dir = build_eval_dir(model_path, floater_dir, iteration)
    cleaned = render_and_evaluate(eval_dir, iteration, source_path, force=force)

    comparison = {
        "params": stats["params"],
        "iteration": iteration,
        "floater_stats": {key: stats[key] for key in
                          ("num_gaussians", "num_floaters", "floater_ratio", "floater_score")},
        "metrics": compare_metrics(original, cleaned),
    }
    with open(floater_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n=== {fm.make_tag(k, w1, w2, n, radius_mult)} | iteration {iteration} ===")
    print(f"floaters removed: {stats['num_floaters']}/{stats['num_gaussians']} "
          f"({stats['floater_ratio'] * 100:.2f}%), floater_score: {stats['floater_score']:.6f}")
    print(f"{'':8s}{'original':>12s}{'cleaned':>12s}{'delta':>12s}")
    for key in METRIC_KEYS:
        m = comparison["metrics"][key]
        print(f"{key:8s}{m['original']:12.5f}{m['cleaned']:12.5f}{m['delta']:+12.5f}")
    print(f"Comparison written to {floater_dir / 'comparison.json'}")
    return comparison


if __name__ == "__main__":
    parser = ArgumentParser(description="Validate floater metric via render/metric comparison")
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--source_path", "-s", type=str, default=None,
                        help="Dataset source path (defaults to the one stored in cfg_args)")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--n", type=float, default=5.0)
    parser.add_argument("--radius_mult", type=float, default=1.0,
                        help="Only gaussians within radius_mult * camera_extent are candidate-"
                             "eligible; <= 0 disables the filter")
    parser.add_argument("--force", action="store_true",
                        help="Recompute floater metric and cleaned renders even if cached")
    args = parser.parse_args()

    validate(args.model_path, args.k, args.w1, args.w2, args.n, radius_mult=args.radius_mult,
             iteration=args.iteration, source_path=args.source_path, force=args.force)
