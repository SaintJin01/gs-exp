#
# Grid search over floater metric parameters (k, w1, w2, n).
#
# For every combination the floater candidates are removed and the cleaned
# point cloud is rendered/evaluated (via floater_metric_validation). The goal
# is to find parameters that remove floaters while minimizing quality loss —
# or ideally improving PSNR/SSIM/LPIPS over the original.
#
# The point cloud and per-k k-NN distances are computed once and reused across
# combinations. Finished combinations (comparison.json present) are skipped,
# so an interrupted search can be resumed. Results are written to:
#   <model_path>/floater/grid_search_results.json / .csv
#
# Usage:
#   python exp_utils/floater_metric_grid_search.py -m output/train_bs_dbp_eval \
#       --k_list 5 10 20 --w1_list 0.5 1.0 --w2_list 0.25 0.5 1.0 --n_list 1 3 5 10
#

import csv
import itertools
import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import floater_metric as fm
import floater_metric_validation as fmv

# sort key per objective: (value extractor, reverse)
OBJECTIVES = {
    "psnr": (lambda row: row["delta_PSNR"], True),
    "ssim": (lambda row: row["delta_SSIM"], True),
    "lpips": (lambda row: row["delta_LPIPS"], False),
}

CSV_FIELDS = ["k", "w1", "w2", "n", "radius_mult", "num_floaters", "floater_ratio",
              "floater_score", "PSNR", "SSIM", "LPIPS",
              "delta_PSNR", "delta_SSIM", "delta_LPIPS"]


def make_row(params, stats, comparison):
    row = dict(params)
    row["num_floaters"] = stats["num_floaters"]
    row["floater_ratio"] = stats["floater_ratio"]
    row["floater_score"] = stats["floater_score"]
    for key in fmv.METRIC_KEYS:
        row[key] = comparison["metrics"][key]["cleaned"]
        row[f"delta_{key}"] = comparison["metrics"][key]["delta"]
    return row


def cleanup_eval_dir(eval_dir, iteration):
    """Drop the bulky per-combo copies (renders + duplicated ply); keep results.json."""
    shutil.rmtree(eval_dir / "test", ignore_errors=True)
    ply = eval_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if ply.exists():
        ply.unlink()


def grid_search(model_path, k_list, w1_list, w2_list, n_list, r_list, iteration=-1,
                source_path=None, objective="psnr", force=False, keep_eval=False):
    ply_path, iteration = fm.resolve_ply_path(model_path, iteration)
    floater_root = ply_path.parents[2] / "floater"
    floater_root.mkdir(parents=True, exist_ok=True)

    print("Evaluating original point cloud (baseline)")
    original = fmv.render_and_evaluate(model_path, iteration, source_path)
    print("Baseline:", {key: round(original[key], 5) for key in fmv.METRIC_KEYS})

    data = None
    knn_cache = {}
    rows = []
    combos = list(itertools.product(k_list, w1_list, w2_list, n_list, r_list))

    for i, (k, w1, w2, n, r) in enumerate(combos):
        params = {"k": k, "w1": w1, "w2": w2, "n": n, "radius_mult": r}
        tag = fm.make_tag(k, w1, w2, n, r)
        floater_dir = floater_root / tag
        comparison_path = floater_dir / "comparison.json"
        print(f"\n[{i + 1}/{len(combos)}] {tag}")

        if not force and comparison_path.exists():
            comparison = json.loads(comparison_path.read_text())
            rows.append(make_row(params, comparison["floater_stats"], comparison))
            print("  cached, skipping")
            continue

        if data is None:
            data = fm.load_gaussians(ply_path)
        if k not in knn_cache:
            print(f"  computing {k}-NN mean distances")
            knn_cache[k] = fm.knn_mean_distance(data["xyz"], k)

        stats, _ = fm.run_metric(ply_path, k, w1, w2, n, radius_mult=r, out_dir=floater_dir,
                                 data=data, knn_dist=knn_cache[k])
        eval_dir = fmv.build_eval_dir(model_path, floater_dir, iteration)
        cleaned = fmv.render_and_evaluate(eval_dir, iteration, source_path, force=force)

        comparison = {
            "params": params,
            "iteration": iteration,
            "floater_stats": {key: stats[key] for key in
                              ("num_gaussians", "num_floaters", "floater_ratio", "floater_score")},
            "metrics": fmv.compare_metrics(original, cleaned),
        }
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        if not keep_eval:
            cleanup_eval_dir(eval_dir, iteration)

        rows.append(make_row(params, stats, comparison))
        save_results(floater_root, original, rows, objective, iteration)

    save_results(floater_root, original, rows, objective, iteration)
    report(original, rows, objective)
    return rows


def save_results(floater_root, original, rows, objective, iteration):
    key_fn, reverse = OBJECTIVES[objective]
    rows_sorted = sorted(rows, key=key_fn, reverse=reverse)
    with open(floater_root / "grid_search_results.json", "w") as f:
        json.dump({"iteration": iteration, "objective": objective,
                   "original": {key: original[key] for key in fmv.METRIC_KEYS},
                   "results": rows_sorted}, f, indent=2)
    with open(floater_root / "grid_search_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows_sorted)


def report(original, rows, objective):
    key_fn, reverse = OBJECTIVES[objective]
    rows_sorted = sorted(rows, key=key_fn, reverse=reverse)

    print(f"\n=== Grid search results (sorted by delta_{objective.upper()}) ===")
    print("Baseline:", {key: round(original[key], 5) for key in fmv.METRIC_KEYS})
    header = (f"{'k':>4s}{'w1':>7s}{'w2':>7s}{'n':>6s}{'r':>6s}{'ratio%':>8s}{'fl_score':>10s}"
              f"{'dPSNR':>9s}{'dSSIM':>9s}{'dLPIPS':>9s}")
    print(header)
    for row in rows_sorted:
        print(f"{row['k']:4d}{row['w1']:7g}{row['w2']:7g}{row['n']:6g}{row['radius_mult']:6g}"
              f"{row['floater_ratio'] * 100:8.2f}{row['floater_score']:10.5f}"
              f"{row['delta_PSNR']:+9.4f}{row['delta_SSIM']:+9.4f}{row['delta_LPIPS']:+9.4f}")

    best = rows_sorted[0]
    print(f"\nBest ({objective}): "
          f"{fm.make_tag(best['k'], best['w1'], best['w2'], best['n'], best['radius_mult'])} "
          f"-> dPSNR {best['delta_PSNR']:+.4f}, dSSIM {best['delta_SSIM']:+.4f}, "
          f"dLPIPS {best['delta_LPIPS']:+.4f}, "
          f"removed {best['num_floaters']} ({best['floater_ratio'] * 100:.2f}%)")


if __name__ == "__main__":
    parser = ArgumentParser(description="Grid search over floater metric parameters")
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--source_path", "-s", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--k_list", type=int, nargs="+", default=[10])
    parser.add_argument("--w1_list", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--w2_list", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--n_list", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    parser.add_argument("--r_list", type=float, nargs="+", default=[1.0],
                        help="radius_mult values (camera-extent multiplier); <= 0 disables")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="psnr",
                        help="Metric delta used to rank parameter combinations")
    parser.add_argument("--force", action="store_true",
                        help="Recompute combinations even if comparison.json exists")
    parser.add_argument("--keep_eval", action="store_true",
                        help="Keep per-combo eval renders and ply copies (uses a lot of disk)")
    args = parser.parse_args()

    grid_search(args.model_path, args.k_list, args.w1_list, args.w2_list, args.n_list,
                args.r_list, iteration=args.iteration, source_path=args.source_path,
                objective=args.objective, force=args.force, keep_eval=args.keep_eval)
