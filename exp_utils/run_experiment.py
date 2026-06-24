#!/usr/bin/env python3
"""
Repulsion-loss experiment runner.

Runs train.py with a given (repuls_margin, lambda_repuls, repuls_k,
repuls_from_iter, repuls_warmup) combination, writes the result into an output
folder whose name encodes those values, and records the training time (and final
Gaussian count) into that folder.

Example:
    python exp_utils/run_experiment.py -s /path/to/dataset \
        --repuls_margin 1.0 --lambda_repuls 1.0 --repuls_k 3 \
        --repuls_from_iter 500 --repuls_warmup 2000

Any extra arguments are forwarded as-is to train.py, e.g.:
    python exp_utils/run_experiment.py -s /path/to/dataset \
        -- --iterations 7000 --eval
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# Repo root (parent of exp_utils/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(REPO_ROOT, "train.py")


def fmt(v):
    """Filesystem-safe compact float/int formatting (1.0 -> '1', 0.01 -> '0p01')."""
    s = f"{v:g}"
    return s.replace(".", "p").replace("-", "m")


def make_run_name(repuls_margin, lambda_repuls, repuls_k,
                  repuls_from_iter, repuls_warmup):
    name = (
        f"repuls_m{fmt(repuls_margin)}"
        f"_w{fmt(lambda_repuls)}"
        f"_k{fmt(repuls_k)}"
    )
    if repuls_from_iter:
        name += f"_from{fmt(repuls_from_iter)}"
    if repuls_warmup:
        name += f"_wu{fmt(repuls_warmup)}"
    return name


def count_gaussians(model_path):
    """Count vertices in the final point_cloud.ply produced by training."""
    pc_root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_root):
        return None
    iters = []
    for name in os.listdir(pc_root):
        if name.startswith("iteration_"):
            try:
                iters.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    if not iters:
        return None
    ply = os.path.join(pc_root, f"iteration_{max(iters)}", "point_cloud.ply")
    if not os.path.isfile(ply):
        return None
    with open(ply, "rb") as f:
        for line in f:
            line = line.strip()
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line == b"end_header":
                break
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run a repulsion-loss training experiment.",
        epilog="Extra args after a literal '--' are forwarded to train.py.",
    )
    parser.add_argument("-s", "--source_path", required=True,
                        help="Dataset source path (passed to train.py as -s).")
    parser.add_argument("--repuls_margin", type=float, default=1.0)
    parser.add_argument("--lambda_repuls", type=float, default=1.0)
    parser.add_argument("--repuls_k", type=int, default=2)
    parser.add_argument("--repuls_from_iter", type=int, default=0)
    parser.add_argument("--repuls_warmup", type=int, default=0)
    parser.add_argument("--output_root", default=os.path.join(REPO_ROOT, "output"),
                        help="Base directory for experiment outputs.")
    parser.add_argument("--no_eval", action="store_true",
                        help="Disable the train/test split (--eval is on by default).")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to train.py (prefix with --).")
    args = parser.parse_args()

    run_name = make_run_name(
        args.repuls_margin, args.lambda_repuls, args.repuls_k,
        args.repuls_from_iter, args.repuls_warmup
    )
    model_path = os.path.join(args.output_root, run_name)
    os.makedirs(model_path, exist_ok=True)

    # Strip a leading literal '--' that argparse.REMAINDER keeps.
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = [
        sys.executable, TRAIN_PY,
        "-s", args.source_path,
        "-m", model_path,
        "--repuls_margin", str(args.repuls_margin),
        "--lambda_repuls", str(args.lambda_repuls),
        "--repuls_k", str(args.repuls_k),
        "--repuls_from_iter", str(args.repuls_from_iter),
        "--repuls_warmup", str(args.repuls_warmup),
    ]
    # Evaluate on a held-out test split by default; --no_eval (or an explicit
    # --eval in the forwarded args) overrides this.
    if not args.no_eval and "--eval" not in extra:
        cmd.append("--eval")
    cmd += extra

    print(f"[run_experiment] output : {model_path}")
    print(f"[run_experiment] command: {' '.join(cmd)}")

    start = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    ret = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - start

    n_gaussians = count_gaussians(model_path)

    metrics = {
        "run_name": run_name,
        "params": {
            "repuls_margin": args.repuls_margin,
            "lambda_repuls": args.lambda_repuls,
            "repuls_k": args.repuls_k,
            "repuls_from_iter": args.repuls_from_iter,
            "repuls_warmup": args.repuls_warmup,
        },
        "source_path": os.path.abspath(args.source_path),
        "model_path": model_path,
        "extra_args": extra,
        "started_at": started_at,
        "return_code": ret.returncode,
        "training_time_sec": round(elapsed, 2),
        "training_time_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "final_gaussian_count": n_gaussians,
    }

    out_file = os.path.join(model_path, "experiment_metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[run_experiment] return code   : {ret.returncode}")
    print(f"[run_experiment] training time : {metrics['training_time_hms']} "
          f"({metrics['training_time_sec']}s)")
    print(f"[run_experiment] gaussians     : {n_gaussians}")
    print(f"[run_experiment] metrics saved : {out_file}")

    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
