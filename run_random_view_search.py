#!/usr/bin/env python3
#
# Random view-sequence search.
#
# Goal: find a training view order that trains well.
# For each experiment a random (duplicate-free) permutation of the training
# views is used as a fixed training order. We train, render the test set,
# compute PSNR/SSIM/LPIPS and a combined score, and keep a registry so that
# re-runs never repeat a previous sequence and always refresh the score list.
#
# Storage: output/random_view_sequence/<run>/
#          output/random_view_sequence/registry.json
#          output/random_view_sequence/summary.json / summary.csv
#

import os
import sys
import json
import glob
import argparse
import subprocess

BASE_DIR_DEFAULT = os.path.join("output", "random_view_sequence")


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_registry(base_dir):
    path = os.path.join(base_dir, "registry.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": {}}


def save_registry(base_dir, registry):
    with open(os.path.join(base_dir, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)


def read_metrics(model_dir, iters):
    """Read PSNR/SSIM/LPIPS from a finished run's results.json (or None)."""
    results_path = os.path.join(model_dir, "results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        results = json.load(f)
    method = f"ours_{iters}"
    entry = results.get(method) or (next(iter(results.values())) if results else None)
    if not entry:
        return None
    return {"PSNR": entry.get("PSNR"), "SSIM": entry.get("SSIM"), "LPIPS": entry.get("LPIPS")}


def read_sequence(model_dir):
    path = os.path.join(model_dir, "view_sequence.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    return data.get("seed"), data.get("sequence")


def scan_existing(base_dir, registry, iters):
    """Refresh the registry from whatever run folders exist on disk."""
    for model_dir in sorted(glob.glob(os.path.join(base_dir, "*"))):
        if not os.path.isdir(model_dir):
            continue
        name = os.path.basename(model_dir)
        seed, sequence = read_sequence(model_dir)
        metrics = read_metrics(model_dir, iters)
        run = registry["runs"].get(name, {})
        run["model_dir"] = model_dir
        if seed is not None:
            run["seed"] = seed
        if sequence is not None:
            run["sequence"] = sequence
        if metrics is not None:
            run.update(metrics)
        registry["runs"][name] = run


def used_seeds(registry):
    seeds = set()
    for run in registry["runs"].values():
        if run.get("seed") is not None:
            seeds.add(int(run["seed"]))
    return seeds


def used_sequences(registry):
    return {tuple(run["sequence"]) for run in registry["runs"].values() if run.get("sequence")}


def compute_scores(registry):
    """Min-max normalize each metric across runs and average into a combined
    score in [0, 1] (higher is better). LPIPS is inverted."""
    runs = [(n, r) for n, r in registry["runs"].items()
            if all(r.get(k) is not None for k in ("PSNR", "SSIM", "LPIPS"))]
    if not runs:
        return

    def norm(values, higher_better):
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [0.5 for _ in values]
        return [((v - lo) / (hi - lo)) if higher_better else (1.0 - (v - lo) / (hi - lo))
                for v in values]

    psnr_n = norm([r["PSNR"] for _, r in runs], True)
    ssim_n = norm([r["SSIM"] for _, r in runs], True)
    lpips_n = norm([r["LPIPS"] for _, r in runs], False)
    for (name, run), p, s, l in zip(runs, psnr_n, ssim_n, lpips_n):
        run["score"] = round((p + s + l) / 3.0, 6)


def write_summary(base_dir, registry, iters):
    rows = []
    for name, run in registry["runs"].items():
        rows.append({
            "run": name,
            "seed": run.get("seed"),
            "PSNR": run.get("PSNR"),
            "SSIM": run.get("SSIM"),
            "LPIPS": run.get("LPIPS"),
            "score": run.get("score"),
        })
    rows.sort(key=lambda r: (r["score"] is not None, r["score"] or 0), reverse=True)

    with open(os.path.join(base_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(base_dir, "summary.csv"), "w") as f:
        f.write("rank,run,seed,PSNR,SSIM,LPIPS,score\n")
        for i, r in enumerate(rows, 1):
            f.write("{},{},{},{},{},{},{}\n".format(
                i, r["run"], r["seed"],
                "" if r["PSNR"] is None else f"{r['PSNR']:.4f}",
                "" if r["SSIM"] is None else f"{r['SSIM']:.4f}",
                "" if r["LPIPS"] is None else f"{r['LPIPS']:.4f}",
                "" if r["score"] is None else f"{r['score']:.4f}"))

    print("\n===== Combined score ranking (iters={}) =====".format(iters))
    print("{:<4} {:<12} {:>6} {:>9} {:>8} {:>8} {:>8}".format(
        "rank", "run", "seed", "PSNR", "SSIM", "LPIPS", "score"))
    for i, r in enumerate(rows, 1):
        print("{:<4} {:<12} {:>6} {:>9} {:>8} {:>8} {:>8}".format(
            i, r["run"], "" if r["seed"] is None else r["seed"],
            "-" if r["PSNR"] is None else f"{r['PSNR']:.3f}",
            "-" if r["SSIM"] is None else f"{r['SSIM']:.4f}",
            "-" if r["LPIPS"] is None else f"{r['LPIPS']:.4f}",
            "-" if r["score"] is None else f"{r['score']:.4f}"))
    print("Summary written to {}".format(os.path.join(base_dir, "summary.csv")))


def run_pipeline(source_path, model_dir, seed, iters, extra_train_args):
    """Train (fixed view order) -> render test -> metrics. Resumable."""
    ply = os.path.join(model_dir, "point_cloud", f"iteration_{iters}", "point_cloud.ply")
    renders = os.path.join(model_dir, "test", f"ours_{iters}", "renders")
    results = os.path.join(model_dir, "results.json")

    if not os.path.exists(ply):
        run([sys.executable, "train.py",
             "-s", source_path, "-m", model_dir, "--eval",
             "--iterations", str(iters),
             "--save_iterations", str(iters),
             "--test_iterations", str(iters),
             "--view_sequence_seed", str(seed),
             "--disable_viewer", "--quiet"] + extra_train_args)

    if not (os.path.isdir(renders) and os.listdir(renders)):
        run([sys.executable, "render.py", "-m", model_dir, "--skip_train", "--quiet"])

    if not os.path.exists(results):
        run([sys.executable, "metrics.py", "-m", model_dir])


def main():
    parser = argparse.ArgumentParser(description="Random view-sequence search")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-n", "--num_sequences", type=int, default=10,
                        help="number of NEW unique view sequences to add this run")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--base_dir", default=BASE_DIR_DEFAULT)
    # Any extra args (e.g. -r 2, --images images_4) are forwarded to train.py
    args, extra_train_args = parser.parse_known_args()

    os.makedirs(args.base_dir, exist_ok=True)
    registry = load_registry(args.base_dir)
    registry["source_path"] = os.path.abspath(args.source_path)
    registry["iterations"] = args.iterations

    # 1) Refresh from anything already on disk (also reads previously-computed metrics)
    scan_existing(args.base_dir, registry, args.iterations)

    # 2) Pick fresh, unused seeds -> unique sequences
    seeds = used_seeds(registry)
    new_seeds = []
    candidate = 1
    while len(new_seeds) < args.num_sequences:
        if candidate not in seeds:
            new_seeds.append(candidate)
            seeds.add(candidate)
        candidate += 1

    # 3) Run each new sequence through the full pipeline
    existing_seqs = used_sequences(registry)
    for seed in new_seeds:
        name = f"seq_{seed:04d}"
        model_dir = os.path.join(args.base_dir, name)
        print(f"\n########## {name} (seed={seed}) ##########", flush=True)
        try:
            run_pipeline(args.source_path, model_dir, seed, args.iterations, extra_train_args)
        except subprocess.CalledProcessError as e:
            print(f"[warn] pipeline failed for {name}: {e}")
            continue

        seed_saved, sequence = read_sequence(model_dir)
        if sequence is not None and tuple(sequence) in existing_seqs:
            print(f"[warn] {name} produced a duplicate sequence; keeping but flagged.")
        if sequence is not None:
            existing_seqs.add(tuple(sequence))
        metrics = read_metrics(model_dir, args.iterations)
        run_entry = {"model_dir": model_dir, "seed": seed, "sequence": sequence}
        if metrics:
            run_entry.update(metrics)
        registry["runs"][name] = run_entry
        save_registry(args.base_dir, registry)

    # 4) Recompute combined scores over ALL runs and refresh the summary
    scan_existing(args.base_dir, registry, args.iterations)
    compute_scores(registry)
    save_registry(args.base_dir, registry)
    write_summary(args.base_dir, registry, args.iterations)


if __name__ == "__main__":
    main()
