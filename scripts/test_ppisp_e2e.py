"""Tiny synthetic-scene smoke test of the real train/render/resume entry points."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="3dgs-ppisp-e2e-") as tmp:
        tmp = Path(tmp)
        data = tmp / "data"
        data.mkdir()
        for split, count in [("train", 3), ("test", 1)]:
            (data / split).mkdir()
            frames = []
            for i in range(count):
                transform = np.eye(4)
                transform[0, 3] = (i - 1) * 0.2
                frames.append({"file_path": f"{split}/r_{i}", "transform_matrix": transform.tolist()})
                pixels = np.full((32, 32, 3), 80 + 30 * i, dtype=np.uint8)
                Image.fromarray(pixels).save(data / split / f"r_{i}.png")
            (data / f"transforms_{split}.json").write_text(json.dumps(
                {"camera_angle_x": 0.7, "frames": frames}))
        points = np.zeros(64, dtype=[(x, "f4") for x in ["x", "y", "z", "nx", "ny", "nz"]]
                          + [(x, "u1") for x in ["red", "green", "blue"]])
        rng = np.random.default_rng(42)
        for field in ["x", "y"]:
            points[field] = rng.uniform(-0.4, 0.4, 64)
        points["z"] = rng.uniform(-2.5, -1.5, 64)
        for field in ["red", "green", "blue"]:
            points[field] = 120
        PlyData([PlyElement.describe(points, "vertex")]).write(data / "points3d.ply")

        def run(*args):
            subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)

        model = tmp / "model"
        common = ["-s", data, "--eval", "--ppisp", "--iterations", "10", "--disable_viewer",
                  "--test_iterations", "5", "10", "--save_iterations", "8", "10",
                  "--checkpoint_iterations", "5", "9", "--densify_until_iter", "7",
                  "--densify_from_iter", "1", "--densification_interval", "2",
                  "--opacity_reset_interval", "100", "--densify_grad_threshold", "100"]
        run("train.py", "-m", model, *common)
        for iteration in [8, 10]:
            run("render.py", "-m", model, "--iteration", iteration)
            assert len(list((model / "test" / f"ours_{iteration}" / "renders").glob("*.png"))) == 1
        for checkpoint in [5, 9]:
            resumed = tmp / f"resume_{checkpoint}"
            run("train.py", "-m", resumed, *common, "--start_checkpoint", model / f"chkpnt{checkpoint}.pth")
            run("render.py", "-m", resumed, "--iteration", "10", "--skip_train")
        baseline = tmp / "baseline"
        run("train.py", "-s", data, "-m", baseline, "--iterations", "2", "--disable_viewer")
        run("render.py", "-m", baseline, "--iteration", "2")
        print("PASS: real 3DGS train/render, early/final ISP snapshots, resume before/after freeze, baseline")


if __name__ == "__main__":
    main()
