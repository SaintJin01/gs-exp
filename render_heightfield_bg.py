"""Render gs-exp checkpoint against the current heightfield cubemap."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import exp_utils.cubemap_bg as cubemap_bg
from exp_utils.heightfield_bg import CubemapBackground

DEFAULT_CUBEMAP = "/home/sj/work/gs-dataset/tandt/train/cubemaps/ablation_01_sky_border_4096_gt25m"
os.environ.setdefault("HEIGHTFIELD_RENDER_SUPERSAMPLE", "2")
cubemap_bg.CubemapBackground = CubemapBackground
if "--cubemap" not in sys.argv:
    sys.argv.extend(["--cubemap", DEFAULT_CUBEMAP])
runpy.run_path(str(Path(__file__).with_name("render_cubemap_bg.py")), run_name="__main__")
