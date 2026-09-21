"""Train gs-exp against the current heightfield cubemap.

Uses train_cubemap_bg's existing per-pixel-background rasterizer and SIBR
network server, replacing only the cubemap sampler.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import exp_utils.cubemap_bg as cubemap_bg
from exp_utils.heightfield_bg import CubemapBackground

DEFAULT_CUBEMAP = "/home/sj/work/gs-dataset/tandt/train/cubemaps/ade20k_tree_exclude_2k_patch_consensus"
DEFAULT_IMAGES = "compensation/images/images"
cubemap_bg.CubemapBackground = CubemapBackground
if "--cubemap" not in sys.argv:
    sys.argv.extend(["--cubemap", DEFAULT_CUBEMAP])
if "--images" not in sys.argv and "-i" not in sys.argv:
    sys.argv.extend(["--images", DEFAULT_IMAGES])
runpy.run_path(str(Path(__file__).with_name("train_cubemap_bg.py")), run_name="__main__")
