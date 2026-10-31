"""Positive unbounded radiance."""
import torch


def sh_to_radiance(sh):
    # Match PPISP's official 3DGRUT SH path: lower bound only, no HDR ceiling.
    return (sh + 0.5).clamp_min(0.0)
