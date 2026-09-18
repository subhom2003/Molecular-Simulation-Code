"""Pair-style library: pluggable short-range interaction kernels."""
from .abc import PairStyle, ForceResult
from .lj_cut import LJCut
from .soft import SoftPair

# Dispatch table for `pair_style <name> ...` commands.
PAIR_STYLES = {
    "lj/cut":     LJCut,
    "soft":       SoftPair,
}

__all__ = ["PairStyle", "ForceResult", "LJCut", "SoftPair", "PAIR_STYLES"]
