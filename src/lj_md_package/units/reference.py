"""
=============================================================================
Literature reference values for the Lennard-Jones benchmark state point
=============================================================================

Single source of truth for the literature comparison printed in the summary
report and overlaid on the validation plots.  Values are for the LJTS fluid
(truncated+shifted at r_c=2.5 sigma) at the Verlet (1967) state point:

    rho* = 0.8442,  T* = 0.722,  N = 108

References:
    - Lustig et al., "Equation of state for the Lennard-Jones truncated
      and shifted fluid", J. Chem. Phys. 154, 034109 (2021) -- LJTS EOS
    - Johnson et al., "The Lennard-Jones equation of state revisited",
      Mol. Phys. 78, 591 (1993) -- full LJ EOS (for comparison)
    - Rahman, "Correlations in the motion of atoms in liquid argon",
      Phys. Rev. 136, A405 (1964) -- diffusion coefficient

Note: The full LJ values (P*=1.731, E/N=-6.236) are for the infinite-cutoff
system and are NOT the correct reference for this package which uses
r_c=2.5 sigma with tail corrections.

Plotting / summary modules should *never* duplicate these literals: import
:const:`LITERATURE_LJ` and seek by the same key.
=============================================================================
"""

from __future__ import annotations

# LJTS (r_c=2.5, truncated+shifted, NO tail correction) reference values at
# rho*=0.8442, T*=0.722.  This is the correct comparison when the simulation
# runs WITHOUT tail corrections.
LITERATURE_LJ: dict[str, tuple[float, str]] = {
    "T_star":   (0.722,   "Verlet 1967 (input target)"),
    "P_star":   (1.138,   "Lustig et al. 2021 LJTS EOS (r_c=2.5, no tail)"),
    "E_per_N":  (-4.846,  "Lustig et al. 2021 LJTS EOS (r_c=2.5, no tail)"),
    "D_star":   (0.018,   "Rahman 1964 (full LJ, approximate)"),
    "U_min":    (-1.0,    "LJ minimum at 2^(1/6) sigma"),
}

# Full-LJ reference values at the same state point - the correct comparison
# when the simulation runs WITH tail corrections (the tail makes the
# truncated potential an estimator of the full LJ fluid).
LITERATURE_FULL_LJ: dict[str, tuple[float, str]] = {
    "T_star":   (0.722,   "Verlet 1967 (input target)"),
    "P_star":   (1.731,   "Allen & Tildesley 2017 (full LJ, tail-corrected)"),
    "E_per_N":  (-6.236,  "Allen & Tildesley 2017 (full LJ, tail-corrected)"),
    "D_star":   (0.018,   "Rahman 1964 (full LJ, approximate)"),
    "U_min":    (-1.0,    "LJ minimum at 2^(1/6) sigma"),
}


def reference_for(use_tail_corrections: bool) -> dict[str, tuple[float, str]]:
    """Pick the literature set matching the tail-correction setting.

    Tail-corrected truncated simulations estimate the FULL LJ fluid;
    without tail corrections the simulated system IS the LJTS fluid.
    """
    return LITERATURE_FULL_LJ if use_tail_corrections else LITERATURE_LJ


def near_solid_liquid_coexistence(rho_star: float, T_star: float) -> bool:
    """Heuristic flag: near the LJTS solid-liquid boundary a small (N~108)
    box falls into the coexistence band and finite-N averages will NOT match
    the homogeneous-liquid EOS references (see REVIEW.md, section P2)."""
    return rho_star > 0.80 and T_star < 0.78


# ---------------------------------------------------------------------------
# LJ phase-boundary reference points for the phase-diagram plot.  These are
# approximations of the LJTS triple and critical points.
# ---------------------------------------------------------------------------
PHASE_POINTS: dict[str, tuple[float, float]] = {
    "triple":     (0.610, 0.850),   # LJTS r_c=2.5 triple point
    "critical":    (1.086, 0.319),  # LJTS r_c=2.5 critical point
}

# Liquid-vapour coexistence for LJTS (rho_l, rho_v, T*) along the saturation curve
# Approximate values from Vrabec et al. 2006 / Lustig et al. 2021
COEXISTENCE_CURVE_T: tuple[float, ...] = (
    0.62, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05
)
COEXISTENCE_RHO_L: tuple[float, ...] = (
    0.85, 0.83, 0.81, 0.79, 0.77, 0.74, 0.71, 0.67, 0.62, 0.55
)
COEXISTENCE_RHO_V: tuple[float, ...] = (
    0.001, 0.002, 0.004, 0.007, 0.011, 0.018, 0.028, 0.045, 0.075, 0.12
)