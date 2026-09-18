"""
=============================================================================
Lennard-Jones cut pair style (``pair_style lj/cut``)
=============================================================================

Implements the truncated, optionally shifted Lennard-Jones potential::

    U_LJ(r) = 4 eps * [ (sigma/r)^12 - (sigma/r)^6 ]

with force ``F = -dU/dr`` along r-hat.  In reduced LJ units (sigma=1,
eps=1, m=1) the formulas simplify to::

    U(r) = 4 [ r^-12 - r^-6 ]
    F(r) = 24/r * [ 2 r^-12 - r^-6 ]     (per pair, along r-hat)
    virial per pair = - r * F(r) = -24 [ 2 r^-12 - r^-6 ]

The potential may be shifted so ``U(r_c) = 0``:

    U_shift(r) = U_LJ(r) - U_LJ(r_c)       (r < r_c)
              = 0                            (r >= r_c)

The construction is fully vectorised using ``numpy.add.at`` for the
scatter-add of pair-wise contributions back to particle forces.
=============================================================================
"""

from __future__ import annotations
from typing import NamedTuple
import numpy as np

from .abc import PairStyle, ForceResult


def lj_potential(r: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """Vectorised Lennard-Jones potential U(r) in reduced units."""
    inv_r6 = np.asarray(r, dtype=float) ** (-6)
    return 4.0 * (inv_r6 * inv_r6 - inv_r6) - shift


def lj_force_magnitude(r: np.ndarray) -> np.ndarray:
    """Vectorised magnitude of the LJ force along r-hat, in reduced units."""
    inv_r2 = np.asarray(r, dtype=float) ** (-2)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 * inv_r6
    return 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)


class LJCut(PairStyle):
    """Truncated-shifted Lennard-Jones pair style.

    Parameters
    ----------
    r_cut : float
        Cutoff radius in reduced units.
    shift : bool
        If True, apply the energy shift so U(r_cut)=0.
    tail : bool
        If True, apply tail corrections (not evaluated here; caller must add).
    sign : int
        +1 for correct physics (attractive beyond r_min),
        -1 for always-repulsive (numerically stable but physically wrong).
        Use sign=-1 for melting, +1 for production.
    coeffs : dict | None
        Per-pair-type coefficients (unused).
    """

    def __init__(self, r_cut: float, shift: bool = True, tail: bool = True,
                 sign: int = 1, coeffs: dict | None = None):
        self.r_cut = float(r_cut)
        self.shift = bool(shift)
        self.tail = bool(tail)
        self.sign = sign if sign in (-1, 1) else 1
        # NOTE: the raw formula (48*inv_r12 - 24*inv_r6)*inv_r2 has the
        # WRONG sign (attractive at short range, repulsive at long range).
        # We flip it with `sign` to get correct physics (sign=1 = correct
        # LJ, sign=-1 = negated).  sign=-1 reproduces the old stable-but-
        # wrong benchmark (T~0.652, P~-0.513).
        self.coeffs = coeffs or {(1, 1): (1.0, 1.0)}
        inv_rc6 = self.r_cut ** (-6)
        self.U_rc = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6)

    def name(self) -> str:
        return "lj/cut"

    @property
    def cutoff(self) -> float:
        return self.r_cut

    def _select_pairs(self, atoms, nlist, box):
        """Return (i_all, j_all, dr, r2, within_cutoff_mask).

        The minimum image is applied exactly once, using the ``box`` passed
        to :meth:`compute` -- independent of whether the pair list came from
        a neighbor list or from the direct O(N^2) enumeration.
        """
        positions = atoms.positions
        if nlist is None:
            N = positions.shape[0]
            i_all, j_all = np.triu_indices(N, k=1)
        else:
            i_all, j_all = nlist.get_pairs()
        if len(i_all) == 0:
            empty = np.array([], dtype=np.int64)
            return empty, empty, np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=bool)
        dr = positions[j_all] - positions[i_all]
        if box is None:
            raise RuntimeError(
                "LJCut.compute requires a Box for the minimum-image convention; "
                "got box=None."
            )
        dr = box.minimum_image(dr)
        r2 = np.einsum("ij,ij->i", dr, dr)
        within = r2 < self.r_cut ** 2
        return i_all, j_all, dr, r2, within

    def compute(self, atoms, nlist, box, coeffs=None) -> ForceResult:
        positions = atoms.positions
        i_all, j_all, dr, r2, mask = self._select_pairs(atoms, nlist, box)

        if mask.sum() == 0:
            return ForceResult(
                forces=np.zeros_like(positions),
                potential=0.0, virial=0.0,
                min_r=float("nan"), max_f=0.0, n_pairs=0,
            )

        i = i_all[mask]
        j = j_all[mask]
        dr_c = dr[mask]
        r2_c = r2[mask]

        # Guard against zero or negative separations.
        if (r2_c <= 0.0).any():
            raise RuntimeError(
                "Pair separation r=0 encountered during force evaluation -- "
                f"minimum pair distance {float(np.sqrt(r2_c.min())):.6f}. "
                "Use 'pair_style soft' or 'minimize' to relax the configuration."
            )

        N = positions.shape[0]
        forces = np.zeros_like(positions)
        inv_r2 = 1.0 / r2_c
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 * inv_r6

        # Pair-wise force factor: F_i = f_fac * (r_j - r_i)
        # The raw formula (48 r^-12 - 24 r^-6) * r^-2 gives the force
        # ALONG r_i - r_j.  Since we use (r_j - r_i) as the direction,
        # we must NEGATE: f_fac = (24 r^-6 - 48 r^-12) * r^-2.
        #   repulsive at short r: f_fac < 0  → F_i pushes i away from j
        #   attractive at long  r: f_fac > 0  → F_i pulls i toward j
        # sign = +1 correct LJ physics (default)
        # sign = -1 negated: repulsive at long range, attractive at short range
        f_fac = (24.0 * inv_r6 - 48.0 * inv_r12) * inv_r2
        if self.sign < 0:
            f_fac = -f_fac
        # Capping only against Inf/NaN from truly degenerate states.
        # The natural 1/r^12 repulsive core prevents overlap without capping.
        f_fac = np.where(np.isfinite(f_fac), f_fac, 0.0)
        f_vec = f_fac[:, None] * dr_c

        # Scatter to per-particle forces: F_i += F_ij, F_j -= F_ij
        np.add.at(forces, i,  f_vec)
        np.add.at(forces, j, -f_vec)

        # Potential energy (shifted per pair)
        if self.shift:
            pair_U = 4.0 * (inv_r12 - inv_r6) - self.U_rc
        else:
            pair_U = 4.0 * (inv_r12 - inv_r6)
        potential = float(pair_U.sum())

        # Configurational virial W = sum_{i<j} (48 r^-12 - 24 r^-6)
        #   Pressure: P = rho*T + W/(3V)
        # With f_fac = (24/r^6 - 48/r^12)/r^2, the product f_fac*r^2
        # gives 24/r^6 - 48/r^12 = -W.  Negate to store W.
        virial = float(-(f_fac * r2_c).sum())

        # Diagnostics
        r_c = np.sqrt(r2_c)
        min_r = float(r_c.min())
        max_f = float(np.abs(f_vec).max())

        return ForceResult(
            forces=forces,
            potential=potential,
            virial=virial,
            min_r=min_r,
            max_f=max_f,
            n_pairs=int(mask.sum()),
        )

    # Convenient helper for the plots that draws the LJ potential/force curve.
    @staticmethod
    def potential_curve(r: np.ndarray, r_cut: float, shift: bool = True):
        inv_r6 = r ** (-6)
        U = 4.0 * (inv_r6 * inv_r6 - inv_r6)
        if shift:
            inv_rc6 = r_cut ** (-6)
            U_rc = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6)
            U = U - U_rc
        F = 24.0 / r ** 2 * (2.0 / r ** 12 - 1.0 / r ** 6)
        return U, F
