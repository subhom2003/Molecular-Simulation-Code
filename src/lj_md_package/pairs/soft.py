"""
Soft pair style.
=============================================================================

A purely repulsive soft potential::

    U_soft(r) = A * [1 + cos(pi * r / r_c)] / 2    for r < r_c
            = 0                                    for r >= r_c

Used as a startup pair style for "minimization" runs to relax overlapping
configurations before switching to a hard potential like LJ.  Mirrors
LAMMPS ``pair_style soft``.

    pair_style soft A
    pair_coeff 1 1 <A_value>
=============================================================================
"""

from __future__ import annotations
import numpy as np
from .abc import PairStyle, ForceResult


class SoftPair(PairStyle):
    def __init__(self, r_cut: float = 1.0, A: float = 1.0):
        self.r_cut = float(r_cut)
        self.A = float(A)

    def name(self) -> str:
        return "soft"

    @property
    def cutoff(self) -> float:
        return self.r_cut

    def compute(self, atoms, nlist, box, coeffs=None) -> ForceResult:
        positions = atoms.positions
        if nlist is None:
            N = positions.shape[0]
            i_all, j_all = np.triu_indices(N, k=1)
        else:
            i_all, j_all = nlist.get_pairs()
        forces = np.zeros_like(positions)
        if len(i_all) == 0:
            return ForceResult(forces, 0.0, 0.0, float("nan"), 0.0, 0)
        dr = positions[j_all] - positions[i_all]
        if box is not None:
            dr = box.minimum_image(dr)
        r2 = np.einsum("ij,ij->i", dr, dr)
        mask = r2 < self.r_cut ** 2
        if mask.sum() == 0:
            return ForceResult(forces, 0.0, 0.0, float("nan"), 0.0, 0)
        i = i_all[mask]; j = j_all[mask]
        dr_c = dr[mask]
        r_c = np.sqrt(r2[mask])
        # U(r) = A/2 (1 + cos(pi r / rc));  dU/dr = -(A pi / 2 rc) sin(pi r / rc) < 0
        U_pair = 0.5 * self.A * (1.0 + np.cos(np.pi * r_c / self.r_cut))
        dU_dr = -0.5 * self.A * np.sin(np.pi * r_c / self.r_cut) * (np.pi / self.r_cut)
        # Force on i: F_i = -dU/dr (r_i - r_j)/r = dU_dr * (r_j - r_i)/r.
        # dU_dr < 0, so F_i points away from j: repulsive, as intended.
        f_vec = dU_dr[:, None] * dr_c / r_c[:, None]
        np.add.at(forces, i,  f_vec)
        np.add.at(forces, j, -f_vec)
        return ForceResult(
            forces=forces,
            potential=float(U_pair.sum()),
            virial=-float((dU_dr * r_c).sum()),
            min_r=float(r_c.min()),
            max_f=float(np.abs(f_vec).max()),
            n_pairs=int(mask.sum()),
        )
