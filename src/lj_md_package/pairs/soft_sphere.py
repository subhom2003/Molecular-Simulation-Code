"""
Soft-sphere repulsive pair style: U(r) = A * (1 - r/r_c)^2  for r < r_c.
Maximally bounded: U(0) = A, F(0) = 2A/r_c.
"""
from __future__ import annotations
import numpy as np
from .abc import PairStyle, ForceResult


class SoftSpherePair(PairStyle):
    def __init__(self, r_cut: float = 2.5, A: float = 100.0):
        self.r_cut = float(r_cut)
        self.A = float(A)

    def name(self) -> str:
        return "soft/sphere"

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
        # U = A * (1 - r/r_c)^2
        # Repulsive magnitude  f = -dU/dr = 2*A/r_c * (1 - r/r_c) > 0.
        # Force on i points away from j:  F_i = -f * (r_j - r_i)/r.
        f_mag = 2.0 * self.A / self.r_cut * (1.0 - r_c / self.r_cut)
        f_vec = -f_mag[:, None] * dr_c / r_c[:, None]
        np.add.at(forces, i,  f_vec)
        np.add.at(forces, j, -f_vec)
        U_pair = self.A * (1.0 - r_c / self.r_cut) ** 2
        return ForceResult(
            forces=forces,
            potential=float(U_pair.sum()),
            virial=float((f_mag * r_c).sum()),
            min_r=float(r_c.min()),
            max_f=float(np.abs(f_vec).max()),
            n_pairs=int(mask.sum()),
        )
