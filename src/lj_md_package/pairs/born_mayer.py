"""
Born-Mayer repulsive pair style: U(r) = A * exp(-r/lambda) for r < r_c.
Bounded at r=0: U(0)=A, F(0)=A/lambda (finite).
Smooth and continuous everywhere.
"""
from __future__ import annotations
import numpy as np
from .abc import PairStyle, ForceResult


class BornMayerPair(PairStyle):
    """U(r) = A exp(-r/lam) - A exp(-rc/lam)  for r < rc (shifted to 0 at rc)."""

    def __init__(self, r_cut: float = 2.5, A: float = 100.0, lam: float = 0.3):
        self.r_cut = float(r_cut)
        self.A = float(A)
        self.lam = float(lam)
        # Energy shift so U(r_c) = 0 (potential continuous; force jumps are
        # small because exp(-rc/lam) is small for sensible parameters).
        self._U_rc = self.A * float(np.exp(-self.r_cut / self.lam))

    def name(self) -> str:
        return "born/mayer"

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
        r2_c = r2[mask]
        r_c = np.sqrt(r2_c)
        exp_arg = -r_c / self.lam
        ef = np.exp(exp_arg)
        # Repulsive magnitude f = -dU/dr = (A/lam) exp(-r/lam) > 0;
        # F_i points away from j:  F_i = -f * (r_j - r_i)/r.
        f_mag = self.A / self.lam * ef
        f_vec = -f_mag[:, None] * dr_c / r_c[:, None]
        np.add.at(forces, i,  f_vec)
        np.add.at(forces, j, -f_vec)
        U_pair = self.A * ef - self._U_rc
        return ForceResult(
            forces=forces,
            potential=float(U_pair.sum()),
            virial=float((f_mag * r_c).sum()),
            min_r=float(r_c.min()),
            max_f=float(np.abs(f_vec).max()),
            n_pairs=int(mask.sum()),
        )
