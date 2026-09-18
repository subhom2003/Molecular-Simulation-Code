"""
ComputeStructureFactor -- static structure factor S(k).

Direct evaluation at the reciprocal lattice vectors of the cubic box:

    k = (2 pi / L) n,   n = (nx, ny, nz) integer
    S(k) = (1/N) <|sum_i exp(-i k . r_i)|^2>

No grid deposition, no aliasing, no self-term subtraction: the formula above
is exact (its k -> infinity limit is 1).  Vectors are grouped into |k| shells
(one shell per integer |n|, i.e. dk = 2 pi/L) and averaged within a shell,
then over accumulated configurations.
"""

from __future__ import annotations
import numpy as np

from .abc import Compute


class ComputeStructureFactor(Compute):
    def __init__(self, compute_id="sk", group="all", every: int = 100,
                 L: float = 1.0, N: int = 0, n_max: int | None = None,
                 k_max: float | None = None, n_bins: int | None = None,
                 start: int = 0):
        super().__init__(compute_id, group, every=every, start=start)
        self.L = float(L)
        self.N = int(N)
        # Accept (and mostly ignore) the historical kwargs n_grid/k_max/n_bins
        # for backward compatibility; the direct method only needs n_max,
        # the maximum integer magnitude of the reciprocal vectors sampled.
        if n_max is None:
            if k_max is not None:
                n_max = max(1, int(round(k_max * self.L / (2.0 * np.pi))))
            else:
                n_max = 15
        self.n_max = int(n_max)

        # enumerate integer triplets with 0 < |n| <= n_max (drop the DC term)
        ns = np.arange(-self.n_max, self.n_max + 1)
        nx, ny, nz = np.meshgrid(ns, ns, ns, indexing="ij")
        nvec = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
        mag2 = (nvec ** 2).sum(axis=1)
        keep = (mag2 > 0) & (mag2 <= self.n_max ** 2)
        self._nvec = nvec[keep].astype(float)
        self._kvecs = (2.0 * np.pi / self.L) * self._nvec          # (M, 3)
        self._kmag = np.sqrt((self._kvecs ** 2).sum(axis=1))       # (M,)

        # shell index per vector: integer |n|
        self._shell = np.sqrt(mag2[keep]).round().astype(int)
        self.shells = np.unique(self._shell)
        self.n_samples = 0
        self._shell_sum = np.zeros(len(self.shells), dtype=np.float64)
        self._shell_cnt = np.zeros(len(self.shells), dtype=np.int64)

    def name(self) -> str:
        return "structure_factor"

    def _shell_index(self) -> np.ndarray:
        return np.searchsorted(self.shells, self._shell)

    def compute(self, state, force_result) -> None:
        positions = state.atoms.positions
        N = positions.shape[0]
        # phases: (N, M)
        ph = positions @ self._kvecs.T
        rho_k = np.exp(-1j * ph).sum(axis=0)                    # (M,)
        s_k = (np.abs(rho_k) ** 2) / N                          # includes self term
        idx = self._shell_index()
        np.add.at(self._shell_sum, idx, s_k)
        np.add.at(self._shell_cnt, idx, 1)
        self.n_samples += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (k, S(k)) over shells; only shells that contain at least
        one reciprocal vector appear - there are no NaN rows."""
        # mean modulus of |k| per shell: average over the VECTORS in the shell
        # (the counts array accumulates per vector per sample and would
        # over-divide the geometric |k|).
        idx = self._shell_index()
        ksum = np.zeros(len(self.shells))
        np.add.at(ksum, idx, self._kmag)
        nvec = np.bincount(idx, minlength=len(self.shells))
        has = self._shell_cnt > 0
        k_shell = np.where(nvec > 0, ksum / np.maximum(nvec, 1), np.nan)
        S = np.where(has, self._shell_sum / np.maximum(self._shell_cnt, 1), np.nan)
        return k_shell[has], S[has]
