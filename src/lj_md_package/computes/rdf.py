"""
ComputeRDF -- radial distribution function accumulator.

g(r) = N(r) / (rho * V * 4 pi r^2 dr * 1/2)   <-- per-particle shell count

We accumulate histogram over many configurations, then normalize at retrieval
time. Multiple time origins are taken at every ``every`` step (the parent
:class:`Compute` schedules them).
"""

from __future__ import annotations
import numpy as np

from .abc import Compute


class ComputeRDF(Compute):
    def __init__(self, compute_id="rdf", group="all", every: int = 50,
                 n_bins: int = 200, L: float = 1.0, N: int = 0, start: int = 0):
        super().__init__(compute_id, group, every=every, start=start)
        self.n_bins = n_bins
        self.r_max = L / 2.0
        self.L = L
        self.N = N
        self.V = L ** 3
        self.dr = self.r_max / n_bins
        self.bins = np.zeros(n_bins, dtype=np.float64)
        self.r = np.linspace(self.dr / 2.0,
                             self.r_max - self.dr / 2.0, n_bins)
        self.n_samples = 0

    def name(self) -> str:
        return "rdf"

    def accumulate(self, positions, L) -> None:
        self.n_samples += 1
        N = positions.shape[0]
        i, j = np.triu_indices(N, k=1)
        dr = positions[j] - positions[i]
        dr -= L * np.round(dr / L)
        r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
        r = r[r < self.r_max]
        hist, _ = np.histogram(r, bins=self.n_bins, range=(0.0, self.r_max))
        self.bins += hist

    def compute(self, state, force_result) -> None:
        self.accumulate(state.atoms.positions, state.box.L)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_samples == 0:
            return self.r, np.full_like(self.r, np.nan)
        shell_vol = (4.0 / 3.0) * np.pi * (
            (self.r + self.dr / 2.0) ** 3 - (self.r - self.dr / 2.0) ** 3
        )
        g = 2.0 * self.V * self.bins / (self.N ** 2 * self.n_samples * shell_vol)
        return self.r, g

    def coordination_number(self, r_max: float = 1.5) -> float:
        """Coordination number NC(r_max) = 4 pi rho integral_0^{r_max} r^2 g(r) dr."""
        r, g = self.finalize()
        return 4.0 * np.pi * self.N / self.V * np.trapezoid(g * r ** 2, r)
