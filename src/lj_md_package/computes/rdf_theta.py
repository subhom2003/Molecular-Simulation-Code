"""
ComputeRDFTheta -- angular-radial pair distribution function g(r, cosθ).

Accumulates a 2D histogram of pair distances and polar angles (measured from
the z-axis) over multiple configurations, then normalizes to give the
3D angular-resolved pair correlation function.

Normalization (for unordered pairs via triu_indices):
    g(r_k, cosθ_l) = V × bins[k,l] / (N² × N_samples × π × r_k² × dr × dcosθ)

For an isotropic ideal gas: g(r, cosθ) → 1 (uniform in cosθ).
Relation to 1D g(r):
    g(r) = ½ ∫_{-1}^{1} g(r, cosθ) dcosθ
"""

from __future__ import annotations
import numpy as np

from .abc import Compute


class ComputeRDFTheta(Compute):
    def __init__(self, compute_id="rdf_theta", group="all", every: int = 50,
                 n_r_bins: int = 100, n_cos_bins: int = 50, L: float = 1.0,
                 N: int = 0, start: int = 0):
        super().__init__(compute_id, group, every=every, start=start)
        self.n_r_bins = n_r_bins
        self.n_cos_bins = n_cos_bins
        self.r_max = L / 2.0
        self.L = L
        self.N = N
        self.V = L ** 3
        self.dr = self.r_max / n_r_bins
        self.dcos = 2.0 / n_cos_bins
        self.r_edges = np.linspace(0.0, self.r_max, n_r_bins + 1)
        self.cos_edges = np.linspace(-1.0, 1.0, n_cos_bins + 1)
        self.r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        self.cos_centers = 0.5 * (self.cos_edges[:-1] + self.cos_edges[1:])
        self.bins = np.zeros((n_r_bins, n_cos_bins), dtype=np.float64)
        self.n_samples = 0

    def name(self) -> str:
        return "rdf_theta"

    def compute(self, state, force_result) -> None:
        self.n_samples += 1
        pos = state.atoms.positions
        L = state.box.L
        N = pos.shape[0]

        i, j = np.triu_indices(N, k=1)
        dr = pos[j] - pos[i]
        dr -= L * np.round(dr / L)

        r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
        cos_theta = dr[:, 2] / np.maximum(r, 1e-10)

        mask = r < self.r_max
        r_vals = r[mask]
        cos_vals = cos_theta[mask]

        if len(r_vals) == 0:
            return

        hist, _, _ = np.histogram2d(
            r_vals, cos_vals,
            bins=[self.r_edges, self.cos_edges],
        )
        self.bins += hist

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.n_samples == 0:
            return (self.r_centers, self.cos_centers,
                    np.full_like(self.r_centers, np.nan),
                    np.full((self.n_r_bins, self.n_cos_bins), np.nan))

        R = self.r_centers[:, None]
        prefactor = self.V / (self.N ** 2 * self.n_samples * np.pi * self.dr * self.dcos)
        r2 = R ** 2
        g = np.where(r2 > 1e-10, prefactor * self.bins / r2, 0.0)

        mask = np.isfinite(g).all(axis=1)
        g_isotropic = np.full(self.n_r_bins, np.nan)
        g_isotropic[mask] = g[mask].mean(axis=1)
        return self.r_centers, self.cos_centers, g_isotropic, g

    def angular_profile(self, r_lo: float, r_hi: float) -> tuple[np.ndarray, np.ndarray]:
        r, cos, _, g = self.finalize()
        mask = (r >= r_lo) & (r <= r_hi)
        if not mask.any():
            return self.cos_centers, np.full_like(self.cos_centers, np.nan)
        g_theta = g[mask, :].mean(axis=0)
        return self.cos_centers, g_theta
