"""
ComputeVACF -- velocity autocorrelation function tracker.

C_v(t) = <v(t0) . v(t0 + t)> / <v(t0) . v(t0)>

Multiple origins are stored, same scheme as :class:`ComputeMSD`.
"""

from __future__ import annotations
from collections import deque
import numpy as np

from .abc import Compute


class ComputeVACF(Compute):
    def __init__(self, compute_id="vacf", group="all", every: int = 50,
                 max_length: int = 500, dt: float = 0.005,
                 normalize: bool = True, start: int = 0):
        super().__init__(compute_id, group, every=every, start=start)
        self.max_length = max_length
        self.dt = dt
        self.normalize = normalize
        self._origins = deque()
        self._sum = np.zeros(max_length + 1, dtype=np.float64)
        self._count = np.zeros(max_length + 1, dtype=np.int64)
        self._v0sq_sum = 0.0
        self._v0sq_count = 0

    def name(self) -> str:
        return "vacf"

    def compute(self, state, force_result) -> None:
        v = state.atoms.velocities
        # add origin
        self._origins.append((state.step, v.copy()))
        # discard origins past max_length
        while self._origins and (state.step - self._origins[0][0]) > self.max_length:
            self._origins.popleft()
        if not self._origins:
            return
        # v_stack shape (n_orig, N, 3)
        s0_stack = np.array([o[0] for o in self._origins], dtype=np.int64)
        v0_stack = np.stack([o[1] for o in self._origins], axis=0)
        lags = state.step - s0_stack
        # v(t0) . v(t0+t) -- mean over atoms and vector components
        corr = (v0_stack * v[None, :, :]).sum(axis=(1, 2)) / state.atoms.N
        for lag, c in zip(lags, corr):
            self._sum[lag] += c
            self._count[lag] += 1
        # accumulate <v0 . v0> for the normalization
        v0sq = (v0_stack ** 2).sum(axis=(1, 2)) / state.atoms.N
        for c in v0sq:
            self._v0sq_sum += c
            self._v0sq_count += 1

    def get_vacf(self, normalize: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, C(t)).  Normalized by default."""
        lags = np.where(self._count > 0)[0]
        times = lags * self.dt
        vacf = self._sum[lags] / np.maximum(self._count[lags], 1)
        do_norm = self.normalize if normalize is None else normalize
        if do_norm and self._v0sq_count > 0:
            vacf = vacf / (self._v0sq_sum / self._v0sq_count)
        return times, vacf

    def green_kubo_diffusion(self) -> float:
        """Green-Kubo diffusion: D = (1/3) int_0^inf C_v(t) dt.

        C_v(t) = <v_i(t) . v_i(0)> is the UNNORMALIZED velocity
        autocorrelation (integrated here).  Since C_v(0) = 3 T in reduced
        units, this is D = T * int C_norm(t) dt.
        """
        times, vacf = self.get_vacf(normalize=False)
        if times.size < 4:
            return float("nan")
        integral = np.trapezoid(vacf, times)
        return float(integral / 3.0)
