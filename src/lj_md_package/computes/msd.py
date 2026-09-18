"""
ComputeMSD -- mean-square displacement tracker with many time origins.

For each origin added (every ``interval`` steps) we store (step0, r0)
and on every subsequent step compute::

    dr = unwrapped_now - r0
    MSD(t-t0) = mean_i |dr_i|^2

The diffusion coefficient D is recovered from the Einstein relation at
long times::

    D = MSD / (6 dt)
"""

from __future__ import annotations
from collections import deque
import numpy as np

from .abc import Compute


class ComputeMSD(Compute):
    def __init__(self, compute_id="msd", group="all", every: int = 100,
                 max_length: int = 2000, dt: float = 0.005, start: int = 0):
        super().__init__(compute_id, group, every=every, start=start)
        self.max_length = max_length
        self.dt = dt
        # origins: (step0, unwrapped0)
        self._origins = deque()
        # msd[lag] = sum of squared displacements
        self._sum = np.zeros(max_length + 1, dtype=np.float64)
        self._count = np.zeros(max_length + 1, dtype=np.int64)

    def name(self) -> str:
        return "msd"

    def compute(self, state, force_result) -> None:
        unwrapped = state.unwrapped
        # add new origin
        self._origins.append((state.step, unwrapped.copy()))
        n_origins = len(self._origins)
        # vectorize over all active origins at this snapshot
        # stack origins as numpy array: shape (n_orig, N, 3)
        r0_stack = np.stack([o[1] for o in self._origins], axis=0)
        s0_stack = np.array([o[0] for o in self._origins], dtype=np.int64)
        lags = state.step - s0_stack             # (n_orig,)
        # drop origins past max_length
        keep = lags <= self.max_length
        if not keep.any():
            self._origins.clear()
            return
        if not keep.all():
            # pop origins whose lag > max_length
            # origins were appended FIFO, the discard must be FIFO too
            while self._origins and (state.step - self._origins[0][0]) > self.max_length:
                self._origins.popleft()
            r0_stack = np.stack([o[1] for o in self._origins], axis=0)
            s0_stack = np.array([o[0] for o in self._origins], dtype=np.int64)
            lags = state.step - s0_stack
        # vectorized MSD computation over origins
        disp = unwrapped[None, :, :] - r0_stack  # (n_orig, N, 3)
        sq = (disp ** 2).sum(axis=2)             # (n_orig, N)
        sq_mean_per_origin = sq.mean(axis=1)     # (n_orig,)
        for lag, val in zip(lags, sq_mean_per_origin):
            self._sum[lag] += val
            self._count[lag] += 1

    def get_msd(self) -> tuple[np.ndarray, np.ndarray]:
        lags = np.where(self._count > 0)[0]
        times = lags * self.dt
        msd = self._sum[lags] / np.maximum(self._count[lags], 1)
        return times, msd

    def diffusion_coefficient(self, t_frac_start: float = 0.3) -> float:
        times, msd = self.get_msd()
        if times.size < 4:
            return float("nan")
        start = int(t_frac_start * times.size)
        # D = d(MSD)/dt / 6
        slope = np.polyfit(times[start:], msd[start:], 1)[0]
        return float(slope / 6.0)
