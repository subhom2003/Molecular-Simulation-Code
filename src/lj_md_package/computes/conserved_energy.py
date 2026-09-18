"""ComputeConservedEnergy -- track H_NH(t)  for NVT runs."""
from __future__ import annotations
import numpy as np

from .abc import Compute


class ComputeConservedEnergy(Compute):
    """Track H_NH(t) = H_system + sum over fixes (conserved_energy)."""

    def __init__(self, compute_id="conserved_energy", group="all", every: int = 10):
        super().__init__(compute_id, group, every=every)
        self.steps = []
        self.times = []
        self.values = []

    def name(self) -> str:
        return "conserved/energy"

    def compute(self, state, force_result) -> None:
        # H_system = KE + PE (without tail corrections)
        KE = state.kinetic_energy
        PE = force_result.potential if force_result is not None else 0.0
        # add conserved contribution from any Nose-Hoover-style fix
        H = KE + PE
        # state.extra_conserved is the sum of fix.conserved_energy() the simulation has cached
        H += getattr(state, "_fix_conserved", 0.0)
        self.steps.append(state.step)
        self.times.append(state.time)
        self.values.append(float(H))

    def get_history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.steps, dtype=np.int64),
            np.asarray(self.times, dtype=float),
            np.asarray(self.values, dtype=float),
        )

    def fractional_drift(self) -> float:
        if len(self.values) < 2:
            return float("nan")
        H0 = self.values[0]
        if abs(H0) < 1e-12:
            return float("nan")
        return abs(self.values[-1] - H0) / abs(H0)
