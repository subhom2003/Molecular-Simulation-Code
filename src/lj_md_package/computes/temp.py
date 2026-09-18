"""ComputeTemp -- instantaneous and running mean of temperature."""
from __future__ import annotations
from .abc import Compute


class ComputeTemp(Compute):
    def __init__(self, compute_id="temp", group="all"):
        super().__init__(compute_id, group)
        self.T = 0.0
        self.T_mean = 0.0
        self.n_samples = 0

    def name(self) -> str:
        return "temp"

    def compute(self, state, force_result) -> None:
        self.T = state.temperature
        self.n_samples += 1
        # incremental mean (avoids storing the whole history)
        self.T_mean += (self.T - self.T_mean) / self.n_samples
