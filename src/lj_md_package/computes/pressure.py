"""ComputePressure -- instantaneous pressure from the virial route::

    P = (N kB T + W) / V
    W = -sum_ij r_ij . F_ij   (canned virial from force kernel)
    Plus long-range tail corrections::

    P_tail = (16 pi / 3) rho*^2 [(2/3) r_c^-9 - r_c^-3]
"""
from __future__ import annotations
from .abc import Compute


class ComputePressure(Compute):
    def __init__(self, compute_id="press", group="all",
                 rho_star: float = 0.8442, P_tail: float = 0.0):
        super().__init__(compute_id, group)
        self.rho_star = rho_star
        self.P_tail = P_tail
        self.P = 0.0
        self.virial = 0.0

    def name(self) -> str:
        return "pressure"

    def compute(self, state, force_result) -> None:
        N = state.atoms.N
        T = state.temperature
        V = self.rho_star and (N / self.rho_star)
        W = force_result.virial if force_result is not None else 0.0
        self.virial = W
        ideal = N * T / V
        self.P = ideal + W / (3.0 * V) + self.P_tail
