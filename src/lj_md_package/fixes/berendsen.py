"""FixBerendsen -- Berendsen (1984) weak-coupling thermostat."""
from __future__ import annotations
import numpy as np
import logging

from .abc import Fix

logger = logging.getLogger("lj_md.fix")


class FixBerendsen(Fix):
    """``fix ID group-ID berendsen N Tstart Tstop tau_T``.

    Apply a Berendsen velocity rescaling every step::

        lambda = sqrt(1 + (dt/tau_T) [(T_target/T_current) - 1])

    so that the instantaneous T relaxes exponentially to T_target with
    time constant tau_T.  Not strictly canonical (suppresses temperature
    fluctuations) but smooth and robust, ideal for equilibration.
    """

    def __init__(self, fix_id: str = "berendsen", group: str = "all",
                 N: int = 1, Tstart: float = 1.0, Tstop: float | None = None,
                 tau_T: float = 0.5):
        super().__init__(fix_id, group, every=int(N))
        self.Tstart = float(Tstart)
        self.Tstop = float(Tstop) if Tstop is not None else float(Tstart)
        self.tau_T = float(tau_T)
        self._run_start = 0
        self._n_total_steps = 0

    def name(self) -> str:
        return "berendsen"

    def on_run_start(self, start_step: int, n_steps: int) -> None:
        self._run_start = int(start_step)
        self._n_total_steps = int(n_steps)

    def set_total_steps(self, n: int) -> None:
        self._n_total_steps = int(n)

    def _target_T(self, step: int) -> float:
        if self._n_total_steps <= 1:
            return self.Tstart
        frac = (step - self._run_start) / self._n_total_steps
        frac = min(max(frac, 0.0), 1.0)
        return self.Tstart + frac * (self.Tstop - self.Tstart)

    def do_end_of_step(self, state) -> None:
        atoms = state.atoms
        T_target = self._target_T(state.step)
        T_current = atoms.temperature()
        if T_current <= 0:
            return
        ratio = T_target / T_current
        lam_sq = 1.0 + (state.dt / self.tau_T) * (ratio - 1.0)
        if lam_sq < 0.0:
            logger.warning("Berendsen lambda^2 < 0; clamping to 0; system very cold")
            lam_sq = 0.0
        lam = np.sqrt(lam_sq)
        atoms.velocities *= lam
