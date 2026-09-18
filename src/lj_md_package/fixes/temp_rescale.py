"""FixTempRescale -- Woodcock (1971) instantaneous velocity rescaling."""
from __future__ import annotations
import numpy as np

from .abc import Fix


class FixTempRescale(Fix):
    """``fix ID group-ID temp/rescale N Tstart Tstop window fraction``
    (LAMMPS argument order).

    Every ``N`` steps, if the instantaneous temperature lies outside the
    window ``T_target +/- window`` (window in *absolute* reduced-temperature
    units), velocities are scaled towards the target: with ``fraction=1``
    the full rescale ``lam = sqrt(T_target/T)`` is applied; with
    ``fraction<1`` only ``lam = 1 + fraction*(lam_full - 1)``.

    If ``Tstart != Tstop``, the target temperature ramps linearly across the
    *current* ``run`` block (the fix learns the block's start/length through
    :meth:`on_run_start`, invoked by :meth:`Simulation.run`).
    """

    def __init__(self, fix_id: str = "temp_rescale", group: str = "all",
                 N: int = 100, Tstart: float = 1.0, Tstop: float | None = None,
                 window: float = 0.0, fraction: float = 1.0):
        super().__init__(fix_id, group, every=int(N))
        self.Tstart = float(Tstart)
        self.Tstop = float(Tstop) if Tstop is not None else float(Tstart)
        self.window = float(window)      # absolute temperature window
        self.fraction = float(fraction)
        self._run_start = 0
        self._n_total_steps = 0  # set by the run loop (on_run_start)

    def name(self) -> str:
        return "temp/rescale"

    # ------------------------------------------------------------- run hook
    def on_run_start(self, start_step: int, n_steps: int) -> None:
        self._run_start = int(start_step)
        self._n_total_steps = int(n_steps)

    # backward-compatible alias used by older code paths
    def set_total_steps(self, n: int) -> None:
        self._n_total_steps = int(n)

    def _target_T(self, step: int) -> float:
        """Linear ramp from Tstart to Tstop across the current run block."""
        if self._n_total_steps <= 1:
            return self.Tstart
        frac = (step - self._run_start) / self._n_total_steps
        frac = min(max(frac, 0.0), 1.0)
        return self.Tstart + frac * (self.Tstop - self.Tstart)

    def do_end_of_step(self, state) -> None:
        atoms = state.atoms
        T_target = self._target_T(state.step)
        T_current = atoms.temperature()
        # absolute temperature window (LAMMPS semantics)
        if abs(T_current - T_target) <= self.window:
            return
        lam = np.sqrt(T_target / T_current)
        if self.fraction < 1.0:
            lam = 1.0 + self.fraction * (lam - 1.0)
        atoms.velocities *= lam
        # keep the cached state temperature in sync with the velocities
        state.kinetic_energy = atoms.kinetic_energy()
        state.temperature = 2.0 * state.kinetic_energy / atoms.dof
