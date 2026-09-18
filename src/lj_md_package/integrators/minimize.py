"""
=============================================================================
Steepest-descent minimizer (``min_style sd`` in LAMMPS)
=============================================================================

Energy minimization by steepest descent.  Each step::

    r <- r - alpha * F   with  alpha = max_displacement / max(|F|) ;

alpha is normalized so that the maximum displacement of any particle per
iteration is bounded by ``max_displacement`` (default 0.1 sigma).
Useful as a startup routine to relax overlapping initial configurations
before running dynamics.
=============================================================================
"""

from __future__ import annotations
import numpy as np
from .abc import Integrator


class MinimizeSD(Integrator):
    def __init__(self, dt: float, box, max_displacement: float = 0.1,
                 energy_tolerance: float = 1e-10, force_tolerance: float = 1e-10,
                 max_steps: int = 1000):
        self.box = box
        self.max_displacement = float(max_displacement)
        self.energy_tolerance = float(energy_tolerance)
        self.force_tolerance = float(force_tolerance)
        self.max_steps = int(max_steps)
        self._prev_potential = None

    def initial_integrate(self, state) -> None:
        atoms = state.atoms
        F = state.force_result.forces
        f_max = float(np.abs(F).max()) if F.size else 0.0
        if f_max < self.force_tolerance:
            return
        alpha = self.max_displacement / f_max
        # Steepest descent moves ALONG the force (F = -grad U).
        atoms.positions += alpha * F
        atoms.positions = self.box.wrap(atoms.positions)
        if state.unwrapped is None:
            state.unwrapped = atoms.positions.copy()
        state.unwrapped += alpha * F
        # NOTE: state.step / state.time are managed by Simulation.run.

    def final_integrate(self, state) -> None:
        # Force-only minimizer; no velocity update needed. KE = 0 by definition.
        state.kinetic_energy = 0.0
        state.temperature = 0.0

    def has_converged(self, state) -> bool:
        potential = state.force_result.potential if state.force_result else float("inf")
        if self._prev_potential is not None:
            dE = abs(potential - self._prev_potential)
            if dE < self.energy_tolerance:
                return True
        self._prev_potential = potential
        f_max = float(np.abs(state.force_result.forces).max())
        return f_max < self.force_tolerance
