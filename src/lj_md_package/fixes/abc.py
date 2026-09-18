"""
Base class for "fixes" -- LAMMPS-style plug-ins attached to the run loop.

A fix is an object that is invoked at specific stages of each timestep:

- :meth:`initial_integrate` -- before the first half-step velocity update.
  Used by FixNVE / FixNVT to actually integrate.
- :meth:`post_force` -- after the force kernel has run (used to apply
  post-force corrections, e.g. force capping or stochastic-Langevin kicks).
- :meth:`final_integrate` -- after the second half-step velocity update.
  NVE's final_integrate does the second velocity-Verlet half-step.
- :meth:`end_of_step` -- after all integration steps in the loop iteration,
  but before computes run. Used by FixMomentum to zero the COM every N steps.

This mirrors LAMMPS's own FIX_AFTER_FORCE / FIX_AFTER_INTEGRATE logic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class Fix(ABC):
    """Abstract base class for all fixes."""

    #: unique fix id (string)
    fix_id: str = ""
    #: group label (e.g. "all") -- currently every fix applies to all atoms.
    group: str = "all"
    #: frequency (steps) at which the end_of_step action is invoked (default every step)
    every: int = 1

    def __init__(self, fix_id: str = "fix", group: str = "all", every: int = 1):
        self.fix_id = fix_id
        self.group = group
        self.every = int(every)

    @abstractmethod
    def name(self) -> str:
        """Short LAMMPS-style fix name (e.g. 'nve', 'temp/rescale')."""

    # -------------------------------------- lifecycle hooks (default no-op)
    def initial_integrate(self, state) -> None:
        pass

    def post_force(self, state) -> None:
        pass

    def final_integrate(self, state) -> None:
        pass

    def end_of_step(self, state) -> None:
        if state.step % self.every == 0:
            self.do_end_of_step(state)

    def do_end_of_step(self, state) -> None:
        pass

    # -------------------- optional contributions to the extended Hamiltonian
    def conserved_energy(self, state) -> float:
        """Return any extra term this fix contributes to the conserved
        Hamiltonian.  For thermostats this is typically the bath kinetic
        + potential energy.  NVE contributes 0."""
        return 0.0
