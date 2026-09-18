"""FixNVE -- velocity-Verlet NVE integrator wrapper (``fix nve`` in LAMMPS)."""
from __future__ import annotations
from .. integrators.velocity_verlet import VelocityVerlet
from .abc import Fix


class FixNVE(Fix):
    """Pure NVE dynamics. Holds the integrator half-steps together."""

    def __init__(self, fix_id: str = "nve", group: str = "all",
                 integrator: VelocityVerlet | None = None):
        super().__init__(fix_id, group)
        self.integrator = integrator  # may be None; will pull from state

    def name(self) -> str:
        return "nve"

    def initial_integrate(self, state) -> None:
        # the simulation owns a shared integrator -- just dispatch to it
        if self.integrator is not None:
            self.integrator.initial_integrate(state)

    def final_integrate(self, state) -> None:
        if self.integrator is not None:
            self.integrator.final_integrate(state)
