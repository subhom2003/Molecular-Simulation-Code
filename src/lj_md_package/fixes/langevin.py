"""
FixLangevin -- stochastic Langevin thermostat with BAOAB (GJF) splitting.

Implements the Leimkuhler-Matthews BAOAB discretization of

    m dv = F dt - m gamma v dt + sqrt(2 m gamma k_B T) dW

as a velocity-Verlet-like sequence compatible with the run loop's
fix hooks:

    B:  v  += (dt/2) F(t)/m
    A:  r  += (dt/2) v
    O:  v   = c1 v + c2 xi,   c1 = exp(-gamma dt),
                              c2 = sqrt(k_B T (1 - c1^2) / m)
    A:  r  += (dt/2) v
    [forces recomputed]
    B:  v  += (dt/2) F(t+dt)/m

The O substep is the *exact* Ornstein-Uhlenbeck propagator, so the kinetic
temperature has no O(dt) bias (the old BBK variant was ~+1% hot at
gamma=1, dt=0.005).  At gamma -> 0 this reduces to velocity Verlet.

This fix REPLACES FixNVE (it performs the full integration).  Do not use
both at once - ``Simulation.run`` will refuse.

#আমার প্রিয় চারু 

References
----------
- B. Leimkuhler & C. Matthews, J. Chem. Phys. 138, 174102 (2013) (BAOAB).
- N. Gronbech-Jensen & O. Farago, Mol. Phys. 111, 983 (2013) (GJF).
"""

from __future__ import annotations
import numpy as np
import logging

from .abc import Fix

logger = logging.getLogger("lj_md.fix")


class FixLangevin(Fix):
    """Stochastic Langevin thermostat (BAOAB), self-integrating.

    Parameters
    ----------
    T_target : float
        Target kinetic temperature (reduced units).
    gamma : float
        Friction coefficient (1/tau).
    Tstop : float | None
        If given, ramp T linearly from T_target to Tstop across the
        current ``run`` block.
    seed : int
        RNG seed for the stochastic force.
    """

    def __init__(self, fix_id: str = "langevin", group: str = "all",
                 T_target: float = 1.0, gamma: float = 1.0,
                 Tstop: float | None = None, seed: int = 42):
        super().__init__(fix_id, group)
        self.T_target = float(T_target)
        self.Tstop = float(Tstop) if Tstop is not None else None
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(int(seed))
        self.integrator = None  # deprecated: BAOAB needs no VelocityVerlet
        self._run_start = 0
        self._n_total_steps = 0

    def name(self) -> str:
        return "langevin"

    # ------------------------------------------------------------- settings
    def set_integrator(self, integrator) -> None:
        """Deprecated no-op kept for API compatibility (BAOAB performs its
        own position update; no VelocityVerlet is needed)."""
        self.integrator = integrator

    def on_run_start(self, start_step: int, n_steps: int) -> None:
        self._run_start = int(start_step)
        self._n_total_steps = int(n_steps)

    def _T_now(self, step: int) -> float:
        """Ramp T_target -> Tstop linearly over the current run block."""
        if self.Tstop is None or self._n_total_steps <= 1:
            return self.T_target
        frac = (step - self._run_start) / self._n_total_steps
        frac = min(max(frac, 0.0), 1.0)
        return self.T_target + frac * (self.Tstop - self.T_target)

    # ------------------------------------------------------------- stepping
    def initial_integrate(self, state) -> None:
        """B half-kick, A half-drift, O (exact OU), A half-drift."""
        atoms = state.atoms
        dt = state.dt
        m = atoms.masses[:, None]
        F = state.force_result.forces if state.force_result is not None else 0.0
        T = self._T_now(state.step)

        # B: half kick from F(t)
        atoms.velocities += 0.5 * dt * F / m
        # A: half drift
        dr1 = 0.5 * dt * atoms.velocities
        atoms.positions += dr1
        # O: exact Ornstein-Uhlenbeck step
        c1 = float(np.exp(-self.gamma * dt))
        c2 = np.sqrt(T * (1.0 - c1 * c1) / atoms.masses)      # (N,)
        noise = self.rng.standard_normal(size=atoms.velocities.shape)
        atoms.velocities = c1 * atoms.velocities + c2[:, None] * noise
        # A: half drift
        dr2 = 0.5 * dt * atoms.velocities
        atoms.positions += dr2

        # PBC wrap + unwrapped tracking
        box = getattr(state, "box", None)
        if box is not None:
            atoms.positions = box.wrap(atoms.positions)
        if state.unwrapped is None:
            state.unwrapped = atoms.positions.copy() - (dr1 + dr2)
        else:
            state.unwrapped += dr1 + dr2

    def final_integrate(self, state) -> None:
        """B: final half kick from F(t+dt)."""
        atoms = state.atoms
        dt = state.dt
        F_new = state.force_result.forces
        atoms.velocities += 0.5 * dt * F_new / atoms.masses[:, None]
        state.kinetic_energy = atoms.kinetic_energy()
        state.temperature = 2.0 * state.kinetic_energy / atoms.dof

    def conserved_energy(self, state) -> float:
        """Langevin has no conserved extended Hamiltonian."""
        return 0.0
