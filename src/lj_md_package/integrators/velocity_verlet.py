"""
=============================================================================
Velocity-Verlet integrator
=============================================================================

Velocity Verlet (Swope 1982) integrates Hamilton's equations::

    r(t+dt) = r(t) + v(t) dt + (1/2m) F(t) dt^2
    v(t+dt) = v(t) + (1/2m) [ F(t) + F(t+dt) ] dt

Equivalently, in the *split* (leapfrog-like) form this driver uses::

    v(t+dt/2) = v(t)      + (1/2m) F(t) dt       # initial_integrate
    r(t+dt)   = r(t)      + v(t+dt/2) dt
    [compute F(t+dt) here]
    v(t+dt)   = v(t+dt/2) + (1/2m) F(t+dt) dt    # final_integrate

Properties
----------
- Symplectic (preserves phase-space volume).
- Time-reversible.
- Local truncation error O(dt^4), global error O(dt^2).
- Energy drift bounded for dt small enough (LJ: dt < ~0.01 tau is safe).

The :class:`VelocityVerlet` here implements the *integrator* half-steps only;
the run loop concatenates them with force evaluations, thermostat (via a
:class:`Fix`) and observables (via a :class:`Compute`).
=============================================================================
"""

from __future__ import annotations
import numpy as np

from .abc import Integrator


class VelocityVerlet(Integrator):
    """Split-step velocity-Verlet integrator.

    Tracks the per-particle masses (LJ reduced units default m=1) and the
    previous force array, plus a copy of the wrapped box so unwrapped
    positions (for MSD) can be tracked across PBC folds.
    """

    def __init__(self, dt: float, box):
        self.dt = float(dt)
        self.box = box

    # ------------------------------------------------------------------- 1
    def initial_integrate(self, state) -> None:
        """Step-1: v_{half} = v + (dt/2) F/m ; r' = r + dt v_{half}. """
        atoms = state.atoms
        F = state.force_result.forces
        dt_half = 0.5 * self.dt
        # velocity half-step
        atoms.velocities += dt_half * F / atoms.masses[:, None]
        # full-step position
        dr = self.dt * atoms.velocities
        atoms.positions += dr
        # wrap into the primary unit cell
        atoms.positions = self.box.wrap(atoms.positions)
        # accumulate unwrapped dr (for MSD / diffusion)
        if state.unwrapped is None:
            state.unwrapped = atoms.positions - dr
        else:
            state.unwrapped += dr
        # NOTE: state.step / state.time are set by the Simulation run loop,
        # not here, to avoid double-counting across fixes.

    # ------------------------------------------------------------------- 2
    def final_integrate(self, state) -> None:
        """Step-2: v_{new} = v_{half} + (dt/2) F_new/m  (call after forces recomputed)."""
        atoms = state.atoms
        F_new = state.force_result.forces
        atoms.velocities += 0.5 * self.dt * F_new / atoms.masses[:, None]
        # cache KE/temperature
        state.kinetic_energy = atoms.kinetic_energy()
        state.temperature = 2.0 * state.kinetic_energy / atoms.dof
