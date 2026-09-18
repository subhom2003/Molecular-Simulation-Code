"""
=============================================================================
SimulationState -- mutable per-step state shared by fixes & computes
=============================================================================

A single object that holds the :class:`Atoms`, the latest :class:`ForceResult`
(or a placeholder before forces are computed), and bookkeeping that fixes and
computes may need to access (current step, sim time, cached kinetic energy,
the Nose-Hoover extended Hamiltonian, etc.).

The integrator's hooks operate on this object, so a fix that wants to
inspect/modify positions/velocities can do so without passing many
arguments around.  Mirrors LAMMPS's domain->atoms and pair->v_t_dot.
=============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationState:
    """Mutable per-step state of a running simulation."""

    #: Per-particle arrays (positions, velocities, types, masses, ...).
    atoms: object = None  # Atoms
    #: Simulation box, used for PBC by fixes/computes.
    box: object = None    # Box
    #: Latest force result (forces, potential, virial, min_r, max_f).
    force_result: object = None

    #: Current integrator step (0-based count of completed steps).
    step: int = 0
    #: Current simulation time = step * dt.
    time: float = 0.0
    #: Time step; kept here since thermostats/fixes need it.
    dt: float = 0.005

    #: Unwrapped positions (for MSD and visualization); shape (N, 3).
    unwrapped: np.ndarray = None

    #: Cached kinetic energy at the current step (set after velocity update).
    kinetic_energy: float = 0.0
    #: Cached instantaneous temperature.
    temperature: float = 0.0

    #: Extended Hamiltonian (only meaningful when a Nose-Hoover fix is active).
    conserved_energy: float = 0.0
