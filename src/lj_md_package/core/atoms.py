"""
=============================================================================
Atoms: per-particle state for a single simulation step
=============================================================================

Container holding all per-particle arrays needed by the engine:
positions, velocities, types, masses, charges, molecule IDs.

The unit of mass is the LJ reduced unit (m=1) unless the user supplies a
type-mass table via :meth:`set_masses_from_types`.
=============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Atoms:
    """
    Per-particle mutable state.

    Parameters
    ----------
    positions : (N, 3) array
        Wrapped positions inside the simulation box.
    velocities : (N, 3) array
        Particle velocities (in reduced units sigma/tau).
    types : (N,) int array, optional
        Particle type id (default: all 1).
    masses : (N,) or (n_types,) array, optional
        Particle masses; default = 1 for all (LJ reduced unit).
    charges : (N,) array, optional
        Particle charges (default 0; not used by current pair styles).
    mol_ids : (N,) int array, optional
        Molecule ids (default 0).
    """

    positions: np.ndarray
    velocities: np.ndarray
    types: np.ndarray = None
    masses: np.ndarray = None
    charges: np.ndarray = None
    mol_ids: np.ndarray = None
    # IDs assigned at construction (1..N), used by dump sort.
    ids: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        N = self.positions.shape[0]
        if self.positions.shape != (N, 3):
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")
        if self.velocities is None:
            self.velocities = np.zeros_like(self.positions)
        if self.types is None:
            self.types = np.ones(N, dtype=np.int32)
        if self.masses is None:
            self.masses = np.ones(N, dtype=float)
        if self.charges is None:
            self.charges = np.zeros(N, dtype=float)
        if self.mol_ids is None:
            self.mol_ids = np.zeros(N, dtype=np.int32)
        if self.ids is None:
            self.ids = np.arange(1, N + 1, dtype=np.int32)

    # -------------------------------------------------------- array accessors
    @property
    def N(self) -> int:
        return int(self.positions.shape[0])

    @property
    def dof(self) -> int:
        """Degrees of freedom: 3N - 3 after COM velocity removal."""
        return 3 * self.N - 3

    def kinetic_energy(self) -> float:
        """Total kinetic energy in reduced units: 0.5 * sum(m v^2)."""
        return float(0.5 * np.einsum(
            "i,ij,ij->", self.masses, self.velocities, self.velocities
        ))

    def temperature(self, dof: int | None = None) -> float:
        """Instantaneous temperature from the kinetic energy.

        Default DOF is :attr:`dof` (3N - 3 after COM removal) which is the
        correct value post-:func:`Atoms.remove_com_motion`.
        """
        ke = self.kinetic_energy()
        n = self.dof if dof is None else int(dof)
        return 2.0 * ke / float(n)

    # ------------------------------------------------------------ mutations
    def remove_com_motion(self) -> np.ndarray:
        """Subtract the center-of-mass linear momentum from velocities.
        Returns the COM momentum vector (length-3)."""
        if np.allclose(self.masses, self.masses[0]):
            v_com = self.velocities.mean(axis=0)
        else:
            total_mass = self.masses.sum()
            v_com = (self.velocities * self.masses[:, None]).sum(axis=0) / total_mass
        self.velocities -= v_com
        return v_com * self.masses.sum()

    def scale_to_temperature(self, T_target: float, dof: int | None = None) -> None:
        """Scale velocities so the instantaneous temperature matches T_target."""
        n = self.dof if dof is None else int(dof)
        ke = self.kinetic_energy()
        T_current = 2.0 * ke / float(n)
        if T_current <= 0:
            raise RuntimeError("Cannot scale zero-temperature velocities.")
        lam = np.sqrt(T_target / T_current)
        self.velocities *= lam

    def __len__(self) -> int:
        return self.N
