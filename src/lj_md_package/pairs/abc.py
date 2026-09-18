"""
PairStyle base class + ForceResult container.

A pair style is a short-range pair interaction kernel that, given a
:class:`Atoms` snapshot, a neighbor list, and a :class:`Box`, returns a
:class:`ForceResult` (per-particle forces + scalar contributions: potential
energy, virial, min_r and max_f diagnostics).

Dispatch is performed by name through :data:`pairs.PAIR_STYLES`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np


class ForceResult(NamedTuple):
    """Per-call result of a pair-style compute call."""

    forces: np.ndarray      #: (N, 3) per-particle force vectors
    potential: float         #: total potential energy (cut/shifted; tail not included)
    virial: float            #: virial = - sum r_ij . F_ij, used for pressure
    min_r: float             #: minimum separation found inside the cutoff
    max_f: float             #: largest |F_ij| applied
    n_pairs: int = 0         #: number of within-cutoff pair interactions counted


class PairStyle(ABC):
    """Short-range pair-style strategy object."""

    @abstractmethod
    def compute(self, atoms, nlist, box, coeffs=None) -> ForceResult:
        """Compute the forces, potential and virial for one snapshot."""

    @abstractmethod
    def name(self) -> str:
        """Return the short-form LAMMPS-style pair-style name (e.g. 'lj/cut')."""

    @property
    def cutoff(self) -> float:
        """Cutoff radius of this pair style."""
        raise NotImplementedError
