"""
NeighborList base class.

All neighbor-list strategies expose the same API: build the list from a
position snapshot, optionally check whether a rebuild is needed, and
return all current pair indices via ``get_pairs()``.

The user-facing pair style only sees the (i, j) pair arrays -- it treats
them identically whether the list was built by an O(N^2) Verlet scheme or
by a linked-cell decomposition.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class NeighborList(ABC):
    """Abstract neighbor-list strategy."""

    @abstractmethod
    def build(self, positions: np.ndarray) -> None:
        """Rebuild the pair list from the current positions."""

    @abstractmethod
    def needs_rebuild(self, positions: np.ndarray) -> bool:
        """Return True if the list should be rebuilt given current positions."""

    @abstractmethod
    def get_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (i, j) integer arrays of all pairs in the list."""

    def update_if_needed(self, positions: np.ndarray) -> bool:
        """Rebuild the list if needed; return True if a rebuild happened."""
        if self.needs_rebuild(positions):
            self.build(positions)
            return True
        return False

    def update_box_length(self, L: float, positions: np.ndarray | None = None) -> None:
        """Notify the list that the (cubic) box length changed (barostat).

        Updates the stored length and forces a rebuild on the next call.
        Subclasses with derived grid geometry must override.
        """
        self.L = float(L)
        self.r_list = self.r_cut + self.r_skin
        self._ref_positions = None  # force rebuild

    def diagnostics(self) -> str:
        return ""
