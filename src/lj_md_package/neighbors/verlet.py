"""
=============================================================================
Verlet neighbor list
=============================================================================

A Verlet (neighbour-)list stores pairs (i,j) with r_ij < r_cut + r_skin
and is rebuilt only when any particle has displaced by more than r_skin/2
since the last build -- guaranteed not to miss a within-cutoff pair.

Construction is O(N^2) in memory and time, but the *use* is O(N * <n_pairs>),
which is the whole point of the technique.  For N > ~3000 you should switch
to :class:`neighbors.cell_list.CellList` (built via ``neighbor 0.3 bin``).

References
----------
- L. Verlet (1967), Phys. Rev. 159, 98.
- Allen & Tildesley (2017), §2.4 and §5.3.
=============================================================================
"""

from __future__ import annotations
import time
import numpy as np
import logging

from .abc import NeighborList

logger = logging.getLogger("lj_md.neighbor")


class VerletNL(NeighborList):
    """Verlet-style neighbor list for a cubic / orthogonal box."""

    def __init__(self, r_cut: float, r_skin: float, L: float):
        self.r_cut = float(r_cut)
        self.r_skin = float(r_skin)
        self.L = float(L)
        self.r_list = self.r_cut + self.r_skin

        self._i_pairs: np.ndarray | None = None
        self._j_pairs: np.ndarray | None = None
        self._ref_positions: np.ndarray | None = None

        # diagnostics
        self._n_builds = 0
        self._total_build_time = 0.0

    # ------------------------------------------------------------------ build
    def build(self, positions: np.ndarray) -> None:
        """Rebuild the pair list from the given positions snapshot."""
        n = positions.shape[0]
        t0 = time.perf_counter()

        # Generate all i < j pairs (memory: N(N-1)/2 int64 pairs).
        i_all, j_all = np.triu_indices(n, k=1)

        if len(i_all) == 0:
            self._i_pairs = np.array([], dtype=np.int64)
            self._j_pairs = np.array([], dtype=np.int64)
            self._ref_positions = positions.copy()
            return

        dr = positions[j_all] - positions[i_all]
        # Minimum-image convention (orthogonal-cubic box)
        dr -= self.L * np.round(dr / self.L)
        r2 = np.einsum("ij,ij->i", dr, dr)

        mask = r2 < self.r_list ** 2
        self._i_pairs = i_all[mask]
        self._j_pairs = j_all[mask]
        self._ref_positions = positions.copy()

        self._n_builds += 1
        self._total_build_time += time.perf_counter() - t0
        logger.debug(
            "Neighbor list rebuilt (%s pairs out of %s); total builds=%d",
            self._i_pairs.size, i_all.size, self._n_builds,
        )

    # --------------------------------------------------------- needs_rebuild
    def needs_rebuild(self, positions: np.ndarray) -> bool:
        """Return True if any particle displaced by > r_skin/2 since last build."""
        if self._ref_positions is None:
            return True
        disp = positions - self._ref_positions
        # apply minimum image to displacement
        disp -= self.L * np.round(disp / self.L)
        max_disp_sq = float(np.einsum("ij,ij->i", disp, disp).max())
        return max_disp_sq > (self.r_skin / 2.0) ** 2

    # --------------------------------------------------------------- pairs
    def get_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        if self._i_pairs is None:
            raise RuntimeError("NeighborList.get_pairs called before build()")
        return self._i_pairs, self._j_pairs

    # ------------------------------------------------------- diagnostics
    def diagnostics(self) -> str:
        avg = (self._total_build_time / max(self._n_builds, 1)) * 1000.0
        return (
            f"Neighbor list rebuilds:    {self._n_builds}\n"
            f"  avg build time:           {avg:.3f} ms\n"
            f"  average pairs per build:  "
            f"{(self._i_pairs.size if self._i_pairs is not None else 0):d}"
        )

    @property
    def n_builds(self) -> int:
        return self._n_builds
