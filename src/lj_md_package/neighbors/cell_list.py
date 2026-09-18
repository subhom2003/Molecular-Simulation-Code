"""
=============================================================================
Linked-cell neighbor list
=============================================================================

A linked-cell decomposition constructs a spatial grid of cells of side
r_list = r_cut + r_skin and only inspects pairs in the same or adjacent
cells.  Construction is O(N) in time and memory; suitable for boxes with
L / r_list >= 3 (i.e. at least 3 cells per dimension - the half-stencil
algorithm double-counts pairs when only 2 cells fit per dimension).

The interface is identical to :class:`neighbors.verlet.VerletNL`.
=============================================================================
"""

from __future__ import annotations
import time
import numpy as np
import logging

from .abc import NeighborList

logger = logging.getLogger("lj_md.neighbor")


class CellList(NeighborList):
    """Linked-cell neighbor list for a cubic / orthogonal box."""

    def __init__(self, r_cut: float, r_skin: float, L: float):
        self.r_cut = float(r_cut)
        self.r_skin = float(r_skin)
        self.L = float(L)
        self.r_list = self.r_cut + self.r_skin

        # Linked-cell needs >= 3 cells per dimension; with n_cells = 2 the
        # 13-offset half stencil wraps around ((cx +/- 1) % 2 coincide) and
        # every cross-cell pair would be counted TWICE (wrong energies and
        # forces).  Use VerletNL for such small boxes instead.
        n = int(np.floor(self.L / self.r_list))
        if n < 3:
            raise ValueError(
                f"CellList requires L/r_list >= 3 cells per dimension "
                f"(L={self.L:.4f}, r_list={self.r_list:.4f}); "
                f"use VerletNL for this box size."
            )
        self.n_cells = n
        self.cell_size = self.L / self.n_cells

        self._i_pairs = None
        self._j_pairs = None
        self._ref_positions = None
        self._n_builds = 0
        self._total_build_time = 0.0

    def build(self, positions: np.ndarray) -> None:
        t0 = time.perf_counter()
        N = positions.shape[0]
        nc = self.n_cells
        L = self.L
        r_list_sq = self.r_list ** 2

        # integer cell index for each particle
        ixc = np.clip((positions[:, 0] / self.cell_size).astype(np.int64), 0, nc - 1)
        iyc = np.clip((positions[:, 1] / self.cell_size).astype(np.int64), 0, nc - 1)
        izc = np.clip((positions[:, 2] / self.cell_size).astype(np.int64), 0, nc - 1)
        flat = ixc * nc * nc + iyc * nc + izc
        n_cells_total = nc * nc * nc

        # Build linked list
        head = -np.ones(n_cells_total, dtype=np.int64)
        next_idx = -np.ones(N, dtype=np.int64)
        for n in range(N):
            cid = flat[n]
            next_idx[n] = head[cid]
            head[cid] = n

        # Half-cell-shell offsets (only forward neighbour cells to avoid double counts)
        cell_offs = [
            (0, 0, 0), (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0),
            (-1, -1, 1), (0, -1, 1), (1, -1, 1),
            (-1,  0, 1), (0,  0, 1), (1,  0, 1),
            (-1,  1, 1), (0,  1, 1), (1,  1, 1),
        ]

        i_list = []
        j_list = []

        for cx in range(nc):
            for cy in range(nc):
                for cz in range(nc):
                    cid = cx * nc * nc + cy * nc + cz
                    n = head[cid]
                    while n != -1:
                        # same cell: only pairs with i < j
                        m = next_idx[n]
                        while m != -1:
                            d = positions[m] - positions[n]
                            d -= L * np.round(d / L)
                            if d.dot(d) < r_list_sq:
                                i_list.append(n); j_list.append(m)
                            m = next_idx[m]
                        # neighbouring cells
                        for ox, oy, oz in cell_offs[1:]:
                            oxp = (cx + ox) % nc
                            oyp = (cy + oy) % nc
                            ozp = (cz + oz) % nc
                            nc_cid = oxp * nc * nc + oyp * nc + ozp
                            m = head[nc_cid]
                            while m != -1:
                                d = positions[m] - positions[n]
                                d -= L * np.round(d / L)
                                if d.dot(d) < r_list_sq:
                                    i_list.append(n); j_list.append(m)
                                m = next_idx[m]
                        n = next_idx[n]

        self._i_pairs = np.array(i_list, dtype=np.int64)
        self._j_pairs = np.array(j_list, dtype=np.int64)
        self._ref_positions = positions.copy()

        self._n_builds += 1
        self._total_build_time += time.perf_counter() - t0
        logger.debug(
            "Cell list rebuilt (cells=%d^3, pairs=%d); total builds=%d",
            nc, self._i_pairs.size, self._n_builds,
        )

    def update_box_length(self, L: float, positions: np.ndarray | None = None) -> None:
        """Barostat hook: re-derive the grid geometry for the new length."""
        self.L = float(L)
        self.r_list = self.r_cut + self.r_skin
        n = int(np.floor(self.L / self.r_list))
        if n < 3:
            raise ValueError(
                f"CellList requires L/r_list >= 3 after box resize "
                f"(L={self.L:.4f}); shrink the barostat step or use VerletNL."
            )
        self.n_cells = n
        self.cell_size = self.L / self.n_cells
        self._ref_positions = None  # force rebuild

    def needs_rebuild(self, positions: np.ndarray) -> bool:
        if self._ref_positions is None:
            return True
        disp = positions - self._ref_positions
        disp -= self.L * np.round(disp / self.L)
        return float(np.einsum("ij,ij->i", disp, disp).max()) > (self.r_skin / 2.0) ** 2

    def get_pairs(self) -> tuple:
        if self._i_pairs is None:
            raise RuntimeError("CellList.get_pairs called before build()")
        return self._i_pairs, self._j_pairs

    @property
    def n_builds(self) -> int:
        return self._n_builds

    def diagnostics(self) -> str:
        avg = (self._total_build_time / max(self._n_builds, 1)) * 1000.0
        return (
            f"Cell list rebuilds:         {self._n_builds}\n"
            f"  grid:                     {self.n_cells}^3 cells, side {self.cell_size:.3f}\n"
            f"  avg build time:           {avg:.3f} ms\n"
        )
