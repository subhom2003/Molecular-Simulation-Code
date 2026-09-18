"""Neighbor list strategies -- pluggable O(N^2)/O(N) pair lookup."""

from .abc import NeighborList
from .verlet import VerletNL
from .cell_list import CellList

NEIGHBOR_STYLES = {
    "verlet": VerletNL,
    "cell":   CellList,
    "nsq":    VerletNL,   # naive N^2 queued through the Verlet (same shape)
}

__all__ = ["NeighborList", "VerletNL", "CellList", "NEIGHBOR_STYLES"]
