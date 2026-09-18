"""DumpEnergy -- simplified energy-only CSV (energies.csv)."""
from __future__ import annotations
from pathlib import Path
import csv

from .abc import Dump


class DumpEnergy(Dump):
    HEADER = ["step", "time", "KE", "PE", "E_total", "E_per_atom"]

    def __init__(self, dump_id: str = "energy", every: int = 10,
                 filename: str = "output/energies.csv", N: int = 108,
                 u_tail: float = 0.0):
        super().__init__(dump_id, "all", every, filename)
        self.N = N
        self.u_tail = u_tail
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(filename, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.HEADER)

    def write(self, step: int, state) -> None:
        fr = state.force_result
        KE = state.kinetic_energy
        PE_truncated = fr.potential if fr is not None else 0.0
        PE = PE_truncated + self.u_tail * self.N
        E = KE + PE
        self._writer.writerow([
            step, step * state.dt, KE, PE, E, E / self.N,
        ])

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()
