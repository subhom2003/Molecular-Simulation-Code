"""DumpThermo -- one CSV per thermo sample, written as a stream."""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np

from .abc import Dump


class DumpThermo(Dump):
    """Mirror LAMMPS ``dump thermo`` style.  Writes a CSV with thermo columns."""

    HEADER = [
        "step", "time", "T", "P", "PE", "KE", "E_total", "E_per_atom",
        "density", "virial", "min_r", "max_f", "p_COM",
    ]

    def __init__(self, dump_id: str = "thermo", every: int = 10,
                 filename: str = "output/thermo.csv",
                 rho_star: float = 0.8442, P_tail: float = 0.0,
                 u_tail: float = 0.0):
        """
        Args:
            dump_id: unique identifier (used by ``undump``).
            every:   CSV row every ``every`` steps.
            filename: CSV path.
            rho_star: number density rho* = N/V.
            P_tail:  pressure long-range tail correction (added to every P).
            u_tail:  per-particle long-range tail correction to the potential
                     energy. The dump adds ``u_tail * N`` to the truncated PE.
        """
        super().__init__(dump_id, "all", every, filename)
        self.rho_star = rho_star
        self.P_tail = P_tail
        self.u_tail = u_tail
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(filename, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.HEADER)

    def write(self, step: int, state) -> None:
        fr = state.force_result
        atoms = state.atoms
        N = atoms.N
        KE = state.kinetic_energy
        T = state.temperature
        PE_truncated = fr.potential if fr is not None else 0.0
        virial = fr.virial if fr is not None else 0.0
        V = N / self.rho_star
        P = self.rho_star * T + virial / (3.0 * V) + self.P_tail
        PE = PE_truncated + self.u_tail * N
        E_total = KE + PE
        p_com = float(np.linalg.norm(atoms.velocities.mean(axis=0) * N))
        min_r = fr.min_r if fr is not None else float("nan")
        max_f = fr.max_f if fr is not None else 0.0
        self._writer.writerow([
            step, step * state.dt, T, P, PE, KE, E_total, E_total / N,
            self.rho_star, virial, min_r, max_f, p_com,
        ])

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()
