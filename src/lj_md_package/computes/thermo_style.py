"""
ThermoStyle -- per-step thermo log + buffered CSV writer.

Mirrors LAMMPS ``thermo_style custom ...`` keyword lists.  Records per-step
data in a dict-of-lists (vs old ThermodynamicRecord's list-of-dicts) for
amortized O(1) appending and O(n) array conversion at the end.
"""

from __future__ import annotations
import numpy as np

from .abc import Compute


_THERMO_KEYS = (
    "step", "time", "T", "P", "PE", "KE", "E_total", "E_per_atom",
    "density", "virial", "min_r", "max_f", "p_COM",
)


class ThermoStyle(Compute):
    """Accumulates per-step thermo quantities. Functions like LAMMPS's
    ``thermo_style`` keyword-subset listing while recording all available
    keys internally for the post-run validation report.
    """

    def __init__(self, compute_id: str = "thermo", group: str = "all",
                 every: int = 1, rho_star: float = 0.8442, P_tail: float = 0.0,
                 u_tail: float = 0.0, n_equil: int = 0,
                 custom_keys: tuple = _THERMO_KEYS):
        super().__init__(compute_id=compute_id, group=group, every=every)
        self.rho_star = rho_star
        self.P_tail = P_tail
        self.u_tail = u_tail
        self.n_equil = int(n_equil)
        self._keys = tuple(custom_keys)
        self._data = {k: [] for k in _THERMO_KEYS}
        self._N = 108

    def name(self) -> str:
        return "thermo"

    def set_atom_count(self, N: int) -> None:
        self._N = int(N)

    def set_equil_steps(self, n_equil: int) -> None:
        self.n_equil = int(n_equil)

    def record(self, step: int, dt: float, KE: float, PE: float,
               virial: float, T: float, P: float, min_r: float,
               max_f: float, p_COM: float) -> None:
        """Record a row of thermo quantities.

        ``P`` and ``PE`` come in as *truncated* (no tail correction); this
        method adds the configured tail corrections ``P_tail`` and
        ``u_tail * N`` to produce the literature-comparable cross-quantities.
        """
        PE_total = PE + self.u_tail * self._N
        E_total = KE + PE_total
        row = {
            "step":     step,
            "time":     step * dt,
            "T":        T,
            "P":        P + self.P_tail,
            "PE":       PE_total,
            "KE":       KE,
            "E_total":  E_total,
            "E_per_atom": E_total / self._N,
            "density":  self.rho_star,
            "virial":   virial,
            "min_r":    min_r,
            "max_f":    max_f,
            "p_COM":    p_COM,
        }
        for k in _THERMO_KEYS:
            self._data[k].append(row[k])

    def compute(self, state, force_result) -> None:
        """Read the state and append a thermo row to internal storage."""
        fr = force_result
        atoms = state.atoms
        N = atoms.N
        KE = state.kinetic_energy
        T = state.temperature
        PE_truncated = fr.potential if fr is not None else 0.0
        virial = fr.virial if fr is not None else 0.0
        V = N / self.rho_star
        P_truncated = self.rho_star * T + virial / (3.0 * V)
        p_com = float(np.linalg.norm(atoms.velocities.mean(axis=0) * N))
        min_r = fr.min_r if fr is not None else float("nan")
        max_f = fr.max_f if fr is not None else 0.0
        self.record(step=state.step, dt=state.dt, KE=KE, PE=PE_truncated,
                    virial=virial, T=T, P=P_truncated,
                    min_r=min_r, max_f=max_f, p_COM=p_com)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(self._data[k], dtype=float) for k in _THERMO_KEYS}

    def production_mask(self) -> np.ndarray:
        return np.asarray(self._data["step"]) > self.n_equil

    def production_averages(self) -> dict[str, float]:
        mask = self.production_mask()
        return {
            k: float(np.mean(np.asarray(self._data[k])[mask]))
            for k in _THERMO_KEYS if len(self._data[k]) > 0
        }

    def energy_drift(self) -> float:
        steps = np.asarray(self._data["step"])
        energy = np.asarray(self._data["E_total"])
        # use the first *production* point as reference
        prod_idx = np.where(steps > self.n_equil)[0]
        if prod_idx.size < 2:
            return float("nan")
        e0 = energy[prod_idx[0]]
        if abs(e0) < 1e-12:
            return float("nan")
        return abs(energy[prod_idx[-1]] - e0) / abs(e0)

    def __len__(self) -> int:
        return len(self._data["step"])

    @property
    def keys(self) -> tuple:
        return self._keys
