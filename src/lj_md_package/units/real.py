"""Real units (kcal/mol, Angstrom, fs, g/mol, K, atm) -- LAMMPS style."""
from __future__ import annotations
from .abc import UnitSystem


class RealUnits(UnitSystem):
    """LAMMPS 'real' unit convention for argon-like LJ fluids."""

    length_label = "Angstrom"
    energy_label = "kcal/mol"
    mass_label = "g/mol"
    time_label = "fs"
    temperature_label = "K"
    pressure_label = "atm"
    density_label = "g/cm^3"

    # 1 reduced -> 1 real (when argon assignments mozzarella-like are assumed)
    # sigma = 3.405 A; eps = 0.238 kcal/mol; m = 39.948 g/mol; tau ~ 2.16 ps
    length_to_real = 3.405         # sigma -> A
    energy_to_real = 0.238         # eps -> kcal/mol
    mass_to_real = 39.948          # m -> g/mol
    time_to_real = 2156.0          # tau -> fs
    temperature_to_real = 119.8    # T* -> K
    pressure_to_real = 413.6       # P* -> atm
    density_to_real = 1.6316       # rho* -> g/cm^3

    def name(self) -> str:
        return "real"

    def reduced_to_real_length(self, value: float) -> float:
        return value * self.length_to_real
