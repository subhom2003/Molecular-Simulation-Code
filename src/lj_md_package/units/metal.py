"""Metal units (eV, Angstrom, ps, amu, K, bar) -- LAMMPS style."""
from __future__ import annotations
from .abc import UnitSystem


class MetalUnits(UnitSystem):
    """LAMMPS 'metal' unit convention for argon-like LJ fluids."""

    length_label = "Angstrom"
    energy_label = "eV"
    mass_label = "amu"
    time_label = "ps"
    temperature_label = "K"
    pressure_label = "bar"
    density_label = "g/cm^3"

    # Argon: sigma=3.405 A, eps=10.35 meV=0.01035 eV, m=39.948 amu, tau~2.16 ps
    length_to_real = 3.405
    energy_to_real = 0.01035
    mass_to_real = 39.948
    time_to_real = 2.156
    temperature_to_real = 119.8
    pressure_to_real = 413.6       # eps/sigma^3 -> bar (approx)
    density_to_real = 1.6316

    def name(self) -> str:
        return "metal"

    def reduced_to_real_length(self, value: float) -> float:
        return value * self.length_to_real
