"""SI units (metre, joule, kg, second, kelvin, pascal) -- LAMMPS style."""

from __future__ import annotations
from .abc import UnitSystem


class SIUnits(UnitSystem):
    """LAMMPS 'si' unit convention for argon-like LJ fluids."""

    length_label = "m"
    energy_label = "J"
    mass_label = "kg"
    time_label = "s"
    temperature_label = "K"
    pressure_label = "Pa"
    density_label = "kg/m^3"

    # Argon reference values (used for LJ-style input conversions)
    length_to_real = 3.405e-10
    energy_to_real = 1.654e-21
    mass_to_real = 6.634e-26
    time_to_real = 2.156e-12
    temperature_to_real = 119.8
    pressure_to_real = 4.193e7
    density_to_real = 1631.6   # kg/m^3

    def name(self) -> str:
        return "si"

    def reduced_to_real_length(self, value: float) -> float:
        return value * self.length_to_real
