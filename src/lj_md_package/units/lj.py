"""


Lennard-Jones reduced units.

Quantities are dimensionless.  Argon reference values (used to translate
to SI for the unit-marks in LaTeX-style axis labels if the user requests
``units lj`` plus ``--real``):
    sigma = 3.405 Angstrom
    eps/k_B = 119.8 K
    m = 39.948 u
    tau = sigma * sqrt(m/eps) ~ 2.156 ps
"""

from __future__ import annotations
from .abc import UnitSystem


class LJUnits(UnitSystem):
    """All reduced units; conversions are identity."""

    length_label = "sigma"
    energy_label = "epsilon"
    mass_label = "m"
    time_label = "tau"
    temperature_label = "T*"
    pressure_label = "P*"
    density_label = "rho*"

    # Reduced -> reduced: identity
    length_to_SI = 3.405e-10   # sigma in metres (argon)
    energy_to_SI = 1.654e-21   # epsilon in joules (argon: k_B*119.8)
    mass_to_SI = 6.634e-26     # argon mass in kg
    time_to_SI = 2.156e-12     # tau in seconds
    temperature_to_SI = 119.8  # eps/kB in kelvin
    pressure_to_SI = 4.193e7   # eps/sigma^3 in pascals

    def name(self) -> str:
        return "lj"

    def reduced_to_real_length(self, value: float) -> float:
        return value
