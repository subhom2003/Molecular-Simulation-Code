"""
Unit system strategy.
=============================================================================

A UnitSystem carries the *physical* units selected for the simulation
(``units lj`` -> LJUnits, ``units real`` -> RealUnits, ``units metal``
-> MetalUnits, ``units si`` -> SIUnits).

All physics inside the engine is computed in LJ reduced units; the
unit system is therefore only consulted at I/O time and at the boundary
between user inputs and the engine.

Concrete subclasses (e.g. :class:`lj_md_package.units.lj.LJUnits`) populate
the conversion factors and label strings.
=============================================================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class UnitSystem(ABC):
    """Base class -- the reduced LJ engine uses the same numerical values
    regardless of the user-chosen ``units``."""

    #: Length unit label (for axis labels etc.)
    length_label: str = "sigma"
    #: Energy unit label
    energy_label: str = "epsilon"
    #: Mass unit label
    mass_label: str = "m"
    #: Time unit label
    time_label: str = "tau"
    #: Temperature unit label
    temperature_label: str = "T*"
    #: Pressure unit label
    pressure_label: str = "P*"
    #: Density unit label
    density_label: str = "rho*"

    # Physical conversion factors (reduced unit -> SI).
    # Subclasses override these.  Length_converted = reduced * length_to_SI.
    length_to_SI: float = 1.0      # LJ: sigma in metres
    energy_to_SI: float = 1.0      # LJ: epsilon in joules
    mass_to_SI: float = 1.0        # LJ: m in kg
    time_to_SI: float = 1.0        # LJ: tau in seconds
    temperature_to_SI: float = 1.0 # LJ: T in kelvin
    pressure_to_SI: float = 1.0    # LJ: P in pascals

    @abstractmethod
    def name(self) -> str: ...

    def reduced_to_real_length(self, value: float) -> float:
        """Convert a reduced length to the current unit system's unit."""
        return value  # everything is in LJ units internally; subclasses override.

    @classmethod
    def from_name(cls, name: str) -> "UnitSystem":
        """Factory-style dispatch -- see lj_md_package.units.get_unit_system."""
        from . import get_unit_system
        return get_unit_system(name)
