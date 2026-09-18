"""Units package: unit-system strategy classes for the MD engine."""
from .abc import UnitSystem
from .lj import LJUnits
from .real import RealUnits
from .metal import MetalUnits
from .si import SIUnits
from .reference import LITERATURE_LJ

__all__ = [
    "UnitSystem", "LJUnits", "RealUnits", "MetalUnits", "SIUnits",
    "LITERATURE_LJ",
    "get_unit_system",
]


def get_unit_system(name: str) -> UnitSystem:
    """Factory dispatch: 'lj' / 'real' / 'metal' / 'si' -> UnitSystem."""
    name = name.lower()
    table = {"lj": LJUnits, "real": RealUnits, "metal": MetalUnits, "si": SIUnits}
    if name not in table:
        raise ValueError(f"Unknown unit system {name!r}; pick one of {list(table)}.")
    return table[name]()
