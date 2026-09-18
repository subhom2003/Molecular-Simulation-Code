"""Fixes: strategies that modify the dynamics (integrators, thermostats, ...)."""
from .abc import Fix
from .nve import FixNVE
from .nvt_nose_hoover import FixNVT
from .temp_rescale import FixTempRescale
from .berendsen import FixBerendsen
from .berendsen_barostat import FixBerendsenBarostat
from .momentum import FixMomentum
from .langevin import FixLangevin

FIXES = {
    "nve":             FixNVE,
    "nvt":             FixNVT,
    "nose_hoover":     FixNVT,
    "temp/rescale":    FixTempRescale,
    "berendsen":       FixBerendsen,
    "press/berendsen": FixBerendsenBarostat,
    "momentum":        FixMomentum,
    "langevin":        FixLangevin,
}

__all__ = [
    "Fix", "FixNVE", "FixNVT", "FixTempRescale", "FixBerendsen",
    "FixBerendsenBarostat", "FixMomentum", "FixLangevin", "FIXES",
]
