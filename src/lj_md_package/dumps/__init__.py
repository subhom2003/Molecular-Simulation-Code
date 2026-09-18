"""Dumps: per-step or interval file outputs (XYZ trajectory, CSV thermo, ...)."""
from .abc import Dump
from .xyz import DumpXYZ
from .thermo import DumpThermo
from .energy import DumpEnergy
from .lammps_data import read_lammps_data, write_lammps_data

DUMPS = {
    "xyz":    DumpXYZ,
    "thermo": DumpThermo,
    "energy": DumpEnergy,
}

__all__ = ["Dump", "DumpXYZ", "DumpThermo", "DumpEnergy",
           "read_lammps_data", "write_lammps_data", "DUMPS"]
