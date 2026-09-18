"""
=============================================================================
lj_md_package — a LAMMPS-style Lennard-Jones molecular dynamics engine
=============================================================================

A pure-Python MD simulator for monatomic Lennard-Jones fluids that can
be driven either by a Pythonic builder API or by LAMMPS-style input
scripts (units/pair_style/fix/compute/dump/run/...).

Top-level public symbols (most users only need these)
-----------------------------------------------------
- :class:`SimConfig`            central configuration dataclass
- :class:`Simulation`           high-level run-loop facade
- :class:`Script`               LAMMPS-style input-script parser
- :func:`run_script`            convenience: parse & execute an ``.in`` file
- :func:`benchmark`             one-shot Verlet (1967) benchmark
- :data:`LITERATURE_LJ`         literature reference values for the benchmark

Examples
--------
Programmatic:
    >>> from lj_md_package import SimConfig, Simulation
    >>> sim = Simulation(SimConfig(n_steps=200, n_equil=50))
    >>> sim.run()

Input script (``in.lj``):
    units         lj
    lattice       fcc 0.8442
    create_atoms  1 box
    pair_style    lj/cut 2.5
    pair_coeff    1 1 1.0 1.0
    neighbor      0.3 bin
    velocity      all create 0.722 42
    fix           1 all nve
    fix           2 all temp/rescale 100 0.722 0.722 0.1 1.0
    thermo        10
    dump          1 all xyz 200 output/trajectory.xyz
    run           5000
    unfix         2
    run           15000

Run from the shell:
    $ lj-md in.lj
    $ lj-md --benchmark
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = [
    "__version__",
    "SimConfig",
    "Atoms",
    "Box",
    "SimulationState",
    "Simulation",
    "Script",
    "run_script",
    "benchmark",
    "LITERATURE_LJ",
    # Fix/Compute/Dump base classes
    "Fix", "Compute", "Dump",
    # Common fixes / computes / dumps
    "FixNVE", "FixNVT", "FixTempRescale", "FixBerendsen", "FixMomentum", "FixLangevin",
    "ComputeTemp", "ComputePressure", "ComputeRDF", "ComputeRDFTheta", "ComputeMSD", "ComputeVACF",
    "ComputeStructureFactor", "ComputeConservedEnergy", "ThermoStyle",
    "DumpXYZ", "DumpThermo", "DumpEnergy",
    # Pair styles + neighbor lists
    "PairStyle", "LJCut", "SoftPair",
    "NeighborList", "VerletNL", "CellList",
    # Integrators
    "Integrator", "VelocityVerlet", "MinimizeSD",
    # Units
    "UnitSystem", "LJUnits", "RealUnits", "MetalUnits",
    # Utilities
    "setup_log",
]

# --- Re-export the public surface -----------------------------------------
from .core.config import SimConfig
from .core.box import Box
from .core.atoms import Atoms
from .core.state import SimulationState
from .core.simulation import Simulation
from .core.logging_setup import setup_log

from .fixes.abc import Fix
from .fixes.nve import FixNVE
from .fixes.nvt_nose_hoover import FixNVT
from .fixes.temp_rescale import FixTempRescale
from .fixes.berendsen import FixBerendsen
from .fixes.momentum import FixMomentum
from .fixes.langevin import FixLangevin

from .computes.abc import Compute
from .computes.temp import ComputeTemp
from .computes.pressure import ComputePressure
from .computes.rdf import ComputeRDF
from .computes.rdf_theta import ComputeRDFTheta
from .computes.msd import ComputeMSD
from .computes.vacf import ComputeVACF
from .computes.structure_factor import ComputeStructureFactor
from .computes.conserved_energy import ComputeConservedEnergy
from .computes.thermo_style import ThermoStyle

from .dumps.abc import Dump
from .dumps.xyz import DumpXYZ
from .dumps.thermo import DumpThermo
from .dumps.energy import DumpEnergy

from .pairs.abc import PairStyle
from .pairs.lj_cut import LJCut
from .pairs.soft import SoftPair

from .neighbors.abc import NeighborList
from .neighbors.verlet import VerletNL
from .neighbors.cell_list import CellList

from .integrators.abc import Integrator
from .integrators.velocity_verlet import VelocityVerlet
from .integrators.minimize import MinimizeSD

from .units.abc import UnitSystem
from .units.lj import LJUnits
from .units.real import RealUnits
from .units.metal import MetalUnits
from .units.reference import LITERATURE_LJ

from .input.script import Script, run_script
from .cli import benchmark
from .plotting import generate_all_plots
