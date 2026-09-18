"""Computes: read-only observables (temperature, pressure, RDF, MSD, ...)."""
from .abc import Compute
from .temp import ComputeTemp
from .pressure import ComputePressure
from .rdf import ComputeRDF
from .rdf_theta import ComputeRDFTheta
from .msd import ComputeMSD
from .vacf import ComputeVACF
from .structure_factor import ComputeStructureFactor
from .conserved_energy import ComputeConservedEnergy
from .thermo_style import ThermoStyle

COMPUTES = {
    "temp":             ComputeTemp,
    "pressure":         ComputePressure,
    "rdf":              ComputeRDF,
    "rdf/theta":        ComputeRDFTheta,
    "msd":              ComputeMSD,
    "vacf":             ComputeVACF,
    "structure_factor": ComputeStructureFactor,
    "conserved/energy": ComputeConservedEnergy,
}

__all__ = [
    "Compute", "ComputeTemp", "ComputePressure", "ComputeRDF", "ComputeRDFTheta",
    "ComputeMSD", "ComputeVACF", "ComputeStructureFactor", "ComputeConservedEnergy",
    "ThermoStyle", "COMPUTES",
]
