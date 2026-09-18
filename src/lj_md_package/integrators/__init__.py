"""Integrator strategies for the engine."""
from .abc import Integrator
from .velocity_verlet import VelocityVerlet
from .minimize import MinimizeSD

INTEGRATOR_STYLES = {
    "velocity_verlet": VelocityVerlet,
    "minimize_sd":     MinimizeSD,
}

__all__ = ["Integrator", "VelocityVerlet", "MinimizeSD", "INTEGRATOR_STYLES"]
