"""Integrator base class -- exposes start_of_step / end_of_step hooks."""

from __future__ import annotations
from abc import ABC, abstractmethod


class Integrator(ABC):
    """Integrator strategy.frame state-less helpers wrapped as a class."""

    @abstractmethod
    def initial_integrate(self, state) -> None:
        """First velocity-Verlet half-step: update positions and half-step velocities."""

    @abstractmethod
    def final_integrate(self, state) -> None:
        """Second velocity-Verlet half-step: complete the velocity update."""
