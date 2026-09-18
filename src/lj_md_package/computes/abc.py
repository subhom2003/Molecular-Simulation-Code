"""Compute base class -- read-only observables computed each step (or every N)."""

from __future__ import annotations
from abc import ABC, abstractmethod


class Compute(ABC):
    """Abstract base class for all computes.

    A compute is invoked each integration step (or every ``every``) and is
    expected to read the state and store results internally.  A compute must
    not modify the state -- that is the job of a :class:`Fix`.
    """

    compute_id: str = "compute"
    group: str = "all"

    def __init__(self, compute_id: str = "compute", group: str = "all",
                 every: int = 1, start: int = 0):
        self.compute_id = compute_id
        self.group = group
        self.every = int(every)
        #: first step at which this compute accumulates data (equilibration
        #: gating; thermostatted frames must not contaminate observables).
        self.start = int(start)

    @abstractmethod
    def name(self) -> str: ...

    def compute_if_due(self, state, force_result) -> None:
        if state.step >= self.start and state.step % self.every == 0:
            self.compute(state, force_result)

    @abstractmethod
    def compute(self, state, force_result) -> None:
        """Read state and update internal accumulators."""
