"""Dump base class -- interval-based output of simulation state."""
from __future__ import annotations
from abc import ABC, abstractmethod


class Dump(ABC):
    """Abstract dump strategy.  Subclasses implement :meth:`write`."""

    def __init__(self, dump_id: str = "dump", group: str = "all",
                 every: int = 100, filename: str = ""):
        self.dump_id = dump_id
        self.group = group
        self.every = int(every)
        self.filename = filename
        self.step_offset = 0  # for unfix/run append behaviour

    def write_if_due(self, step: int, state) -> bool:
        """Return True if this step triggered a write."""
        if step % self.every == 0:
            self.write(step, state)
            return True
        return False

    @abstractmethod
    def write(self, step: int, state) -> None:
        ...

    def close(self) -> None:
        pass
