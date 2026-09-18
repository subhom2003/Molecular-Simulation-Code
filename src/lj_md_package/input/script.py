"""
=============================================================================
Script -- LAMMPS-style input-script runner
=============================================================================

A :class:`Script` holds:

- a :class:`Simulation` object (created lazily)
- ``variables`` dict for ``variable`` and ``${var}`` substitution
- a registry of available commands

The :meth:`run` method reads a text file line-by-line, tokenizes each
physical line via :func:`input.tokenizer.tokenize`, looks up the command
name in :data:`registry`, and dispatches it.

:func:`run_script` is a convenience API: open the file, run it, finish
the simulation.
=============================================================================
"""

from __future__ import annotations
import os
from pathlib import Path

from .registry import registry
from .tokenizer import tokenize
from . import commands as _commands  # noqa: F401  (ensures registration)
from ..core.simulation import Simulation
from ..core.config import SimConfig


class Script:
    """LAMMPS-style input-script execution context."""

    def __init__(self):
        from ..core.simulation import Simulation as S
        self.cfg = SimConfig()
        self.sim = S(self.cfg)
        self.variables: dict[str, str] = {}
        self._run_count = 0
        self._equil_n_thermostat_off = 0
        # flags to track user-set fields
        self._configured_units = False
        self._atoms_set = False
        self._pair_set = False
        self._neighbor_set = False
        self._box_set = False
        self._vec_compress_counts = 4
        self._custom_thermo_keys = None

    # ------------------------------------------------------ public methods
    def set_variable(self, name: str, value: str) -> None:
        self.variables[name] = str(value)

    def get_variable(self, name: str) -> str | None:
        return self.variables.get(name)

    # The run loop:
    def run_string(self, text: str) -> None:
        for line_no, tokens in tokenize(text, self.variables):
            if not tokens:
                continue
            cmd = tokens[0]
            try:
                registry.dispatch(cmd, self, tokens[1:])
            except Exception as e:
                raise type(e)(f"line {line_no}: {cmd}: {e}") from e

    def run_file(self, filename: str) -> None:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Input script not found: {filename}")
        text = path.read_text()
        # include-file support: rebase relative includes against this dir
        self._script_dir = path.parent
        self.run_string(text)

    def finish(self) -> None:
        """Finalize: close dumps, write RDF/MSD/VACF CSVs and summary.txt."""
        import time
        # wall time is unknown in this context; use 0
        self.sim.finish(0.0)


# ---------------------------------------------------------------------------
def run_script(filename: str) -> Simulation:
    """Convenience wrapper: parse & run an input script, return Simulation."""
    s = Script()
    s.run_file(filename)
    # force a final call to ensure summary files are populated if no run() was issued
    from ..core.simulation import Simulation as Sim
    if isinstance(s.sim, Sim):
        try:
            s.sim.finish(getattr(s, "_wall_time", 0.0))
        except Exception:
            pass
    return s.sim
