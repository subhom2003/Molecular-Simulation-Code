"""Input-script package: parse and execute LAMMPS-style .in scripts."""
from .registry import CommandRegistry, registry
from .script import Script, run_script
from . import commands  # registers all command handlers

__all__ = ["Script", "run_script", "CommandRegistry", "registry"]
