"""
Command registry for the input-script parser.

Commands register themselves with the module-level :data:`registry` via the
:func:`register` decorator, e.g.::

    @registry.register("units")
    def cmd_units(script, args):
        ...

At execution time the dispatcher looks up the command name and calls
``handler(script, args)`` with the *remaining* tokens after the command name.
"""

from __future__ import annotations
from typing import Callable


class CommandRegistry:
    def __init__(self) -> None:
        self._table: dict[str, Callable] = {}

    def register(self, name: str) -> Callable[[Callable], Callable]:
        def decorator(fn: Callable) -> Callable:
            self._table[name] = fn
            return fn
        return decorator

    def dispatch(self, name: str, script, args: list[str]):
        try:
            cmd = self._table[name]
        except KeyError:
            raise KeyError(f"Unknown input-script command: {name!r}")
        return cmd(script, args)

    def names(self):
        return list(self._table.keys())


registry = CommandRegistry()
