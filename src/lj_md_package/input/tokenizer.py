"""
=============================================================================
LAMMPS-style input-script tokenizer
=============================================================================

A lightweight tokenizer that handles:

- Comments (everything after ``#`` is discarded).
- Line continuation using a trailing ``&``.
- ``${var}`` and ``$var`` variable substitution (shell-style, set via
  ``variable name value``).
- Whitespace splitting.

Returns a flat list of (line_no, tokens) entries with ``tokens`` the
whitespace-separated fields of each *physical* line (after continuation
merging).  Comments and blank lines are skipped.
=============================================================================
"""

from __future__ import annotations
import re
from typing import Iterator


_CONTINUATION_RE = re.compile(r"&\s*$")
_VAR_BRACE_RE = re.compile(r"\$\{([A-Za-z_]\w*)\}")
_VAR_PLAIN_RE = re.compile(r"\$([A-Za-z_]\w*)")


def _substitute_vars(line: str, variables: dict[str, str]) -> str:
    """Expand $var / ${var} using ``variables``; recurse once for nested vars."""
    def repl_brace(m):
        return str(variables.get(m.group(1), m.group(0)))
    def repl_plain(m):
        return str(variables.get(m.group(1), m.group(0)))
    for _ in range(4):  # max 4 levels of nesting
        new = _VAR_BRACE_RE.sub(repl_brace, line)
        new = _VAR_PLAIN_RE.sub(repl_plain, new)
        if new == line:
            break
        line = new
    return line


def tokenize(text: str, variables: dict[str, str] | None = None
              ) -> Iterator[tuple[int, list[str]]]:
    """Yield (line_number, tokens) for each physical line in `text`."""
    variables = variables or {}
    lines = text.splitlines()
    line_no = 0
    accumulated = ""
    start_line = 0
    for raw in lines:
        line_no += 1
        # strip comment
        hpos = raw.find("#")
        if hpos != -1:
            raw = raw[:hpos]
        raw = raw.rstrip()
        # handle continuation
        if _CONTINUATION_RE.search(raw):
            accumulated += _CONTINUATION_RE.sub("", raw)
            if start_line == 0:
                start_line = line_no
            continue
        accumulated += raw
        physical_line = accumulated.strip()
        if not physical_line:
            accumulated = ""
            continue
        physical_line = _substitute_vars(physical_line, variables)
        tokens = physical_line.split()
        out_line_no = start_line if start_line else line_no
        accumulated = ""
        start_line = 0
        yield out_line_no, tokens
    # final flush (just in case of trailing no-continuation)
    if accumulated.strip():
        physical_line = _substitute_vars(accumulated.strip(), variables)
        yield start_line or line_no, physical_line.split()
