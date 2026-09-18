"""
Idempotent logging setup for the lj_md package.
=============================================================================

Uses a *named* logger ``"lj_md"`` so we never touch the root logger or other
libraries.  Multiple :func:`setup_log` calls replace any prior handlers we
added (so re-running inside the same Python process -- e.g. a temperature
scan -- does not stack duplicate handlers and produce doubled log lines).
"""

from __future__ import annotations
import logging
from pathlib import Path

_LJ_MD_LOGGER_NAME = "lj_md"
_LJ_MD_TAG = "_lj_md_handler"


def setup_log(log_file: str | None = "output/simulation.log",
              level: int = logging.INFO) -> logging.Logger:
    """Configure the lj_md named logger. Idempotent: tag handlers, replace on recall."""
    logger = logging.getLogger(_LJ_MD_LOGGER_NAME)
    logger.setLevel(level)
    # remove our previously-added handlers only (keep others -- e.g. pytest)
    for h in list(logger.handlers):
        if getattr(h, _LJ_MD_TAG, False):
            logger.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        setattr(fh, _LJ_MD_TAG, True)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    setattr(ch, _LJ_MD_TAG, True)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
