"""Readers for the CSV files the dumper produces."""
from __future__ import annotations
import numpy as np


def _read_csv(filename: str) -> dict[str, np.ndarray]:
    """Generic CSV reader with header row, returning a dict by column name."""
    with open(filename, "r") as f:
        lines = f.readlines()
    header = lines[0].lstrip("# ").strip().split(",")
    cols = {h: [] for h in header}
    for ln in lines[1:]:
        toks = ln.strip().split(",")
        if len(toks) != len(header):
            continue
        for h, v in zip(header, toks):
            try:
                cols[h].append(float(v))
            except ValueError:
                cols[h].append(np.nan)
    return {h: np.asarray(v, dtype=float) for h, v in cols.items()}


def read_thermo_csv(filename: str) -> dict[str, np.ndarray]:
    return _read_csv(filename)


def read_energies_csv(filename: str) -> dict[str, np.ndarray]:
    return _read_csv(filename)


def read_rdf_csv(filename: str) -> dict[str, np.ndarray]:
    return _read_csv(filename)
