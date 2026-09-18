"""
Read/write LAMMPS-style ``data`` files (Atoms/Velocities/Masses sections).
=============================================================================

Supports the LAMMPS ``read_data`` command syntax for monatomic and simple
multi-type systems.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from ..core.atoms import Atoms
from ..core.box import Box


def read_lammps_data(filename: str) -> tuple[Atoms, Box, dict]:
    """Parse a LAMMPS-data file and return (Atoms, Box, header_dict)."""
    header = {}
    atoms_lines = []
    vel_lines = []
    mass_lines = []
    section = None
    with open(filename, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # header key: count of "<X> <i>"
            if section is None:
                # might be header
                lower = stripped.lower()
                found = False
                for key in ("atoms", "atom types", "bonds", "bond types",
                            "angles", "angle types",
                            "xlo xhi", "ylo yhi", "zlo zhi",
                            "xy", "yz", "xz"):
                    if lower.endswith(key):
                        try:
                            header[key] = [float(x) for x in stripped[:-(len(key))].split()]
                        except ValueError:
                            pass
                        found = True
                        break
                if found:
                    continue
                if stripped in ("Masses", "Atoms", "Velocities",
                                "Atoms # atomic", "Atoms # full"):
                    section = stripped.split()[0]
                    continue
            else:
                if stripped in ("Masses", "Atoms", "Velocities",
                                "Atoms # atomic", "Atoms # full"):
                    section = stripped.split()[0]
                    continue
                if section == "Masses":
                    mass_lines.append(stripped.split())
                elif section == "Atoms":
                    atoms_lines.append(stripped.split())
                elif section == "Velocities":
                    vel_lines.append(stripped.split())
    # Build Box
    xlo, xhi = header.get("xlo xhi", [0.0, 1.0])
    ylo, yhi = header.get("ylo yhi", [0.0, 1.0])
    zlo, zhi = header.get("zlo zhi", [0.0, 1.0])
    xy = header.get("xy", [0.0])[0]
    yz = header.get("yz", [0.0])[0]
    xz = header.get("xz", [0.0])[0]
    box = Box(xlo, xhi, ylo, yhi, zlo, zhi, xy=xy, yz=yz, xz=xz)
    # Build Atoms
    atoms_arr = np.array(atoms_lines, dtype=float)
    N = atoms_arr.shape[0]
    ids = atoms_arr[:, 0].astype(int)
    order = np.argsort(ids)
    atoms_arr = atoms_arr[order]
    # assume "atomic" style: id type x y z
    types = atoms_arr[:, 1].astype(int)
    positions = atoms_arr[:, 2:5]
    if atoms_arr.shape[1] >= 7:
        # full style: id mol type charge x y z
        types = atoms_arr[:, 2].astype(int)
        positions = atoms_arr[:, 4:7]
    velocities = np.zeros_like(positions)
    if vel_lines:
        v_arr = np.array(vel_lines, dtype=float)
        v_arr = v_arr[np.argsort(v_arr[:, 0].astype(int))]
        velocities = v_arr[:, 1:4]
    # Masses
    masses = np.ones(N, dtype=float)
    if mass_lines:
        mass_table = {int(r[0]): float(r[1]) for r in mass_lines}
        for i, t in enumerate(types):
            masses[i] = mass_table.get(t, 1.0)
    atoms = Atoms(positions=positions, velocities=velocities,
                  types=types, masses=masses, ids=ids)
    return atoms, box, header


def write_lammps_data(filename: str, atoms: Atoms, box: Box,
                       atom_style: str = "atomic") -> None:
    """Write a LAMMPS-data file from an Atoms+Box snapshot."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    N = atoms.N
    n_types = int(atoms.types.max())
    with open(filename, "w") as f:
        f.write(f"# LJ-MD data file -- {N} atoms\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"{box.xlo} {box.xhi} xlo xhi\n")
        f.write(f"{box.ylo} {box.yhi} ylo yhi\n")
        f.write(f"{box.zlo} {box.zhi} zlo zhi\n")
        if box.xy or box.yz or box.xz:
            f.write(f"{box.xy} {box.yz} {box.xz} xy yz xz\n")
        f.write("\nMasses\n\n")
        unique_types = sorted(set(atoms.types.tolist()))
        for t in unique_types:
            m = atoms.masses[atoms.types == t][0]
            f.write(f"{t} {m}\n")
        f.write("\nAtoms # atomic\n\n")
        for i in range(N):
            f.write(f"{atoms.ids[i]} {atoms.types[i]} "
                    f"{atoms.positions[i, 0]:.10f} "
                    f"{atoms.positions[i, 1]:.10f} "
                    f"{atoms.positions[i, 2]:.10f}\n")
        f.write("\nVelocities\n\n")
        for i in range(N):
            f.write(f"{atoms.ids[i]} {atoms.velocities[i, 0]:.10f} "
                    f"{atoms.velocities[i, 1]:.10f} "
                    f"{atoms.velocities[i, 2]:.10f}\n")
