"""
=============================================================================
Simulation box with periodic boundary conditions
=============================================================================

Provides:
- :class:`Box`  -- a triclinic simulation box, supports cuboid (orthogonal)
  and full tilt-factor (xy, yz, xz) representations used by LAMMPS.

Methods:
- :meth:`Box.wrap`           apply periodic boundary conditions to positions
- :meth:`Box.minimum_image`  return the minimum-image displacement vector
- :meth:`Box.volume`         box volume
- :meth:`Box.from_density`   build a cubic box for a given N and rho*

The LAMMPS convention is used: a particle at exactly xhi wraps to xlo,
the half-box excludes the upper edge so a particle cannot interact with
its own image across a periodic boundary.
=============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    """
    Triclinic simulation box in LAMMPS conventions.

    Parameters
    ----------
    xlo, xhi, ylo, yhi, zlo, zhi : float
        Box edges.
    xy, yz, xz : float
        Tilt factors (xy is the displacement of the y-edge along x, etc.).
        Default 0 -> orthogonal cuboid box.

    The three box edge vectors are:
        a = (xhi - xlo, 0, 0)
        b = (xy, yhi - ylo, 0)
        c = (xz, yz, zhi - zlo)
    """

    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float
    xy: float = 0.0
    yz: float = 0.0
    xz: float = 0.0

    # ----------------------------------------------------------- constructors
    @classmethod
    def cubic(cls, L: float) -> "Box":
        """Build a cubic box of side L covering [0, L]^3."""
        return cls(0.0, float(L), 0.0, float(L), 0.0, float(L))

    @classmethod
    def from_density(cls, N: int, rho_star: float) -> "Box":
        """Build the cubic box implied by N particles at reduced density rho*."""
        V = float(N) / float(rho_star)
        L = V ** (1.0 / 3.0)
        return cls.cubic(L)

    # ------------------------------------------------------------- geometry
    @property
    def Lx(self) -> float:
        return self.xhi - self.xlo

    @property
    def Ly(self) -> float:
        return self.yhi - self.ylo

    @property
    def Lz(self) -> float:
        return self.zhi - self.zlo

    @property
    def L(self) -> float:
        """Side length for a cubic box; raises if non-cubic."""
        if not (np.isclose(self.Lx, self.Ly) and np.isclose(self.Ly, self.Lz)
                and self.xy == 0 and self.yz == 0 and self.xz == 0):
            raise ValueError("Box is non-cubic; use Lx/Ly/Lz instead of L.")
        return self.Lx

    @property
    def edges(self) -> np.ndarray:
        """Return the 3x3 matrix of box edge vectors [a, b, c] as rows."""
        a = np.array([self.Lx, 0.0, 0.0])
        b = np.array([self.xy, self.Ly, 0.0])
        c = np.array([self.xz, self.yz, self.Lz])
        return np.array([a, b, c])

    def volume(self) -> float:
        """Box volume V = a . (b x c)."""
        a, b, c = self.edges
        return float(abs(np.dot(a, np.cross(b, c))))

    # --------------------------------------------------------- PBC / images
    def wrap(self, positions: np.ndarray) -> np.ndarray:
        """Wrap positions into the primary unit cell.

        For an orthogonal box: ``p = lo + (p - lo) % L``.
        For a triclinic box the LAMMPS wrapping order is used: unwrap in x,
        then y, then z, folding each axis via the tilt factors.
        """
        p = np.asarray(positions, dtype=float).copy()
        if self.xy == 0.0 and self.yz == 0.0 and self.xz == 0.0:
            p[:, 0] = self.xlo + (p[:, 0] - self.xlo) % self.Lx
            p[:, 1] = self.ylo + (p[:, 1] - self.ylo) % self.Ly
            p[:, 2] = self.zlo + (p[:, 2] - self.zlo) % self.Lz
            return p
        # triclinic LAMMPS-style wrap
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        p[:, 2] = self.zlo + (p[:, 2] - self.zlo) % Lz
        p[:, 1] = self.ylo + (p[:, 1] + (self.yz / Lz) * (p[:, 2] - self.zlo) - self.ylo) % Ly
        p[:, 0] = self.xlo + (p[:, 0] + (self.xz / Lz) * (p[:, 2] - self.zlo)
                              + (self.xy / Ly) * (p[:, 1] - self.ylo) - self.xlo) % Lx
        return p

    def minimum_image(self, dr: np.ndarray) -> np.ndarray:
        """Apply minimum-image convention to displacement vectors.

        For an orthogonal box: ``d -= L * round(d / L)``.
        For a triclinic box: convert to fractional coords, wrap, convert back.
        Accepts both 1D (single vector) and 2D arrays.
        """
        d = np.asarray(dr, dtype=float)
        one_d = d.ndim == 1
        if one_d:
            d = d.reshape(1, 3)
        if self.xy == 0.0 and self.yz == 0.0 and self.xz == 0.0:
            d[:, 0] -= self.Lx * np.round(d[:, 0] / self.Lx)
            d[:, 1] -= self.Ly * np.round(d[:, 1] / self.Ly)
            d[:, 2] -= self.Lz * np.round(d[:, 2] / self.Lz)
        else:
            inv = np.linalg.inv(self.edges.T)
            f = d @ inv.T
            f -= np.round(f)
            d = f @ self.edges
        if one_d:
            return d.ravel()
        return d

    def divide(self) -> tuple[float, float, float]:
        """Return (Lx, Ly, Lz) tuple (convenience)."""
        return self.Lx, self.Ly, self.Lz
