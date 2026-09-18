"""
FixBerendsenBarostat -- isotropic Berendsen (1984) weak-coupling barostat.

``fix ID all press/berendsen P_target tau_P [beta]``

Each ``N`` steps the cubic box and all positions are scaled by

    mu = [1 - beta * (dt/tau_P) * (P_target - P(t))]^(1/3)

so that high internal pressure compresses the box (mu < 1) and vice versa.
``beta`` is an effective isothermal compressibility LJ estimate ~0.05-0.2
(in P*^-1 units); larger beta = stiffer response.

Notes
-----
- Only cubic boxes are supported.
- The box object is shared with the simulation; neighbor lists are updated
  via :meth:`NeighborList.update_box_length` and rebuilt immediately.
- Position-warping means observables that cached the box length at
  construction (RDF, S(k), ...) measure against the NEW box only after
  their next computation; use the barostat for equilibration, then unfix
  it and run production at the mean density.
- Not a canonical NPT sampler (like the Berendsen thermostat): use for
  equilibration only.

Reference: Berendsen et al., J. Chem. Phys. 81, 3684 (1984).
"""

from __future__ import annotations
import logging
import numpy as np

from .abc import Fix

logger = logging.getLogger("lj_md.fix")


class FixBerendsenBarostat(Fix):
    """Isotropic Berendsen barostat on a cubic box."""

    def __init__(self, fix_id: str = "barostat", group: str = "all",
                 P_target: float = 1.0, tau_P: float = 5.0,
                 beta: float = 0.1, N: int = 1):
        super().__init__(fix_id, group, every=int(N))
        self.P_target = float(P_target)
        self.tau_P = float(tau_P)
        self.beta = float(beta)
        self.sim = None  # set via attach()

    def name(self) -> str:
        return "press/berendsen"

    def attach(self, sim) -> None:
        """Give the fix access to the simulation (neighbor list refresh)."""
        self.sim = sim

    def do_end_of_step(self, state) -> None:
        atoms = state.atoms
        box = state.box
        if box is None:
            raise RuntimeError("Berendsen barostat needs state.box set.")
        if not (np.isclose(box.Lx, box.Ly) and np.isclose(box.Ly, box.Lz)):
            raise RuntimeError("Berendsen barostat supports cubic boxes only.")

        fr = state.force_result
        if fr is None:
            return
        N = atoms.N
        V = box.volume()
        # instantaneous pressure from the latest force evaluation
        P_inst = N * state.temperature / V + fr.virial / (3.0 * V)

        dt = state.dt
        x = 1.0 - self.beta * (dt / self.tau_P) * (self.P_target - P_inst)
        if x <= 0.0:
            # way out of the weak-coupling regime; clamp gently
            logger.warning("Berendsen barostat scale factor invalid (x=%g); "
                           "clamping to 0.999.", x)
            x = 0.999
        mu = x ** (1.0 / 3.0)
        if abs(mu - 1.0) < 1e-12:
            return

        # scale box + positions about the lower corner (xlo,ylo,zlo)
        lo = np.array([box.xlo, box.ylo, box.zlo])
        atoms.positions = lo + (atoms.positions - lo) * mu
        box.xhi = box.xlo + (box.xhi - box.xlo) * mu
        box.yhi = box.ylo + (box.yhi - box.ylo) * mu
        box.zhi = box.zlo + (box.zhi - box.zlo) * mu
        atoms.positions = box.wrap(atoms.positions)

        # update neighbor list geometry and force a rebuild
        if self.sim is not None and self.sim.neighbor_list is not None:
            self.sim.neighbor_list.update_box_length(box.Lx)
            self.sim.neighbor_list.build(atoms.positions)
        # unwrapped positions track the affine scaling too
        if state.unwrapped is not None:
            state.unwrapped = lo + (state.unwrapped - lo) * mu
