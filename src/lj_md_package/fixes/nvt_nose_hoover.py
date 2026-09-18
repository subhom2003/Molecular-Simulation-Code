"""
=============================================================================
FixNVT -- Nose-Hoover canonical (NVT) thermostat
=============================================================================

Single-bath Nose-Hoover thermostat with a time-reversible Strang
(Martyna-Tuckerman) splitting around a velocity-Verlet step:

    xi  += (dt/4) * G(K)  / Q          # quarter step, G = 2K - dof*k_B*T
    v   *= exp(-xi * dt/2)             # half-step velocity scaling
    --- velocity-Verlet full step (v half-kick, drift, force, half-kick)
    v   *= exp(-xi * dt/2)             # mirrored half-step scaling
    xi  += (dt/4) * G(K_new) / Q       # quarter step

This palindromic form is second-order accurate and time-reversible
(replacing the previous asymmetric first-order split).  The conserved
quantity is the extended Hamiltonian

    H_NH = K + U + (1/2) Q xi^2 + dof * k_B T * eta,     eta = integral xi dt

``conserved_energy()`` returns ONLY the bath terms; KE+PE are accounted by
the caller (see :class:`ComputeConservedEnergy`).

References
----------
- Wm. G. Hoover, Phys. Rev. A 31, 1695 (1985).
- G. J. Martyna, M. L. Klein, M. E. Tuckerman, J. Chem. Phys. 97, 2635 (1992).
=============================================================================
"""

from __future__ import annotations
import numpy as np

from .abc import Fix


class FixNVT(Fix):
    """Nose-Hoover NVT fix.  Works alongside a FixNVE's velocity-Verlet step."""

    def __init__(self, fix_id: str = "nvt", group: str = "all",
                 T_target: float = 1.0, Q: float = 2.0, dof: int | None = None):
        super().__init__(fix_id, group)
        self.T_target = float(T_target)
        self.Q = float(Q)
        self.dof = dof
        self.xi = 0.0          # thermostat friction variable
        self.eta = 0.0         # time-integral of xi (log of bath coordinate s)
        self._dof_cache: int | None = None

    def name(self) -> str:
        return "nvt"

    # ----------------------------------------------------------- helpers
    def _dof(self, state) -> int:
        if self.dof is not None:
            return self.dof
        if self._dof_cache is None:
            self._dof_cache = state.atoms.dof
        return self._dof_cache

    def _G(self, KE: float, dof: int) -> float:
        """G = 2 K - dof * k_B T_target   (k_B = 1 in LJ units)."""
        return 2.0 * KE - dof * self.T_target

    # ----------------------------------------- thermostat half-steps
    def initial_integrate(self, state) -> None:
        """First thermostat half-step (before the VV step)."""
        atoms = state.atoms
        dt = state.dt
        dof = self._dof(state)
        KE = atoms.kinetic_energy()
        self.xi += 0.25 * dt * self._G(KE, dof) / self.Q
        scale = np.exp(-0.5 * self.xi * dt)
        atoms.velocities *= scale
        self.eta += 0.5 * self.xi * dt

    def final_integrate(self, state) -> None:
        """Mirrored thermostat half-step (after the VV step)."""
        atoms = state.atoms
        dt = state.dt
        scale = np.exp(-0.5 * self.xi * dt)
        atoms.velocities *= scale
        dof = self._dof(state)
        KE_new = atoms.kinetic_energy()
        self.xi += 0.25 * dt * self._G(KE_new, dof) / self.Q
        self.eta += 0.5 * self.xi * dt
        # cache temperature for thermo output
        state.kinetic_energy = KE_new
        state.temperature = 2.0 * KE_new / dof

    # ------------------------------------------- conserved extended Hamiltonian
    def conserved_energy(self, state) -> float:
        """Bath contribution only:  (1/2) Q xi^2 + dof * T_target * eta."""
        dof = self._dof(state)
        return 0.5 * self.Q * self.xi ** 2 + dof * self.T_target * self.eta

    def diagnostics(self) -> str:
        return f"xi={self.xi:+.6f}  eta={self.eta:+.6f}"
