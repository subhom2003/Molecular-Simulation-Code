"""
=============================================================================
Simulation -- high-level facade and the LAMMPS-style event loop
=============================================================================

The :class:`Simulation` object collects:

- ``cfg``     : :class:`SimConfig`
- ``box``     : :class:`Box`
- ``atoms``   : :class:`Atoms` (positions, velocities, types, masses)
- ``pair_style`` :class:`PairStyle`
- ``neighbor_list`` :class:`NeighborList`
- ``fixes``   : list of :class:`Fix`  (including the integrator as FixNVE/NVT)
- ``computes``: list of :class:`Compute`
- ``dumps``   : list of :class:`Dump`

The :meth:`run` method iterates the standard LAMMPS event loop::

    for step in [1, n_steps]:
        for f in self.fixes:  f.initial_integrate(state)        # VV step 1
        if neighbor_list.needs_rebuild:    rebuild
        force_result = pair_style.compute(atoms, neighbor_list, box)
        state.force_result = force_result
        for f in self.fixes:  f.post_force(state)               # thermostat kicks
        for f in self.fixes:  f.final_integrate(state)          # VV step 2
        state.step = step
        state.kinetic_energy = atoms.kinetic_energy()
        state.temperature   = 2 KE / dof
        for f in self.fixes:  f.end_of_step(state)              # periodic ops
        state._fix_conserved = sum fix.conserved_energy(state)
        for c in self.computes: c.compute_if_due(state, fr)
        for d in self.dumps:   d.write_if_due(step, state)
        progress()
    finish()

Backward-compatibility helpers (:meth:`from_config`) produce a default
Simulation matching the legacy Pythonic API.
=============================================================================
"""

from __future__ import annotations
import time
import logging
import numpy as np

from .config import SimConfig
from .box import Box
from .atoms import Atoms
from .state import SimulationState

from ..pairs.lj_cut import LJCut
from ..pairs.soft import SoftPair
from ..neighbors.verlet import VerletNL
from ..neighbors.cell_list import CellList
from ..integrators.velocity_verlet import VelocityVerlet
from ..integrators.minimize import MinimizeSD
from ..fixes.abc import Fix
from ..fixes.nve import FixNVE
from ..fixes.temp_rescale import FixTempRescale
from ..fixes.berendsen import FixBerendsen
from ..fixes.momentum import FixMomentum
from ..computes.abc import Compute
from ..computes.rdf import ComputeRDF
from ..computes.rdf_theta import ComputeRDFTheta
from ..computes.msd import ComputeMSD
from ..computes.vacf import ComputeVACF
from ..computes.structure_factor import ComputeStructureFactor
from ..computes.conserved_energy import ComputeConservedEnergy
from ..computes.thermo_style import ThermoStyle
from ..dumps.abc import Dump

from ..initialize import initialize_system
from ..analysis.summary import build_validation_report

logger = logging.getLogger("lj_md.sim")


def _make_neighbor_list(cfg: SimConfig):
    """Build a neighbor list honoring cfg.neighbor_style, falling back to
    VerletNL when the box is too small for a safe CellList (< 3 cells/dim -
    the half-stencil would double-count pairs there)."""
    if cfg.neighbor_style == "cell":
        if cfg.L / cfg.r_list >= 3.0:
            return CellList(cfg.r_cut, cfg.r_skin, cfg.L)
        logger.info(
            "neighbor_style='cell' requested but L/r_list=%.2f < 3 -> "
            "using VerletNL (cell list would double-count pairs at this size).",
            cfg.L / cfg.r_list,
        )
    return VerletNL(cfg.r_cut, cfg.r_skin, cfg.L)


def _make_thermostat_fix(cfg: SimConfig):
    """Construct the equilibration thermostat fix selected by
    ``cfg.thermostat_type`` (fix id ``"thermostat"``), or None."""
    from ..fixes.langevin import FixLangevin
    from ..fixes.nvt_nose_hoover import FixNVT

    t = cfg.thermostat_type
    if t == "none":
        return None
    if t == "rescale":
        return FixTempRescale(fix_id="thermostat", group="all",
                              N=cfg.rescale_interval,
                              Tstart=cfg.T_star, Tstop=cfg.T_star,
                              fraction=1.0, window=0.0)
    if t == "berendsen":
        return FixBerendsen(fix_id="thermostat", group="all",
                            N=1, Tstart=cfg.T_star, Tstop=cfg.T_star,
                            tau_T=cfg.tau_T)
    if t == "nose_hoover":
        return FixNVT(fix_id="thermostat", group="all",
                      T_target=cfg.T_star, Q=cfg.nose_hoover_Q)
    if t == "langevin":
        # BAOAB: self-integrating thermostat; must REPLACE the NVE fix.
        fix = FixLangevin(fix_id="thermostat", group="all",
                          T_target=cfg.T_star, gamma=cfg.langevin_gamma,
                          seed=cfg.random_seed)
        return fix
    raise ValueError(f"Unknown thermostat_type {t!r}")


class Simulation:
    """High-level facade wrapping the LAMMPS-style event loop."""

    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.box: Box | None = None
        self.atoms: Atoms | None = None
        self.pair_style = None
        self.neighbor_list = None
        self.integrator = None

        self.fixes: list[Fix] = []
        self.computes: list[Compute] = []
        self.dumps: list[Dump] = []

        # bookkeeping
        self.state = SimulationState(dt=self.cfg.dt)
        self._n_total_steps = 0  # we keep going across multiple run blocks
        self._thermo: ThermoStyle | None = None
        self._equilibrated = False
        self._n_equil_in_current = 0
        self._n_prod_in_current = 0
        self._run_E0: float | None = None

    # ==================================================== setup hooks
    def set_box(self, box: Box) -> None:
        self.box = box
        self.state.box = box

    def set_atoms(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self.state.atoms = atoms
        self.state.unwrapped = atoms.positions.copy()

    def set_pair_style(self, ps) -> None:
        self.pair_style = ps

    def set_neighbor_list(self, nl) -> None:
        self.neighbor_list = nl

    def set_integrator(self, integ) -> None:
        self.integrator = integ

    def add_fix(self, fix: Fix) -> None:
        self.fixes.append(fix)

    def remove_fix(self, fix_id: str) -> bool:
        for i, f in enumerate(self.fixes):
            if f.fix_id == fix_id:
                self.fixes.pop(i)
                return True
        return False

    def add_compute(self, comp: Compute) -> None:
        self.computes.append(comp)

    def remove_compute(self, compute_id: str) -> bool:
        for i, c in enumerate(self.computes):
            if c.compute_id == compute_id:
                self.computes.pop(i)
                return True
        return False

    def add_dump(self, dump: Dump) -> None:
        self.dumps.append(dump)

    def remove_dump(self, dump_id: str) -> bool:
        for i, d in enumerate(self.dumps):
            if d.dump_id == dump_id:
                d.close()
                self.dumps.pop(i)
                return True
        return False

    # ===================================================== setup helpers
    @classmethod
    def from_config(cls, cfg: SimConfig | None = None) -> "Simulation":
        """Convenience factory: replicate the legacy Pythonic default
        Simulation (fcc lattice, LJ-cut pair, Verlet neighbor list,
        rescale-during-equil thermostat, default computes+dumps)."""
        cfg = cfg or SimConfig()
        cfg.validate()
        sim = cls(cfg)

        # box + initial atoms
        sim.set_box(Box.cubic(cfg.L))
        positions, velocities = initialize_system(cfg)
        atoms = Atoms(positions=positions, velocities=velocities)
        atoms._box_ref = sim.box
        sim.set_atoms(atoms)

        # pair style + neighbor list
        sim.set_pair_style(LJCut(
            r_cut=cfg.r_cut, shift=cfg.shift_potential, tail=cfg.use_tail_corrections
        ))
        sim.pair_style._u_tail_per_atom = cfg.u_tail
        sim.pair_style._N = cfg.N

        sim.set_neighbor_list(_make_neighbor_list(cfg))
        sim.neighbor_list.build(atoms.positions)

        # integrator + integration fix (so the run loop actually integrates)
        sim.set_integrator(VelocityVerlet(cfg.dt, sim.box))

        # equilibration thermostat (id "thermostat"; remove it for NVE production)
        thermo_fix = _make_thermostat_fix(cfg)
        from ..fixes.langevin import FixLangevin as _Lang
        if isinstance(thermo_fix, _Lang):
            # BAOAB Langevin integrates itself - it must be the only
            # integrating fix (no FixNVE alongside).
            thermo_fix.set_integrator(sim.integrator)
            sim.add_fix(thermo_fix)
        else:
            sim.add_fix(FixNVE(fix_id="nve", group="all", integrator=sim.integrator))
            if thermo_fix is not None:
                sim.add_fix(thermo_fix)

        # initial force evaluation
        fr = sim.pair_style.compute(sim.atoms, sim.neighbor_list, sim.box)
        sim.state.force_result = fr
        sim.state.kinetic_energy = atoms.kinetic_energy()
        sim.state.temperature = 2.0 * sim.state.kinetic_energy / atoms.dof

        # cross-check parity: neighbor-list vs direct force kernel
        # (brute O(N^2) reference - skip if N too large)
        if cfg.N <= 512:
            fr_ref = sim.pair_style.compute(sim.atoms, None, sim.box)
            max_diff = float(np.abs(fr.forces - fr_ref.forces).max())
            if not np.allclose(fr.forces, fr_ref.forces, atol=1e-8):
                logger.warning(
                    "Neighbor-list vs direct force kernel disagree (max=%g); "
                    "expected at float precision.", max_diff
                )
            if max_diff > 1e-6:
                raise RuntimeError(
                    f"Force kernel divergence {max_diff:.3g} > 1e-6; "
                    f"neighbor list may be corrupt."
                )

        # safety check on initial force magnitude
        if fr.max_f > cfg.max_initial_force:
            raise RuntimeError(
                f"Initial max force {fr.max_f:.1f} > {cfg.max_initial_force}: "
                f"use lattice_type='fcc' or pair_style soft+minimize."
            )

        # thermodynamic recorder (always on)
        sim._thermo = ThermoStyle(
            rho_star=cfg.rho_star, P_tail=cfg.P_tail,
            u_tail=cfg.u_tail, n_equil=cfg.n_equil
        )
        sim._thermo.set_atom_count(cfg.N)

        # default computes (accumulate only after production starts)
        sim.add_compute(sim._thermo)
        sim.add_compute(ComputeRDF(every=cfg.rdf_interval, n_bins=cfg.rdf_n_bins,
                                    L=cfg.L, N=cfg.N, start=cfg.rdf_start))
        sim.add_compute(ComputeMSD(every=cfg.msd_interval, max_length=cfg.msd_length,
                                    dt=cfg.dt, start=cfg.msd_start))
        sim.add_compute(ComputeVACF(every=cfg.vacf_interval, max_length=cfg.vacf_length,
                                     dt=cfg.dt, start=cfg.vacf_start))
        sim.add_compute(ComputeStructureFactor(every=max(200, cfg.sample_interval * 4),
                                                L=cfg.L, N=cfg.N, start=cfg.rdf_start))
        sim.add_compute(ComputeRDFTheta(every=cfg.rdf_interval,
                                         n_r_bins=cfg.rdf_n_bins // 2,
                                         n_cos_bins=50,
                                         L=cfg.L, N=cfg.N, start=cfg.rdf_start))

        return sim

    # ===================================================== run loop
    def run(self, n_steps: int) -> None:
        """Run n_steps additional integration steps."""
        if self.atoms is None or self.pair_style is None:
            raise RuntimeError("Simulation.run() called before setup")

        self.state.dt = self.cfg.dt
        start_step = self._n_total_steps
        end_step = start_step + n_steps

        # sanity: exactly one time-integrating fix (FixNVE or self-
        # integrating thermostat like FixLangevin).
        from ..fixes.nve import FixNVE as _FNVE
        from ..fixes.langevin import FixLangevin as _FLang
        integrating = [f for f in self.fixes if isinstance(f, (_FNVE, _FLang))]
        if len(integrating) == 0:
            raise RuntimeError(
                "No integrating fix (nve / langevin) present - the system "
                "would not move. Add e.g. FixNVE('nve','all',sim.integrator)."
            )
        if len(integrating) > 1:
            raise RuntimeError(
                f"Multiple integrating fixes present "
                f"({[f.fix_id for f in integrating]}): positions would be "
                f"updated twice per step. Keep exactly one."
            )

        # let ramped fixes learn the (start, length) of THIS run block
        for f in self.fixes:
            hook = getattr(f, "on_run_start", None)
            if callable(hook):
                hook(start_step, n_steps)

        # segment-local energy reference for the progress drift readout
        if self.state.force_result is not None:
            self._run_E0 = self.state.kinetic_energy + self.state.force_result.potential
        else:
            self._run_E0 = None

        is_equilibration = not self._equilibrated
        if is_equilibration:
            self._n_equil_in_current = n_steps
        else:
            self._n_prod_in_current = n_steps

        # ensure a fresh rebuild at run start
        if self.neighbor_list is not None and self.neighbor_list.needs_rebuild(self.atoms.positions):
            self.neighbor_list.build(self.atoms.positions)

        logger.info(
            "run: %d steps (start=%d, end=%d) %s",
            n_steps, start_step, end_step,
            "EQUILIBRATION" if is_equilibration else "PRODUCTION"
        )

        t_start = time.perf_counter()
        prev_progress_step = start_step

        for step in range(start_step + 1, end_step + 1):
            self._n_total_steps = step

            # ----- LAMMPS event loop -----
            for f in self.fixes:
                f.initial_integrate(self.state)

            # rebuild neighbor list if needed
            if self.neighbor_list.update_if_needed(self.atoms.positions):
                pass

            # forces
            fr = self.pair_style.compute(self.atoms, self.neighbor_list, self.box)
            self.state.force_result = fr
            # update cached KE/T before post_force thermostats operate
            self.state.kinetic_energy = self.atoms.kinetic_energy()
            self.state.temperature = 2.0 * self.state.kinetic_energy / self.atoms.dof

            for f in self.fixes:
                f.post_force(self.state)

            for f in self.fixes:
                f.final_integrate(self.state)
            # refresh KE/T after velocity update
            self.state.kinetic_energy = self.atoms.kinetic_energy()
            self.state.temperature = 2.0 * self.state.kinetic_energy / self.atoms.dof

            self.state.step = step
            self.state.time = step * self.cfg.dt

            for f in self.fixes:
                f.end_of_step(self.state)

            # capture conserved contributions from any thermostat-style fix
            self.state._fix_conserved = sum(
                f.conserved_energy(self.state) for f in self.fixes
                if hasattr(f, "conserved_energy")
            )

            for c in self.computes:
                c.compute_if_due(self.state, fr)

            for d in self.dumps:
                d.write_if_due(step, self.state)

            # progress
            if step - prev_progress_step >= self.cfg.output_interval:
                self._progress(step, t_start)
                prev_progress_step = step

        self._progress(end_step, t_start, final=True)
        # if equilibration just finished, switch equilibrated=True for next run
        if is_equilibration:
            self._equilibrated = True

    # ===================================================== internals
    def _progress(self, step: int, t0: float, final: bool = False) -> None:
        fr = self.state.force_result
        atoms = self.atoms
        KE = self.state.kinetic_energy
        T = self.state.temperature
        PE = fr.potential if fr is not None else 0.0
        E = KE + PE
        # drift is measured against the first sample of the CURRENT run
        # segment - comparing against a thermostatted phase is meaningless
        E0 = self._run_E0 if getattr(self, "_run_E0", None) is not None else E
        drift = abs(E - E0) / abs(E0) if abs(E0) > 0 else 0.0
        n_builds = self.neighbor_list.n_builds if self.neighbor_list else 0
        elapsed = time.perf_counter() - t0
        # Compute pressure from virial
        if fr is not None and self.cfg.rho_star > 0:
            V = self.cfg.N / self.cfg.rho_star
            P = self.cfg.rho_star * T + fr.virial / (3.0 * V) + self.cfg.P_tail
        else:
            P = 0.0
        if final:
            logger.info(
                "[%6d] T=%.4f P=%+.4f E=%.4f drift=%+.2e builds=%d  (elapsed %.2fs)",
                step, T, P, E, drift, n_builds, elapsed
            )
        else:
            logger.info(
                "[%6d] T=%.4f P=%+.4f E=%.4f drift=%+.2e builds=%d",
                step, T, P, E, drift, n_builds
            )

    def finish(self, wall_time_s: float) -> None:
        """End-run post-processing: write RDF/MSD/VACF CSVs, build summary."""
        cfg = self.cfg
        from pathlib import Path
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # write RDF
        for c in self.computes:
            if isinstance(c, ComputeRDF):
                r, g = c.finalize()
                np.savetxt(out / "rdf.csv", np.column_stack([r, g]),
                           delimiter=",", header="r,g", comments="")
            elif isinstance(c, ComputeMSD):
                t, msd = c.get_msd()
                np.savetxt(out / "msd.csv", np.column_stack([t, msd]),
                           delimiter=",", header="t,MSD", comments="")
            elif isinstance(c, ComputeVACF):
                t, vacf = c.get_vacf()
                np.savetxt(out / "vacf.csv", np.column_stack([t, vacf]),
                           delimiter=",", header="t,VACF", comments="")
            elif isinstance(c, ComputeStructureFactor):
                k, s = c.finalize()
                np.savetxt(out / "sk.csv", np.column_stack([k, s]),
                           delimiter=",", header="k,S(k)", comments="")
            elif isinstance(c, ComputeRDFTheta):
                r, cos, g_iso, g_2d = c.finalize()
                np.savez(out / "rdf_theta.npz",
                         r=r, cos=cos, g_2d=g_2d, g_isotropic=g_iso)

        # close dump files
        for d in self.dumps:
            d.close()

        # build validation summary (None -> auto-picks full-LJ vs LJTS
        # reference set based on cfg.use_tail_corrections)
        summary = build_validation_report(self, wall_time_s, None)
        with open(out / "summary.txt", "w") as f:
            f.write(summary)
        logger.info("Validation summary:\n%s", summary)

        # generate plots
        if cfg.generate_plots:
            try:
                from ..plotting import generate_all_plots
                generate_all_plots(self)
            except Exception as e:
                logger.warning("Plotting failed: %s", e)
