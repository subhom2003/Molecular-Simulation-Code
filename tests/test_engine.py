"""Engine-level regressions (review B1, B3, B4, P3)."""
from __future__ import annotations

import logging
import numpy as np
import pytest

from lj_md_package import SimConfig, Simulation
from lj_md_package.core.atoms import Atoms
from lj_md_package.core.box import Box
from lj_md_package.input.script import Script
from lj_md_package.initialize import create_fcc_lattice


def _quiet():
    logging.getLogger("lj_md").setLevel(logging.CRITICAL)
    logging.disable(logging.CRITICAL)


def test_from_config_actually_integrates():
    """B1 regression: a from_config simulation must move atoms."""
    _quiet()
    cfg = SimConfig(N=108, n_steps=6000, n_equil=1000, generate_plots=False)
    sim = Simulation.from_config(cfg)
    ids = [f.fix_id for f in sim.fixes]
    assert "nve" in ids or "thermostat" in ids
    p0 = sim.atoms.positions.copy()
    sim.run(500)
    assert np.abs(sim.atoms.positions - p0).max() > 0.0
    # thermostat id documented in the READMEs must exist
    assert sim.remove_fix("thermostat") is True


def test_nve_energy_conservation():
    """Velocity-Verlet on a pre-melted liquid at dt=0.002: regression-slope
    drift < 1e-4 relative over a 20-tau window.  (dt=0.005 is fine for
    production work, but its bounded O(dt^2) error term adds ~3e-4 at the
    benchmark densities, so the regression test uses a finer setting that
    separates integration error from thermostat relaxation.)"""
    _quiet()
    dt = 0.002
    melt, prod = 3750, 8000
    cfg = SimConfig(N=108, rho_star=0.7, T_star=1.5, n_steps=melt + prod,
                    n_equil=melt, generate_plots=False, dt=dt,
                    thermostat_type="langevin")
    sim = Simulation.from_config(cfg)
    sim.run(melt)                       # melt with Langevin (self-integrating)
    sim.remove_fix("thermostat")       # switch to plain NVE
    from lj_md_package import FixNVE
    sim.add_fix(FixNVE("nve", "all", sim.integrator))
    sim.run(prod)
    th = sim._thermo.as_arrays()
    E = np.asarray(th["E_total"])[melt:]
    n = np.arange(len(E))
    slope = np.polyfit(n, E, 1)[0]
    assert abs(slope) * len(E) < 1e-4 * abs(E[0])


def test_langevin_input_script_moves_atoms():
    """B3 regression."""
    _quiet()
    text = """
    units lj
    lattice fcc 0.8442
    create_atoms 1 box
    pair_style lj/cut 2.5
    pair_coeff 1 1 1.0 1.0
    neighbor 0.3 nsq
    velocity all create 2.0 42
    fix 1 all langevin 2.0 1.0 42
    run 500
    """
    s = Script(); s.run_string(text)
    cfg = SimConfig()
    lat = create_fcc_lattice(cfg)
    dev = np.abs((s.sim.atoms.positions - lat + cfg.L / 2) % cfg.L - cfg.L / 2).max()
    assert dev > 0.05
    # must not carry a FixNVE at the same time
    names = [f.name() for f in s.sim.fixes]
    assert "nve" not in names


def test_double_integrator_rejected():
    _quiet()
    from lj_md_package import FixLangevin
    cfg = SimConfig(N=108, n_steps=6000, n_equil=1000, generate_plots=False)
    sim = Simulation.from_config(cfg)  # has FixNVE + rescale thermostat
    sim.add_fix(FixLangevin("l2", "all", T_target=1.0, gamma=1.0, seed=1))
    with pytest.raises(RuntimeError):
        sim.run(10)


def test_no_integrating_fix_rejected():
    _quiet()
    cfg = SimConfig(N=108, n_steps=6000, n_equil=1000, generate_plots=False,
                    thermostat_type="none")
    sim = Simulation.from_config(cfg)
    sim.remove_fix("nve")
    with pytest.raises(RuntimeError):
        sim.run(10)


def test_fix_nvt_argument_order():
    """B4 regression: `fix 2 all nvt 0.722 5.0` must set T=0.722, Q=5.0."""
    _quiet()
    s = Script()
    s.run_string("""
    units lj
    lattice fcc 0.8442
    create_atoms 1 box
    pair_style lj/cut 2.5
    pair_coeff 1 1 1.0 1.0
    neighbor 0.3 nsq
    velocity all create 0.722 42
    fix 1 all nve
    fix 2 all nvt 0.722 5.0
    """)
    nvt = [f for f in s.sim.fixes if f.name() == "nvt"][0]
    assert nvt.T_target == pytest.approx(0.722)
    assert nvt.Q == pytest.approx(5.0)


def test_temp_rescale_lammps_order_and_window():
    """P3: window is the 4th arg (absolute), fraction the 5th."""
    _quiet()
    from lj_md_package.fixes.temp_rescale import FixTempRescale
    s = Script()
    s.run_string("""
    units lj
    lattice fcc 0.8442
    create_atoms 1 box
    pair_style lj/cut 2.5
    pair_coeff 1 1 1.0 1.0
    neighbor 0.3 nsq
    velocity all create 1.2 42
    fix 1 all nve
    fix 2 all temp/rescale 10 0.8 0.8 0.05 0.5
    """)
    tr = [f for f in s.sim.fixes if f.name() == "temp/rescale"][0]
    assert tr.window == pytest.approx(0.05)
    assert tr.fraction == pytest.approx(0.5)
    # window is ABSOLUTE: |1.2 - 0.8| > 0.05 -> must rescale, not skip
    atoms = s.sim.atoms
    T0 = atoms.temperature()
    tr.do_end_of_step(s.sim.state)
    assert atoms.temperature() != pytest.approx(T0)


def test_mic_without_neighbor_list_uses_box():
    """B7 regression: pair.compute(atoms, None, box) must apply MIC via box."""
    from lj_md_package.pairs.lj_cut import LJCut
    pos = np.array([[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]])  # MIC neighbor of atom 0
    atoms = Atoms(positions=pos, velocities=np.zeros((2, 3)))
    box = Box.cubic(10.0)
    fr = LJCut(2.5).compute(atoms, None, box)
    # MIC distance is sqrt(3*0.2^2) ~ 0.346, NOT the raw 16.97
    assert fr.min_r == pytest.approx(np.sqrt(3) * 0.2, rel=1e-9)
    assert fr.potential != 0.0


def test_minimize_descends():
    """B5 regression: steep-descent must DECREASE the energy."""
    _quiet()
    from lj_md_package.integrators.minimize import MinimizeSD
    from lj_md_package.fixes.nve import FixNVE
    # small random config (overlap-y): relax it
    cfg = SimConfig(N=108, rho_star=0.3, T_star=1.0, lattice_type="random",
                    n_steps=6000, n_equil=100, generate_plots=False,
                    thermostat_type="none", max_initial_force=1e8)
    sim = Simulation.from_config(cfg)
    sim.remove_fix("nve")
    mini = MinimizeSD(dt=cfg.dt, box=sim.box)
    sim.add_fix(FixNVE("min", "all", mini))
    U0 = sim.state.force_result.potential
    sim.run(300)
    U1 = sim.state.force_result.potential
    assert U1 < U0


def test_cell_neighborhood_nsq_naming():
    text = """
    units lj
    lattice fcc 0.8442
    create_atoms 1 box
    pair_style lj/cut 2.5
    pair_coeff 1 1 1.0 1.0
    neighbor 0.3 bin
    velocity all create 0.722 42
    fix 1 all nve
    run 5
    """
    # 'bin' at N=108 must fall back to VerletNL (nc=2 would double-count)
    _quiet()
    s = Script(); s.run_string(text)
    from lj_md_package.neighbors.verlet import VerletNL
    assert isinstance(s.sim.neighbor_list, VerletNL)
