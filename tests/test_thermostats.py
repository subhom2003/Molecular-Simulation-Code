"""Thermostat physics tests (review P6, P5)."""
from __future__ import annotations

import logging
import numpy as np
import pytest

from lj_md_package import SimConfig, Simulation, FixLangevin, FixNVT
from lj_md_package.computes.conserved_energy import ComputeConservedEnergy


def _quiet():
    logging.getLogger("lj_md").setLevel(logging.CRITICAL)
    logging.disable(logging.CRITICAL)


def test_baoab_temperature_unbiased():
    """BAOAB: kinetic T unbiased at O(dt) even at gamma=1."""
    _quiet()
    cfg = SimConfig(N=108, rho_star=0.02, T_star=1.0, n_steps=9000,
                    n_equil=1000, generate_plots=False, thermostat_type="none")
    sim = Simulation.from_config(cfg)
    sim.remove_fix("nve")
    sim.add_fix(FixLangevin("langevin", "all", T_target=1.0, gamma=1.0, seed=9))
    sim.run(8000)
    T = np.asarray(sim._thermo.as_arrays()["T"])
    mean_T = T[1000:].mean()
    # statistical error of the mean ~ T*sqrt(2/dof)*sqrt(2 tau_int / n) ~ 0.01
    assert abs(mean_T - 1.0) < 0.02


def test_nose_hoover_conserves_extended_hamiltonian():
    """MTTK split: H_NH drift must be small and time-reversible-ish."""
    _quiet()
    cfg = SimConfig(N=108, T_star=1.2, rho_star=0.6, n_steps=8000,
                    n_equil=1000, generate_plots=False, thermostat_type="none")
    sim = Simulation.from_config(cfg)
    nh = FixNVT("nh", "all", T_target=1.2, Q=50.0)
    sim.add_fix(nh)
    ce = ComputeConservedEnergy(every=10)
    sim.add_compute(ce)
    sim.run(6000)
    _, _, H = ce.get_history()
    drift = abs((H[-1] - H[0]) / H[0])
    assert drift < 5e-3


def test_nose_hoover_hits_target_T():
    _quiet()
    cfg = SimConfig(N=108, T_star=1.2, rho_star=0.6, n_steps=8000,
                    n_equil=1000, generate_plots=False, thermostat_type="none")
    sim = Simulation.from_config(cfg)
    sim.add_fix(FixNVT("nh", "all", T_target=1.2, Q=50.0))
    sim.run(6000)
    T = np.asarray(sim._thermo.as_arrays()["T"])
    assert abs(T[1000:].mean() - 1.2) < 0.05


def test_conserved_energy_includes_bath_once():
    """P5 regression: H_NH must equal KE + PE + bath, not 2*(KE+PE)+bath."""
    _quiet()
    cfg = SimConfig(N=108, T_star=1.2, rho_star=0.6, n_steps=8000,
                    n_equil=1000, generate_plots=False, thermostat_type="none")
    sim = Simulation.from_config(cfg)
    nh = FixNVT("nh", "all", T_target=1.2, Q=50.0)
    sim.add_fix(nh)
    ce = ComputeConservedEnergy(every=1)
    sim.add_compute(ce)
    sim.run(200)
    _, _, H = ce.get_history()
    th = sim._thermo.as_arrays()
    KE = float(th["KE"][-1])
    PE_trunc = float(th["PE"][-1]) - cfg.N * cfg.u_tail  # H uses truncated PE
    bath = nh.conserved_energy(sim.state)
    assert H[-1] == pytest.approx(KE + PE_trunc + bath, rel=1e-9)
