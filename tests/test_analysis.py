"""Analysis-layer tests (review P1, B6, P4)."""
from __future__ import annotations

import numpy as np
import pytest

from lj_md_package.computes.vacf import ComputeVACF
from lj_md_package.computes.msd import ComputeMSD
from lj_md_package.computes.rdf import ComputeRDF
from lj_md_package.computes.structure_factor import ComputeStructureFactor
from lj_md_package.core.atoms import Atoms
from lj_md_package.core.box import Box
from lj_md_package.core.state import SimulationState


def _fake_state(step, N=108, L=10.0, T=1.0, seed=0):
    rng = np.random.default_rng(seed)
    atoms = Atoms(positions=rng.uniform(0, L, (N, 3)),
                  velocities=rng.normal(0, np.sqrt(T), (N, 3)))
    st = SimulationState(dt=0.005)
    st.atoms = atoms
    st.box = Box.cubic(L)
    st.step = step
    st.time = step * 0.005
    st.unwrapped = atoms.positions.copy()
    return st


def test_vacf_green_kubo_factor():
    """B6 regression: D = (1/3) int C(t) dt with C UNNORMALIZED.

    Synthetic C(t) = 3T exp(-t/tau): D = T*tau.
    """
    vacf = ComputeVACF(max_length=1000, dt=0.01)
    T, tau = 0.722, 0.3
    nlag = vacf.max_length + 1
    t = np.arange(nlag) * 0.01
    C_un = 3 * T * np.exp(-t / tau)
    vacf._sum[:] = C_un * 10      # pretend 10 origins contributed each lag
    vacf._count[:] = 10
    D = vacf.green_kubo_diffusion()
    assert D == pytest.approx(T * tau, rel=0.02)


def test_compute_start_gating():
    """P1: computes must not accumulate before `start`."""
    rdf = ComputeRDF(every=1, n_bins=50, L=10.0, N=108, start=100)
    for step in range(200):
        st = _fake_state(step)
        rdf.compute_if_due(st, None)
    assert rdf.n_samples == 100  # steps 100..199


def test_ideal_gas_rdf_flat():
    """Uniform random (ideal-gas) frames: g(r) ~ 1 within statistics."""
    rdf = ComputeRDF(every=1, n_bins=60, L=10.0, N=108)
    rng = np.random.default_rng(7)
    for s in range(300):
        rdf.compute(_fake_state(s, seed=int(rng.integers(1e9))), None)
    r, g = rdf.finalize()
    mid = (r > 1.0) & (r < 4.5)
    # ideal gas: mean ~1 with ~sqrt fluctuations; tolerance generous
    assert abs(g[mid].mean() - 1.0) < 0.07


def test_structure_factor_no_nan_and_plateau():
    """P4: no NaN bins; ideal-gas S(k) ~ 1."""
    skc = ComputeStructureFactor(every=1, L=10.0, N=108)
    rng = np.random.default_rng(3)
    for s in range(50):
        skc.compute(_fake_state(s, seed=int(rng.integers(1e9))), None)
    k, sk = skc.finalize()
    assert not np.isnan(sk).any()
    # ideal gas plateau: mean S(k>4) ~ 1
    mask = k > 4.0
    assert abs(sk[mask].mean() - 1.0) < 0.15


def test_msd_uses_unwrapped_positions():
    """MSD must count particles that cross periodic boundaries."""
    msd = ComputeMSD(every=1, max_length=10, dt=0.005)
    N = 4
    st = _fake_state(0, N=N, L=2.0)
    st.atoms.positions[:] = 0.1
    st.unwrapped = np.full((N, 3), 0.1)
    msd.compute_if_due(st, None)
    # each step: +0.4 in x, wrapping across the L=2 boundary at step 5
    for s in range(1, 6):
        x = 0.1 + 0.4 * s  # unwrapped; wrapped value is x % 2
        st = _fake_state(s, N=N, L=2.0)
        st.atoms.positions[:] = np.array([x % 2.0, 0.1, 0.1])
        st.unwrapped = np.tile(np.array([x, 0.1, 0.1]), (N, 1))
        msd.compute_if_due(st, None)
    t, m = msd.get_msd()
    # at lag n: MSD = (0.4 n)^2; lag 5 includes the boundary crossing
    assert m[5] == pytest.approx((0.4 * 5) ** 2, rel=1e-10)
