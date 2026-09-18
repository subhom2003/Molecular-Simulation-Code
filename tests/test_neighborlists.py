"""Neighbor-list correctness (review B2)."""
from __future__ import annotations

import numpy as np
import pytest

from lj_md_package.core.atoms import Atoms
from lj_md_package.core.box import Box
from lj_md_package.core.config import SimConfig
from lj_md_package.initialize import create_fcc_lattice
from lj_md_package.neighbors.verlet import VerletNL
from lj_md_package.neighbors.cell_list import CellList
from lj_md_package.pairs.lj_cut import LJCut


def _random_config(N, rho, seed=0):
    cfg = SimConfig(N=N, rho_star=rho)
    pos = create_fcc_lattice(cfg) + np.random.default_rng(seed).normal(0, 0.05, (N, 3))
    pos %= cfg.L
    return cfg, pos


def test_cell_list_matches_verlet_for_large_box():
    """N=864 -> nc=3: CellList must reproduce VerletNL exactly."""
    cfg, pos = _random_config(864, 0.8442)
    box = Box.cubic(cfg.L)
    atoms = Atoms(positions=pos, velocities=np.zeros((864, 3)))
    ps = LJCut(2.5)
    vn = VerletNL(2.5, 0.3, cfg.L); vn.build(pos)
    cl = CellList(2.5, 0.3, cfg.L); cl.build(pos)
    fv = ps.compute(atoms, vn, box)
    fc = ps.compute(atoms, cl, box)
    assert fc.potential == pytest.approx(fv.potential, rel=1e-12)
    assert fc.virial == pytest.approx(fv.virial, rel=1e-12)
    assert np.allclose(fc.forces, fv.forces, atol=1e-12)
    # and no pair duplication
    ic, jc = cl.get_pairs()
    keys = np.minimum(ic, jc) * 10_000_000 + np.maximum(ic, jc)
    assert len(np.unique(keys)) == len(keys)


def test_cell_list_rejects_tiny_boxes():
    """nc=2 (e.g. N=108 at rho=0.8442) must refuse: it would double-count."""
    with pytest.raises(ValueError):
        CellList(2.5, 0.3, 5.0)


def test_factory_falls_back_to_verlet():
    from lj_md_package.core.simulation import _make_neighbor_list, VerletNL as V
    cfg = SimConfig(N=108, neighbor_style="cell")
    nl = _make_neighbor_list(cfg)
    assert isinstance(nl, V)


def test_cell_list_no_misses_vs_bruteforce():
    """Every pair with MIC distance < r_list must appear."""
    cfg, pos = _random_config(864, 0.8442, seed=5)
    cl = CellList(2.5, 0.3, cfg.L); cl.build(pos)
    ic, jc = cl.get_pairs()
    got = set(zip(np.minimum(ic, jc).tolist(), np.maximum(ic, jc).tolist()))
    i, j = np.triu_indices(864, k=1)
    dr = pos[j] - pos[i]
    dr -= cfg.L * np.round(dr / cfg.L)
    r2 = np.einsum("ij,ij->i", dr, dr)
    want = set(zip(i[r2 < 2.8 ** 2].tolist(), j[r2 < 2.8 ** 2].tolist()))
    assert want <= got
