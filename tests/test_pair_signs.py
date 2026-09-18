"""Finite-difference force-sign tests for every pair style (review B5).

Force on atom 0 of an isolated pair at separation r must satisfy
F_x(0) = +dU/dr  (atom 0 sits at the origin of the pair axis).
"""
from __future__ import annotations

import numpy as np
import pytest

from lj_md_package.core.atoms import Atoms
from lj_md_package.core.box import Box
from lj_md_package.pairs.lj_cut import LJCut
from lj_md_package.pairs.soft import SoftPair
from lj_md_package.pairs.soft_sphere import SoftSpherePair
from lj_md_package.pairs.born_mayer import BornMayerPair

BOX = Box.cubic(20.0)
STYLES = [
    ("lj/cut", lambda: LJCut(2.5), 1.5),
    ("lj/cut(rep)", lambda: LJCut(2.5), 1.0),
    ("soft", lambda: SoftPair(r_cut=2.5, A=1.0), 1.5),
    ("soft/sphere", lambda: SoftSpherePair(r_cut=2.5, A=100.0), 1.5),
    ("born/mayer", lambda: BornMayerPair(r_cut=2.5, A=100.0, lam=0.3), 1.5),
]


def _pair_state(style, r):
    atoms = Atoms(positions=np.array([[10.0, 10.0, 10.0],
                                      [10.0 + r, 10.0, 10.0]]),
                  velocities=np.zeros((2, 3)))
    return style.compute(atoms, None, BOX)


@pytest.mark.parametrize("name,make,r", STYLES)
def test_force_matches_finite_difference(name, make, r):
    style = make()
    fr = _pair_state(style, r)
    d = 1e-6
    up = _pair_state(style, r + d).potential
    um = _pair_state(style, r - d).potential
    dudr = (up - um) / (2 * d)
    assert fr.forces[0, 0] == pytest.approx(dudr, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("name,make", [
    ("soft", lambda: SoftPair(r_cut=2.5, A=1.0)),
    ("soft/sphere", lambda: SoftSpherePair(r_cut=2.5, A=100.0)),
    ("born/mayer", lambda: BornMayerPair(r_cut=2.5, A=100.0, lam=0.3)),
])
def test_repulsive_styles_push_apart(name, make):
    style = make()
    fr = _pair_state(style, 1.5)
    # repulsive: force on atom 0 points in -x (away from atom 1)
    assert fr.forces[0, 0] < 0.0
    # net force zero (Newton's third law)
    assert np.allclose(fr.forces.sum(axis=0), 0.0, atol=1e-12)


def test_born_mayer_shifted_energy_at_cutoff():
    style = BornMayerPair(r_cut=2.5, A=100.0, lam=0.3)
    fr_in = _pair_state(style, 2.5 - 1e-9)
    fr_out = _pair_state(style, 2.5 + 1e-9)
    assert fr_in.potential == pytest.approx(0.0, abs=1e-9)
    assert fr_out.potential == 0.0


def test_lj_virial_sign_attractive_region():
    # attractive region: pair force pulls together -> virial W < 0
    fr = _pair_state(LJCut(2.5), 1.5)
    assert fr.virial < 0.0
    # repulsive region: W > 0
    fr = _pair_state(LJCut(2.5), 1.0)
    assert fr.virial > 0.0
