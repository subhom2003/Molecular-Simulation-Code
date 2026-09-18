"""
Initialization: lattice construction + Maxwell-Boltzmann velocities + MC liquid
"""
from __future__ import annotations
import numpy as np
import logging

from .. core.config import SimConfig

logger = logging.getLogger("lj_md.init")


def create_fcc_lattice(cfg: SimConfig) -> np.ndarray:
    N = cfg.N
    n_cells_per_dim = round((N / 4) ** (1.0 / 3.0))
    if 4 * n_cells_per_dim ** 3 != N:
        raise ValueError(
            f"For FCC lattice N must be 4*k^3 for integer k; got N={N} "
            f"nearest valid values: {[4*k**3 for k in range(2, 8)]}"
        )
    L = cfg.L
    a = L / n_cells_per_dim
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]) * a
    positions = []
    for ix in range(n_cells_per_dim):
        for iy in range(n_cells_per_dim):
            for iz in range(n_cells_per_dim):
                origin = np.array([ix, iy, iz], dtype=float) * a
                for b in basis:
                    positions.append(origin + b)
    positions = np.array(positions)
    positions = positions % L
    return positions


def create_sc_lattice(cfg: SimConfig) -> np.ndarray:
    N = cfg.N
    n_cells = round(N ** (1.0 / 3.0))
    if n_cells ** 3 != N:
        raise ValueError(f"For SC lattice N must be k^3; got N={N}")
    L = cfg.L
    a = L / n_cells
    coords = np.linspace(a / 2.0, L - a / 2.0, n_cells)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return positions


def create_random_lattice(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    N = cfg.N
    L = cfg.L
    min_d = cfg.min_pair_distance
    positions = np.zeros((N, 3))
    placed = 0
    max_attempts = 100_000
    while placed < N:
        for _ in range(max_attempts):
            cand = rng.uniform(0, L, size=3)
            ok = True
            for j in range(placed):
                d = cand - positions[j]
                d -= L * np.round(d / L)
                if d.dot(d) < min_d ** 2:
                    ok = False
                    break
            if ok or placed == 0:
                positions[placed] = cand
                placed += 1
                break
        else:
            raise RuntimeError(
                "Random lattice: failed to place particle after "
                f"{max_attempts} attempts (N={N}, density too high?)."
            )
    return positions


def maxwell_boltzmann_velocities(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    sigma = float(np.sqrt(cfg.T_star))
    return rng.normal(loc=0.0, scale=sigma, size=(cfg.N, 3))


def remove_com_velocity(v: np.ndarray) -> np.ndarray:
    v = v.copy()
    v -= v.mean(axis=0)
    return v


def scale_velocities_to_temperature(v: np.ndarray, T_target: float, dof: int) -> np.ndarray:
    v = v.copy()
    ke = 0.5 * float(np.einsum("ij,ij->", v, v))
    T_current = 2.0 * ke / float(dof)
    if T_current <= 0:
        raise RuntimeError("Cannot scale zero-T velocities.")
    lam = np.sqrt(T_target / T_current)
    v *= lam
    return v


def _check_overlaps(positions: np.ndarray, cfg: SimConfig) -> None:
    N = positions.shape[0]
    min_d = cfg.min_pair_distance
    L = cfg.L
    if N <= 320:
        i, j = np.triu_indices(N, k=1)
        d = positions[j] - positions[i]
        d -= L * np.round(d / L)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        min_r = float(r.min())
        if min_r < min_d:
            raise RuntimeError(
                f"Initial overlap detected: min pair distance {min_r:.4f} sigma "
                f"< threshold {min_d:.2f}."
            )
        logger.debug("Initial min pair distance: %.4f sigma", min_r)
    else:
        rng = np.random.default_rng(cfg.random_seed)
        n_check = 50_000
        ip = rng.integers(0, N, n_check)
        jp = rng.integers(0, N, n_check)
        mask = ip != jp
        d = positions[jp[mask]] - positions[ip[mask]]
        d -= L * np.round(d / L)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        logger.debug("Sampled min pair distance: %.4f sigma", float(r.min()))


def initialize_system(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    if cfg.lattice_type == "fcc":
        positions = create_fcc_lattice(cfg)
    elif cfg.lattice_type == "sc":
        positions = create_sc_lattice(cfg)
    elif cfg.lattice_type == "random":
        positions = create_random_lattice(cfg, rng)
    else:
        raise ValueError(f"Unknown lattice_type {cfg.lattice_type!r}")

    _check_overlaps(positions, cfg)
    positions = positions % cfg.L

    velocities = maxwell_boltzmann_velocities(cfg, rng)
    velocities = remove_com_velocity(velocities)
    velocities = scale_velocities_to_temperature(velocities, cfg.T_star,
                                                  dof=3 * cfg.N - 3)
    com_p = float(np.linalg.norm(velocities.sum(axis=0)))
    logger.debug(
        "Init velocities: T*=%.4f (target %.4f), |P_COM|=%.2e",
        2.0 * 0.5 * float(np.einsum("ij,ij->", velocities, velocities)) / (3 * cfg.N - 3),
        cfg.T_star, com_p
    )
    return positions, velocities
