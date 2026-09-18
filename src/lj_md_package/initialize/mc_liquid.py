"""
NVT Metropolis Monte Carlo liquid generator (vectorized).

Uses the correct LJ potential with proper Metropolis sampling.
No integrator stability issues because MC accepts/rejects moves
based on energy, not force integration.

The delta-energy computation is fully vectorized (O(N) numpy ops per step).
"""
from __future__ import annotations
import numpy as np
from .. core.config import SimConfig
from .. core.box import Box
from .. core.atoms import Atoms


def _pair_energy_vector(positions: np.ndarray, box: Box, r_cut: float) -> float:
    """Total pair energy via O(N²) vectorized sum.  Used only for diagnostics."""
    N = positions.shape[0]
    i, j = np.triu_indices(N, k=1)
    dr = positions[j] - positions[i]
    dr = box.minimum_image(dr)
    r2 = np.einsum("ij,ij->i", dr, dr)
    mask = r2 < r_cut * r_cut
    if mask.sum() == 0:
        return 0.0
    r2_c = r2[mask]
    inv_r6 = 1.0 / (r2_c * r2_c * r2_c)
    return float((4.0 * (inv_r6 * inv_r6 - inv_r6)).sum())


def _delta_energy_vector(
    positions: np.ndarray, box: Box, i: int, new_pos: np.ndarray, r_cut: float
) -> float:
    """Energy change for moving atom i → new_pos, fully vectorized O(N)."""
    old_pos = positions[i]
    N = positions.shape[0]

    dr_old = positions - old_pos
    dr_old = box.minimum_image(dr_old)
    r2_old = np.einsum("ij,ij->i", dr_old, dr_old)

    dr_new = positions - new_pos
    dr_new = box.minimum_image(dr_new)
    r2_new = np.einsum("ij,ij->i", dr_new, dr_new)

    mask = np.arange(N) != i

    def u_arr(r2_arr):
        in_cut = r2_arr < r_cut * r_cut
        inv_r6 = np.zeros_like(r2_arr)
        inv_r6[in_cut] = 1.0 / (r2_arr[in_cut] ** 3)
        return 4.0 * (inv_r6 * inv_r6 - inv_r6)

    old_u = u_arr(r2_old[mask]).sum()
    new_u = u_arr(r2_new[mask]).sum()
    return float(new_u - old_u)


def generate_liquid_mc(
    cfg: SimConfig,
    n_mc_steps: int = 100_000,
    max_disp: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> Atoms:
    """Generate a liquid configuration at the target density/temperature.

    Runs NVT Metropolis MC with the correct LJ potential.  Starting
    config is FCC for N divisible by 4, SC for N = k^3, otherwise a
    random non-overlapping grid.  Returns an Atoms object with zero
    velocities (caller must assign MB velocities and remove COM).
    """
    N = cfg.N
    L = cfg.L
    n_cells = int(round(N ** (1.0 / 3.0)))
    from .. initialize import create_fcc_lattice, create_sc_lattice
    try:
        positions = create_fcc_lattice(cfg)
    except ValueError:
        try:
            positions = create_sc_lattice(cfg)
        except ValueError:
            # Grid placement for arbitrary N
            spacing = L / n_cells
            positions = []
            for iz in range(n_cells):
                for iy in range(n_cells):
                    for ix in range(n_cells):
                        if len(positions) < N:
                            positions.append([ix*spacing, iy*spacing, iz*spacing])
            positions = np.array(positions[:N])
    box = Box.cubic(cfg.L)
    rng = np.random.default_rng(seed)
    N = cfg.N
    beta = 1.0 / cfg.T_star

    n_accept = 0
    n_reject = 0

    for step in range(1, n_mc_steps + 1):
        i = rng.integers(N)
        new_pos = positions[i] + rng.uniform(-max_disp, max_disp, size=3)
        new_pos = new_pos % cfg.L

        dU = _delta_energy_vector(positions, box, i, new_pos, cfg.r_cut)
        if dU < 0 or rng.uniform() < np.exp(-beta * dU):
            positions[i] = new_pos
            n_accept += 1
        else:
            n_reject += 1

        if verbose and step % max(1, n_mc_steps // 10) == 0:
            U_curr = _pair_energy_vector(positions, box, cfg.r_cut)
            min_r2 = _min_pair_dist2_vector(positions, box)
            print(f"  MC step {step:7d}: U={U_curr:+.2f}  min_r={np.sqrt(min_r2):.4f}  "
                  f"acc={n_accept / (n_accept + n_reject + 1):.3f}")

    U_final = _pair_energy_vector(positions, box, cfg.r_cut)
    min_r2 = _min_pair_dist2_vector(positions, box)
    if verbose:
        print(f"MC done: U_final={U_final:+.2f} min_r={np.sqrt(min_r2):.4f} "
              f"acc_rate={n_accept / (n_accept + n_reject + 1):.3f}")
    return Atoms(positions=positions, velocities=np.zeros((N, 3)))


def _min_pair_dist2_vector(positions: np.ndarray, box: Box) -> float:
    """Minimum pair separation squared, vectorized O(N²)."""
    N = positions.shape[0]
    i, j = np.triu_indices(N, k=1)
    dr = positions[j] - positions[i]
    dr = box.minimum_image(dr)
    r2 = np.einsum("ij,ij->i", dr, dr)
    return float(r2.min())
