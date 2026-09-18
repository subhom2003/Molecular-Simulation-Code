"""
Example: Run the Verlet (1967) LJ liquid benchmark and variants.

Usage:
    python examples/run_lj_liquid.py --run benchmark
    python examples/run_lj_liquid.py --run supercritical
    python examples/run_lj_liquid.py --run scan
    python examples/run_lj_liquid.py --run all
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lj_md_package import SimConfig, Simulation


def run_benchmark():
    """Verlet (1967) point with the melt-anneal-NVE protocol.

    NOTE: (rho*=0.8442, T*=0.722, N=108) borders the solid-liquid
    coexistence band for the LJTS model, so a rescale-only equilibration
    never melts the FCC start - see REVIEW.md for the full discussion.
    We melt at T*=2.0 (Langevin), anneal to T*=0.722, then run NVE.
    """
    from lj_md_package import FixNVE, FixLangevin

    print("\n" + "=" * 60)
    print("  BENCHMARK: rho*=0.8442, T*=0.722, N=108 (melt-anneal-NVE)")
    print("=" * 60)

    n_equil = 6500  # 1500 melt + 5000 anneal
    n_prod = 13500
    cfg = SimConfig(
        N=108,
        rho_star=0.8442,
        T_star=0.722,
        dt=0.005,
        n_steps=n_equil + n_prod,
        n_equil=n_equil,
        r_cut=2.5,
        r_skin=0.3,
        thermostat_type="none",
        shift_potential=True,
        use_tail_corrections=True,
        lattice_type="fcc",
        random_seed=42,
        sample_interval=10,
        output_interval=500,
        output_dir="output/benchmark",
    )
    cfg.rdf_start = n_equil
    cfg.msd_start = n_equil
    cfg.vacf_start = n_equil
    sim = Simulation.from_config(cfg)
    # Langevin integrates itself: remove the plain-NVE fix while it is active
    sim.remove_fix("nve")

    lang = FixLangevin("thermostat", "all", T_target=2.0, gamma=1.0, seed=42)
    sim.add_fix(lang)
    sim.run(1500)                     # melt
    lang.T_target = 0.722             # anneal to the target
    sim.run(5000)
    sim.remove_fix("thermostat")
    sim.add_fix(FixNVE("nve", "all", sim.integrator))
    sim.run(n_prod)                   # production NVE
    sim.finish(0.0)


def run_supercritical():
    """Supercritical LJ fluid at rho*=0.3, T*=1.5."""
    print("\n" + "=" * 60)
    print("  SUPERCRITICAL: rho*=0.300, T*=1.500, N=108")
    print("=" * 60)

    cfg = SimConfig(
        N=108,
        rho_star=0.300,
        T_star=1.500,
        dt=0.005,
        n_steps=15000,
        n_equil=3000,
        r_cut=2.5,
        r_skin=0.4,
        thermostat_type="rescale",
        rescale_interval=50,
        shift_potential=True,
        use_tail_corrections=True,
        lattice_type="sc",
        random_seed=123,
        sample_interval=10,
        output_interval=500,
        output_dir="output/supercritical",
    )
    sim = Simulation.from_config(cfg)
    sim.run(cfg.n_equil)
    sim.remove_fix("thermostat")
    sim.run(cfg.n_steps - cfg.n_equil)
    sim.finish(0.0)


def run_temperature_scan():
    """Temperature scan at fixed rho*=0.8442."""
    T_values = [0.75, 0.90, 1.10, 1.35]
    print("\n" + "=" * 60)
    print(f"  TEMPERATURE SCAN: rho*=0.8442, T* in {T_values}")
    print("=" * 60)

    results = []
    for T_star in T_values:
        cfg = SimConfig(
            N=108,
            rho_star=0.8442,
            T_star=T_star,
            dt=0.005,
            n_steps=8000,
            n_equil=2000,
            r_cut=2.5,
            r_skin=0.3,
            thermostat_type="rescale",
            rescale_interval=50,
            shift_potential=True,
            use_tail_corrections=True,
            lattice_type="fcc",
            random_seed=42,
            sample_interval=10,
            output_interval=1000,
            output_dir=f"output/scan_T{T_star:.2f}",
        )
        sim = Simulation.from_config(cfg)
        sim.run(cfg.n_equil)
        sim.remove_fix("thermostat")
        sim.run(cfg.n_steps - cfg.n_equil)
        sim.finish(0.0)

        thermo = sim._thermo.as_arrays()
        prod_mask = thermo["step"] > cfg.n_equil
        if np.any(prod_mask):
            results.append({
                "T_star": T_star,
                "E_mean": float(thermo["E_per_atom"][prod_mask].mean()),
                "P_mean": float(thermo["P"][prod_mask].mean()),
            })

    print("\n" + "=" * 60)
    print("  ISOCHORE RESULTS  (rho*=0.8442)")
    print(f"  {'T*':>6}  {'<E*/N>':>10}  {'<P*>':>10}")
    print("  " + "-" * 32)
    for r in results:
        print(f"  {r['T_star']:>6.3f}  {r['E_mean']:>+10.4f}  {r['P_mean']:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LJ-MD example simulations")
    parser.add_argument(
        "--run",
        choices=["benchmark", "supercritical", "scan", "all"],
        default="benchmark",
        help="Which example to run (default: benchmark)",
    )
    args = parser.parse_args()

    if args.run == "benchmark":
        run_benchmark()
    elif args.run == "supercritical":
        run_supercritical()
    elif args.run == "scan":
        run_temperature_scan()
    elif args.run == "all":
        run_benchmark()
        run_supercritical()
        run_temperature_scan()