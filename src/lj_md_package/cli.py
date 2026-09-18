"""
=============================================================================
lj-md CLI
=============================================================================

Usage::

    lj-md in.lj                - run a LAMMPS-style input file
    lj-md --benchmark          - run the default Verlet (1967) benchmark
    lj-md --benchmark --steps N - run the benchmark with N production steps
    lj-md --strict             - raise on validation failures

Builds the Simulation from a config file or runs directly from defaults.
=============================================================================
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import logging

from .core.config import SimConfig
from .core.simulation import Simulation
from .core.logging_setup import setup_log
from .input.script import run_script


def benchmark(steps: int = 20_000, n_equil: int = 5_000, strict: bool = False,
              output_dir: str = "output", no_plots: bool = False):
    """Run the Verlet (1967) LJ benchmark: rho*=0.8442, T*=0.722, N=108.

    from_config adds: FixNVE ("nve"), the velocity-rescale thermostat
    ("thermostat", cfg.thermostat_type='rescale'), and the analysis computes
    gated at rdf_start/msd_start/vacf_start.  Production runs are NVE after
    removing the thermostat fix.
    """
    import numpy as np
    cfg = SimConfig(
        rho_star=0.8442, T_star=0.722, N=108,
        n_steps=steps, n_equil=n_equil,
        output_dir=output_dir,
        strict=strict,
        generate_plots=not no_plots,
    )
    # Sampling starts must be <= n_steps; clamp for short custom runs.
    for attr in ("rdf_start", "msd_start", "vacf_start"):
        if getattr(cfg, attr) > cfg.n_steps:
            setattr(cfg, attr, cfg.n_equil)
    setup_log(f"{output_dir}/simulation.log", logging.INFO)
    sim = Simulation.from_config(cfg)
    from .dumps.thermo import DumpThermo
    from .dumps.energy import DumpEnergy
    from .dumps.xyz import DumpXYZ
    from .fixes.momentum import FixMomentum

    # zero momentum every 1000 steps
    sim.add_fix(FixMomentum(fix_id="mom", group="all", N=1000))
    # dumps
    sim.add_dump(DumpThermo(dump_id="thermo", every=cfg.sample_interval,
                             filename=f"{output_dir}/thermo.csv",
                             rho_star=cfg.rho_star, P_tail=cfg.P_tail,
                             u_tail=cfg.u_tail))
    sim.add_dump(DumpEnergy(dump_id="energy", every=cfg.sample_interval,
                             filename=f"{output_dir}/energies.csv",
                             N=cfg.N, u_tail=cfg.u_tail))
    sim.add_dump(DumpXYZ(dump_id="traj", every=cfg.traj_interval,
                          filename=f"{output_dir}/trajectory.xyz"))

    # record initial thermodynamic sample (P WITHOUT tail - record() adds it)
    sim._thermo.record(
        step=0, dt=cfg.dt, KE=sim.state.kinetic_energy,
        PE=sim.state.force_result.potential, virial=sim.state.force_result.virial,
        T=sim.state.temperature,
        P=(sim.state.force_result.virial / (3.0 * cfg.V) +
           cfg.N * sim.state.temperature / cfg.V),
        min_r=sim.state.force_result.min_r,
        max_f=sim.state.force_result.max_f,
        p_COM=float(np.linalg.norm(sim.atoms.velocities.mean(axis=0) * sim.atoms.N))
    )

    # ----- Equilibration with thermostat
    print(f"[equilibration] {n_equil} steps with thermostat ({cfg.thermostat_type})")
    t0 = time.perf_counter()
    sim.run(n_equil)
    print(f"[equilibration] done in {time.perf_counter() - t0:.2f}s")

    # ----- remove thermostat, run production NVE
    sim.remove_fix("thermostat")
    print(f"[production] {steps - n_equil} steps NVE")
    t0 = time.perf_counter()
    sim.run(steps - n_equil)
    print(f"[production] done in {time.perf_counter() - t0:.2f}s")

    sim.finish(time.perf_counter() - t0)
    return sim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lj-md",
        description="LAMMPS-style Lennard-Jones molecular dynamics in pure Python",
    )
    parser.add_argument("input_file", nargs="?",
                        help="LAMMPS-style input script (.in) to run")
    parser.add_argument("--benchmark", action="store_true",
                        help="run the default Verlet (1967) LJ benchmark")
    parser.add_argument("--steps", type=int, default=20_000,
                        help="production steps for --benchmark")
    parser.add_argument("--equil", type=int, default=5_000,
                        help="equilibration steps for --benchmark")
    parser.add_argument("--output", "-o", default="output",
                        help="output directory")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip generating publication-quality plots")
    parser.add_argument("--strict", action="store_true",
                        help="raise on validation failure (otherwise log warning)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="logging verbosity")
    parser.add_argument("--version", action="store_true")

    args = parser.parse_args(argv)
    if args.version:
        from . import __version__
        print(__version__)
        return 0
    setup_log(f"{args.output}/simulation.log",
              getattr(logging, args.log_level))
    if args.benchmark:
        benchmark(steps=args.steps, n_equil=args.equil,
                  strict=args.strict, output_dir=args.output,
                  no_plots=args.no_plots)
        return 0
    if args.input_file:
        if not os.path.isfile(args.input_file):
            print(f"lj-md: input file not found: {args.input_file}", file=sys.stderr)
            return 1
        run_script(args.input_file)
        return 0
    # no-op -- show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
