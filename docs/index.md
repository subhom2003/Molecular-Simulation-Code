# LJ-MD Package — Documentation

**lj-md-package** is a pure-Python LAMMPS-style molecular dynamics engine for
monatomic Lennard-Jones fluids. It supports both a Pythonic builder API and
LAMMPS-style input scripts.

## Quick Start

### Installation

```bash
pip install -e .
# With plotting support:
pip install -e .[plot]
```

### Programmatic Usage

```python
from lj_md_package import SimConfig, Simulation

cfg = SimConfig(
    N=108,
    rho_star=0.8442,
    T_star=0.722,
    n_steps=20000,
    n_equil=5000,
)
sim = Simulation.from_config(cfg)
sim.run(cfg.n_equil)          # equilibration with thermostat
sim.remove_fix("thermostat")  # switch to NVE
sim.run(cfg.n_steps - cfg.n_equil)  # production
sim.finish(0.0)               # write outputs
```

### Input Script (LAMMPS-style)

Create `in.lj`:

```lammps
units         lj
lattice       fcc 0.8442
create_atoms  1 box
pair_style    lj/cut 2.5
pair_coeff    1 1 1.0 1.0
neighbor      0.3 bin
velocity      all create 0.722 42
fix           1 all nve
fix           2 all temp/rescale 100 0.722 0.722 0.1 1.0
thermo        10
dump          1 all xyz 200 output/trajectory.xyz
run           5000
unfix         2
run           15000
```

Run it:

```bash
lj-md in.lj
```

### Built-in Benchmark

```bash
lj-md --benchmark --steps 20000 --equil 5000
```

Outputs go to `output/`: `thermo.csv`, `energies.csv`, `trajectory.xyz`,
`rdf.csv`, `msd.csv`, `vacf.csv`, `sk.csv`, `summary.txt`.

## Key Features

- **Correct LJ physics**: Truncated+shifted LJ at r_c=2.5 sigma with analytic tail corrections
- **Multiple thermostats**: Velocity rescale, Berendsen, Langevin (BBK), Nose-Hoover
- **Analysis computes**: RDF, MSD (Einstein diffusion), VACF (Green-Kubo diffusion), S(k), energy drift
- **Validation**: Automatic block-averaged comparison to LJTS literature values
- **LAMMPS-style parser**: Familiar input script syntax with variable substitution
- **Fast vectorized core**: NumPy-based force kernel with Verlet neighbor list
- **Installable package**: `pip install -e .`, CLI entry point `lj-md`

## Documentation Map

- `README.md` / `README.txt` — Overview, install, quickstart
- `docs/quickstart.md` — 5-minute walkthrough
- `docs/theory.md` — LJ potential, integrator, virial, observables
- `docs/thermostats_barostats.md` — Thermostat and barostat guide
- `docs/input_script_reference.md` — LAMMPS-style command reference
- `REVIEW.md` (repository root) — code-review findings and fixes log

## Architecture

```
lj_md_package/
  core/           SimConfig, Box, Atoms, Simulation, State
  pairs/          lj/cut (Lennard-Jones truncated+shifted)
  neighbors/      VerletNL, CellList
  integrators/    VelocityVerlet, MinimizeSD
  fixes/          nve, temp/rescale, berendsen, langevin, nose_hoover, momentum
  computes/       temp, pressure, rdf, msd, vacf, structure_factor, thermo
  dumps/          xyz, thermo, energy
  initialize/     FCC/SC/random lattice, MB velocities, MC liquid
  input/          LAMMPS-style parser (units, lattice, pair_style, fix, run...)
  analysis/       Block averaging, autocorrelation, validation report
  units/          LJ / real / metal / si + literature reference values
  cli.py          CLI entry point and benchmark()

examples/         run_lj_liquid.py (benchmark, supercritical, T-scan)
docs/             Sphinx documentation
```

## License

MIT