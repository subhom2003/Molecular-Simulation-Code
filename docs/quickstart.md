# Quickstart — LJ-MD

## Installation

```bash
pip install -e .
# Optional:
pip install -e .[plot]
```

Requires Python >= 3.10 and NumPy >= 1.24.

## Your First Simulation

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
sim.run(cfg.n_equil)              # equilibration with thermostat
sim.remove_fix("thermostat")      # switch to NVE
sim.run(cfg.n_steps - cfg.n_equil) # production
sim.finish(0.0)                   # write outputs
```

Outputs go to `output/`:

- `thermo.csv` -- thermodynamic data (T, P, E, etc.)
- `energies.csv` -- energy components
- `trajectory.xyz` -- extended-XYZ trajectory
- `rdf.csv` -- radial distribution function
- `msd.csv` -- mean-squared displacement
- `vacf.csv` -- velocity autocorrelation
- `sk.csv` -- static structure factor
- `summary.txt` -- validation report

## Run the Built-in Benchmark

```bash
lj-md --benchmark --steps 20000 --equil 5000
```

## Run an Input Script (LAMMPS-style)

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

## Key Configuration Parameters

| Parameter      | Default | Description                          |
|----------------|---------|--------------------------------------|
| `N`            | 108     | Number of particles (FCC: 4*k^3)     |
| `rho_star`     | 0.8442  | Reduced density                      |
| `T_star`       | 0.722   | Reduced temperature                  |
| `r_cut`        | 2.5     | LJ cutoff                            |
| `dt`           | 0.005   | Timestep                             |
| `n_steps`      | 20000   | Total integration steps              |
| `n_equil`      | 5000    | Equilibration steps                  |
| `thermostat_type` | "rescale" | rescale/berendsen/nose_hoover/langevin/none |
| `shift_potential` | True | Shift U so U(r_c)=0                 |
| `use_tail_corrections` | True | Apply g(r)=1 tail corrections    |

## Next Steps

- `docs/configuration.md` — All SimConfig parameters
- `docs/thermostats.md` — Choosing a thermostat
- `docs/input_script.md` — Full LAMMPS-style command reference
- `docs/validation.md` — Benchmark validation protocol