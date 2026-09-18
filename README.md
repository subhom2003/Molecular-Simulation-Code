# lj-md — A Simple Lennard-Jones Molecular Dynamics Engine

A small, simple (NumPy-only), pure-Python molecular dynamics engine
for monatomic **Lennard-Jones (LJ) fluids**, with a LAMMPS-style input-script
interface, a Pythonic builder API, and built-in validation against literature
reference data.

> ⚠️ This is a research-grade MD simulation engine. 

---

## Table of Contents

1. [What does it do?](#1-what-does-it-do)
2. [Features](#2-features)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Input Script Reference (summary)](#5-input-script-reference-summary)
6. [Python API](#6-python-api)
7. [Outputs](#7-outputs)
8. [Physics background](#8-physics-background)
9. [Thermostats and barostat](#9-thermostats-and-barostat)
10. [Validation and the benchmark caveat](#10-validation-and-the-benchmark-caveat)
11. [Testing](#11-testing)
12. [Project layout](#12-project-layout)
13. [Troubleshooting](#13-troubleshooting)
14. [References](#14-references)
15. [License](#15-license)

---

## 1. What does it do?

It integrates Newton's equations of motion for `N` particles interacting via
the Lennard-Jones 12-6 potential, in periodic boundary conditions, in LJ
reduced units:

```
U(r) = 4ε [ (σ/r)^12 − (σ/r)^6 ]
```

The default benchmark reproduces the classic **Verlet (1967)** state point

```
ρ* = 0.8442,  T* = 0.722,  N = 108
```

with the truncated-and-shifted potential at `r_c = 2.5 σ` and analytic
`g(r)=1` long-range tail corrections.

You can drive it three ways:

| Interface | Entry point | Best for |
|---|---|---|
| CLI benchmark | `lj-md --benchmark` | checking the install works |
| Input scripts | `lj-md in.lj` | LAMMPS-like workflows |
| Python API | `Simulation.from_config(...)` | scripted studies, parameter scans |

---

## 2. Features

- **Correct LJ physics** — truncated + shifted potential with analytic tail
  corrections; virial pressure; energy conservation monitored (typical NVE
  drift `|dE/E0| ≲ 1e-4`).
- **Integrators** — velocity Verlet; steepest-descent minimizer; BAOAB
  (GJF) Langevin as a self-integrating stochastic scheme.
- **Thermostats** — velocity rescale, Berendsen, Nosé–Hoover (time-reversible
  Martyna–Tuckerman split), Langevin (BAOAB, no O(Δt) kinetic-temperature
  bias).
- **Barostat** — isotropic Berendsen (`press/berendsen`) for equilibration.
- **Neighbor lists** — O(N²) Verlet half-list; O(N) linked-cell list with
  automatic safe fallback for small boxes.
- **Analysis computes** — thermo time series, RDF g(r), angular g(r, cosθ),
  MSD (Einstein D), VACF (Green–Kubo D), static structure factor S(k) by
  direct k-space summation, Nosé–Hoover conserved-energy tracking.
- **Statistics** — Flyvbjerg–Petersen block-averaged error bars and
  integrated autocorrelation times in the validation report.
- **Unit labels** — LJ reduced internally; `lj`, `real`, `metal`, `si`
  labels accepted in scripts.
- **Tests** — a `pytest` suite covering force signs, neighbor lists,
  thermostat statistics, and analysis correctness.

---

## 3. Installation

Requirements: **Python ≥ 3.10**, **NumPy ≥ 1.24** (plots need matplotlib).

```bash
git clone <your-fork-url>
cd lj_md_package

# core engine only
pip install -e .

# with plotting support
pip install -e .[plot]

# developer extras (pytest)
pip install -e .[plot,test]
```

Verify:

```bash
lj-md --version        # -> 0.2.0
python -m pytest tests/ -q
```

---

## 4. Quick Start

### 4a. One-line benchmark (fastest check)

```bash
lj-md --benchmark --steps 20000 --equil 5000 --output output/benchmark
```

Then inspect `output/benchmark/summary.txt` and the PNG plots.

### 4b. From an input script

`in.lj`:

```lammps
units         lj
lattice       fcc 0.8442
create_atoms  1 box
pair_style    lj/cut 2.5
pair_coeff    1 1 1.0 1.0
neighbor      0.3 nsq
velocity      all create 0.722 42
fix           1 all nve
fix           2 all temp/rescale 100 0.722 0.722 0.1 1.0
thermo        10
dump          1 all xyz 200 output/trajectory.xyz
run           5000
unfix         2
run           15000
```

```bash
lj-md in.lj
```

### 4c. Python API

```python
from lj_md_package import SimConfig, Simulation

cfg = SimConfig(N=108, rho_star=0.8442, T_star=0.722,
                n_steps=20000, n_equil=5000,
                output_dir="output/api_run")
sim = Simulation.from_config(cfg)
sim.run(cfg.n_equil)               # equilibration (rescale thermostat on)
sim.remove_fix("thermostat")       # switch thermostat off
sim.run(cfg.n_steps - cfg.n_equil) # NVE production
sim.finish(0.0)                    # write outputs + validation report
```

### 4d. The recommended liquid protocol (melt-anneal-NVE)

At dense/low-T state points an FCC start will not melt by itself — use a
hot Langevin melt, anneal, then NVE production:

```python
from lj_md_package import SimConfig, Simulation, FixNVE, FixLangevin

cfg = SimConfig(N=108, rho_star=0.8442, T_star=0.722,
                n_steps=70000, n_equil=20000, thermostat_type="none",
                output_dir="output/melt_anneal")
sim = Simulation.from_config(cfg)
sim.remove_fix("nve")                         # Langevin integrates itself

lang = FixLangevin("thermostat", "all", T_target=2.0, gamma=1.0, seed=42)
sim.add_fix(lang)
sim.run(5000)                                  # melt at T* = 2.0
lang.T_target = 0.722                          # anneal to the target
sim.run(15000)
sim.remove_fix("thermostat")
sim.add_fix(FixNVE("nve", "all", sim.integrator))
sim.run(50000)                                 # NVE production
sim.finish(0.0)
```

---

## 5. Input Script Reference (summary)

Commands are processed line by line; `#` starts a comment; `&` continues a
line; `${var}` expands variables set with `variable`.

| Command | Syntax | Notes |
|---|---|---|
| `units` | `lj` / `real` / `metal` / `si` | internally always LJ units |
| `atom_style` | `atomic` | only `atomic` |
| `boundary` | `p p p` | only fully periodic |
| `lattice` | `fcc RHO` or `sc RHO` | density in reduced units |
| `region` | `ID block xlo xhi ylo yhi zlo zhi` | cubic only |
| `create_box` | `N_TYPES region-ID` | |
| `create_atoms` | `TYPE box` | fills the lattice |
| `read_data` | `FILE` | LAMMPS data file |
| `pair_style` | `lj/cut RC` / `soft RC [A]` / `soft/sphere` / `born/mayer` | |
| `pair_coeff` | `I J EPS SIGMA` | stored per type pair |
| `pair_modify` | `shift yes/no` `tail yes/no` | |
| `neighbor` | `SKIN bin` / `SKIN nsq` | `bin` = cell list (needs ≥3 cells/dim, else auto-fallback) |
| `neigh_modify` | … | accepted, ignored |
| `timestep` | `DT` | mutates the live integrator's dt |
| `velocity` | `all create T SEED` / `all scale T` / `all zero linear` | |
| `fix` | see fix table below | |
| `unfix` / `uncompute` / `undump` | `ID` | |
| `compute` | `temp` / `pressure` / `rdf [NBINS]` / `msd` / `vacf` / `structure_factor` / `conserved/energy` | analysis computes auto-added at first `run` |
| `thermo` | `N` | console/log interval |
| `thermo_style` | `custom K1 K2 ...` / `default` | |
| `dump` | `ID all xyz/thermo/energy EVERY FILE` | |
| `dump_modify` | … | accepted, mostly ignored |
| `run` | `N_STEPS` | first `run` counts as equilibration |
| `minimize` | `ETOL FTOL MAXITER MAXEVAL` | steepest descent; removes other integrators |
| `variable`, `print`, `include`, `set_output_dir`, `seed`, `reset_timestep` | | |

### Fix styles

| Style | Syntax | Ensemble / use |
|---|---|---|
| `nve` | `fix ID all nve` | NVE integrator |
| `temp/rescale` | `fix ID all temp/rescale N Tstart Tstop window fraction` | equilibration (LAMMPS argument order; `window` is an absolute T) |
| `berendsen` | `fix ID all berendsen N Tstart Tstop tau_T` | equilibration |
| `nvt` / `nose_hoover` | `fix ID all nvt Tstar [Q]` | canonical NVT |
| `langevin` | `fix ID all langevin Tstar gamma [seed]` | NVT; **self-integrating — replaces `nve`** |
| `press/berendsen` | `fix ID all press/berendsen P_target tau_P [beta]` | equilibration barostat (cubic box) |
| `momentum` | `fix ID all momentum N` | remove COM drift every N steps |

The full reference is in
[`docs/input_script_reference.md`](docs/input_script_reference.md).

---

## 6. Python API

Everything is driven by a single dataclass plus the `Simulation` facade:

```python
from lj_md_package import SimConfig, Simulation
cfg = SimConfig(N=108, rho_star=0.8442, T_star=0.722, ...)
sim = Simulation.from_config(cfg)
```

`from_config` wires: box, FCC/SC/random lattice, Maxwell–Boltzmann
velocities (COM removed), `lj/cut` pair, a safe neighbor list, the NVE fix,
the chosen thermostat (id `"thermostat"`), and gated analysis computes.

Key `SimConfig` fields (defaults in parentheses):

| Group | Field | Meaning |
|---|---|---|
| System | `N` (108), `rho_star` (0.8442), `T_star` (0.722) | state point |
| Force field | `r_cut` (2.5), `shift_potential` (True), `use_tail_corrections` (True) | |
| Neighbor list | `neighbor_style` (`"verlet"`/`"cell"`), `r_skin` (0.3) | |
| Integration | `dt` (0.005), `n_steps` (20000), `n_equil` (5000) | |
| Thermostat | `thermostat_type` (`"rescale"`), `rescale_interval`, `tau_T`, `nose_hoover_Q`, `langevin_gamma` | |
| Sampling starts | `rdf_start`, `msd_start`, `vacf_start` (all 5000) | first production step used by computes |
| Sampling rates | `sample_interval`, `rdf_interval`, `msd_interval`, `vacf_interval` | |
| Analysis sizes | `rdf_n_bins` (200), `msd_length` (2000), `vacf_length` (500) | |
| Output | `output_dir` (`"output"`), `traj_interval`, `generate_plots`, `plot_dpi` | |
| Initialization | `lattice_type` (`"fcc"`), `random_seed` (42) | |
| Validation | `energy_drift_tol`, `strict` | |

Useful runtime objects:

- `sim.fixes` / `sim.add_fix(...)` / `sim.remove_fix(id)`
- `sim.computes` / `sim.add_compute(...)` (computes obey `step >= start`)
- `sim.dumps` / `sim.add_dump(...)`
- `sim._thermo.as_arrays()` — full time series as NumPy dict
- `sim.state.force_result` — forces, truncated PE, virial of last step

---

## 7. Outputs

Written into `cfg.output_dir` (default `output/`):

| File | Contents |
|---|---|
| `thermo.csv` | step, time, T*, P*, PE, KE, E_total, E/N, ρ, virial, min_r, max_f, COM momentum |
| `energies.csv` | KE / PE / E_total per sample |
| `trajectory.xyz` | extended-XYZ (VMD/OVITO-compatible) frames |
| `rdf.csv` | g(r), production-gated |
| `rdf_theta.npz` | angular g(r, cosθ) |
| `msd.csv` | mean-squared displacement vs lag time |
| `vacf.csv` | normalized velocity autocorrelation |
| `sk.csv` | S(k) at allowed k-shells (no NaN gaps) |
| `simulation.log` | full run log |
| `summary.txt` | validation report (block averages, autocorrelation times, energy drift, literature comparison) |
| `*.png` | thermo series, production panel, energy drift, T histogram, g(r), S(k), structure panel, MSD/VACF/transport figures |

---

## 8. Physics background

- **Truncation & shift**: `U(r) = U_LJ(r) − U_LJ(r_c)` for `r < r_c`.
- **Tail corrections** (`use_tail_corrections=True`, the default):

  ```
  u_tail/N = (8πρ/3) [ (1/3) r_c⁻⁹ − r_c⁻³ ]
  P_tail   = (16πρ²/3) [ (2/3) r_c⁻⁹ − r_c⁻³ ]
  ```

  There is a tail-correction mismatch concept you must keep straight:
  **with** tails you simulate the *full* LJ fluid; **without** tails you
  simulate the LJTS fluid. The validation report selects the matching
  reference set automatically (`units/reference.py`).

- **Pressure**: `P = ρT + W/(3V) + P_tail`, with `W = Σ r_ij·F_ij`.
- **Temperature**: `T = 2 KE / (3N − 3)` (COM momentum removed).
- **Integration**: velocity Verlet; BAOAB for Langevin; Strang-split NH.
- **Structure**: g(r), S(k) from exact k-summation at k = 2πn/L.
- **Diffusion**: Einstein from MSD slope; Green–Kubo from the
  *un-normalized* VACF integral.

Details: [`docs/theory.md`](docs/theory.md),
[`docs/thermostats_barostats.md`](docs/thermostats_barostats.md).

---

## 9. Thermostats and barostat

| Thermostat | Canonical sample? | Deterministic? | Use for |
|---|---|---|---|
| velocity rescale | no | yes | fast equilibration |
| Berendsen | no | yes | gentle equilibration |
| Langevin (BAOAB) | yes | no (stochastic) | melting + NVT production |
| Nosé–Hoover (MTTK) | yes | yes | NVT production when dynamics matter |

Rules of thumb:

- Never run production under rescale/Berendsen — remove them first.
- `fix langevin` **replaces** `fix nve` (it does the integration itself); the
  engine refuses two integrating fixes at once.
- The barostat is for equilibration only; unfix it and run NVT/NVE for
  production.

---

## 10. Validation and the benchmark caveat

References used by `summary.txt` at the benchmark state point
(ρ* = 0.8442, T* = 0.722):

| Tails | Model | E/N | P* | Source |
|---|---|---|---|---|
| off | LJTS, r_c=2.5 | −4.846 | 1.138 | Lustig et al., Mol. Phys. (2021) |
| on | full LJ | −6.236 | 1.731 | Allen & Tildesley (2017) |

**Important physics caveat.** At N = 108 this point sits essentially on the
LJTS solid–liquid coexistence band. Without a melt-anneal protocol the FCC
start freezes and the measured P* ≈ 0, E/N ≈ −4.6 are *coexistence*
averages — correct NVE, wrong phase. The summary prints an explicit caveat
when the state point is in this regime. For clean liquid benchmarks either
follow the melt-anneal-NVE recipe (§4d) and verify the melt, or pick a
safely-liquid state point (e.g. ρ* = 0.8442, T* = 1.0, or ρ* = 0.8,
T* = 1.0), or move to N ≥ 864.

Typical sanity numbers for a well-prepared NVE run at dt = 0.005:
`|dE/E0| ≲ 1e-4` over 20k steps.

---

## 11. Testing

```bash
python -m pytest tests/ -q
```

Covers: force/potential sign consistency for every pair style (finite
difference), CellList vs VerletNL exactness at N=864 and the safe-fallback
at small boxes, engine regressions (Python API moves atoms, script Langevin
moves atoms, fix-argument order, MIC without a neighbor list, minimizer
descends), ideal-gas g(r)/S(k), Green–Kubo normalization, BAOAB kinetic-
temperature accuracy, and Nosé–Hoover conserved-Hamiltonian behavior.

---

## 12. Project layout

```
lj_md_package/
├── src/lj_md_package/
│   ├── core/         SimConfig, Box, Atoms, Simulation, State
│   ├── pairs/        lj/cut, soft, soft/sphere, born/mayer
│   ├── neighbors/    VerletNL, CellList
│   ├── integrators/  VelocityVerlet, MinimizeSD
│   ├── fixes/        nve, temp/rescale, berendsen, langevin, nose_hoover,
│   │                 press/berendsen, momentum
│   ├── computes/     temp, pressure, rdf, rdf_theta, msd, vacf,
│   │                 structure_factor, thermo, conserved/energy
│   ├── dumps/        xyz, thermo, energy, lammps-data I/O
│   ├── initialize/   lattices, MB velocities, MC liquid generator
│   ├── input/        LAMMPS-style parser (tokenizer, registry, commands)
│   ├── analysis/     block averaging, autocorrelation, validation report
│   ├── units/        lj / real / metal / si + literature references
│   ├── cli.py        `lj-md` entry point & benchmark()
│   └── plotting.py
├── examples/         benchmark / supercritical / T-scan driver
├── tests/            pytest suite
├── docs/             theory, thermostat and input-script references
├── REVIEW.md         code-review findings & fix log (this release)
├── RUN_COMMANDS.txt  copy-paste command cheat sheet
└── in.lj             sample input script
```

---

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `No integrating fix (nve / langevin) present` | you removed `nve`; add it back (`Simulation.add_fix(FixNVE("nve","all",sim.integrator))`) |
| `Multiple integrating fixes present` | both `nve` and `langevin` defined — remove one |
| `CellList requires L/r_list >= 3` | box too small for `neighbor ... bin`; use `nsq` (auto-fallback exists in factories) |
| FCC never melts at T*=0.722 | use melt-anneal-NVE (§4d) — expected physics, not a bug |
| P* ≫ literature errors near 0.722/0.8442 | two-phase box at N=108 (see §10) |
| Large energy drift | reduce dt (try 0.002); check shift/tail consistency |
| `rdf_start must be in [0, n_steps]` | shorten-incompatible config; clamp starts ≤ n_steps |

Log file: `output_dir/simulation.log`. Increase verbosity with
`lj-md --log-level DEBUG`.

---

## 14. References

1. L. Verlet, *Phys. Rev.* **159**, 98 (1967).
2. M. P. Allen & D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
   Oxford (2017).
3. D. Frenkel & B. Smit, *Understanding Molecular Simulation*, 2nd ed.,
   Academic Press (2002).
4. R. Lustig et al., *Mol. Phys.* (2021) — LJTS equation of state (r_c=2.5).
5. B. Leimkuhler & C. Matthews, *J. Chem. Phys.* **138**, 174102 (2013)
   (BAOAB).
6. G. J. Martyna, M. L. Klein, M. E. Tuckerman, *J. Chem. Phys.* **97**,
   2635 (1992) (Nosé–Hoover splitting).
7. H. J. C. Berendsen et al., *J. Chem. Phys.* **81**, 3684 (1984).
8. A. Rahman, *Phys. Rev.* **136**, A405 (1964).
9. H. Flyvbjerg & H. G. Petersen, *J. Chem. Phys.* **91**, 461 (1989).


