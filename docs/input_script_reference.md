# LAMMPS-Style Input Script Reference

The `lj-md` package parses a subset of the LAMMPS input script language.
Scripts are plain text files processed line-by-line.  Blank lines and
comments beginning with ``#`` are ignored.

## Quick Example

```lammps
units         lj                     # LJ reduced units
lattice       fcc 0.8442             # FCC lattice at rho*=0.8442
create_atoms  1 box                  # fill box with type-1 atoms
pair_style    lj/cut 2.5             # LJ truncated+shifted at r_c=2.5
pair_coeff    1 1 1.0 1.0            # eps=1, sigma=1
neighbor      0.3 bin                # Verlet skin = 0.3, linked cells
velocity      all create 0.722 42    # Maxwell-Boltzmann at T*=0.722
fix           1 all nve              # NVE integrator (microcanonical)
fix           2 all temp/rescale 100 0.722 0.722 0.1 1.0   # thermostat
thermo        10                     # log every 10 steps
dump          1 all xyz 200 output/traj.xyz   # trajectory
run           5000                   # equilibrate
unfix         2                      # remove thermostat
run           15000                  # NVE production
```

Run it:

```bash
lj-md in.lj
```

---

## Command Reference

### `units`

**Syntax:** `units lj | real | metal | si`

Sets the unit system.  Only `lj` (LJ reduced units) is fully
implemented.  `real` and `metal` are accepted as labels but
internally operate in LJ units.

```lammps
units lj
```

---

### `atom_style`

**Syntax:** `atom_style atomic`

Only `atomic` (point particles) is supported.  Other styles are
accepted silently.

```lammps
atom_style atomic
```

---

### `boundary`

**Syntax:** `boundary p p p`

Sets periodic boundary conditions.  Only `p p p` is supported
(periodic in all three dimensions).

```lammps
boundary p p p
```

---

### `lattice`

**Syntax:** `lattice (fcc|sc) RHO_STAR`

Defines the crystal lattice and number density.  `fcc` = face-centred
cubic, `sc` = simple cubic.  The density determines the box size at
`create_box` / `create_atoms` time.

```lammps
lattice fcc 0.8442
lattice sc 0.300
```

---

### `region`

**Syntax:** `region ID block XLO XHI YLO YHI ZLO ZHI`

Defines a rectangular box region.  Only `block` style is supported.
The default region is the simulation box.

```lammps
region mybox block 0.0 5.0388 0.0 5.0388 0.0 5.0388
```

---

### `create_box`

**Syntax:** `create_box N_TYPES REGION_ID`

Creates the simulation box.  N_TYPES is the number of atom types
(only `1` is tested).

```lammps
create_box 1 box
```

---

### `create_atoms`

**Syntax:** `create_atoms TYPE box`

Fills the box with atoms on the pre-set lattice.  Only one type
is supported; atoms are placed at FCC or SC lattice points.

```lammps
lattice fcc 0.8442
create_atoms 1 box
```

---

### `read_data`

**Syntax:** `read_data FILENAME`

Reads a LAMMPS data file (Atoms, Box, Masses sections).

```lammps
read_data conf.data
```

---

### `pair_style`

**Syntax:** `pair_style lj/cut RCUT | soft RCUT [A]`

Sets the pair interaction.  `lj/cut` is the truncated (and optionally
shifted) Lennard-Jones potential.  `soft` is a purely repulsive cosine
pair used for minimisation.

```lammps
pair_style lj/cut 2.5
pair_style soft 1.0 1.0
```

---

### `pair_coeff`

**Syntax:** `pair_coeff TYPE_I TYPE_J EPS SIGMA`

Sets the epsilon and sigma coefficients for a pair of atom types.
Coefficients are stored per-type pair.

```lammps
pair_coeff 1 1 1.0 1.0
```

---

### `pair_modify`

**Syntax:** `pair_modify (shift yes|no) (tail yes|no)`

Controls the truncation-shift and long-range tail corrections for the
pair potential:

- `shift yes` — shift potential so U(r_c) = 0
- `tail yes` — apply analytic g(r)=1 tail corrections to U and P

```lammps
pair_modify shift yes tail yes
```

---

### `neighbor`

**Syntax:** `neighbor SKIN (bin|nsq)`

Sets the Verlet skin distance and the neighbor-list algorithm:

- `skin` — buffer distance for neighbor list rebuilds
- `bin` / `cell` — linked-cell list (fast for N > 500)
- `nsq` — Verlet O(N²) half-list (fast for N < 500, default)

```lammps
neighbor 0.3 bin
neighbor 0.4 nsq
```

---

### `neigh_modify`

**Syntax:** `neigh_modify ...`

Accepted but all sub-options are currently ignored.  Neighbour lists
rebuild automatically when any atom moves more than skin/2.

---

### `timestep`

**Syntax:** `timestep DT`

Sets the integration time step in LJ reduced time units.

```lammps
timestep 0.005
```

---

### `velocity`

**Syntax:**
- `velocity all create T_SEED dist gaussian`
- `velocity all scale T_TARGET`
- `velocity all zero linear`

Initialises or modifies velocities:

- `create` — assign Maxwell-Boltzmann velocities at temperature T.
  The `dist gaussian` keyword is accepted but ignored (always Gaussian).
- `scale` — rescale all velocities to target temperature T.
- `zero linear` — remove centre-of-mass linear momentum.

```lammps
velocity all create 0.722 42
velocity all scale 1.500
velocity all zero linear
```

---

### `fix`

**Syntax:** `fix ID GROUP STYLE ARGS...`

Attaches a thermostat, barostat, or integrator.

| Style | Syntax | Description |
|-------|--------|-------------|
| `nve` | `fix ID all nve` | NVE (microcanonical) integrator |
| `nvt` | `fix ID all nvt T_STAR [Q]` | Nose-Hoover NVT, MTTK split (optional mass Q) |
| `nose_hoover` | `fix ID all nose_hoover T_STAR [Q]` | Same as `nvt` |
| `temp/rescale` | `fix ID all temp/rescale N TSTART TSTOP [WINDOW] [FRAC]` | Velocity rescale every N steps (LAMMPS order, WINDOW is absolute) |
| `berendsen` | `fix ID all berendsen N TSTART TSTOP TAU_T` | Berendsen weak-coupling thermostat |
| `press/berendsen` | `fix ID all press/berendsen P_TARGET TAU_P [BETA]` | Isotropic Berendsen barostat (equilibration) |
| `langevin` | `fix ID all langevin T_STAR GAMMA [SEED]` | Langevin (BAOAB) stochastic thermostat; self-integrating, replaces `nve` |
| `momentum` | `fix ID all momentum N` | Zero COM motion every N steps |

`fix langevin` performs both the position and velocity updates; adding it
automatically *replaces* any existing `nve` fix (a warning is logged). Using
both is rejected by `run`.

```lammps
fix 1 all nve
fix 2 all temp/rescale 100 0.722 0.722 0.1 1.0     # window=0.1, fraction=1.0
fix 2 all nvt 0.722 2.0
fix 2 all berendsen 1 0.722 0.722 0.5
fix 2 all langevin 0.722 1.0 42                     # replaces fix 1
fix 2 all press/berendsen 1.14 5.0 0.1
fix 3 all momentum 1000
```

---

### `unfix`

**Syntax:** `unfix FIX_ID`

Removes a previously added fix.

```lammps
unfix 2
```

---

### `compute`

**Syntax:** `compute ID GROUP STYLE [ARGS]`

Registers a per-step observable calculator.

| Style | Description |
|-------|-------------|
| `temp` | Running temperature average |
| `pressure` | Pressure from the virial route |
| `rdf [N_BINS]` | Radial distribution function g(r) |
| `msd` | Mean-squared displacement |
| `vacf` | Velocity autocorrelation function |
| `structure_factor` | Static structure factor S(k) |
| `conserved/energy` | Nose-Hoover conserved Hamiltonian |

Analysis computes (`rdf`, `msd`, `vacf`, `structure_factor`) are added
automatically at the first `run` command if none are present.

```lammps
compute myrdf all rdf 200
compute mymsd all msd
```

---

### `uncompute`

**Syntax:** `uncompute COMPUTE_ID`

Removes a previously added compute.

```lammps
uncompute myrdf
```

---

### `thermo`

**Syntax:** `thermo N`

Sets the thermodynamic output interval.  Every N steps a line with
temperature, pressure, energies, etc. is logged and written to
`thermo.csv` and `energies.csv`.

```lammps
thermo 10
```

---

### `thermo_style`

**Syntax:** `thermo_style custom KEY1 KEY2 ...`
**Syntax:** `thermo_style default`

Controls which quantities appear in the thermo output.  `default`
uses a built-in set of keys.  `custom` expects a space-separated
list of keys:

```
step, time, T, P, PE, KE, E_total, E_per_atom, density, virial
```

```lammps
thermo_style custom step T P E_total
```

---

### `dump`

**Syntax:** `dump ID GROUP (xyz|thermo|energy) EVERY FILENAME`

Writes data periodically:

- `xyz` — extended XYZ trajectory (VMD/OVITO compatible)
- `thermo` — CSV: step, time, T, P, PE, KE, E_total, ...
- `energy` — CSV: step, time, KE, PE, E_total

```lammps
dump 1 all xyz 200 output/traj.xyz
dump 2 all thermo 10 output/thermo.csv
dump 3 all energy 10 output/energies.csv
```

---

### `dump_modify`

**Syntax:** `dump_modify ...`

Accepted but most sub-options are currently ignored.

---

### `undump`

**Syntax:** `undump DUMP_ID`

Removes a previously added dump and closes its file.

```lammps
undump 1
```

---

### `run`

**Syntax:** `run N_STEPS`

Runs the simulation for N_STEPS integration steps.  The first `run`
command is automatically treated as equilibration (thermostat active).
Subsequent `run` commands are production (NVE unless another fix is
active).

```lammps
run 5000      # equilibration
unfix 2
run 15000     # production
```

---

### `minimize`

**Syntax:** `minimize ENERGY_TOL FORCE_TOL MAX_ITER MAX_STEPS`

Performs energy minimisation via steepest descent (`r += alpha F`).  Any
existing integrating fix is removed first; re-add `fix nve` afterwards to
continue dynamics.  A `soft` start is a good way to untangle overlaps before
switching to the hard LJ core.

```lammps
pair_style soft 1.0 1.0
pair_coeff 1 1 10.0 1.0
minimize 1e-6 1e-8 100 1000
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
```

---

### `print`

**Syntax:** `print STRING`

Logs a message.

```lammps
print "Starting production run"
```

---

### `variable`

**Syntax:** `variable NAME (equal|string) VALUE`

Defines a named variable for `${NAME}` substitution in subsequent
commands.

```lammps
variable T equal 0.722
variable nsteps equal 5000
timestep 0.005
run ${nsteps}
```

---

### `include`

**Syntax:** `include FILENAME`

Reads and executes commands from another file (relative paths are
resolved against the parent input file's directory).

```lammps
include potential.params
```

---

### `set_output_dir`

**Syntax:** `set_output_dir PATH`

Changes the output directory for all subsequent file output.

```lammps
set_output_dir results/run1
```

---

### `seed`

**Syntax:** `seed INTEGER`

Sets the random number seed.

```lammps
seed 12345
```

---

### `reset_timestep`

**Syntax:** `reset_timestep [STEP]`

Resets the internal step counter.  If no argument is given, resets
to 0.

```lammps
reset_timestep 0
```

---

## Notes

- All quantities are in **LJ reduced units** (length = σ, energy = ε,
  time = σ √(m/ε), temperature = ε/k_B).
- The first `run` is automatically equilibration.  Switch off
  the thermostat with `unfix` before the production `run`.
- Default analysis computes (RDF, RDF-theta, MSD, VACF, S(k)) are
  injected at the first `run` if none were added explicitly with
  `compute`.  All output goes to `output/`.
- Plotting is automatic after the simulation finishes.  Use
  `lj-md --no-plots` to skip.
