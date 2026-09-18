# Thermostats, Barostats, And Ensembles In Molecular Dynamics

## Thermodynamic Ensembles

An ensemble is a collection of microstates consistent with given macroscopic
constraints. MD simulations sample different ensembles depending on which
quantities are held constant.

| Ensemble | Fixed | Symbol        | Physical meaning                     |
|----------|-------|---------------|--------------------------------------|
| **NVE**  | N,V,E | Microcanonical | Isolated system, no heat exchange    |
| **NVT**  | N,V,T | Canonical    | Constant temperature (heat bath)    |
| **NPT**  | N,P,T | Isobaric-isothermal | Fixed pressure and temperature |
| **NPH**  | N,P,H | Isobaric-isoenthalpic | Fixed pressure and enthalpy |

By default our package runs in **NVE** (no thermostat). With a thermostat
fix added, the ensemble becomes **NVT**. A barostat fix would produce **NPT**
or **NPH**.

---

## NVE Ensemble (Microcanonical)

The simplest ensemble: Newton's equations are integrated without any
thermostat or barostat. Total energy $E = K + U$ is conserved up to
integration error.

**When to use:**
- Production runs after equilibration
- Measuring dynamics (diffusion, viscosity, thermal conductivity)
- Verifying code correctness (energy drift check)

**Implementation in our package:**
```python
from lj_md_package.fixes.nve import FixNVE
sim.add_fix(FixNVE("nve", "all", integrator))
```

The NVE fix simply calls the integrator's `initial_integrate` and
`final_integrate` methods with no thermostat modification.

---

## Thermostats (NVT Ensemble)

A thermostat controls temperature by modifying particle velocities to target
a desired kinetic energy:

$$\left\langle\frac{1}{2}\sum_i m_i v_i^2\right\rangle = \frac{3}{2}Nk_BT$$

The instantaneous temperature is:

$$T(t) = \frac{2}{3Nk_B}\sum_i \frac{1}{2} m_i v_i^2(t)$$

---

### 1. Velocity Rescale (Andersen-style / Berendsen-like)

The simplest thermostat: at every $N_{rescale}$ steps, all velocities are
uniformly scaled to match the target temperature.

**Algorithm:**
$$\lambda = \sqrt{\frac{T_{target}}{T(t)}}$$
$$\mathbf{v}_i \leftarrow \lambda \mathbf{v}_i$$

**Properties:**
- **Pros:** Simple, fast, guaranteed convergence
- **Cons:** Does not sample the canonical ensemble correctly (velocity
  fluctuations are suppressed)
- **Use case:** Rapid equilibration; **never use for production**

**Our implementation:**
```python
sim.add_fix(FixTempRescale("thermostat", "all",
                           T_target=0.722,
                           fraction=0.1))
```
Temperature is checked every step; if $|T - T_{target}| > \text{tol}$,
velocities are scaled by $1 + \text{fraction} \cdot (T_{target}/T - 1)$.

---

### 2. Berendsen Thermostat (Weak Coupling)

The Berendsen thermostat couples the system to an external heat bath with
weak first-order kinetics:

**Algorithm:**
$$\lambda = \sqrt{1 + \frac{\Delta t}{\tau_T}
\left(\frac{T_{target}}{T(t)} - 1\right)}$$
$$\mathbf{v}_i \leftarrow \lambda \mathbf{v}_i$$

where $\tau_T$ is the coupling time constant (in timesteps). A large
$\tau_T$ gives weak coupling (slow equilibration, less perturbation).
A small $\tau_T$ gives strong coupling.

**Properties:**
- **Pros:** Smooth, tunable coupling strength
- **Cons:** Does **not** sample the canonical ensemble (velocity
  distribution is artificially narrowed)
- **Use case:** Only for equilibration, never for production

**Our implementation:**
```python
from lj_md_package.fixes.berendsen import FixBerendsen
sim.add_fix(FixBerendsen("berend", "all",
                         T_target=0.722,
                         tau_T=100.0))
```

At every step, the scale factor $\lambda$ is applied. The default
$\tau_T = 100$ timesteps gives gentle coupling.

---

### 3. Langevin Thermostat (BAOAB Integrator)

The Langevin thermostat adds stochastic (random) and dissipative (friction)
forces to model interaction with an implicit heat bath:

$$m\ddot{\mathbf{r}} = \mathbf{F} - m\gamma\dot{\mathbf{r}} + \mathbf{R}(t)$$

where:
- $\gamma$ is the friction coefficient ($1/\gamma$ sets the relaxation time)
- $\mathbf{R}(t)$ is Gaussian white noise (the "random force"):
  $$\langle R_\alpha(t) \rangle = 0$$
  $$\langle R_\alpha(t) R_\beta(t') \rangle = 2m\gamma k_BT \, \delta_{\alpha\beta} \delta(t-t')$$

**BAOAB / GJF discretization (Leimkuhler-Matthews):**

Our implementation uses the BAOAB splitting, whose Ornstein-Uhlenbeck
substep is integrated *exactly*:

1. Half-kick: $\mathbf{v} \mathrel{+}= \frac{\Delta t}{2m}\mathbf{F}(t)$
2. Half-drift: $\mathbf{r} \mathrel{+}= \frac{\Delta t}{2}\mathbf{v}$
3. Exact OU: $\mathbf{v} = c_1 \mathbf{v} + c_2 \boldsymbol{\xi}$,
   $c_1 = e^{-\gamma\Delta t}$, $c_2 = \sqrt{k_B T (1-c_1^2)/m}$,
   $\boldsymbol{\xi}\sim\mathcal{N}(0,1)$
4. Half-drift: $\mathbf{r} \mathrel{+}= \frac{\Delta t}{2}\mathbf{v}$
5. Compute $\mathbf{F}(t+\Delta t)$, then half-kick:
   $\mathbf{v} \mathrel{+}= \frac{\Delta t}{2m}\mathbf{F}(t+\Delta t)$

Unlike the older BBK scheme, the measured kinetic temperature has **no
$O(\gamma\Delta t)$ bias** (measured +1% with BBK at $\gamma=1$,
$\Delta t=0.005$; $<0.5\%$ with BAOAB, statistical noise included).

**Properties:**
- **Pros:** Correct canonical sampling (NVT)
- **Pros:** Can also melt crystals very effectively (stochastic kicks
  break lattice symmetry)
- **Cons:** Dynamics are not Hamiltonian (no time-reversibility preserved)
- **Use case:** Equilibration AND production (if NVT is acceptable);
  best thermostat for melting solids

**Selection of $\gamma$:**
- $\gamma = 0.1$ -- weak coupling, close to Newtonian dynamics
- $\gamma = 1.0$ -- moderate coupling (our default for melting)
- $\gamma = 10.0$ -- strong coupling, overdamped dynamics

**Important:** The Langevin fix replaces the NVE fix. It IS the integrator
for position updates. Do NOT use FixLangevin alongside FixNVE.

**Our implementation:**
```python
from lj_md_package.fixes.langevin import FixLangevin
lang = FixLangevin("langevin", "all",
                   T_target=0.722,
                   gamma=1.0,
                   seed=42)
sim.add_fix(lang)      # replaces any FixNVE (it integrates by itself)
```

`FixLangevin` is fully self-contained (BAOAB performs the position updates);
`set_integrator()` remains as a no-op for backward compatibility.
`Tstop` may be given to ramp the target temperature linearly across a
`run` block.

---

### 4. Nose-Hoover Thermostat (Extended Lagrangian)

The Nose-Hoover thermostat introduces an auxiliary variable $s$ (the heat
bath) with an effective mass $Q = N_{dof} k_B T \tau_T^2$:

**Extended Hamiltonian:**
$$\mathcal{H} = \sum_i \frac{p_i^2}{2ms^2} + U(\mathbf{r}) + \frac{p_s^2}{2Q}
+ N_{dof} k_B T \ln s$$

**Equations of motion (Nose-Hoover formulation):**
$$\dot{\mathbf{r}}_i = \frac{\mathbf{p}_i}{m}$$
$$\dot{\mathbf{p}}_i = \mathbf{F}_i - \xi \mathbf{p}_i$$
$$\dot{\xi} = \frac{1}{Q}\left(\sum_i \frac{p_i^2}{m} - N_{dof} k_B T\right)$$
where $\xi = \dot{s}/s$ is the friction coefficient.

**Properties:**
- **Pros:** Rigorous canonical sampling (NVT) -- proven to be ergodic
- **Pros:** Deterministic and time-reversible
- **Cons:** More complex parameter tuning ($\tau_T$)
- **Cons:** Can have oscillations if $\tau_T$ is poorly chosen
- **Use case:** Production runs requiring NVT statistics

**Parameter selection:**
$$\tau_T \approx \sqrt{Q/(N_{dof} k_B T)}$$

A good starting point is $\tau_T = 100$ timesteps.

**Our implementation:**
```python
from lj_md_package.fixes.nvt_nose_hoover import FixNVT
sim.add_fix(FixNVT("nh", "all", T_target=0.722, Q=50.0))
```

The implementation uses a time-reversible Strang (MTTK-style) splitting
around the velocity-Verlet step: quarter-$\xi$ / half-scale / [VV] /
half-scale / quarter-$\xi$, and the fix reports the bath energy
$\frac{1}{2}Q\xi^2 + N_{dof} k_B T\,\eta$ (with $\eta=\int\xi dt$) through
``conserved_energy()`` so that ``compute conserved/energy`` assembles
the full extended Hamiltonian $H_{NH} = K + U + \text{bath}$ exactly once.

---

### Comparison Of Thermostats

| Thermostat       | Canonical? | Deterministic? | Dynamics? | Best For                       |
|------------------|------------|----------------|-----------|--------------------------------|
| Velocity Rescale | No         | Yes            | No        | Quick equilibration            |
| Berendsen        | No         | Yes            | No        | Gentle equilibration           |
| Langevin (BAOAB) | Yes        | No (stochastic)| Yes       | Melting + NVT production       |
| Nose-Hoover      | Yes        | Yes            | Yes       | NVT production (need dynamics) |

---

### Momentum Removal

After thermostat operations (especially rescale and Berendsen), the total
momentum may drift due to roundoff. The momentum fix removes center-of-mass
motion:

$$\mathbf{v}_i \leftarrow \mathbf{v}_i - \frac{1}{M}\sum_j m_j\mathbf{v}_j$$

This is applied automatically at the start of each timestep in our code
when the fix is active:

```python
from lj_md_package.fixes.momentum import FixMomentum
sim.add_fix(FixMomentum("mom", "all"))
```

---

## Barostats (NPT Ensemble)

A barostat controls the system pressure by adjusting the simulation box
volume. Pressure control requires the virial:

$$P = \rho k_B T + \frac{W}{V}$$

where $W$ is the virial from pair forces.

### Nose-Hoover Barostat (Extended Lagrangian)

The Nose-Hoover barostat couples the box volume to an auxiliary variable
with effective mass $W_{baro} = N_{dof} k_B T \tau_P^2$:

**Equations of motion (box volume $V$ and strain rate $\eta$):**
$$\dot{\mathbf{r}}_i = \frac{\mathbf{p}_i}{m_i} + \eta (\mathbf{r}_i - \mathbf{R}_0)$$
$$\dot{\mathbf{p}}_i = \mathbf{F}_i - \xi \mathbf{p}_i - \eta \mathbf{p}_i$$
$$\dot{V} = 3\eta V$$
$$\dot{\eta} = \frac{V(P-P_{target})}{W_{baro}} - \xi\eta$$
$$\dot{\xi} = \frac{1}{Q}\left(\sum_i \frac{p_i^2}{m} - N_{dof} k_B T\right)$$

**Status in this package:**
A full Nose-Hoover barostat is not yet implemented. NPT simulations can be
approximated by pre-setting the density to known values (from experiment or
literature) and running NVT.

For the Verlet (1967) benchmark state point:
- Set $\rho^* = 0.8442$ (from literature)
- Run NVT at $T^* = 0.722$
- The pressure converges to $P^* \approx 1.138$ (LJTS reference)

This approach is standard practice in LJ simulations since the EOS is
well-characterized.

---

### Berendsen Barostat (Weak Coupling)

Similar to the Berendsen thermostat, the Berendsen barostat scales the
box and particle positions isotropically:

$$\mu = \left[1 - \beta\frac{\Delta t}{\tau_P}(P_{target} - P(t))\right]^{1/3}$$
$$\mathbf{r}_i \leftarrow \mu \mathbf{r}_i$$
$$L \leftarrow \mu L$$

**Status in this package:** implemented as
``fix ID all press/berendsen P_target tau_P [beta]`` on cubic boxes.  The
box object, neighbor list and positions are updated in place.  Like the
Berendsen thermostat it does not sample the NPT distribution exactly - use
it for equilibration, then unfix and run NVT/NVE production at the mean
density.  A Nose-Hoover barostat remains future work.

---

## Equilibration Protocol (Practical Guide)

The recommended workflow for producing a well-equilibrated LJ liquid:

```
Step 1: Create initial configuration (FCC lattice)
                               |
Step 2: Melt at high T (Langevin, T=2.0, 5000+ steps)
                               |
Step 3: Anneal to target T (Langevin, 10000+ steps)
                               |
Step 4: NVE production (remove thermostat, 50000+ steps)
```

**Why this works:**

1. The FCC crystal at $\rho^*=0.8442$, $T^*=0.722$ is metastable -- the
   system is below the melting point and will not spontaneously melt with
   a deterministic thermostat.

2. The Langevin thermostat at $T^*=2.0$ provides stochastic kicks that
   break lattice symmetry, melting the crystal.

3. Gradual annealing (cooling) from $T^*=2.0$ to $T^*=0.722$ prevents the
   system from getting stuck in glassy configurations.

4. The final NVE run gives true microcanonical dynamics for measuring
   transport properties.

**Failure mode:** If a velocity rescale or Berendsen thermostat is used
directly at $T^*=0.722$ on an FCC lattice, the system stays ordered and
the pressure will be negative (incorrect). This is a known feature of the
LJ state point, not a code bug.

---

## Temperature And Pressure Calculation Details

### Instantaneous Temperature

The instantaneous temperature is computed from kinetic energy:

$$T(t) = \frac{2}{3Nk_B}\sum_{i=1}^N \frac{1}{2} m_i \mathbf{v}_i^2$$

In reduced units ($k_B=1$, $m=1$):

$$T(t) = \frac{1}{3N}\sum_{i=1}^N \mathbf{v}_i^2$$

### Instantaneous Pressure

$$P(t) = \rho k_B T(t) + \frac{1}{3V}\sum_{i<j} \mathbf{r}_{ij} \cdot \mathbf{F}_{ij}$$

The second term is the **virial** contribution divided by $3V$.

### Production Averages

During production, block averages (Flyvbjerg-Petersen) are reported with
error bars:

$$\langle T \rangle \pm \sigma_T \quad \langle P \rangle \pm \sigma_P
\quad \langle E/N \rangle \pm \sigma_E$$

Autocorrelation times $\tau_{int}$ are computed for each observable and
printed in the validation report.

---

## Summary: Which Fixes To Use And When

| Goal                         | Fix                                      | Ensemble |
|------------------------------|------------------------------------------|----------|
| Energy conservation check    | `FixNVE`                                 | NVE      |
| Quick equilibration          | `FixTempRescale` + `FixNVE`             | NVT (approx) |
| Gentle equilibration         | `FixBerendsen` + `FixNVE`               | NVT (approx) |
| Melting a crystal            | `FixLangevin` alone                     | NVT      |
| NVT production (stochastic)  | `FixLangevin` alone                     | NVT      |
| NVT production (deterministic)| `FixNoseHoover` + `FixNVE`             | NVT      |
| Remove COM drift             | `FixMomentum`                            | any      |

**Note:** FixLangevin replaces FixNVE -- it handles both the thermostat and
the integration. All other thermostats sit alongside FixNVE and modify
velocities after the NVE integration step.

---

## References

1. Andersen, H.C. (1980). Molecular dynamics simulations at constant
   pressure and/or temperature. *J. Chem. Phys.*, **72**, 2384.

2. Berendsen, H.J.C. et al. (1984). Molecular dynamics with coupling to an
   external bath. *J. Chem. Phys.*, **81**, 3684.

3. Nose, S. (1984). A unified formulation of the constant temperature
   molecular dynamics methods. *J. Chem. Phys.*, **81**, 511.

4. Hoover, W.G. (1985). Canonical dynamics: equilibrium phase-space
   distributions. *Phys. Rev. A*, **31**, 1695.

5. Schneider, T. & Stoll, E. (1978). Molecular-dynamics study of a
   three-dimensional one-component model for distortive phase transitions.
   *Phys. Rev. B*, **17**, 1302.

6. Brunger, A., Brooks, C.L., & Karplus, M. (1984). Stochastic boundary
   conditions for molecular dynamics simulations of ST2 water.
   *Chem. Phys. Lett.*, **105**, 495.

7. Frenkel, D. & Smit, B. (2002). *Understanding Molecular Simulation*,
   2nd ed. Academic Press.
