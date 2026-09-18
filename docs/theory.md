# Theory Of Molecular Dynamics Simulation

## Lennard-Jones Potential

The Lennard-Jones (LJ) 12-6 potential models the interaction between a pair of
neutral atoms or molecules:

$$U_{LJ}(r) = 4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12} -
\left(\frac{\sigma}{r}\right)^{6}\right]$$

- $\epsilon$ is the depth of the potential well (energy scale)
- $\sigma$ is the distance where $U=0$ (length scale)
- $r$ is the inter-particle distance

The $r^{-12}$ term models Pauli repulsion at short range (steric exclusion), and
the $r^{-6}$ term models van der Waals attraction (induced dipole-dipole).

The force on particle $i$ due to particle $j$ is:

$$\mathbf{F}_{ij} = -\nabla U = \frac{24\epsilon}{r^2}
\left[2\left(\frac{\sigma}{r}\right)^{12} -
\left(\frac{\sigma}{r}\right)^{6}\right] \mathbf{r}_{ij}$$

---

### Reduced Units

All quantities are expressed in LJ reduced units:

| Quantity   | Symbol | Definition            | SI conversion                          |
|------------|--------|-----------------------|----------------------------------------|
| Length     | $r^*$  | $r/\sigma$            | $\sigma$ = 3.405 A (Ar)               |
| Energy     | $E^*$  | $E/\epsilon$          | $\epsilon/k_B$ = 119.8 K (Ar)         |
| Mass       | $m^*$  | $m/m_0$               | $m_0$ = 39.948 amu (Ar)              |
| Time       | $t^*$  | $t\sqrt{\epsilon/(m\sigma^2)}$ | $\tau\approx 2.16$ ps (Ar) |
| Temperature| $T^*$  | $k_B T/\epsilon$      |                                         |
| Pressure   | $P^*$  | $P\sigma^3/\epsilon$  |                                         |
| Density    | $\rho^*$| $\rho\sigma^3$        |                                         |

The force equation in reduced units (dropping asterisks):

$$f_{ij} = \frac{24}{r^2}\left[2r^{-12} - r^{-6}\right] r_{ij}$$

---

### Truncation And Shift

The LJ potential is truncated (cut off) at $r = r_c$ to reduce computational
cost with long-range interactions. Two common schemes:

**Truncated (no shift):**
$U(r) = U_{LJ}(r)$ for $r < r_c$, $U(r) = 0$ for $r \ge r_c$.

**Truncated + Shifted (our default):**
$U(r) = U_{LJ}(r) - U_{LJ}(r_c)$ for $r < r_c$, $U(r) = 0$ for $r \ge r_c$.

Shifting ensures $U(r_c) = 0$, removing the discontinuity in energy at the
cutoff. This package defaults to $r_c = 2.5$ with shifting.

---

### Long-Range Tail Corrections

Truncation removes the tail of the LJ potential for $r > r_c$. For an isotropic
fluid with radial distribution $g(r) \to 1$ beyond $r_c$, analytic corrections
are applied:

**Energy tail correction:**

$$U_{tail} = \frac{8\pi N \rho \epsilon}{3}
\left[\frac{1}{3}\left(\frac{\sigma}{r_c}\right)^9 -
\left(\frac{\sigma}{r_c}\right)^3\right]$$

**Pressure tail correction:**

$$P_{tail} = \frac{16\pi \rho^2 \epsilon}{3}
\left[\frac{2}{3}\left(\frac{\sigma}{r_c}\right)^9 -
\left(\frac{\sigma}{r_c}\right)^3\right]$$

These corrections are essential for obtaining accurate pressure and energy
values. In this package they are controlled by the
`use_tail_corrections` parameter in `SimConfig`.

---

## Integration: Velocity Verlet Algorithm

Newton's second law for a system of $N$ particles:

$$m_i \frac{d^2\mathbf{r}_i}{dt^2} = \mathbf{F}_i$$

The Velocity Verlet algorithm integrates these equations in discrete timesteps
$\Delta t$:

**Step 1 - Update positions:**

$$\mathbf{r}_i(t+\Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i(t)\Delta t +
\frac{1}{2m_i}\mathbf{F}_i(t)\Delta t^2$$

**Step 2 - Compute forces $\mathbf{F}_i(t+\Delta t)$ from new positions.**

**Step 3 - Update velocities:**

$$\mathbf{v}_i(t+\Delta t) = \mathbf{v}_i(t) +
\frac{1}{2m_i}\left[\mathbf{F}_i(t) + \mathbf{F}_i(t+\Delta t)\right]\Delta t$$

**Properties:**
- **Symplectic** -- conserves phase-space volume (important for long-time stability)
- **Time-reversible**
- **Global error** $O(\Delta t^2)$ in positions and velocities
- **Simplectic** nature means energy fluctuates but does not systematically drift in NVE

The implementation in this package stores the "half-step kick" pattern:

```
positions += velocities * dt + 0.5 * forces * dt^2  / mass
compute_forces()
velocities += 0.5 * forces * dt / mass
```

---

## Periodic Boundary Conditions And Minimum Image Convention

Periodic boundary conditions (PBC) replicate the simulation box in all
directions, eliminating surface effects. When a particle exits one face,
it re-enters from the opposite face.

The minimum image convention ensures each particle interacts with the nearest
image of every other particle:

$$r_{ij} = \text{box\_image}(r_j - r_i, L) = r_j - r_i - L \cdot
\text{nint}\left(\frac{r_j - r_i}{L}\right)$$

For the minimum image to be valid, the cutoff must satisfy $r_c \le L/2$.
If $r_c > L/2$, a particle could interact with both a neighbor and its
periodic image, violating the convention.

---

## Virial And Pressure

The **virial** $W$ is computed from pair forces:

$$W = -\frac{1}{3}\sum_{i<j} \mathbf{r}_{ij}\cdot\mathbf{F}_{ij}$$

For the LJ pair force $\mathbf{F}_{ij} = f_{ij}\mathbf{r}_{ij}$, this
simplifies to:

$$W = -\frac{1}{3}\sum_{i<j} f_{ij} \cdot r_{ij}^2$$

The **pressure** follows from the virial theorem:

$$P = \rho k_B T + \frac{W}{V} + P_{tail}$$

where $\rho = N/V$ is the number density.

In reduced units ($k_B = 1$):

$$P^* = \rho^* T^* + \frac{W}{V^*}$$

---

## Energy And Drift

**Kinetic energy:**

$$K = \sum_{i=1}^N \frac{1}{2} m_i v_i^2$$

**Potential energy:**
Sum of pair interactions (shifted + tail correction):

$$U = \sum_{i<j} U_{LJ}(r_{ij}) + U_{tail}$$

**Total energy:** $E = K + U$

**Energy drift** is defined as:

$$\text{drift} = \frac{|E(t) - E(0)|}{|E(0)|}$$

A drift $< 10^{-4}$ over $10^4$ steps indicates good integration stability.
Our package reports this automatically in the validation summary.

---

## Neighbor Lists

Direct computation of all $N(N-1)/2$ pair forces scales as $O(N^2)$ and is
prohibitively expensive for large systems. Neighbor lists reduce the cost by
only evaluating pairs within $r_{list} = r_c + r_{skin}$.

### Verlet Neighbor List (our default for $N<500$)

For each particle $i$, store a list of all particles $j > i$ with
$r_{ij} < r_{list}$. Forces are computed only from pairs in this list.

The **skin thickness** $r_{skin}$ ensures the list remains valid for multiple
steps. The list is rebuilt when accumulated particle displacement exceeds
$r_{skin}/2$.

Our implementation uses **half-list** storage (each pair stored once, $i < j$),
halving the force computations compared to a full list.

### Cell List / Linked Cells (our default for $N \ge 500$)

The simulation box is divided into a 3D grid of cells of size $\ge r_{list}$.
Each particle is assigned to its cell. Pairs are only checked between particles
in the same or adjacent ($3^3 - 1 = 26$) neighbor cells.

Scaling: $O(N)$ for the cell assignment, $O(N \cdot n_{cell})$ for pair search
(where $n_{cell}$ is the typical number of particles per cell).

Implementation uses:
$$n_{cells} = \max\left(\lfloor L / r_{list}\rfloor, 2\right)$$

The $n_{cells} \ge 2$ constraint prevents degenerate single-cell layouts
that would behave as $O(N^2)$.

---

## Force Computation With The Half-List Convention

When using the half Verlet list (each pair $(i,j)$ stored once with $i<j$),
Newton's third law is applied directly:

$$\mathbf{F}_i += f_{ij} \cdot \mathbf{r}_{ij}$$
$$\mathbf{F}_j -= f_{ij} \cdot \mathbf{r}_{ij}$$

The virial contribution is accumulated per pair:

$$W += -f_{ij} \cdot r_{ij}^2$$

where $f_{ij}$ is the scalar force factor computed by the pair potential.

---

## Radial Distribution Function (g(r))

The radial distribution function measures the probability of finding a particle
at distance $r$ from a reference particle, relative to an ideal gas:

$$g(r) = \frac{1}{4\pi r^2 \rho N}
\left\langle\sum_{i=1}^N\sum_{j\neq i} \delta(r - r_{ij})\right\rangle$$

In practice, a histogram is built by binning pair distances:

$$g(r_k) = \frac{n_k}{4\pi r_k^2 \Delta r \cdot N \cdot N_{frames}}$$

where $n_k$ is the number of pairs in bin $k$ at radius $r_k$ with width
$\Delta r$.

**Properties:**
- $g(r) \to 0$ as $r \to 0$ (steric exclusion)
- $g(r) \to 1$ as $r \to \infty$ (no long-range order)
- Peak at $r \approx 1.12\sigma$ for the LJ liquid (first coordination shell)
- Integral gives coordination number:
  $$N_{coord} = 4\pi\rho\int_0^{r_m} r^2 g(r) dr$$

---

## Mean Square Displacement And Diffusion

The mean squared displacement tracks particle mobility:

$$\text{MSD}(t) = \langle|\mathbf{r}_i(t) - \mathbf{r}_i(0)|^2\rangle$$

For a liquid in the diffusive regime:

$$\text{MSD}(t) = 6D_{\text{Einstein}} t + C$$

The **diffusion coefficient** is extracted from the linear slope:

$$D_{\text{Einstein}} = \lim_{t\to\infty} \frac{\text{MSD}(t)}{6t}$$

The **Green-Kubo relation** also gives $D$ from the velocity autocorrelation:

$$D_{\text{GK}} = \frac{1}{3}\int_0^\infty \langle\mathbf{v}(t)\cdot\mathbf{v}(0)\rangle dt$$

In practice, we compute:
1. Einstein $D$: linear fit to MSD vs $t$ over the diffusive window
2. Green-Kubo $D$: integrate VACF using the trapezoidal rule

The two should agree within statistical error for a well-equilibrated system.

---

## Velocity Autocorrelation Function (VACF)

$$C_{vv}(t) = \langle\mathbf{v}_i(t)\cdot\mathbf{v}_i(0)\rangle$$

Normalized:

$$C_{vv}^n(t) = \frac{\langle\mathbf{v}_i(t)\cdot\mathbf{v}_i(0)\rangle}
{\langle\mathbf{v}_i(0)\cdot\mathbf{v}_i(0)\rangle}$$

**Properties:**
- $C_{vv}(0) = 3k_BT/m$ (from equipartition)
- Decays to zero in the liquid (particles lose memory of initial velocity)
- May show a negative dip at short times (backscattering in dense fluids)

---

## Static Structure Factor (S(k))

The structure factor is the Fourier transform of $g(r)$:

$$S(k) = 1 + 4\pi\rho\int_0^\infty r^2 [g(r)-1] \frac{\sin(kr)}{kr} dr$$

Directly from particle positions:

$$S(\mathbf{k}) = \frac{1}{N}\left\langle
\left|\sum_{i=1}^N e^{-i\mathbf{k}\cdot\mathbf{r}_i}\right|^2\right\rangle$$

The first peak in $S(k)$ corresponds to the first peak in $g(r)$:
$k_{peak} \approx 2\pi/r_{peak}$.

For a liquid, $S(k \to 0)$ gives the isothermal compressibility:
$$S(0) = \rho k_B T \kappa_T$$

---

## Block Averaging (Flyvbjerg-Petersen)

Time-series data from MD simulations is serially correlated. The true
statistical error is larger than the naive $\sigma/\sqrt{N}$ estimate.

The Flyvbjerg-Petersen block averaging method:

1. Divide the $M$ data points into blocks of size $n_b$
2. Compute block averages
3. Compute standard deviation of block averages
4. Repeat with increasing block size
5. The plateau value of $\sigma(n_b)$ gives the true standard error

The optimal block size is where $\sigma / \sqrt{N_{blocks}}$ stabilizes.
This is used in the validation report to compute error bars on
$T^*$, $P^*$, and $E/N$.

---

## Integrated Autocorrelation Time

The integrated autocorrelation time quantifies how many steps are needed to
obtain independent samples:

$$\tau_{int} = \frac{1}{2} + \sum_{t=1}^{\infty} \rho(t)$$

where $\rho(t)$ is the normalized autocorrelation function of the observable.
The effective number of independent samples:

$$N_{eff} = \frac{N}{2\tau_{int}}$$

This affects the statistical error:

$$\sigma_{\langle A \rangle} = \sigma_A\sqrt{\frac{2\tau_{int}}{N}}$$

---

## References

1. Verlet, L. (1967). Computer "experiments" on classical fluids. I.
   Thermodynamical properties of Lennard-Jones molecules. *Phys. Rev.*,
   **159**(1), 98.

2. Frenkel, D. & Smit, B. (2002). *Understanding Molecular Simulation*,
   2nd ed. Academic Press.

3. Allen, M.P. & Tildesley, D.J. (2017). *Computer Simulation of Liquids*,
   2nd ed. Oxford University Press.

4. Plimpton, S. (1995). Fast parallel algorithms for short-range molecular
   dynamics. *J. Comp. Phys.*, **117**, 1-19.

5. Lustig, R. et al. (2021). LJTS equation of state. *Mol. Phys.*, (r_c=2.5).

6. Flyvbjerg, H. & Petersen, H.G. (1989). Error estimates on averages of
   correlated data. *J. Chem. Phys.*, **91**, 461.
