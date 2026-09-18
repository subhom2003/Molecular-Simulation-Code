"""
=============================================================================
Configuration dataclass for LJ-MD simulations
=============================================================================

All parameters in Lennard-Jones reduced units unless otherwise noted.
Reduced units convention:

    Quantity      Symbol    Reduced form       Argon equivalent
    ────────────  ────────  ─────────────────  ────────────────
    Length        sigma*    r* = r/sigma       sigma = 3.405 Angstrom
    Energy        eps*      U* = U/eps         eps/k_B = 119.8 K
    Mass          m*        m* = m/m           m = 39.948 u
    Time          tau*      t* = t/tau         tau = sigma*sqrt(m/eps) ~ 2.156 ps
    Temperature   T*        T* = k_B T/eps     1 T* ~ 119.8 K
    Pressure      P*        P* = P sigma^3/eps 1 P* ~ 41.9 MPa
    Density       rho*      rho* = N sigma^3/V 1 rho* ~ 1.6 g/cm^3

The classical Verlet (1967) state point is ρ*=0.8442, T*=0.722.
References: Verlet 1967; Allen & Tildesley 2017; Frenkel & Smit 2002;
Johnson, Zollweg & Gubbins 1993 (EOS).
=============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger("lj_md.config")


# ---------------------------------------------------------------------------
# Allowed values for enum-like fields
# ---------------------------------------------------------------------------
THERMOSTATS = {"none", "rescale", "berendsen", "nose_hoover", "langevin"}
LATTICES = {"fcc", "sc", "random"}


@dataclass
class SimConfig:
    """
    Master configuration dataclass.

    Every field is documented inline with its physical meaning,
    units (in reduced LJ units), recommended range, and references.
    Derived quantities (L, V, r_list, U_rc, u_tail, P_tail, dof) are
    computed automatically in :meth:`__post_init__`.

    Example
    -------
    >>> cfg = SimConfig(N=108, rho_star=0.8442, T_star=0.722)
    >>> print(cfg.summary())
    """

    # --- System -----------------------------------------------------------
    #: Number of particles. 108 = 4 x 3^3 (smallest cubic FCC supercell).
    N: int = 108
    #: Reduced number density rho* = N sigma^3 / V (default: Verlet 1967).
    rho_star: float = 0.8442
    #: Reduced temperature T* = k_B T / eps (default: Verlet 1967).
    T_star: float = 0.722

    # --- Force field ------------------------------------------------------
    #: Lennard-Jones cutoff radius r_c in sigma units.
    r_cut: float = 2.5
    #: Truncate-and-shift potential so U(r_c) = 0.
    shift_potential: bool = True
    #: Apply analytic long-range tail corrections to U and P (g(r)=1 for r > r_c).
    use_tail_corrections: bool = True

    # --- Neighbor list ----------------------------------------------------
    #: Verlet skin distance in sigma units (rebuild buffer).
    r_skin: float = 0.3
    #: Neighbor-list style: "verlet" (O(N^2) memory, all pairs) or "cell" (linked-cell).
    neighbor_style: str = "verlet"

    # --- Time integration -------------------------------------------------
    #: Time step Delta_t in reduced time units tau.
    dt: float = 0.005
    #: Total number of integration steps.
    n_steps: int = 20_000
    #: Equilibration steps (thermostat active, no production data collected).
    n_equil: int = 5_000
    #: Steps before the end of equilibration during which the thermostat is
    #: smoothly switched off ("coasting") so the system settles into NVE
    #: equilibrium and the production temperature is unbiased. Default 0 means
    #: hold the thermostat all the way through; the legacy benchmark used 200.
    n_thermo_off: int = 0
    #: Integrator style: "velocity_verlet" (only one currently supported).
    integrator_style: str = "velocity_verlet"

    # --- Thermostats ------------------------------------------------------
    #: Thermostat during equilibration: none/rescale/berendsen/nose_hoover/langevin.
    thermostat_type: str = "rescale"
    #: Steps between velocity rescaling events (rescale thermostat only).
    rescale_interval: int = 100
    #: Berendsen thermostat time constant tau_T (in tau).
    tau_T: float = 0.5
    #: Nosé-Hoover thermostat mass Q (fictitious heat-bath variable).
    nose_hoover_Q: float = 2.0
    #: Langevin friction gamma (in 1/tau).
    langevin_gamma: float = 1.0

    # --- Sampling ---------------------------------------------------------
    #: Steps between thermodynamic data collection.
    sample_interval: int = 10
    #: Step at which RDF accumulation begins.
    rdf_start: int = 5_000
    #: Steps between RDF histogram accumulations.
    rdf_interval: int = 50
    #: Number of radial bins for g(r).
    rdf_n_bins: int = 200
    #: Step at which MSD origin sampling begins.
    msd_start: int = 5_000
    #: Steps between MSD time-origin selections.
    msd_interval: int = 100
    #: Number of steps tracked per MSD origin.
    msd_length: int = 2_000
    #: Step at which VACF origin sampling begins.
    vacf_start: int = 5_000
    #: Steps between VACF time-origin selections.
    vacf_interval: int = 50
    #: Number of steps tracked per VACF origin.
    vacf_length: int = 500

    # --- Dumps ------------------------------------------------------------
    #: Steps between XYZ trajectory frames.
    traj_interval: int = 200
    #: Steps between console/log progress messages.
    output_interval: int = 500

    # --- Initialization ---------------------------------------------------
    #: Lattice type: fcc, sc, random.
    lattice_type: str = "fcc"
    #: RNG seed for reproducibility.
    random_seed: int = 42

    # --- Output -----------------------------------------------------------
    #: Directory for all output files. Created if it does not exist.
    output_dir: str = "output"
    #: Generate publication-quality plots after simulation finishes.
    generate_plots: bool = True
    #: DPI for saved figures (use 300 for publication).
    plot_dpi: int = 150
    #: Default figure size in inches (width, height).
    figure_size: tuple = field(default_factory=lambda: (8, 6))

    # --- Numerical tolerances / validation --------------------------------
    #: Maximum acceptable fractional drift |DE/E0|.
    energy_drift_tol: float = 0.01
    #: Minimum allowed initial pair separation (sigma).
    min_pair_distance: float = 0.5
    #: Maximum allowed initial force magnitude (eps/sigma).
    max_initial_force: float = 1000.0
    #: If True, validation checks raise instead of warning.
    strict: bool = False

    # --- Derived (computed in __post_init__) ------------------------------
    #: Box length L = (N/rho)^{1/3}.
    L: float = field(init=False, repr=False)
    #: Box volume V = N / rho.
    V: float = field(init=False, repr=False)
    #: Neighbor-list cutoff r_list = r_cut + r_skin.
    r_list: float = field(init=False, repr=False)
    #: LJ potential U(r_cut) (used for shift).
    U_rc: float = field(init=False, repr=False)
    #: Per-particle long-range tail correction to potential energy.
    u_tail: float = field(init=False, repr=False)
    #: Long-range tail correction to pressure.
    P_tail: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compute all derived simulation parameters."""
        self.N = int(self.N)
        self.V = float(self.N) / float(self.rho_star)
        self.L = self.V ** (1.0 / 3.0)

        if self.r_cut >= self.L / 2.0:
            raise ValueError(
                f"r_cut={self.r_cut} must be strictly < L/2={self.L/2:.4f}. "
                f"Reduce r_cut or increase N."
            )

        self.r_list = self.r_cut + self.r_skin

        inv_rc6 = self.r_cut ** (-6)
        inv_rc12 = self.r_cut ** (-12)
        self.U_rc = 4.0 * (inv_rc12 - inv_rc6)

        rc3, rc9 = self.r_cut ** 3, self.r_cut ** 9
        if self.use_tail_corrections:
            self.u_tail = (8.0 * np.pi / 3.0) * self.rho_star * (
                1.0 / (3.0 * rc9) - 1.0 / rc3
            )
            self.P_tail = (16.0 * np.pi / 3.0) * self.rho_star ** 2 * (
                2.0 / (3.0 * rc9) - 1.0 / rc3
            )
        else:
            self.u_tail = 0.0
            self.P_tail = 0.0

        logger.debug(
            "SimConfig initialised: N=%d, rho*=%.4f, T*=%.4f, L*=%.4f",
            self.N, self.rho_star, self.T_star, self.L,
        )

    # ------------------------------------------------------------------ dof
    @property
    def dof(self) -> int:
        """Degrees of freedom for temperature arithmetic. 3N - 3 after COM removal."""
        return 3 * self.N - 3

    # -------------------------------------------------------------- validate
    def validate(self) -> None:
        """Validate configuration; raise ``ValueError`` on bad inputs."""
        if self.N < 4:
            raise ValueError(f"N must be at least 4 (got {self.N}).")
        if self.rho_star <= 0:
            raise ValueError(f"rho_star must be positive (got {self.rho_star}).")
        if self.T_star <= 0:
            raise ValueError(f"T_star must be positive (got {self.T_star}).")
        if not (0.0 < self.dt <= 0.02):
            logger.warning(
                "dt=%s is outside the recommended 0.001-0.010 range for LJ; "
                "energy conservation may degrade.", self.dt
            )
        if self.n_equil >= self.n_steps:
            raise ValueError(
                f"n_equil={self.n_equil} must be < n_steps={self.n_steps}."
            )
        if self.rdf_start < 0 or self.rdf_start > self.n_steps:
            raise ValueError(
                f"rdf_start={self.rdf_start} must be in [0, n_steps={self.n_steps}]."
            )
        if self.r_skin <= 0:
            raise ValueError(f"r_skin must be positive (got {self.r_skin}).")
        if self.thermostat_type not in THERMOSTATS:
            raise ValueError(
                f"thermostat_type={self.thermostat_type!r} not in {sorted(THERMOSTATS)}."
            )
        if self.lattice_type not in LATTICES:
            raise ValueError(
                f"lattice_type={self.lattice_type!r} not in {sorted(LATTICES)}."
            )
        if self.neighbor_style not in ("verlet", "cell"):
            raise ValueError(
                f"neighbor_style={self.neighbor_style!r} must be 'verlet' or 'cell'."
            )
        if self.thermostat_type == "nose_hoover" and self.nose_hoover_Q <= 0:
            raise ValueError(f"nose_hoover_Q must be positive.")
        if self.thermostat_type == "berendsen" and self.tau_T <= 0:
            raise ValueError(f"tau_T must be positive for Berendsen.")
        if self.n_thermo_off < 0 or self.n_thermo_off > self.n_equil:
            raise ValueError(
                f"n_thermo_off={self.n_thermo_off} must be in [0, n_equil]."
            )
        logger.debug("Configuration validated successfully.")

    # -------------------------------------------------------------- summary
    def summary(self) -> str:
        """Return a human-readable summary of all configuration parameters."""
        sep = "=" * 70
        return "\n".join([
            sep,
            " LENNARD-JONES MD SIMULATION -- CONFIGURATION SUMMARY",
            sep,
            f"  System          N={self.N:4d}  rho*={self.rho_star:.4f}  T*={self.T_star:.4f}",
            f"  Box             L*={self.L:.4f} sigma  V*={self.V:.4f} sigma^3",
            f"  Force field     r_c*={self.r_cut} sigma  shift={self.shift_potential}",
            f"  Tail corr.      u_tail/N={self.u_tail:+.4f} eps  P_tail={self.P_tail:+.4f} eps/sigma^3",
            f"  Neighbor list   style={self.neighbor_style}  r_skin={self.r_skin} sigma  r_list={self.r_list:.4f}",
            f"  Integration     dt*={self.dt}  n_steps={self.n_steps:,}",
            f"  Equilibration   n_equil={self.n_equil:,}  thermostat='{self.thermostat_type}'",
            f"  Production      NVE (no thermostat), last {self.n_thermo_off} steps of equil switch off",
            f"  Sampling        every {self.sample_interval} steps",
            f"  RDF             {self.rdf_n_bins} bins, start at step {self.rdf_start}",
            f"  MSD/VACF        starts at step {self.msd_start}",
            f"  Output          dir='{self.output_dir}'  dpi={self.plot_dpi}",
            f"  RNG seed        {self.random_seed}",
            f"  Strict mode     {self.strict}",
            sep,
        ])


# ---------------------------------------------------------------------------
# Default config factory -- avoids import-time side effects of constructing
# a SimConfig at module load. Callers should use get_default_config().
# ---------------------------------------------------------------------------
def get_default_config() -> SimConfig:
    """Return a fresh copy of the default Verlet (1967) benchmark config."""
    return SimConfig()


# Backward-compat alias for older scripts (lazy: a property would be cleaner,
# but many users do `from lj_md_package.constants import DEFAULT_CONFIG`).
# We expose it as a module-level None sentinel that the users would not be
# expected to use directly; the new API is get_default_config().
DEFAULT_CONFIG: SimConfig | None = None
