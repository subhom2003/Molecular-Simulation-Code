"""
=============================================================================
Validation report builder (``summary.txt``)
=============================================================================

Compares block-averaged production averages of T*, P*, E/N, D*, |DE/E0|
against literature values from :data:`units.reference.LITERATURE_LJ`.
Returns a printable ASCII report (also written to ``summary.txt``).
=============================================================================
"""

from __future__ import annotations
import numpy as np

from .block_average import block_average
from .autocorrelation import integrated_autocorrelation_time


def _fmt(val: float, err: float | None = None, fmt: str = "%.4f",
         ref: float | None = None, ref_name: str = "") -> str:
    s = fmt % val
    if err is not None and not np.isnan(err):
        s += f" +- {fmt % err}"
    if ref is not None and not np.isnan(ref) and val == val:
        err_pct = abs(val - ref) / abs(ref) * 100 if abs(ref) > 0 else float("nan")
        s += f"  |  literature {ref:.4f} ({ref_name}) | error {err_pct:.1f}%"
    return s


def build_validation_report(sim, wall_time_s: float,
                             literature: dict | None = None) -> str:
    """Build the ASCII validation report for a finished simulation."""
    cfg = sim.cfg
    out: list[str] = []
    sep = "=" * 76
    out.append(sep)
    out.append(" LJ-MD -- SIMULATION VALIDATION REPORT")
    out.append(sep)
    out.append(cfg.summary())

    if literature is None:
        # pick the model the simulation actually represents:
        # tail corrections ON  -> full-LJ reference
        # tail corrections OFF -> LJTS (rc=2.5) reference
        from ..units.reference import reference_for
        literature = reference_for(getattr(cfg, "use_tail_corrections", False))
        out.append(f"  Reference model: "
                   f"{'full LJ (with tails)' if getattr(cfg, 'use_tail_corrections', False) else 'LJTS (no tails)'}")

    thermo = sim._thermo.as_arrays() if sim._thermo else {}
    prod_mask = thermo["step"] > cfg.n_equil if "step" in thermo else np.array([])
    if prod_mask.size == 0 or not np.any(prod_mask):
        out.append("\nWARNING: no production data available "
                   "(all recorded steps are <= n_equil? runs too short?).")
        return "\n".join(out + [sep])

    from ..units.reference import near_solid_liquid_coexistence
    if near_solid_liquid_coexistence(cfg.rho_star, cfg.T_star) and cfg.N < 500:
        out.append("")
        out.append(" NOTE: rho*/T* are near the solid-liquid coexistence band and")
        out.append(" N is small: expect coexistence-like averages (intermediate P,")
        out.append(" depressed g(r) peaks) unless the melt-anneal-NVE protocol from")
        out.append(" README.txt section 7 was followed and stayed liquid.")
        out.append("")

    out.append("\n")
    out.append(" Production Averages (block-averaged with Flyvbjerg-Petersen)")
    out.append("-" * 76)

    def report_key(key: str, label: str, fmt: str = "%.4f", ref_key: str | None = None):
        if key not in thermo:
            return
        series = thermo[key][prod_mask]
        if series.size == 0:
            return
        mean, err, _ = block_average(series)
        ref_val, ref_name = literature.get(ref_key or key, (None, ""))
        out.append(
            f"  {label:<15s} {_fmt(mean, err, fmt, ref_val, ref_name)}"
        )

    report_key("T",       "T*",  ref_key="T_star")
    report_key("P",       "P*",  ref_key="P_star")
    report_key("E_per_atom", "E/N",     ref_key="E_per_N")

    from ..computes.msd import ComputeMSD
    from ..computes.vacf import ComputeVACF
    from ..computes.rdf import ComputeRDF

    D_lit_val, D_lit_ref = literature.get("D_star", (None, ""))
    for c in sim.computes:
        if isinstance(c, ComputeMSD):
            D = c.diffusion_coefficient()
            out.append(f"  {'D* (Einstein)':<15s} {_fmt(D, fmt='%.4f', ref=D_lit_val, ref_name=D_lit_ref)}")
        elif isinstance(c, ComputeVACF):
            Dgk = c.green_kubo_diffusion()
            out.append(f"  {'D* (Green-Kubo)':<15s} {_fmt(Dgk, fmt='%.4f', ref=D_lit_val, ref_name=D_lit_ref)}")
        elif isinstance(c, ComputeRDF):
            try:
                r, g = c.finalize()
                peak_r = float(r[g.argmax()])
                peak_g = float(g.max())
                out.append(f"  {'g(r) peak r*':<15s} {peak_r:.4f}  | g(r) peak height = {peak_g:.3f}")
            except Exception:
                pass

    # equilibration-quality indicator: production mean T vs thermostat target
    if "T" in thermo:
        T_prod = float(np.mean(thermo["T"][prod_mask]))
        T_tgt = cfg.T_star
        if T_tgt > 0 and abs(T_prod - T_tgt) / T_tgt > 0.05:
            out.append("")
            out.append(f"  !! Production T*={T_prod:.4f} deviates {(T_prod-T_tgt)/T_tgt*100:+.1f}% "
                       f"from target {T_tgt:.4f} - equilibration likely incomplete")
            out.append("     (structure still relaxing during production; lengthen the"
                       " anneal stage or check the thermostat).")

    out.append("\n")
    out.append(" Energy Conservation")
    out.append("-" * 76)
    drift = sim._thermo.energy_drift()
    out.append(f"  |dE/E0|  {drift:.3e}  "
               f"{'- OK' if drift < cfg.energy_drift_tol else '!' * 2 + ' LARGE DRIFT'}")

    out.append("\n")
    out.append(" Autocorrelation Times")
    out.append("-" * 76)
    for key, label in [("T", "T*"), ("P", "P*"), ("E_total", "E_total")]:
        if key not in thermo:
            continue
        iat = integrated_autocorrelation_time(thermo[key][prod_mask])
        out.append(f"  tau({label:>9s}) = {iat:7.2f} steps (~{iat * cfg.dt:6.3f} tau)")

    out.append("\n")
    out.append(" Run Stats")
    out.append("-" * 76)
    out.append(f"  Wall time          {wall_time_s:.2f} s")
    if sim.neighbor_list is not None:
        out.append(f"  Neighbor list builds: {sim.neighbor_list.n_builds}")
    out.append(sep)
    return "\n".join(out)