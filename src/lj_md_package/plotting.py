from __future__ import annotations
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("lj_md.plotting")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    plt = None


_COLORS = {
    "T":      "#E24A33",
    "P":      "#348ABD",
    "KE":     "#E24A33",
    "PE":     "#348ABD",
    "E":      "#188487",
    "eq_line": "#AAAAAA",
    "ref":    "#988ED5",
    "fit":    "#8EBA42",
    "hist":   "#467821",
}


def _configure_style(dpi=150, figsize=(8, 6)):
    rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.15",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.color": "0.75",
        "xtick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "legend.frameon": True,
        "legend.edgecolor": "0.5",
        "legend.fancybox": False,
        "legend.framealpha": 0.9,
    })


def _boundary_time(thermo, n_equil, dt):
    time = thermo["time"]
    step = thermo["step"]
    if np.any(step == n_equil):
        return float(time[step == n_equil][0])
    idx = np.searchsorted(step, n_equil)
    if idx < len(time):
        return float(time[idx])
    return float(n_equil * dt)


def _annotate_prod_avg(ax, x_data, y_data, prod_mask, precision=4):
    if not prod_mask.any():
        return
    prod_y = y_data[prod_mask]
    if len(prod_y) == 0:
        return
    avg = float(np.mean(prod_y))
    ax.axhline(avg, color=_COLORS["ref"], linestyle=":", linewidth=1.0, alpha=0.8)
    x_center = float(x_data[prod_mask].mean()) if np.any(prod_mask) else 0.0
    y_range = ax.get_ylim()
    y_pos = avg + 0.02 * (y_range[1] - y_range[0])
    ax.text(x_center, y_pos, f"avg = {avg:.{precision}f}",
            color=_COLORS["ref"], fontsize=9, ha="center", va="bottom",
            style="italic")


def plot_thermo_timeseries(thermo, n_equil, cfg, output_dir, dpi=150, figsize=(8, 10)):
    time = thermo["time"]
    step = thermo["step"]
    T = thermo["T"]
    P = thermo["P"]
    KE = thermo["KE"]
    PE = thermo["PE"]
    E = thermo["E_total"]
    E_per = thermo["E_per_atom"]

    N = cfg.N
    dt = cfg.dt
    t_boundary = _boundary_time(thermo, n_equil, dt)
    prod_mask = step > n_equil

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    ax = axes[0]
    ax.plot(time, T, color=_COLORS["T"], linewidth=0.7, rasterized=True)
    ax.axvline(t_boundary, color=_COLORS["eq_line"], linestyle="--", linewidth=1.0)
    _annotate_prod_avg(ax, time, T, prod_mask)
    ax.set_ylabel("$T^*$")
    ax.set_title("Thermodynamic Time Series", fontsize=14, fontweight="bold")

    ax = axes[1]
    ax.plot(time, P, color=_COLORS["P"], linewidth=0.7, rasterized=True)
    ax.axvline(t_boundary, color=_COLORS["eq_line"], linestyle="--", linewidth=1.0)
    _annotate_prod_avg(ax, time, P, prod_mask)
    ax.set_ylabel("$P^*$")

    ax = axes[2]
    ax.plot(time, KE / N, color=_COLORS["KE"], linewidth=0.6, alpha=0.65,
            label="$K / N$", rasterized=True)
    ax.plot(time, PE / N, color=_COLORS["PE"], linewidth=0.6, alpha=0.65,
            label="$U / N$", rasterized=True)
    ax.plot(time, E_per, color=_COLORS["E"], linewidth=0.9,
            label="$E_{\\mathrm{tot}} / N$", rasterized=True)
    ax.axvline(t_boundary, color=_COLORS["eq_line"], linestyle="--", linewidth=1.0)
    _annotate_prod_avg(ax, time, E_per, prod_mask)
    ax.set_xlabel("$t$ ($\\tau$)")
    ax.set_ylabel("$E / N$ ($\\epsilon$)")
    ax.legend(loc="lower right", ncol=3)

    for ax in axes:
        ax.set_xlim(time.min(), time.max())

    fig.savefig(Path(output_dir) / "thermo_timeseries.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved thermo_timeseries.png")


def plot_production_thermo(thermo, n_equil, cfg, output_dir, dpi=150, figsize=(10, 6)):
    step = thermo["step"]
    time = thermo["time"]
    T = thermo["T"]
    P = thermo["P"]
    E = thermo["E_total"]

    prod_mask = step > n_equil
    if not prod_mask.any():
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.35)

    t_prod = time[prod_mask]
    T_prod = T[prod_mask]
    P_prod = P[prod_mask]
    E_prod = E[prod_mask]

    ax1.plot(t_prod, T_prod, color=_COLORS["T"], linewidth=0.8, rasterized=True)
    T_avg, T_std = float(np.mean(T_prod)), float(np.std(T_prod))
    ax1.axhline(T_avg, color=_COLORS["ref"], linestyle=":", linewidth=1.0)
    ax1.fill_between(t_prod, T_avg - T_std, T_avg + T_std,
                     alpha=0.12, color=_COLORS["ref"])
    ax1.set_xlabel("$t$ ($\\tau$)")
    ax1.set_ylabel("$T^*$")
    ax1.set_title(f"$\\langle T^* \\rangle = {T_avg:.4f} \\pm {T_std:.4f}$",
                  fontsize=11)

    ax2.plot(t_prod, P_prod, color=_COLORS["P"], linewidth=0.8, rasterized=True)
    P_avg, P_std = float(np.mean(P_prod)), float(np.std(P_prod))
    ax2.axhline(P_avg, color=_COLORS["ref"], linestyle=":", linewidth=1.0)
    ax2.fill_between(t_prod, P_avg - P_std, P_avg + P_std,
                     alpha=0.12, color=_COLORS["ref"])
    ax2.set_xlabel("$t$ ($\\tau$)")
    ax2.set_ylabel("$P^*$")
    ax2.set_title(f"$\\langle P^* \\rangle = {P_avg:.4f} \\pm {P_std:.4f}$",
                  fontsize=11)

    ax3.plot(t_prod, E_prod / cfg.N, color=_COLORS["E"], linewidth=0.8, rasterized=True)
    E_avg, E_std = float(np.mean(E_prod)), float(np.std(E_prod))
    ax3.axhline(E_avg / cfg.N, color=_COLORS["ref"], linestyle=":", linewidth=1.0)
    ax3.fill_between(t_prod, (E_avg - E_std) / cfg.N, (E_avg + E_std) / cfg.N,
                     alpha=0.12, color=_COLORS["ref"])
    ax3.set_xlabel("$t$ ($\\tau$)")
    ax3.set_ylabel("$E_{\\mathrm{tot}} / N$ ($\\epsilon$)")
    ax3.set_title(f"$\\langle E/N \\rangle = {E_avg / cfg.N:.4f} \\pm {E_std / cfg.N:.4f}$",
                  fontsize=11)

    fig.suptitle("Production Phase (NVE)", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(Path(output_dir) / "thermo_production.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved thermo_production.png")


def plot_rdf(r, g, output_dir, dpi=150, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(r, g, color=_COLORS["PE"], linewidth=1.5, rasterized=True)
    ax.fill_between(r, g, alpha=0.08, color=_COLORS["PE"])

    peak_idx = int(np.argmax(g))
    peak_r, peak_g = float(r[peak_idx]), float(g[peak_idx])
    ax.annotate(
        f"Peak: $r={peak_r:.3f}\\sigma$, $g={peak_g:.3f}$",
        xy=(peak_r, peak_g),
        xytext=(peak_r + 0.25 * r.max(), peak_g * 0.65),
        arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                               edgecolor="0.5", alpha=0.8),
    )

    ax.set_xlabel("$r$ ($\\sigma$)")
    ax.set_ylabel("$g(r)$")
    ax.set_xlim(0, r.max())
    ax.set_ylim(0, g.max() * 1.15)
    ax.set_title("Radial Distribution Function", fontsize=14, fontweight="bold")

    fig.savefig(Path(output_dir) / "rdf.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved rdf.png")


def plot_structure_factor(k, sk, output_dir, dpi=150, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(k, sk, color=_COLORS["T"], linewidth=1.5, rasterized=True)

    valid = k > 0.1
    if valid.any():
        peak_idx = int(np.argmax(sk[valid])) + int(np.where(valid)[0][0])
        peak_k, peak_s = float(k[peak_idx]), float(sk[peak_idx])
        ax.annotate(
            f"$k={peak_k:.2f}\\sigma^{{-1}}$\n$S(k)={peak_s:.3f}$",
            xy=(peak_k, peak_s),
            xytext=(peak_k * 1.3, peak_s * 0.7),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                                   edgecolor="0.5", alpha=0.8),
        )

    ax.set_xlabel("$k$ ($\\sigma^{-1}$)")
    ax.set_ylabel("$S(k)$")
    ax.set_xlim(0, k.max() if len(k) > 0 else 20)
    ax.set_title("Static Structure Factor", fontsize=14, fontweight="bold")

    fig.savefig(Path(output_dir) / "sk.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved sk.png")


def plot_structure(r, g, k, sk, output_dir, dpi=150, figsize=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.30)

    ax1.plot(r, g, color=_COLORS["PE"], linewidth=1.5, rasterized=True)
    ax1.fill_between(r, g, alpha=0.08, color=_COLORS["PE"])
    peak_idx = int(np.argmax(g))
    peak_r, peak_g = float(r[peak_idx]), float(g[peak_idx])
    ax1.annotate(f"$r={peak_r:.3f}\\sigma$\n$g={peak_g:.3f}$",
                 xy=(peak_r, peak_g), xytext=(peak_r + 0.5, peak_g * 0.6),
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                                       edgecolor="0.5", alpha=0.8))
    ax1.set_xlabel("$r$ ($\\sigma$)")
    ax1.set_ylabel("$g(r)$")
    ax1.set_xlim(0, r.max())
    ax1.set_ylim(0, g.max() * 1.15)
    ax1.set_title("Radial Distribution Function", fontsize=13, fontweight="bold")

    ax2.plot(k, sk, color=_COLORS["T"], linewidth=1.5, rasterized=True)
    valid = k > 0.1
    if valid.any():
        peak_idx2 = int(np.argmax(sk[valid])) + int(np.where(valid)[0][0])
        peak_k, peak_s = float(k[peak_idx2]), float(sk[peak_idx2])
        ax2.annotate(f"$k={peak_k:.2f}\\sigma^{{-1}}$\n$S(k)={peak_s:.3f}$",
                     xy=(peak_k, peak_s), xytext=(peak_k * 1.4, peak_s * 0.6),
                     arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                                           edgecolor="0.5", alpha=0.8))
    ax2.set_xlabel("$k$ ($\\sigma^{-1}$)")
    ax2.set_ylabel("$S(k)$")
    ax2.set_xlim(0, k.max() if len(k) > 0 else 20)
    ax2.set_title("Static Structure Factor", fontsize=13, fontweight="bold")

    fig.savefig(Path(output_dir) / "structure.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved structure.png")


def plot_rdf_phi_polar(sim, output_dir, dpi=150, figsize=(8, 7)):
    L = sim.box.L
    N = sim.atoms.N
    pos = sim.atoms.positions
    dr_vec = pos[:, None, :] - pos[None, :, :]
    dr_vec -= L * np.round(dr_vec / L)
    r_xy = np.sqrt(dr_vec[:, :, 0] ** 2 + dr_vec[:, :, 1] ** 2)
    phi = np.degrees(np.arctan2(dr_vec[:, :, 1], dr_vec[:, :, 0]))

    r_max = sim.cfg.r_cut
    n_r = 60
    n_phi = 72
    r_edges = np.linspace(0.0, r_max, n_r + 1)
    phi_edges = np.linspace(-180.0, 180.0, n_phi + 1)
    dr = r_edges[1] - r_edges[0]
    dphi = np.radians(phi_edges[1] - phi_edges[0])

    mask = ~np.eye(N, dtype=bool)
    r_flat = r_xy[mask]
    phi_flat = phi[mask]
    keep = (r_flat > 0) & (r_flat <= r_max)
    hist, _, _ = np.histogram2d(r_flat[keep], phi_flat[keep],
                                bins=[r_edges, phi_edges])

    A = L ** 2
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    g = np.zeros_like(hist)
    mask_r = r_centers > 1e-10
    g[mask_r] = (2.0 * A * hist[mask_r]) / (N ** 2 * r_centers[mask_r, None] * dr * dphi)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    theta_rad_edges = np.radians(phi_edges)
    R_grid, THETA_grid = np.meshgrid(r_edges, theta_rad_edges, indexing="ij")
    vmax = np.percentile(g, 99.5)
    pcm = ax.pcolormesh(THETA_grid, R_grid, g, cmap="inferno",
                        shading="auto", vmin=0, vmax=vmax)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-180)
    ax.set_thetamax(180)
    ax.set_title("$g(r_{xy}, \\phi)$ — In-Plane Structure", pad=20,
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.12, shrink=0.82)
    cbar.set_label("$g(r, \\phi)$", fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "rdf_phi_polar.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved rdf_phi_polar.png")


def plot_rdf_theta(sim, output_dir, dpi=150, figsize=(10, 8)):
    from .computes.rdf_theta import ComputeRDFTheta

    r_centers = None
    cos_centers = None
    g_2d = None
    for c in sim.computes:
        if isinstance(c, ComputeRDFTheta):
            r_centers, cos_centers, g_iso, g_2d = c.finalize()
            break

    if g_2d is None or r_centers is None:
        logger.warning("No ComputeRDFTheta data available for g(r,theta) plot.")
        return

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[1, 0.7],
                          hspace=0.3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    extent = [r_centers[0] - (r_centers[1] - r_centers[0]) / 2,
              r_centers[-1] + (r_centers[1] - r_centers[0]) / 2,
              cos_centers[0] - (cos_centers[1] - cos_centers[0]) / 2,
              cos_centers[-1] + (cos_centers[1] - cos_centers[0]) / 2]
    vmax = np.percentile(g_2d, 99.5)
    im = ax1.imshow(g_2d.T, origin="lower", aspect="auto", extent=extent,
                    cmap="viridis", vmin=0, vmax=vmax, interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax1, shrink=0.9, pad=0.02)
    cbar.set_label("$g(r, \\cos\\theta)$", fontsize=11)
    ax1.set_xlabel("$r$ ($\\sigma$)")
    ax1.set_ylabel("$\\cos\\theta$")
    ax1.set_title("$g(r, \\cos\\theta)$", fontsize=13, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    r_full = r_centers
    ax2.plot(r_full, g_iso, color=_COLORS["PE"], linewidth=1.5)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("$r$ ($\\sigma$)")
    ax2.set_ylabel("$g(r)$ (angle-averaged)")
    ax2.set_title("Isotropic $g(r)$ Check", fontsize=13, fontweight="bold")

    ax3 = fig.add_subplot(gs[1, :])
    first_shell = (1.0, 1.6)
    g_theta = g_2d[(r_centers >= first_shell[0]) & (r_centers <= first_shell[1]), :]
    if g_theta.shape[0] > 0:
        g_theta_mean = g_theta.mean(axis=0)
        ax3.plot(cos_centers, g_theta_mean, color=_COLORS["T"], linewidth=1.5)
        ax3.fill_between(cos_centers, g_theta_mean, alpha=0.2, color=_COLORS["T"])
        ax3.axhline(g_theta_mean.mean(), color="gray", ls="--", lw=0.8,
                    label=f"mean = {g_theta_mean.mean():.3f}")
    ax3.set_xlabel("$\\cos\\theta$")
    ax3.set_ylabel("$g(\\cos\\theta)$")
    ax3.set_title(f"Angular Profile ($1.0 \\leq r \\leq 1.6\\,\\sigma$)",
                  fontsize=13, fontweight="bold")
    ax3.legend()

    fig.suptitle("Angular-Resolved Pair Distribution $g(r, \\cos\\theta)$",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(Path(output_dir) / "rdf_theta.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved rdf_theta.png")


def plot_transport(t_msd, msd, t_vacf, vacf, D_einstein, D_gk,
                   output_dir, dpi=150, figsize=(12, 5), D_ref=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.30)

    ax1.plot(t_msd, msd, color=_COLORS["PE"], linewidth=1.8, rasterized=True)
    if t_msd.size > 4:
        start = max(1, int(0.3 * t_msd.size))
        coeffs = np.polyfit(t_msd[start:], msd[start:], 1)
        fit_line = np.polyval(coeffs, t_msd)
        label = f"Fit: $D^*_E = {D_einstein:.4f}$"
        if D_ref is not None:
            label += f"\n(ref: {D_ref:.4f})"
        ax1.plot(t_msd, fit_line, "--", color=_COLORS["fit"], linewidth=1.2,
                 label=label)
    if D_ref is not None and t_msd.size > 4:
        ref_line = 6.0 * D_ref * t_msd
        ax1.plot(t_msd, ref_line, ":", color=_COLORS["ref"], linewidth=1.0,
                 alpha=0.7, label=f"$D^*_R = {D_ref}$")
    ax1.set_xlabel("$t$ ($\\tau$)")
    ax1.set_ylabel("$\\langle \\Delta r(t)^2 \\rangle$ ($\\sigma^2$)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Mean-Squared Displacement", fontsize=13, fontweight="bold")

    ax2.plot(t_vacf, vacf, color=_COLORS["T"], linewidth=1.8, rasterized=True)
    ax2.fill_between(t_vacf, vacf, 0, alpha=0.08, color=_COLORS["T"])
    ax2.axhline(0, color="0.4", linewidth=0.5, linestyle="--")
    text = f"$D_{{\\mathrm{{GK}}}} = {D_gk:.4f}$"
    if D_ref is not None:
        text += f"\nref: {D_ref:.4f}"
    if not np.isnan(D_gk):
        ax2.text(0.95, 0.95, text,
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                          edgecolor="0.5", alpha=0.8))
    ax2.set_xlabel("$t$ ($\\tau$)")
    ax2.set_ylabel("$C_{vv}(t) / C_{vv}(0)$")
    ax2.set_title("Velocity Autocorrelation Function", fontsize=13, fontweight="bold")

    fig.savefig(Path(output_dir) / "transport.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved transport.png")


def plot_energy_drift(thermo, n_equil, cfg, output_dir, dpi=150, figsize=(8, 4)):
    step = thermo["step"]
    time = thermo["time"]
    E = thermo["E_total"]

    prod_mask = step > n_equil
    if not prod_mask.any():
        return

    t_prod = time[prod_mask]
    E_prod = E[prod_mask]
    E0 = E_prod[0]
    if abs(E0) < 1e-12:
        return
    drift = np.abs(E_prod - E0) / abs(E0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_prod, drift, color=_COLORS["E"], linewidth=0.8, rasterized=True)
    ax.set_xlabel("$t$ ($\\tau$)")
    ax.set_ylabel("$|E(t) - E_0| / |E_0|$")
    ax.set_yscale("log")
    ax.set_title("Energy Drift (Production Phase)", fontsize=14, fontweight="bold")

    final_drift = drift[-1]
    ax.text(0.95, 0.95, f"Final drift: {final_drift:.3e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                     edgecolor="0.5", alpha=0.8))

    fig.savefig(Path(output_dir) / "energy_drift.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved energy_drift.png")


def plot_temperature_histogram(thermo, n_equil, cfg, output_dir, dpi=150, figsize=(7, 5)):
    step = thermo["step"]
    T = thermo["T"]
    prod_mask = step > n_equil
    if not prod_mask.any():
        return
    T_prod = T[prod_mask]
    T_avg = float(np.mean(T_prod))
    T_std = float(np.std(T_prod))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(T_prod, bins=30, density=True, color=_COLORS["T"], alpha=0.7,
            edgecolor="white", linewidth=0.5, label="Production T*")
    x_range = np.linspace(T_prod.min(), T_prod.max(), 200)
    gauss = np.exp(-0.5 * ((x_range - T_avg) / T_std) ** 2)
    gauss /= gauss.sum() * (x_range[1] - x_range[0])
    ax.plot(x_range, gauss, "--", color=_COLORS["ref"], linewidth=1.5,
            label="Gaussian fit")
    ax.axvline(T_avg, color=_COLORS["T"], linestyle=":", linewidth=1.2)
    ax.set_xlabel("$T^*$")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Temperature Distribution\n$\\langle T^* \\rangle = {T_avg:.4f} \\pm {T_std:.4f}$",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    fig.savefig(Path(output_dir) / "temperature_histogram.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved temperature_histogram.png")


def plot_msd_individual(t_msd, msd, D_einstein, output_dir, dpi=150,
                        figsize=(7, 5), D_ref=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_msd, msd, color=_COLORS["PE"], linewidth=1.8, rasterized=True)
    if t_msd.size > 4:
        start = max(1, int(0.3 * t_msd.size))
        coeffs = np.polyfit(t_msd[start:], msd[start:], 1)
        fit_line = np.polyval(coeffs, t_msd)
        label = f"$D^*_{{\\mathrm{{Einstein}}}} = {D_einstein:.4f}$"
        if D_ref is not None:
            label += f"\n(ref: {D_ref:.4f})"
        ax.plot(t_msd, fit_line, "--", color=_COLORS["fit"], linewidth=1.2,
                label=label)
    if D_ref is not None and t_msd.size > 4:
        ref_line = 6.0 * D_ref * t_msd
        ax.plot(t_msd, ref_line, ":", color=_COLORS["ref"], linewidth=1.0,
                alpha=0.7, label=f"$D^*_{{\\mathrm{{ref}}}} = {D_ref}$")
    ax.set_xlabel("$t$ ($\\tau$)")
    ax.set_ylabel("$\\langle \\Delta r(t)^2 \\rangle$ ($\\sigma^2$)")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Mean-Squared Displacement", fontsize=14, fontweight="bold")

    fig.savefig(Path(output_dir) / "msd.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved msd.png")


def plot_vacf_individual(t_vacf, vacf, D_gk, output_dir, dpi=150,
                         figsize=(7, 5), D_ref=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_vacf, vacf, color=_COLORS["T"], linewidth=1.8, rasterized=True)
    ax.fill_between(t_vacf, vacf, 0, alpha=0.08, color=_COLORS["T"])
    ax.axhline(0, color="0.4", linewidth=0.5, linestyle="--")
    text = f"$D_{{\\mathrm{{GK}}}} = {D_gk:.4f}$"
    if D_ref is not None:
        text += f"\nref: {D_ref:.4f}"
    if not np.isnan(D_gk):
        ax.text(0.95, 0.95, text,
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                         edgecolor="0.5", alpha=0.8))
    ax.set_xlabel("$t$ ($\\tau$)")
    ax.set_ylabel("$C_{vv}(t) / C_{vv}(0)$")
    ax.set_title("Velocity Autocorrelation Function", fontsize=14, fontweight="bold")

    fig.savefig(Path(output_dir) / "vacf.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved vacf.png")


def generate_all_plots(sim, output_dir=None):
    if not HAVE_MPL:
        logger.warning(
            "matplotlib not available. Install with: pip install -e .[plot]"
        )
        return

    cfg = sim.cfg
    _configure_style(dpi=cfg.plot_dpi, figsize=cfg.figure_size)
    output_dir = output_dir or cfg.output_dir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    D_ref = None
    try:
        from .units.reference import LITERATURE_LJ
        D_ref = float(LITERATURE_LJ["D_star"][0])
    except Exception:
        pass

    if sim._thermo is None:
        logger.warning("No thermo data available for plotting.")
        return

    thermo = sim._thermo.as_arrays()
    n_equil = cfg.n_equil
    dpi = cfg.plot_dpi

    try:
        plot_thermo_timeseries(thermo, n_equil, cfg, output_dir, dpi)
    except Exception as e:
        logger.warning("Failed to plot thermo timeseries: %s", e)

    try:
        plot_production_thermo(thermo, n_equil, cfg, output_dir, dpi)
    except Exception as e:
        logger.warning("Failed to plot production thermo: %s", e)

    try:
        plot_energy_drift(thermo, n_equil, cfg, output_dir, dpi)
    except Exception as e:
        logger.warning("Failed to plot energy drift: %s", e)

    try:
        plot_temperature_histogram(thermo, n_equil, cfg, output_dir, dpi)
    except Exception as e:
        logger.warning("Failed to plot temperature histogram: %s", e)

    rdf_data = None
    sk_data = None
    msd_data = None
    vacf_data = None
    D_einstein = float("nan")
    D_gk = float("nan")

    from .computes.rdf import ComputeRDF
    from .computes.structure_factor import ComputeStructureFactor
    from .computes.msd import ComputeMSD
    from .computes.vacf import ComputeVACF

    for c in sim.computes:
        if isinstance(c, ComputeRDF):
            try:
                r, g = c.finalize()
                rdf_data = (r, g)
            except Exception as e:
                logger.warning("Failed to get RDF data: %s", e)
        elif isinstance(c, ComputeStructureFactor):
            try:
                k, sk = c.finalize()
                sk_data = (k, sk)
            except Exception as e:
                logger.warning("Failed to get S(k) data: %s", e)
        elif isinstance(c, ComputeMSD):
            try:
                t, msd = c.get_msd()
                D_einstein = c.diffusion_coefficient()
                msd_data = (t, msd)
            except Exception as e:
                logger.warning("Failed to get MSD data: %s", e)
        elif isinstance(c, ComputeVACF):
            try:
                t, vacf = c.get_vacf()
                D_gk = c.green_kubo_diffusion()
                vacf_data = (t, vacf)
            except Exception as e:
                logger.warning("Failed to get VACF data: %s", e)

    if rdf_data is not None:
        r, g = rdf_data
        try:
            plot_rdf(r, g, output_dir, dpi)
        except Exception as e:
            logger.warning("Failed to plot RDF: %s", e)

    if sk_data is not None:
        k, sk = sk_data
        try:
            plot_structure_factor(k, sk, output_dir, dpi)
        except Exception as e:
            logger.warning("Failed to plot S(k): %s", e)

    if rdf_data is not None and sk_data is not None:
        r, g = rdf_data
        k, sk = sk_data
        try:
            plot_structure(r, g, k, sk, output_dir, dpi)
        except Exception as e:
            logger.warning("Failed to plot combined structure: %s", e)

    if msd_data is not None:
        t, msd = msd_data
        try:
            plot_msd_individual(t, msd, D_einstein, output_dir, dpi, D_ref=D_ref)
        except Exception as e:
            logger.warning("Failed to plot MSD: %s", e)

    if vacf_data is not None:
        t, vacf = vacf_data
        try:
            plot_vacf_individual(t, vacf, D_gk, output_dir, dpi, D_ref=D_ref)
        except Exception as e:
            logger.warning("Failed to plot VACF: %s", e)

    if msd_data is not None and vacf_data is not None:
        t_msd, msd = msd_data
        t_vacf, vacf = vacf_data
        try:
            plot_transport(t_msd, msd, t_vacf, vacf,
                           D_einstein, D_gk, output_dir, dpi, D_ref=D_ref)
        except Exception as e:
            logger.warning("Failed to plot combined transport: %s", e)

    try:
        plot_rdf_theta(sim, output_dir, dpi)
    except Exception as e:
        logger.warning("Failed to plot g(r,theta): %s", e)

    try:
        plot_rdf_phi_polar(sim, output_dir, dpi)
    except Exception as e:
        logger.warning("Failed to plot polar g(r,phi): %s", e)

    logger.info("All plots generated in %s/", output_dir)
