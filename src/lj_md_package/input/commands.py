"""
=============================================================================
LAMMPS-style commands registered with the script registry
=============================================================================

Each handler has the signature::

    def cmd_<name>(script: Script, args: list[str]) -> None

The handlers mutate the ``script.sim`` instance (configuring it) and, in the
case of ``run`` / ``unfix`` / ``dump`` etc., execute live actions.

Supported commands:
    units, atom_style, boundary,
    lattice, region, create_box, create_atoms, read_data,
    pair_style, pair_coeff, pair_modify,
    neighbor, neigh_modify,
    timestep, velocity, fix, unfix, compute, uncompute,
    thermo, thermo_style, dump, dump_modify, undump,
    restart, run, minimize, print, variable, set_output_dir, seed.
=============================================================================
"""

from __future__ import annotations
import logging
import numpy as np

from .registry import registry
from ..core.box import Box
from ..core.atoms import Atoms
from ..core.config import SimConfig
from ..core.simulation import Simulation
from ..initialize import (create_fcc_lattice, create_sc_lattice,
                          create_random_lattice, remove_com_velocity,
                          scale_velocities_to_temperature,
                          maxwell_boltzmann_velocities)
from ..pairs.lj_cut import LJCut
from ..pairs.soft import SoftPair
from ..neighbors.verlet import VerletNL
from ..neighbors.cell_list import CellList
from ..fixes.nve import FixNVE
from ..fixes.nvt_nose_hoover import FixNVT
from ..fixes.temp_rescale import FixTempRescale
from ..fixes.berendsen import FixBerendsen
from ..fixes.momentum import FixMomentum
from ..fixes.langevin import FixLangevin
from ..integrators.velocity_verlet import VelocityVerlet
from ..integrators.minimize import MinimizeSD
from ..computes.rdf import ComputeRDF
from ..computes.rdf_theta import ComputeRDFTheta
from ..computes.msd import ComputeMSD
from ..computes.vacf import ComputeVACF
from ..computes.structure_factor import ComputeStructureFactor
from ..computes.conserved_energy import ComputeConservedEnergy
from ..computes.temp import ComputeTemp
from ..computes.pressure import ComputePressure
from ..computes.thermo_style import ThermoStyle, _THERMO_KEYS
from ..dumps.xyz import DumpXYZ
from ..dumps.thermo import DumpThermo
from ..dumps.energy import DumpEnergy
from ..dumps.lammps_data import read_lammps_data
from ..units.abc import UnitSystem
from ..units.lj import LJUnits
from ..units.real import RealUnits
from ..units.metal import MetalUnits
from ..units.si import SIUnits

logger = logging.getLogger("lj_md.script")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def _to_floats(args: list[str]) -> list[float]:
    return [float(a) for a in args]


def _to_ints(args: list[str]) -> list[int]:
    return [int(float(a)) for a in args]


def _config(script: Script) -> SimConfig:
    return script.cfg


def _sim(script: Script) -> Simulation:
    return script.sim


def _ensure_atoms_set(script: Script) -> None:
    if not script._atoms_set:
        raise RuntimeError("_atoms not initialised: use 'lattice'/'create_atoms'/'read_data' before 'run'.")


def _ensure_pair_set(script: Script) -> None:
    if not script._pair_set:
        raise RuntimeError("pair_style not specified.")


def _ensure_neighbor_set(script: Script) -> None:
    if not script._neighbor_set:
        _build_default_neighbor_list(script)


def _build_default_neighbor_list(script: Script) -> None:
    cfg = _config(script)
    # For small systems, VerletNL (O(N²) build, O(N) use) is faster and more
    # reliable than CellList. CellList becomes preferable for N > ~500.
    use_cell = (cfg.neighbor_style == "cell" and cfg.N >= 500)
    if use_cell:
        nl = CellList(cfg.r_cut, cfg.r_skin, cfg.L)
    else:
        nl = VerletNL(cfg.r_cut, cfg.r_skin, cfg.L)
    sim = _sim(script)
    sim.set_neighbor_list(nl)
    nl.build(sim.atoms.positions)
    script._neighbor_set = True


# =============================================================================
# commands
# =============================================================================

# --------------------------------------------------------------- units
@registry.register("units")
def cmd_units(script: Script, args: list[str]) -> None:
    """``units lj|real|metal|si`` -- sets the unit convention. Currently only
    LJ-reduced is implemented internally; the other names accept their unit
    labels (and so that conversion to/from happens at the I/O level)."""
    cfg = _config(script)
    name = args[0].lower()
    if name == "lj":
        cfg._unit_system = LJUnits()
    elif name == "real":
        cfg._unit_system = RealUnits()
    elif name == "metal":
        cfg._unit_system = MetalUnits()
    elif name == "si":
        cfg._unit_system = SIUnits()
    else:
        raise ValueError(f"units {name!r} not supported yet (use 'lj').")
    script._configured_units = True


# ----------------------------------------------------------- atom_style
@registry.register("atom_style")
def cmd_atom_style(script, args):
    name = args[0].lower()
    if name != "atomic":
        raise NotImplementedError(f"atom_style {name!r} not supported (only 'atomic').")
    # otherwise no-op for now


# ----------------------------------------------------------- boundary
@registry.register("boundary")
def cmd_boundary(script, args):
    """``boundary p p p`` -- set periodic boundary flags. Only 'p p p' supported."""
    iflen = len(args)
    if iflen != 3:
        raise ValueError(f"boundary: expected 3 args, got {args}")
    if not all(b.lower() in ("p", "f", "s", "m") for b in args):
        raise ValueError(f"boundary: only periodic/fixed/shrink-wrap supported, got {args}")
    if not all(b.lower() == "p" for b in args):
        raise NotImplementedError("boundary: only 'p p p' (fully periodic) supported yet.")


# ------------------------------------------------------------ lattice
@registry.register("lattice")
def cmd_lattice(script, args):
    """``lattice (fcc|sc) rho*`` -- define the lattice and density."""
    if len(args) < 2:
        raise ValueError("lattice type rho*  ... expected")
    style = args[0].lower()
    rho = float(args[1])
    cfg = _config(script)
    cfg.lattice_type = "sc" if style == "sc" else "fcc"
    cfg.rho_star = rho
    # recompute derived fields (L, V, r_list, U_rc, u_tail, P_tail)
    cfg.__post_init__()
    script._lattice_set = True


# --------------------------------------------------------------- region
@registry.register("region")
def cmd_region(script, args):
    """``region ID block xlo xhi ylo yhi zlo zhi`` -- define a region.
    We only support a single box region; the ID 'box' is the canonical name."""
    rid = args[0]
    kind = args[1].lower()
    if kind != "block":
        raise NotImplementedError(f"region {kind!r} not supported; only 'block'.")
    cfg = _config(script)
    # args: xlo xhi ylo yhi zlo zhi  (after 'block')
    v = [float(a) for a in args[2:8]]
    cfg._region = v
    script._box_set = True


# --------------------------------------------------------- create_box
@registry.register("create_box")
def cmd_create_box(script, args):
    """``create_box Ntypes region-ID`` -- creates the simulation box from a region.
    In the legacy repo this only synthesizes atoms at lattice points; here we
    actually instantiate the box in our config and store the n_types.
    """
    n_types = int(args[0])
    region_id = args[1] if len(args) > 1 else "box"
    cfg = _config(script)
    if hasattr(cfg, "_region") and cfg._region is not None:
        v = cfg._region
        # In LAMMPS, "0 N" creates automatically. Accept the box region values.
        # If specified lattice density would conflict we reconcile here.
        Lx = v[1] - v[0]
        Ly = v[3] - v[2]
        Lz = v[5] - v[4]
        if Ly == Lz == Lx:
            cfg.rho_star = cfg.N / (Lx ** 3)
            cfg.__post_init__()
        else:
            raise NotImplementedError("Non-cubic region box not supported yet.")
    else:
        # if no region, data was likely already prepopulated by create_atoms using rho.
        pass
    cfg._n_atom_types = n_types
    sim = _sim(script)
    sim.set_box(Box.cubic(cfg.L))


# ---------------------------------------------------------- create_atoms
@registry.register("create_atoms")
def cmd_create_atoms(script, args):
    """``create_atoms TYPE box`` -- fill the box with atoms using the pre-set lattice.
    Accept only the simplest form (one box-region specification)."""
    n_type = int(args[0])
    region_spec = args[1:]  # e.g. ["box"] or ["region", "ID"]
    if region_spec and region_spec[0] not in ("box", "region"):
        raise NotImplementedError(f"create_atoms {region_spec[0]!r} not supported; use 'box'.")
    cfg = _config(script)
    # Build lattice based on lattice_type and N=best-fit FCC (4n^3) for the current density
    # If user has not specified N yet, default to 108 if FCC (4*3^3=108)
    if not hasattr(cfg, "_N_set"):
        # Pick a default N based on density that fits the box if region given; else 108
        cfg.N = 108
    sim = _sim(script)
    # determine lattice
    if cfg.lattice_type == "fcc":
        positions = create_fcc_lattice(cfg)
    elif cfg.lattice_type == "sc":
        positions = create_sc_lattice(cfg)
    elif cfg.lattice_type == "random":
        rng = np.random.default_rng(cfg.random_seed)
        positions = create_random_lattice(cfg, rng)
    else:
        raise ValueError(f"Unknown lattice_type: {cfg.lattice_type}")
    atoms = Atoms(positions=positions, velocities=np.zeros_like(positions),
                  types=np.full(positions.shape[0], n_type, dtype=np.int32))
    atoms._box_ref = sim.box
    sim.set_atoms(atoms)
    script._atoms_set = True
    cfg._n_atom_types = n_type
    # Recreate the box just in case
    sim.set_box(Box.cubic(cfg.L))


# ----------------------------------------------------------- read_data
@registry.register("read_data")
def cmd_read_data(script, args):
    """``read_data FILE`` -- load a LAMMPS data file directly."""
    cfg = _config(script)
    sim = _sim(script)
    filename = args[0]
    atoms, box, header = read_lammps_data(filename)
    sim.set_box(box)
    sim.set_atoms(atoms)
    cfg.N = atoms.N
    cfg.rho_star = atoms.N / box.volume()
    cfg.rho_star = float(cfg.rho_star)
    logger.info("read_data: loaded N=%d from %s (header=%s)", atoms.N, filename, header)
    script._atoms_set = True


# --------------------------------------------------------- pair_style
@registry.register("pair_style")
def cmd_pair_style(script, args):
    name = args[0]
    rest = args[1:]
    if name == "lj/cut":
        r_cut = float(rest[0])
        pair = LJCut(r_cut=r_cut, shift=True, tail=True)
    elif name == "soft":
        r_cut = float(rest[0]) if rest else 1.0
        A = float(rest[1]) if len(rest) > 1 else 1.0
        pair = SoftPair(r_cut=r_cut, A=A)
    else:
        raise NotImplementedError(f"pair_style {name!r} not implemented.")
    sim = _sim(script)
    sim.set_pair_style(pair)
    # Sync to config (sample etc. might be picked up later)
    cfg = _config(script)
    cfg.r_cut = pair.cutoff
    script._pair_set = True


# --------------------------------------------------------- pair_coeff
@registry.register("pair_coeff")
def cmd_pair_coeff(script, args):
    """``pair_coeff TYPE1 TYPE2 EPS SIG`` -- just a placeholder for now since
    we only support single-type LJ eps=1, sigma=1."""
    if len(args) < 4:
        raise ValueError("pair_coeff TYPE1 TYPE2 eps sigma")
    i, j = int(args[0]), int(args[1])
    eps, sigma = float(args[2]), float(args[3])
    sim = _sim(script)
    if sim.pair_style is None:
        raise RuntimeError("pair_coeff before pair_style")
    sim.pair_style.coeffs[(i, j)] = (eps, sigma)
    sim.pair_style.coeffs[(j, i)] = (eps, sigma)


# --------------------------------------------------------- pair_modify
@registry.register("pair_modify")
def cmd_pair_modify(script, args):
    """``pair_modify (shift yes/no) (tail yes/no)`` -- set shift/tail flags."""
    sim = _sim(script)
    pair = sim.pair_style
    if pair is None:
        raise RuntimeError("pair_modify before pair_style")
    rest = args
    for i, token in enumerate(rest):
        if token == "shift":
            pair.shift = (rest[i + 1].lower() == "yes")
        elif token == "tail":
            pair.tail = (rest[i + 1].lower() == "yes")
    cfg = _config(script)
    cfg.shift_potential = bool(pair.shift)
    cfg.use_tail_corrections = bool(pair.tail)


# ------------------------------------------------------------ neighbor
@registry.register("neighbor")
def cmd_neighbor(script, args):
    """``neighbor SKIN (bin|nsq)`` -- set skin and (optionally) neighbor-list style."""
    cfg = _config(script)
    skin = float(args[0])
    cfg.r_skin = skin
    if len(args) > 1:
        style = args[1]
        if style in ("bin", "cell"):
            cfg.neighbor_style = "cell"
        elif style == "nsq":
            cfg.neighbor_style = "verlet"
        else:
            raise ValueError(f"neighbor style {style!r} not supported.")
    script._neighbor_set = False  # force rebuild in case atoms already set


# ------------------------------------------------------------ neigh_modify
@registry.register("neigh_modify")
def cmd_neigh_modify(script, args):
    # Accept and ignore most sub-options; we always rebuild on movement.
    logger.debug("neigh_modify %s -- mostly ignored", args)


# --------------------------------------------------------------- timestep
@registry.register("timestep")
def cmd_timestep(script, args):
    cfg = _config(script)
    cfg.dt = float(args[0])
    sim = _sim(script)
    sim.state.dt = cfg.dt
    # Mutate the live integrator so fixes that hold a reference to it
    # continue to use the new timestep.
    if sim.integrator is not None:
        sim.integrator.dt = cfg.dt
    else:
        sim.set_integrator(VelocityVerlet(cfg.dt, sim.box))


# --------------------------------------------------------------- velocity
@registry.register("velocity")
def cmd_velocity(script, args):
    """``velocity GROUP-ID create T_seed dist gaussian`` or
    ``velocity GROUP-ID scale T_target`` or
    ``velocity GROUP-ID zero linear``."""
    group_id = args[0]
    if group_id != "all":
        raise NotImplementedError(f"velocity {group_id!r} not supported -- only 'all'.")
    action = args[1]
    cfg = _config(script)
    atoms = _sim(script).atoms
    if action == "create":
        T = float(args[2])
        seed = int(float(args[3]))
        cfg.T_star = T
        rng = np.random.default_rng(seed)
        v = maxwell_boltzmann_velocities(cfg, rng)
        v = remove_com_velocity(v)
        v = _RescaleToT(v, T)
        atoms.velocities = v
    elif action == "scale":
        T = float(args[2])
        atoms.scale_to_temperature(T)
    elif action == "zero":
        # only support linear
        atoms.remove_com_motion()
    else:
        raise NotImplementedError(f"velocity {action!r} not supported.")


def _RescaleToT(v: np.ndarray, T_target: float) -> np.ndarray:
    dof = 3 * v.shape[0] - 3
    ke = 0.5 * float(np.einsum("ij,ij->", v, v))
    T_cur = 2.0 * ke / dof
    if T_cur <= 0:
        raise RuntimeError("Cannot scale zero-T velocities.")
    return v * np.sqrt(T_target / T_cur)


# --------------------------------------------------------------- fix
def _drop_integrating_fixes(sim, keep_id: str | None = None) -> None:
    """Remove existing integrating fixes (nve, langevin) other than keep_id."""
    from ..fixes.nve import FixNVE as _FNVE
    from ..fixes.langevin import FixLangevin as _FLang
    for f in list(sim.fixes):
        if isinstance(f, (_FNVE, _FLang)) and f.fix_id != keep_id:
            logger.warning("fix: removing integrating fix %r (replaced).", f.fix_id)
            sim.remove_fix(f.fix_id)


@registry.register("fix")
def cmd_fix(script, args):
    fix_id = args[0]
    group_id = args[1]
    fix_name = args[2]
    fix_args = args[3:]
    cfg = _config(script)
    sim = _sim(script)
    if fix_name == "nve":
        # Build the integrator only if not already created from `timestep` before
        if sim.integrator is None:
            sim.set_integrator(VelocityVerlet(cfg.dt, sim.box))
        _drop_integrating_fixes(sim, keep_id=None)
        target = FixNVE(fix_id=fix_id, group=group_id, integrator=sim.integrator)
        sim.add_fix(target)
    elif fix_name in ("nvt", "nose_hoover"):
        # fix ID all nvt T_TARGET [Q]
        T_target = float(fix_args[0])
        Q = float(fix_args[1]) if len(fix_args) > 1 else cfg.nose_hoover_Q
        sim.add_fix(FixNVT(fix_id=fix_id, group=group_id, T_target=T_target, Q=Q))
    elif fix_name == "temp/rescale":
        # LAMMPS argument order: N Tstart Tstop window fraction
        N = int(fix_args[0])
        Tstart = float(fix_args[1])
        Tstop = float(fix_args[2])
        window = float(fix_args[3]) if len(fix_args) > 3 else 0.0
        fraction = float(fix_args[4]) if len(fix_args) > 4 else 1.0
        sim.add_fix(FixTempRescale(fix_id=fix_id, group=group_id, N=N,
                                   Tstart=Tstart, Tstop=Tstop,
                                   fraction=fraction, window=window))
    elif fix_name == "berendsen":
        N = int(fix_args[0])
        Tstart = float(fix_args[1])
        Tstop = float(fix_args[2])
        tau_T = float(fix_args[3])
        sim.add_fix(FixBerendsen(fix_id=fix_id, group=group_id, N=N,
                                  Tstart=Tstart, Tstop=Tstop, tau_T=tau_T))
    elif fix_name in ("press/berendsen", "berendsen/barostat"):
        # fix ID all press/berendsen P_target tau_P [beta]
        from ..fixes.berendsen_barostat import FixBerendsenBarostat
        P_target = float(fix_args[0])
        tau_P = float(fix_args[1])
        beta = float(fix_args[2]) if len(fix_args) > 2 else 0.1
        baro = FixBerendsenBarostat(fix_id=fix_id, group=group_id,
                                    P_target=P_target, tau_P=tau_P, beta=beta)
        baro.attach(sim)
        sim.add_fix(baro)
        logger.warning("press/berendsen is intended for equilibration; "
                       "computes that cached the box length (rdf, sk) are "
                       "stale under a changing box.")
    elif fix_name == "langevin":
        # fix ID all langevin T_TARGET gamma [seed]
        T_target = float(fix_args[0])
        gamma = float(fix_args[1])
        seed = int(fix_args[2]) if len(fix_args) > 2 else cfg.random_seed
        # Langevin integrates itself (BAOAB): remove any existing nve fix
        _drop_integrating_fixes(sim, keep_id=None)
        if sim.integrator is None:
            sim.set_integrator(VelocityVerlet(cfg.dt, sim.box))
        lang = FixLangevin(fix_id=fix_id, group=group_id,
                           T_target=T_target, gamma=gamma, seed=seed)
        lang.set_integrator(sim.integrator)
        sim.add_fix(lang)
    elif fix_name == "momentum":
        N = int(fix_args[0])
        sim.add_fix(FixMomentum(fix_id=fix_id, group=group_id, N=N))
    else:
        raise NotImplementedError(f"fix {fix_name!r} not implemented.")


# -------------------------------------------------------------- unfix
@registry.register("unfix")
def cmd_unfix(script, args):
    fix_id = args[0]
    found = _sim(script).remove_fix(fix_id)
    if not found:
        raise RuntimeError(f"unfix: no fix with id={fix_id!r}.")


# ---------------------------------------------------------- compute
@registry.register("compute")
def cmd_compute(script, args):
    compute_id = args[0]
    group_id = args[1]
    compute_name = args[2]
    rest = args[3:]
    cfg = _config(script)
    sim = _sim(script)
    if compute_name == "temp":
        sim.add_compute(ComputeTemp(compute_id=compute_id, group=group_id))
    elif compute_name == "pressure":
        sim.add_compute(ComputePressure(compute_id=compute_id, group=group_id,
                                         rho_star=cfg.rho_star, P_tail=cfg.P_tail))
    elif compute_name == "rdf":
        n_bins = int(rest[0]) if rest else cfg.rdf_n_bins
        sim.add_compute(ComputeRDF(compute_id=compute_id, group=group_id,
                                   every=cfg.rdf_interval, n_bins=n_bins,
                                   L=cfg.L, N=cfg.N))
    elif compute_name == "msd":
        sim.add_compute(ComputeMSD(compute_id=compute_id, group=group_id,
                                   every=cfg.msd_interval, max_length=cfg.msd_length,
                                   dt=cfg.dt))
    elif compute_name == "vacf":
        sim.add_compute(ComputeVACF(compute_id=compute_id, group=group_id,
                                    every=cfg.vacf_interval, max_length=cfg.vacf_length,
                                    dt=cfg.dt))
    elif compute_name == "structure_factor":
        sim.add_compute(ComputeStructureFactor(compute_id=compute_id,
                                                group=group_id, L=cfg.L, N=cfg.N))
    elif compute_name in ("conserved/energy", "conserved_energy"):
        sim.add_compute(ComputeConservedEnergy(compute_id=compute_id, group=group_id))
    else:
        raise NotImplementedError(f"compute {compute_name!r} not implemented.")


# ----------------------------------------------------------- uncompute
@registry.register("uncompute")
def cmd_uncompute(script, args):
    found = _sim(script).remove_compute(args[0])
    if not found:
        raise RuntimeError(f"uncompute: no compute with id={args[0]!r}.")


# ------------------------------------------------------------ thermo
@registry.register("thermo")
def cmd_thermo(script, args):
    """``thermo N`` -- set thermo output interval."""
    cfg = _config(script)
    if args and args[0].isdigit():
        cfg.output_interval = int(args[0])
        cfg.sample_interval = int(args[0])
    sim = _sim(script)
    # Set up a ThermoStyle compute if not present
    if sim._thermo is None:
        sim._thermo = ThermoStyle(rho_star=cfg.rho_star, P_tail=cfg.P_tail,
                                  u_tail=cfg.u_tail)
        sim._thermo.set_atom_count(cfg.N)
        sim.add_compute(sim._thermo)


# --------------------------------------------------------- thermo_style
@registry.register("thermo_style")
def cmd_thermo_style(script, args):
    """``thermo_style (custom|default) KEY1 KEY2 ...``"""
    cfg = _config(script)
    sim = _sim(script)
    if args[0] == "default":
        keys = _THERMO_KEYS
        custom = _THERMO_KEYS
    elif args[0] == "custom":
        custom = tuple(args[1:])
        keys = tuple(a for a in custom if a in _THERMO_KEYS)
    else:
        raise NotImplementedError(f"thermo_style {args[0]!r} not supported; use 'custom'.")
    sim._thermo = ThermoStyle(rho_star=cfg.rho_star, P_tail=cfg.P_tail,
                              u_tail=cfg.u_tail, custom_keys=keys)
    sim._thermo.set_atom_count(cfg.N)
    # Replace existing thermo-style compute if present.
    for c in sim.computes:
        if isinstance(c, ThermoStyle):
            sim.computes.remove(c)
            break
    sim.add_compute(sim._thermo)


# ----------------------------------------------------------------- dump
@registry.register("dump")
def cmd_dump(script, args):
    """``dump ID GROUP (xyz) N FILE`` (only xyz/thermo/energy supported)."""
    dump_id = args[0]
    group_id = args[1]
    style = args[2]
    every = int(args[3])
    filename = args[4]
    sim = _sim(script)
    cfg = _config(script)
    if style == "xyz":
        d = DumpXYZ(dump_id=dump_id, group=group_id, every=every, filename=filename)
    elif style == "thermo":
        d = DumpThermo(dump_id=dump_id, every=every, filename=filename,
                       rho_star=cfg.rho_star, P_tail=cfg.P_tail, u_tail=cfg.u_tail)
    elif style == "energy":
        d = DumpEnergy(dump_id=dump_id, every=every, filename=filename,
                       N=cfg.N, u_tail=cfg.u_tail)
    else:
        raise NotImplementedError(f"dump {style!r} not supported; only xyz/thermo/energy.")
    sim.add_dump(d)


# ----------------------------------------------------------- dump_modify
@registry.register("dump_modify")
def cmd_dump_modify(script, args):
    # Accept and ignore most sub-options; only element is silently supported.
    logger.debug("dump_modify %s -- mostly ignored", args)


# ------------------------------------------------------------- undump
@registry.register("undump")
def cmd_undump(script, args):
    dump_id = args[0]
    if not _sim(script).remove_dump(dump_id):
        raise RuntimeError(f"undump: no dump with id={dump_id!r}.")


# ------------------------------------------------------------- run
@registry.register("run")
def cmd_run(script, args):
    _ensure_atoms_set(script)
    _ensure_pair_set(script)
    _ensure_neighbor_set(script)
    cfg = _config(script)
    sim = _sim(script)
    # cross-config of equilibration length is implicit: the user just calls
    # run() sequentially. The first run() call is treated as "equilibration".
    n = int(args[0])
    if sim._equilibrated is False:
        # first run() is equilibration
        cfg.n_equil = n
    # Analysis must only see production frames: gate compute starts at
    # n_equil (i.e. the end of the first run block).
    prod_start = cfg.n_equil

    # Add default analysis computes if none exist.
    has_analysis = any(isinstance(c, (ComputeRDF, ComputeRDFTheta, ComputeMSD,
                                       ComputeVACF, ComputeStructureFactor))
                       for c in sim.computes)
    if not has_analysis:
        if not any(isinstance(c, ComputeRDF) for c in sim.computes):
            sim.add_compute(ComputeRDF(every=cfg.rdf_interval, n_bins=cfg.rdf_n_bins,
                                       L=cfg.L, N=cfg.N, start=prod_start))
        if not any(isinstance(c, ComputeRDFTheta) for c in sim.computes):
            sim.add_compute(ComputeRDFTheta(every=cfg.rdf_interval,
                                            n_r_bins=cfg.rdf_n_bins // 2,
                                            n_cos_bins=50,
                                            L=cfg.L, N=cfg.N, start=prod_start))
        if not any(isinstance(c, ComputeMSD) for c in sim.computes):
            sim.add_compute(ComputeMSD(every=cfg.msd_interval,
                                       max_length=cfg.msd_length, dt=cfg.dt,
                                       start=prod_start))
        if not any(isinstance(c, ComputeVACF) for c in sim.computes):
            sim.add_compute(ComputeVACF(every=cfg.vacf_interval,
                                        max_length=cfg.vacf_length, dt=cfg.dt,
                                        start=prod_start))
        if not any(isinstance(c, ComputeStructureFactor) for c in sim.computes):
            sim.add_compute(ComputeStructureFactor(every=max(200, cfg.sample_interval * 4),
                                                   L=cfg.L, N=cfg.N,
                                                   start=prod_start))
        logger.info("Added default analysis computes (start=%d) for input-script run.",
                    prod_start)

    # initial force evaluation if not yet computed
    if sim.state.force_result is None:
        fr = sim.pair_style.compute(sim.atoms, sim.neighbor_list, sim.box)
        sim.state.force_result = fr
        sim.state.kinetic_energy = sim.atoms.kinetic_energy()
        sim.state.temperature = 2.0 * sim.state.kinetic_energy / sim.atoms.dof

    # default integrating fix if the user hasn't added one
    if not sim.fixes:
        logger.warning("run: no fix defined; adding 'fix integrator all nve'.")
        if sim.integrator is None:
            sim.set_integrator(VelocityVerlet(cfg.dt, sim.box))
        sim.add_fix(FixNVE(fix_id="integrator", group="all",
                           integrator=sim.integrator))

    # always keep a ThermoStyle recorder (the summary relies on it)
    if sim._thermo is None:
        sim._thermo = ThermoStyle(rho_star=cfg.rho_star, P_tail=cfg.P_tail,
                                  u_tail=cfg.u_tail, n_equil=cfg.n_equil)
        sim._thermo.set_atom_count(cfg.N)
        sim.add_compute(sim._thermo)
    sim._thermo.n_equil = cfg.n_equil

    sim.run(n)


# ------------------------------------------------------------- minimize
@registry.register("minimize")
def cmd_minimize(script, args):
    """``minimize energy_tol force_tol max_iter max_steps``"""
    e_tol = float(args[0])
    f_tol = float(args[1])
    max_iter = int(args[2])
    max_steps = int(args[3])
    cfg = _config(script)
    sim = _sim(script)
    _ensure_atoms_set(script)
    _ensure_pair_set(script)
    _ensure_neighbor_set(script)
    # fire the initial force evaluation if needed (steady descent needs forces)
    if sim.state.force_result is None:
        fr = sim.pair_style.compute(sim.atoms, sim.neighbor_list, sim.box)
        sim.state.force_result = fr
    # remove any other integrating fix: minimization owns the update
    _drop_integrating_fixes(sim, keep_id=None)
    sim.set_integrator(MinimizeSD(dt=cfg.dt, box=sim.box,
                                  energy_tolerance=e_tol, force_tolerance=f_tol,
                                  max_steps=max_steps))
    target = FixNVE(fix_id="min", group="all", integrator=sim.integrator)
    sim.add_fix(target)
    sim.run(max_iter)


# ------------------------------------------------------------- print
@registry.register("print")
def cmd_print(script, args):
    # Print only -- just log
    logger.info("print: %s", " ".join(args))


# --------------------------------------------------------- variable
@registry.register("variable")
def cmd_variable(script, args):
    """``variable NAME value`` -- store a named string for ${...} substitution."""
    name = args[0]
    kind = args[1]
    if kind == "equal":
        # only simple numeric values supported
        try:
            value = float(args[2])
            script.set_variable(name, repr(value))
        except (ValueError, IndexError):
            raise NotImplementedError("variable equal non-numeric -- complex expressions not supported.")
    else:
        # store everything after kind as the variable value
        script.set_variable(name, " ".join(args[2:]))


# ------------------------------------------------------------ include
@registry.register("include")
def cmd_include(script, args):
    filename = args[0]
    base = getattr(script, "_script_dir", None)
    if base:
        path = base / filename
        if not path.is_absolute() and not path.is_file():
            path = base / filename
    else:
        path = __import__("pathlib").Path(filename)
    text = path.read_text()
    script.run_string(text)


# ------------------------------------------------------------ set_output_dir
@registry.register("set_output_dir")
def cmd_set_output_dir(script, args):
    cfg = _config(script)
    cfg.output_dir = args[0]


# ------------------------------------------------------------- seed
@registry.register("seed")
def cmd_seed(script, args):
    cfg = _config(script)
    cfg.random_seed = int(float(args[0]))


# ------------------------------------------------------------- reset_timestep
@registry.register("reset_timestep")
def cmd_reset_timestep(script, args):
    sim = _sim(script)
    if sim.state is not None:
        if args:
            sim.state.step = int(args[0])
        else:
            sim.state.step = 0


logger.info("LAMMPS-style command registry populated with: %s", sorted(registry.names()))
