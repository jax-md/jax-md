"""
JaxMD Martini Coarse-Grained Simulation of Chignolin

Usage: python martini_nvt_simulation.py pdb_file_location gromacs_directory_location output_directory_location
Defaults are: 
PDB_FILE = "data/martini-data/step3_charmm2gmx.pdb"
TEST_DIR = "data/martini-data/gromacs/"
OUTPUT_DIR = "output/"

"""

from functools import partial
import sys
import time
from jax import debug, lax
from jax_md import space
import numpy as np

import jax

from jax_md.mm_forcefields.martini.lincs import (
    LincsTopology,
    make_lincs_apply_fn,
)

from jax_md.mm_forcefields.martini.topology import create_topology

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit

import mdtraj as md

from pathlib import Path

from jax_md import minimize, simulate
from jax_md.mm_forcefields import neighbor

from jax_md.mm_forcefields.martini.cm_motion_remover import make_cm_remover
from jax_md.mm_forcefields.martini.energy import energy_fn
from jax_md.mm_forcefields.martini.top_file_parser import GromacsTopFile, Atom
from jax_md.units import gromacs_unit_system

unit = gromacs_unit_system()

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
PDB_FILE = "data/martini-data/step3_charmm2gmx.pdb"
TEST_DIR = "data/martini-data/gromacs/"
OUTPUT_DIR = "output/"

TEMPERATURE_K = 310.0  # kelvin
kT = TEMPERATURE_K * unit["temperature"]

MINIMIZE_STEPS = 5_000
MINIMIZE_TOL = 1.0  # kJ mol⁻¹ nm⁻¹  (force norm convergence criterion)

EQUILIBRATION_STEPS = 50_000
PRODUCTION_STEPS = 100_000
DT = 0.01 * unit["time"]  # 10 fs
FRICTION = 10 / unit["time"]  # 10 ps⁻¹  (Langevin friction coefficient)

SAVE_EVERY = 1000  # save a frame every N production steps

R_CUT = 1.1 * unit["distance"]  # 1.1 nm
EPSILON_R = 15


def load_structure(pdb_file: str):
    """
    Load a Martini CG PDB with mdTraj.

    Returns
    -------
    positions : jnp.ndarray, shape (N, 3), float32, units nm
    box       : jnp.ndarray, shape (3,),   float32, units nm  (orthorhombic lengths)
    traj      : mdtraj.Trajectory  (retained for topology access downstream)
    """
    traj = md.load(pdb_file)

    positions = jnp.array(traj.xyz[0], dtype=jnp.float64)

    if traj.unitcell_lengths is None:
        raise ValueError(
            "No unit cell found in PDB. "
            "Ensure a CRYST1 record is present in the file."
        )
    box = jnp.array(traj.unitcell_lengths[0], dtype=jnp.float64)
    return positions, box, traj


def map_positions_to_ff_ordering(
    traj: md.Trajectory,
    ff_atoms: list[Atom],
) -> jnp.ndarray:
    """
    Verify that the PDB atom sequence matches the FF atom ordering produced
    by _build_atom_and_mass_list, then return positions in that order.

    Matching is done by (residue_name, atom_name) at each flat index position.

    Parameters
    ----------
    traj     : mdTraj Trajectory parsed from the PDB
    ff_atoms : list[Atom] in the flat order from _build_atom_and_mass_list

    Returns
    -------
    positions : jnp.ndarray, shape (N, 3), float32, nm, in FF atom order
    """
    mdtraj_atoms = list(traj.topology.atoms)

    if len(mdtraj_atoms) != len(ff_atoms):
        raise ValueError(
            f"Atom count mismatch: PDB has {len(mdtraj_atoms)} atoms, "
            f"FF topology has {len(ff_atoms)}. Cannot continue."
        )

    for flat_idx, (mda, ffa) in enumerate(zip(mdtraj_atoms, ff_atoms)):
        pdb_resname = mda.residue.name
        pdb_atmname = mda.name
        ff_resname = ffa.residue_name
        ff_atmname = ffa.atom_name

        if (
            pdb_resname != ff_resname
            and ff_resname
            != "ION"  # Ions have specific residue names in the pdb file but not in the top file
        ) or pdb_atmname != ff_atmname:
            raise ValueError(
                f"Atom mismatch at flat index {flat_idx}: "
                f"PDB has (resname='{pdb_resname}', name='{pdb_atmname}'), "
                f"FF topology expects (resname='{ff_resname}', name='{ff_atmname}'). "
                f"Cannot continue."
            )

    return jnp.array(traj.xyz[0], dtype=jnp.float64)


def run_energy_minimization(
    E_jit,
    box,
    positions,
    shift_fn,
    neighbor_fn,
    martini_topology,
    displacement_fn,
    lincs_topology,
):

    @partial(
        jit,
        static_argnums=(2,),
    )
    def step_fn(i, state_nbrs, apply_fn):
        state, nbrs = state_nbrs
        t = i * DT
        state = apply_fn(
            i,
            state,
            box=box,
            neighbor=nbrs,
        )
        nbrs = nbrs.update(state.position)
        return state, nbrs

    n_beads = positions.shape[0]
    n_list = neighbor_fn.allocate(positions)
    t0 = time.time()

    vsite_mask = martini_topology.masses > 0

    init_min, apply_min = minimize.fire_descent(
        E_jit, shift_fn, dt_start=0.0001, dt_max=0.01
    )
    min_state = init_min(positions, martini_topology.masses, box=box, neighbor=n_list)

    apply_min_w_lincs = make_lincs_apply_fn(
        apply_min, lincs_topology, DT, displacement_fn, shift_fn, nrec=4
    )

    apply_min_with_cm_removal = make_cm_remover(apply_min_w_lincs, freq=1)

    norm_safe = lambda norm: jnp.maximum(norm, 1.0)

    real_positions = jnp.where(vsite_mask[:, None], min_state.position, 0.0)

    normalized_tolerance = MINIMIZE_TOL / norm_safe(
        (jnp.sum(jnp.linalg.norm(real_positions, axis=-1) ** 2) / n_beads)
    )
    print("Normalized tolerance is: ", normalized_tolerance)

    for step in range(MINIMIZE_STEPS):
        min_state, n_list = step_fn(
            step, (min_state, n_list), apply_min_with_cm_removal
        )
        if n_list.did_buffer_overflow:
            n_list = neighbor_fn.allocate(min_state.position)

        if (step + 1) % 100 == 0:
            cur_E = E_jit(min_state.position, box=box, neighbor=n_list)
            g_norm = jnp.linalg.norm(min_state.force)
            real_positions = jnp.where(vsite_mask[:, None], min_state.position, 0.0)
            x_norm = norm_safe(jnp.linalg.norm(real_positions, ord="fro"))
            force_norm = g_norm / x_norm

            print(
                f"      step {step+1:5d} | "
                f"E = {cur_E:10.3f} kJ/mol | "
                f"Force_norm = {force_norm:.4f} kJ/mol/nm"
            )
            if force_norm < normalized_tolerance:
                print(f"      Converged at step {step + 1}.")
                break

    min_pos = min_state.position
    print(f"      Minimisation wall time : {time.time() - t0:.1f} s")
    print(
        f"      Final energy           : {E_jit(min_pos, box=box, neighbor=n_list):.3f} kJ/mol"
    )
    return min_pos, vsite_mask[:, None]


def run_nvt(
    E_jit,
    shift_fn,
    displacement_fn,
    box,
    positions,
    neighbor_fn,
    martini_topology,
    t0,
    lincs_topology,
):
    n = positions.shape[0]

    key = jax.random.key(42)

    init_sim, apply_sim = simulate.nvt_langevin(
        E_jit,
        shift_fn,
        dt=DT,
        kT=kT,
        gamma=FRICTION,
    )

    key, subkey = jax.random.split(key)
    n_list = neighbor_fn.allocate(positions)
    sim_state = init_sim(
        subkey,
        positions,
        box=box,
        neighbor=n_list,
        mass=martini_topology.masses,
        kT=kT,
    )

    apply_sim_w_lincs = make_lincs_apply_fn(
        apply_sim, lincs_topology, DT, displacement_fn, shift_fn, nrec=4
    )

    apply_fn_with_cm_removal = make_cm_remover(apply_sim_w_lincs, freq=1)
    apply_fn = jit(apply_fn_with_cm_removal)

    last_time = t0

    log = {
        "E": jnp.zeros((PRODUCTION_STEPS // SAVE_EVERY,)),
        "Pos": jnp.zeros((PRODUCTION_STEPS // SAVE_EVERY, n, 3)),
        "kT": jnp.zeros((PRODUCTION_STEPS // SAVE_EVERY,)),
    }

    def inner_step_fn(i, state_nbrs):
        state, nbrs = state_nbrs
        t = i * DT
        state = apply_fn(i, state, box=box, neighbor=nbrs)
        nbrs = nbrs.update(state.position)
        return state, nbrs

    @jit
    def outer_sim_fn(j, state_nbrs_log):
        state, nbrs, log = state_nbrs_log
        E = E_jit(state.position, box=box, neighbor=nbrs)

        log["E"] = log["E"].at[j].set(E)
        log["Pos"] = log["Pos"].at[j].set(state.position)

        def inner_sim_fn(i, state_nbrs):
            return inner_step_fn(i, state_nbrs)

        state, nbrs = lax.fori_loop(0, SAVE_EVERY, inner_sim_fn, (state, nbrs))

        return state, nbrs, log

    nbrs = neighbor_fn.allocate(sim_state.position)
    for j in range(int(PRODUCTION_STEPS // SAVE_EVERY)):
        sim_state, nbrs, log = outer_sim_fn(j, (sim_state, nbrs, log))

        now = time.time()
        elapsed_since_last = now - last_time
        last_time = now
        ns_per_day_curr = SAVE_EVERY * DT / 1000.0 / elapsed_since_last * 86400

        debug.print(
            "Step = {j} | Total Energy = {T} | perf = {ns_per_day_curr:.2f} ns/day",
            j=j * SAVE_EVERY,
            T=log["E"][j],
            ns_per_day_curr=ns_per_day_curr,
        )
        if nbrs.did_buffer_overflow:
            nbrs = neighbor_fn.allocate(sim_state.position)
            debug.print(f"Neighbor list overflow at step {j * SAVE_EVERY}")
    return sim_state.position, log["Pos"], log["E"]


def run_simulation(
    pdb_file: Path = Path(PDB_FILE),
    test_dir: Path = Path(TEST_DIR),
    output_dir: Path = Path(OUTPUT_DIR),
):
    print("=" * 60)
    print("  Chignolin Martini MD — JaxMD")
    print("=" * 60)

    print(f"\n[1/4] Loading structure: {pdb_file}")
    positions, box, traj = load_structure(pdb_file)

    print(f"      Beads      : {positions.shape[0]}")
    print(f"      Box (nm)   : {np.around(np.array(box), 3)}")

    print("\n[2/4] Initialising neighbour list and energy fn …")
    displacement_fn, shift_fn = space.periodic_general(
        box, fractional_coordinates=False
    )
    neighbor_fn = neighbor.create_neighbor_list(
        displacement_fn,
        box,
        R_CUT,
    )

    topology_file = GromacsTopFile(test_dir / "system.top", epsilon_r=EPSILON_R)

    martini_topology = create_topology(
        topology_file, nonbonded_cutoff=R_CUT, epsilon_r=EPSILON_R
    )

    lincs_topology = LincsTopology.from_topology(topology_file, martini_topology.masses)

    positions = map_positions_to_ff_ordering(traj, martini_topology.atoms)

    martini_energy_fn = energy_fn(martini_topology, displacement_fn, shift_fn)

    E_jit = jit(martini_energy_fn)

    n_list = neighbor_fn.allocate(positions)

    print(
        f"      Initial energy : {E_jit(positions, box=box, neighbor=n_list):.3f} kJ/mol"
    )

    print(
        f"\n[3/4] Energy minimisation "
        f"({MINIMIZE_STEPS} steps, tol = {MINIMIZE_TOL} kJ/mol/nm) …"
    )
    min_pos, vsite_mask = run_energy_minimization(
        E_jit,
        box,
        positions,
        shift_fn,
        neighbor_fn,
        martini_topology,
        displacement_fn,
        lincs_topology,
    )

    print(
        f"\n[4/4] Langevin NVT MD — "
        f"{TEMPERATURE_K} K | {PRODUCTION_STEPS} steps × {DT} ps …"
    )
    t0 = time.time()
    positions, trajectory, energies = run_nvt(
        E_jit,
        shift_fn,
        displacement_fn,
        box,
        min_pos,
        neighbor_fn,
        martini_topology,
        t0,
        lincs_topology,
    )

    # -- Save outputs ---------------------------------------------------------
    traj_arr = np.array(trajectory)  # (n_frames, n_beads, 3)
    traj_xtc = xtc_from_positions(traj.topology, trajectory, output_dir, vsite_mask)
    traj_xtc.save_xtc(f"{output_dir}/positions.xtc")

    print(f"\nTrajectory saved → {output_dir}/positions.xtc shape={traj_arr.shape}")
    print(f"Total wall time  : {time.time() - t0:.1f} s")

    return traj_arr


def xtc_from_positions(topology, positions, output_dir, vsite_mask):
    positions = np.where(vsite_mask, positions, 0.0)
    traj_out = md.Trajectory(
        xyz=np.array(positions),
        topology=topology,
        time=np.arange(len(positions)) * DT,  # optional, in picoseconds
    )
    return traj_out


if __name__ == "__main__":
    pdb = sys.argv[1] if len(sys.argv) > 1 else PDB_FILE
    test_dir = sys.argv[2] if len(sys.argv) > 2 else TEST_DIR
    output_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR
    run_simulation(Path(pdb), Path(test_dir), Path(output_dir))
