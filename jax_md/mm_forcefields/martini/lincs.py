"""
This module provides a Jax-compatible implementation of the LINCS
(LINear Constraint Solver) algorithm. Implementation based on
Hess et al., J. Comp. Chem. 18:1463-1472 (1997).

Position_step overrides provided will enable NVT and NVE simulations with
LINCS. To run NPT simulations, similar functionality must be added directly into
simulate.npt_nose_hoover.
"""

from __future__ import annotations

from typing import NamedTuple
from collections import defaultdict

from jax import vmap
import jax.numpy as jnp
from jax_md import space
from jax_md.util import safe_norm, Array
import numpy as np

from jax_md.mm_forcefields.martini.top_file_parser import GromacsTopFile


class LincsTopology(NamedTuple):
  """Pre-computed, static topology data for LINCS.

  All integer arrays are plain NumPy so they can serve as static tracers
  inside jit-compiled functions (passed via partial / closure).
  The float arrays (constraint_lengths, Sdiag, coef) are JAX arrays.

  Attributes:
    atom1: First atom of each constraint. Shape (K,).
    atom2: Second atom of each constraint. Shape (K,).
    constraint_lengths: Target constraint lengths d_i. Shape (K,).
    Sdiag: 1/sqrt(1/m_{i1} + 1/m_{i2}). Shape (K,).
    masses: Atom masses. Shape (N,).
    inv_masses: Reciprocal atom masses 1/m_i. Shape (N,).
    ncc: Number of constraints coupled to each constraint. Shape (K,).
    conn: Index of the j-th coupled constraint, 0-padded. Shape (K, cmax).
    coef_sign: +1 or -1 sign for each coupled pair. Shape (K, cmax).
    coef_mass: invmass[shared_atom] * Sdiag_i * Sdiag_j, 0 for padded entries.
      Shape (K, cmax).
    cmax: Padded width of connectivity arrays.
  """

  atom1: np.ndarray
  atom2: np.ndarray
  constraint_lengths: Array
  Sdiag: Array
  masses: Array
  inv_masses: Array
  ncc: np.ndarray
  conn: np.ndarray
  coef_sign: Array
  coef_mass: Array
  cmax: int

  @classmethod
  def from_topology(cls, top: GromacsTopFile, masses: np.ndarray | Array):
    """Build LincsTopology from a Gromacs Topology file.

    Args:
      top: Gromacs Topology File.
      masses: Masses of atoms.

    Returns:
      topo: LincsTopology object.
    """
    if isinstance(masses, Array):
      masses = np.asarray(masses)
    constraint_pairs = []
    constraint_lengths = []
    base_atom_index = 0
    for mol_idx, mol in enumerate(top._molecules):
      mol_type = top._moleculeTypes[mol.name]
      n_atoms = len(mol_type.atoms)
      for _ in range(mol.count):
        for constraint in mol_type.constraints:
          constraint_atoms = []
          for atom in constraint.atoms:
            constraint_atoms.append(base_atom_index + atom)
          constraint_pairs.append(constraint_atoms)
          constraint_lengths.append(constraint.distance)
        base_atom_index += n_atoms
    constraint_pairs = np.asarray(constraint_pairs)
    constraint_lengths = np.asarray(constraint_lengths)
    return cls.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )

  @classmethod
  def build_lincs_topology(
    cls,
    constraint_pairs: np.ndarray,
    masses: np.ndarray,
    constraint_lengths: np.ndarray,
  ) -> LincsTopology:
    """Pre-compute all static topology data needed by lincs_step.

    Args:
      constraint_pairs: Atom index pairs for each constraint. Shape (K, 2).
      masses: Atom masses. Shape (N,).
      constraint_lengths: Target constraint lengths. Shape (K,).

    Returns:
      topo: LincsTopology named tuple ready to pass into lincs_step.
    """
    constraint_pairs = np.asarray(constraint_pairs, dtype=np.int32)
    constraint_lengths = np.asarray(constraint_lengths, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)

    vsite_mask = masses == 0.0
    constrained_atoms = np.unique(constraint_pairs)
    bad = constrained_atoms[vsite_mask[constrained_atoms]]
    if len(bad) > 0:
      raise ValueError(
        f'Atoms {bad.tolist()} have zero mass (virtual sites) but appear in constraint pairs.'
        'LINCS constraints do not support virtual sites.'
      )

    safe_masses = np.where(vsite_mask, np.inf, masses)
    K = len(constraint_pairs)
    inv_masses = 1.0 / safe_masses

    a1s = constraint_pairs[:, 0]
    a2s = constraint_pairs[:, 1]
    Sdiag = 1.0 / np.sqrt(inv_masses[a1s] + inv_masses[a2s])

    conn_lists: list[list[int]] = [[] for _ in range(K)]
    sign_lists: list[list[float]] = [[] for _ in range(K)]
    cmass_lists: list[list[float]] = [[] for _ in range(K)]

    atom_to_constraints: dict[int, list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(constraint_pairs):
      atom_to_constraints[int(a)].append(i)
      atom_to_constraints[int(b)].append(i)

    for i in range(K):
      a1_i, a2_i = int(constraint_pairs[i, 0]), int(constraint_pairs[i, 1])
      neighbours = set()
      for shared_atom in (a1_i, a2_i):
        for j in atom_to_constraints[shared_atom]:
          if j == i or j in neighbours:
            continue
          neighbours.add(j)
          a1_j, a2_j = int(constraint_pairs[j, 0]), int(constraint_pairs[j, 1])
          if (shared_atom == a1_i and shared_atom == a1_j) or (
            shared_atom == a2_i and shared_atom == a2_j
          ):
            sign = -1.0
          else:
            sign = 1.0
          cmass = inv_masses[shared_atom] * Sdiag[i] * Sdiag[j]
          conn_lists[i].append(j)
          sign_lists[i].append(sign)
          cmass_lists[i].append(cmass)

    cmax = max((len(c) for c in conn_lists), default=1)

    ncc = np.array([len(c) for c in conn_lists], dtype=np.int32)
    conn = np.zeros((K, cmax), dtype=np.int32)
    coef_sign = np.zeros((K, cmax), dtype=np.float64)
    coef_mass = np.zeros((K, cmax), dtype=np.float64)

    for i in range(K):
      n = len(conn_lists[i])
      if n > 0:
        conn[i, :n] = conn_lists[i]
        coef_sign[i, :n] = sign_lists[i]
        coef_mass[i, :n] = cmass_lists[i]

    return LincsTopology(
      atom1=a1s,
      atom2=a2s,
      constraint_lengths=jnp.array(constraint_lengths),
      Sdiag=jnp.array(Sdiag),
      masses=jnp.array(masses),
      inv_masses=jnp.array(inv_masses),
      ncc=ncc,
      conn=conn,
      coef_sign=jnp.array(coef_sign),
      coef_mass=jnp.array(coef_mass),
      cmax=cmax,
    )


def _compute_B(
  positions: Array,
  topo: LincsTopology,
  displacement_fn: space.DisplacementOrMetricFn,
) -> Array:
  """Compute unit constraint direction vectors from previous positions.

  Args:
    positions: Previous atom positions. Shape (N, 3).
    topo: Pre-built LincsTopology.
    displacement_fn: Displacement function respecting boundary conditions.

  Returns:
    B: Unit constraint vectors B[i] = (r_{a1} - r_{a2}) / |...|. Shape (K, 3).
  """
  r1 = positions[topo.atom1]
  r2 = positions[topo.atom2]
  diff = vmap(displacement_fn)(r1, r2)
  norm = safe_norm(diff, axis=-1, keepdims=True)
  return diff / norm


def _build_A(B: Array, topo: LincsTopology) -> Array:
  """Build the normalised coupling matrix.

  Args:
    B: Unit constraint direction vectors. Shape (K, 3).
    topo: Pre-built LincsTopology.

  Returns:
    A: Coupling matrix A[i, j] = coef_sign * coef_mass * (B_i · B_j).
      Shape (K, cmax). Padded entries are zero.
  """
  B_conn = B[topo.conn]
  B_i = B[:, None, :]
  Bdot = jnp.sum(B_i * B_conn, axis=-1)
  return topo.coef_sign * topo.coef_mass * Bdot


def _solve(
  rhs_init: Array,
  A: Array,
  topo: LincsTopology,
  nrec: int,
) -> Array:
  """Apply the Neumann series approximation of (I - A)^{-1}.

  Computes sol = sum_{k=0}^{nrec} A^k * rhs_init via iterative expansion.
  Implements the SOLVE subroutine from Appendix 3.B.

  Args:
    rhs_init: Initial right-hand side vector. Shape (K,).
    A: Coupling matrix. Shape (K, cmax).
    topo: Pre-built LincsTopology.
    nrec: Number of expansion iterations. Must be a concrete Python int.

  Returns:
    sol: Approximate solution. Shape (K,).
  """
  col_idx = jnp.arange(topo.cmax)[None, :]
  mask = jnp.array(col_idx < topo.ncc[:, None])
  sol = rhs_init
  prev_rhs = rhs_init
  for _ in range(nrec):
    prev_rhs_conn = jnp.where(mask, prev_rhs[topo.conn], 0.0)
    new_rhs = jnp.sum(A * prev_rhs_conn, axis=-1)
    sol = sol + new_rhs
    prev_rhs = new_rhs
  return sol


def _apply_sol(
  positions: Array,
  sol: Array,
  B: Array,
  topo: LincsTopology,
) -> Array:
  """Scatter constraint corrections back to atom positions.

  xp[a1] -= inv_mass[a1] * B[i] * Sdiag[i] * sol[i]
  xp[a2] += inv_mass[a2] * B[i] * Sdiag[i] * sol[i]

  Args:
    positions: Current atom positions. Shape (N, 3).
    sol: Constraint solution vector. Shape (K,).
    B: Unit constraint direction vectors. Shape (K, 3).
    topo: Pre-built LincsTopology.

  Returns:
    positions: Corrected atom positions. Shape (N, 3).
  """
  scale = topo.Sdiag * sol
  delta = B * scale[:, None]
  corr = jnp.zeros_like(positions)
  corr = corr.at[topo.atom1].add(-topo.inv_masses[topo.atom1, None] * delta)
  corr = corr.at[topo.atom2].add(topo.inv_masses[topo.atom2, None] * delta)
  return positions + corr


def lincs_positions(
  positions_old: Array,
  positions_unc: Array,
  topo: LincsTopology,
  displacement_fn: space.DisplacementOrMetricFn,
  nrec: int = 4,
) -> tuple[Array, Array, Array]:
  """Apply LINCS to correct constraint lengths in the new positions.

  Args:
    positions_old: Positions at the start of the timestep r_n. Shape (N, 3).
    positions_unc: Unconstrained positions after the force/velocity update.
      Shape (N, 3).
    topo: Pre-built LincsTopology.
    displacement_fn: Displacement function respecting boundary conditions.
    nrec: Expansion order.

  Returns:
    positions_con: Constrained positions. Shape (N, 3).
  """
  B = _compute_B(positions_old, topo, displacement_fn)
  A = _build_A(B, topo)

  # ---- Step 1: primary projection ----------------------------------------
  r1_unc = positions_unc[topo.atom1]
  r2_unc = positions_unc[topo.atom2]
  constraint_proj = jnp.sum(B * vmap(displacement_fn)(r1_unc, r2_unc), axis=-1)
  rhs1 = topo.Sdiag * (constraint_proj - topo.constraint_lengths)
  sol1 = _solve(rhs1, A, topo, nrec)
  pos_con = _apply_sol(positions_unc, sol1, B, topo)

  # ---- Step 2: rotational lengthening correction -------------------------
  r1_con = pos_con[topo.atom1]
  r2_con = pos_con[topo.atom2]
  l2 = jnp.sum(vmap(displacement_fn)(r1_con, r2_con) ** 2, axis=-1)
  p = jnp.sqrt(jnp.maximum(2.0 * topo.constraint_lengths**2 - l2, 0.0))
  rhs2 = topo.Sdiag * (topo.constraint_lengths - p)
  sol2 = _solve(rhs2, A, topo, nrec)
  pos_con = _apply_sol(pos_con, sol2, B, topo)

  mlambda = topo.Sdiag * (sol1 + sol2)

  return pos_con, B, mlambda


def apply_lincs(
  R_old_frac,
  R_unc_frac,
  topo,
  displacement_fn,
  dt,
  box=None,
  fractional_coordinates=False,
):
  """Calculate constrained positions and correct velocities to match new positions.

  Args:
    R_old_frac: Positions at the start of the timestep, optionally in fractional
      coordinates. Shape (N, 3).
    R_unc_frac: Unconstrained positions after the force/velocity update, optionally
      in fractional coordinates. Shape (N, 3).
    topo: Pre-built LincsTopology.
    displacement_fn: Displacement function respecting boundary conditions.
    dt: Integration timestep.
    box: Box tensor for periodic boundary conditions. Required when fractional=True.
    fractional: Whether positions are given in fractional coordinates.

  Returns:
    R_constrained: Constrained positions in the same coordinate system as input.
      Shape (N, 3).
    dV: Velocity correction in the same coordinate system as input. Shape (N, 3).
  """
  if fractional_coordinates and box is not None:
    R_old = space.transform(box, R_old_frac)
    R_unc = space.transform(box, R_unc_frac)
    displacement_fn, _ = space.periodic_general(
      box, fractional_coordinates=False
    )
  else:
    R_old = R_old_frac
    R_unc = R_unc_frac

  def real_disp_fn(Ra, Rb, **kwargs):
    if box is not None:
      kwargs = dict(kwargs, box=box)
    return displacement_fn(Ra, Rb, **kwargs)

  R_constrained_real, B, mlambda = lincs_positions(
    R_old, R_unc, topo, real_disp_fn, nrec=4
  )

  R_constrained_real, B, mlambda = lincs_positions(
    R_old, R_unc, topo, real_disp_fn, nrec=4
  )

  if fractional_coordinates and box is not None:
    inv_box = space.inverse(box)
    R_constrained = space.transform(inv_box, R_constrained_real)
  else:
    R_constrained = R_constrained_real

  scale = mlambda / dt
  delta = B * scale[:, None]
  dV_real = jnp.zeros((topo.inv_masses.shape[0], 3))
  dV_real = dV_real.at[topo.atom1].add(
    -topo.inv_masses[topo.atom1, None] * delta
  )
  dV_real = dV_real.at[topo.atom2].add(
    topo.inv_masses[topo.atom2, None] * delta
  )

  if fractional_coordinates and box is not None:
    inv_box = space.inverse(box)
    dV = space.transform(inv_box, dV_real)
  else:
    dV = dV_real

  return R_constrained, dV


def make_lincs_apply_fn(
  apply_fn,
  topo: LincsTopology,
  dt: float,
  displacement_fn,
  shift_fn,
  nrec: int = 4,
):
  """Wrap a JaxMD integrator apply function with LINCS position constraints.

  Args:
    apply_fn: JaxMD integrator apply function acting on a JaxMD simulation state
              (ex. NVEState or NVTState).
    topo: Pre-built LincsTopology.
    dt: Simulation timestep.
    displacement_fn: Displacement function respecting boundary conditions.
    shift_fn: Shift function for wrapping positions into the simulation box.
    nrec: Expansion order for the LINCS solver.

  Returns:
    apply_fn_w_lincs: Function with the same signature as apply_fn
      that returns a state with constrained positions.
  """

  def apply_fn_w_lincs(state, **kwargs):
    """Apply the integrator and apply correct positions using LINCS.

    Args:
      state: JaxMD NVEState or NVTState with a position field.
      **kwargs: Additional keyword arguments forwarded to apply_fn.

    Returns:
      state: Updated state with constrained and box-wrapped positions.
    """
    positions_old = state.position
    state = apply_fn(state, **kwargs)

    pos_con, dV = apply_lincs(
      positions_old, state.position, topo, displacement_fn, dt
    )
    pos_con_wrapped = shift_fn(positions_old, pos_con - positions_old)
    return state.set(
      position=pos_con_wrapped,
      momentum=state.momentum + state.mass * dV,
    )

  return apply_fn_w_lincs
