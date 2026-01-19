""" This module implements common distance constraint algorithms.

References:
SHAKE - Ryckaert (1977)
https://doi.org/10.1016/0021-9991(77)90098-5

RATTLE - Andersen (1983)
https://doi.org/10.1016/0021-9991(83)90014-1

SETTLE - Miyamoto (1992)
https://doi.org/10.1002/jcc.540130805

CCMA - Eastman (2010)
https://pmc.ncbi.nlm.nih.gov/articles/PMC2885791/

SETTLE and CCMA are based off OpenMM implementation
in ReferenceSETTLEAlgorithm.cpp and ReferenceCCMAAlgorithm.cpp
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp

from jax_md import dataclasses
from jax_md.util import Array, normalize
from jax_md.mm_forcefields.reaxff.reaxff_helper import safe_sqrt


def settle(
  pos: Array,
  pos_p: Array,
  settle_data: 'SettleData',
  masses: Array,
  displacement_fn,
  shift_fn,
  *,
  box: Optional[Array] = None,
  use_periodic_general: bool = False,
) -> Array:
  """Apply SETTLE position constraints for 3-atom rigid clusters."""
  return settle_apply_positions(
    pos,
    pos_p,
    settle_data,
    masses,
    displacement_fn,
    shift_fn,
    box=box,
    use_periodic_general=use_periodic_general,
  )


def ccma(
  pos: Array,
  constraints: 'CCMAData',
  masses: Array,
  displacement_fn,
  shift_fn,
  *,
  box: Optional[Array] = None,
  use_periodic_general: bool = False,
  tolerance: float = 1e-6,
  max_iters: int = 100,
) -> Array:
  """Apply general distance constraints."""
  return ccma_apply_positions(
    pos,
    constraints,
    masses,
    displacement_fn,
    shift_fn,
    box=box,
    use_periodic_general=use_periodic_general,
    tolerance=tolerance,
    max_iters=max_iters,
  )


@dataclasses.dataclass
class SettleData:
  """Parameters for SETTLE clusters (rigid 3-atom distance-constraint loops).

  This mirrors the data extracted by OpenMM's ReferenceConstraints when it
  detects 3-atom closed loops of constraints that can be solved analytically.

  Attributes:
    atom1: (n_cluster,) central atom indices.
    atom2: (n_cluster,) partner atom indices.
    atom3: (n_cluster,) partner atom indices.
    distance1: (n_cluster,) distance from atom1 to atom2/atom3 (A).
    distance2: (n_cluster,) distance between atom2 and atom3 (A).
  """

  atom1: Array
  atom2: Array
  atom3: Array
  distance1: Array
  distance2: Array


@dataclasses.dataclass
class CCMAData:
  """Parameters for general distance constraints solved iteratively.

  This is an intentionally simplified, JAX-friendly analogue of OpenMM's CCMA
  implementation. It uses the same per-constraint update equations as OpenMM's
  ReferenceCCMAAlgorithm::applyConstraints(), but does not include the (very
  expensive) sparse inverse coupling matrix precomputation used to accelerate
  convergence in OpenMM.

  Attributes:
    idx: (n_constraint, 2) atom index pairs.
    dist: (n_constraint,) target distances (A).
  """

  idx: Array
  dist: Array


def prepare_settle_ccma(
  constraint_idx: onp.ndarray,
  constraint_dist: onp.ndarray,
  masses: onp.ndarray,
) -> Tuple[SettleData, CCMAData]:
  """Split OpenMM System distance constraints into SETTLE clusters and the rest.

  This ports the cluster-detection logic from OpenMM's Reference platform:
  ReferenceConstraints.cpp.

  Args:
    constraint_idx: (n_constraint, 2) atom index pairs.
    constraint_dist: (n_constraint,) target distances (A).
    masses: (n_atom,) masses (daltons/amu).

  Returns:
    settle: SettleData with 0 or more clusters.
    ccma: CCMAData with all remaining constraints not handled by SETTLE.
  """
  n_atoms = int(masses.shape[0])
  if constraint_idx.size == 0:
    empty = jnp.asarray(onp.zeros((0,), dtype=onp.int32))
    empty_f = jnp.asarray(onp.zeros((0,), dtype=onp.float64))
    settle = SettleData(empty, empty, empty, empty_f, empty_f)
    ccma = CCMAData(jnp.asarray(onp.zeros((0, 2), dtype=onp.int32)), empty_f)
    return settle, ccma

  idx = onp.asarray(constraint_idx, dtype=onp.int32)
  dist = onp.asarray(constraint_dist, dtype=onp.float64)
  masses = onp.asarray(masses, dtype=onp.float64)

  # Same as OpenMM: skip constraints between two massless particles.
  keep = ~((masses[idx[:, 0]] == 0.0) & (masses[idx[:, 1]] == 0.0))
  idx = idx[keep]
  dist = dist[keep]

  # Count how many constraints touch each atom.
  # NOTE repeated index accumulation can produce unusual behavior
  # https://stackoverflow.com/questions/2004364/increment-numpy-array-with-repeated-indices
  constraint_count = onp.zeros((n_atoms,), dtype=onp.int32)
  onp.add.at(constraint_count, idx[:, 0], 1)
  onp.add.at(constraint_count, idx[:, 1], 1)

  # Build adjacency only for atoms involved in exactly 2 constraints
  settle_adj = [dict() for _ in range(n_atoms)]
  for (a, b), d in zip(idx, dist):
    if constraint_count[a] == 2 and constraint_count[b] == 2:
      settle_adj[a][int(b)] = float(d)
      settle_adj[b][int(a)] = float(d)

  # Keep only closed 3-atom loops and pick lowest idx as representative
  settle_centers = []
  for a in range(n_atoms):
    if len(settle_adj[a]) != 2:
      settle_adj[a].clear()
      continue
    partners = list(settle_adj[a].keys())
    b, c = partners[0], partners[1]
    if len(settle_adj[b]) != 2 or len(settle_adj[c]) != 2 or c not in settle_adj[b]:
      settle_adj[a].clear()
      continue
    if a < b and a < c:
      settle_centers.append(a)

  is_settle_atom = onp.zeros((n_atoms,), dtype=onp.bool_)
  a1, a2, a3, d1, d2 = [], [], [], [], []

  for a in settle_centers:
    partners = list(settle_adj[a].keys())
    b, c = partners[0], partners[1]
    dist_ab = float(settle_adj[a][b])
    dist_ac = float(settle_adj[a][c])
    dist_bc = float(settle_adj[b][c])

    # Identify the central atom as the one with two equal distances
    if dist_ab == dist_ac:
      c1, c2, c3 = a, b, c
      cd1, cd2 = dist_ab, dist_bc
    elif dist_ab == dist_bc:
      c1, c2, c3 = b, a, c
      cd1, cd2 = dist_ab, dist_ac
    elif dist_ac == dist_bc:
      c1, c2, c3 = c, a, b
      cd1, cd2 = dist_ac, dist_ab
    else:
      continue

    a1.append(c1)
    a2.append(c2)
    a3.append(c3)
    d1.append(cd1)
    d2.append(cd2)
    is_settle_atom[a] = True
    is_settle_atom[b] = True
    is_settle_atom[c] = True

  # Remaining constraints go to CCMA. OpenMM identifies SETTLE clusters and then
  # excludes constraints that belong to those clusters.
  settle_edges = set()
  for c1, c2, c3 in zip(a1, a2, a3):
    settle_edges.add(tuple(sorted((int(c1), int(c2)))))
    settle_edges.add(tuple(sorted((int(c1), int(c3)))))
    settle_edges.add(tuple(sorted((int(c2), int(c3)))))
  if settle_edges:
    edge_keys = [tuple(sorted((int(x), int(y)))) for x, y in idx]
    ccma_keep = onp.asarray([ek not in settle_edges for ek in edge_keys], dtype=onp.bool_)
    ccma_idx = idx[ccma_keep]
    ccma_dist = dist[ccma_keep]
  else:
    ccma_idx = idx
    ccma_dist = dist

  settle = SettleData(
    atom1=jnp.asarray(onp.asarray(a1, dtype=onp.int32)),
    atom2=jnp.asarray(onp.asarray(a2, dtype=onp.int32)),
    atom3=jnp.asarray(onp.asarray(a3, dtype=onp.int32)),
    distance1=jnp.asarray(onp.asarray(d1, dtype=onp.float64)),
    distance2=jnp.asarray(onp.asarray(d2, dtype=onp.float64)),
  )
  ccma = CCMAData(
    idx=jnp.asarray(onp.asarray(ccma_idx, dtype=onp.int32)),
    dist=jnp.asarray(onp.asarray(ccma_dist, dtype=onp.float64)),
  )
  return settle, ccma


def ccma_apply_positions(
  pos: Array,
  constraints: CCMAData,
  masses: Array,
  displacement_fn,
  shift_fn,
  *,
  box: Optional[Array] = None,
  use_periodic_general: bool = False,
  tolerance: float = 1e-6,
  max_iters: int = 150,
) -> Array:
  """Apply distance constraints to positions (CCMA-style iterative projection).

  This enforces constraints on the current positions, analogous to OpenMM
  context.applyConstraints() on the Reference platform.
  """
  if constraints is None or constraints.idx.size == 0:
    return pos

  box_kwargs = {'box': box} if use_periodic_general else {}
  idx = constraints.idx
  dist = constraints.dist
  i = idx[:, 0]
  j = idx[:, 1]

  inv_mass = jnp.where(masses > 0, 1.0 / masses, 0.0)

  # Reference bond vectors r_ij and their squared norms
  # TODO using metric would be nice, but you need both disps and dists
  r_ij = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(pos[i], pos[j])
  d_ij2 = jnp.sum(r_ij * r_ij, axis=1)
  dist2 = dist * dist

  # OpenMM uses reducedMass = 0.5/(invMass_i + invMass_j)
  reduced_mass = 0.5 / (inv_mass[i] + inv_mass[j] + 1e-30)

  lower = 1.0 - 2.0 * tolerance + tolerance * tolerance
  upper = 1.0 + 2.0 * tolerance + tolerance * tolerance

  def cond_fn(state):
    it, _, converged = state
    return jnp.logical_and(it < max_iters, jnp.logical_not(converged))

  def body_fn(state):
    it, pos_p, _ = state
    rp_ij = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(pos_p[i], pos_p[j])
    rp2 = jnp.sum(rp_ij * rp_ij, axis=1)
    rrpr = jnp.sum(rp_ij * r_ij, axis=1)

    diff = dist2 - rp2
    delta = reduced_mass * diff / (rrpr + 1e-30)

    dr = r_ij * delta[:, None]
    dRi = dr * inv_mass[i, None]
    dRj = -dr * inv_mass[j, None]

    atom_ids = jnp.concatenate([i, j], axis=0)
    dR_all = jnp.concatenate([dRi, dRj], axis=0)
    dR_sum = jax.ops.segment_sum(dR_all, atom_ids, num_segments=pos.shape[0])

    pos_p = shift_fn(pos_p, dR_sum, **box_kwargs)
    converged = jnp.all(jnp.logical_and(rp2 >= lower * dist2, rp2 <= upper * dist2))
    return (it + 1, pos_p, converged)

  _, pos_out, _ = jax.lax.while_loop(cond_fn, body_fn, (0, pos, False))
  return pos_out


def ccma_apply_velocities(
  pos: Array,
  vel: Array,
  constraints: CCMAData,
  masses: Array,
  displacement_fn,
  *,
  box: Optional[Array] = None,
  use_periodic_general: bool = False,
  tolerance: float = 1e-6,
  max_iters: int = 150,
) -> Array:
  """Apply velocity constraints (RATTLE-style projection) for distance constraints."""
  if constraints is None or constraints.idx.size == 0:
    return vel

  box_kwargs = {'box': box} if use_periodic_general else {}
  idx = constraints.idx
  i = idx[:, 0]
  j = idx[:, 1]

  inv_mass = jnp.where(masses > 0, 1.0 / masses, 0.0)

  r_ij = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(pos[i], pos[j])
  d_ij2 = jnp.sum(r_ij * r_ij, axis=1)
  reduced_mass = 0.5 / (inv_mass[i] + inv_mass[j] + 1e-30)

  def cond_fn(state):
    it, _, converged = state
    return jnp.logical_and(it < max_iters, jnp.logical_not(converged))

  def body_fn(state):
    it, vel_p, _ = state
    rp_ij = vel_p[i] - vel_p[j]
    rrpr = jnp.sum(rp_ij * r_ij, axis=1)
    delta = -2.0 * reduced_mass * rrpr / (d_ij2 + 1e-30)

    converged = jnp.all(jnp.abs(delta) <= tolerance)

    dr = r_ij * delta[:, None]
    dVi = dr * inv_mass[i, None]
    dVj = -dr * inv_mass[j, None]
    atom_ids = jnp.concatenate([i, j], axis=0)
    dV_all = jnp.concatenate([dVi, dVj], axis=0)
    dV_sum = jax.ops.segment_sum(dV_all, atom_ids, num_segments=vel.shape[0])
    vel_p = vel_p + dV_sum
    return (it + 1, vel_p, converged)

  _, vel_out, _ = jax.lax.while_loop(cond_fn, body_fn, (0, vel, False))
  return vel_out


def settle_apply_positions(
  pos: Array,
  pos_p: Array,
  settle: SettleData,
  masses: Array,
  displacement_fn,
  shift_fn,
  *,
  box: Optional[Array] = None,
  use_periodic_general: bool = False,
) -> Array:
  """Apply SETTLE position constraints for identified 3-atom clusters.

  This is a direct port of OpenMM's Reference SETTLE implementation. It operates
  on displacements computed by displacement_fn, so it is compatible with
  wrapped periodic positions (including fractional coordinates).
  """
  if settle is None or settle.atom1.size == 0:
    return pos_p

  box_kwargs = {'box': box} if use_periodic_general else {}

  a0 = settle.atom1
  a1 = settle.atom2
  a2 = settle.atom3

  apos0 = pos[a0]
  apos1 = pos[a1]
  apos2 = pos[a2]

  # Predicted displacements from original positions
  xp0 = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(pos_p[a0], apos0)
  xp1 = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(pos_p[a1], apos1)
  xp2 = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(pos_p[a2], apos2)

  # Bond vectors in the original configuration
  b0 = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(apos1, apos0)
  c0 = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(apos2, apos0)

  m0 = masses[a0]
  m1 = masses[a1]
  m2 = masses[a2]
  inv_total = 1.0 / (m0 + m1 + m2)

  # Center of mass of the displacement vectors
  xcom = (xp0 * m0[:, None] + (b0 + xp1) * m1[:, None] + (c0 + xp2) * m2[:, None]) * inv_total[:, None]

  a1v = xp0 - xcom
  b1v = b0 + xp1 - xcom
  c1v = c0 + xp2 - xcom

  # Local orthonormal basis (trns mat) built from original b0/c0
  aks_zd = jnp.cross(b0, c0)
  axlng = jnp.sqrt(jnp.sum(b0 * b0, axis=1))
  azlng = jnp.sqrt(jnp.sum(aks_zd * aks_zd, axis=1))
  axlng = jnp.where(axlng > 0, axlng, 1.0)
  azlng = jnp.where(azlng > 0, azlng, 1.0)

  trns11 = b0[:, 0] / axlng
  trns21 = b0[:, 1] / axlng
  trns31 = b0[:, 2] / axlng

  trns13 = aks_zd[:, 0] / azlng
  trns23 = aks_zd[:, 1] / azlng
  trns33 = aks_zd[:, 2] / azlng

  trns12 = trns23 * trns31 - trns33 * trns21
  trns22 = trns33 * trns11 - trns13 * trns31
  trns32 = trns13 * trns21 - trns23 * trns11

  # Transform selected vectors into the local frame
  def _dot_trns(v):
    x = trns11 * v[:, 0] + trns21 * v[:, 1] + trns31 * v[:, 2]
    y = trns12 * v[:, 0] + trns22 * v[:, 1] + trns32 * v[:, 2]
    z = trns13 * v[:, 0] + trns23 * v[:, 1] + trns33 * v[:, 2]
    return x, y, z

  xb0d, yb0d, _ = _dot_trns(b0)
  xc0d, yc0d, _ = _dot_trns(c0)
  _, _, za1d = _dot_trns(a1v)
  xb1d, yb1d, zb1d = _dot_trns(b1v)
  xc1d, yc1d, zc1d = _dot_trns(c1v)

  rc = 0.5 * settle.distance2
  rb = safe_sqrt(settle.distance1 * settle.distance1 - rc * rc)
  ra = rb * (m1 + m2) * inv_total
  rb = rb - ra

  sinphi = za1d / ra
  sinphi = jnp.clip(sinphi, -1.0, 1.0)
  cosphi = safe_sqrt(1.0 - sinphi * sinphi)
  sinpsi = (zb1d - zc1d) / (2.0 * rc * cosphi)
  sinpsi = jnp.clip(sinpsi, -1.0, 1.0)
  cospsi = safe_sqrt(1.0 - sinpsi * sinpsi)

  ya2d = ra * cosphi
  xb2d = -rc * cospsi
  yb2d = -rb * cosphi - rc * sinpsi * sinphi
  yc2d = -rb * cosphi + rc * sinpsi * sinphi

  xb2d2 = xb2d * xb2d
  hh2 = 4.0 * xb2d2 + (yb2d - yc2d) * (yb2d - yc2d) + (zb1d - zc1d) * (zb1d - zc1d)
  deltx = 2.0 * xb2d + safe_sqrt(4.0 * xb2d2 - hh2 + settle.distance2 * settle.distance2)
  xb2d = xb2d - 0.5 * deltx

  alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d
  beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d
  gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d

  al2be2 = alpha * alpha + beta * beta
  sqrt_term = safe_sqrt(jnp.maximum(al2be2 - gamma * gamma, 0.0))
  sintheta = (alpha * gamma - beta * sqrt_term) / (al2be2 + 1e-30)
  sintheta = jnp.clip(sintheta, -1.0, 1.0)
  costheta = safe_sqrt(1.0 - sintheta * sintheta)

  xa3d = -ya2d * sintheta
  ya3d = ya2d * costheta
  za3d = za1d

  xb3d = xb2d * costheta - yb2d * sintheta
  yb3d = xb2d * sintheta + yb2d * costheta
  zb3d = zb1d

  xc3d = -xb2d * costheta - yc2d * sintheta
  yc3d = -xb2d * sintheta + yc2d * costheta
  zc3d = zc1d

  # Transform back to the original frame
  xa3 = trns11 * xa3d + trns12 * ya3d + trns13 * za3d
  ya3 = trns21 * xa3d + trns22 * ya3d + trns23 * za3d
  za3 = trns31 * xa3d + trns32 * ya3d + trns33 * za3d

  xb3 = trns11 * xb3d + trns12 * yb3d + trns13 * zb3d
  yb3 = trns21 * xb3d + trns22 * yb3d + trns23 * zb3d
  zb3 = trns31 * xb3d + trns32 * yb3d + trns33 * zb3d

  xc3 = trns11 * xc3d + trns12 * yc3d + trns13 * zc3d
  yc3 = trns21 * xc3d + trns22 * yc3d + trns23 * zc3d
  zc3 = trns31 * xc3d + trns32 * yc3d + trns33 * zc3d

  xp0_new = jnp.stack([xcom[:, 0] + xa3, xcom[:, 1] + ya3, xcom[:, 2] + za3], axis=1)
  xp1_new = jnp.stack([xcom[:, 0] + xb3 - b0[:, 0], xcom[:, 1] + yb3 - b0[:, 1], xcom[:, 2] + zb3 - b0[:, 2]], axis=1)
  xp2_new = jnp.stack([xcom[:, 0] + xc3 - c0[:, 0], xcom[:, 1] + yc3 - c0[:, 1], xcom[:, 2] + zc3 - c0[:, 2]], axis=1)

  # Apply displacements to original positions and scatter back into pos_p
  new0 = shift_fn(apos0, xp0_new, **box_kwargs)
  new1 = shift_fn(apos1, xp1_new, **box_kwargs)
  new2 = shift_fn(apos2, xp2_new, **box_kwargs)

  pos_p = pos_p.at[a0].set(new0)
  pos_p = pos_p.at[a1].set(new1)
  pos_p = pos_p.at[a2].set(new2)
  return pos_p


def settle_apply_velocities(
  pos: Array,
  vel: Array,
  settle: SettleData,
  masses: Array,
  displacement_fn,
  *,
  box: Optional[Array] = None,
  use_periodic_general: bool = False,
  tolerance: float = 1e-6,
) -> Array:
  """Apply velocity constraints for SETTLE clusters."""
  if settle is None or settle.atom1.size == 0:
    return vel

  box_kwargs = {'box': box} if use_periodic_general else {}
  a0 = settle.atom1
  a1 = settle.atom2
  a2 = settle.atom3

  apos0 = pos[a0]
  apos1 = pos[a1]
  apos2 = pos[a2]
  v0 = vel[a0]
  v1 = vel[a1]
  v2 = vel[a2]

  mA = masses[a0]
  mB = masses[a1]
  mC = masses[a2]

  inv_m0 = jnp.where(mA > 0, 1.0 / mA, 0.0)
  inv_m1 = jnp.where(mB > 0, 1.0 / mB, 0.0)
  inv_m2 = jnp.where(mC > 0, 1.0 / mC, 0.0)

  # OpenMM computes velocity constraints using the bond directions in the
  # current constrained geometry
  eAB = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(apos1, apos0)
  eBC = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(apos2, apos1)
  eCA = jax.vmap(lambda a, b: displacement_fn(a, b, **box_kwargs))(apos0, apos2)
  eAB = normalize(eAB)
  eBC = normalize(eBC)
  eCA = normalize(eCA)

  vAB = jnp.sum((v1 - v0) * eAB, axis=1)
  vBC = jnp.sum((v2 - v1) * eBC, axis=1)
  vCA = jnp.sum((v0 - v2) * eCA, axis=1)

  cA = -jnp.sum(eAB * eCA, axis=1)
  cB = -jnp.sum(eAB * eBC, axis=1)
  cC = -jnp.sum(eBC * eCA, axis=1)
  s2A = 1.0 - cA * cA
  s2B = 1.0 - cB * cB
  s2C = 1.0 - cC * cC

  # Port of ReferenceSETTLEAlgorithm::applyToVelocities
  mABCinv = 1.0 / (mA * mB * mC)
  denom = (
    (((s2A * mB + s2B * mA) * mC +
      (s2A * mB * mB + 2.0 * (cA * cB * cC + 1.0) * mA * mB + s2B * mA * mA)) * mC +
     s2C * mA * mB * (mA + mB)) * mABCinv
  )

  tab = (
    ((cB * cC * mA - cA * mB - cA * mC) * vCA +
     (cA * cC * mB - cB * mC - cB * mA) * vBC +
     (s2C * mA * mA * mB * mB * mABCinv + (mA + mB + mC)) * vAB) / denom
  )
  tbc = (
    ((cA * cB * mC - cC * mB - cC * mA) * vCA +
     (s2A * mB * mB * mC * mC * mABCinv + (mA + mB + mC)) * vBC +
     (cA * cC * mB - cB * mA - cB * mC) * vAB) / denom
  )
  tca = (
    ((s2B * mA * mA * mC * mC * mABCinv + (mA + mB + mC)) * vCA +
     (cA * cB * mC - cC * mB - cC * mA) * vBC +
     (cB * cC * mA - cA * mB - cA * mC) * vAB) / denom
  )

  v0_new = v0 + (eAB * tab[:, None] - eCA * tca[:, None]) * inv_m0[:, None]
  v1_new = v1 + (eBC * tbc[:, None] - eAB * tab[:, None]) * inv_m1[:, None]
  v2_new = v2 + (eCA * tca[:, None] - eBC * tbc[:, None]) * inv_m2[:, None]

  vel = vel.at[a0].set(v0_new)
  vel = vel.at[a1].set(v1_new)
  vel = vel.at[a2].set(v2_new)
  return vel
