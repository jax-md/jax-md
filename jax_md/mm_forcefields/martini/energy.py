"""Energy functions for the Martini force field."""

from __future__ import annotations

import math
from typing import Dict

import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

from jax_md import space
from jax_md.quantity import EnergyFn
from jax_md.util import Array, normalize, safe_arccos
from jax_md.mm_forcefields.nonbonded.electrostatics import (
  NeighborList,
  RFCoulomb,
)

from jax_md.mm_forcefields.martini.topology import MartiniTopology


def _batch_dihedral(
  ri: jnp.ndarray,
  rj: jnp.ndarray,
  rk: jnp.ndarray,
  rl: jnp.ndarray,
  displacement_fn: space.DisplacementFn,
  **kwargs,
) -> jnp.ndarray:
  """Compute dihedral angles for batches of 4-atom groups using the atan2 method.
  Based on Praxeolitic formula.

  Args:
    ri: Cartesian coordinates of atom i, shape (N, 3).
    rj: Cartesian coordinates of atom j, shape (N, 3).
    rk: Cartesian coordinates of atom k, shape (N, 3).
    rl: Cartesian coordinates of atom l, shape (N, 3).
    displacement_fn: JAX MD displacement function that correctly handles
      periodic boundary conditions (e.g. minimum image convention).
      Signature: ``displacement_fn(a, b) -> a - b`` respecting PBC.
    **kwargs: Additional keyword arguments (unused, accepted for API
      compatibility).

  Returns:
    Dihedral angles in radians, shape (N,), in the range (-π, π].
    Positive values correspond to clockwise rotation when looking along j→k.

  Notes:
  Degenerate case: if atoms i, j, k (or j, k, l) are collinear, b1 has zero
  length after projection and the angle is undefined. This will produce NaN.
  Such configurations are geometrically invalid for a bonded dihedral.
  """

  b0 = -1.0 * vmap(displacement_fn)(rj, ri)
  b1 = vmap(displacement_fn)(rk, rj)
  b2 = vmap(displacement_fn)(rl, rk)

  b1_hat = normalize(b1, axis=-1)

  v = b0 - jnp.sum(b0 * b1_hat, axis=-1, keepdims=True) * b1_hat

  w = b2 - jnp.sum(b2 * b1_hat, axis=-1, keepdims=True) * b1_hat

  x = jnp.sum(v * w, axis=-1)
  y = jnp.sum(jnp.cross(b1_hat, v) * w, axis=-1)

  return jnp.arctan2(y, x)


def _cos_theta(
  ri: jnp.ndarray,
  rj: jnp.ndarray,
  rk: jnp.ndarray,
  displacement_fn: space.DisplacementFn,
):
  """Compute the cosine of the angle formed by vectors ri→rj and rk→rj.

  Args:
    ri: Cartesian coordinates of atom i, shape (N, 3).
    rj: Cartesian coordinates of the central atom j, shape (N, 3).
    rk: Cartesian coordinates of atom k, shape (N, 3).
    displacement_fn: JAX MD displacement function that correctly handles
      periodic boundary conditions. Signature:
      ``displacement_fn(a, b) -> a - b`` respecting PBC.

  Returns:
    Cosine of the angle at vertex rj, shape (N,), clipped to [-1, 1].
  """
  r_ij = vmap(displacement_fn)(ri, rj)
  r_kj = vmap(displacement_fn)(rk, rj)
  rij_norm = normalize(r_ij)
  rkj_norm = normalize(r_kj)
  cos_th = jnp.sum(rij_norm * rkj_norm, axis=-1)
  return jnp.clip(cos_th, -1.0, 1.0)


def build_energies(topology: MartiniTopology):
  bond_energy = HarmonicBondEnergy(
    indices=topology.bond_indices,
    rs=topology.bond_rs,
    ks=topology.bond_ks,
  )
  harm_angle_energy = HarmonicAngleEnergy(
    indices=topology.harm_angle_indices,
    theta0s=topology.harm_angle_theta0s,
    ks=topology.harm_angle_ks,
  )
  g96_angle_energy = G96AngleEnergy(
    indices=topology.g96_angle_indices,
    cos_theta0s=topology.g96_angle_cos_theta0s,
    ks=topology.g96_angle_ks,
  )
  rest_angle_energy = RestrictedAngleEnergy(
    indices=topology.rest_angle_indices,
    cos_theta0s=topology.rest_angle_cos_theta0s,
    ks=topology.rest_angle_ks,
  )
  per_dihedral_energy = PeriodicDihedralEnergy(
    indices=topology.per_dihedral_indices,
    ns=topology.per_dihedral_ns,
    ks=topology.per_dihedral_ks,
    phi0s=topology.per_dihedral_phi0s,
  )
  harm_dihedral_energy = HarmonicDihedralEnergy(
    indices=topology.harm_dihedral_indices,
    ks=topology.harm_dihedral_ks,
    phi0s=topology.harm_dihedral_phi0s,
  )
  rb_dihedral_energy = RBFourierEnergy(
    indices=topology.rb_dihedral_indices,
    cs=topology.rb_dihedral_cs,
  )
  cbt_energy = CBTEnergy(
    indices=topology.cbt_indices,
    a_coeffs=topology.cbt_a_coeffs,
    ks=topology.cbt_ks,
  )
  cmap_energy = CMapEnergy(
    indices=topology.cmap_indices,
    map_ids=topology.cmap_map_ids,
    grids=topology.cmap_grids,
    coeffs=topology.cmap_coeffs,
  )
  lj_energy = LJEnergy(
    n_atoms=topology.n_atoms,
    atom_type_indices=topology.atom_type_indices,
    charges=topology.charges,
    C6_table=topology.C6_table,
    C12_table=topology.C12_table,
    exclusion_pairs=topology.exclusion_pairs,
    exception_pairs=topology.exception_pairs,
    exception_q=topology.exception_q,
    exception_C6=topology.exception_C6,
    exception_C12=topology.exception_C12,
    epsilon_r=topology.epsilon_r,
    r_cut=topology.r_cut,
    excl_mask=topology.excl_mask,
  )
  coulomb = RFCoulomb(
    topology.r_cut * 10.0,  # Convert to Angstroms
    topology.epsilon_r,
    topology.exception_pairs,
    topology.exclusion_pairs,
    topology.exception_q,
    topology.exception_C6,
    topology.exception_C12,
  )

  bonded_terms: Dict[str, nn.Module] = {}
  if bond_energy.indices.shape[0]:
    bonded_terms['Bond'] = bond_energy
  if harm_angle_energy.indices.shape[0]:
    bonded_terms['Angle'] = harm_angle_energy
  if g96_angle_energy.indices.shape[0]:
    bonded_terms['G96Angle'] = g96_angle_energy
  if rest_angle_energy.indices.shape[0]:
    bonded_terms['Restr. Angles'] = rest_angle_energy
  if per_dihedral_energy.indices.shape[0]:
    bonded_terms['Periodic Dih.'] = per_dihedral_energy
  if harm_dihedral_energy.indices.shape[0]:
    bonded_terms['Improper Dih.'] = harm_dihedral_energy
  if rb_dihedral_energy.indices.shape[0]:
    bonded_terms['RB-Fourier'] = rb_dihedral_energy
  if cbt_energy.indices.shape[0]:
    bonded_terms['CBT Dih.'] = cbt_energy
  if cmap_energy.indices.shape[0]:
    bonded_terms['CMAP Dih.'] = cmap_energy
  return bonded_terms, lj_energy, coulomb


def energy_fn(
  topology: MartiniTopology,
  displacement_fn: space.DisplacementFn,
  shift_fn: space.ShiftFn,
  include_vsites: bool = True,
) -> EnergyFn:
  """Return a JAX-differentiable energy function for a Martini system.

  Args:
    topology: Pre-parsed topology produced by
      :func:`jax_md.mm_forcefields.martini.topology.create_topology`.
    displacement_fn: JAX-MD displacement function (e.g. from
      ``space.periodic_general``).
    include_vsites: If ``True``, virtual-site positions are recomputed
      from real-atom positions before evaluating energy terms.

  Returns:
    Callable of the form ``f(positions, box, perturbation, neighbor)``
    returning a scalar energy in kJ/mol.
  """
  bonded_terms, lj_energy, coulomb = build_energies(topology)

  apply_vsites = topology.apply_vsites
  excl_mask = topology.excl_mask
  charges = topology.charges

  def _energy(
    positions: jnp.ndarray,
    box: jnp.ndarray,
    neighbor: NeighborList,
    perturbation=None,
  ) -> jnp.ndarray:
    total = jnp.zeros(())

    def disp_fn(a, b):
      return displacement_fn(a, b, perturbation=perturbation)

    pos = (
      apply_vsites(positions, disp_fn, shift_fn)
      if include_vsites
      else positions
    )

    total = total + lj_energy(pos, neighbor, disp_fn)
    for fn_name in bonded_terms.keys():
      total = total + bonded_terms[fn_name](pos, disp_fn)

    # Coulomb setup to use angstroms and return in kcal/mol
    total = (
      total
      + coulomb.energy(
        pos * 10.0, charges, box * 10.0, excl_mask, None, neighbor, None
      )
      * 4.184
    )
    return total

  return _energy


class HarmonicBondEnergy(nn.Module):
  """Harmonic bond energy: E = ½ k (r - r₀)²."""

  indices: jnp.ndarray
  rs: jnp.ndarray
  ks: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute harmonic bond energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total harmonic bond energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    ri, rj = pos[self.indices[:, 0]], pos[self.indices[:, 1]]
    disp = vmap(displacement_fn)(ri, rj)
    dr = space.distance(disp)
    return jnp.sum(0.5 * self.ks * (dr - self.rs) ** 2)


class HarmonicAngleEnergy(nn.Module):
  """Harmonic angle energy: E = ½ k (θ − θ₀)²."""

  indices: jnp.ndarray
  theta0s: jnp.ndarray
  ks: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute harmonic angle energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total harmonic angle energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    ri = pos[self.indices[:, 0]]
    rj = pos[self.indices[:, 1]]
    rk = pos[self.indices[:, 2]]
    cos_th = _cos_theta(ri, rj, rk, displacement_fn, **kwargs)
    theta = safe_arccos(cos_th)
    return jnp.sum(0.5 * self.ks * (theta - self.theta0s) ** 2)


class G96AngleEnergy(nn.Module):
  """G96 angle energy: E = ½ k (cos θ - cos θ₀)²."""

  indices: jnp.ndarray
  cos_theta0s: jnp.ndarray
  ks: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute G96 angle energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total G96 angle energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    ri = pos[self.indices[:, 0]]
    rj = pos[self.indices[:, 1]]
    rk = pos[self.indices[:, 2]]
    cos_th = _cos_theta(ri, rj, rk, displacement_fn, **kwargs)
    return jnp.sum(0.5 * self.ks * (cos_th - self.cos_theta0s) ** 2)


class RestrictedAngleEnergy(nn.Module):
  """Restricted bending angle energy (func type 10)."""

  indices: jnp.ndarray
  cos_theta0s: jnp.ndarray
  ks: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute restricted bending angle energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total restricted bending angle energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    ri = pos[self.indices[:, 0]]
    rj = pos[self.indices[:, 1]]
    rk = pos[self.indices[:, 2]]
    cos_th = _cos_theta(ri, rj, rk, displacement_fn, **kwargs)
    cos_th = jnp.clip(cos_th, -1.0 + 1e-8, 1.0 - 1e-8)
    sin2_th = 1.0 - cos_th**2
    return jnp.sum(0.5 * self.ks * (cos_th - self.cos_theta0s) ** 2 / sin2_th)


class PeriodicDihedralEnergy(nn.Module):
  """Proper / periodic dihedral energy: E = k [1 + cos(n φ − φ₀)]."""

  indices: jnp.ndarray
  ns: jnp.ndarray
  ks: jnp.ndarray
  phi0s: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute periodic dihedral energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total periodic dihedral energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    ri = pos[self.indices[:, 0]]
    rj = pos[self.indices[:, 1]]
    rk = pos[self.indices[:, 2]]
    rl = pos[self.indices[:, 3]]
    phi = _batch_dihedral(ri, rj, rk, rl, displacement_fn, **kwargs)
    return jnp.sum(self.ks * (1.0 + jnp.cos(self.ns * phi - self.phi0s)))


class HarmonicDihedralEnergy(nn.Module):
  """Harmonic improper dihedral energy: E = ½ k (φ - φ₀)² (periodic wrap)."""

  indices: jnp.ndarray
  ks: jnp.ndarray
  phi0s: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute harmonic improper dihedral energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total harmonic improper dihedral energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    ri = pos[self.indices[:, 0]]
    rj = pos[self.indices[:, 1]]
    rk = pos[self.indices[:, 2]]
    rl = pos[self.indices[:, 3]]
    phi = _batch_dihedral(ri, rj, rk, rl, displacement_fn, **kwargs)
    delta = phi - self.phi0s
    delta = delta - 2 * math.pi * jnp.round(delta / (2 * math.pi))
    return jnp.sum(0.5 * self.ks * delta**2)


class RBFourierEnergy(nn.Module):
  """Ryckaert-Bellemans torsion energy."""

  indices: jnp.ndarray
  cs: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute Ryckaert-Bellemans torsion energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total RB torsion energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    ri = pos[self.indices[:, 0]]
    rj = pos[self.indices[:, 1]]
    rk = pos[self.indices[:, 2]]
    rl = pos[self.indices[:, 3]]
    theta = _batch_dihedral(ri, rj, rk, rl, displacement_fn, **kwargs)
    phi = theta - math.pi
    cos_phi = jnp.cos(phi)
    cs = self.cs
    energy = (
      cs[:, 0]
      + cs[:, 1] * cos_phi
      + cs[:, 2] * cos_phi**2
      + cs[:, 3] * cos_phi**3
      + cs[:, 4] * cos_phi**4
      + cs[:, 5] * cos_phi**5
    )
    return jnp.sum(energy)


class CBTEnergy(nn.Module):
  """Combined bending-torsion (CBT) dihedral energy."""

  indices: jnp.ndarray
  a_coeffs: jnp.ndarray
  ks: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute combined bending-torsion dihedral energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total CBT dihedral energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    ri = pos[self.indices[:, 0]]
    rj = pos[self.indices[:, 1]]
    rk = pos[self.indices[:, 2]]
    rl = pos[self.indices[:, 3]]

    cos_th0 = _cos_theta(ri, rj, rk, displacement_fn, **kwargs)
    sin_th0 = jnp.sqrt(jnp.clip(1.0 - cos_th0**2, 0.0, None))
    cos_th1 = _cos_theta(rj, rk, rl, displacement_fn, **kwargs)
    sin_th1 = jnp.sqrt(jnp.clip(1.0 - cos_th1**2, 0.0, None))

    phi = _batch_dihedral(ri, rj, rk, rl, displacement_fn, **kwargs)
    cos_phi = jnp.cos(phi)
    a = self.a_coeffs
    poly = (
      a[:, 0]
      + a[:, 1] * cos_phi
      + a[:, 2] * cos_phi**2
      + a[:, 3] * cos_phi**3
      + a[:, 4] * cos_phi**4
    )
    return jnp.sum(self.ks * sin_th0**3 * sin_th1**3 * poly)


class CMapEnergy(nn.Module):
  """CMAP bicubic-spline correction map energy."""

  indices: jnp.ndarray
  map_ids: jnp.ndarray
  grids: jnp.ndarray
  coeffs: jnp.ndarray

  def __call__(
    self, pos: jnp.ndarray, displacement_fn: space.DisplacementFn, **kwargs
  ) -> jnp.ndarray:
    """Compute CMAP correction map energy.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total CMAP energy in kJ/mol.
    """
    if self.indices.shape[0] == 0:
      return jnp.zeros(())
    idx = self.indices
    ri = pos[idx[:, 0]]
    rj = pos[idx[:, 1]]
    rk = pos[idx[:, 2]]
    rl = pos[idx[:, 3]]
    rm = pos[idx[:, 4]]
    phi1 = _batch_dihedral(ri, rj, rk, rl, displacement_fn, **kwargs)
    phi2 = _batch_dihedral(rj, rk, rl, rm, displacement_fn, **kwargs)
    energies = vmap(self._bicubic_interpolate)(self.map_ids, phi1, phi2)
    return jnp.sum(energies)

  def _bicubic_interpolate(
    self, map_id: int, phi1: float, phi2: float
  ) -> Array:
    """Bicubic interpolation on a periodic (S x S) CMAP grid.

    Args:
      map_id: Index into the CMAP coefficient array.
      phi1: First dihedral angle in radians, in [-π, π].
      phi2: Second dihedral angle in radians, in [-π, π].

    Returns:
      Interpolated CMAP energy for the given dihedral pair.
    """
    S = self.coeffs.shape[1]
    x = jnp.mod(phi1 + 2 * jnp.pi, 2 * jnp.pi)
    y = jnp.mod(phi2 + 2 * jnp.pi, 2 * jnp.pi)
    delta = 2 * jnp.pi / S
    s = jnp.floor(jnp.minimum(x / delta, S - 1)).astype(jnp.int32)
    t = jnp.floor(jnp.minimum(y / delta, S - 1)).astype(jnp.int32)
    c = self.coeffs[map_id, s, t]
    da = x / delta - s
    db = y / delta - t

    energy = jnp.zeros(())
    for i in range(3, -1, -1):
      energy = (
        da * energy
        + ((c[i * 4 + 3] * db + c[i * 4 + 2]) * db + c[i * 4 + 1]) * db
        + c[i * 4 + 0]
      )
    return energy


class LJEnergy(nn.Module):
  """Lennard-Jones 12-6 energy with exclusions and pair exceptions."""

  n_atoms: int
  atom_type_indices: jnp.ndarray
  charges: jnp.ndarray
  C6_table: jnp.ndarray
  C12_table: jnp.ndarray
  # exclusion pairs as a boolean mask or pair list
  exclusion_pairs: jnp.ndarray
  # exception pairs with explicit q, C6, C12
  exception_pairs: jnp.ndarray
  exception_q: jnp.ndarray
  exception_C6: jnp.ndarray
  exception_C12: jnp.ndarray
  epsilon_r: float
  r_cut: float
  excl_mask: jnp.ndarray

  def __call__(
    self,
    pos: jnp.ndarray,
    nlist: NeighborList,
    displacement_fn: space.DisplacementFn,
    **kwargs,
  ) -> jnp.ndarray:
    """Compute Lennard-Jones energy with exclusions and exceptions.

    Args:
      pos: Atomic positions, shape (n_atoms, 3).
      nlist: Neighbor list.
      displacement_fn: JAX-MD displacement function.

    Returns:
      Total LJ energy in kJ/mol.
    """
    r_cut = self.r_cut
    r_cut2 = r_cut * r_cut
    r_cut6 = r_cut2 * r_cut2 * r_cut2

    n_atoms = pos.shape[0]
    idx_j = nlist.idx
    idx_i = jnp.broadcast_to(
      jnp.arange(n_atoms, dtype=jnp.int32)[:, None], idx_j.shape
    )

    valid = (idx_j >= 0) & (idx_j < n_atoms)
    idx_j_safe = jnp.where(valid, idx_j, 0)
    idx_i_safe = jnp.where(valid, idx_i, 0)

    ri = pos[idx_i_safe]
    rj = pos[idx_j_safe]

    batched_disp = vmap(vmap(displacement_fn, in_axes=(0, 0)), in_axes=(0, 0))
    dr = batched_disp(ri, rj)
    r2 = jnp.sum(dr * dr, axis=-1)
    r2_safe = jnp.where(r2 > 0.0, r2, 1.0)
    r6 = r2_safe * r2_safe * r2_safe
    r12 = r6 * r6

    atom_types = self.atom_type_indices
    c12 = self.C12_table[atom_types[idx_i_safe], atom_types[idx_j_safe]]
    c6 = self.C6_table[atom_types[idx_i_safe], atom_types[idx_j_safe]]

    lj = c12 / r12 - c6 / r6
    lj_corr = c12 / (r_cut6 * r_cut6) - c6 / r_cut6

    same = idx_i_safe == idx_j_safe
    excluded = self.excl_mask[idx_i_safe, idx_j_safe]
    include = valid & (~same) & (~excluded) & (r2 < r_cut2)

    total_lj = 0.5 * jnp.sum(jnp.where(include, lj - lj_corr, 0.0))

    exc_lj = jnp.zeros(())
    if self.exception_pairs.shape[0] > 0:
      ep = self.exception_pairs
      ri_exc = pos[ep[:, 0]]
      rj_exc = pos[ep[:, 1]]
      dr_exc = vmap(displacement_fn)(ri_exc, rj_exc)
      r_exc2 = jnp.sum(dr_exc * dr_exc, axis=-1)
      r_exc2_safe = jnp.where(r_exc2 == 0.0, 1.0, r_exc2)

      c6_exc = self.exception_C6
      c12_exc = self.exception_C12
      valid_lj = (c12_exc != 0) & (c6_exc != 0)

      lj_exc = c12_exc / (r_exc2_safe**6) - c6_exc / (r_exc2_safe**3)
      lj_exc_corr = c12_exc / (r_cut6 * r_cut6) - c6_exc / r_cut6
      exc_lj = jnp.sum(
        jnp.where(valid_lj & (r_exc2 < r_cut2), lj_exc - lj_exc_corr, 0.0)
      )

    return total_lj + exc_lj
