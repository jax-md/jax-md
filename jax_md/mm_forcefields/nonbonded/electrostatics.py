"""Electrostatic interactions for molecular mechanics forcefields.

This module provides coulomb energy functions that support:
- Exclusion masks (e.g., for bonded/angle pairs)
- 1-4 interaction scaling
- Units in kcal/mol (common for MM forcefields)
"""

import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import erfc
from jax_md import space
from jax_md.util import Array
from jax_md.partition import NeighborList


# Conversion factor: e²/Å to kcal/mol
COULOMB_CONSTANT = 332.06371  # kcal·Å/(mol·e²)


class CoulombHandler:
  """Base class for coulomb energy handlers."""

  def energy(
    self,
    positions: Array,
    charges: Array,
    box: Array,
    exclusion_mask: Array,
    pair_14_mask: Array,
    nlist: NeighborList,
    scale_14: float = 0.5,
  ) -> Array:
    """Compute coulomb energy.

    Args:
        positions: Atomic positions, shape (n_atoms, 3).
        charges: Partial charges, shape (n_atoms,).
        box: Simulation box (scalar or array).
        exclusion_mask: Boolean mask for excluded pairs, shape (n_atoms, n_atoms).
        pair_14_mask: Boolean mask for 1-4 pairs, shape (n_atoms, n_atoms).
        nlist: Neighbor list.
        scale_14: Scaling factor for 1-4 interactions.

    Returns:
        Coulomb energy in kcal/mol.
    """
    raise NotImplementedError


class CutoffCoulomb(CoulombHandler):
  """Simple cutoff coulomb with optional complementary error function."""

  def __init__(
    self, r_cut: float = 12.0, use_erfc: bool = False, alpha: float = 0.3
  ):
    """Initialize cutoff coulomb handler.

    Args:
        r_cut: Cutoff distance (Å).
        use_erfc: Whether to use erfc damping.
        alpha: Damping parameter for erfc (Å⁻¹).
    """
    self.r_cut = r_cut
    self.use_erfc = use_erfc
    self.alpha = alpha

  def pair_energy(self, qi: Array, qj: Array, r: Array) -> Array:
    """Compute pairwise coulomb energy."""
    energy = COULOMB_CONSTANT * (qi * qj / r)
    if self.use_erfc:
      energy *= erfc(self.alpha * r)
    return energy

  def energy(
    self,
    positions: Array,
    charges: Array,
    box: Array,
    exclusion_mask: Array,
    pair_14_mask: Array,
    nlist: NeighborList,
    scale_14: float = 0.5,
  ) -> Array:
    """Compute cutoff coulomb energy."""
    n_atoms = positions.shape[0]
    max_neighbors = nlist.idx.shape[1]

    displacement_fn, _ = space.periodic(box)

    # Prepare neighbor indices
    idx_i = jnp.repeat(jnp.arange(n_atoms)[:, None], max_neighbors, axis=1)
    idx_j = nlist.idx

    # Valid neighbor mask
    valid = (idx_j >= 0) & (idx_j < n_atoms)
    idx_j_safe = jnp.where(valid, idx_j, 0)
    idx_i_safe = jnp.where(valid, idx_i, 0)

    # Get charges and positions
    qi = charges[idx_i_safe]
    qj = charges[idx_j_safe]
    ri = positions[idx_i_safe]
    rj = positions[idx_j_safe]

    # Compute distances
    batched_disp = vmap(vmap(displacement_fn, in_axes=(0, 0)), in_axes=(0, 0))
    disp = batched_disp(ri, rj)
    r_sq = jnp.sum(disp**2, axis=-1)
    r_sq_safe = jnp.maximum(r_sq, 1e-12)
    r = jnp.sqrt(r_sq_safe)

    # Check exclusions and scaling
    same = idx_i_safe == idx_j_safe
    excluded = vmap(lambda i, j: exclusion_mask[i, j])(idx_i_safe, idx_j_safe)
    is_14 = vmap(lambda i, j: pair_14_mask[i, j])(idx_i_safe, idx_j_safe)

    # Include mask
    include = valid & (~same) & (~excluded) & (r < self.r_cut)

    # Apply 1-4 scaling
    scale = jnp.where(is_14, scale_14, 1.0)

    # Compute energy
    energy_raw = self.pair_energy(qi, qj, r) * scale
    energy = jnp.where(include, energy_raw, 0.0)

    # Factor of 0.5 to avoid double counting
    return 0.5 * jnp.sum(energy)


class EwaldCoulomb(CoulombHandler):
  """Ewald summation for coulomb interactions."""

  def __init__(self, alpha: float = 0.23, kmax: int = 5, r_cut: float = 12.0):
    """Initialize Ewald coulomb handler.

    Args:
        alpha: Ewald damping parameter (Å⁻¹).
        kmax: Maximum k-vector index for reciprocal sum.
        r_cut: Real-space cutoff (Å).
    """
    self.alpha = alpha
    self.kmax = kmax
    self.r_cut = r_cut

  def reciprocal_energy(
    self, positions: Array, charges: Array, box: Array
  ) -> Array:
    """Compute reciprocal space energy."""
    vol = jnp.prod(box)

    # Generate k-vectors
    k_range = jnp.arange(-self.kmax, self.kmax + 1)
    KX, KY, KZ = jnp.meshgrid(k_range, k_range, k_range, indexing='ij')
    kvecs = jnp.stack([KX, KY, KZ], axis=-1).reshape(-1, 3)

    def compute_term(k):
      is_zero = jnp.all(k == 0)
      k_cart = 2 * jnp.pi * k / box
      k2 = jnp.dot(k_cart, k_cart)
      rho_k = jnp.sum(charges * jnp.exp(1j * jnp.dot(positions, k_cart)))
      factor = jnp.where(is_zero, 0.0, jnp.exp(-k2 / (4 * self.alpha**2)) / k2)
      return (4 * jnp.pi / vol) * factor * jnp.abs(rho_k) ** 2

    energy_terms = vmap(compute_term)(kvecs)
    return 0.5 * jnp.sum(energy_terms) * COULOMB_CONSTANT

  def self_energy(self, charges: Array) -> Array:
    """Compute self-energy correction."""
    return (
      -self.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges**2) * COULOMB_CONSTANT
    )

  def real_energy(
    self,
    positions: Array,
    charges: Array,
    box: Array,
    exclusion_mask: Array,
    pair_14_mask: Array,
    nlist: NeighborList,
    scale_14: float,
  ) -> Array:
    """Compute real-space energy."""
    n_atoms = positions.shape[0]
    max_neighbors = nlist.idx.shape[1]

    displacement_fn, _ = space.periodic(box)

    # Prepare neighbor indices
    idx_i = jnp.repeat(jnp.arange(n_atoms)[:, None], max_neighbors, axis=1)
    idx_j = nlist.idx

    # Valid neighbor mask
    valid = (idx_j >= 0) & (idx_j < n_atoms)
    idx_j_safe = jnp.where(valid, idx_j, 0)
    idx_i_safe = jnp.where(valid, idx_i, 0)

    # Get charges and positions
    qi = charges[idx_i_safe]
    qj = charges[idx_j_safe]
    ri = positions[idx_i_safe]
    rj = positions[idx_j_safe]

    # Compute distances
    batched_disp = vmap(vmap(displacement_fn, in_axes=(0, 0)), in_axes=(0, 0))
    disp = batched_disp(ri, rj)
    r_sq = jnp.sum(disp**2, axis=-1)
    r_sq_safe = jnp.maximum(r_sq, 1e-12)
    r = jnp.sqrt(r_sq_safe)

    # Check exclusions and scaling
    same = idx_i_safe == idx_j_safe
    excluded = vmap(lambda i, j: exclusion_mask[i, j])(idx_i_safe, idx_j_safe)
    is_14 = vmap(lambda i, j: pair_14_mask[i, j])(idx_i_safe, idx_j_safe)

    # Include mask
    include = valid & (~same) & (r < self.r_cut)

    # Coulomb scaling factors
    factor_coul = jnp.where(is_14, scale_14, 1.0)
    factor_coul = jnp.where(excluded, 0.0, factor_coul)

    # Erfc term with proper scaling
    erfc_val = erfc(self.alpha * r)
    energy_raw = (
      COULOMB_CONSTANT * qi * qj / r * (erfc_val - (1.0 - factor_coul))
    )

    # Mask out invalid pairs
    energy = jnp.where(include, energy_raw, 0.0)

    return 0.5 * jnp.sum(energy)

  def energy(
    self,
    positions: Array,
    charges: Array,
    box: Array,
    exclusion_mask: Array,
    pair_14_mask: Array,
    nlist: NeighborList,
    scale_14: float = 0.5,
  ) -> Array:
    """Compute total Ewald coulomb energy."""
    e_real = self.real_energy(
      positions, charges, box, exclusion_mask, pair_14_mask, nlist, scale_14
    )
    e_recip = self.reciprocal_energy(positions, charges, box)
    e_self = self.self_energy(charges)
    return e_real + e_recip + e_self


class PMECoulomb(CoulombHandler):
  """Particle Mesh Ewald for coulomb interactions."""

  def __init__(
    self, grid_size: int = 32, alpha: float = 0.3, r_cut: float = 12.0
  ):
    """Initialize PME coulomb handler.

    Args:
        grid_size: Number of grid points per dimension.
        alpha: Ewald damping parameter (Å⁻¹).
        r_cut: Real-space cutoff (Å).
    """
    self.grid_size = grid_size
    self.alpha = alpha
    self.r_cut = r_cut

  def structure_factor(
    self, charges: Array, positions: Array, box: Array
  ) -> Array:
    """Map charges to grid using linear interpolation."""
    scaled_pos = positions / box * self.grid_size
    base = jnp.floor(scaled_pos).astype(int)
    frac = scaled_pos - base

    def deposit_single(charge, b, f):
      ix = (b[0] + jnp.array([0, 1])) % self.grid_size
      iy = (b[1] + jnp.array([0, 1])) % self.grid_size
      iz = (b[2] + jnp.array([0, 1])) % self.grid_size

      wx = jnp.array([1 - f[0], f[0]])
      wy = jnp.array([1 - f[1], f[1]])
      wz = jnp.array([1 - f[2], f[2]])

      grid = jnp.zeros((self.grid_size, self.grid_size, self.grid_size))

      for dx in range(2):
        for dy in range(2):
          for dz in range(2):
            weight = charge * wx[dx] * wy[dy] * wz[dz]
            grid = grid.at[ix[dx], iy[dy], iz[dz]].add(weight)
      return grid

    grids = vmap(deposit_single)(charges, base, frac)
    return jnp.sum(grids, axis=0)

  def reciprocal_energy(self, rho_k: Array, box: Array) -> Array:
    """Compute reciprocal space energy from FFT of charge grid."""
    vol = jnp.prod(box)

    # Frequency grids
    freq = jnp.fft.fftfreq(self.grid_size) * self.grid_size
    Gx, Gy, Gz = jnp.meshgrid(freq, freq, freq, indexing='ij')
    G2 = (2 * jnp.pi) ** 2 * (
      Gx**2 / box[0] ** 2 + Gy**2 / box[1] ** 2 + Gz**2 / box[2] ** 2
    )

    # Compute energy in reciprocal space
    mask = G2 > 0
    factor = jnp.where(mask, jnp.exp(-G2 / (4 * self.alpha**2)) / G2, 0.0)
    rho_sq = jnp.abs(rho_k) ** 2
    energy = (4 * jnp.pi / vol) * jnp.sum(factor * rho_sq)

    return 0.5 * COULOMB_CONSTANT * energy

  def self_energy(self, charges: Array) -> Array:
    """Compute self-energy correction."""
    return (
      -self.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges**2) * COULOMB_CONSTANT
    )

  def real_energy(
    self,
    positions: Array,
    charges: Array,
    box: Array,
    exclusion_mask: Array,
    pair_14_mask: Array,
    nlist: NeighborList,
    scale_14: float,
  ) -> Array:
    """Compute real-space energy (same as Ewald)."""
    n_atoms = positions.shape[0]
    max_neighbors = nlist.idx.shape[1]

    displacement_fn, _ = space.periodic(box)

    # Prepare neighbor indices
    idx_i = jnp.repeat(jnp.arange(n_atoms)[:, None], max_neighbors, axis=1)
    idx_j = nlist.idx

    # Valid neighbor mask
    valid = (idx_j >= 0) & (idx_j < n_atoms)
    idx_j_safe = jnp.where(valid, idx_j, 0)
    idx_i_safe = jnp.where(valid, idx_i, 0)

    # Get charges and positions
    qi = charges[idx_i_safe]
    qj = charges[idx_j_safe]
    ri = positions[idx_i_safe]
    rj = positions[idx_j_safe]

    # Compute distances
    batched_disp = vmap(vmap(displacement_fn, in_axes=(0, 0)), in_axes=(0, 0))
    disp = batched_disp(ri, rj)
    r_sq = jnp.sum(disp**2, axis=-1)
    r_sq_safe = jnp.maximum(r_sq, 1e-12)
    r = jnp.sqrt(r_sq_safe)

    # Check exclusions and scaling
    same = idx_i_safe == idx_j_safe
    excluded = vmap(lambda i, j: exclusion_mask[i, j])(idx_i_safe, idx_j_safe)
    is_14 = vmap(lambda i, j: pair_14_mask[i, j])(idx_i_safe, idx_j_safe)

    # Include mask
    include = valid & (~same) & (r < self.r_cut)

    # Coulomb scaling factors
    factor_coul = jnp.where(is_14, scale_14, 1.0)
    factor_coul = jnp.where(excluded, 0.0, factor_coul)

    # Erfc term with proper scaling
    erfc_val = erfc(self.alpha * r)
    energy_raw = (
      COULOMB_CONSTANT * qi * qj / r * (erfc_val - (1.0 - factor_coul))
    )

    # Mask out invalid pairs
    energy = jnp.where(include, energy_raw, 0.0)

    return 0.5 * jnp.sum(energy)

  def energy(
    self,
    positions: Array,
    charges: Array,
    box: Array,
    exclusion_mask: Array,
    pair_14_mask: Array,
    nlist: NeighborList,
    scale_14: float = 0.5,
  ) -> Array:
    """Compute total PME coulomb energy."""
    # Real space
    e_real = self.real_energy(
      positions, charges, box, exclusion_mask, pair_14_mask, nlist, scale_14
    )

    # Reciprocal space
    rho_real = self.structure_factor(charges, positions, box)
    rho_k = jnp.fft.fftn(rho_real)
    e_recip = self.reciprocal_energy(rho_k, box)

    # Self energy
    e_self = self.self_energy(charges)

    return e_real + e_recip + e_self
