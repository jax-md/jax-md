"""Electrostatic interactions for molecular mechanics forcefields.

This module provides coulomb energy functions that support:
- Exclusion masks (e.g., for bonded/angle pairs)
- 1-4 interaction scaling
- Units in kcal/mol (common for MM forcefields)
"""

from functools import partial
from typing import Callable
import jax.numpy as jnp
import numpy as onp
from jax import vmap, custom_jvp
from jax.scipy.special import erf, erfc
from jax_md import space, smap
from jax_md.util import Array, high_precision_sum
from jax_md.partition import NeighborList
from jax_md.mm_forcefields.base import combine_product


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
  
  def prepare_smap(self, charges, box, exc_charge_prod, displacement_fn, cutoff_fn, fractional_coordinates):
    def pair_energy_map(dr, charge_sq, **unused_kwargs):
      """Compute pairwise coulomb energy."""
      energy = COULOMB_CONSTANT * (charge_sq / dr)
      if self.use_erfc:
        energy *= erfc(self.alpha * dr)
      return energy
    
    def pair_plain_map(dr, charge_sq, **unused_kwargs):
      """Compute pairwise coulomb energy."""
      energy = COULOMB_CONSTANT * (charge_sq / dr)
      return energy
    
    pair_coul_fn = smap.pair_neighbor_list(
      cutoff_fn(pair_energy_map),
      space.canonicalize_displacement_or_metric(displacement_fn),
      ignore_unused_parameters=True,
      charge_sq=(combine_product, None),
    )

    ### Modified (or zero) coulomb interaction for exceptions
    bond_coul_fn = smap.bond(
      pair_plain_map,
      space.canonicalize_displacement_or_metric(displacement_fn),
      charge_sq=None,
    )

    return (pair_coul_fn, bond_coul_fn)

  def energy_smap(self,
    positions: Array,
    charges: Array,
    nlist: NeighborList,
    #box: Array,
    box_kwarg: any,
    exc_pairs: Array,
    exc_charge_prod: Array,
    return_components: bool = False,
    coulomb_fns: any = None,
  ) -> Array:
    pair_coul_fn, bond_coul_fn = coulomb_fns

    energy = pair_coul_fn(positions, nlist, charge_sq=charges, **box_kwarg)
    e_exceptions = bond_coul_fn(positions, exc_pairs, charge_sq=exc_charge_prod, **box_kwarg)

    return energy, e_exceptions

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

# TODO custom jvp's and class methods don't seem to work well together
@custom_jvp
def transform_gradients(box, coords):
  # This function acts as a no-op in the forward pass, but it transforms the
  # gradients into fractional coordinates in the backward pass.
  return coords


@transform_gradients.defjvp
def _(primals, tangents):
  box, coords = primals
  dbox, dcoords = tangents
  return coords, space.transform(dbox, coords) + space.transform(box, dcoords)

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

  # B-Spline and charge (or structure factor) smearing code.
  # copied from _energy/electrostatics.py with modifications
  # for clarity and higher b-spline order support

  @staticmethod
  @partial(jnp.vectorize, signature='()->(p)')
  def optimized_bspline_4(w):
    """Order 4 cardinal B-spline coefficients for PME.

    This function computes the 1D charge-assignment weights, evaluated
    at the fractional coordinate w = [0, 1). Some reordering or exploiting
    partition of unity (sum of weights is 1.0) can be used to slightly reduce
    flops, but explicit recursion is used here for clarity.
    """
    coeffs = jnp.zeros((4,))

    # Initialize order-2 weights - linear on [0,1)
    coeffs = coeffs.at[1].set(w)
    coeffs = coeffs.at[0].set(1.0 - w)

    # Elevate order 2 -> 3 (divide by 2)
    div = 0.5
    coeffs = coeffs.at[2].set(div * w * coeffs[1])
    coeffs = coeffs.at[1].set(div * ((w + 1.0) * coeffs[0] + (2.0 - w) * coeffs[1]))
    coeffs = coeffs.at[0].set(div * (1.0 - w) * coeffs[0])

    # Elevate order 3 -> 4 (divide by 3)
    div = 1.0 / 3.0
    coeffs = coeffs.at[3].set(div * w * coeffs[2])
    coeffs = coeffs.at[2].set(div * ((w + 1.0) * coeffs[1] + (3.0 - w) * coeffs[2]))
    coeffs = coeffs.at[1].set(div * ((w + 2.0) * coeffs[0] + (2.0 - w) * coeffs[1]))
    coeffs = coeffs.at[0].set(div * (1.0 - w) * coeffs[0])

    return coeffs
  

  @staticmethod
  @partial(jnp.vectorize, signature='()->(p)')
  def optimized_bspline_5(w):
    """Order 5 cardinal B-spline coefficients for PME.

    This function computes the 1D charge-assignment weights, evaluated
    at the fractional coordinate w = [0, 1). Some reordering or exploiting
    partition of unity (sum of weights is 1.0) can be used to slightly reduce
    flops, but explicit recursion is used here for clarity.
    """
    coeffs = jnp.zeros((5,))

    # Initialize order-2 weights - linear on [0,1)
    coeffs = coeffs.at[1].set(w)
    coeffs = coeffs.at[0].set(1.0 - w)

    # Elevate order 2 -> 3 (divide by 2)
    div = 0.5
    coeffs = coeffs.at[2].set(div * w * coeffs[1])
    coeffs = coeffs.at[1].set(div * ((w + 1.0) * coeffs[0] + (2.0 - w) * coeffs[1]))
    coeffs = coeffs.at[0].set(div * (1.0 - w) * coeffs[0])

    # Elevate order 3 -> 4 (divide by 3)
    div = 1.0 / 3.0
    coeffs = coeffs.at[3].set(div * w * coeffs[2])
    coeffs = coeffs.at[2].set(div * ((w + 1.0) * coeffs[1] + (3.0 - w) * coeffs[2]))
    coeffs = coeffs.at[1].set(div * ((w + 2.0) * coeffs[0] + (2.0 - w) * coeffs[1]))
    coeffs = coeffs.at[0].set(div * (1.0 - w) * coeffs[0])

    # Elevate order 4 -> 5 (divide by 4).
    div = 0.25
    coeffs = coeffs.at[4].set(div * w * coeffs[3])
    coeffs = coeffs.at[3].set(div * ((w + 1.0) * coeffs[2] + (4.0 - w) * coeffs[3]))
    coeffs = coeffs.at[2].set(div * ((w + 2.0) * coeffs[1] + (3.0 - w) * coeffs[2]))
    coeffs = coeffs.at[1].set(div * ((w + 3.0) * coeffs[0] + (2.0 - w) * coeffs[1]))
    coeffs = coeffs.at[0].set(div * (1.0 - w) * coeffs[0])

    return coeffs
  

  @staticmethod
  def map_charges_to_grid(
    position: Array,
    charge: Array,
    inverse_box: space.Box,
    grid_dimensions: Array,
    fractional_coordinates: bool,
    order: int,
  ) -> Array:
    """Smears charges over a grid of specified dimensions."""
    Q = jnp.zeros(grid_dimensions)
    N = position.shape[0]

    @partial(jnp.vectorize, signature='(),()->(p)')
    def grid_position(u, K):
      grid = jnp.floor(u).astype(jnp.int32)
      grid = jnp.arange(order) + grid
      return jnp.mod(grid, K)

    @partial(jnp.vectorize, signature='(d),()->(p,p,p,d),(p,p,p)')
    def map_particle_to_grid(position, charge):
      if fractional_coordinates:
        u = transform_gradients(inverse_box, position) * grid_dimensions
      else:
        u = space.raw_transform(inverse_box, position) * grid_dimensions

      w = u - jnp.floor(u)
      if order == 4:
        coeffs = PMECoulomb.optimized_bspline_4(w)
      elif order == 5:
        coeffs = PMECoulomb.optimized_bspline_5(w)
      else:
        raise ValueError(f'Unsupported PME spline order {order}.')

      grid_pos = grid_position(u, grid_dimensions)

      accum = charge * (
        coeffs[0, :, None, None]
        * coeffs[1, None, :, None]
        * coeffs[2, None, None, :]
      )
      grid_pos = jnp.concatenate(
        (
          jnp.broadcast_to(grid_pos[[0], :, None, None], (1, order, order, order)),
          jnp.broadcast_to(grid_pos[[1], None, :, None], (1, order, order, order)),
          jnp.broadcast_to(grid_pos[[2], None, None, :], (1, order, order, order)),
        ),
        axis=0,
      )
      grid_pos = jnp.transpose(grid_pos, (1, 2, 3, 0))

      return grid_pos, accum

    gp, ac = map_particle_to_grid(position, charge)
    gp = jnp.reshape(gp, (-1, 3))
    ac = jnp.reshape(ac, (-1,))

    return Q.at[gp[:, 0], gp[:, 1], gp[:, 2]].add(ac)


  @staticmethod
  @partial(jnp.vectorize, signature='()->()')
  def b(m, n=4):
    if n not in (4, 5):
      raise ValueError(f'Unsupported PME spline order {n}.')
    k = jnp.arange(n - 1)
    if n == 4:
      M = PMECoulomb.optimized_bspline_4(1.0)[1:][::-1]
    else:
      M = PMECoulomb.optimized_bspline_5(1.0)[1:][::-1]
    prefix = jnp.exp(2 * jnp.pi * 1j * (n - 1) * m)
    return prefix / jnp.sum(M * jnp.exp(2 * jnp.pi * 1j * m * k))

  @staticmethod
  def B(mx, my, mz, n=4):
    """Compute the B factors from Essmann et al. equation 4.7."""
    b_x = PMECoulomb.b(mx, n=n)
    b_y = PMECoulomb.b(my, n=n)
    b_z = PMECoulomb.b(mz, n=n)
    return jnp.abs(b_x) ** 2 * jnp.abs(b_y) ** 2 * jnp.abs(b_z) ** 2

  def coulomb_recip_pme(
    self,
    charge: Array,
    box: space.Box,
    grid_points: Array,
    fractional_coordinates: bool = False,
    alpha: float = 0.34,
    order: int = 4,
  ) -> Callable[[Array], Array]:
    _ibox = space.inverse(box)

    def energy_fn(R, **kwargs):
      q = kwargs.pop('charge', charge)
      box_overridden = 'box' in kwargs
      _box = kwargs.pop('box', box)
      perturbation = kwargs.pop('perturbation', None)
      # Explicit handing for perturbation
      if perturbation is not None:
        _box = _box * perturbation
        box_overridden = True
      ibox = space.inverse(_box) if box_overridden else _ibox
      dim = R.shape[-1]
      # Required to avoid tracing of jax.ArrayImpl instance
      grid_dimensions = onp.array(grid_points)

      grid = PMECoulomb.map_charges_to_grid(
        R, q, ibox, grid_dimensions, fractional_coordinates, order=order
      )
      Fgrid = jnp.fft.fftn(grid)

      # Use matrix ('ij') indexing so (mx,my,mz) axes align with FFT output axes.
      mx, my, mz = jnp.meshgrid(
        *[jnp.fft.fftfreq(int(g)) for g in grid_dimensions], indexing='ij'
      )

      if jnp.isscalar(_box) or getattr(_box, "ndim", 0) == 0:
        m_2 = (mx**2 + my**2 + mz**2) * (grid_dimensions[0] * ibox) ** 2
        V = (1.0 * _box) ** dim
      elif _box.ndim == 1:
        # Orthorhombic box given as side lengths (Lx, Ly, Lz)
        m_2 = (
          (mx * (grid_dimensions[0] * ibox[0])) ** 2
          + (my * (grid_dimensions[1] * ibox[1])) ** 2
          + (mz * (grid_dimensions[2] * ibox[2])) ** 2
        )
        V = jnp.prod(_box)
      else:
        # Triclinic box as a 3x3 matrix
        m = (
          ibox[None, None, None, 0] * mx[:, :, :, None] * grid_dimensions[0]
          + ibox[None, None, None, 1] * my[:, :, :, None] * grid_dimensions[1]
          + ibox[None, None, None, 2] * mz[:, :, :, None] * grid_dimensions[2]
        )
        m_2 = jnp.sum(m**2, axis=-1)
        V = jnp.linalg.det(_box)

      mask = m_2 != 0
      # NOTE m_2 = 0 at the (0,0,0) mode - this can lead to nan gradients 
      # during autodiff with dx/dV - this masking may still not be safe
      m_2 = jnp.where(mask, m_2, 1.0)
      exp_m = (
        1 / (2 * jnp.pi * V) * jnp.exp(-(jnp.pi**2) * m_2 / alpha**2) / m_2
      )
      return high_precision_sum(
        mask * exp_m * PMECoulomb.B(mx, my, mz, n=order) * jnp.abs(Fgrid) ** 2
      )

    return energy_fn

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

  def prepare_smap(self,
    charges: Array,
    box: Array,
    exc_charge_prod: Array,
    return_components: bool = False,
    displacement_fn: any = None,
    cutoff_fn: any = None,
    fractional_coordinates: bool = False,
  ) -> Array:
    # TODO look at smap and all 1/r terms to figure out most
    # robust masking scheme to avoid bad energy/forces
    def pair_plain_map(dr, charge_sq, **unused_kwargs):
      """Compute pairwise coulomb energy."""
      dr = jnp.where(jnp.isclose(dr, 0.), 1, dr)
      energy = COULOMB_CONSTANT * (charge_sq / dr)
      return energy
    
    def pair_energy_map(dr, charge_sq, **unused_kwargs):
      """Compute pairwise coulomb energy."""
      dr = jnp.where(jnp.isclose(dr, 0.), 1, dr)
      energy = COULOMB_CONSTANT * (charge_sq / dr)
      energy *= erfc(self.alpha * dr)
      return energy
    
    def pair_correction(dr, charge_sq, **unused_kwargs):
      dr = jnp.where(jnp.isclose(dr, 0.), 1, dr)
      return -(COULOMB_CONSTANT * charge_sq * erf(self.alpha * dr) / dr)
    
    ### Real space contribution
    pair_coul_fn = smap.pair_neighbor_list(
      cutoff_fn(pair_energy_map),
      space.canonicalize_displacement_or_metric(displacement_fn),
      ignore_unused_parameters=True,
      charge_sq=(combine_product, None),
    )

    ### Reciprocal space contribution
    # TODO it may make more sense to separate dynamic parameters from the
    # generator as they're passed by closure into the step function
    recip_fn = self.coulomb_recip_pme(charges,
                                      box,
                                      self.grid_size,
                                      fractional_coordinates=fractional_coordinates,
                                      alpha=self.alpha)

    ### Remove exceptions implicitly included in reciprocal term
    bond_corr_fn = smap.bond(
      pair_correction,
      space.canonicalize_displacement_or_metric(displacement_fn),
      charge_sq=(combine_product, None)
    )

    ### Modified (or zero) coulomb interaction for exceptions
    bond_coul_fn = smap.bond(
      pair_plain_map,
      space.canonicalize_displacement_or_metric(displacement_fn),
      charge_sq=None,
    )

    return pair_coul_fn, recip_fn, bond_corr_fn, bond_coul_fn


  def energy_smap(self,
    positions: Array,
    charges: Array,
    nlist: NeighborList,
    box_kwarg: any,
    exc_pairs: Array,
    exc_charge_prod: Array,
    return_components: bool = False,
    coulomb_fns: any = None,
  ) -> Array:
    pair_coul_fn, recip_fn, bond_corr_fn, bond_coul_fn = coulomb_fns
    
    ### Real space contribution
    e_real = pair_coul_fn(positions, nlist, charge_sq=charges, **box_kwarg)

    ### Reciprocal space contribution
    e_recip = COULOMB_CONSTANT * recip_fn(positions, charge=charges, **box_kwarg)

    ### Self energy contribution
    e_self = self.self_energy(charges)

    ### Remove exceptions implicitly included in reciprocal term
    e_corr = bond_corr_fn(positions, exc_pairs, charge_sq=charges, **box_kwarg)

    ### Modified (or zero) coulomb interaction for exceptions
    e_exceptions = bond_coul_fn(positions, exc_pairs, charge_sq=exc_charge_prod, **box_kwarg)

    total_e = e_real + e_recip + e_self + e_corr
    return total_e, e_exceptions

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
