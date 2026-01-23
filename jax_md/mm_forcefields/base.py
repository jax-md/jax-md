"""Base dataclasses and types for molecular mechanics forcefields."""

from functools import wraps, partial
from typing import Callable, NamedTuple, Optional
import jax.numpy as jnp
from jax_md.util import Array, safe_mask
from jax_md.partition import NeighborListFormat
from jax_md.mm_forcefields.reaxff.reaxff_helper import safe_sqrt


class Topology(NamedTuple):
  """Molecular topology defining bonded connectivity.

  Attributes:
      n_atoms: Number of atoms in the system.
      bonds: Array of shape (n_bonds, 2) with atom indices.
      angles: Array of shape (n_angles, 3) with atom indices.
      torsions: Array of shape (n_torsions, 4) with atom indices.
      impropers: Array of shape (n_impropers, 4) with atom indices.
      exclusion_mask: Boolean array of shape (n_atoms, n_atoms) indicating
          pairs that should be excluded from nonbonded interactions.
      pair_14_mask: Boolean array of shape (n_atoms, n_atoms) indicating
          1-4 pairs that get special scaling.
      molecule_id: Array of shape (n_atoms,) assigning each atom to a molecule.
      cmap_atoms: Array of shape (n_cmap, 8) with atom indices.
      cmap_map_idx: Array of shape (n_cmap,) with map indices.
      exc_pairs: Array of shape (n_exception, 2) with atom indices.
      nbfix_atom_type: Array of shape (n_atom,) with NBFIX LJ indices.
  """

  n_atoms: int
  bonds: Array
  angles: Array
  torsions: Array
  impropers: Array
  exclusion_mask: Array
  pair_14_mask: Array
  molecule_id: Optional[Array] = None
  cmap_atoms: Optional[Array] = None
  cmap_map_idx: Optional[Array] = None
  exc_pairs: Optional[Array] = None
  nbfix_atom_type: Optional[Array] = None


class BondedParameters(NamedTuple):
  """Force constants and equilibrium values for bonded terms.

  All arrays should have shape matching the corresponding topology arrays.

  Attributes:
      bond_k: Bond force constants (kcal/mol/Å²).
      bond_r0: Equilibrium bond lengths (Å).
      angle_k: Angle force constants (kcal/mol/rad²).
      angle_theta0: Equilibrium angles (radians).
      torsion_k: Torsion force constants (kcal/mol).
      torsion_n: Torsion periodicity.
      torsion_gamma: Torsion phase (radians).
      improper_k: Improper force constants (kcal/mol).
      improper_n: Improper periodicity.
      improper_gamma: Improper phase (radians).
      cmap_maps: CMAP energy maps (kcal/mol).
  """

  bond_k: Array
  bond_r0: Array
  angle_k: Array
  angle_theta0: Array
  torsion_k: Array
  torsion_n: Array
  torsion_gamma: Array
  improper_k: Array
  improper_n: Array
  improper_gamma: Array
  cmap_maps: Optional[Array] = None


class NonbondedParameters(NamedTuple):
  """Nonbonded interaction parameters.

  Attributes:
      charges: Partial charges for each atom (e).
      sigma: Lennard-Jones sigma parameters (Å).
      epsilon: Lennard-Jones epsilon parameters (kcal/mol).
      exc_charge_prod: Charge products for pairs of nonbonded exceptions (e).
      exc_sigma: Sigma values for exception LJ term.
      exc_epsilon: Epsilon values for exception LJ term.
      nbfix_acoef: LJ A coefficient for NBFIX pairs. #TODO unit?
      nbfix_bcoef: LJ B coefficient for NBFIX pairs. #TODO unit?
  """

  charges: Array
  sigma: Array
  epsilon: Array
  exc_charge_prod: Optional[Array] = None
  exc_sigma: Optional[Array] = None
  exc_epsilon: Optional[Array] = None
  nbfix_acoef: Optional[Array] = None
  nbfix_bcoef: Optional[Array] = None


class NonbondedOptions(NamedTuple):
  """Options for nonbonded interaction calculations.

  Attributes:
      r_cut: Cutoff distance for nonbonded interactions (Å).
      dr_threshold: Neighbor list skin distance (Å).
      use_soft_lj: Whether to use soft-core LJ potential.
      lj_cap: Cap value for soft-core LJ (kcal/mol).
      use_shift_lj: Whether to shift LJ potential to zero at cutoff.
      scale_14_lj: Scaling factor for 1-4 LJ interactions (typically 0.5).
      scale_14_coul: Scaling factor for 1-4 Coulomb interactions (typically 0.5).
      nb_format: Format to store neighbor indices, instance of partition.NeighborListFormat.
      use_pbc: Flag to enable or disable periodic boundary conditions.
      use_periodic_general: Flag to enable or disable the use of PBCs on a parallelepiped.
      fractional_coordinates: Flag to enable the use of fractional coordinates for neighbor list.
      wrapped_space: Flag to enable or disable wrapping for periodic unit cell.
      disp_coef: dispersion coefficient for coulomb interaction.
      r_switch: Relevant switching distance if enabled, must be less than r_cut.
  """

  r_cut: float = 12.0
  dr_threshold: float = 0.5
  use_soft_lj: bool = False
  lj_cap: float = 1000.0
  use_shift_lj: bool = False
  scale_14_lj: float = 0.5
  scale_14_coul: float = 0.5
  nb_format: NeighborListFormat = (
    NeighborListFormat.Dense
  )  # TODO make all optional
  use_pbc: bool = True
  use_periodic_general: bool = False
  fractional_coordinates: bool = False
  wrapped_space: bool = True
  disp_coef: float = 0.0
  r_switch: float = 0.0


# Common combinators for nonbonded mixing
# TODO add waldman-hagler and other combining rules if necessary
def combine_lorentz(v1: Array, v2: Array) -> Array:
  """Lorentz mixing rule (arithmetic mean).

  Args:
    v1: First per-particle parameter.
    v2: Second per-particle parameter.

  Returns:
    Mixed parameter value.
  """
  return 0.5 * (v1 + v2)


def combine_berthelot(v1: Array, v2: Array) -> Array:
  """Berthelot mixing rule (geometric mean).

  Args:
    v1: First per-particle parameter.
    v2: Second per-particle parameter.

  Returns:
    Mixed parameter value.
  """
  return safe_sqrt(v1 * v2)


def combine_product(q1: Array, q2: Array) -> Array:
  """Simple product rule.

  Args:
    q1: First value.
    q2: Second value.

  Returns:
    Product q1*q2.
  """
  return q1 * q2


def compute_angle(dr_12: Array, dr_32: Array) -> Array:
  """Compute the angle (in radians) from two displacement vectors.

  This returns the angle formed by the three atoms (1-2-3) given displacement
  vectors from the central atom (2) to the outer atoms (1 and 3).

  Args:
    dr_12: Displacement vector from atom 2 to atom 1, shape (3,).
    dr_32: Displacement vector from atom 2 to atom 3, shape (3,).

  Returns:
    Angle in radians in [0, pi].
  """
  d_12 = jnp.linalg.norm(dr_12 + 1e-7)
  d_32 = jnp.linalg.norm(dr_32 + 1e-7)
  cos_angle = jnp.dot(dr_12, dr_32) / (d_12 * d_32)
  return safe_mask((cos_angle < 1) & (cos_angle > -1), jnp.arccos, cos_angle)


def compute_dihedral(v1: Array, v2: Array, v3: Array) -> Array:
  """Compute a signed dihedral angle (in radians) from three displacement vectors.

  This implements a common "praxeolitic" dihedral definition using displacement
  vectors, which avoids explicit position wrapping issues under PBCs.

  Args:
    v1: Displacement vector r_01 (from atom 1 to atom 0), shape (3,).
    v2: Displacement vector r_12 (from atom 2 to atom 1), shape (3,).
    v3: Displacement vector r_23 (from atom 3 to atom 2), shape (3,).

  Returns:
    Signed dihedral angle in radians in (-pi, pi].
  """
  # using displacements instead of positions avoids periodicity issues
  b0 = -1.0 * (v1)
  b1 = v2
  b2 = v3

  # normalize b1 so that it does not influence magnitude of vector
  # rejections that come next
  b1 /= jnp.linalg.norm(b1 + 1e-10)

  # vector rejections
  # v = projection of b0 onto plane perpendicular to b1
  #   = b0 minus component that aligns with b1
  # w = projection of b2 onto plane perpendicular to b1
  #   = b2 minus component that aligns with b1
  v = b0 - jnp.dot(b0, b1) * b1
  w = b2 - jnp.dot(b2, b1) * b1

  # angle between v and w in a plane is the torsion angle
  # v and w may not be normalized but that's fine since tan is y/x
  x = jnp.dot(v, w)
  y = jnp.dot(jnp.cross(b1, v), w)
  r = jnp.arctan2(y, x + 1e-10)

  return r


# Cutoff / switching utilities
# TODO look into other common functions to add here

EnergyFn = Callable[..., Array]

# TODO based off of multiplicative_isotropic_cut but may need some changes
def hard_cutoff(fn: EnergyFn, r_cut: float) -> EnergyFn:
  """Wrap a pairwise function with a hard cutoff.

  Args:
    fn: Function of the form `fn(dr, *args, **kwargs) -> energy`.
    r_cut: Cutoff radius (same units as `dr`).

  Returns:
    A wrapped function that multiplies `fn` by an indicator `(dr < r_cut)`.
  """

  # TODO double where needed? why mask?
  def smooth_fn(dr: Array) -> Array:
    return jnp.where(dr < r_cut, 1, 0)

  @wraps(fn)
  def cutoff_fn(dr: Array, *args, **kwargs) -> Array:
    return smooth_fn(dr) * fn(dr, *args, **kwargs)

  return cutoff_fn


def force_switch(fn: EnergyFn, r_on: float, r_off: float) -> EnergyFn:
  """Wrap a pairwise function with a CHARMM-style force-switching function.

  The switching function is 1 at `r_on`, goes smoothly to 0 at `r_off`, and is
  clamped outside [r_on, r_off].

  Args:
    fn: Function of the form `fn(dr, *args, **kwargs) -> energy`.
    r_on: Switch-on radius.
    r_off: Switch-off radius (typically equals the neighbor cutoff).

  Returns:
    A wrapped function that multiplies `fn` by the switching polynomial.
  """

  def smooth_fn(dr: Array) -> Array:
    s = jnp.clip((dr - r_on) / (r_off - r_on), 0.0, 1.0)
    switch = 1 - 10 * s**3 + 15 * s**4 - 6 * s**5
    return switch

  @wraps(fn)
  def cutoff_fn(dr: Array, *args, **kwargs) -> Array:
    return smooth_fn(dr) * fn(dr, *args, **kwargs)

  return cutoff_fn
