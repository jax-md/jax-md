# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Definitions of various standard energy functions."""

from functools import wraps, partial

from typing import Callable, Tuple, TextIO, Dict, Any, Optional

import jax
import jax.numpy as jnp
from jax import ops
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk
from jax.scipy.special import erfc  # error function
from jax_md import space, smap, partition, nn, quantity, interpolate, util

maybe_downcast = util.maybe_downcast

# Types


f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat


# Energy Functions


def simple_spring(dr: Array,
                  length: Array=1,
                  epsilon: Array=1,
                  alpha: Array=2,
                  **unused_kwargs) -> Array:
  """Isotropic spring potential with a given rest length.

  We define `simple_spring` to be a generalized Hookean spring with
  agreement when `alpha = 2`.
  """
  return epsilon / alpha * jnp.abs(dr - length) ** alpha


def simple_spring_bond(displacement_or_metric: DisplacementOrMetricFn,
                       bond: Array,
                       bond_type: Optional[Array]=None,
                       length: Array=1,
                       epsilon: Array=1,
                       alpha: Array=2) -> Callable[[Array], Array]:
  """Convenience wrapper to compute energy of particles bonded by springs."""
  length = maybe_downcast(length)
  epsilon = maybe_downcast(epsilon)
  alpha = maybe_downcast(alpha)
  return smap.bond(
    simple_spring,
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    bond,
    bond_type,
    ignore_unused_parameters=True,
    length=length,
    epsilon=epsilon,
    alpha=alpha)


def soft_sphere(dr: Array,
                sigma: Array=1,
                epsilon: Array=1,
                alpha: Array=2,
                **unused_kwargs) -> Array:
  """.. _soft-sphere:

  Finite ranged repulsive interaction between soft spheres.

  Args:
    dr: An ndarray of shape `[n, m]` of pairwise distances between particles.
    sigma: Particle diameter. Should either be a floating point scalar or an
      ndarray whose shape is `[n, m]`.
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is `[n, m]`.
    alpha: Exponent specifying interaction stiffness. Should either be a float
      point scalar or an ndarray whose shape is `[n, m]`.
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
  Returns:
    Matrix of energies whose shape is `[n, m]`.
  """

  dr = dr / sigma
  fn = lambda dr: epsilon / alpha * (f32(1.0) - dr) ** alpha

  if isinstance(alpha, int) or issubclass(type(alpha.dtype), jnp.integer):
    return jnp.where(dr < 1.0, fn(dr), f32(0.0))

  return util.safe_mask(dr < 1.0, fn, dr, f32(0.0))


def soft_sphere_pair(displacement_or_metric: DisplacementOrMetricFn,
                     species: Optional[Array]=None,
                     sigma: Array=1.0,
                     epsilon: Array=1.0,
                     alpha: Array=2.0,
                     per_particle: bool=False) -> Callable[[Array], Array]:
  """Convenience wrapper to compute :ref:`soft sphere energy <soft-sphere>` over a system."""
  sigma = maybe_downcast(sigma)
  epsilon = maybe_downcast(epsilon)
  alpha = maybe_downcast(alpha)
  return smap.pair(
      soft_sphere,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      ignore_unused_parameters=True,
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      alpha=alpha,
      reduce_axis=(1,) if per_particle else None)


def soft_sphere_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: Box,
    species: Optional[Array]=None,
    sigma: Array=1.0,
    epsilon: Array=1.0,
    alpha: Array=2.0,
    dr_threshold: float=0.2,
    per_particle: bool=False,
    fractional_coordinates: bool=False,
    format: NeighborListFormat=partition.OrderedSparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`soft spheres <soft-sphere>` using a neighbor list."""
  sigma = maybe_downcast(sigma)
  epsilon = maybe_downcast(epsilon)
  alpha = maybe_downcast(alpha)
  list_cutoff = jnp.max(sigma)
  dr_threshold = maybe_downcast(dr_threshold)

  neighbor_fn = partition.neighbor_list(
    displacement_or_metric,
    box_size,
    list_cutoff,
    dr_threshold,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **neighbor_kwargs)
  energy_fn = smap.pair_neighbor_list(
    soft_sphere,
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    ignore_unused_parameters=True,
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha,
    reduce_axis=(1,) if per_particle else None)

  return neighbor_fn, energy_fn


def lennard_jones(dr: Array,
                  sigma: Array=1,
                  epsilon: Array=1,
                  **unused_kwargs) -> Array:
  """.. _lj-pot:

  Lennard-Jones interaction between particles with a minimum at `sigma`.

  Args:
    dr: An ndarray of shape `[n, m]` of pairwise distances between particles.
    sigma: Distance between particles where the energy has a minimum. Should
      either be a floating point scalar or an ndarray whose shape is `[n, m]`.
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is `[n, m]`.
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
  Returns:
    Matrix of energies of shape `[n, m]`.
  """
  idr = (sigma / dr)
  idr = idr * idr
  idr6 = idr * idr * idr
  idr12 = idr6 * idr6
  # TODO(schsam): This seems potentially dangerous. We should do ErrorChecking
  # here.
  return jnp.nan_to_num(f32(4) * epsilon * (idr12 - idr6))


def lennard_jones_pair(displacement_or_metric: DisplacementOrMetricFn,
                       species: Optional[Array]=None,
                       sigma: Array=1.0,
                       epsilon: Array=1.0,
                       r_onset: Array=2.0,
                       r_cutoff: Array=2.5,
                       per_particle: bool=False) -> Callable[[Array], Array]:
  """Convenience wrapper to compute :ref:`Lennard-Jones energy <lj-pot>` over a system."""
  sigma = maybe_downcast(sigma)
  epsilon = maybe_downcast(epsilon)
  r_onset = maybe_downcast(r_onset) * jnp.max(sigma)
  r_cutoff = maybe_downcast(r_cutoff) * jnp.max(sigma)
  return smap.pair(
    multiplicative_isotropic_cutoff(lennard_jones, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    ignore_unused_parameters=True,
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    reduce_axis=(1,) if per_particle else None)


def lennard_jones_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: Box,
    species: Optional[Array]=None,
    sigma: Array=1.0,
    epsilon: Array=1.0,
    alpha: Array=2.0,
    r_onset: float=2.0,
    r_cutoff: float=2.5,
    dr_threshold: float=0.5,
    per_particle: bool=False,
    fractional_coordinates: bool=False,
    format: partition.NeighborListFormat=partition.OrderedSparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`Lennard-Jones <lj-pot>` using a neighbor list."""
  sigma = maybe_downcast(sigma)
  epsilon = maybe_downcast(epsilon)
  r_onset = maybe_downcast(r_onset) * jnp.max(sigma)
  r_cutoff = maybe_downcast(r_cutoff) * jnp.max(sigma)
  dr_threshold = maybe_downcast(dr_threshold)

  neighbor_fn = partition.neighbor_list(
    displacement_or_metric,
    box_size,
    r_cutoff,
    dr_threshold,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **neighbor_kwargs)
  energy_fn = smap.pair_neighbor_list(
    multiplicative_isotropic_cutoff(lennard_jones, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    ignore_unused_parameters=True,
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    reduce_axis=(1,) if per_particle else None)

  return neighbor_fn, energy_fn


def morse(dr: Array,
          sigma: Array=1.0,
          epsilon: Array=5.0,
          alpha: Array=5.0,
          **unused_kwargs) -> Array:
  """.. _morse-pot:

  Morse interaction between particles with a minimum at `sigma`.

  Args:
    dr: An ndarray of shape `[n, m]` of pairwise distances between particles.
    sigma: Distance between particles where the energy has a minimum. Should
      either be a floating point scalar or an ndarray whose shape is `[n, m]`.
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is `[n, m]`.
    alpha: Range parameter. Should either be a floating point scalar or an
      ndarray whose shape is `[n, m]`.
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
  Returns:
    Matrix of energies of shape `[n, m]`.
  """
  U = epsilon * (f32(1) - jnp.exp(-alpha * (dr - sigma)))**f32(2) - epsilon
  # TODO(cpgoodri): ErrorChecking following lennard_jones
  return jnp.nan_to_num(jnp.array(U, dtype=dr.dtype))


def morse_pair(displacement_or_metric: DisplacementOrMetricFn,
               species: Optional[Array]=None,
               sigma: Array=1.0,
               epsilon: Array=5.0,
               alpha: Array=5.0,
               r_onset: float=2.0,
               r_cutoff: float=2.5,
               per_particle: bool=False) -> Callable[[Array], Array]:
  """Convenience wrapper to compute :ref:`Morse energy <morse-pot>` over a system."""
  sigma = maybe_downcast(sigma)
  epsilon = maybe_downcast(epsilon)
  alpha = maybe_downcast(alpha)
  return smap.pair(
    multiplicative_isotropic_cutoff(morse, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    ignore_unused_parameters=True,
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha,
    reduce_axis=(1,) if per_particle else None)


def morse_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: Box,
    species: Optional[Array]=None,
    sigma: Array=1.0,
    epsilon: Array=5.0,
    alpha: Array=5.0,
    r_onset: float=2.0,
    r_cutoff: float=2.5,
    dr_threshold: float=0.5,
    per_particle: bool=False,
    fractional_coordinates: bool=False,
    format: partition.NeighborListFormat=partition.OrderedSparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`Morse <morse-pot>` using a neighbor list."""
  sigma = maybe_downcast(sigma)
  epsilon = maybe_downcast(epsilon)
  alpha = maybe_downcast(alpha)
  r_onset = maybe_downcast(r_onset)
  r_cutoff = maybe_downcast(r_cutoff)
  dr_threshold = maybe_downcast(dr_threshold)

  neighbor_fn = partition.neighbor_list(
    displacement_or_metric,
    box_size,
    r_cutoff,
    dr_threshold,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **neighbor_kwargs)
  energy_fn = smap.pair_neighbor_list(
    multiplicative_isotropic_cutoff(morse, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    ignore_unused_parameters=True,
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha,
    reduce_axis=(1,) if per_particle else None)

  return neighbor_fn, energy_fn


def gupta_potential(displacement, p, q, r_0n, U_n, A, cutoff):
  """.. _gupta-pot:

  Gupta potential with default parameters for Au_55 cluster. Gupta
  potential was introduced by R. P. Gupta [#gupta]_. This potential uses parameters
  that were fit for bulk gold by Jellinek [#jellinek]_. This particular implementation
  of the Gupta potential was introduced by Garzon and Posada-Amarillas [#garzon]_.

  Args:
    displacement: Function to compute displacement between two positions.
    p: Gupta potential parameter of the repulsive term that was fitted for
      bulk gold.
    q: Gupta potential parameter of the attractive term that was fitted for
      bulk gold.
    r_0n:
      Parameter that determines the length scale of the potential. This
      value was particularly fit for gold clusters of size 55 atoms.
    U_n:
      Parameter that determines the energy scale, fit particularly for
      gold clusters of size 55 atoms.
    A: Parameter that was obtained using the cohesive energy of the fcc gold
      metal.
    cutoff:
      Pairwise interactions that are farther than the cutoff distance will be
      ignored.

  Returns:
    A function that takes in positions of gold atoms (shape `[n, 3]` where `n` is
    the number of atoms) and returns the total energy of the system in units
    of eV.

  .. rubric:: References
  .. [#gupta] R.P. Gupta, Phys. Rev. B 23, 6265 (1981)
  .. [#jellinek] J. Jellinek, in Metal-Ligand Interactions, edited by N. Russo and
    D. R. Salahub (Kluwer Academic, Dordrecht, 1996), p. 325.
  .. [#garzon] I.L. Garzon, A. Posada-Amarillas, Phys. Rev. B 54, 16 (1996)
  """
  def _gupta_term1(r, p, r_0n, cutoff):
    """Repulsive term in Gupta potential."""
    within_cutoff = (r > 0) & (r < cutoff)
    term1 = jnp.exp(-1.0 * p * (r / r_0n - 1))
    return jnp.where(within_cutoff, term1, 0.0)

  def _gupta_term2(r, q, r_0n, cutoff):
    """Attractive term in Gupta potential."""
    within_cutoff = (r > 0) & (r < cutoff)
    term2 = jnp.exp(-2.0 * q * (r / r_0n - 1))
    return jnp.where(within_cutoff, term2, 0.0)

  def compute_fn(R):
    dR = space.map_product(displacement)(R, R)
    dr = space.distance(dR)
    first_term = A * jnp.sum(_gupta_term1(dr, p, r_0n, cutoff), axis=1)
    # Safe sqrt used in order to ensure that force calculations are not nan
    # when the particles are too widely separated at initialization
    # (corresponding to the case where the attractive term is 0.).
    attractive_term = jnp.sum(_gupta_term2(dr, q, r_0n, cutoff), axis=1)
    second_term = util.safe_mask(attractive_term > 0,
                                 jnp.sqrt, attractive_term)
    return U_n / 2.0 * jnp.sum(first_term - second_term)

  return compute_fn


GUPTA_GOLD55_DICT = {
    'p' : 10.15,
    'q' : 4.13,
    'r_0n' : 2.96,
    'U_n' : 3.454,
    'A' : 0.118428,
}


def gupta_gold55(displacement,
                 cutoff=8.0):
  gupta_gold_fn = gupta_potential(displacement,
                                  cutoff=cutoff,
                                  **GUPTA_GOLD55_DICT)
  def energy_fn(R, **unused_kwargs):
    return gupta_gold_fn(R)
  return energy_fn


def multiplicative_isotropic_cutoff(fn: Callable[..., Array],
                                    r_onset: float,
                                    r_cutoff: float) -> Callable[..., Array]:
  """Takes an isotropic function and constructs a truncated function.

  Given a function `f:R -> R`, we construct a new function `f':R -> R` such that
  `f'(r) = f(r)` for `r < r_onset`, `f'(r) = 0` for `r > r_cutoff`, and `f(r)` is :math:`C^1`
  everywhere. To do this, we follow the approach outlined in HOOMD Blue  [#hoomd]_
  (thanks to Carl Goodrich for the pointer). We construct a function `S(r)` such
  that `S(r) = 1` for `r < r_onset`, `S(r) = 0` for `r > r_cutoff`, and `S(r)` is :math:`C^1`.
  Then `f'(r) = S(r)f(r)`.

  Args:
    fn: A function that takes an ndarray of distances of shape `[n, m]` as well
      as varargs.
    r_onset: A float specifying the distance marking the onset of deformation.
    r_cutoff: A float specifying the cutoff distance.

  Returns:
    A new function with the same signature as fn, with the properties outlined
    above.

  .. rubric:: References
  .. [#hoomd] HOOMD Blue documentation. Accessed on 05/31/2019.
      https://hoomd-blue.readthedocs.io/en/stable/module-md-pair.html#hoomd.md.pair.pair
  """

  r_c = r_cutoff ** f32(2)
  r_o = r_onset ** f32(2)

  def smooth_fn(dr):
    r = dr ** f32(2)

    inner = jnp.where(dr < r_cutoff,
                     (r_c - r)**2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o)**3,
                     0)

    return jnp.where(dr < r_onset, 1, inner)

  @wraps(fn)
  def cutoff_fn(dr, *args, **kwargs):
    return smooth_fn(dr) * fn(dr, *args, **kwargs)

  return cutoff_fn


def dsf_coulomb(r: Array,
                Q_sq: Array,
                alpha: Array=0.25,
                cutoff: float=8.0) -> Array:
  """Damped-shifted-force approximation of the coulombic interaction."""
  qqr2e = 332.06371  # Coulombic conversion factor: 1/(4*pi*epo).

  cutoffsq = cutoff * cutoff
  erfcc = erfc(alpha * cutoff)
  erfcd = jnp.exp(-alpha * alpha * cutoffsq)
  f_shift = -(erfcc / cutoffsq + 2 / jnp.sqrt(jnp.pi) * alpha * erfcd / cutoff)
  e_shift = erfcc / cutoff - f_shift * cutoff

  e = qqr2e * Q_sq / r * (erfc(alpha * r) - r * e_shift - r**2 * f_shift)
  return jnp.where(r < cutoff, e, 0.0)


def bks(dr: Array,
        Q_sq: Array,
        exp_coeff: Array,
        exp_decay: Array,
        attractive_coeff: Array,
        repulsive_coeff: Array,
        coulomb_alpha: Array,
        cutoff: float,
        **unused_kwargs) -> Array:
  """.. _bks-pot:

  Beest-Kramer-van Santen (BKS) potential [#bks]_ which is commonly used to
  model silicas. This function computes the interaction between two
  given atoms within the Buckingham form [#carre]_ , following the implementation
  from Liu et al. [#liu]_ .

  Args:
    dr: An ndarray of shape `[n, m]` of pairwise distances between particles.
    Q_sq: An ndarray of shape `[n, m]` of pairwise product of partial charges.
    exp_coeff: An ndarray of shape `[n, m]` that sets the scale of the
      exponential decay of the short-range interaction.
    attractive_coeff: An ndarray of shape `[n, m]` for the coefficient of the
      attractive 6th order term.
    repulsive_coeff: An ndarray of shape `[n, m]` for the coefficient of the
      repulsive 24th order term, to prevent the unphysical fusion of atoms.
    coulomb_alpha: Damping parameter for the approximation of the long-range
      coulombic interactions (a scalar).
    cutoff: Cutoff distance for considering pairwise interactions.
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.

  Returns:
    Matrix of energies of shape `[n, m]`.

  .. rubric:: References
  .. [#bks] Van Beest, B. W. H., Gert Jan Kramer, and R. A. Van Santen. "Force fields
    for silicas and aluminophosphates based on ab initio calculations." Physical
    Review Letters 64.16 (1990): 1955.
  .. [#carre] Carré, Antoine, et al. "Developing empirical potentials from ab initio
    simulations: The case of amorphous silica." Computational Materials Science
    124 (2016): 323-334.
  .. [#liu] Liu, Han, et al. "Machine learning Forcefield for silicate glasses."
    arXiv preprint arXiv:1902.03486 (2019).
  """
  energy = (dsf_coulomb(dr, Q_sq, coulomb_alpha, cutoff) + \
            exp_coeff * jnp.exp(-dr / exp_decay) + \
            attractive_coeff / dr ** 6 + repulsive_coeff / dr ** 24)
  return  jnp.where(dr < cutoff, energy, 0.0)


def bks_pair(displacement_or_metric: DisplacementOrMetricFn,
             species: Array,
             Q_sq: Array,
             exp_coeff: Array,
             exp_decay: Array,
             attractive_coeff: Array,
             repulsive_coeff: Array,
             coulomb_alpha: Array,
             cutoff: float) -> Callable[[Array], Array]:
  """Convenience wrapper to compute :ref:`BKS energy <bks-pot>` over a system."""
  Q_sq = maybe_downcast(Q_sq)
  exp_coeff = maybe_downcast(exp_coeff)
  exp_decay = maybe_downcast(exp_decay)
  attractive_coeff = maybe_downcast(attractive_coeff)
  repulsive_coeff = maybe_downcast(repulsive_coeff)

  return smap.pair(bks, displacement_or_metric,
                   species=species,
                   ignore_unused_parameters=True,
                   Q_sq=Q_sq,
                   exp_coeff=exp_coeff,
                   exp_decay=exp_decay,
                   attractive_coeff=attractive_coeff,
                   repulsive_coeff=repulsive_coeff,
                   coulomb_alpha=coulomb_alpha,
                   cutoff=cutoff)


def bks_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: Box,
    species: Array,
    Q_sq: Array,
    exp_coeff: Array,
    exp_decay: Array,
    attractive_coeff: Array,
    repulsive_coeff: Array,
    coulomb_alpha: Array,
    cutoff: float,
    dr_threshold: float=0.8,
    fractional_coordinates: bool=False,
    format: partition.NeighborListFormat=partition.OrderedSparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`BKS energy <bks-pot>` using a neighbor list."""
  Q_sq = maybe_downcast(Q_sq)
  exp_coeff = maybe_downcast(exp_coeff)
  exp_decay = maybe_downcast(exp_decay)
  attractive_coeff = maybe_downcast(attractive_coeff)
  repulsive_coeff = maybe_downcast(repulsive_coeff)
  dr_threshold = maybe_downcast(dr_threshold)

  neighbor_fn = partition.neighbor_list(
    displacement_or_metric,
    box_size,
    cutoff,
    dr_threshold,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **neighbor_kwargs)

  energy_fn = smap.pair_neighbor_list(
      bks,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      ignore_unused_parameters=True,
      Q_sq=Q_sq,
      exp_coeff=exp_coeff,
      exp_decay=exp_decay,
      attractive_coeff=attractive_coeff,
      repulsive_coeff=repulsive_coeff,
      coulomb_alpha=coulomb_alpha,
      cutoff=cutoff)

  return neighbor_fn, energy_fn

# BKS Potential Parameters.
# Coefficients given in kcal/mol.

CHARGE_OXYGEN = -0.977476019
CHARGE_SILICON = 1.954952037

BKS_SILICA_DICT = {
    'Q_sq' : [[CHARGE_SILICON**2, CHARGE_SILICON*CHARGE_OXYGEN],
              [CHARGE_SILICON*CHARGE_OXYGEN, CHARGE_OXYGEN**2]],
    'exp_coeff' : [[0, 471671.1243 ],
                   [471671.1243, 23138.64826]],
    'exp_decay' : [[1, 0.19173537],
                   [0.19173537, 0.356855265]],
    'attractive_coeff' : [[0, -2156.074422],
                          [-2156.074422, -1879.223108]],
    'repulsive_coeff' : [[78940848.06, 668.7557239],
                         [668.7557239, 2605.841269]],
    'coulomb_alpha' : 0.25,
}


def _bks_silica_self(Q_sq: Array, alpha: Array, cutoff: float) -> Array:
  """Function for computing the self-energy contributions to BKS."""
  cutoffsq = cutoff * cutoff
  erfcc = erfc(alpha * cutoff)
  erfcd = jnp.exp(-alpha * alpha * cutoffsq)
  f_shift = -(erfcc / cutoffsq + 2.0 / jnp.sqrt(jnp.pi) * alpha * erfcd / cutoff)
  e_shift = erfcc / cutoff - f_shift * cutoff
  qqr2e = 332.06371  # kcal/mol coulombic conversion factor: 1/(4*pi*epo)
  return -(e_shift / 2.0 + alpha / jnp.sqrt(jnp.pi)) * Q_sq * qqr2e


def bks_silica_pair(displacement_or_metric: DisplacementOrMetricFn,
                    species: Array,
                    cutoff: float=8.0):
  """Convenience wrapper to compute :ref:`BKS energy <bks-pot>` for SiO2."""
  bks_pair_fn = bks_pair(displacement_or_metric,
                         species,
                         cutoff=cutoff,
                         **BKS_SILICA_DICT)
  N_0 = jnp.sum(species==0)
  N_1 = jnp.sum(species==1)

  e_self = partial(_bks_silica_self, alpha=0.25, cutoff=cutoff)

  def energy_fn(R, **kwargs):
    return (bks_pair_fn(R, **kwargs) +
            N_0 * e_self(CHARGE_SILICON**2) +
            N_1 * e_self(CHARGE_OXYGEN**2))


  return energy_fn


def bks_silica_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: Box,
    species: Array,
    cutoff: float = 8.0,
    dr_threshold: float = 1.0,
    fractional_coordinates: bool=False,
    format: partition.NeighborListFormat=partition.OrderedSparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`BKS energy <bks-pot>` using neighbor lists."""
  kwargs = {**BKS_SILICA_DICT, **neighbor_kwargs}
  neighbor_fn, bks_pair_fn = bks_neighbor_list(
    displacement_or_metric,
    box_size,
    species,
    cutoff=cutoff,
    dr_threshold=dr_threshold,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **kwargs)
  N_0 = jnp.sum(species==0)
  N_1 = jnp.sum(species==1)

  e_self = partial(_bks_silica_self, alpha=0.25, cutoff=cutoff)

  def energy_fn(R, neighbor, **kwargs):
    return (bks_pair_fn(R, neighbor, **kwargs) +
            N_0 * e_self(CHARGE_SILICON ** 2) +
            N_1 * e_self(CHARGE_OXYGEN ** 2))

  return neighbor_fn, energy_fn


# Stillinger-Weber Potential


def _sw_angle_interaction(gamma: float, sigma: float, cutoff: float,
                          dR12: Array, dR13: Array) -> Array:
  """The angular interaction for the Stillinger-Weber potential.
  This function is defined only for interaction with a pair of
  neighbors. We then vmap this function three times below to make
  it work on the whole system of atoms.
  Args:
    gamma: A scalar used to fit the angle interaction.
    sigma: A scalar that sets the distance scale between neighbors.
    cutoff: The cutoff beyond which the interactions are not
      considered. The default value should not be changed for the
      default SW potential.
    dR12: A d-dimensional vector that specifies the displacement
      of the first neighbor. This potential is usually used in three
      dimensions.
    dR13: A d-dimensional vector that specifies the displacement
      of the second neighbor.

  Returns:
    Angular interaction energy for one pair of neighbors.
  """
  a = cutoff / sigma
  dr12 = space.distance(dR12)
  dr13 = space.distance(dR13)
  dr12 = jnp.where(dr12 < cutoff, dr12, 0)
  dr13 = jnp.where(dr13 < cutoff, dr13, 0)
  term1 = jnp.exp(gamma / (dr12 / sigma - a) + gamma / (dr13 / sigma - a))
  cos_angle = quantity.cosine_angle_between_two_vectors(dR12, dR13)
  term2 = (cos_angle + 1./3)**2
  within_cutoff = (dr12>0) & (dr13>0) & (jnp.linalg.norm(dR12-dR13)>1e-5)
  return jnp.where(within_cutoff, term1 * term2, 0)
sw_three_body_term = vmap(vmap(vmap(
    _sw_angle_interaction, (0, None)), (None, 0)), 0)


def _sw_radial_interaction(sigma: float, B: float, cutoff: float, r: Array
                           ) -> Array:
  """The two body term of the Stillinger-Weber potential."""
  a = cutoff / sigma
  p = 4
  term1 = (B * (r / sigma)**(-p) - 1.0)
  within_cutoff = (r > 0) & (r < cutoff)
  r = jnp.where(within_cutoff, r, 0)
  term2 = jnp.exp(1 / (r / sigma - a))
  return jnp.where(within_cutoff, term1 * term2, 0.0)


def stillinger_weber(displacement: DisplacementFn,
                     sigma: float = 2.0951,
                     A: float = 7.049556277,
                     B: float = 0.6022245584,
                     lam: float = 21.0,
                     gamma: float = 1.2,
                     epsilon: float = 2.16826,
                     three_body_strength: float =1.0,
                     cutoff: float = 3.77118) -> Callable[[Array], Array]:
  """.. _sw-pot:

  Computes the Stillinger-Weber potential.

  The Stillinger-Weber (SW) potential [#stillinger]_ which is commonly used to
  model silicon and similar systems. This function uses the default SW
  parameters from the original paper. The SW potential was originally proposed
  to model diamond in the diamond crystal phase and the liquid phase, and is
  known to give unphysical amorphous configurations [#holender]_ [#barkema]_ .
  For this reason, we provide a `three_body_strength` parameter. Changing this
  number to `1.5` or `2.0` has been know to produce more physical amorphous
  phase, preventing most atoms from having more than four nearest neighbors.
  Note that this function currently assumes nearest-image-convention.

  Args:
    displacement: The displacement function for the space.
    sigma: A scalar that sets the distance scale between neighbors.
    A: A scalar that determines the scale of two-body term.
    B: A scalar that determines the scale of the :math:`1 / r^p` term.
    lam: A scalar that determines the scale of the three-body term.
    epsilon: A scalar that sets the total energy scale.
    gamma: A scalar used to fit the angle interaction.
    three_body_strength:
      A scalar that determines the relative strength
      of the angular interaction. Default value is `1.0`, which works well
      for the diamond crystal and liquid phases. `1.5` and `2.0` have been used
      to model amorphous silicon.
  Returns:
    A function that computes the total energy.

  .. rubric:: References
  .. [#stillinger] Stillinger, Frank H., and Thomas A. Weber. "Computer
    simulation of local order in condensed phases of silicon."
    Physical review B 31.8 (1985): 5262.
  .. [#holender] Holender, J. M., and G. J. Morgan. "Generation of a large
    structure (105 atoms) of amorphous Si using molecular dynamics." Journal of
    Physics: Condensed Matter 3.38 (1991): 7241.
  .. [#barkema] Barkema, G. T., and Normand Mousseau. "Event-based relaxation of
    continuous disordered systems." Physical review letters 77.21 (1996): 4358.
  """
  two_body_fn = partial(_sw_radial_interaction, sigma, B, cutoff)
  three_body_fn = partial(_sw_angle_interaction, gamma, sigma, cutoff)
  three_body_fn = vmap(vmap(vmap(three_body_fn, (0, None)), (None, 0)))

  def compute_fn(R, **kwargs):
    d = partial(displacement, **kwargs)
    dR = space.map_product(d)(R, R)
    dr = space.distance(dR)
    first_term = util.high_precision_sum(two_body_fn(dr)) / 2.0 * A
    second_term = lam *  util.high_precision_sum(three_body_fn(dR, dR)) / 2.0
    return epsilon * (first_term + three_body_strength * second_term)
  return compute_fn


def stillinger_weber_neighbor_list(
    displacement: DisplacementFn,
    box_size: float,
    sigma: float = 2.0951,
    A: float = 7.049556277,
    B: float = 0.6022245584,
    lam: float = 21.0,
    gamma: float = 1.2,
    epsilon: float = 2.16826,
    three_body_strength:float = 1.0,
    cutoff: float = 3.77118,
    dr_threshold: float = 0.5,
    fractional_coordinates: bool=False,
    format: NeighborListFormat=partition.Dense,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`Stillinger-Weber <sw-pot>`
  using a neighbor list.
  """
  two_body_fn = partial(_sw_radial_interaction, sigma, B, cutoff)
  three_body_fn = partial(_sw_angle_interaction, gamma, sigma, cutoff)

  neighbor_fn = partition.neighbor_list(displacement,
                                        box_size,
                                        cutoff,
                                        dr_threshold,
                                        format=format,
                                        **neighbor_kwargs)

  def compute_fn(R, neighbor, **kwargs):
    d = partial(displacement, **kwargs)
    mask = partition.neighbor_list_mask(neighbor)

    if neighbor.format is partition.Dense:
      _three_body_fn = vmap(vmap(vmap(three_body_fn, (0, None)), (None, 0)))
      dR = space.map_neighbor(d)(R, R[neighbor.idx])
      dr = space.distance(dR)
      first_term = util.high_precision_sum(two_body_fn(dr) * mask) / 2.0 * A
      mask_ijk = mask[:, None, :] * mask[:, :, None]
      second_term = lam *  util.high_precision_sum(
        _three_body_fn(dR, dR) * mask_ijk) / 2.0
    else:
      raise NotImplementedError('Stillinger-Weber potential only implemented '
                                'with Dense neighbor lists.')

    return epsilon * (first_term + three_body_strength * second_term)
  return neighbor_fn, compute_fn


# Tersoff model


def load_lammps_tersoff_parameters(file: TextIO) -> Array:
  """.. _ts-lammps:

  Reads Tersoff parameters from a LAMMPS file and returns parameter tables.

  This function reads multi-element original Tersoff potential parameters
  from a file.

  Args:
    file: A parameter file that is written with lammps format.

  Returns:
    params: An array that contains Tersoff parameters.
  """
  # start to read file.
  # todo: params_per_line becomes input variables.
  #       depending on the various type of tersoff model.
  params = []
  params_per_line = 17

  # read parameters.
  # skip if the line has \# or empty
  # if the number of parameters in one line is less than params_per_line,
  # additional line is appended to match.
  skip = False
  for line in file.read().split('\n'):
    words = line.strip().split()
    nwords = len(words)
    if '#' in words or nwords == 0:
      continue

    if nwords < params_per_line and skip is False:
      line_keep = line
      skip = True
      continue

    line_keep += ' ' + line
    words = line_keep.strip().split()
    nwords = len(words)

    if nwords != params_per_line:
      raise ValueError('Incorrect format: %d not in %d'
        % (nwords, params_per_line))
    else:
      skip = False

    words[3:] = f32(words[3:])
    params.append({
        'element1': words[0],
        'element2': words[1],
        'element3': words[2],
        'mTf': words[3],
        'gamma': words[4],
        'lam3': words[5],
        'cTf': words[6],
        'dTf': words[7],
        'hTf': words[8],
        'nTf': words[9],
        'beta': words[10],
        'lam2': words[11],
        'B': words[12],
        'R': words[13],
        'D': words[14],
        'lam1': words[15],
        'A': words[16],
    })
  return params

def _ters_cutoff(dr, R, D) -> Array:
  """The cut-off function of the Tersoff potential.
  Args:
    R: A Parameter that is the average of inner and outer cutoff radii
    D: A Parameter that is the half of the difference
       between inner and outer cutoff radii

  Returns:
    cut-off values
  """
  outer = jnp.where(dr < R + D,
                    0.5*(1.0 - jnp.sin(jnp.pi/2*(dr - R)/D)),
                    0)
  inner = jnp.where(dr < R - D, 1, outer)
  return inner

def _ters_bij(R, D, c, d, h, lam3, beta, n, m,
              dRij, dRik, mask_ijk) -> Array:
  """The bond-order term of the Tersoff potential.
  Args:
    # parameters for cut-off functions
    R: A Parameter that is the average of inner and outer cutoff radii
    D: A Parameter that is the half of the difference
       between inner and outer cutoff radii

    # parameters related to the angle Penalty function in the bond-order
    # function
    # h(\theta) = 1 + c^2/d^2 + c^2/(d^2 + (h - cos(\theta)^2))
    c: A Parameter that determines angle penalty
    d: A Parameter that determines angle penalty
    h: A cosine value that is a desirable angle between 3 atoms.

    # parameters related to the distance penalty function in the bond-order
    # function
    lam3: A Parameter that determines distance penalty value
    m: A Parameter that determines distance penalty value

    # parameters related to the bond-order function
    beta: A Parameter that determines bond-order value
    n: A Parameter that determines bond-order value

    dRij: A ndarray of shape [n, neighbors, dim] of pairwise distances between
     particles
    dRik: A ndarray of shape [n, neighbors, dim] of pairwise distances between
      particles TODO - Currently, it is the same as the dRij


  Returns:
    Bond-order values between i and j atoms
  """
  drij = space.distance(dRij)
  drik = space.distance(dRik)

  # compute g_ijk - angle penalty value
  costheta = quantity.cosine_angles(dRij)
  gijk = 1.0 + (c**2 / d**2) - (c**2 / (d**2 + (h - costheta)**2))
  gijk *= mask_ijk
  gijk = vmap(lambda x: x.at[jnp.diag_indices(x.shape[0])].set(0))(gijk)

  # compute exponential term - distance penalty value
  dr_diff = vmap(vmap(jnp.subtract, (None, 0)))(drij, drik)
  explr3 = jnp.exp(lam3**m * dr_diff**m)
  explr3 *= mask_ijk
  explr3 = vmap(lambda x: x.at[jnp.diag_indices(x.shape[0])].set(0))(explr3)

  # compute fC with dr_ik
  fC = _ters_cutoff(drik, R, D)

  # compute zeta without diagonal term
  tmp = jnp.multiply(gijk, explr3)
  zeta_ij = jnp.einsum('ik,ikj->ij', fC, tmp)
  bij = (1 + (beta*zeta_ij)**n)**(-1 / 2 / n)

  return bij

def _ters_attractive(B: f64, lam2: f64,
                     R: f64, D: f64,
                     c: f64, d: f64, h: f64,
                     lam3: f64, beta: f64, n: f64, m: f64,
                     dR12: Array, dR13: Array,
                     mask_ijk) -> Array:
  """The attractive term of the Tersoff potential.
  Args:
    dR12: A ndarray of shape [n, neighbors, dim] of pairwise distnaces between
      particles.
    dR13: A ndarray of shape [n, neighbors, dim] of pairwise distnaces between
      particles. TODO - Currently, it is the same as the dR12
    R: A Parameter that is the average of inner and outer cutoff radii.
    D: A Parameter that is the half of the difference.
       between inner and outer cutoff radii.

    # parameters related to the angle Penalty function in the bond-order
    # function.
    # h(\theta) = 1 + c^2/d^2 + c^2/(d^2 + (h - cos(\theta)^2))
    c: A Parameter that determines angle penalty.
    d: A Parameter that determines angle penalty.
    h: A cosine value that is a desirable angle between 3 atoms.

    # parameters related to the distance penalty function in the bond-order
    # function.
    lam3: A Parameter that determines distance penalty value.
    m: A Parameter that determines distance penalty value.

    # parameters related to the bond-order function
    beta: A Parameter that determines bond-order value.
    n: A Parameter that determines bond-order value.

  Returns:
    Attractive interaction energy for one pair of neighbors.
  """

  dr12 = space.distance(dR12)
  fC = _ters_cutoff(dr12, R, D)
  fA = -B*jnp.exp(-lam2*dr12)
  bij = _ters_bij(R, D, c, d, h, lam3, beta, n, m,
                  dR12, dR13, mask_ijk)
  return 0.5*fC*bij*fA

def _ters_repulsive(A: f64, lam1: f64, R: f64, D: f64,
                    dr: Array) -> Array:
  """The repulsive term of the Tersoff potential.
  Args:
    A: A scalar that determines repulsive energy (eV).
    lam1: A scalar that determines the scale two-body distance (Angstrom).
    R: A scalar that is the average of inner and outer cutoff radii.
    D: A scalar that is the half of the difference
       between inner and outer cutoff radii.
  Returns:
    Repulsive interaction energy for one pair of neighbors.
  """

  fC = _ters_cutoff(dr, R, D)
  fR = A * jnp.exp(-lam1 * dr)
  return 0.5 * fC * fR

def tersoff(displacement: DisplacementFn,
            params: Array,
            species: Optional[Array]=None,
            ) -> Callable[[Array], Array]:
  """Computes the Tersoff potential.

  The Tersoff potential [1] which is commonly used to model
  semiconducting materials. The Tersoff potential was originally proposed to
  model various types of lattice with a simple functional form.
  For this reason, Tersoff model was introduced bond-order function
  to determine the strength of repulsive and attractive forces between atoms.

  Args:
    displacement: The displacement function for the space.
    params: A dictionary of parameters for the tersoff potential. Usually this
      should be loaded from lammps using the
      :ref:`load_lammps_tersoff_parameters <ts-lammps>` function.
    species: An array of species. Currently only `None` is supported.

  Returns:
    A function that computes the total energy.

  [1] J. Tersoff "New empirical approach for the structure and energy of
  covalent systems" Physical review B 37.12 (1988): 6991.
  """
  # check number of parameters set.
  if species is None:
    params = params[0]
  else:
    raise NotImplementedError('Multiple species is not implemented yet.'
                              ' Please raise an issue if this is important for'
                              ' you.')

  # define a repulsive and an attractive function with given parameters.
  repulsive_fn = partial(_ters_repulsive,
                         params['A'], params['lam1'],
                         params['R'], params['D'])
  attractive_fn = partial(_ters_attractive,
                          params['B'], params['lam2'],
                          params['R'], params['D'],
                          params['cTf'], params['dTf'], params['hTf'],
                          params['lam3'], params['beta'],
                          params['nTf'], params['mTf'])

  # define compute functions.
  def compute_fn(R, **kwargs):
    d = partial(displacement, **kwargs)
    dR = space.map_neighbor(d)(R, R)
    dr = space.distance(dR)
    N = R.shape[0]
    neighbor = jnp.broadcast_to(jnp.arange(N)[None, :], (N, N))
    mask = partition.neighbor_list_mask(neighbor, mask_self=True)
    mask_ijk = mask[:, None, :] * mask[:, :, None]
    first_term = util.high_precision_sum(repulsive_fn(dr) * mask)
    second_term = util.high_precision_sum(attractive_fn(dR, dR, mask_ijk)
                                          * mask)
    return first_term + second_term
  return compute_fn

  return compute_fn

def tersoff_neighbor_list(displacement: DisplacementFn,
                          box_size: float,
                          params: Array,
                          species: Optional[Array]=None,
                          dr_threshold: float=0.5,
                          fractional_coordinates: bool=True,
                          format: NeighborListFormat=partition.Dense,
                          **neighbor_kwargs
                          ) -> Tuple[NeighborFn,
                               Callable[[Array, NeighborList], Array]]:
  """Computes the Tersoff potential.

  The Tersoff potential [1] which is commonly used to model
  semiconducting materials. The Tersoff potential was originally proposed to
  model various types of lattice with a simple functional form.
  For this reason, Tersoff model was introduced bond-order function
  to determine the strength of repulsive and attractive forces between atoms.

  Args:
    displacement: The displacement function for the space.
    box_size: A float or vector specifying the size of the simulation box.
    params: A dictionary of parameters for the tersoff potential. Usually this
      should be loaded from lammps using the
      :ref:`load_lammps_tersoff_parameters <ts-lammps>` function.
    species: An array of species. Currently only `None` is supported.
    dr_threshold: A distance threshold that controls how often the neighor list
      is recomputed.
    fractional_coordinates: A boolean specifying whether coordinates are stored
      in the unit cube.
    format: Format of the neighbor list.

  Returns:
    A pair of functions. One that builds the neighbor list and one that
    computes the total energy.

  [1] J. Tersoff "New empirical approach for the structure and energy of
  covalent systems" Physical review B 37.12 (1988): 6991.
  """
  # check number of parameters set
  if species is None:
    params = params[0]
    nparams = 1
  else:
    raise NotImplementedError('Multiple species were not implemented yet.')

  # define a repulsive and an attractive function with given parameters
  repulsive_fn = partial(_ters_repulsive,
                         params['A'], params['lam1'],
                         params['R'], params['D'])
  attractive_fn = partial(_ters_attractive,
                          params['B'], params['lam2'],
                          params['R'], params['D'],
                          params['cTf'], params['dTf'], params['hTf'],
                          params['lam3'], params['beta'],
                          params['nTf'], params['mTf'])

  # define a neighbor function.
  # TODO: other neighbor list construction method will be implemented.
  if format is partition.Dense:
    neighbor_fn = partition.neighbor_list(
      displacement,
      box_size,
      params['R'] + params['D'],
      dr_threshold,
      disable_cell_list=True,
      fractional_coordinates=fractional_coordinates,
      format=format,
      **neighbor_kwargs)
  else:
    raise NotImplementedError('Tersoff potential only implemented '
                                'with Dense neighbor lists.')

  # define compute functions
  def compute_fn(R, neighbor, **kwargs):
    d = partial(displacement, **kwargs)
    mask = partition.neighbor_list_mask(neighbor, mask_self=True)
    mask_ijk = mask[:, None, :] * mask[:, :, None]

    dR = space.map_neighbor(d)(R, R[neighbor.idx])
    dr = space.distance(dR)
    first_term = util.high_precision_sum(repulsive_fn(dr) * mask)
    second_term = util.high_precision_sum(attractive_fn(dR, dR, mask_ijk)
                                          * mask)
    return first_term + second_term
  return neighbor_fn, compute_fn


def tersoff_from_lammps_parameters_neighbor_list(
    displacement: DisplacementFn,
    box_size: float,
    f: TextIO,
    dr_threshold: float=0.5,
    fractional_coordinates=True,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute Tersoff energy with LAMMPS parameters."""
  return tersoff_neighbor_list(displacement,
                               box_size,
                               load_lammps_tersoff_parameters(f),
                               dr_threshold=dr_threshold,
                               fractional_coordinates=fractional_coordinates,
                               **neighbor_kwargs)


# Embedded Atom Method


def load_lammps_eam_parameters(file: TextIO) -> Tuple[Callable[[Array], Array],
                                                      Callable[[Array], Array],
                                                      Callable[[Array], Array],
                                                      float]:
  """Reads EAM parameters from a LAMMPS file and returns relevant spline fits.

  This function reads single-element EAM potential fit parameters from a file
  in DYNAMO funcl format. In summary, the file contains:

  * Line 1-3: Comments
  * Line 4: Number of elements and the element type
  * Line 5: The number of charge values that the embedding energy is evaluated
    on (`num_drho`), interval between the charge values (`drho`), the number of
    distances the pairwise energy and the charge density is evaluated on (`num_dr`),
    the interval between these distances (`dr`), and the cutoff distance (`cutoff`).

  The lines that come after are the embedding function evaluated on `num_drho`
  charge values, charge function evaluated at `num_dr` distance values, and
  pairwise energy evaluated at `num_dr` distance values. Note that the pairwise
  energy is multiplied by distance (in units of eV x Angstroms).

  For more details of the DYNAMO file format, see:
  https://sites.google.com/a/ncsu.edu/cjobrien/tutorials-and-guides/eam

  Args:
    f: File handle for the EAM parameters text file.

  Returns:
    A tuple containing three functions and a cutoff distance.

    charge_fn:
      A function that takes an ndarray of shape `[n, m]` of distances
      between particles and returns a matrix of charge contributions.
    embedding_fn:
      Function that takes an ndarray of shape `[n]` of charges and
      returns an ndarray of shape `[n]` of the energy cost of embedding an atom
      into the charge.
    pairwise_fn:
      A function that takes an ndarray of shape `[n, m]` of distances
      and returns an ndarray of shape `[n, m]` of pairwise energies.
    cutoff:
      Cutoff distance for the `embedding_fn` and `pairwise_fn`.
  """
  raw_text = file.read().split('\n')
  if 'setfl' not in raw_text[0]:
    raise ValueError('File format is incorrect, expected LAMMPS setfl format.')
  temp_params = raw_text[4].split()
  num_drho, num_dr = int(temp_params[0]), int(temp_params[2])
  drho, dr, cutoff = float(temp_params[1]), float(temp_params[3]), float(
      temp_params[4])
  data = maybe_downcast([float(i) for i in raw_text[6:-1]])
  embedding_fn = interpolate.spline(data[:num_drho], drho)
  charge_fn = interpolate.spline(data[num_drho:num_drho + num_dr], dr)
  # LAMMPS EAM parameters file lists pairwise energies after multiplying by
  # distance, in units of eV*Angstrom. We divide the energy by distance below,
  distances = jnp.arange(num_dr) * dr
  # Prevent dividing by zero at zero distance, which will not
  # affect the calculation
  distances = jnp.where(distances == 0, f32(0.001), distances)
  pairwise_fn = interpolate.spline(
      data[num_dr + num_drho:num_drho + 2 * num_dr] / distances,
      dr)
  return charge_fn, embedding_fn, pairwise_fn, cutoff


def eam(displacement: DisplacementFn,
        charge_fn: Callable[[Array], Array],
        embedding_fn: Callable[[Array], Array],
        pairwise_fn: Callable[[Array], Array],
        axis: Optional[Tuple[int, ...]]=None) -> Callable[[Array], Array]:
  """.. _eam-pot:

  Interatomic potential as approximated by embedded atom model (EAM).

  This code implements the EAM approximation to interactions between metallic
  atoms. In EAM, the potential energy of an atom is given by two terms: a
  pairwise energy and an embedding energy due to the interaction between the
  atom and background charge density. The EAM potential for a single atomic
  species is often determined by three functions:

  1) Charge density contribution of an atom as a function of distance.
  2) Energy of embedding an atom in the background charge density.
  3) Pairwise energy.

  These three functions are usually provided as spline fits, and we follow the
  implementation and spline fits given by Mishin et al. [#mishin]_
  Note that in current implementation, the three functions listed above
  can also be expressed by a any function with the correct signature,
  including neural networks.

  Args:
    displacement: A function that produces an ndarray of shape `[n, m,
      spatial_dimension]` of particle displacements from particle positions
      specified as an ndarray of shape `[n, spatial_dimension]` and `[m,
      spatial_dimension]` respectively.
    box_size: The size of the simulation box.
    charge_fn: A function that takes an ndarray of shape `[n, m]` of distances
      between particles and returns a matrix of charge contributions.
    embedding_fn: Function that takes an ndarray of shape `[n]` of charges and
      returns an ndarray of shape `[n]` of the energy cost of embedding an atom
      into the charge.
    pairwise_fn: A function that takes an ndarray of shape `[n, m]` of distances
      and returns an ndarray of shape `[n, m]` of pairwise energies.
    cutoff: A float specifying the maximum interaction distance.
    dr_threshold: A float specifying the halo in the neighbor list.
    axis: Specifies which axis the total energy should be summed over.
    fractional_coordinates: A boolean specifying whether or not the coordinates
      will be in the unit cube.
    format: The format of the neighbor list.

  Returns:
    A tuple containing a function to build the neighbor list and function that
    computes the EAM energy of a set of atoms with positions given by an
    `[n, spatial_dimension]` ndarray.

  .. rubric:: References
  .. [#mishin] Y. Mishin, D. Farkas, M.J. Mehl, DA Papaconstantopoulos, "Interatomic
    potentials for monoatomic metals from experimental data and ab initio
    calculations." Physical Review B, 59 (1999)
  """
  metric = space.map_product(space.metric(displacement))

  def energy(R, **kwargs):
    dr = metric(R, R, **kwargs)
    total_charge = util.high_precision_sum(charge_fn(dr), axis=1)
    embedding_energy = embedding_fn(total_charge)
    pairwise_energy = util.high_precision_sum(smap._diagonal_mask(
        pairwise_fn(dr)), axis=1) / f32(2.0)
    return util.high_precision_sum(
        embedding_energy + pairwise_energy, axis=axis)

  return energy


def eam_from_lammps_parameters(displacement: DisplacementFn,
                               f: TextIO) -> Callable[[Array], Array]:
  """Convenience wrapper to compute :ref:`EAM energy <eam-pot>` with LAMMPS parameters."""
  return eam(displacement, *load_lammps_eam_parameters(f)[:-1])


def eam_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: float,
    charge_fn: Callable[[Array], Array],
    embedding_fn: Callable[[Array], Array],
    pairwise_fn: Callable[[Array], Array],
    cutoff: float,
    dr_threshold: float = 0.5,
    axis: Optional[Tuple[int, ...]] = None,
    fractional_coordinates: bool = True,
    format: partition.NeighborListFormat = partition.Sparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`EAM <eam-pot>` using a neighbor list."""
  metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

  neighbor_fn = partition.neighbor_list(displacement_or_metric,
                                        box_size,
                                        cutoff,
                                        dr_threshold,
                                        mask_self=False,
                                        format=format,
                                        **neighbor_kwargs)

  def energy_fn(R, neighbor, **kwargs):
    mask = partition.neighbor_list_mask(neighbor)
    self_mask = partition.neighbor_list_mask(neighbor, mask_self=True)

    if neighbor.format is partition.Dense:
      dr = space.map_neighbor(metric)(R, R[neighbor.idx], **kwargs)
      total_charge = util.high_precision_sum(charge_fn(dr) * mask, axis=1)
      embedding_energy = embedding_fn(total_charge)
      pairwise_energy = util.high_precision_sum(pairwise_fn(dr) * self_mask,
                                                axis=1)
    elif neighbor.format is partition.Sparse:
      N = len(R)
      dr = space.map_bond(metric)(R[neighbor.idx[0]], R[neighbor.idx[1]],
                                  **kwargs)
      total_charge = ops.segment_sum(charge_fn(dr) * mask, neighbor.idx[0], N)
      embedding_energy = embedding_fn(total_charge)
      pairwise_energy = ops.segment_sum(pairwise_fn(dr) * self_mask,
                                        neighbor.idx[0], N)
    else:
      raise NotImplementedError('EAM potential not implemented for '
                                'OrderedSparse neighbor lists.')

    return util.high_precision_sum(
      embedding_energy + pairwise_energy / 2.0, axis=axis)

  return neighbor_fn, energy_fn


def eam_from_lammps_parameters_neighbor_list(
    displacement: DisplacementFn,
    box_size, float,
    f: TextIO,
    axis=None,
    dr_threshold: float=0.5,
    fractional_coordinates=True,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`EAM energy <eam-pot>`
  with parameters from LAMMPS using a neighbor list.."""
  return eam_neighbor_list(displacement,
                           box_size,
                           *load_lammps_eam_parameters(f),
                           dr_threshold=dr_threshold,
                           **neighbor_kwargs)


def behler_parrinello(displacement: DisplacementFn,
                      species: Optional[Array]=None,
                      mlp_sizes: Tuple[int, ...]=(30, 30),
                      mlp_kwargs: Optional[Dict[str, Any]]=None,
                      sym_kwargs: Optional[Dict[str, Any]]=None,
                      per_particle: bool=False
                      ) -> Tuple[nn.InitFn,
                                 Callable[[PyTree, Array], Array]]:
  if sym_kwargs is None:
    sym_kwargs = {}
  if mlp_kwargs is None:
    mlp_kwargs = {
        'activation': jnp.tanh
    }

  sym_fn = nn.behler_parrinello_symmetry_functions(displacement,
                                                   species,
                                                   **sym_kwargs)

  @hk.without_apply_rng
  @hk.transform
  def model(R, **kwargs):
    embedding_fn = hk.nets.MLP(output_sizes=mlp_sizes+(1,),
                               activate_final=False,
                               name='BPEncoder',
                               **mlp_kwargs)
    embedding_fn = vmap(embedding_fn)
    sym = sym_fn(R, **kwargs)
    readout = embedding_fn(sym)
    if per_particle:
      return readout
    return jnp.sum(readout)
  return model.init, model.apply


def behler_parrinello_neighbor_list(
    displacement: DisplacementFn,
    box_size: float,
    species: Optional[Array]=None,
    mlp_sizes: Tuple[int, ...]=(30, 30),
    mlp_kwargs: Optional[Dict[str, Any]]=None,
    sym_kwargs: Optional[Dict[str, Any]]=None,
    dr_threshold: float=0.5,
    fractional_coordinates: bool=False,
    format: partition.NeighborListFormat=partition.Sparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, nn.InitFn, Callable[[PyTree, Array, NeighborList],
                                               Array]]:
  if sym_kwargs is None:
    sym_kwargs = {}
  if mlp_kwargs is None:
    mlp_kwargs = {
        'activation': jnp.tanh
    }

  cutoff_distance = 8.0
  if 'cutoff_distance' in sym_kwargs:
    cutoff_distance = sym_kwargs['cutoff_distance']

  neighbor_fn = partition.neighbor_list(
    displacement,
    box_size,
    cutoff_distance,
    dr_threshold,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **neighbor_kwargs)

  sym_fn = nn.behler_parrinello_symmetry_functions_neighbor_list(displacement,
                                                                 species,
                                                                 **sym_kwargs)

  @hk.without_apply_rng
  @hk.transform
  def model(R, neighbor, **kwargs):
    embedding_fn = hk.nets.MLP(output_sizes=mlp_sizes+(1,),
                               activate_final=False,
                               name='BPEncoder',
                               **mlp_kwargs)
    embedding_fn = vmap(embedding_fn)
    sym = sym_fn(R, neighbor, **kwargs)
    readout = embedding_fn(sym)
    return jnp.sum(readout)
  return neighbor_fn, model.init, model.apply


class EnergyGraphNet(hk.Module):
  """Implements a Graph Neural Network for energy fitting.

  This model uses a GraphNetEmbedding combined with a decoder applied to the
  global state.
  """
  def __init__(self,
               n_recurrences: int,
               mlp_sizes: Tuple[int, ...],
               mlp_kwargs: Optional[Dict[str, Any]]=None,
               format: partition.NeighborListFormat=partition.Dense,
               name: str='Energy'):
    super(EnergyGraphNet, self).__init__(name=name)

    if mlp_kwargs is None:
      mlp_kwargs = {
        'w_init': hk.initializers.VarianceScaling(),
        'b_init': hk.initializers.VarianceScaling(0.1),
        'activation': jax.nn.softplus
      }
    self._format = format
    self._graph_net = nn.GraphNetEncoder(n_recurrences,
                                         mlp_sizes,
                                         mlp_kwargs,
                                         format)
    self._decoder = hk.nets.MLP(output_sizes=mlp_sizes + (1,),
                                activate_final=False,
                                name='GlobalDecoder',
                                **mlp_kwargs)

  def __call__(self, graph: nn.GraphsTuple) -> jnp.ndarray:
    output = self._graph_net(graph)
    output = jnp.squeeze(self._decoder(output.globals), axis=-1)
    if self._format is partition.Sparse:
      output = output[0]
    return output


def _canonicalize_node_state(nodes: Optional[Array]) -> Optional[Array]:
  if nodes is None:
    return nodes

  if nodes.ndim == 1:
    nodes = nodes[:, jnp.newaxis]

  if nodes.ndim != 2:
    raise ValueError(
      'Nodes must be a [N, node_dim] array. Found {}.'.format(nodes.shape))

  return nodes


def graph_network(displacement_fn: DisplacementFn,
                  r_cutoff: float,
                  nodes: Optional[Array]=None,
                  n_recurrences: int=2,
                  mlp_sizes: Tuple[int, ...]=(64, 64),
                  mlp_kwargs: Optional[Dict[str, Any]]=None
                  ) -> Tuple[nn.InitFn,
                             Callable[[PyTree, Array], Array]]:
  """Convenience wrapper around EnergyGraphNet model.

  Args:
    displacement_fn: Function to compute displacement between two positions.
    r_cutoff: A floating point cutoff; Edges will be added to the graph
      for pairs of particles whose separation is smaller than the cutoff.
    nodes: None or an ndarray of shape `[N, node_dim]` specifying the state
      of the nodes. If None this is set to the zeros vector. Often, for a
      system with multiple species, this could be the species id.
    n_recurrences: The number of steps of message passing in the graph network.
    mlp_sizes: A tuple specifying the layer-widths for the fully-connected
      networks used to update the states in the graph network.
    mlp_kwargs: A dict specifying args for the fully-connected networks used to
      update the states in the graph network.

  Returns:
    A tuple of functions. An `params = init_fn(key, R)` that instantiates the
    model parameters and an `E = apply_fn(params, R)` that computes the energy
    for a particular state.
  """

  nodes = _canonicalize_node_state(nodes)

  @hk.without_apply_rng
  @hk.transform
  def model(R: Array, **kwargs) -> Array:
    N = R.shape[0]

    d = partial(displacement_fn, **kwargs)
    d = space.map_product(d)
    dR = d(R, R)

    dr_2 = space.square_distance(dR)

    if 'nodes' in kwargs:
      _nodes = _canonicalize_node_state(kwargs['nodes'])
    else:
      _nodes = jnp.zeros((N, 1), R.dtype) if nodes is None else nodes

    edge_idx = jnp.broadcast_to(jnp.arange(N)[jnp.newaxis, :], (N, N))
    edge_idx = jnp.where(dr_2 < r_cutoff ** 2, edge_idx, N)

    _globals = jnp.zeros((1,), R.dtype)

    net = EnergyGraphNet(n_recurrences, mlp_sizes, mlp_kwargs)
    return net(nn.GraphsTuple(_nodes, dR, _globals, edge_idx))  # pytype: disable=wrong-arg-count

  return model.init, model.apply


def graph_network_neighbor_list(
    displacement_fn: DisplacementFn,
    box_size: Box,
    r_cutoff: float,
    dr_threshold: float,
    nodes: Optional[Array]=None,
    n_recurrences: int=2,
    mlp_sizes: Tuple[int, ...]=(64, 64),
    mlp_kwargs: Optional[Dict[str, Any]]=None,
    fractional_coordinates: bool=False,
    format: partition.NeighborListFormat=partition.Sparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, nn.InitFn, Callable[[PyTree, Array, NeighborList],
                                               Array]]:
  """Convenience wrapper around EnergyGraphNet model using neighbor lists.

  Args:
    displacement_fn: Function to compute displacement between two positions.
    box_size: The size of the simulation volume, used to construct neighbor
      list.
    r_cutoff: A floating point cutoff; Edges will be added to the graph
      for pairs of particles whose separation is smaller than the cutoff.
    dr_threshold: A floating point number specifying a "halo" radius that we use
      for neighbor list construction. See `neighbor_list` for details.
    nodes: None or an ndarray of shape `[N, node_dim]` specifying the state
      of the nodes. If None this is set to the zeroes vector. Often, for a
      system with multiple species, this could be the species id.
    n_recurrences: The number of steps of message passing in the graph network.
    mlp_sizes: A tuple specifying the layer-widths for the fully-connected
      networks used to update the states in the graph network.
    mlp_kwargs: A dict specifying args for the fully-connected networks used to
      update the states in the graph network.
    fractional_coordinates: A boolean specifying whether or not the coordinates
      will be in the unit cube.
    format: The format of the neighbor list. See `partition.NeighborListFormat`
      for details. Only `Dense` and `Sparse` formats are accepted. If the `Dense`
      format is used, then the graph network is constructed using the JAX MD
      backend, otherwise Jraph is used.

  Returns:
    A pair of functions. An `params = init_fn(key, R)` that instantiates the
    model parameters and an `E = apply_fn(params, R)` that computes the energy
    for a particular state.
  """

  nodes = _canonicalize_node_state(nodes)

  @hk.without_apply_rng
  @hk.transform
  def model(R, neighbor, **kwargs):
    N = R.shape[0]
    d = partial(displacement_fn, **kwargs)

    if 'nodes' in kwargs:
      _nodes = _canonicalize_node_state(kwargs['nodes'])
    else:
      _nodes = jnp.zeros((N, 1), R.dtype) if nodes is None else nodes

    _globals = jnp.zeros((1,), R.dtype)

    if format is partition.Dense:
      d = space.map_neighbor(d)
      R_neigh = R[neighbor.idx]
      dR = d(R, R_neigh)

      dr_2 = space.square_distance(dR)
      edge_idx = jnp.where(dr_2 < r_cutoff ** 2, neighbor.idx, N)
      graph = nn.GraphsTuple(_nodes, dR, _globals, edge_idx)
    else:
      d = space.map_bond(d)
      dR = d(R[neighbor.idx[0]], R[neighbor.idx[1]])
      if dr_threshold > 0.0:
        dr_2 = space.square_distance(dR)
        mask = dr_2 < r_cutoff ** 2 + 1e-5
        graph = partition.to_jraph(neighbor, mask)
        # TODO(schsam): It seems wasteful to recompute dR after we remask the
        # edges. If I can think of a clean way to get rid of this, I should.
        dR = d(R[graph.receivers], R[graph.senders])
      else:
        graph = partition.to_jraph(neighbor)

      graph = graph._replace(
        nodes=jnp.concatenate((_nodes,
                               jnp.zeros((1,) + _nodes.shape[1:], R.dtype)),
                             axis=0),
        edges=dR,
        globals=jnp.broadcast_to(_globals[:, None], (2, 1))
      )

    net = EnergyGraphNet(n_recurrences, mlp_sizes, mlp_kwargs, format)
    return net(graph)  # pytype: disable=wrong-arg-count

  neighbor_fn = partition.neighbor_list(
    displacement_fn,
    box_size,
    r_cutoff,
    dr_threshold,
    mask_self=False,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **neighbor_kwargs)
  init_fn, apply_fn = model.init, model.apply

  return neighbor_fn, init_fn, apply_fn
