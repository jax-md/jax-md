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

"""Describes different physical quantities."""


from typing import TypeVar, Callable, Union, Tuple, Optional, Any

from absl import logging
from brainunit import Quantity
from brainunit.autograd import grad
import brainunit as u

from jax import vmap, eval_shape
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
from jax import ops
from jax import ShapeDtypeStruct
from jax.tree_util import tree_map, tree_reduce
from jax.scipy.special import gammaln

from jax_md import space, dataclasses, partition, util

import functools
import operator

partial = functools.partial

# Types


Array = util.Array
f32 = util.f32
f64 = util.f64

DisplacementFn = space.DisplacementFn
MetricFn = space.MetricFn
Box = space.Box

EnergyFn = Callable[..., Array]
ForceFn = Callable[..., Array]

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]


# Functions


def force(energy_fn: EnergyFn) -> ForceFn:
  """Computes the force as the negative gradient of an energy."""
  return grad(lambda R, *args, **kwargs: -energy_fn(R, *args, **kwargs))


def clipped_force(energy_fn: EnergyFn, max_force: float) -> ForceFn:
  force_fn = force(energy_fn)
  def wrapped_force_fn(R, *args, **kwargs):
    force = force_fn(R, *args, **kwargs)
    force_norm = jnp.linalg.norm(force, axis=-1, keepdims=True)
    return jnp.where(force_norm > max_force,
                     force / force_norm * max_force,
                     force)

  return wrapped_force_fn


def canonicalize_force(energy_or_force_fn: Union[EnergyFn, ForceFn]) -> ForceFn:
  _force_fn = None
  def force_fn(R, **kwargs):
    nonlocal _force_fn
    if _force_fn is None:
      out_shaped = eval_shape(energy_or_force_fn, R, **kwargs)
      try:
        out_shaped = out_shaped.mantissa
      except Exception:
        pass
      if isinstance(out_shaped, ShapeDtypeStruct) and out_shaped.shape == ():
        _force_fn = force(energy_or_force_fn)
      else:
        # Check that the output has the right shape to be a force.
        is_valid_force = tree_reduce(
          lambda x, y: x and y,
          tree_map(lambda x, y: x.shape == y.shape, out_shaped, R),
          True
        )
        if not is_valid_force:
          raise ValueError('Provided function should be compatible with '
                           'either an energy or a force. Found a function '
                           f'whose output has shape {out_shaped}.')

        _force_fn = energy_or_force_fn
    return _force_fn(R, **kwargs)

  return force_fn


@functools.singledispatch
def count_dof(position: Array) -> int:
  util.check_custom_simulation_type(position)
  return tree_reduce(lambda accum, x: accum + x.size, position, 0)


def volume(dimension: int, box: Box) -> Array:
  if jnp.isscalar(box) or not box.ndim:
    return box ** dimension
  elif box.ndim == 1:
    return u.math.prod(box)
  elif box.ndim == 2:
    return u.linalg.det(box)
  raise ValueError(('Box must be either: a scalar, a vector, or a matrix. '
                    f'Found {box}.'))


def kinetic_energy(*unused_args,
                   momentum: Array=None,
                   velocity: Array=None,
                   mass: Array=1.0,
                   ) -> float:
  """Computes the kinetic energy of a system.

  To avoid ambiguity, either momentum or velocity must be passed explicitly
  as a keyword argument.

  Args:
    momentum: Array specifying the momentum of the system.
    velocity: Array specifying the velocity of the system.
    mass: Array specifying the mass of the constituents.

  Returns:
    The kinetic energy of the system.
  """
  if unused_args:
    raise ValueError('To use the kinetic energy function, you must explicitly '
                     'pass either momentum or velocity as a keyword argument.')
  if momentum is not None and velocity is not None:
    raise ValueError('To use the kinetic energy function, you must pass either'
                     ' a momentum or a velocity.')
  return_quantity = False
  if isinstance(momentum, Quantity):
    momentum = momentum.to_decimal(u.IMF * u.angstrom)
    return_quantity = True
  if isinstance(mass, Quantity):
    mass = mass.to_decimal(u.atomic_mass)
    return_quantity = True
  if isinstance(velocity, Quantity):
    velocity = velocity.to_decimal(u.angstrom / u.fsecond)
    return_quantity = True

  k = (lambda v, m: v**2 * m) if momentum is None else (lambda p, m: p**2 / m)
  q = velocity if momentum is None else momentum
  util.check_custom_simulation_type(q)

  ke = tree_map(lambda m, q: 0.5 * util.high_precision_sum(k(q, m)), mass, q)
  r = tree_reduce(operator.add, ke, 0.0)
  return r * u.eV if return_quantity else r


def temperature(*unused_args,
                momentum: Array=None,
                velocity: Array=None,
                mass: Array=1.0,
                ) -> float:
  """Computes the temperature of a system.

  To avoid ambiguity, either momentum or velocity must be passed explicitly
  as a keyword argument.

  Args:
    momentum: Array specifying the momentum of the system.
    velocity: Array specifying the velocity of the system.
    mass: Array specifying the mass of the constituents.

  Returns:
    The temperature of the system in units of the Boltzmann constant.
  """
  if unused_args:
    raise ValueError('To use the kinetic energy function, you must explicitly '
                     'pass either momentum or velocity as a keyword argument.')
  if momentum is not None and velocity is not None:
    raise ValueError('To use the kinetic energy function, you must pass either'
                     ' a momentum or a velocity.')

  return_quantity = False
  if isinstance(momentum, Quantity):
    momentum = momentum.to_decimal(u.IMF * u.angstrom)
    return_quantity = True
  if isinstance(mass, Quantity):
    mass = mass.to_decimal(u.atomic_mass)
    return_quantity = True
  if isinstance(velocity, Quantity):
    velocity = velocity.to_decimal(u.angstrom / u.fsecond)
    return_quantity = True

  t = (lambda v, m: v**2 * m) if momentum is None else (lambda p, m: p**2 / m)
  q = velocity if momentum is None else momentum
  util.check_custom_simulation_type(q)

  dof = count_dof(q)

  kT = tree_map(lambda m, q: util.high_precision_sum(t(q, m)) / dof, mass, q)
  r = tree_reduce(operator.add, kT, 0.0)
  return r * u.kelvin if return_quantity else r


def pressure(energy_fn: EnergyFn, position: Array, box: Box,
             kinetic_energy: float=0.0, **kwargs) -> float:
  """Computes the internal pressure of a system.

  Args:
    energy_fn: A function that computes the energy of the system. This
      function must take as an argument `perturbation` which perturbs the
      box shape. Any energy function constructed using `smap` or in `energy.py`
      with a standard space will satisfy this property.
    position: An array of particle positions.
    box: A box specifying the shape of the simulation volume. Used to infer the
      volume of the unit cell.
    kinetic_energy: A float specifying the kinetic energy of the system.

  Returns:
    A float specifying the pressure of the system.
  """
  return_quantity = False
  if isinstance(position, Quantity) or isinstance(box, Quantity):
    return_quantity = True

  dim = position.shape[1]

  def U(eps):
    try:
      return energy_fn(position, box=box, perturbation=(1 + eps), **kwargs)
    except space.UnexpectedBoxException:
      return energy_fn(position, perturbation=(1 + eps), **kwargs)

  dUdV = grad(U)
  vol_0 = volume(dim, box)

  return 1 / (dim * u.get_mantissa(vol_0)) * (2 * u.get_mantissa(kinetic_energy) - u.get_mantissa(dUdV(0.0))) * (u.eV/u.angstrom**3) if return_quantity \
    else 1 / (dim * vol_0) * (2 * kinetic_energy - dUdV(0.0))


def stress(energy_fn: EnergyFn, position: Array, box: Box,
           mass: Array=1.0, velocity: Optional[Array]=None, **kwargs
           ) -> Array:
  """Computes the internal stress of a system.

  Args:
    energy_fn: A function that computes the energy of the system. This
      function must take as an argument `perturbation` which perturbs the
      box shape. Any energy function constructed using `smap` or in `energy.py`
      with a standard space will satisfy this property.
    position: An array of particle positions.
    box: A box specifying the shape of the simulation volume. Used to infer the
      volume of the unit cell.
    mass: The mass of the particles; only used to compute the kinetic
      contribution if `velocity` is not `None`.
    velocity: An array of atomic velocities.

  Returns:
    A float specifying the pressure of the system.
  """
  return_quantity = False
  if isinstance(position, Quantity) or isinstance(box, Quantity):
    return_quantity = True

  dim = position.shape[1]

  zero = jnp.zeros((dim, dim), position.dtype)
  I = jnp.eye(dim, dtype=position.dtype)

  def U(eps):
    try:
      return energy_fn(position, box=box, perturbation=(I + eps), **kwargs)
    except space.UnexpectedBoxException:
      return energy_fn(position, perturbation=(I + eps), **kwargs)

  dUdV = grad(U)
  vol_0 = volume(dim, box)

  VxV = 0.0
  if velocity is not None:
    V = velocity
    VxV = util.high_precision_sum(mass * V[:, None, :] * V[:, :, None], axis=0)

  return 1 / u.get_mantissa(vol_0) * (u.get_mantissa(VxV) - u.get_mantissa(dUdV(zero))) * (u.eV/u.angstrom**3) if return_quantity \
    else 1 / vol_0 * (VxV - dUdV(zero))


def cosine_angle_between_two_vectors(dR_12: Array, dR_13: Array) -> Array:
  dr_12 = space.distance(dR_12) + 1e-7
  dr_13 = space.distance(dR_13) + 1e-7
  cos_angle = jnp.dot(dR_12, dR_13) / dr_12 / dr_13
  return jnp.clip(cos_angle, -1.0, 1.0)


def cosine_angles(dR: Array) -> Array:
  """Returns cosine of angles for all atom triplets.

  Args:
    dR: Matrix of displacements; `ndarray(shape=[num_atoms, num_neighbors,
      spatial_dim])`.

  Returns:
    Tensor of cosine of angles;
    `ndarray(shape=[num_atoms, num_neighbors, num_neighbors])`.
  """

  angles_between_all_triplets = vmap(
      vmap(vmap(cosine_angle_between_two_vectors, (0, None)), (None, 0)), 0)
  return angles_between_all_triplets(dR, dR)


def is_integer(x: Array) -> bool:
  return x.dtype == jnp.int32 or x.dtype == jnp.int64


def average_pair_correlation_results(gofr, species=None):
  """ Calculate species-based averages of pair correlations.

  Average the results of pair_correlation or pair_correlation_neighbor_list,
  appropriately taking species information into account.

  When species=None, gofr is expected to be an array of shape (N,nr), where N is
  the number of species and nr is the number of radii to be considered. The
  average is calculated over all particles, so an array of shape (nr,) is
  returned.

  When species is specified, gofr is expected to be a list of nspecies arrays,
  each of shape (N,nr), where nspecies is the number of unique species types.
  Here, the average is carried out separately for every pair of species, so the
  returned array has shape (nspecies, nspecies, nr).

  Args:
    gofr: array of shape (N,nr) or a list of arrays of shape (N,nr), where nr is
      the number of radii for which :math:`g(r)` is calculated.
    species: Optional. Array of shape (N,) specifying the species of each
      particle.

  Returns:
    An array of shape (nr,) for species=None, otherwise an array of shape
      (nspecies, nspecies, nr), where nspecies is the number of unique species.
  """
  if species is None:
    return jnp.mean(gofr, axis=0)
  species_types = jnp.unique(species) #note: this returns in sorted order
  return jnp.array([ [ jnp.mean(gofr[si][species==s], axis=0) \
      for s in species_types] for si in range(species_types.size)])


def pair_correlation(displacement_or_metric: Union[DisplacementFn, MetricFn],
                     radii: Array,
                     sigma: float,
                     species: Array = None,
                     eps: float = 1e-7,
                     compute_average: bool = False):
  """Computes the pair correlation function at a mesh of distances.

  The pair correlation function measures the number of particles at a given
  distance from a central particle. The pair correlation function is defined by

  .. math::
    g(r) = <\sum_{i \\neq j}\delta(r - |r_i - r_j|)>

  We make the approximation,

  .. math::
    \delta(r) \\approx {1 \over \sqrt{2\pi\sigma^2}e^{-r / (2\sigma^2)}}

  Args:
    displacement_or_metric:
      A function that computes the displacement or distance between two points.
    radii: An array of radii at which we would like to compute :math:`g(r)`.
    sigima: A float specifying the width of the approximating Gaussian.
    species: An optional array specifying the species of each particle. If
      species is None then we compute a single :math:`g(r)` for all particles,
      otherwise we compute one :math:`g(r)` for each species.
    eps: A small additive constant used to ensure stability if the radius is
      zero.

  Returns:
    A function `g_fn` that computes the pair correlation function for a
    collection of particles.

  :math:`g(r)` is calculated separately for each particle. For species=None, the
  output of `g_fn` is an array of shape (N, nr), where N is the number of
  particles passed to `g_fn` and nr is the size of radii (the number of points
  at which we calculate :math:`g(r)`. When species is specified, the output is a
  list of nspecies arrays, each of shape (N, nr), where nspecies is the number
  of unique species. If `gofr` is the output of `g_fn`, then gofr[si][i] gives
  the :math:`g(r)` for particle i considering only pair particles of species si.

  Note: when species is specified, the returned list is in the order of the
  sorted unique species indices, not the order in which they appear.

  """
  d = space.canonicalize_displacement_or_metric(displacement_or_metric)
  d = space.map_product(d)

  inv_rad = 1 / (radii + eps)

  def pairwise(dr, dim):
    return jnp.exp(-f32(0.5) * (dr - radii)**2 / sigma**2) * inv_rad**(dim - 1)
  pairwise = vmap(vmap(pairwise, (0, None)), (0, None))

  if species is None:
    def g_fn(R):
      dim = R.shape[-1]
      mask = 1 - jnp.eye(R.shape[0], dtype=R.dtype)
      g_R = jnp.sum(mask[:, :, jnp.newaxis] *
                    pairwise(d(R, R), dim), axis=(1,))
      if compute_average:
        g_R = average_pair_correlation_results(g_R, species)
      return g_R
  else:
    if not (isinstance(species, jnp.ndarray) and is_integer(species)):
      raise TypeError('Malformed species; expecting array of integers.')
    species_types = jnp.unique(species)
    def g_fn(R):
      dim = R.shape[-1]
      g_R = []
      mask = 1 - jnp.eye(R.shape[0], dtype=R.dtype)
      for s in species_types:
        Rs = R[species == s]
        mask_s = mask[:, species == s, jnp.newaxis]
        g_R += [jnp.sum(mask_s * pairwise(d(Rs, R), dim), axis=(1,))]
      if compute_average:
        g_R = average_pair_correlation_results(g_R, species)
      return g_R
  return g_fn


def pair_correlation_neighbor_list(
    displacement_or_metric: Union[DisplacementFn, MetricFn],
    box_size: Box,
    radii: Array,
    sigma: float,
    species: Array = None,
    dr_threshold: float = 0.5,
    eps: float = 1e-7,
    fractional_coordinates: bool=False,
    format: partition.NeighborListFormat=partition.Dense,
    compute_average: bool = False):
  """Computes the pair correlation function at a mesh of distances.

  The pair correlation function measures the number of particles at a given
  distance from a central particle. The pair correlation function is defined by

  .. math::
    g(r) = <\sum_{i \\neq j}\delta(r - |r_i - r_j|)>

  We make the approximation,

  .. math::
    \delta(r) \\approx {1 \over \sqrt{2\pi\sigma^2}e^{-r / (2\sigma^2)}}

  This function uses neighbor lists to speed up the calculation.

  Args:
    displacement_or_metric: A function that computes the displacement or
      distance between two points.
    box_size: The size of the box containing the particles.
    radii: An array of radii at which we would like to compute :math:`g(r)`.
    sigima: A float specifying the width of the approximating Gaussian.
    species: An optional array specifying the species of each particle. If
      species is None then we compute a single :math:`g(r)` for all particles,
      otherwise we compute one :math:`g(r)` for each species.
    dr_threshold: A float specifying the halo size of the neighbor list.
    eps: A small additive constant used to ensure stability if the radius is
      zero.
    fractional_coordinates: Bool determining whether positions are stored in
      the unit cube or not.
    format: The format of the neighbor lists. Must be `Dense` or `Sparse`.

  Returns:
    A pair of functions: `neighbor_fn` that constructs a neighbor list (see
    `neighbor_list` in `partition.py` for details). `g_fn` that computes the
    pair correlation function for a collection of particles given their
    position and a neighbor list.

  :math:`g(r)` is calculated separately for each particle. For species=None, the
  output of `g_fn` is an array of shape (N, nr), where N is the number of
  particles passed to `g_fn` and nr is the size of radii (the number of points
  at which we calculate :math:`g(r)`. When species is specified, the output is a
  list of nspecies arrays, each of shape (N, nr), where nspecies is the number
  of unique species. If `gofr` is the output of `g_fn`, then gofr[si][i] gives
  the :math:`g(r)` for particle i considering only pair particles of species si.

  Note: when species is specified, the returned list is in the order of the
  sorted unique species indices, not the order in which they appear.
  """
  metric = space.canonicalize_displacement_or_metric(displacement_or_metric)
  inv_rad = 1 / (radii + eps)
  def pairwise(dr, dim):
    return jnp.exp(-f32(0.5) * (dr - radii)**2 / sigma**2) * inv_rad**(dim - 1)

  neighbor_fn = partition.neighbor_list(displacement_or_metric,
                                        box_size,
                                        jnp.max(radii) + sigma,
                                        dr_threshold,
                                        format=format)

  if species is None:
    def g_fn(R, neighbor):
      N, dim = R.shape
      mask = partition.neighbor_list_mask(neighbor)
      if neighbor.format is partition.Dense:
        R_neigh = R[neighbor.idx]
        d = space.map_neighbor(metric)
        _pairwise = vmap(vmap(pairwise, (0, None)), (0, None))
        g_R = jnp.sum(mask[:, :, None] *
                      _pairwise(d(R, R_neigh), dim), axis=(1,))
        if compute_average:
          g_R = average_pair_correlation_results(g_R, species)
        return g_R
      elif neighbor.format is partition.Sparse:
        dr = space.map_bond(metric)(R[neighbor.idx[0]], R[neighbor.idx[1]])
        _pairwise = vmap(pairwise, (0, None))
        g_R = ops.segment_sum(mask[:, None] * _pairwise(dr, dim),
                              neighbor.idx[0],
                              N)
        if compute_average:
          g_R = average_pair_correlation_results(g_R, species)
        return g_R
      else:
        raise NotImplementedError('Pair correlation function does not support '
                                  'OrderedSparse neighbor lists.')

  else:
    if not (isinstance(species, jnp.ndarray) and is_integer(species)):
      raise TypeError('Malformed species; expecting array of integers.')
    species_types = jnp.unique(species)
    def g_fn(R, neighbor):
      N, dim = R.shape
      g_R = []
      mask = partition.neighbor_list_mask(neighbor)
      if neighbor.format is partition.Dense:
        neighbor_species = species[neighbor.idx]
        R_neigh = R[neighbor.idx]
        d = space.map_neighbor(metric)
        _pairwise = vmap(vmap(pairwise, (0, None)), (0, None))
        for s in species_types:
          mask_s = mask * (neighbor_species == s)
          g_R += [jnp.sum(mask_s[:, :, jnp.newaxis] *
                          _pairwise(d(R, R_neigh), dim), axis=(1,))]
      elif neighbor.format is partition.Sparse:
        neighbor_species = species[neighbor.idx[1]]
        dr = space.map_bond(metric)(R[neighbor.idx[0]], R[neighbor.idx[1]])
        _pairwise = vmap(pairwise, (0, None))
        for s in species_types:
          mask_s = mask * (neighbor_species == s)
          g_R += [ops.segment_sum(mask_s[:, None] *
                                  _pairwise(dr, dim), neighbor.idx[0], N)]
      else:
        raise NotImplementedError('Pair correlation function does not support '
                                  'OrderedSparse neighbor lists.')

      if compute_average:
        g_R = average_pair_correlation_results(g_R, species)
      return g_R
  return neighbor_fn, g_fn

def nball_unit_volume(spatial_dimension: int) -> float:
  """ Return the volume of a unit sphere in arbitrary dimensions
  """
  return jnp.power(jnp.pi, spatial_dimension / 2) / \
    jnp.exp( gammaln(spatial_dimension / 2 + 1))

def particle_volume(radii: Array,
                    spatial_dimension: int,
                    particle_count: Array = 1,
                    species: Array = None) -> float:
  """ Calculate the volume of a collection of particles

  Args:
    radii: array of shape (n,) giving particle radii, where n can be 1, the
      number of species, or the number of particles depending on the values of
      particle_count and species.
    spatial_dimension: int giving the spatial dimension
    particle_count: number of particles with each radii. Broadcastable to radii.
    species: list of particle species. If provided, this overrides
      particle_count and radii is expected to give per-species radii

  Returns: the sum of the volume of all the particles
  """
  V_unit = nball_unit_volume(spatial_dimension)
  V_particle = V_unit * radii**spatial_dimension

  if species is not None:
    particle_count = jnp.bincount(species)

  return jnp.sum(particle_count * V_particle)

def volume_fraction(box: Box,
                    radii: Array,
                    spatial_dimension: int,
                    particle_count: Array = 1,
                    species: Array = None) -> float:
  """ Calculate the volume fraction

  See documentation for particle_volume for explanation of parameters
  """
  Vparticle = particle_volume(radii, spatial_dimension, particle_count, species)
  return Vparticle / volume(spatial_dimension, box)

def box_size_at_volume_fraction(volume_fraction: float,
                                radii: Array,
                                spatial_dimension: int,
                                particle_count: Array = 1,
                                species: Array = None) -> float:
  """ Calculate box_size to obtain a desired volume fraction

  See documentation for particle_volume for explanation of parameters
  """
  Vparticle = particle_volume(radii, spatial_dimension, particle_count, species)
  return jnp.power( Vparticle / volume_fraction, 1 / spatial_dimension)

def box_size_at_number_density(particle_count: int,
                               number_density: float | Quantity,
                               spatial_dimension: int) -> float:
  if isinstance(number_density, Quantity):
    number_density = number_density.to_decimal(1/u.angstrom**spatial_dimension)
  return jnp.power(particle_count / number_density, 1 / spatial_dimension) * u.angstrom


def box_from_parameters(a: float, b: float, c: float,
                        alpha: float, beta: float, gamma: float) -> Box:
  alpha = alpha * jnp.pi / 180
  beta = beta * jnp.pi / 180
  gamma = gamma * jnp.pi / 180
  yy = b * jnp.sin(gamma)
  xy = b * jnp.cos(gamma)
  xz = c * jnp.cos(beta)
  yz = (b * c * jnp.cos(alpha) - xy * xz) / yy
  zz = jnp.sqrt(c**2 - xz**2 - yz**2)
  return jnp.array([
      [a, xy, xz],
      [0, yy, yz],
      [0, 0,  zz]
  ])

def bulk_modulus(elastic_tensor: Array) -> float:
  return jnp.einsum('iijj->', elastic_tensor) / elastic_tensor.shape[0] ** 2


@dataclasses.dataclass
class PHopState:
    position_buffer: jnp.ndarray
    phop: jnp.ndarray

InitFn = Callable[[Array], PHopState]
ApplyFn = Callable[[PHopState, Array], PHopState]
PHopCalculator = Tuple[InitFn, ApplyFn]

def phop(displacement: DisplacementFn, window_size: int) -> PHopCalculator:
  """Computes the phop indicator of rearrangements.

  phop is an indicator function that is effective at detecting when particles
  in a quiescent system have experienced a rearrangement. Qualitatively, phop
  measures when the average position of a particle has changed significantly.

  Formally, given a window of size :math:`\Delta t` we two averages before and after
  the current time,

  .. math::
    E_A[f] = E_{t\in[t - \Delta t / 2, t]}[f(t)]
    E_B[f] = E_{t\in[t, t + \Delta t / 2]}[f(t)].

  In terms of these expectations, phop is given by,

  .. math::
    phop = \sqrt{E_A[(R_i(t) - E_B[R_i(t)])^2]E_B[(R_i(t) - E_A[R_i(t)])^2]}

  phop was first introduced in Candelier et al. [#candelier]_

  Args:
    displacement: A function that computes displacements between pairs of
      particles. See `spaces.py` for details.
    window_size: An integer specifying the number of positions that constitute
      the window.

  Returns:
    A pair of functions, `(init_fn, update_fn)` that initialize the state of a
    phop measurement and update the state of a phop measurement to include new
    positions.

  .. rubric:: References
  .. [#candelier] R. Candelier et al.
    "Spatiotemporal Hierarchy of Relaxation Events, Dynamical Heterogeneities,
    and Structural Reorganization in a Supercooled Liquid"
    Physical Review Letters 105, 135702 (2010).
  """

  half_window_size = window_size // 2
  displacement = space.map_bond(displacement)

  def init_fn(position: Array) -> PHopState:
    position_buffer = jnp.tile(position, (window_size, 1, 1))
    assert position_buffer.shape == ((window_size,) + position.shape)
    return PHopState(position_buffer, jnp.zeros((position.shape[0],)))  # pytype: disable=wrong-arg-count

  def update_fn(state: PHopState, position: Array) -> PHopState:
    # Compute phop.
    a_pos = state.position_buffer[:half_window_size]
    a_mean = jnp.mean(a_pos, axis=0)
    b_pos = state.position_buffer[half_window_size:]
    b_mean = jnp.mean(b_pos, axis=0)

    phop = jnp.sqrt(jnp.mean((a_pos - b_mean) ** 2 * (b_pos - a_mean) ** 2,
                             axis=(0, 2)))

    # Unwrap position.
    buff = state.position_buffer
    position = displacement(position, buff[-1]) + buff[-1]

    # Add position to the list.
    buff = jnp.concatenate((buff, position[jnp.newaxis, :, :]))[1:]

    return PHopState(buff, phop)  # pytype: disable=wrong-arg-count

  return init_fn, update_fn
