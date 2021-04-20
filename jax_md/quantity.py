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


from typing import TypeVar, Callable, Union, Tuple

from jax.api import grad, vmap, eval_shape
import jax.numpy as jnp

from jax_md import space, dataclasses, partition, util

from functools import partial


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


def canonicalize_force(energy_or_force_fn: Union[EnergyFn, ForceFn]) -> ForceFn:
  _force_fn = None
  def force_fn(R, **kwargs):
    nonlocal _force_fn
    if _force_fn is None:
      out_shape = eval_shape(energy_or_force_fn, R, **kwargs).shape
      if out_shape == ():
        _force_fn = force(energy_or_force_fn)
      else:
        _force_fn = energy_or_force_fn
    return _force_fn(R, **kwargs)

  return force_fn


def volume(dimension: int, box: Box) -> float:
  if jnp.isscalar(box) or not box.ndim:
    return box ** dimension
  elif box.ndim == 1:
    return jnp.prod(box)
  elif box.ndim == 2:
    return jnp.linalg.det(box)
  raise ValueError(('Box must be either: a scalar, a vector, or a matrix. '
                    f'Found {box}.'))


def kinetic_energy(velocity: Array, mass: Array=1.0) -> float:
  """Computes the kinetic energy of a system with some velocities."""
  return 0.5 * util.high_precision_sum(mass * velocity ** 2)


def temperature(velocity: Array, mass: Array=1.0) -> float:
  """Computes the temperature of a system with some velocities."""
  N, dim = velocity.shape
  return util.high_precision_sum(mass * velocity ** 2) / (N * dim)


def pressure(energy_fn: EnergyFn, position: Array, box: Box,
             kinetic_energy: float=0.0, **kwargs) -> float:
  """Computes the internal pressure of a system.

  Note: This function requires that `energy_fn` take a `box` keyword argument.
  Most frequently, this is accomplished by using `periodic_general` boundary
  conditions combined with any of the energy functions in `energy.py`. This
  will not work with `space.periodic`.
  """
  dim = position.shape[1]

  vol_0 = volume(dim, box)
  box_fn = lambda vol: (vol / vol_0) ** (1 / dim) * box

  def U(vol):
    return energy_fn(position, box=box_fn(vol), **kwargs)

  dUdV = grad(U)
  KE = kinetic_energy
  F = force(energy_fn)(position, box=box, **kwargs)
  R = space.transform(box, position)
  RdotF = util.high_precision_sum(R * F)

  return 1 / (dim * vol_0) * (2 * KE + RdotF - dim * vol_0 * dUdV(vol_0))


def canonicalize_mass(mass: Union[float, Array]) -> Union[float, Array]:
  if isinstance(mass, float):
    return mass
  elif isinstance(mass, jnp.ndarray):
    if len(mass.shape) == 2 and mass.shape[1] == 1:
      return mass
    elif len(mass.shape) == 1:
      return jnp.reshape(mass, (mass.shape[0], 1))
    elif len(mass.shape) == 0:
      return mass
  elif isinstance(mass, f32) or isinstance(mass, f64):
    return mass
  msg = (
      'Expected mass to be either a floating point number or a one-dimensional'
      'ndarray. Found {}.'.format(mass)
      )
  raise ValueError(msg)


def angle_between_two_vectors(dR_12: Array, dR_13: Array) -> Array:
  dr_12 = space.distance(dR_12) + 1e-7
  dr_13 = space.distance(dR_13) + 1e-7
  cos_angle = jnp.dot(dR_12, dR_13) / dr_12 / dr_13
  return jnp.clip(cos_angle, -1.0, 1.0)


def cosine_angles(dR: Array) -> Array:
  """Returns cosine of angles for all atom triplets.

  Args:
    dR: Matrix of displacements; ndarray(shape=[num_atoms, num_neighbors,
      spatial_dim]).

  Returns:
    Tensor of cosine of angles;
    ndarray(shape=[num_atoms, num_neighbors, num_neighbors]).
  """

  angles_between_all_triplets = vmap(
      vmap(vmap(angle_between_two_vectors, (0, None)), (None, 0)), 0)
  return angles_between_all_triplets(dR, dR)


def is_integer(x: Array) -> bool:
  return x.dtype == jnp.int32 or x.dtype == jnp.int64


def pair_correlation(displacement_or_metric: Union[DisplacementFn, MetricFn],
                     radii: Array,
                     sigma: float,
                     species: Array = None,
                     eps: float = 1e-7):
  """Computes the pair correlation function at a mesh of distances.

  The pair correlation function measures the number of particles at a given
  distance from a central particle. The pair correlation function is defined
  by $g(r) = <\sum_{i\neq j}\delta(r - |r_i - r_j|)>.$ We make the
  approximation
  $\delta(r) \approx {1 \over \sqrt{2\pi\sigma^2}e^{-r / (2\sigma^2)}}$.

  Args:
    displacement_or_metric: A function that computes the displacement or
      distance between two points.
    radii: An array of radii at which we would like to compute g(r).
    sigima: A float specifying the width of the approximating Gaussian.
    species: An optional array specifying the species of each particle. If
      species is None then we compute a single g(r) for all particles,
      otherwise we compute one g(r) for each species.
    eps: A small additive constant used to ensure stability if the radius is
      zero.

  Returns:
    A function `g_fn` that computes the pair correlation function for a
    collection of particles.
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
      return jnp.sum(mask[:, :, jnp.newaxis] *
                     pairwise(d(R, R), dim), axis=(1,))
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
      return g_R
  return g_fn


def pair_correlation_neighbor_list(
    displacement_or_metric: Union[DisplacementFn, MetricFn],
    box_size: Box,
    radii: Array,
    sigma: float,
    species: Array = None,
    dr_threshold: float = 0.5,
    eps: float = 1e-7):
  """Computes the pair correlation function at a mesh of distances.

  The pair correlation function measures the number of particles at a given
  distance from a central particle. The pair correlation function is defined
  by $g(r) = <\sum_{i\neq j}\delta(r - |r_i - r_j|)>.$ We make the
  approximation,
  $\delta(r) \approx {1 \over \sqrt{2\pi\sigma^2}e^{-r / (2\sigma^2)}}$.

  This function uses neighbor lists to speed up the calculation.

  Args:
    displacement_or_metric: A function that computes the displacement or
      distance between two points.
    box_size: The size of the box containing the particles.
    radii: An array of radii at which we would like to compute g(r).
    sigima: A float specifying the width of the approximating Gaussian.
    species: An optional array specifying the species of each particle. If
      species is None then we compute a single g(r) for all particles,
      otherwise we compute one g(r) for each species.
    dr_threshold: A float specifying the halo size of the neighobr list.
    eps: A small additive constant used to ensure stability if the radius is
      zero.

  Returns:
    A pair of functions: `neighbor_fn` that constructs a neighbor list (see
    `neighbor_list` in `partition.py` for details). `g_fn` that computes the
    pair correlation function for a collection of particles given their
    position and a neighbor list.
  """
  d = space.canonicalize_displacement_or_metric(displacement_or_metric)
  d = space.map_neighbor(d)
  inv_rad = 1 / (radii + eps)
  def pairwise(dr, dim):
    return jnp.exp(-f32(0.5) * (dr - radii)**2 / sigma**2) * inv_rad**(dim - 1)
  pairwise = vmap(vmap(pairwise, (0, None)), (0, None))

  neighbor_fn = partition.neighbor_list(displacement_or_metric,
                                        box_size,
                                        jnp.max(radii) + sigma,
                                        dr_threshold)

  if species is None:
    def g_fn(R, neighbor):
      dim = R.shape[-1]
      R_neigh = R[neighbor.idx]
      mask = neighbor.idx < R.shape[0]
      return jnp.sum(mask[:, :, jnp.newaxis] *
                     pairwise(d(R, R_neigh), dim), axis=(1,))
  else:
    if not (isinstance(species, jnp.ndarray) and is_integer(species)):
      raise TypeError('Malformed species; expecting array of integers.')
    species_types = jnp.unique(species)
    def g_fn(R, neighbor):
      dim = R.shape[-1]
      g_R = []
      mask = neighbor.idx < R.shape[0]
      neighbor_species = species[neighbor.idx]
      R_neigh = R[neighbor.idx]
      for s in species_types:
        mask_s = mask * (neighbor_species == s)
        g_R += [jnp.sum(mask_s[:, :, jnp.newaxis] *
                        pairwise(d(R, R_neigh), dim), axis=(1,))]
      return g_R
  return neighbor_fn, g_fn


def box_size_at_number_density(particle_count: int,
                               number_density: float,
                               spatial_dimension: int) -> float:
  return jnp.power(particle_count / number_density, 1 / spatial_dimension)


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

  Formally, given a window of size \Delta t we two averages before and after
  the current time,

    E_A[f] = E_{t\in[t - \Delta t / 2, t]}[f(t)]
    E_B[f] = E_{t\in[t, t + \Delta t / 2]}[f(t)].

  In terms of these expectations, phop is given by,
    phop = \sqrt{E_A[(R_i(t) - E_B[R_i(t)])^2]E_B[(R_i(t) - E_A[R_i(t)])^2]}.

  phop was first introduced in

    R. Candelier et al.
    "Spatiotemporal Hierarchy of Relaxation Events, Dynamical Heterogeneities,
     and Structural Reorganization in a Supercooled Liquid"
    Physical Review Letters 105, 135702 (2010).

  Args:
    displacement: A function that computes displacements between pairs of
      particles. See `spaces.py` for details.
    window_size: An integer specifying the number of positions that constitute
      the window.

  Returns:
    A pair of functions, `(init_fn, update_fn)` that initialize the state of a
    phop measurement and update the state of a phop measurement to include new
    positions.
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
