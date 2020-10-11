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

from jax import grad, vmap
import jax.numpy as np

from jax_md import space, dataclasses, partition
from jax_md.util import *

from functools import partial

def force(energy):
  """Computes the force as the negative gradient of an energy."""
  return grad(lambda R, *args, **kwargs: -energy(R, *args, **kwargs))


def canonicalize_force(energy_or_force, quantity):
  if quantity is Force:
    return energy_or_force
  elif quantity is Energy:
    return force(energy_or_force)

  raise ValueError(
      'Expected quantity to be Energy or Force, but found {}'.format(quantity))


class Force(object):
  """Dummy object to denote whether a quantity is a force."""
  pass
Force = Force()


class Energy(object):
  """Dummy object to denote whether a quantity is an energy."""
  pass
Energy = Energy()


class Dynamic(object):
  """Object used to denote dynamic shapes and species."""
  pass
Dynamic = Dynamic()


def kinetic_energy(V, mass=1.0):
  """Computes the kinetic energy of a system with some velocities."""
  return 0.5 * np.sum(mass * V ** 2)


def temperature(V, mass=1.0):
  """Computes the temperature of a system with some velocities."""
  N, dim = V.shape
  return np.sum(mass * V ** 2) / (N * dim)


def canonicalize_mass(mass):
  if isinstance(mass, float):
    return mass
  elif isinstance(mass, np.ndarray):
    if len(mass.shape) == 2 and mass.shape[1] == 1:
      return mass
    elif len(mass.shape) == 1:
      return np.reshape(mass, (mass.shape[0], 1))
    elif len(mass.shape) == 0:
      return mass
  elif (isinstance(mass, f32) or
        isinstance(mass, f64)):
    return mass
  msg = (
      'Expected mass to be either a floating point number or a one-dimensional'
      'ndarray. Found {}.'.format(mass)
      )
  raise ValueError(msg)


def angle_between_two_vectors(dR_12, dR_13):
  dr_12 = space.distance(dR_12) + 1e-7
  dr_13 = space.distance(dR_13) + 1e-7
  cos_angle = np.dot(dR_12, dR_13) / dr_12 / dr_13
  return np.clip(cos_angle, -1.0, 1.0)


def cosine_angles(dR):
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


def is_integer(x):
  return x.dtype == np.int32 or x.dtype == np.int64


def pair_correlation(displacement_or_metric, rs, sigma, species=None):
  """Computes the pair correlation function at a mesh of distances.

  The pair correlation function measures the number of particles at a given
  distance from a central particle. The pair correlation function is defined
  by $g(r) = <\sum_{i\neq j}\delta(r - |r_i - r_j|)>.$ We make the
  approximation
  $\delta(r) \approx {1 \over \sqrt{2\pi\sigma^2}e^{-r / (2\sigma^2)}}$.

  Args:
    displacement_or_metric: A function that computes the displacement or
      distance between two points.
    rs: An array of radii at which we would like to compute g(r).
    sigima: A float specifying the width of the approximating Gaussian.
    species: An optional array specifying the species of each particle. If
      species is None then we compute a single g(r) for all particles,
      otherwise we compute one g(r) for each species.

  Returns:
    A function `g_fn` that computes the pair correlation function for a
    collection of particles.
  """
  d = space.canonicalize_displacement_or_metric(displacement_or_metric)
  d = space.map_product(d)
  def pairwise(dr, dim):
    return np.exp(-f32(0.5) * (dr - rs) ** 2 / sigma ** 2) / rs ** (dim - 1)
  pairwise = vmap(vmap(pairwise, (0, None)), (0, None))

  if species is None:
    def g_fn(R):
      dim = R.shape[-1]
      mask = 1 - np.eye(R.shape[0], dtype=R.dtype)
      return np.sum(mask[:, :, np.newaxis] * pairwise(d(R, R), dim), axis=(1,))
  else:
    if not (isinstance(species, np.ndarray) and is_integer(species)):
      raise TypeError('Malformed species; expecting array of integers.')
    species_types = np.unique(species)
    def g_fn(R):
      dim = R.shape[-1]
      g_R = []
      mask = 1 - np.eye(R.shape[0], dtype=R.dtype)
      for s in species_types:
        Rs = R[species == s]
        mask_s = mask[:, species == s, np.newaxis]
        g_R += [np.sum(mask_s * pairwise(d(Rs, R), dim), axis=(1,))]
      return g_R
  return g_fn


def pair_correlation_neighbor_list(displacement_or_metric,
                                   box_size,
                                   rs,
                                   sigma,
                                   species=None,
                                   dr_threshold=0.5):
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
    rs: An array of radii at which we would like to compute g(r).
    sigima: A float specifying the width of the approximating Gaussian.
    species: An optional array specifying the species of each particle. If
      species is None then we compute a single g(r) for all particles,
      otherwise we compute one g(r) for each species.
    dr_threshold: A float specifying the halo size of the neighobr list.

  Returns:
    A pair of functions: `neighbor_fn` that constructs a neighbor list (see
    `neighbor_list` in `partition.py` for details). `g_fn` that computes the
    pair correlation function for a collection of particles given their
    position and a neighbor list.
  """
  d = space.canonicalize_displacement_or_metric(displacement_or_metric)
  d = space.map_neighbor(d)
  def pairwise(dr, dim):
    return np.exp(-f32(0.5) * (dr - rs) ** 2 / sigma ** 2) / rs ** (dim - 1)
  pairwise = vmap(vmap(pairwise, (0, None)), (0, None))

  neighbor_fn = partition.neighbor_list(displacement_or_metric, box_size, np.max(rs) + sigma, dr_threshold)

  if species is None:
    def g_fn(R, neighbor):
      dim = R.shape[-1]
      R_neigh = R[neighbor.idx]
      mask = neighbor.idx < R.shape[0]
      return np.sum(mask[:, :, np.newaxis] * pairwise(d(R, R_neigh), dim), axis=(1,))
  else:
    if not (isinstance(species, np.ndarray) and is_integer(species)):
      raise TypeError('Malformed species; expecting array of integers.')
    species_types = np.unique(species)
    def g_fn(R, neighbor):
      dim = R.shape[-1]
      g_R = []
      mask = neighbor.idx < R.shape[0]
      neighbor_species = species[neighbor.idx]
      R_neigh = R[neighbor.idx]
      for s in species_types:
        mask_s = mask * (neighbor_species == s)
        g_R += [np.sum(mask_s[:, :, np.newaxis] * pairwise(d(R, R_neigh), dim), axis=(1,))]
      return g_R
  return neighbor_fn, g_fn


def box_size_at_number_density(
    particle_count, number_density, spatial_dimension):
  return np.power(particle_count / number_density, 1 / spatial_dimension)


def bulk_modulus(elastic_tensor):
  return np.einsum('iijj->', elastic_tensor) / elastic_tensor.shape[0] ** 2


@dataclasses.dataclass
class PHopState:
    position_buffer: np.ndarray
    phop: np.ndarray

def phop(displacement, window_size):
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

  def init_fn(position):
    position_buffer = np.tile(position, (window_size, 1, 1))
    assert position_buffer.shape == ((window_size,) + position.shape)
    return PHopState(
        position_buffer = position_buffer,
        phop = np.zeros((position.shape[0],)),
    )

  def update_fn(state, position):
    # Compute phop.
    a_pos = state.position_buffer[:half_window_size]
    a_mean = np.mean(a_pos, axis=0)
    b_pos = state.position_buffer[half_window_size:]
    b_mean = np.mean(b_pos, axis=0)

    phop = np.sqrt(np.mean((a_pos - b_mean) ** 2 * (b_pos - a_mean) ** 2,
                           axis=(0, 2)))
    
    # Unwrap position.
    buff = state.position_buffer
    position = displacement(position, buff[-1]) + buff[-1]

    # Add position to the list.
    buff = np.concatenate((buff, position[np.newaxis, :, :]))[1:]
     
    return PHopState(buff, phop)

  return init_fn, update_fn
