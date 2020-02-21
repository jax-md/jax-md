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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import grad, vmap
import jax.numpy as np

from jax_md import space
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


def cosine_angles(dR):
  """Returns cosine of angles for all atom triplets.

  Args:
    dR: Matrix of displacements; ndarray(shape=[num_atoms, num_neighbors,
      spatial_dim]).

  Returns:
    Tensor of cosine of angles;
    ndarray(shape=[num_atoms, num_neighbors, num_neighbors]).
  """

  def angle_between_two_vectors(dR_12, dR_13):
    dr_12 = space.distance(dR_12) + 1e-7
    dr_13 = space.distance(dR_13) + 1e-7
    cos_angle = np.dot(dR_12, dR_13) / dr_12 / dr_13
    return np.clip(cos_angle, -1.0, 1.0)

  angles_between_all_triplets = vmap(
      vmap(vmap(angle_between_two_vectors, (0, None)), (None, 0)), 0)
  return angles_between_all_triplets(dR, dR)


def pair_correlation(displacement_or_metric, rs, sigma):
  metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

  sigma = np.array(sigma, f32)
  # NOTE(schsam): This seems rather harmless, but possibly something to look at
  rs = np.array(rs + 1e-7, f32)

  # TODO(schsam): Get this working with cell list .
  def compute_fun(R, **dynamic_kwargs):
    _metric = partial(metric, **dynamic_kwargs)
    _metric = space.map_product(_metric)
    dr = _metric(R, R)
    # TODO(schsam): Clean up.
    dr = np.where(dr > f32(1e-7), dr, f32(1e7))
    dim = R.shape[1]
    exp = np.exp(-f32(0.5) * (dr[:, :, np.newaxis] - rs) ** 2 / sigma ** 2)
    e = np.exp(dr / sigma ** 2)
    gaussian_distances = exp / np.sqrt(2 * np.pi * sigma ** 2)
    return np.mean(gaussian_distances, axis=1) / rs ** (dim - 1)
  return compute_fun


def box_size_at_number_density(
    particle_count, number_density, spatial_dimension):
  return np.power(particle_count / number_density, 1 / spatial_dimension)


def bulk_modulus(elastic_tensor):
  return np.einsum('iijj->', elastic_tensor) / elastic_tensor.shape[0] ** 2
