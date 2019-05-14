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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np

from jax_md import space, smap


def soft_sphere(dR, sigma=1.0, epsilon=1.0, alpha=2.0):
  """Finite ranged repulsive interaction between soft spheres.

  Args:
    dR: An ndarray of shape [n, m, spatial_dimension] of displacement vectors
      between particles.
    sigma: Particle radii. Should either be a floating point scalar or an
      ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
    alpha: Exponent specifying interaction stiffness. Should either be a float
      point scalar or an ndarray whose shape is [n, m].

  Returns:
    Matrix of energies whose shape is [n, m].
  """
  dr = space.distance(dR)
  dr = dr / sigma
  U = epsilon * np.where(dr < 1.0, 1.0 / alpha * (1.0 - dr) ** alpha, 0.0)
  # NOTE(schsam): This seems a little bit janky. However, it seems possibly
  # necessary because constants seemed to be upcast to float64.
  return np.array(U, dtype=dr.dtype)


def soft_sphere_pairwise(
    metric, species=None, sigma=1.0, epsilon=1.0, alpha=2.0):
  """Convenience wrapper to compute soft sphere energy over a system."""
  return smap.pairwise(
      soft_sphere,
      metric,
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      alpha=alpha)


def lennard_jones(dR, sigma, epsilon):
  """Lennard-Jones interaction between particles with a minimum at sigma.

  Args:
    dR: An ndarray of shape [n, m, spatial_dimension] of displacement vectors
      between particles.
    sigma: Distance between particles where the energy has a minimum. Should
      either be a floating point scalar or an ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
  Returns:
    Matrix of energies of shape [n, m].
  """
  dr = space.square_distance(dR)
  dr = sigma ** 2 / dr
  idr6 = dr ** 3.0
  idr12 = idr6 ** 2.0
  return epsilon * (idr12 - 2 * idr6)


def lennard_jones_pairwise(
    metric, species=None, sigma=1.0, epsilon=1.0):
  """Convenience wrapper to compute Lennard-Jones energy over a system."""
  return smap.pairwise(
      lennard_jones, metric, species=species, sigma=sigma, epsilon=epsilon)
