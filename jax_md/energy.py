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

from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe

from jax_md import space, smap
from jax_md.util import *


def _canonicalize_displacement_or_metric(displacement_or_metric):
  """Checks whether or not a displacement or metric was provided."""
  for dim in range(4):
    try:
      R = ShapedArray((1, dim), f32)
      dR_or_dr = pe.abstract_eval_fun(displacement_or_metric, R, R, t=0)
      if len(dR_or_dr.shape) == 2:
        return displacement_or_metric
      else:
        return space.metric(displacement_or_metric)
    except ValueError:
      continue
  raise ValueError(
    'Canonicalize displacement not implemented for spatial dimension larger'
    'than 4.')


def soft_sphere(dr, sigma=f32(1.0), epsilon=f32(1.0), alpha=f32(2.0)):
  """Finite ranged repulsive interaction between soft spheres.

  Args:
    dr: An ndarray of shape [n, m] of pairwise distances between particles.
    sigma: Particle radii. Should either be a floating point scalar or an
      ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
    alpha: Exponent specifying interaction stiffness. Should either be a float
      point scalar or an ndarray whose shape is [n, m].

  Returns:
    Matrix of energies whose shape is [n, m].
  """
  dr = dr / sigma
  U = epsilon * np.where(
    dr < 1.0, f32(1.0) / alpha * (f32(1.0) - dr) ** alpha, f32(0.0))
  return U


def soft_sphere_pairwise(
    displacement_or_metric, species=None, sigma=1.0, epsilon=1.0, alpha=2.0): 
  """Convenience wrapper to compute soft sphere energy over a system."""
  sigma = np.array(sigma, dtype=f32)
  epsilon = np.array(epsilon, dtype=f32)
  alpha = np.array(alpha, dtype=f32)
  return smap.pairwise(
      soft_sphere,
      _canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      alpha=alpha)


def lennard_jones(dr, sigma, epsilon):
  """Lennard-Jones interaction between particles with a minimum at sigma.

  Args:
    dr: An ndarray of shape [n, m] of pairwise distances between particles.
    sigma: Distance between particles where the energy has a minimum. Should
      either be a floating point scalar or an ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
  Returns:
    Matrix of energies of shape [n, m].
  """
  dr = (sigma / dr) ** f32(2)
  idr6 = dr ** f32(3)
  idr12 = idr6 ** f32(2)
  return f32(4) * epsilon * (idr12 - idr6)


def lennard_jones_pairwise(
    displacement_or_metric, species=None, sigma=1.0, epsilon=1.0):
  """Convenience wrapper to compute Lennard-Jones energy over a system."""
  sigma = np.array(sigma, dtype=f32)
  epsilon = np.array(epsilon, dtype=f32)
  return smap.pairwise(
    lennard_jones,
    _canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    sigma=sigma,
    epsilon=epsilon)


def smooth_cutoff(dr, r_onset, r_cutoff):
  f2 = f32(2)
  f3 = f32(3)
  return np.where(
    dr < r_onset,
    f32(1),
    np.where(
      dr < r_cutoff,
      (r_cutoff ** f2 - r ** f2) ** f2 * (
        r_cutoff ** f2 + f2 * r ** f2 - f3 * r_onset ** f2) / (
          r_cutoff ** f2 - r_onset ** f2) ** f3,
      f32(0)
    )
  )
