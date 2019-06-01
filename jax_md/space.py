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

"""Code to define different spaces in which particles are simulated.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import ad_util
from jax import custom_transforms
from jax.interpreters import ad

import jax.numpy as np

from jax_md.util import *

# Primitive Spatial Transforms


# pylint: disable=invalid-name
def _check_transform_shapes(T, v=None):
  """Check whether a transform and collection of vectors have valid shape."""
  if len(T.shape) != 2:
    raise ValueError(
        ('Transform has invalid rank.'
         ' Found rank {}, expected rank 2.'.format(len(T.shape))))

  if T.shape[0] != T.shape[1]:
    raise ValueError('Found non-square transform.')

  if v is not None and v.shape[-1] != T.shape[1]:
    raise ValueError(
        ('Transform and vectors have incommensurate spatial dimension. '
         'Found {} and {} respectively.'.format(T.shape[1], v.shape[-1])))


def _small_inverse(T):
  """Compute the inverse of a small matrix."""
  _check_transform_shapes(T)
  dim = T.shape[0]

  # TODO(schsam): Check whether matrices are singular. @ErrorChecking

  if dim == 2:
    det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    return np.array([[T[1, 1], -T[0, 1]], [-T[1, 0], T[0, 0]]], dtype=T.dtype) / det

  # TODO(schsam): Fill in the 3x3 case by hand.

  return np.linalg.inv(T)


@custom_transforms
def transform(T, v):
  """Apply a linear transformation, T, to a collection of vectors, v.

  Transform is written such that it acts as the identity during gradient
  backpropagation.

  Args:
    T: Transformation; ndarray(shape=[spatial_dim, spatial_dim]).
    v: Collection of vectors; ndarray(shape=[..., spatial_dim]).

  Returns:
    Transformed vectors; ndarray(shape=[..., spatial_dim]).
  """
  _check_transform_shapes(T, v)
  return np.dot(v, T)
ad.defjvp(
    transform.primitive,
    lambda g, T, v: ad_util.zero,
    lambda g, T, v: g
    )


def pairwise_displacement(Ra, Rb):
  """Compute a matrix of pairwise displacements given two sets of positions.

  Args:
    Ra: Vector of positions; ndarray(shape=[n, spatial_dim]).
    Rb: Vector of positions; ndarray(shape=[m, spatial_dim]).

  Returns:
    Matrix of displacements; ndarray(shape=[n, m, spatial_dim]).
  """
  return Ra[:, np.newaxis, :] - Rb[np.newaxis, :, :]


def periodic_displacement(side, dR):
  """Wraps displacement vectors into a hypercube.

  Args:
    side: Specification of hypercube size. Either,
      (a) float if all sides have equal length.
      (b) ndarray(spatial_dim) if sides have different lengths.
    dR: Matrix of displacements; ndarray(shape=[..., spatial_dim]).
  Returns:
    Matrix of wrapped displacements; ndarray(shape=[..., spatial_dim]).
  """
  return np.mod(dR + side * f32(0.5), side) - f32(0.5) * side


def square_distance(dR):
  """Computes square distances.

  Args:
    dR: Matrix of displacements; ndarray(shape=[..., spatial_dim]).
  Returns:
    Matrix of squared distances; ndarray(shape=[...]).
  """
  return np.sum(dR ** 2, axis=-1)


def distance(dR):
  """Computes distances.

  Args:
    dR: Matrix of displacements; ndarray(shape=[..., spatial_dim]).
  Returns:
    Matrix of distances; ndarray(shape=[...]).
  """
  return np.sqrt(square_distance(dR))


def periodic_shift(side, R, dR):
  """Shifts positions, wrapping them back within a periodic hypercube."""
  return np.mod(R + dR, side)


"""
Spaces

  The following functions provide the necessary transformations to perform
  simulations in different spaces.

  Spaces are tuples containing:
      displacement_fn(Ra, Rb, **kwargs): Computes displacements between pairs of
        particles. Ra and Rb should be ndarrays of shape [N, spatial_dim] and
        [M, spatial_dim] respectively. Returns an ndarray of shape
        [N, M, spatial_dim].

      shift_fn(R, dR, **kwargs): Moves points at position R by an amount dR.

  In each case, **kwargs is optional keyword arguments that can be supplied to
  the different functions. In cases where the space features time dependence
  this will be passed through a "t" keyword argument.
"""


def free():
  """Free boundary conditions."""
  def displacement_fn(Ra, Rb, **unused_kwargs):
    return pairwise_displacement(Ra, Rb)
  def shift_fn(R, dR, **unused_kwargs):
    return R + dR
  return displacement_fn, shift_fn


def periodic(side, wrapped=True):
  """Periodic boundary conditions on a hypercube of sidelength side.

  Args:
    side: Either a float or an ndarray of shape [spatial_dimension] specifying
      the size of each side of the periodic box.
    wrapped: A boolean specifying whether or not particle positions are
      remapped back into the box after each step
  Returns:
    (displacement_fn, shift_fn) tuple.
  """
  def displacement_fn(Ra, Rb, **unused_kwargs):
    return periodic_displacement(side, pairwise_displacement(Ra, Rb))
  if wrapped:
    def shift_fn(R, dR, **unused_kwargs):
      return periodic_shift(side, R, dR)
  else:
    def shift_fn(R, dR, **unused_kwargs):
      return R + dR
  return displacement_fn, shift_fn


def _check_time_dependence(t):
  if t is None:
    msg = ('Space has time-dependent transform, but no time has been '
           'provided. (t = {})'.format(t))
    raise ValueError(msg)


def periodic_general(T, wrapped=True):
  """Periodic boundary conditions on a parallelepiped.

  This function defines a simulation on a parellelepiped formed by applying an
  affine transformation to the unit hypercube [0, 1]^spatial_dimension.

  When using periodic_general, particles positions should be stored in the unit
  hypercube. To get real positions from the simulation you should call
  R_sim = space.transform(T, R_unit_cube).

  The affine transformation can feature time dependence (if T is a function
  instead of a scalar). In this case the resulting space will also be time
  dependent. This can be useful for simulating systems under mechanical strain.

  Args:
    T: An affine transformation.
       Either:
         1) An ndarray of shape [spatial_dim, spatial_dim].
         2) A function that takes floating point times and produces ndarrays of
            shape [spatial_dim, spatial_dim].
    wrapped: A boolean specifying whether or not particle positions are
      remapped back into the box after each step
  Returns:
    (displacement_fn, shift_fn) tuple.
  """
  if callable(T):
    def displacement(Ra, Rb, t=None, **unused_kwargs):
      _check_time_dependence(t)
      dR = periodic_displacement(f32(1.0), pairwise_displacement(Ra, Rb))
      return transform(T(t), dR)
    # Can we cache the inverse? @Optimization
    if wrapped:
      def shift(R, dR, t=None, **unused_kwargs):
        _check_time_dependence(t)
        return periodic_shift(f32(1.0), R, transform(_small_inverse(T(t)), dR))
    else:
      def shift(R, dR, t=None, **unused_kwargs):
        _check_time_dependence(t)
        return R + transform(_small_inverse(T(t)), dR)
  else:
    T_inv = _small_inverse(T)
    def displacement(Ra, Rb, **unused_kwargs):
      dR = periodic_displacement(f32(1.0), pairwise_displacement(Ra, Rb))
      return transform(T, dR)
    if wrapped:
      def shift(R, dR, **unused_kwargs):
        return periodic_shift(f32(1.0), R, transform(T_inv, dR))
    else:
      def shift(R, dR, **unused_kwargs):
        return R + transform(T_inv, dR)
  return displacement, shift


def metric(displacement):
  """Takes a displacement function and creates a metric."""
  return lambda Ra, Rb, **kwargs: distance(displacement(Ra, Rb, **kwargs))
