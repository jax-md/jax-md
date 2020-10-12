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

Spaces are pairs of functions containing:

  * `displacement_fn(Ra, Rb, **kwargs)`
    Computes displacements between pairs of particles. Ra and Rb should
    be ndarrays of shape [spatial_dim]. Returns an ndarray of shape
    [spatial_dim]. To compute displacements over sets of positions, use
    vmap. Soon (TODO) we will add convenience functions to do this where
    needed.

  * `shift_fn(R, dR, **kwargs)` Moves points at position R by an amount dR.

In each case, **kwargs is optional keyword arguments that can be supplied to
the different functions. In cases where the space features time dependence
this will be passed through a "t" keyword argument.
"""

from typing import Callable, Union, Tuple, Any

from jax.abstract_arrays import ShapedArray

from jax import eval_shape
from jax import vmap
from jax import custom_jvp

import jax

import jax.numpy as jnp

from jax_md.util import Array
from jax_md.util import f32
from jax_md.util import f64
from jax_md.util import safe_mask


# Types


DisplacementFn = Callable[[Array, Array], Array]
MetricFn = Callable[[Array, Array], float]
DisplacementOrMetricFn = Union[DisplacementFn, MetricFn]

ShiftFn = Callable[[Array, Array], Array]

Space = Tuple[DisplacementFn, ShiftFn]

Box = Union[float, Array]


# Primitive Spatial Transforms


def _check_transform_shapes(T: Array, v: Array=None):
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


def _small_inverse(T: Array) -> Array:
  """Compute the inverse of a small matrix."""
  _check_transform_shapes(T)
  dim = T.shape[0]
  # TODO(schsam): Check whether matrices are singular. @ErrorChecking
  return jnp.linalg.inv(T)


@custom_jvp
def transform(T: Array, v: Array) -> Array:
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
  return jnp.dot(v, T)


@transform.defjvp
def transform_jvp(primals: Tuple[Array, Array],
                  tangents: Tuple[Array, Array]) -> Tuple[Array, Array]:
  T, v = primals
  dT, dv = tangents
  return transform(T, v), dv


def pairwise_displacement(Ra: Array, Rb: Array) -> Array:
  """Compute a matrix of pairwise displacements given two sets of positions.

  Args:
    Ra: Vector of positions; ndarray(shape=[spatial_dim]).
    Rb: Vector of positions; ndarray(shape=[spatial_dim]).

  Returns:
    Matrix of displacements; ndarray(shape=[spatial_dim]).
  """
  if len(Ra.shape) != 1:
    msg = (
      'Can only compute displacements between vectors. To compute '
      'displacements between sets of vectors use vmap or TODO.'
    )
    raise ValueError(msg)

  if Ra.shape != Rb.shape:
    msg = 'Can only compute displacement between vectors of equal dimension.'
    raise ValueError(msg)

  return Ra - Rb


def periodic_displacement(side: Box, dR: Array) -> Array:
  """Wraps displacement vectors into a hypercube.

  Args:
    side: Specification of hypercube size. Either,
      (a) float if all sides have equal length.
      (b) ndarray(spatial_dim) if sides have different lengths.
    dR: Matrix of displacements; ndarray(shape=[..., spatial_dim]).
  Returns:
    Matrix of wrapped displacements; ndarray(shape=[..., spatial_dim]).
  """
  return jnp.mod(dR + side * f32(0.5), side) - f32(0.5) * side


def square_distance(dR: Array) -> Array:
  """Computes square distances.

  Args:
    dR: Matrix of displacements; ndarray(shape=[..., spatial_dim]).
  Returns:
    Matrix of squared distances; ndarray(shape=[...]).
  """
  return jnp.sum(dR ** 2, axis=-1)


def distance(dR: Array) -> Array:
  """Computes distances.

  Args:
    dR: Matrix of displacements; ndarray(shape=[..., spatial_dim]).
  Returns:
    Matrix of distances; ndarray(shape=[...]).
  """
  dr = square_distance(dR)
  return safe_mask(dr > 0, jnp.sqrt, dr)


def periodic_shift(side: Box, R: Array, dR: Array) -> Array:
  """Shifts positions, wrapping them back within a periodic hypercube."""
  return jnp.mod(R + dR, side)


""" Spaces """


def free() -> Space:
  """Free boundary conditions."""
  def displacement_fn(Ra: Array, Rb: Array, **unused_kwargs) -> Array:
    return pairwise_displacement(Ra, Rb)
  def shift_fn(R: Array, dR: Array, **unused_kwargs) -> Array:
    return R + dR
  return displacement_fn, shift_fn


def periodic(side: Box, wrapped: bool=True) -> Space:
  """Periodic boundary conditions on a hypercube of sidelength side.

  Args:
    side: Either a float or an ndarray of shape [spatial_dimension] specifying
      the size of each side of the periodic box.
    wrapped: A boolean specifying whether or not particle positions are
      remapped back into the box after each step
  Returns:
    (displacement_fn, shift_fn) tuple.
  """
  def displacement_fn(Ra: Array, Rb: Array, **unused_kwargs) -> Array:
    return periodic_displacement(side, pairwise_displacement(Ra, Rb))
  if wrapped:
    def shift_fn(R: Array, dR: Array, **unused_kwargs) -> Array:
      return periodic_shift(side, R, dR)
  else:
    def shift_fn(R: Array, dR: Array, **unused_kwargs) -> Array:
      return R + dR
  return displacement_fn, shift_fn


def periodic_general(T: Union[Array, Callable[..., Array]],
                     wrapped: bool=True) -> Space:
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
    def displacement(Ra: Array, Rb: Array, **kwargs) -> Array:
      dR = periodic_displacement(f32(1.0), pairwise_displacement(Ra, Rb))
      return transform(T(**kwargs), dR)
    # Can we cache the inverse? @Optimization
    if wrapped:
      def shift(R: Array, dR: Array, **kwargs) -> Array:
        return periodic_shift(f32(1.0),
                              R,
                              transform(_small_inverse(T(**kwargs)), dR))
    else:
      def shift(R: Array, dR: Array, **kwargs) -> Array:
        return R + transform(_small_inverse(T(**kwargs)), dR)
  else:
    T_inv = _small_inverse(T)
    def displacement(Ra: Array, Rb: Array, **unused_kwargs) -> Array:
      dR = periodic_displacement(f32(1.0), pairwise_displacement(Ra, Rb))
      return transform(T, dR)
    if wrapped:
      def shift(R: Array, dR: Array, **unused_kwargs) -> Array:
        return periodic_shift(f32(1.0), R, transform(T_inv, dR))
    else:
      def shift(R: Array, dR: Array, **unused_kwargs) -> Array:
        return R + transform(T_inv, dR)
  return displacement, shift


def metric(displacement: DisplacementFn) -> MetricFn:
  """Takes a displacement function and creates a metric."""
  return lambda Ra, Rb, **kwargs: distance(displacement(Ra, Rb, **kwargs))


def map_product(metric_or_displacement: DisplacementOrMetricFn) -> DisplacementOrMetricFn:
  return vmap(vmap(metric_or_displacement, (0, None), 0), (None, 0), 0)


def map_bond(metric_or_displacement: DisplacementOrMetricFn) -> DisplacementOrMetricFn:
  return vmap(metric_or_displacement, (0, 0), 0)


def map_neighbor(metric_or_displacement: DisplacementOrMetricFn) -> DisplacementOrMetricFn:
  def wrapped_fn(Ra, Rb, **kwargs):
    return vmap(vmap(metric_or_displacement, (None, 0)))(-Ra, -Rb, **kwargs)
  return wrapped_fn


def canonicalize_displacement_or_metric(displacement_or_metric):
  """Checks whether or not a displacement or metric was provided."""
  for dim in range(1, 4):
    try:
      R = ShapedArray((dim,), f32)
      dR_or_dr = eval_shape(displacement_or_metric, R, R, t=0)
      if len(dR_or_dr.shape) == 0:
        return displacement_or_metric
      else:
        return metric(displacement_or_metric)
    except TypeError:
      continue
    except ValueError:
      continue
  raise ValueError(
    'Canonicalize displacement not implemented for spatial dimension larger'
    'than 4.')

