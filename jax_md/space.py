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

"""Spaces in which particles are simulated.

Spaces are pairs of functions containing:

  * `displacement_fn(Ra, Rb, **kwargs)`
    Computes displacements between pairs of particles. Ra and Rb should
    be ndarrays of shape [spatial_dim]. Returns an ndarray of shape
    [spatial_dim]. To compute the displacement over more than one particle
    at a time see the `map_product`, `map_bond`, and `map_neighbor` functions.

  * `shift_fn(R, dR, **kwargs)` Moves points at position R by an amount dR.

Spaces can accept keyword arguments allowing the space to be changed over the
course of a simulation. For an example of this use see `periodic_general`.

Although displacement functions are compute the displacement between two
points, it is often useful to compute displacements between multiple particles
in a vectorized fashion. To do this we provide three functions: `map_product`,
`map_bond`, and `map_neighbor`.
  * `map_pair` computes displacements betweeen all pairs of points such that if
    Ra has shape [n, spatial_dim] and Rb has shape `[m, spatial_dim]` then the
    output has shape `[n, m, spatial_dim]`.
  * `map_bond` computes displacements between all points in a list such that if
    Ra has shape [n, spatial_dim] and Rb has shape [m, spatial_dim] then the
    otuput has shape [n, spatial_dim].
  * `map_neighbor` computes displacements between points and all of their
    neighbors such that if Ra has shape [n, spatial_dim] and Rb has shape
    [n, neighbors, spatial_dim] then the output has shape
    [n, neighbors, spatial_dim].
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


def inverse(box: Box) -> Box:
  """Compute the inverse of an affine transformation."""
  if jnp.isscalar(box) or box.size == 1:
    return 1 / box
  elif box.ndim == 1:
    return 1 / box
  elif box.ndim == 2:
    return jnp.linalg.inv(box)
  raise ValueError(('Box must be either: a scalar, a vector, or a matrix. '
                    f'Found {box}.'))


def _get_free_indices(n: int) -> str:
  return ''.join([chr(ord('a') + i) for i in range(n)])


@custom_jvp
def transform(box: Box, R: Array) -> Array:
  """Apply an affine transformation to positions.

  See `periodic_general` for a description of the semantics of `box`.

  Args:
    box: An affine transformation described in `periodic_general`.
    R: Array of positions. Should have  shape `(..., spatial_dimension)`.

  Returns:
    A transformed array positions of shape `(..., spatial_dimension)`.
  """
  if jnp.isscalar(box) or box.size == 1:
    return R * box
  elif box.ndim == 1:
    indices = _get_free_indices(R.ndim - 1) + 'i'
    return jnp.einsum(f'i,{indices}->{indices}', box, R)
  elif box.ndim == 2:
    free_indices = _get_free_indices(R.ndim - 1)
    left_indices = free_indices + 'j'
    right_indices = free_indices + 'i'
    return jnp.einsum(f'ij,{left_indices}->{right_indices}', box, R)
  raise ValueError(('Box must be either: a scalar, a vector, or a matrix. '
                    f'Found {box}.'))


@transform.defjvp
def transform_jvp(primals, tangents):
  box, R = primals
  dbox, dR = tangents
  return (transform(box, R), dR + transform(dbox, R))


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
    if 'box' in unused_kwargs:
      raise ValueError(('`space.periodic` does not accept a box argument.'
                        'Perhaps you maent to use `space.periodic_general`?'))
    return periodic_displacement(side, pairwise_displacement(Ra, Rb))
  if wrapped:
    def shift_fn(R: Array, dR: Array, **unused_kwargs) -> Array:
      if 'box' in unused_kwargs:
        raise ValueError(('`space.periodic` does not accept a box argument.'
                          'Perhaps you maent to use `space.periodic_general`?'))
      return periodic_shift(side, R, dR)
  else:
    def shift_fn(R: Array, dR: Array, **unused_kwargs) -> Array:
      if 'box' in unused_kwargs:
        raise ValueError(('`space.periodic` does not accept a box argument.'
                          'Perhaps you maent to use `space.periodic_general`?'))
      return R + dR
  return displacement_fn, shift_fn


def periodic_general(box: Box,
                     fractional_coordinates: bool=True,
                     wrapped: bool=True) -> Space:
  """Periodic boundary conditions on a parallelepiped.

  This function defines a simulation on a parallelepiped, $X$, formed by
  applying an affine transformation, $T$, to the unit hypercube
  $U = [0, 1]^d$ along with periodic boundary conditions across all
  of the faces.

  Formally, the space is defined such that $X = {Tu : u \in [0, 1]^d}$.

  The affine transformation, $T$, can be specified in a number of different
  ways. For a parallelepiped that is: 1) a cube of side length $L$, the affine
  transformation can simply be a scalar; 2) an orthorhombic unit cell can be
  specified by a vector `[Lx, Ly, Lz]` of lengths for each axis; 3) a general
  triclinic cell can be specified by an upper triangular matrix.

  There are a number of ways to parameterize a simulation on $X$.
  `periodic_general` supports two parametrizations of $X$ that can be selected
  using the `fractional_coordinates` keyword argument.
    1) When `fractional_coordinates=True`, particle positions are stored in the
       unit cube, $u\in U$. Here, the displacement function computes the
       displacement between $x, y \in X$ as $d_X(x, y) = Td_U(u, v)$ where
       $d_U$ is the displacement function on the unit cube, $U$, $x = Tu$, and
       $v = Tv$ with $u, v\in U$. The derivative of the displacement function
       is defined so that derivatives live in $X$ (as opposed to being
       backpropagated to $U$). The shift function, `shift_fn(R, dR)` is defined
       so that $R$ is expected to lie in $U$ while $dR$ should lie in $X$. This
       combination enables code such as `shift_fn(R, force_fn(R))` to work as
       intended.

    2) When `fractional_coordinates=False`, particle positions are stored in
       the parallelepiped $X$. Here, for $x, y\in X$, the displacement function
       is defined as $d_X(x, y) = Td_U(T^{-1}x, T^{-1}y)$. Since there is an
       extra multiplication by $T^{-1}$, this parameterization is typically
       slower than `fractional_coordinates=False`. As in 1), the displacement
       function is defined to compute derivatives in $X$. The shift function
       is defined so that $R$ and $dR$ should both lie in $X$.


  Example:
  ```python
    from jax import random
    side_length = 10.0
    disp_frac, shift_frac = periodic_general(side_length,
                                             fractional_coordinates=True)
    disp_real, shift_real = periodic_general(side_length,
                                             fractional_coordinates=False)

    # Instantiate random positions in both parameterizations.
    R_frac = random.uniform(random.PRNGKey(0), (4, 3))
    R_real = side_length * R_frac

    # Make some shfit vectors.
    dR = random.normal(random.PRNGKey(0), (4, 3))

    disp_real(R_real[0], R_real[1]) == disp_frac(R_frac[0], R_frac[1])
    transform(side_length, shift_frac(R_frac, 1.0)) == shift_real(R_real, 1.0)
  ```

  It is often desirable to deform a simulation cell either: using a finite
  deformation during a simulation, or using an infinitesimal deformation while
  computing elastic constants. To do this using fractional coordinates, we can
  supply a new affine transformation as `displacement_fn(Ra, Rb, box=new_box)`.
  When using real coordinates, we can specify positions in a space $X$ defined
  by an affine transformation $T$ and compute displacements in a deformed space
  $X'$ defined by an affine transformation $T'$. This is done by writing
  `displacement_fn(Ra, Rb, new_box=new_box)`.

  There are a few caveats when using `periodic_general`. `periodic_general`
  uses the minimum image convention, and so it will fail for potentials whose
  cutoff is longer than the half of the side-length of the box. It will also
  fail to find the correct image when the box is too deformed. We hope to add a
  more robust box for small simulations soon (TODO) along with better error
  checking. In the meantime caution is recommended.

  Args:
    box: A `(spatial_dim, spatial_dim)` affine transformation.
    fractional_coordinates: A boolean specifying whether positions are stored
      in the parallelepiped or the unit cube.
    wrapped: A boolean specifying whether or not particle positions are
      remapped back into the box after each step
  Returns:
    (displacement_fn, shift_fn) tuple.
  """
  inv_box = inverse(box)

  def displacement_fn(Ra, Rb, **kwargs):
    _box, _inv_box = box, inv_box

    if 'box' in kwargs:
      _box = kwargs['box']

      if not fractional_coordinates:
        _inv_box = inverse(_box)

    if 'new_box' in kwargs:
      _box = kwargs['new_box']

    if not fractional_coordinates:
      Ra = transform(_inv_box, Ra)
      Rb = transform(_inv_box, Rb)

    dR = periodic_displacement(f32(1.0), pairwise_displacement(Ra, Rb))
    return transform(_box, dR)

  def u(R, dR):
    if wrapped:
      return periodic_shift(f32(1.0), R, dR)
    return R + dR

  def shift_fn(R, dR, **kwargs):
    if not fractional_coordinates and not wrapped:
      return R + dR

    _box, _inv_box = box, inv_box
    if 'box' in kwargs:
      _box = kwargs['box']
      _inv_box = inverse(_box)

    if 'new_box' in kwargs:
      _box = kwargs['new_box']

    dR = transform(_inv_box, dR)
    if not fractional_coordinates:
      R = transform(_inv_box, R)

    R = u(R, dR)

    if not fractional_coordinates:
      R = transform(_box, R)
    return R

  return displacement_fn, shift_fn


def metric(displacement: DisplacementFn) -> MetricFn:
  """Takes a displacement function and creates a metric."""
  return lambda Ra, Rb, **kwargs: distance(displacement(Ra, Rb, **kwargs))


def map_product(metric_or_displacement: DisplacementOrMetricFn
                ) -> DisplacementOrMetricFn:
  """Vectorizes a metric or displacement function over all pairs."""
  return vmap(vmap(metric_or_displacement, (0, None), 0), (None, 0), 0)


def map_bond(metric_or_displacement: DisplacementOrMetricFn
             ) -> DisplacementOrMetricFn:
  """Vectorizes a metric or displacement function over bonds."""
  return vmap(metric_or_displacement, (0, 0), 0)


def map_neighbor(metric_or_displacement: DisplacementOrMetricFn
                 ) -> DisplacementOrMetricFn:
  """Vectorizes a metric or displacement function over neighborhoods."""
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

