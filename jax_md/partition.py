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

"""Code to transform functions on individual tuples of particles to sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce, partial
from collections import namedtuple
import math
from operator import mul

import numpy as onp

from jax import lax, ops, vmap, eval_shape
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
import jax.numpy as np

from jax_md import quantity, space
from jax_md.util import *


class CellList(namedtuple(
    'CellList', [
        'R_buffer',
        'id_buffer',
        'kwarg_buffers'
    ])):
  """Stores the spatial partition of a system into a cell list.

  See cell_list(...) for details on the construction / specification.

  Cell list buffers all have a common shape, S, where:
    S = [cell_count_x, cell_count_y, cell_capacity]
  or
    S = [cell_count_x, cell_count_y, cell_count_z, cell_capacity]
  in two- and three-dimensions respectively. It is assumed that each cell has
  the same capacity.

  Attributes:
    R_buffer: An ndarray of floating point positions with shape
      S + [spatial_dimension]. 
    id_buffer: An ndarray of int32 particle ids of shape S. Note that empty
      slots are specified by id = N where N is the number of particles in the
      system.
    kwarg_buffers: A dictionary of ndarrays of shape S + [...]. This contains
      side data placed into the cell list.
  """

  def __new__(cls, position_buffer, id_buffer, kwarg_buffers):
    return super(CellList, cls).__new__(
      cls, position_buffer, id_buffer, kwarg_buffers)

 
def _cell_dimensions(spatial_dimension, box_size, minimum_cell_size):
  """Compute the number of cells-per-side and total number of cells in a box."""
  if isinstance(box_size, int) or isinstance(box_size, float):
    box_size = f32(box_size)

  # NOTE(schsam): Should we auto-cast based on box_size? I can't imagine a case
  # in which the box_size would not be accurately represented by an f32.
  if (isinstance(box_size, np.ndarray) and
      (box_size.dtype == np.int32 or box_size.dtype == np.int64)):
    box_size = f32(box_size)

  cells_per_side = np.floor(box_size / minimum_cell_size)
  cell_size = box_size / cells_per_side
  cells_per_side = np.array(cells_per_side, dtype=np.int64)

  if isinstance(box_size, np.ndarray):
    if box_size.ndim == 1 or box_size.ndim == 2:
      assert box_size.size == spatial_dimension
      flat_cells_per_side = np.reshape(cells_per_side, (-1,))
      for cells in flat_cells_per_side:
        if cells < 3:
          raise ValueError(
            ('Box must be at least 3x the size of the grid spacing in each '
             'dimension.'))
      cell_count = reduce(mul, flat_cells_per_side, 1)
    elif box_size.ndim == 0:
      cell_count = cells_per_side ** spatial_dimension
    else:
      raise ValueError('Box must either be a scalar or a vector.')
  else:
    cell_count = cells_per_side ** spatial_dimension

  return box_size, cell_size, cells_per_side, int(cell_count)


def count_cell_filling(R, box_size, minimum_cell_size):
  """Counts the number of particles per-cell in a spatial partition."""
  dim = i32(R.shape[1])
  box_size, cell_size, cells_per_side, cell_count = \
      _cell_dimensions(dim, box_size, minimum_cell_size)

  hash_multipliers = _compute_hash_constants(dim, cells_per_side)

  particle_index = np.array(R / cell_size, dtype=np.int64)
  particle_hash = np.sum(particle_index * hash_multipliers, axis=1)
  filling = np.zeros((cell_count,), dtype=np.int64)

  def count(cell_hash, filling):
    count = np.sum(particle_hash == cell_hash)
    filling = ops.index_update(filling, ops.index[cell_hash], count)
    return filling

  return lax.fori_loop(0, cell_count, count, filling)


def _is_variable_compatible_with_positions(R):
  if (isinstance(R, np.ndarray) and
      len(R.shape) == 2 and
      np.issubdtype(R.dtype, np.floating)):
    return True

  return False


def _compute_hash_constants(spatial_dimension, cells_per_side):
  if cells_per_side.size == 1:
    return np.array([[
        cells_per_side ** d for d in range(spatial_dimension)]], dtype=np.int64)
  elif cells_per_side.size == spatial_dimension:
    one = np.array([[1]], dtype=np.int32)
    cells_per_side = np.concatenate((one, cells_per_side[:, :-1]), axis=1)
    return np.array(np.cumprod(cells_per_side), dtype=np.int64)
  else:
    raise ValueError()


def _neighboring_cells(dimension):
  for dindex in onp.ndindex(*([3] * dimension)):
    yield np.array(dindex, dtype=np.int64) - 1


def _estimate_cell_capacity(R, box_size, cell_size, buffer_size_multiplier):
  # TODO(schsam): We might want to do something more sophisticated here or at
  # least expose this constant.
  spatial_dim = R.shape[-1]
  cell_capacity = np.max(count_cell_filling(R, box_size, cell_size))
  return int(cell_capacity * buffer_size_multiplier)


def _unflatten_cell_buffer(arr, cells_per_side, dim):
  if (isinstance(cells_per_side, int) or
      isinstance(cells_per_side, float) or
      (isinstance(cells_per_side, np.ndarray) and not cells_per_side.shape)):
    cells_per_side = (int(cells_per_side),) * dim
  elif isinstance(cells_per_side, np.ndarray) and len(cells_per_side.shape) == 1:
    cells_per_side = tuple([int(x) for x in cells_per_side[::-1]])
  elif isinstance(cells_per_side, np.ndarray) and len(cells_per_side.shape) == 2:
    cells_per_side = tuple([int(x) for x in cells_per_side[0][::-1]])
  else:
    raise ValueError() # TODO
  return np.reshape(arr, cells_per_side + (-1,) + arr.shape[1:])


def _shift_array(arr, dindex):
  if len(dindex) == 2:
    dx, dy = dindex
    dz = 0
  elif len(dindex) == 3:
    dx, dy, dz = dindex

  if dx < 0:
    arr = np.concatenate((arr[1:], arr[:1]))
  elif dx > 0:
    arr = np.concatenate((arr[-1:], arr[:-1]))

  if dy < 0:
    arr = np.concatenate((arr[:, 1:], arr[:, :1]), axis=1)
  elif dy > 0:
    arr = np.concatenate((arr[:, -1:], arr[:, :-1]), axis=1)

  if dz < 0:
    arr = np.concatenate((arr[:, :, 1:], arr[:, :, :1]), axis=2)
  elif dz > 0:
    arr = np.concatenate((arr[:, :, -1:], arr[:, :, :-1]), axis=2)

  return arr


def _vectorize(f, dim):
  if dim == 2:
    return vmap(vmap(f, 0, 0), 0, 0)
  elif dim == 3:
    return vmap(vmap(vmap(f, 0, 0), 0, 0), 0, 0)
  raise ValueError('Cell list only supports 2d or 3d.')


def cell_list(
    box_size, minimum_cell_size,
    cell_capacity_or_example_R, buffer_size_multiplier=1.1):
  r"""Returns a function that partitions point data spatially. 

  Given a set of points {x_i \in R^d} with associated data {k_i \in R^m} it is
  often useful to partition the points / data spatially. A simple partitioning
  that can be implemented efficiently within XLA is a dense partition into a
  uniform grid called a cell list.

  Since XLA requires that shapes be statically specified, we allocate fixed
  sized buffers for each cell. The size of this buffer can either be specified
  manually or it can be estimated automatically from a set of positions. Note,
  if the distribution of points changes significantly it is likely the buffer
  the buffer sizes will have to be adjusted.

  This partitioning will likely form the groundwork for parallelizing
  simulations over different accelerators.

  Args:
    box_size: A float or an ndarray of shape [spatial_dimension] specifying the
      size of the system. Note, this code is written for the case where the
      boundaries are periodic. If this is not the case, then the current code
      will be slightly less efficient.
    minimum_cell_size: A float specifying the minimum side length of each cell.
      Cells are enlarged so that they exactly fill the box.
    cell_capacity_or_example_R: Either an integer specifying the size
      number of particles that can be stored in each cell or an ndarray of
      positions of shape [particle_count, spatial_dimension] that is used to
      estimate the cell_capacity.
    buffer_size_multiplier: A floating point multiplier that multiplies the
      estimated cell capacity to allow for fluctuations in the maximum cell
      occupancy.
  Returns:
    A function `cell_list_fn(R, **kwargs)` that partitions positions, `R`, and
    side data specified by kwargs into a cell list. Returns a CellList
    containing the partition.
  """

  if isinstance(box_size, np.ndarray) and len(box_size.shape) == 1:
    box_size = np.reshape(box_size, (1, -1))

  cell_capacity = cell_capacity_or_example_R
  if _is_variable_compatible_with_positions(cell_capacity):
    cell_capacity = _estimate_cell_capacity(
      cell_capacity, box_size, minimum_cell_size, buffer_size_multiplier)
  elif not isinstance(cell_capacity, int):
    msg = (
        'cell_capacity_or_example_positions must either be an integer '
        'specifying the cell capacity or a set of positions that will be used '
        'to estimate a cell capacity. Found {}.'.format(type(cell_capacity))
        )
    raise ValueError(msg)

  def build_cells(R, **kwargs):
    N = R.shape[0]
    dim = R.shape[1]

    if dim != 2 and dim != 3:
      # NOTE(schsam): Do we want to check this in compute_fn as well?
      raise ValueError(
          'Cell list spatial dimension must be 2 or 3. Found {}'.format(dim))

    neighborhood_tile_count = 3 ** dim

    _, cell_size, cells_per_side, cell_count = \
        _cell_dimensions(dim, box_size, minimum_cell_size)

    hash_multipliers = _compute_hash_constants(dim, cells_per_side)

    # Create cell list data.
    particle_id = lax.iota(np.int64, N)
    # NOTE(schsam): We use the convention that particles that are successfully,
    # copied have their true id whereas particles empty slots have id = N.
    # Then when we copy data back from the grid, copy it to an array of shape
    # [N + 1, output_dimension] and then truncate it to an array of shape
    # [N, output_dimension] which ignores the empty slots.
    mask_id = np.ones((N,), np.int64) * N
    cell_R = np.zeros((cell_count * cell_capacity, dim), dtype=R.dtype)
    cell_id = N * np.ones((cell_count * cell_capacity, 1), dtype=i32)

    # It might be worth adding an occupied mask. However, that will involve
    # more compute since often we will do a mask for species that will include
    # an occupancy test. It seems easier to design around this empty_data_value
    # for now and revisit the issue if it comes up later.
    empty_kwarg_value = 10 ** 5
    cell_kwargs = {}
    for k, v in kwargs.items():
      if not isinstance(v, np.ndarray):
        raise ValueError((
          'Data must be specified as an ndarry. Found "{}" with '
          'type {}'.format(k, type(v))))
      if v.shape[0] != R.shape[0]:
        raise ValueError(
          ('Data must be specified per-particle (an ndarray with shape '
           '(R.shape[0], ...)). Found "{}" with shape {}'.format(k, v.shape)))
      kwarg_shape = v.shape[1:] if v.ndim > 1 else (1,)
      cell_kwargs[k] = empty_kwarg_value * np.ones(
        (cell_count * cell_capacity,) + kwarg_shape, v.dtype)

    indices = np.array(R / cell_size, dtype=i32)
    hashes = np.sum(indices * hash_multipliers, axis=1)

    # Copy the particle data into the grid. Here we use a trick to allow us to
    # copy into all cells simultaneously using a single lax.scatter call. To do
    # this we first sort particles by their cell hash. We then assign each
    # particle to have a cell id = hash * cell_capacity + grid_id where grid_id
    # is a flat list that repeats 0, .., cell_capacity. So long as there are
    # fewer than cell_capacity particles per cell, each particle is guarenteed
    # to get a cell id that is unique.
    sort_map = np.argsort(hashes)
    sorted_R = R[sort_map]
    sorted_hash = hashes[sort_map]
    sorted_id = particle_id[sort_map]

    sorted_kwargs = {}
    for k, v in kwargs.items():
      sorted_kwargs[k] = v[sort_map]

    sorted_cell_id = np.mod(lax.iota(np.int64, N), cell_capacity)
    sorted_cell_id = sorted_hash * cell_capacity + sorted_cell_id

    cell_R = ops.index_update(cell_R, sorted_cell_id, sorted_R)
    sorted_id = np.reshape(sorted_id, (N, 1))
    cell_id = ops.index_update(
        cell_id, sorted_cell_id, sorted_id)
    cell_R = _unflatten_cell_buffer(cell_R, cells_per_side, dim)
    cell_id = _unflatten_cell_buffer(cell_id, cells_per_side, dim)

    for k, v in sorted_kwargs.items():
      if v.ndim == 1:
        v = np.reshape(v, v.shape + (1,))
      cell_kwargs[k] = ops.index_update(cell_kwargs[k], sorted_cell_id, v)
      cell_kwargs[k] = _unflatten_cell_buffer(
        cell_kwargs[k], cells_per_side, dim)

    return CellList(cell_R, cell_id, cell_kwargs)
  return build_cells


def _displacement_or_metric_to_metric_sq(displacement_or_metric):
  """Checks whether or not a displacement or metric was provided."""
  for dim in range(1, 4):
    try:
      R = ShapedArray((dim,), f32)
      dR_or_dr = eval_shape(displacement_or_metric, R, R, t=0)
      if len(dR_or_dr.shape) == 0:
        return lambda Ra, Rb, **kwargs: \
          displacement_or_metric(Ra, Rb, **kwargs) ** 2
      else:
        return lambda Ra, Rb, **kwargs: space.square_distance(
          displacement_or_metric(Ra, Rb, **kwargs))
    except TypeError:
      continue
    except ValueError:
      continue
  raise ValueError(
    'Canonicalize displacement not implemented for spatial dimension larger'
    'than 4.')


def neighbor_list(
    displacement_or_metric, box_size, cutoff, example_R,
    buffer_size_multiplier=1.1, cell_size=None, **static_kwargs):
  """Returns a function that builds a list neighbors for each point.

  Since XLA requires fixed shape, we use example point configurations to
  estimate the maximum number of points within a neighborhood. However, if the
  configuration changes substantially over time it might be necessary to
  revise this estimate.

  Args:
    displacement: A function `d(R_a, R_b)` that computes the displacement
      between pairs of points.
    box_size: Either a float specifying the size of the box or an array of
      shape [spatial_dim] specifying the box size in each spatial dimension.
    cutoff: A scalar specifying the neighborhood radius.
    example_R: An ndarray of example points of shape [point_count, spatial_dim]
      used to estimate a maximum neighborhood size.
    buffer_size_multiplier: A floating point scalar specifying the fractional
      increase in maximum neighborhood occupancy we allocate compared with the
      maximum in the example positions.
    cell_size: A scalar specifying the size of cells in the cell list used
      in an intermediate step.
    **static_kwargs: kwargs that get threaded through the calculation of
      example positions.

  Returns:
    An ndarray of shape [point_count, maximum_neighbors_per_point] of ids
    specifying points in the neighborhood of each point. Empty elements are
    given an id = point_count.
  """
  box_size = f32(box_size)

  cutoff_sq = cutoff ** 2
  metric_sq = _displacement_or_metric_to_metric_sq(displacement_or_metric)

  if cell_size is None:
    cell_size = cutoff
  cell_list_fn = cell_list(
    box_size, cell_size, example_R, buffer_size_multiplier)

  def neighbor_list_candidate_fn(R, **kwargs):
    cl = cell_list_fn(R)

    N, dim = R.shape

    R = cl.R_buffer
    idx = cl.id_buffer

    cell_idx = [idx]

    for dindex in _neighboring_cells(dim):
      if onp.all(dindex == 0):
        continue
      cell_idx += [_shift_array(idx, dindex)]

    cell_idx = np.concatenate(cell_idx, axis=-2)
    cell_idx = cell_idx[..., np.newaxis, :, :]
    cell_idx = np.broadcast_to(cell_idx, idx.shape[:-1] + cell_idx.shape[-2:])

    def copy_values_from_cell(value, cell_value, cell_id):
      scatter_indices = np.reshape(cell_id, (-1,))
      cell_value = np.reshape(cell_value, (-1,) + cell_value.shape[-2:])
      return ops.index_update(value, scatter_indices, cell_value)

    # NOTE(schsam): Currently, this makes a verlet list that is larger than
    # needed since the idx buffer inherets its size from the cell-list. In
    # three-dimensions this seems to translate into an occupancy of ~70%. We
    # can make this more efficient by shrinking the verlet list at the cost of
    # another sort. However, this seems possibly less efficient than just
    # computing everything.

    neighbor_idx = np.zeros((N + 1,) + cell_idx.shape[-2:], np.int32)
    neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx)
    return neighbor_idx[:-1, :, 0]

  # Use the example positions to estimate the maximum occupancy of the verlet
  # list.
  d_ex = partial(metric_sq, **static_kwargs)
  d_ex = vmap(vmap(d_ex, (None, 0)))
  N = example_R.shape[0]
  example_idx = neighbor_list_candidate_fn(example_R)
  example_neigh_R = example_R[example_idx]
  example_neigh_dR = d_ex(example_R, example_neigh_R)
  mask = np.logical_and(example_neigh_dR < cutoff_sq, example_idx < N)
  max_occupancy = np.max(np.sum(mask, axis=1))
  max_occupancy = int(max_occupancy * buffer_size_multiplier)

  def neighbor_list_fn(R, **kwargs):
    idx = neighbor_list_candidate_fn(R, **kwargs)

    d = partial(metric_sq, **kwargs)
    d = vmap(vmap(d, (None, 0)))

    neigh_R = R[idx]
    dR = d(R, neigh_R)

    argsort = np.argsort(
      f32(1) - np.logical_and(dR < cutoff_sq, idx < N), axis=1)
    # TODO(schsam): Error checking for list exceeding maximum occupancy.
    idx = np.take_along_axis(idx, argsort, axis=1)
    idx = idx[:, :max_occupancy]

    self_mask = idx == np.reshape(np.arange(idx.shape[0]), (idx.shape[0], 1))
    idx = np.where(self_mask, idx.shape[0], idx)

    return idx

  return neighbor_list_fn
