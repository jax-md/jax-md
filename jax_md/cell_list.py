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

  Attributes:
    particle_count: Integer specifying the total number of particles in the
      system.
    spatial_dimension: Integer.
    cell_count: Integer specifying total number of cells in the grid.
    cell_position_buffer: ndarray of floats of shape
      [cell_count, buffer_size, spatial_dimension] containing the position of
      particles in each cell.
    cell_species_buffer: ndarray of integers of shape [cell_count, buffer_size]
      specifying the species of each particle in the grid.
    cell_id_buffer: ndarray of integers of shape [cell_Count, buffer_size]
      specifying the id of each particle in the grid such that the positions may
      copied back into an ndarray of shape [particle_count, spatial_dimension].
      We use the convention that cell_id_buffer contains the true id for
      particles that came from the cell in question, and contains
      particle_count + 1 for particles that were copied from the halo cells.
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
    flat_cells_per_side = np.reshape(cells_per_side, (-1,))
    for cells in flat_cells_per_side:
      if cells < 3:
        raise ValueError(
            ('Box must be at least 3x the size of the grid spacing in each '
             'dimension.'))

    cell_count = reduce(mul, flat_cells_per_side, 1)
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
      np.issubdtype(R, np.floating)):
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


def _estimate_cell_capacity(R, box_size, cell_size):
  # TODO(schsam): We might want to do something more sophisticated here or at
  # least expose this constant.
  excess_storage_fraction = 1.1
  spatial_dim = R.shape[-1]
  cell_capacity = np.max(count_cell_filling(R, box_size, cell_size))
  return int(cell_capacity * excess_storage_fraction)


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


def cell_list(box_size, minimum_cell_size, cell_capacity_or_example_R):
  if isinstance(box_size, np.ndarray) and len(box_size.shape) == 1:
    box_size = np.reshape(box_size, (1, -1))

  cell_capacity = cell_capacity_or_example_R
  if _is_variable_compatible_with_positions(cell_capacity):
    cell_capacity = _estimate_cell_capacity(
      cell_capacity, box_size, minimum_cell_size)
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
      cell_kwargs[k] = _unflatten_cell_buffer(cell_kwargs[k], cells_per_side, dim)

    return CellList(cell_R, cell_id, cell_kwargs)
  return build_cells

def verlet_list(displacement_fn, cell_list_fn, cutoff):
  def verlet_list_fn(R, **kwargs):
    cl = cell_list_fn(R, **kwargs)

    N = R.shape[0]
    dim = cl.R_buffer.shape[-1]

    d = space.map_product(displacement_fn)
    d = vmap(d, (None, 0))

    R = cl.R_buffer
    idx = cl.id_buffer

    R_neighbors = [R]
    idx_neighbors = [idx]

    for dindex in _neighboring_cells(dim):
      if onp.all(dindex == 0):
        continue
      R_neighbors += [_shift_array(R, dindex)]
      idx_neighbors += [_shift_array(idx, dindex)]

    R_neighbors = np.concatenate(R_neighbors, axis=-2)
    R_neighbors = R_neighbors[..., np.newaxis, :, :]
    R_neighbors = np.broadcast_to(
        R_neighbors, R.shape[:-1] + R_neighbors.shape[-2:])

    idx_neighbors = np.concatenate(idx_neighbors, axis=-2)
    idx_neighbors = idx_neighbors[..., np.newaxis, :, :]
    idx_neighbors = np.broadcast_to(
        idx_neighbors, idx.shape[:-1] + idx_neighbors.shape[-2:])

    def copy_values_from_cell(value, cell_value, cell_id):
      scatter_indices = np.reshape(cell_id, (-1,))
      cell_value = np.reshape(cell_value, (-1,) + cell_value.shape[-2:])
      return ops.index_update(value, scatter_indices, cell_value)

    # NOTE(schsam): Currently, this makes a verlet list that is larger than
    # needed since the idx buffer inherets its size from the cell-list. In
    # three-dimensions this seems to translate into an occupancy of ~70%. We can
    # make this more efficient by shrinking the verlet list at the cost of
    # another sort. However, this seems possibly less efficient than just
    # computing everything.

    idx_verlet_list = np.zeros((N + 1,) + idx_neighbors.shape[-2:], np.int32)
    idx_verlet_list = copy_values_from_cell(idx_verlet_list, idx_neighbors, idx)
    return idx_verlet_list[:-1]
  return verlet_list_fn
