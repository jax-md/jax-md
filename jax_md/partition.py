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

from absl import logging

from functools import reduce, partial
from collections import namedtuple

from enum import Enum

from typing import Any, Callable, Optional, Dict, Tuple, Generator, Union

import math
from operator import mul

import numpy as onp

from jax import lax
from jax import ops
from jax import jit, vmap, eval_shape
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp

from jax_md import space
from jax_md import dataclasses
from jax_md import util

import jraph


# Types


Array = util.Array
f32 = util.f32
f64 = util.f64

i32 = util.i32
i64 = util.i64

Box = space.Box
DisplacementOrMetricFn = space.DisplacementOrMetricFn
MetricFn = space.MetricFn
MaskFn = Callable[[Array], Array]

# Cell List


@dataclasses.dataclass
class CellList:
  """Stores the spatial partition of a system into a cell list.

  See cell_list(...) for details on the construction / specification.
  Cell list buffers all have a common shape, S, where
    * `S = [cell_count_x, cell_count_y, cell_capacity]`
    * `S = [cell_count_x, cell_count_y, cell_count_z, cell_capacity]`
  in two- and three-dimensions respectively. It is assumed that each cell has
  the same capacity.

  Attributes:
    position_buffer: An ndarray of floating point positions with shape
      S + [spatial_dimension].
    id_buffer: An ndarray of int32 particle ids of shape S. Note that empty
      slots are specified by id = N where N is the number of particles in the
      system.
    kwarg_buffers: A dictionary of ndarrays of shape S + [...]. This contains
      side data placed into the cell list.
    did_buffer_overflow: A boolean specifying whether or not the cell list
      exceeded the maximum allocated capacity.
    cell_capacity: An integer specifying the maximum capacity of each cell in
      the cell list.
    update_fn: A function that updates the cell list at a fixed capacity.
  """
  position_buffer: Array
  id_buffer: Array
  kwarg_buffers: Dict[str, Array]

  did_buffer_overflow: Array

  cell_capacity: int = dataclasses.static_field()

  update_fn: Callable[..., 'CellList'] = \
      dataclasses.static_field()

  def update(self, position: Array, **kwargs) -> 'CellList':
    cl_data = (self.cell_capacity, self.did_buffer_overflow, self.update_fn)
    return self.update_fn(position, cl_data, **kwargs)


@dataclasses.dataclass
class CellListFns:
  allocate: Callable[..., CellList] = dataclasses.static_field()
  update: Callable[[Array, Union[CellList, int]],
                    CellList] = dataclasses.static_field()

  def __iter__(self):
    return iter((self.allocate, self.update))


def _cell_dimensions(spatial_dimension: int,
                     box_size: Box,
                     minimum_cell_size: float) -> Tuple[Box, Array, Array, int]:
  """Compute the number of cells-per-side and total number of cells in a box."""
  if isinstance(box_size, int) or isinstance(box_size, float):
    box_size = float(box_size)

  # NOTE(schsam): Should we auto-cast based on box_size? I can't imagine a case
  # in which the box_size would not be accurately represented by an f32.
  if (isinstance(box_size, onp.ndarray) and
      (box_size.dtype == i32 or box_size.dtype == i64)):
    box_size = float(box_size)

  cells_per_side = onp.floor(box_size / minimum_cell_size)
  cell_size = box_size / cells_per_side
  cells_per_side = onp.array(cells_per_side, dtype=i32)

  if isinstance(box_size, onp.ndarray):
    if box_size.ndim == 1 or box_size.ndim == 2:
      assert box_size.size == spatial_dimension
      flat_cells_per_side = onp.reshape(cells_per_side, (-1,))
      for cells in flat_cells_per_side:
        if cells < 3:
          msg = ('Box must be at least 3x the size of the grid spacing in each '
                 'dimension.')
          raise ValueError(msg)
      cell_count = reduce(mul, flat_cells_per_side, 1)
    elif box_size.ndim == 0:
      cell_count = cells_per_side ** spatial_dimension
    else:
      raise ValueError('Box must either be a scalar or a vector.')
  else:
    cell_count = cells_per_side ** spatial_dimension

  return box_size, cell_size, cells_per_side, int(cell_count)


def count_cell_filling(position: Array,
                       box_size: Box,
                       minimum_cell_size: float) -> Array:
  """Counts the number of particles per-cell in a spatial partition."""
  dim = int(position.shape[1])
  box_size, cell_size, cells_per_side, cell_count = \
      _cell_dimensions(dim, box_size, minimum_cell_size)

  hash_multipliers = _compute_hash_constants(dim, cells_per_side)

  particle_index = jnp.array(position / cell_size, dtype=i32)
  particle_hash = jnp.sum(particle_index * hash_multipliers, axis=1)

  filling = ops.segment_sum(jnp.ones_like(particle_hash),
                            particle_hash,
                            cell_count)
  return filling


def _compute_hash_constants(spatial_dimension: int,
                            cells_per_side: Array) -> Array:
  if cells_per_side.size == 1:
    return jnp.array([[cells_per_side ** d for d in range(spatial_dimension)]],
                     dtype=i32)
  elif cells_per_side.size == spatial_dimension:
    one = jnp.array([[1]], dtype=i32)
    cells_per_side = jnp.concatenate((one, cells_per_side[:, :-1]), axis=1)
    return jnp.array(jnp.cumprod(cells_per_side), dtype=i32)
  else:
    raise ValueError()


def _neighboring_cells(dimension: int) -> Generator[onp.ndarray, None, None]:
  for dindex in onp.ndindex(*([3] * dimension)):
    yield onp.array(dindex, dtype=i32) - 1


def _estimate_cell_capacity(position: Array,
                            box_size: Box,
                            cell_size: float,
                            buffer_size_multiplier: float) -> int:
  cell_capacity = onp.max(count_cell_filling(position, box_size, cell_size))
  return int(cell_capacity * buffer_size_multiplier)


def _shift_array(arr: Array, dindex: Array) -> Array:
  if len(dindex) == 2:
    dx, dy = dindex
    dz = 0
  elif len(dindex) == 3:
    dx, dy, dz = dindex

  if dx < 0:
    arr = jnp.concatenate((arr[1:], arr[:1]))
  elif dx > 0:
    arr = jnp.concatenate((arr[-1:], arr[:-1]))

  if dy < 0:
    arr = jnp.concatenate((arr[:, 1:], arr[:, :1]), axis=1)
  elif dy > 0:
    arr = jnp.concatenate((arr[:, -1:], arr[:, :-1]), axis=1)

  if dz < 0:
    arr = jnp.concatenate((arr[:, :, 1:], arr[:, :, :1]), axis=2)
  elif dz > 0:
    arr = jnp.concatenate((arr[:, :, -1:], arr[:, :, :-1]), axis=2)

  return arr


def _unflatten_cell_buffer(arr: Array,
                           cells_per_side: Array,
                           dim: int) -> Array:
  if (isinstance(cells_per_side, int) or
      isinstance(cells_per_side, float) or
      (util.is_array(cells_per_side) and not cells_per_side.shape)):
    cells_per_side = (int(cells_per_side),) * dim
  elif util.is_array(cells_per_side) and len(cells_per_side.shape) == 1:
    cells_per_side = tuple([int(x) for x in cells_per_side[::-1]])
  elif util.is_array(cells_per_side) and len(cells_per_side.shape) == 2:
    cells_per_side = tuple([int(x) for x in cells_per_side[0][::-1]])
  else:
    raise ValueError()
  return jnp.reshape(arr, cells_per_side + (-1,) + arr.shape[1:])


def cell_list(box_size: Box,
              minimum_cell_size: float,
              buffer_size_multiplier: float = 1.25
              ) -> CellListFns:
  r"""Returns a function that partitions point data spatially.

  Given a set of points {x_i \in R^d} with associated data {k_i \in R^m} it is
  often useful to partition the points / data spatially. A simple partitioning
  that can be implemented efficiently within XLA is a dense partition into a
  uniform grid called a cell list.

  Since XLA requires that shapes be statically specified inside of a JIT block,
  the cell list code can operate in two modes: allocation and update.

  Allocation creates a new cell list that uses a set of input positions to
  estimate the capacity of the cell list. This capacity can be adjusted by
  setting the `buffer_size_multiplier` or setting the `extra_capacity`.
  Allocation cannot be JIT.

  Updating takes a previously allocated cell list and places a new set of
  particles in the cells. Updating cannot resize the cell list and is therefore
  compatible with JIT. However, if the configuration has changed substantially
  it is possible that the existing cell list won't be large enough to
  accommodate all of the particles. In this case the `did_buffer_overflow` bit
  will be set to True.

  Args:
    box_size: A float or an ndarray of shape [spatial_dimension] specifying the
      size of the system. Note, this code is written for the case where the
      boundaries are periodic. If this is not the case, then the current code
      will be slightly less efficient.
    minimum_cell_size: A float specifying the minimum side length of each cell.
      Cells are enlarged so that they exactly fill the box.
    buffer_size_multiplier: A floating point multiplier that multiplies the
      estimated cell capacity to allow for fluctuations in the maximum cell
      occupancy.
  Returns:
    A CellListFns object that contains two methods, one to allocate the cell
    list and one to update the cell list. The update function can be called
    with either a cell list from which the capacity can be inferred or with
    an explicit integer denoting the capacity. Note that an existing cell list
    can also be updated by calling `cell_list.update(position)`.
  """

  if util.is_array(box_size):
    box_size = onp.array(box_size)
    if len(box_size.shape) == 1:
      box_size = jnp.reshape(box_size, (1, -1))

  if util.is_array(minimum_cell_size):
    minimum_cell_size = onp.array(minimum_cell_size)

  def cell_list_fn(position: Array,
                   capacity_overflow_update: Optional[
                       Tuple[int, bool, Callable[..., CellList]]] = None,
                   extra_capacity: int = 0, **kwargs) -> CellList:
    N = position.shape[0]
    dim = position.shape[1]

    if dim != 2 and dim != 3:
      # NOTE(schsam): Do we want to check this in compute_fn as well?
      raise ValueError(
          f'Cell list spatial dimension must be 2 or 3. Found {dim}.')

    _, cell_size, cells_per_side, cell_count = \
        _cell_dimensions(dim, box_size, minimum_cell_size)

    if capacity_overflow_update is None:
      cell_capacity = _estimate_cell_capacity(position, box_size, cell_size,
                                              buffer_size_multiplier)
      cell_capacity += extra_capacity
      overflow = False
      update_fn = cell_list_fn
    else:
      cell_capacity, overflow, update_fn = capacity_overflow_update

    hash_multipliers = _compute_hash_constants(dim, cells_per_side)

    # Create cell list data.
    particle_id = lax.iota(i32, N)
    # NOTE(schsam): We use the convention that particles that are successfully,
    # copied have their true id whereas particles empty slots have id = N.
    # Then when we copy data back from the grid, copy it to an array of shape
    # [N + 1, output_dimension] and then truncate it to an array of shape
    # [N, output_dimension] which ignores the empty slots.
    cell_position = jnp.zeros((cell_count * cell_capacity, dim),
                              dtype=position.dtype)
    cell_id = N * jnp.ones((cell_count * cell_capacity, 1), dtype=i32)

    # It might be worth adding an occupied mask. However, that will involve
    # more compute since often we will do a mask for species that will include
    # an occupancy test. It seems easier to design around this empty_data_value
    # for now and revisit the issue if it comes up later.
    empty_kwarg_value = 10 ** 5
    cell_kwargs = {}
    #  pytype: disable=attribute-error
    for k, v in kwargs.items():
      if not util.is_array(v):
        raise ValueError((f'Data must be specified as an ndarray. Found "{k}" '
                          f'with type {type(v)}.'))
      if v.shape[0] != position.shape[0]:
        raise ValueError(('Data must be specified per-particle (an ndarray '
                          f'with shape ({N}, ...)). Found "{k}" with '
                          f'shape {v.shape}.'))
      kwarg_shape = v.shape[1:] if v.ndim > 1 else (1,)
      cell_kwargs[k] = empty_kwarg_value * jnp.ones(
          (cell_count * cell_capacity,) + kwarg_shape, v.dtype)
    #  pytype: enable=attribute-error
    indices = jnp.array(position / cell_size, dtype=i32)
    hashes = jnp.sum(indices * hash_multipliers, axis=1)

    # Copy the particle data into the grid. Here we use a trick to allow us to
    # copy into all cells simultaneously using a single lax.scatter call. To do
    # this we first sort particles by their cell hash. We then assign each
    # particle to have a cell id = hash * cell_capacity + grid_id where
    # grid_id is a flat list that repeats 0, .., cell_capacity. So long as
    # there are fewer than cell_capacity particles per cell, each particle is
    # guarenteed to get a cell id that is unique.
    sort_map = jnp.argsort(hashes)
    sorted_position = position[sort_map]
    sorted_hash = hashes[sort_map]
    sorted_id = particle_id[sort_map]

    sorted_kwargs = {}
    for k, v in kwargs.items():
      sorted_kwargs[k] = v[sort_map]

    sorted_cell_id = jnp.mod(lax.iota(i32, N), cell_capacity)
    sorted_cell_id = sorted_hash * cell_capacity + sorted_cell_id

    cell_position = cell_position.at[sorted_cell_id].set(sorted_position)
    sorted_id = jnp.reshape(sorted_id, (N, 1))
    cell_id = cell_id.at[sorted_cell_id].set(sorted_id)
    cell_position = _unflatten_cell_buffer(cell_position, cells_per_side, dim)
    cell_id = _unflatten_cell_buffer(cell_id, cells_per_side, dim)

    for k, v in sorted_kwargs.items():
      if v.ndim == 1:
        v = jnp.reshape(v, v.shape + (1,))
      cell_kwargs[k] = cell_kwargs[k].at[sorted_cell_id].set(v)
      cell_kwargs[k] = _unflatten_cell_buffer(
          cell_kwargs[k], cells_per_side, dim)

    occupancy = ops.segment_sum(jnp.ones_like(hashes), hashes, cell_count)
    max_occupancy = jnp.max(occupancy)
    overflow = overflow | (max_occupancy >= cell_capacity)

    return CellList(cell_position, cell_id, cell_kwargs,
                    overflow, cell_capacity, update_fn)  # pytype: disable=wrong-arg-count

  def allocate_fn(position: Array, extra_capacity: int = 0, **kwargs
                  ) -> CellList:
    return cell_list_fn(position, extra_capacity=extra_capacity, **kwargs)

  def update_fn(position: Array, cl_or_capacity: Union[CellList, int], **kwargs
                ) -> CellList:
    if isinstance(cl_or_capacity, int):
      capacity = int(cl_or_capacity)
      return cell_list_fn(position, (capacity, False, cell_list_fn), **kwargs)
    cl = cl_or_capacity
    cl_data = (cl.cell_capacity, cl.did_buffer_overflow, cl.update_fn)
    return cell_list_fn(position, cl_data, **kwargs)

  return CellListFns(allocate_fn, update_fn)  # pytype: disable=wrong-arg-count


def _displacement_or_metric_to_metric_sq(
    displacement_or_metric: DisplacementOrMetricFn) -> MetricFn:
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


class NeighborListFormat(Enum):
  """An enum listing the different neighbor list formats.

  Attributes:
    Dense: A dense neighbor list where the ids are a square matrix
      of shape `(N, max_neighbors_per_atom)`. Here the capacity of the neighbor
      list must scale with the highest connectivity neighbor.
    Sparse: A sparse neighbor list where the ids are a rectangular
      matrix of shape `(2, max_neighbors)` specifying the start / end particle
      of each neighbor pair.
    OrderedSparse: A sparse neighbor list whose format is the same as `Sparse`
      where only bonds with i < j are included.
  """
  Dense = 0
  Sparse = 1
  OrderedSparse = 2


def is_sparse(fmt: NeighborListFormat) -> bool:
  return (fmt is NeighborListFormat.Sparse or
          fmt is NeighborListFormat.OrderedSparse)


def is_format_valid(fmt: NeighborListFormat):
  if fmt not in list(NeighborListFormat):
    raise ValueError((
        'Neighbor list format must be a member of NeighorListFormat'
        f' found {fmt}.'))


@dataclasses.dataclass
class NeighborList(object):
  """A struct containing the state of a Neighbor List.

  Attributes:
    idx: For an N particle system this is an `[N, max_occupancy]` array of
      integers such that `idx[i, j]` is the jth neighbor of particle i.
    reference_position: The positions of particles when the neighbor list was
      constructed. This is used to decide whether the neighbor list ought to be
      updated.
    did_buffer_overflow: A boolean that starts out False. If there are ever
      more neighbors than max_neighbors this is set to true to indicate that
      there was a buffer overflow. If this happens, it means that the results
      of the simulation will be incorrect and the simulation needs to be rerun
      using a larger buffer.
    cell_list_capacity: An optional integer specifying the capacity of the cell
      list used as an intermediate step in the creation of the neighbor list.
    max_occupancy: A static integer specifying the maximum size of the
      neighbor list. Changing this will invoke a recompilation.
    format: A NeighborListFormat enum specifying the format of the neighbor
      list.
    update_fn: A static python function used to update the neighbor list.
  """
  idx: Array

  reference_position: Array

  did_buffer_overflow: Array

  cell_list_capacity: Optional[int] = dataclasses.static_field()

  max_occupancy: int = dataclasses.static_field()

  format: NeighborListFormat = dataclasses.static_field()
  update_fn: Callable[[Array, 'NeighborList'],
                      'NeighborList'] = dataclasses.static_field()

  def update(self, position: Array, **kwargs) -> 'NeighborList':
    return self.update_fn(position, self, **kwargs)


@dataclasses.dataclass
class NeighborListFns:
  """A struct containing functions to allocate and update neighbor lists.

  Attributes:
    allocate: A function to allocate a new neighbor list. This function cannot
      be compiled, since it uses the values of positions to infer the shapes.
    update: A function to update a neighbor list given a new set of positions
      and a previously allocated neighbor list.
  """
  allocate: Callable[..., NeighborList] = dataclasses.static_field()
  update: Callable[[Array, NeighborList],
                   NeighborList] = dataclasses.static_field()

  def __call__(self,
               position: Array,
               neighbors: Optional[NeighborList] = None,
               extra_capacity: int = 0,
               **kwargs) -> NeighborList:
    """A function for backward compatibility with previous neighbor lists.

    Args:
      position: An `(N, dim)` array of particle positions.
      neighbors: An optional neighbor list object. If it is provided then
        the function updates the neighbor list, otherwise it allocates a new
        neighbor list.
      extra_capacity: Extra capacity to add if allocating the neighbor list.
    Returns:
      A neighbor list object.
    """
    logging.warning('Using a depricated code path to create / update neighbor '
                    'lists. It will be removed in a later version of JAX MD. '
                    'Using `neighbor_fn.allocate` and `neighbor_fn.update` '
                    'is preferred.')
    if neighbors is None:
      return self.allocate(position, extra_capacity, **kwargs)
    return self.update(position, neighbors, **kwargs)

  def __iter__(self):
    return iter((self.allocate, self.update))


NeighborFn = Callable[[Array, Optional[NeighborList], Optional[int]],
                      NeighborList]


def neighbor_list(displacement_or_metric: DisplacementOrMetricFn,
                  box_size: Box,
                  r_cutoff: float,
                  dr_threshold: float,
                  capacity_multiplier: float = 1.25,
                  disable_cell_list: bool = False,
                  mask_self: bool = True,
                  custom_mask_function: Optional[MaskFn] = None,
                  fractional_coordinates: bool = False,
                  format: NeighborListFormat = NeighborListFormat.Dense,
                  **static_kwargs) -> NeighborFn:
  """Returns a function that builds a list neighbors for collections of points.

  Neighbor lists must balance the need to be jit compatable with the fact that
  under a jit the maximum number of neighbors cannot change (owing to static
  shape requirements). To deal with this, our `neighbor_list` returns a
  `NeighborListFns` object that contains two functions: 1)
  `neighbor_fn.allocate` create a new neighbor list and 2) `neighbor_fn.update`
  updates an existing neighbor list. Neighbor lists themselves additionally
  have a convenience `update` member function.

  Note that allocation of a new neighbor list cannot be jit compiled since it
  uses the positions to infer the maximum number of neighbors (along with
  additional space specified by the `capacity_multiplier`). Updating the
  neighbor list can be jit compiled; if the neighbor list capacity is not
  sufficient to store all the neighbors, the `did_buffer_overflow` bit
  will be set to `True` and a new neighbor list will need to be reallocated.

  Here is a typical example of a simulation loop with neighbor lists:

  >>> init_fn, apply_fn = simulate.nve(energy_fn, shift, 1e-3)
  >>> exact_init_fn, exact_apply_fn = simulate.nve(exact_energy_fn, shift, 1e-3)
  >>>
  >>> nbrs = neighbor_fn.allocate(R)
  >>> state = init_fn(random.PRNGKey(0), R, neighbor_idx=nbrs.idx)
  >>>
  >>> def body_fn(i, state):
  >>>   state, nbrs = state
  >>>   nbrs = nbrs.update(state.position)
  >>>   state = apply_fn(state, neighbor_idx=nbrs.idx)
  >>>   return state, nbrs
  >>>
  >>> step = 0
  >>> for _ in range(20):
  >>>   new_state, nbrs = lax.fori_loop(0, 100, body_fn, (state, nbrs))
  >>>   if nbrs.did_buffer_overflow:
  >>>     nbrs = neighbor_fn.allocate(state.position)
  >>>   else:
  >>>     state = new_state
  >>>     step += 1

  Args:
    displacement: A function `d(R_a, R_b)` that computes the displacement
      between pairs of points.
    box_size: Either a float specifying the size of the box or an array of
      shape [spatial_dim] specifying the box size in each spatial dimension.
    r_cutoff: A scalar specifying the neighborhood radius.
    dr_threshold: A scalar specifying the maximum distance particles can move
      before rebuilding the neighbor list.
    capacity_multiplier: A floating point scalar specifying the fractional
      increase in maximum neighborhood occupancy we allocate compared with the
      maximum in the example positions.
    disable_cell_list: An optional boolean. If set to True then the neighbor
      list is constructed using only distances. This can be useful for
      debugging but should generally be left as False.
    mask_self: An optional boolean. Determines whether points can consider
      themselves to be their own neighbors.
    custom_mask_function: An optional function. Takes the neighbor array
      and masks selected elements. Note: The input array to the function is
      (n_particles, m) where the index of particle 1 is in index in the first
      dimension of the array, the index of particle 2 is given by the value in
      the array
    fractional_coordinates: An optional boolean. Specifies whether positions
      will be supplied in fractional coordinates in the unit cube, [0, 1]^d.
      If this is set to True then the box_size will be set to 1.0 and the
      cell size used in the cell list will be set to cutoff / box_size.
    format: The format of the neighbor list; see the NeighborListFormat enum
      for details about the different choices for formats. Defaults to `Dense`.
    **static_kwargs: kwargs that get threaded through the calculation of
      example positions.
  Returns:
    A NeighborListFns object that contains a method to allocate a new neighbor
    list and a method to update an existing neighbor list.
  """
  is_format_valid(format)
  box_size = lax.stop_gradient(box_size)
  r_cutoff = lax.stop_gradient(r_cutoff)
  dr_threshold = lax.stop_gradient(dr_threshold)

  box_size = f32(box_size)

  cutoff = r_cutoff + dr_threshold
  cutoff_sq = cutoff ** 2
  threshold_sq = (dr_threshold / f32(2)) ** 2
  metric_sq = _displacement_or_metric_to_metric_sq(displacement_or_metric)

  cell_size = cutoff
  if fractional_coordinates:
    cell_size = cutoff / box_size
    box_size = f32(1)

  use_cell_list = jnp.all(cell_size < box_size / 3.) and not disable_cell_list
  if use_cell_list:
    cl_fn = cell_list(box_size, cell_size, capacity_multiplier)

  @jit
  def candidate_fn(position: Array) -> Array:
    candidates = jnp.arange(position.shape[0])
    return jnp.broadcast_to(candidates[None, :],
                            (position.shape[0], position.shape[0]))

  @jit
  def cell_list_candidate_fn(cl: CellList, position: Array) -> Array:
    N, dim = position.shape

    idx = cl.id_buffer

    cell_idx = [idx]

    for dindex in _neighboring_cells(dim):
      if onp.all(dindex == 0):
        continue
      cell_idx += [_shift_array(idx, dindex)]

    cell_idx = jnp.concatenate(cell_idx, axis=-2)
    cell_idx = cell_idx[..., jnp.newaxis, :, :]
    cell_idx = jnp.broadcast_to(cell_idx, idx.shape[:-1] + cell_idx.shape[-2:])

    def copy_values_from_cell(value, cell_value, cell_id):
      scatter_indices = jnp.reshape(cell_id, (-1,))
      cell_value = jnp.reshape(cell_value, (-1,) + cell_value.shape[-2:])
      return value.at[scatter_indices].set(cell_value)

    neighbor_idx = jnp.zeros((N + 1,) + cell_idx.shape[-2:], i32)
    neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx)
    return neighbor_idx[:-1, :, 0]

  @jit
  def mask_self_fn(idx: Array) -> Array:
    self_mask = idx == jnp.reshape(jnp.arange(idx.shape[0], dtype=i32), (idx.shape[0], 1))
    return jnp.where(self_mask, idx.shape[0], idx)

  @jit
  def prune_neighbor_list_dense(position: Array, idx: Array, **kwargs) -> Array:
    d = partial(metric_sq, **kwargs)
    d = space.map_neighbor(d)

    N = position.shape[0]
    neigh_position = position[idx]
    dR = d(position, neigh_position)

    mask = (dR < cutoff_sq) & (idx < N)
    out_idx = N * jnp.ones(idx.shape, i32)

    cumsum = jnp.cumsum(mask, axis=1)
    index = jnp.where(mask, cumsum - 1, idx.shape[1] - 1)
    p_index = jnp.arange(idx.shape[0])[:, None]
    out_idx = out_idx.at[p_index, index].set(idx)
    max_occupancy = jnp.max(cumsum[:, -1])

    return out_idx[:, :-1], max_occupancy

  @jit
  def prune_neighbor_list_sparse(position: Array, idx: Array, **kwargs
                                 ) -> Array:
    d = partial(metric_sq, **kwargs)
    d = space.map_bond(d)

    N = position.shape[0]
    sender_idx = jnp.broadcast_to(jnp.arange(N)[:, None], idx.shape)

    sender_idx = jnp.reshape(sender_idx, (-1,))
    receiver_idx = jnp.reshape(idx, (-1,))
    dR = d(position[sender_idx], position[receiver_idx])

    mask = (dR < cutoff_sq) & (receiver_idx < N)
    if format is NeighborListFormat.OrderedSparse:
      mask = mask & (receiver_idx < sender_idx)

    out_idx = N * jnp.ones(receiver_idx.shape, i32)

    cumsum = jnp.cumsum(mask)
    index = jnp.where(mask, cumsum - 1, len(receiver_idx) - 1)
    receiver_idx = out_idx.at[index].set(receiver_idx)
    sender_idx = out_idx.at[index].set(sender_idx)
    max_occupancy = cumsum[-1]

    return jnp.stack((receiver_idx[:-1], sender_idx[:-1])), max_occupancy

  def neighbor_list_fn(position: Array,
                       neighbors: Optional[NeighborList] = None,
                       extra_capacity: int = 0,
                       **kwargs) -> NeighborList:
    nbrs = neighbors
    def neighbor_fn(position_and_overflow, max_occupancy=None):
      position, overflow = position_and_overflow
      N = position.shape[0]

      if use_cell_list:
        if neighbors is None:
          cl = cl_fn.allocate(position, extra_capacity=extra_capacity)
        else:
          cl = cl_fn.update(position, neighbors.cell_list_capacity)
        overflow = overflow | cl.did_buffer_overflow
        idx = cell_list_candidate_fn(cl, position)
        cl_capacity = cl.cell_capacity
      else:
        cl_capacity = None
        idx = candidate_fn(position)

      if mask_self:
        idx = mask_self_fn(idx)
      if custom_mask_function is not None:
        idx = custom_mask_function(idx)

      if is_sparse(format):
        idx, occupancy = prune_neighbor_list_sparse(position, idx, **kwargs)
      else:
        idx, occupancy = prune_neighbor_list_dense(position, idx, **kwargs)

      if max_occupancy is None:
        _extra_capacity = (extra_capacity if not is_sparse(format)
                           else N * extra_capacity)
        max_occupancy = int(occupancy * capacity_multiplier + _extra_capacity)
        if max_occupancy > position.shape[0] and not is_sparse(format):
          max_occupancy = position.shape[0]
        if max_occupancy > occupancy:
          padding = max_occupancy - occupancy
          pad = N * jnp.ones((idx.shape[0], padding), dtype=idx.dtype)
          idx = jnp.concatenate([idx, pad], axis=1)
      idx = idx[:, :max_occupancy]
      update_fn = (neighbor_list_fn if neighbors is None else
                   neighbors.update_fn)
      return NeighborList(
          idx,
          position,
          overflow | (occupancy >= max_occupancy),
          cl_capacity,
          max_occupancy,
          format,
          update_fn)  # pytype: disable=wrong-arg-count

    if nbrs is None:
      return neighbor_fn((position, False))

    neighbor_fn = partial(neighbor_fn, max_occupancy=nbrs.max_occupancy)

    d = partial(metric_sq, **kwargs)
    d = vmap(d)
    return lax.cond(
        jnp.any(d(position, nbrs.reference_position) > threshold_sq),
        (position, nbrs.did_buffer_overflow), neighbor_fn,
        nbrs, lambda x: x)

  def allocate_fn(position: Array, extra_capacity: int = 0, **kwargs
                  ) -> NeighborList:
    return neighbor_list_fn(position, extra_capacity=extra_capacity, **kwargs)

  def update_fn(position: Array, neighbors: NeighborList, **kwargs
                ) -> NeighborList:
    return neighbor_list_fn(position, neighbors, **kwargs)

  return NeighborListFns(allocate_fn, update_fn)  # pytype: disable=wrong-arg-count


def neighbor_list_mask(neighbor: NeighborList, mask_self: bool = False
                       ) -> Array:
  """Compute a mask for neighbor list."""
  if is_sparse(neighbor.format):
    mask = neighbor.idx[0] < len(neighbor.reference_position)
    if mask_self:
      mask = mask & (neighbor.idx[0] != neighbor.idx[1])
    return mask

  mask = neighbor.idx < len(neighbor.idx)
  if mask_self:
    N = len(neighbor.reference_position)
    self_mask = neighbor.idx != jnp.reshape(jnp.arange(N, dtype=i32), (N, 1))
    mask = mask & self_mask
  return mask


def to_jraph(neighbor: NeighborList, mask: Array = None) -> jraph.GraphsTuple:
  """Convert a sparse neighbor list to a `jraph.GraphsTuple`.

  As in jraph, padding here is accomplished by adding a ficticious graph with a
  single node.

  Args:
    neighbor: A neighbor list that we will convert to the jraph format. Must be
      sparse.
    mask: An optional mask on the edges.

  Returns:
    A `jraph.GraphsTuple` that contains the topology of the neighbor list.
  """
  if not is_sparse(neighbor.format):
    raise ValueError('Cannot convert a dense neighbor list to jraph format. '
                     'Please use either NeighborListFormat.Sparse or '
                     'NeighborListFormat.OrderedSparse.')

  receivers, senders = neighbor.idx
  N = len(neighbor.reference_position)

  _mask = neighbor_list_mask(neighbor)

  if mask is not None:
    _mask = _mask & mask
    cumsum = jnp.cumsum(_mask)
    index = jnp.where(_mask, cumsum - 1, len(receivers))
    ordered = N * jnp.ones((len(receivers) + 1,), i32)
    receivers = ordered.at[index].set(receivers)[:-1]
    senders = ordered.at[index].set(senders)[:-1]
    mask = receivers < N

  return jraph.GraphsTuple(
      nodes=None,
      edges=None,
      receivers=receivers,
      senders=senders,
      globals=None,
      n_node=jnp.array([N, 1]),
      n_edge=jnp.array([jnp.sum(_mask), jnp.sum(~_mask)]),
  )


def to_dense(neighbor: NeighborList) -> Array:
  """Converts a sparse neighbor list to dense ids. Cannot be JIT."""
  if neighbor.format is not Sparse:
    raise ValueError('Can only convert sparse neighbor lists to dense ones.')

  receivers, senders = neighbor.idx
  mask = neighbor_list_mask(neighbor)

  receivers = receivers[mask]
  senders = senders[mask]

  N = len(neighbor.reference_position)
  count = ops.segment_sum(jnp.ones(len(receivers), i32), receivers, N)
  max_count = jnp.max(count)
  offset = jnp.tile(jnp.arange(max_count), N)[:len(senders)]
  hashes = senders * max_count + offset
  dense_idx = N * jnp.ones((N * max_count,), i32)
  dense_idx = dense_idx.at[hashes].set(receivers).reshape((N, max_count))
  return dense_idx


Dense = NeighborListFormat.Dense
Sparse = NeighborListFormat.Sparse
OrderedSparse = NeighborListFormat.OrderedSparse
