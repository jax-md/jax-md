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

from functools import reduce, partial
from collections import namedtuple

from typing import Any, Callable, Optional, Dict, Tuple, Generator, Union

import math
from operator import mul

import numpy as onp

from jax import lax
from jax import ops
from jax.api import jit, vmap, eval_shape
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
import jax.numpy as np

from jax_md import quantity, space, dataclasses, util


# Types


Array = util.Array
f32 = util.f32
f64 = util.f64

i32 = util.i32
i64 = util.i64

Box = space.Box
DisplacementOrMetricFn = space.DisplacementOrMetricFn
MetricFn = space.MetricFn


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
  """
  position_buffer: Array
  id_buffer: Array
  kwarg_buffers: Dict[str, Array]


def _cell_dimensions(spatial_dimension: int,
                     box_size: Box,
                     minimum_cell_size: float) -> Tuple[Box, Array, Array, int]:
  """Compute the number of cells-per-side and total number of cells in a box."""
  if isinstance(box_size, int) or isinstance(box_size, float):
    box_size = float(box_size)

  # NOTE(schsam): Should we auto-cast based on box_size? I can't imagine a case
  # in which the box_size would not be accurately represented by an f32.
  if (isinstance(box_size, onp.ndarray) and
      (box_size.dtype == np.int32 or box_size.dtype == np.int64)):
    box_size = float(box_size)

  cells_per_side = onp.floor(box_size / minimum_cell_size)
  cell_size = box_size / cells_per_side
  cells_per_side = onp.array(cells_per_side, dtype=np.int64)

  if isinstance(box_size, onp.ndarray):
    if box_size.ndim == 1 or box_size.ndim == 2:
      assert box_size.size == spatial_dimension
      flat_cells_per_side = onp.reshape(cells_per_side, (-1,))
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


def count_cell_filling(R: Array,
                       box_size: Box,
                       minimum_cell_size: float) -> Array:
  """Counts the number of particles per-cell in a spatial partition."""
  dim = int(R.shape[1])
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


def _is_variable_compatible_with_positions(R: Array) -> bool:
  if (isinstance(R, np.ndarray) and
      len(R.shape) == 2 and
      np.issubdtype(R.dtype, np.floating)):
    return True

  return False


def _compute_hash_constants(spatial_dimension: int,
                            cells_per_side: Array) -> Array:
  if cells_per_side.size == 1:
    return np.array([[cells_per_side ** d for d in range(spatial_dimension)]],
                    dtype=np.int64)
  elif cells_per_side.size == spatial_dimension:
    one = np.array([[1]], dtype=np.int32)
    cells_per_side = np.concatenate((one, cells_per_side[:, :-1]), axis=1)
    return np.array(np.cumprod(cells_per_side), dtype=np.int64)
  else:
    raise ValueError()


def _neighboring_cells(dimension: int) -> Generator[onp.ndarray, None, None]:
  for dindex in onp.ndindex(*([3] * dimension)):
    yield onp.array(dindex, dtype=np.int64) - 1


def _estimate_cell_capacity(R: Array,
                            box_size: Box,
                            cell_size: float,
                            buffer_size_multiplier: float) -> int:
  # TODO(schsam): We might want to do something more sophisticated here or at
  # least expose this constant.
  spatial_dim = R.shape[-1]
  cell_capacity = onp.max(count_cell_filling(R, box_size, cell_size))
  return int(cell_capacity * buffer_size_multiplier)


def _unflatten_cell_buffer(arr: Array,
                           cells_per_side: Array,
                           dim: int) -> Array:
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


def _shift_array(arr: onp.ndarray, dindex: Array) -> Array:
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


def _vectorize(f: Callable, dim: int) -> Callable:
  if dim == 2:
    return vmap(vmap(f, 0, 0), 0, 0)
  elif dim == 3:
    return vmap(vmap(vmap(f, 0, 0), 0, 0), 0, 0)
  raise ValueError('Cell list only supports 2d or 3d.')


def cell_list(box_size: Box,
              minimum_cell_size: float,
              cell_capacity_or_example_R: Union[int, Array],
              buffer_size_multiplier: float=1.1
              ) -> Callable[[Array], CellList]:
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

  if isinstance(box_size, np.ndarray):
    box_size = onp.array(box_size)
    if len(box_size.shape) == 1:
      box_size = np.reshape(box_size, (1, -1))

  if isinstance(minimum_cell_size, np.ndarray):
    minimum_cell_size = onp.array(minimum_cell_size)

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

    return CellList(cell_R, cell_id, cell_kwargs)  # pytype: disable=wrong-arg-count
  return build_cells


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
    max_occupancy: A static integer specifying the maximum size of the
      neighbor list. Changing this will involk a recompilation.
    cell_list_fn: A static python callable that is used to construct a cell
      list used in an intermediate step of the neighbor list calculation.
  """
  idx: Array
  reference_position: Array
  did_buffer_overflow: Array
  max_occupancy: int = dataclasses.static_field()
  cell_list_fn: Callable[[Array], CellList] = dataclasses.static_field()


NeighborFn = Callable[[Array, Optional[NeighborList], Optional[int]],
                      NeighborList]


def neighbor_list(displacement_or_metric: DisplacementOrMetricFn,
                  box_size: Box,
                  r_cutoff: float,
                  dr_threshold: float,
                  capacity_multiplier: float=1.25,
                  disable_cell_list: bool=False,
                  mask_self: bool=True,
                  fractional_coordinates: bool=False,
                  **static_kwargs) -> NeighborFn:
  """Returns a function that builds a list neighbors for collections of points.

  Neighbor lists must balance the need to be jit compatable with the fact that
  under a jit the maximum number of neighbors cannot change (owing to static
  shape requirements). To deal with this, our `neighbor_list` returns a
  function `neighbor_fn` that can operate in two modes: 1) create a new
  neighbor list or 2) update an existing neighbor list. Case 1) cannot be jit
  and it creates a neighbor list with a maximum neighbor count of the current
  neighbor count times capacity_multiplier. Case 2) is jit compatable, if any
  particle has more neighbors than the maximum, the `did_buffer_overflow` bit
  will be set to `True` and a new neighbor list will need to be created.

  Here is a typical example of a simulation loop with neighbor lists:

  >>> init_fn, apply_fn = simulate.nve(energy_fn, shift, 1e-3)
  >>> exact_init_fn, exact_apply_fn = simulate.nve(exact_energy_fn, shift, 1e-3)
  >>>
  >>> nbrs = neighbor_fn(R)
  >>> state = init_fn(random.PRNGKey(0), R, neighbor_idx=nbrs.idx)
  >>>
  >>> def body_fn(i, state):
  >>>   state, nbrs = state
  >>>   nbrs = neighbor_fn(state.position, nbrs)
  >>>   state = apply_fn(state, neighbor_idx=nbrs.idx)
  >>>   return state, nbrs
  >>>
  >>> step = 0
  >>> for _ in range(20):
  >>>   new_state, nbrs = lax.fori_loop(0, 100, body_fn, (state, nbrs))
  >>>   if nbrs.did_buffer_overflow:
  >>>     nbrs = neighbor_fn(state.position)
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
    fractional_coordinates: An optional boolean. Specifies whether positions
      will be supplied in fractional coordinates in the unit cube, [0, 1]^d.
      If this is set to True then the box_size will be set to 1.0 and the
      cell size used in the cell list will be set to cutoff / box_size.
    **static_kwargs: kwargs that get threaded through the calculation of
      example positions.
  Returns:
    A pair. The first element is a NeighborList containing the current neighbor
    list. The second element contains a function 
    `neighbor_list_fn(R, neighbor_list=None)` that will update the neighbor
    list. If neighbor_list is None then the function will construct a new
    neighbor list whose capacity is inferred from R. If neighbor_list is given
    then it will update the neighbor list (with fixed capacity) if any particle
    has moved more than dr_threshold / 2. Note that only
    `neighbor_list_fn(R, neighbor_list)` can be `jit` since it keeps array
    shapes fixed.
  """

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

  use_cell_list = np.all(cell_size < box_size / 3.) and not disable_cell_list

  @jit
  def candidate_fn(R, **kwargs):
    return np.broadcast_to(np.reshape(np.arange(R.shape[0]), (1, R.shape[0])),
                           (R.shape[0], R.shape[0]))

  @jit
  def cell_list_candidate_fn(cl, R, **kwargs):
    N, dim = R.shape

    R = cl.position_buffer
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

  @jit
  def prune_neighbor_list(R, idx, **kwargs):
    d = partial(metric_sq, **kwargs)
    d = vmap(vmap(d, (None, 0)))

    N = R.shape[0]
    neigh_R = R[idx]
    dR = d(R, neigh_R)

    mask = np.logical_and(dR < cutoff_sq, idx < N)
    out_idx = N * np.ones(idx.shape, np.int32)

    cumsum = np.cumsum(mask, axis=1)
    index = np.where(mask, cumsum - 1, idx.shape[1] - 1)
    p_index = np.arange(idx.shape[0])[:, None]
    out_idx = ops.index_update(out_idx, ops.index[p_index, index], idx)
    max_occupancy = np.max(cumsum[:, -1])

    return out_idx, max_occupancy

  @jit
  def mask_self_fn(idx):
    self_mask = idx == np.reshape(np.arange(idx.shape[0]), (idx.shape[0], 1))
    return np.where(self_mask, idx.shape[0], idx)

  def neighbor_list_fn(R: Array,
                       neighbor_list: NeighborList=None,
                       extra_capacity: int=0,
                       **kwargs) -> NeighborList:
    nbrs = neighbor_list
    def neighbor_fn(R_and_overflow, max_occupancy=None):
      R, overflow = R_and_overflow
      if cell_list_fn is not None:
        cl = cell_list_fn(R)
        idx = cell_list_candidate_fn(cl, R, **kwargs)
      else:
        idx = candidate_fn(R, **kwargs)
      idx, occupancy = prune_neighbor_list(R, idx, **kwargs)
      if max_occupancy is None:
        max_occupancy = int(occupancy * capacity_multiplier + extra_capacity)
        padding = max_occupancy - occupancy
        N = R.shape[0]
        if max_occupancy > occupancy:
          idx = np.concatenate(
            [idx, N * np.ones((N, padding), dtype=idx.dtype)], axis=1)
      idx = idx[:, :max_occupancy]
      return NeighborList(
          mask_self_fn(idx) if mask_self else idx,
          R,
          np.logical_or(overflow, (max_occupancy < occupancy)),
          max_occupancy,
          cell_list_fn)  # pytype: disable=wrong-arg-count

    if nbrs is None:
      cell_list_fn = (cell_list(box_size, cell_size, R, capacity_multiplier) if
                      use_cell_list else None)
      return neighbor_fn((R, False))
    else:
      cell_list_fn = nbrs.cell_list_fn
      neighbor_fn = partial(neighbor_fn, max_occupancy=nbrs.max_occupancy)

    d = partial(metric_sq, **kwargs)
    d = vmap(d)
    return lax.cond(
      np.any(d(R, nbrs.reference_position) > threshold_sq),
      (R, nbrs.did_buffer_overflow), neighbor_fn,
      nbrs, lambda x: x)

  return neighbor_list_fn
