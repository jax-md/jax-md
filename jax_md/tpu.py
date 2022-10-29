# Copyright 2022 Google LLC
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
"""Prototype implementation of molecular dynamics on TPU.

This code implements an approach to molecular dynamics that can be run on TPU
and scale to multiple TPUs. To accomplish this, there are several notable
features:
  1) Particles are stored in pixel / voxel cells where each cell stores the
     the offset of the particle from the center and the ID of the particle.

  2) Displacements from each particle to its neighbor are computed using
     either convolutions or seperable convolutions depending on the use case.
     Seperable convolutions have significantly better memory footprints, but
     involve more operations and can be unfriendly for some computations.

  3) During simulations, particles move by switching which cell they are in when
     their offset exceeds the bounds of the current cell. While the dynamics
     try to keep each particle in the correct cell (aka, the offset should be
     less than half the cell size), this is not always possible. In this case,
     the dynamics can be frustrated. As long as the system isn't so frustrated
     that particles lose neighbors, the simulation will be correct.
"""

import functools

import math

from typing import Optional, Union, Any, Tuple, Callable

import einops

import jax
from jax import jit, vmap, grad, pmap
from jax import lax
from jax import random
from jax.experimental import maps
from jax.experimental import mesh_utils
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray

import jax.numpy as np
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

from jax_md import dataclasses
from jax_md import energy
from jax_md import simulate
from jax_md import space

from absl import logging

import numpy as onp

partial = functools.partial


# Types

Array = space.Array
PyTree = Any
TreeDef = Any
Simulator = simulate.Simulator

# TPU Grid Type
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class TPUGrid:
  """Stores the state of a system discritized into pixels / voxels on a grid.

  For simplicity, we will describe the format of the TPUGrid in two-dimensions,
  however it generalizes in a straight forward manner to three-dimensions.

  Each pixel has three components; the first two components are the x and y
  offset of the particle from the center of the pixel, the third component is
  the ID of the particle.

  Naively one might store the data as an `(L, L, 3)` array. However, this would
  cause XLA to pad the array out to `(8, L, L, 128)` which would lead to
  significant  waste. To ameliorate this issue, rather than store the data
  contiguously, we "fold" the data into patches as `(128, l_x, l_y, 3)` where
  `l_x` and `l_y` are chosen so that `f_x l_x = f_y l_y = L` and
  `f_x f_y = 128`. The `(f_x, f_y, f_z)` are referred to as "fold factors".
  Often it is impossible to choose factors that satisfy these constraints. In
  these cases we settle for batch size < 128, knowing that this leads to reduced
  efficiency.

  This choice incurs a cost of padding the patches at each step of the
  simulation. More work is needed to determine 1) how much of the work in
  padding can be improved / removed and 2) how this cost compares to the waste
  on the MXU if we were to store the data contiguously.

  In the multi-TPU setting, the `cell_data` has shape
  `(N_x, N_y, 128, l_x, l_y, 3)` where `(N_x, N_y)` are the number of devices
  in the x- and y-dimensions respectively.


  Attributes:
    cell_data: An array storing the pixel / voxel data. The format of
      `cell_data` is described above.
    topology: A tuple specifying the number of devices per-dimension. Should be
      `num_dims` in length.
    factors: A tuple describing how the image is subdivided into patches, as
      described above.
    box_size: A float specifying the size of the simulation region.
    cell_size: A float specifying the pixel / voxel size.
    max_grid_distance: An integer specifying how many pixels / voxels we will
      consider when computing distances. This is set by the range of the
      interactions.
    num_dims: The number of spatial dimensions.
  """
  cell_data: Array

  topology: Tuple[int, ...] = dataclasses.static_field()
  factors: Tuple[int, ...] = dataclasses.static_field()
  box_size_in_cells: Union[int, Tuple[int, ...]] = dataclasses.static_field()
  cell_size: float = dataclasses.static_field()
  max_grid_distance: int = dataclasses.static_field()
  num_dims: int = dataclasses.static_field()


# Core Public API
# -----------------------------------------------------------------------------


def to_grid(positions: Array,
            box_size_in_cells: Union[int, Tuple[int, ...]],
            cell_size: float,
            max_interaction_distance: float,
            topology: Optional[Tuple[int, ...]] = None,
            aux: Optional[PyTree] = None,
            strategy: str = 'closest',
            ) -> Union[TPUGrid, Tuple[TPUGrid, PyTree]]:
  """Place particles, and optionally auxiliary data, into a TPUGrid.

  Args:
    positions: An `(N, spatial_dimension)` array of particle positions.
    box_size_in_cells: An integer specifying the size of the simulation volume.
    cell_size: A float specifying the pixel / voxel discretization size.
    max_interaction_distance: The maximum range of the interactions that will be
      considered. This will be used to compute how many cells need to be
      considered when computing nearest neighbor distances.
    topology: The topology of the TPU mesh. Must be a tuple of length
      `spatial_dimensions`.
    aux: A PyTree of auxiliary data that we would also like to add to the grid.

  Returns:
    A TPUGrid object containing the state of the system (see the TPUGrid
    documentation for more information). If `aux` is set, then it will also
    return a PyTree with the same strucutre as `aux` with each leaf array placed
    into the grid.
  """
  num_dims = positions.shape[-1]
  # Targetting batch_size = 128 always seems optimal on TPU.
  batch_size = 128
  ids = np.arange(len(positions))

  if topology is None:
    topology = ()

  max_grid_distance = max_interaction_distance / cell_size
  if not onp.isclose(max_grid_distance, onp.round(max_grid_distance)):
    logging.warning(f'max_interaction_distance ({max_interaction_distance}) did'
                    f' not evenly divide the cell_size ({cell_size}). This will'
                    ' mean that the max grid distance will be padded.')
  max_grid_distance = int(onp.ceil(max_grid_distance) + 1)

  cell_data = _positions_to_grid(positions, box_size_in_cells, cell_size, ids + 1,
                                 aux, strategy)

  if np.sum(cell_data[..., -1] > 0) < len(positions):
    print(np.sum(cell_data[..., -1] > 0))
    print(len(positions))
    raise ValueError('Lost particles while placing into grid, due to '
                     'collision. Consider either using a better initial '
                     'configuration or setting the placement strategy to '
                     '`linear` (note that the linear strategy can take a '
                     'long time for large systems).')

  # Function to fold the grid on each device separately.
  inner_fold_fn = lambda grid: _fold_grid(grid, max_grid_distance, batch_size)

  if aux is not None:
    aux_tree, aux_sizes = _get_aux_spec(num_dims, aux)

  if topology:
    # Split grid across TPU mesh and push `inner_fold_fn` to act per-device.
    cell_data, _ = _fold_grid(cell_data,
                              max_grid_distance,
                              factors=topology,
                              inner_fold=False)
    inner_fold_fn = parallelize(inner_fold_fn, topology)

  cell_data, factors = inner_fold_fn(cell_data)

  # Factors is static, and will be the same for each shard.
  factors = factors[(0,) * len(topology)]
  factors = tuple(int(f) for f in factors)

  grid = TPUGrid(cell_data=cell_data,
                 factors=factors,
                 box_size_in_cells=box_size_in_cells,
                 cell_size=cell_size,
                 topology=topology,
                 max_grid_distance=max_grid_distance,
                 num_dims=num_dims)  # pytype: disable=wrong-keyword-args

  grid = _settle_particle_locations(grid)

  if aux is not None:
    cell_data, aux = _get_aux(grid.cell_data, aux_tree, aux_sizes)
    grid = dataclasses.replace(grid, cell_data=cell_data)
    return grid, aux

  return grid


def from_grid(grid: TPUGrid, aux: Optional[PyTree] = None
              ) -> Union[Array, Tuple[Array, PyTree]]:
  """Extract positions and, optionally, auxiliary data from a grid."""
  box_size_in_cells = grid.box_size_in_cells
  # TODO(schsam, jaschasd): Maybe store the box size as a multidimensional array
  # by default.
  box_size_in_cells = (onp.array([box_size_in_cells] * grid.num_dims)
                       if onp.isscalar(box_size_in_cells) else onp.array(box_size_in_cells))

  cell_size = grid.cell_size
  num_dims = grid.num_dims

  if aux is not None:
    data = grid.cell_data
    data, aux_tree, aux_sizes = _set_aux(data, aux)
    grid = dataclasses.replace(grid, cell_data=data)

  data = unfold_mesh(grid.cell_data, grid)

  grid_centers = _grid_centers(box_size_in_cells, cell_size, num_dims)

  data = np.reshape(data, (-1, data.shape[-1]))
  grid_centers = np.reshape(grid_centers, (-1, num_dims))

  valid = data[:, -1] > 0.

  data = data[valid]
  grid_centers = grid_centers[valid]

  if aux is not None:
    pos, aux = _get_aux(data, aux_tree, aux_sizes)
  else:
    pos = data

  pos, particle_ids = pos[:, :num_dims], pos[:, -1]
  idx = np.argsort(particle_ids)

  pos = pos + grid_centers
  # Avoid negative positions.
  box_size = box_size_in_cells * cell_size
  pos = np.mod(pos, box_size[None, :])
  pos = pos[idx]

  if aux is not None:
    aux = aux[idx]
    return pos, aux

  return pos


def random_grid(key: Array,
                density: float,
                box_size_in_cells: Union[int, Tuple[int, ...]],
                cell_size: float,
                max_interaction_distance: float,
                topology: Optional[Tuple[int, ...]] = None,
                ) -> TPUGrid:
  """Place particles, and optionally auxiliary data, into a TPUGrid.

  Args:
    positions: An `(N, spatial_dimension)` array of particle positions.
    box_size: A float specifying the size of the simulation volume.
    cell_size: A float specifying the pixel / voxel discretization size.
    max_interaction_distance: The maximum range of the interactions that will be
      considered. This will be used to compute how many cells need to be
      considered when computing nearest neighbor distances.
    topology: The topology of the TPU mesh. Must be a tuple of length
      `spatial_dimensions`.
    aux: A PyTree of auxiliary data that we would also like to add to the grid.

  Returns:
    A TPUGrid object containing the state of the system (see the TPUGrid
    documentation for more information). If `aux` is set, then it will also
    return a PyTree with the same strucutre as `aux` with each leaf array placed
    into the grid.
  """
  num_dims = len(topology)
  # Targetting batch_size = 128 always seems optimal on TPU.
  batch_size = 128

  if topology is None:
    topology = ()

  max_grid_distance = max_interaction_distance / cell_size
  if not onp.isclose(max_grid_distance, onp.round(max_grid_distance)):
    logging.warning(f'max_interaction_distance ({max_interaction_distance}) did'
                    f' not evenly divide the cell_size ({cell_size}). This will'
                    ' mean that the max grid distance will be padded.')
  max_grid_distance = int(onp.ceil(max_grid_distance) + 1)

  arr_box_size = (onp.array([box_size_in_cells] * num_dims)
                  if onp.isscalar(box_size_in_cells) else onp.array(box_size_in_cells))
  arr_box_size = arr_box_size / onp.array(topology)
  arr_box_size += 2 * max_grid_distance
  assert np.all(np.isclose(arr_box_size, np.round(arr_box_size)))
  arr_box_size = tuple(arr_box_size.astype(np.int32))

  pkeys = random.split(key, onp.prod(topology))
  pkeys = np.reshape(pkeys, topology + (2,))

  def create_instance_by_key(key):
    pos_key, occ_key = random.split(key)
    grid = random.uniform(pos_key, arr_box_size + (num_dims,),
                          minval=-cell_size/2., maxval=cell_size/2.)
    occupied = random.bernoulli(occ_key, density, arr_box_size + (1,))
    grid *= occupied
    return np.concatenate((grid, occupied), axis=-1)

  def create_instance(index):
    index_into_topology = index[:len(topology)]
    key = np.squeeze(pkeys[index_into_topology])
    return np.expand_dims(create_instance_by_key(key), range(len(topology)))

  mesh, axes = mesh_and_axes(topology)
  cell_data = GlobalDeviceArray.from_callback(topology + arr_box_size + (num_dims + 1,),
                                              mesh,
                                              axes,
                                              create_instance)

  # Function to fold the grid on each device separately.
  inner_fold_fn = lambda grid: _fold_grid(grid, max_grid_distance, batch_size)
  inner_fold_fn = parallelize(inner_fold_fn, topology)

  cell_data, factors = inner_fold_fn(cell_data)

  # Factors is static, and will be the same for each shard.
  factors = factors[(0,) * len(topology)]
  factors = tuple(int(f) for f in factors)

  grid = TPUGrid(cell_data=cell_data,
                 factors=factors,
                 box_size_in_cells=box_size_in_cells,
                 cell_size=cell_size,
                 topology=topology,
                 max_grid_distance=max_grid_distance,
                 num_dims=num_dims)  # pytype: disable=wrong-keyword-args

  return _settle_particle_locations(grid)


def compute_displacement(grid: TPUGrid) -> Tuple[Array, Array]:
  """Compute the displacement between all pairs of neighboring particles.

  TODO(schsam): Eventually, we may want to generalize this function if we want
  to aggregate pairwise auxiliary data.

  Args:
    grid: A TPUGrid whose positions are used to compute displacements.

  Returns:
    A pair containing the displacements and a mask which is non-zero for pairs
    of cells that both contain a particle. The shape of the both arrays is
    `grid.cell_data.shape[:-1] + (max_neighbors,)`.
  """

  data = grid.cell_data
  num_dims = grid.num_dims

  # Compute pairwise offsets within window.
  # TODO(jaschasd, schsam): Make ordering of dimensions consistent throughout
  # code. avoid unnecessary transposes.
  data_transposed = np.transpose(data, (0, data.ndim - 1,) +
                                 tuple(range(1, num_dims + 1)))
  offsets = _get_pairwise_displacement(data_transposed, grid)
  # The shape of pairwise_dat will be B x L x C x X x Y x ...

  # The mask will have entries of 1 if particles exist at both grid points
  # and entries of 0 otherwise.
  mask = (offsets[:, :, [-1], ...] > 0.5)
  mask = mask & (data_transposed[:, None, [-1], ...] > 0.5)

  reordering = (0,) + tuple(range(3, 3 + num_dims)) + (1, 2)
  offsets = np.transpose(offsets, reordering)
  mask = np.transpose(mask, reordering)

  # Compute the displacement from the particle (rather than the center) to its
  # neighbors.
  offsets = offsets.at[..., :num_dims].add(-data[..., None, :num_dims])
  offsets = offsets[..., :-1]

  return offsets, mask


def accumulate_on_grid(accum_fn: Callable[[Array, Array, Array], Array],
                       grid: TPUGrid) -> Array:
  """Sums the result of `accum_fn` applied to batches of pairwise displacements.

  Unlike `compute_displacement`, this function evaluates the `accum_fn` serially
  over batches of displacement vectors and sums the result. Typically this will
  take significantly less memory, especially in three-dimensions, than computing
  all of the displacements upfront.

  Args:
    accum_fn: A function that takes displacement vectors and pairs of particle
      IDs.
    grid: A TPUGrid whose positions are used to compute the displacements.

  Returns:
    The result of summing `accum_fn` over all neighbors in the system.
  """

  # TODO(jaschasd, schsam): Make ordering of dimensions consistent throughout
  # code. avoid unnecessary transposes.
  # data has shape [B x X x Y x ... C].
  data = grid.cell_data
  # TODO(jaschasd, schsam): This code is only useful if there is aux data. Will
  # there ever be aux data when we call this?
  data = np.concatenate((data[..., :grid.num_dims], data[..., -1:]), axis=-1)
  data_0 = data[..., np.newaxis, :]

  # TODO(jaschasd, schsam): Could move some of these into the recursion, to
  # reduce peak memory usage
  for dim in range(grid.num_dims):
    data = _pad_axis_channel_last(data, grid.factors, dim,
                                  grid.max_grid_distance)

  return _accumulate_recursion(accum_fn, data_0, data, 0, grid)


def mesh_and_axes(topology):
  labels = ['X', 'Y', 'Z']
  n = len(topology)
  labels = labels[:n]

  devs = mesh_utils.create_device_mesh(topology)
  mesh = maps.Mesh(devs, labels)
  return mesh, P(*labels)


def parallelize(f: Callable, topology: Tuple[int]) -> Callable:
  """Apply pmap for each axis over which the computation is distributed."""

  if not topology:
    return f

  labels = ['X', 'Y', 'Z']
  n = len(topology)
  labels = labels[:n]

  devs = mesh_utils.create_device_mesh(topology)
  mesh = maps.Mesh(devs, labels)

  return mesh(maps.xmap(
      f,
      in_axes=labels + [...],
      out_axes=labels + [...],
      axis_resources={l: l for l in labels}
  ))


def _psum(x: Array, topology: Tuple[int]) -> Array:
  labels = ['X', 'Y', 'Z']
  n = len(topology)
  labels = labels[:n]
  for label in labels:
    x = lax.psum(x, axis_name=label)
  return x


def unfold_mesh(cell_data: Array, grid: TPUGrid) -> Array:
  """Unfolds patches of data from a grid into a contiguous block.

  See the description of folded vs unfolded data representations in `TPUGrid`
  for details.

  Args:
    cell_data: An array of data that includes a batch dimension. The data can be
      sharded or it can be localized on one device.
    grid: A grid that contains information about the system.

  Returns:
    A contiguous version of the data.
  """
  if len(cell_data.shape) == len(grid.cell_data.shape) - 1:
    cell_data = cell_data[..., None]
  assert len(cell_data.shape) == len(grid.cell_data.shape)

  unfold_closed = partial(_unfold_grid, grid=grid, inner_fold=True)
  if grid.topology:
    unfold_closed = parallelize(unfold_closed, grid.topology)
  cell_data = unfold_closed(cell_data)
  if grid.topology:
    cell_data = _unfold_grid(cell_data, grid, False)

  return cell_data


def shift(grid: TPUGrid, displacement: Array, aux: PyTree = None
          ) -> Union[TPUGrid, Tuple[TPUGrid, PyTree]]:
  """Moves particles in a grid by a given displacement.

  Args:
    grid: The current state of the grid and a description of the system.
    displacement: An array of displacement vectors to move each particle by.
    aux: An optional PyTree of auxiliary data. If provided, this data gets moved
      between cells along with the positional data stored in the grid.

  Returns:
    A new TPUGrid state with all of the particles having been moved. This
    function appropriately moves particle occupancy between grid cells and
    transports auxiliary data, if provided, along with the positional data. If
    auxiliary data is provided, then it is returned along with the TPUGrid state
    as a PyTree with the same structure as the inputs.
  """

  # NOTE: Here we use two "mesh transport" calls. The first makes sure that
  # all of the displacements are the same on all copies of particles in the
  # halo. The second makes sure that all of the final positions are the same
  # in all copies of particles in the halo.

  # TODO(jaschasd, schsam): We might be able to do something clever to remove
  # one of the mesh_transport calls? (I think we could remove the second call if
  # we padded max_grid_distance by 1).

  data = grid.cell_data
  data = data.at[..., :grid.num_dims].add(displacement)

  if aux is not None:
    data, aux_tree, aux_sizes = _set_aux(data, aux)

  data = _mesh_transport(data, grid)
  data = _update_grid_locations(data, grid)
  data = _mesh_transport(data, grid)

  if aux is not None:
    data, aux = _get_aux(data, aux_tree, aux_sizes)

  shifted_grid = dataclasses.replace(grid, cell_data=data)

  if aux is not None:
    return shifted_grid, aux

  return shifted_grid


def nearest_valid_grid_size(target_box_size_in_cells: Union[int, Tuple[int, ...]],
                            topology: Union[int, Tuple],
                            max_grid_distance: int,
                            factors: Optional[Tuple[int, ...]]=None,
                            dimension: Optional[int]=None):
  if factors is None:
    if dimension is None:
      if topology:
        dimension = len(topology)
      elif not np.isscalar(target_box_size_in_cells):
        dimension = len(target_box_size_in_cells)
      else:
        raise ValueError('Need to (implicitly) specify dimension of space, by '
                         'passing dimension keyword, or by making '
                         'target_box_size_in_cells or topology a tuple.')
    if dimension == 1:
      factors = (128,)
    elif dimension == 2:
      factors = (16, 8)
    elif dimension == 3:
      factors = (8, 4, 4)
  folded_size = _outer_grid_size_to_inner_grid_size(target_box_size_in_cells,
                                                    topology,
                                                    factors,
                                                    max_grid_distance)
  folded_size = onp.round(folded_size)
  folded_size = onp.maximum(folded_size, max_grid_distance+1)
  folded_size += folded_size % 2  # Make sure folded_size is even.
  new_size = _inner_grid_size_to_outer_grid_size(folded_size,
                                                 topology,
                                                 factors,
                                                 max_grid_distance)
  return new_size.astype(onp.int32)



# Functionality on top of API
# -----------------------------------------------------------------------------

# Code in this section will change once energy functions / simulation
# environments in JAX MD have been generalized sufficiently.

# Energy Functions.


GridFn = Callable[[TPUGrid], Array]


def pair_potential(fn: Callable, **kwargs) -> Tuple[GridFn, GridFn]:
  """Takes the form of a pair potential and computes it over a system on TPU.

  Args:
    fn: A function that computes energy as a function of pairwise separation.
      This function can be parameterized by keyword arguments.

  Returns:
    A pair of functions, one that computes the energy and one that computes the
    force.
  """
  e_fn = partial(fn, **kwargs)

  def accum_fn(displacement, id_a, id_b):
    mask = (id_a > 0) & (id_b > 0) & (id_a != id_b)
    return e_fn(space.distance(displacement)) * mask

  def per_particle_energy_fn(grid):
    accum_vec_fn = np.vectorize(accum_fn, signature='(m),(),()->()')
    return accumulate_on_grid(accum_vec_fn, grid)

  def energy_fn(grid):
    if grid.topology and len(grid.cell_data.shape) > grid.num_dims + 2:
      e = parallelize(per_particle_energy_fn, grid.topology)(grid)
      return np.sum(unfold_mesh(e, grid)) / 2.0
    return np.sum(per_particle_energy_fn(grid)) / 2.0

  def grad_energy_fn(grid):
    g_accum_fn = grad(accum_fn)
    g_accum_vec_fn = np.vectorize(g_accum_fn, signature='(m),(),()->(m)')
    g = accumulate_on_grid(g_accum_vec_fn, grid)
    return g

  def force_fn(grid):
    if grid.topology and len(grid.cell_data.shape) > grid.num_dims + 2:
      f = parallelize(grad_energy_fn, grid.topology)(grid)
    else:
      f = grad_energy_fn(grid)
    return -f

  return energy_fn, force_fn


def soft_sphere(sigma: float = 1.0, epsilon: float = 1., alpha: float = 2
                ) -> Tuple[GridFn, GridFn]:
  """Compute the soft sphere potential over a TPUGrid.

  See `jax_md.energy.soft_sphere` for details about the potential.
  """
  return pair_potential(energy.soft_sphere,
                        sigma=sigma, epsilon=epsilon, alpha=alpha)


def lennard_jones(sigma: float = 1.0,
                  epsilon: float = 1.,
                  r_onset: float = 2.0,
                  r_cutoff: float = 3.0) -> Tuple[GridFn, GridFn]:
  """Compute the truncated Lennard-Jones potential over a TPUGrid.

  See `jax_md.energy.lennard_jones` for details about the potential.
  """
  e_fn = energy.multiplicative_isotropic_cutoff(energy.lennard_jones,
                                                r_onset, r_cutoff)
  return pair_potential(e_fn, sigma=sigma, epsilon=epsilon)


# Simulation Environments.


@dataclasses.dataclass
class NVEState:
  position: TPUGrid
  velocity: Array
  force: Array


def nve(force_fn: GridFn, dt: float) -> Simulator:
  """Performs constant energy simulation on TPU.

  See `jax_md.simulate.nve` for details about the simulation environment.
  """
  dt_2 = 0.5 * dt ** 2
  def init_fn(key, grid: TPUGrid, kT: float) -> NVEState:
    position = grid.cell_data
    mask = position[..., [-1]] > 0
    v = np.sqrt(kT) * random.normal(key,
                                    position.shape[:-1] + (grid.num_dims,),
                                    dtype=position.dtype)
    return NVEState(grid, v * mask, force_fn(grid))  # pytype: disable=wrong-arg-count
  def single_core_apply_fn(state):
    R = state.position
    V = state.velocity
    F = state.force

    R, (V, F) = shift(R, V * dt + F * dt_2, (V, F))  # pytype: disable=attribute-error
    F_new = force_fn(R)
    V = V + 0.5 * (F + F_new) * dt

    return NVEState(R, V, F_new)  # pytype: disable=wrong-arg-count

  def apply_fn(state):
    grid = state.position
    if grid.topology and len(grid.cell_data.shape) > grid.num_dims + 2:
      return parallelize(single_core_apply_fn, grid.topology)(state)
    return single_core_apply_fn(state)

  return init_fn, apply_fn


def kinetic_energy(state: NVEState) -> float:
  grid = state.position
  if grid.topology and len(grid.cell_data.shape) > grid.num_dims + 2:
    return 0.5 * np.sum(unfold_mesh(state.velocity, grid) ** 2)
  return 0.5 * np.sum(state.velocity ** 2)


# Library Code
# -----------------------------------------------------------------------------

# Grid Packing / Unpacking.


def _grid_centers(box_size_in_cells: Array, cell_size: float, num_dims: int) -> Array:
  """Computes the center position of each grid cell."""
  grid_centers = onp.zeros(tuple(box_size_in_cells) + (num_dims,))

  for i in range(num_dims):
    cell_idx = onp.arange(box_size_in_cells[i])
    new_shape = (1,) * i + (box_size_in_cells[i],) + (1,) * (num_dims - i - 1)
    grid_centers[..., i] = (onp.reshape(cell_idx, new_shape) + 0.5) * cell_size

  return np.array(grid_centers)


def cell_hash(cell_index, cells, num_dims):
  if num_dims == 1:
    return cell_index[:, 0]
  elif num_dims == 2:
    return cell_index[:, 1] + cell_index[:, 0] * cells[1]
  elif num_dims == 3:
    return (cell_index[:, 2] +
            cell_index[:, 1] * cells[2] +
            cell_index[:, 0] * cells[1] * cells[2])
  else:
    raise ValueError('TPU only supports one-, two-, or three-dimensional '
                     f'systems. Found {num_dims}.')


@partial(jit, static_argnums=(1, 2, 5), backend='cpu')
def _positions_to_grid(position: Array,
                       box_size_in_cells: int,
                       cell_size: float,
                       particle_id: Array,
                       aux: Optional[PyTree] = None,
                       strategy: str = 'closest',
                       ) -> Array:
  """Place particles in the first `particle_count` grid cells."""
  # This will instantiate all of the atoms (and worse still, the whole grid) on
  # a single host CPU, which will probably run out of memory. So something
  # smarter would be better here.
  count, num_dims = position.shape

  cells = (onp.array([box_size_in_cells] * num_dims)
           if onp.isscalar(box_size_in_cells) else onp.array(box_size_in_cells))
  total_cells = int(onp.prod(cells))

  if strategy == 'linear':
    cell_contents = np.concatenate([position, particle_id[:, None]], axis=-1)

    if aux is not None:
      cell_contents, *_ = _set_aux(cell_contents, aux)

    grid = np.zeros((total_cells, cell_contents.shape[-1]))
    centers = _grid_centers(cells, cell_size, num_dims)
    centers = np.reshape(centers, (total_cells, num_dims))

    grid = grid.at[:count, :].set(cell_contents)
    grid = grid.at[:count, :num_dims].add(-centers[:count])
    grid = np.reshape(grid, tuple(cells) + (-1,))
  elif strategy == 'closest':
    cell_index = np.array(position / cell_size, np.int32)
    centers = cell_index * cell_size + 0.5 * cell_size
    cell_index_flattened = cell_hash(cell_index, cells, num_dims)
    cell_contents = position - centers
    cell_contents = np.concatenate([cell_contents, particle_id[:, None]],
                                   axis=-1)

    if aux is not None:
      cell_contents, *_ = _set_aux(cell_contents, aux)

    grid = np.zeros((total_cells, cell_contents.shape[-1]))
    grid = grid.at[cell_index_flattened, :].set(cell_contents)
    grid = np.reshape(grid, tuple(cells) + (-1,))
  else:
    raise ValueError('Placement strategy must be either "closest" or "linear".'
                     f' Found {strategy}.')

  return grid


def _settle_particle_locations(grid: TPUGrid) -> TPUGrid:
  # To construct the initial grid, we first place particles naively in the first
  # `particle_count` cells. Then, we move the particles until they have settled
  # into a configuration where no local move will reduce the amount of
  # frustration. We do this rather than try to place particles directly in the
  # appropriate cells because otherwise some particles might end up in the same
  # cell.
  def body_fn(grid_old_grid):
    grid, _ = grid_old_grid

    grid_data = _mesh_transport(grid.cell_data, grid)
    grid_data = _update_grid_locations(grid_data, grid)
    grid_data = _mesh_transport(grid_data, grid)

    return dataclasses.replace(grid, cell_data=grid_data), grid

  def cond_fn(grid_old_grid):
    grid, old_grid = grid_old_grid
    cond = np.any(np.abs(grid.cell_data - old_grid.cell_data) > 1e-8)
    cond = _psum(cond, grid.topology)
    return cond > 0

  @jit
  def move_fn(grid):
    old_grid = dataclasses.replace(grid, cell_data=np.zeros_like(grid.cell_data))
    return lax.while_loop(cond_fn, body_fn, (grid, old_grid))[0]

  if grid.topology:
    move_fn = parallelize(move_fn, grid.topology)

  return move_fn(grid)


# Building kernels for convolution operations.


def _generate_offset_kernel_channel_last(axis: str, grid: TPUGrid
                                         ) -> Tuple[Array, Array]:
  """Generates weights and biases to get displacements with convolutions.

  These will act on an array of shape [BL x X x Y ... x C] and are used in the
  setting where we can phrase our problem as an accumulation of displacements.
  This will have lower memory footpring than the case where we compute all the
  displacements outright.

  Args:
    axis: A string specifying which spatial dimension the convolution will act
      on. Should be 'X', 'Y', or 'Z'.
    grid: A TPUGrid object specifying the dimensions of the current system.

  Returns:
    A pair of arrays containing the weights of the CNN and the biases.
  """

  max_grid_distance = grid.max_grid_distance
  cell_size = grid.cell_size
  num_dims = grid.num_dims
  input_channels = num_dims + 1

  axis = ord(axis) - ord('X')

  kernel_width = max_grid_distance * 2 + 1

  kernel = []
  bias = []

  offsets = onp.arange(-max_grid_distance, max_grid_distance + 1)
  # We always want to put the center entry first.
  order = onp.argsort(onp.abs(offsets))
  offsets = offsets[order]

  for offset in offsets:
    shift = [0] * (1 + num_dims)
    shift[axis] = cell_size * offset
    shift = onp.reshape(shift, (1,) * (num_dims + 1) + (input_channels,))
    bias.append(shift)

    # After concatenation W will have shape [C_in x C_in K x K], where
    # K = [kernel width]. Here we compute one plane, with shape
    # [C_in x C_in x K].
    position_selector = onp.zeros((kernel_width,))
    position_selector[offset + max_grid_distance] = 1

    k = onp.eye(input_channels).reshape((input_channels, input_channels, 1))
    k = k * position_selector.reshape((1, 1, kernel_width))
    kernel.append(k)

  kernel = onp.concatenate(kernel, axis=1)  # Shape [C_in x C_in K x K].
  bias = onp.concatenate(bias, axis=-1)  # [1 x 1 x 1 x ... x C_in K].

  if num_dims == 1:
    k_shape = kernel.shape
  if num_dims == 2:
    if axis == 0:
      k_shape = kernel.shape + (1,)
    else:
      k_shape = kernel.shape[:2] + (1,) + kernel.shape[2:]
  elif num_dims == 3:
    if axis == 0:
      k_shape = kernel.shape + (1, 1)
    elif axis == 1:
      k_shape = kernel.shape[:2] + (1,) + kernel.shape[2:] + (1,)
    else:
      k_shape = kernel.shape[:2] + (1, 1) + kernel.shape[2:]

  kernel = onp.reshape(kernel, k_shape)
  return kernel, bias


def _generate_offset_kernel(axis: str, grid: TPUGrid) -> Tuple[Array, Array]:
  """Generates weights and biases to get displacements with convolutions.

  These will act on an array of shape [BL x C x H x W] and are used in the
  setting where compute all of the displacements upfront. This is necessary to,
  for example, compute bond angles.

  NOTE(schsam): There is some code duplication with the above function. Should
  we refactor some functionality?

  Args:
    axis: An integer specifying which spatial dimension the convolution will act
      on.
    grid: A TPUGrid object specifying the dimensions of the current system.

  Returns:
    A pair of arrays containing the weights of the CNN and the biases.
  """

  max_grid_distance = grid.max_grid_distance
  cell_size = grid.cell_size
  num_dims = grid.num_dims
  input_channels = num_dims + 1

  axis = ord(axis) - ord('X')

  kernel_width = max_grid_distance * 2 + 1

  kernel = []
  bias = []

  offsets = onp.arange(-max_grid_distance, max_grid_distance + 1)
  # We always want to put the center entry first.
  order = onp.argsort(onp.abs(offsets))
  offsets = offsets[order]

  for offset in offsets:
    shift = [0] * (1 + num_dims)
    shift[axis] = cell_size * offset
    shift = onp.reshape(shift, (1,) + (input_channels,) + (1,) * num_dims)
    bias.append(shift)

    # After concatenation W will have shape [C_in x C_in K x K], where
    # K = [kernel width]. Here we compute one plane, with shape
    # [C_in x C_in x K].
    position_selector = onp.zeros((kernel_width,))
    position_selector[offset + max_grid_distance] = 1

    k = onp.eye(input_channels).reshape((input_channels, input_channels, 1))
    k = k * position_selector.reshape((1, 1, kernel_width))
    kernel.append(k)

  kernel = onp.concatenate(kernel, axis=1)  # [C_in x C_in K x K]
  bias = onp.concatenate(bias, axis=1)  # [1 x C_in K x 1 x 1 x ...]

  if num_dims == 1:
    k_shape = kernel.shape
  if num_dims == 2:
    if axis == 0:
      k_shape = kernel.shape + (1,)
    else:
      k_shape = kernel.shape[:2] + (1,) + kernel.shape[2:]
  elif num_dims == 3:
    if axis == 0:
      k_shape = kernel.shape + (1, 1)
    elif axis == 1:
      k_shape = kernel.shape[:2] + (1,) + kernel.shape[2:] + (1,)
    else:
      k_shape = kernel.shape[:2] + (1, 1) + kernel.shape[2:]

  kernel = onp.reshape(kernel, k_shape)
  return kernel, bias


# Padding

CONV_AXES_STRING = ['X', 'XY', 'XYZ']
EINOPS_AXES_STRING = ['x', 'x y', 'x y z']
EINOPS_FACTORS_STRING = ['fx', 'fx fy', 'fx fy fz']


def _pad_axis_channel_last(data: Array,
                           factors: Tuple[int, ...],
                           axis: int,
                           padding: int = 1) -> Array:
  """Pad an array along a specified axis when the channels are trailing."""

  # As the name suggests, `data` has shape [B x X x Y x ... C].
  num_dims = len(factors)

  # For convenience we extract string representations of the axes and the
  # factors.
  axes_s = EINOPS_AXES_STRING[num_dims - 1]
  factors_s = EINOPS_FACTORS_STRING[num_dims - 1]

  # A dictionary mapping factors symbols to the size of the dimensions.
  fact_d = {}
  for i, f in enumerate(factors_s.split(' ')):
    fact_d[f] = factors[i]

  # Einops command to split and combine factors.
  split_factors = f'({factors_s}) {axes_s} c -> {factors_s} {axes_s} c'
  combine_factors = f'{factors_s} {axes_s} c -> ({factors_s}) {axes_s} c'

  # Axes before and after the padded axis.
  pre = (0,) * axis
  post = (0,) * (data.ndim - axis - 1)

  # All entries for all dimensions before the target axis.
  all_slice = tuple(slice(data.shape[j]) for j in range(axis + 1))

  # Pad with the end halo along the target axis.
  idx = all_slice + (slice(-padding, data.shape[axis + 1]),)
  end_pad = data[idx]
  end_pad = einops.rearrange(end_pad, split_factors, **fact_d)
  end_pad = np.roll(end_pad, pre + (1,) + post, axis=range(data.ndim))
  end_pad = einops.rearrange(end_pad, combine_factors)

  # Pad with the start halo along the target axis.
  idx = all_slice + (slice(0, padding),)
  start_pad = data[idx]
  start_pad = einops.rearrange(start_pad, split_factors, **fact_d)
  start_pad = np.roll(start_pad, pre + (-1,) + post, axis=range(data.ndim))
  start_pad = einops.rearrange(start_pad, combine_factors)

  data = np.concatenate([end_pad, data, start_pad], axis=1 + axis)

  return data


def _pad_axis(data: Array, factors: Tuple[int, ...], axis: int, padding: int
              ) -> Array:
  """Pad an array along a specified axis when the channels are in the middle."""

  # As the name suggests, data has shape [B x L x C x X x Y x ...].

  num_dims = len(factors)

  # For convenience we extract string representations of the axes and the
  # factors.
  axes_s = EINOPS_AXES_STRING[num_dims - 1]
  factors_s = EINOPS_FACTORS_STRING[num_dims - 1]

  # A dictionary mapping factors symbols to the size of the dimensions.
  fact_d = {}
  for i, f in enumerate(factors_s.split(' ')):
    fact_d[f] = factors[i]

  # Einops command to split and combine factors.
  split_factors = f'({factors_s}) l c {axes_s} -> {factors_s} l c {axes_s}'
  combine_factors = f'{factors_s} l c {axes_s} -> ({factors_s}) l c {axes_s}'

  data = einops.rearrange(data, split_factors, **fact_d)

  # Axes before and after the padded axis.
  pre = (0,) * axis
  post = (0,) * (data.ndim - axis - 1)

  # All entries for all dimensions before the target axis.
  all_slice = tuple(slice(data.shape[j]) for j in range(axis + num_dims + 2))

  # Pad with the end halo along the target axis.
  idx = all_slice + (slice(-padding, data.shape[axis + num_dims + 2]),)
  # Roll the end halo along the corresponding factors axis.
  end_pad = np.roll(data[idx], pre + (1,) + post, axis=range(data.ndim))

  # Pad with the start halo along the target axis.
  idx = all_slice + (slice(0, padding),)
  # Roll the start halo along the corresponding factors axis.
  start_pad = np.roll(data[idx], pre + (-1,) + post, axis=range(data.ndim))

  # extend the grid to include the halo
  data = np.concatenate([end_pad, data, start_pad], axis=num_dims + axis + 2)

  data = einops.rearrange(data, combine_factors)

  return data


# Pairwise computation.


def _get_pairwise_displacement(data: Array, grid: TPUGrid) -> Array:
  """Compute displacements over pairs of neighboring particles given a grid."""

  # TODO(jaschasd, schsam): if all we care about is square distance, and not
  # vector displacement, then we are carrying around more channel dimension than
  # we need.
  max_grid_distance = grid.max_grid_distance
  num_dims = grid.num_dims

  data = data[:, np.newaxis]  # [B x L x C x X x Y x ...]

  kernel_width = max_grid_distance * 2 + 1

  axes = ''.join([chr(ord('X') + i) for i in range(num_dims)])

  for axis_ind, axis in enumerate(axes):
    shp = data.shape
    w, b = _generate_offset_kernel(axis, grid)
    dimension_numbers = ('NC' + axes, 'IO' + axes, 'NC' + axes)

    spatial_shape = shp[-num_dims:]

    data = _pad_axis(data, grid.factors, axis_ind, max_grid_distance)
    data = data.reshape((shp[0]*shp[1], shp[2]) + data.shape[-num_dims:])

    data = lax.conv_general_dilated(data, w, (1,) * num_dims, 'VALID',
                                    dimension_numbers=dimension_numbers,
                                    precision=lax.Precision.HIGHEST)
    data += b
    data = data.reshape((shp[0], shp[1]*kernel_width, shp[2]) + spatial_shape)

    # TODO(jaschasd, schsam): if we have an upper bound on number of neighbors,
    # we could use jax.lax.top_k to throw away entries in L corresponding to no
    # particle, while still remaining jit-able.

  # Mask out the self-interaction from the offset calculation.
  data = data.at[:, 0, -1].set(0)

  return data


# Displacement computation utilities.


def _accumulate_recursion(accum_fn: Callable[[Array, Array, Array], Array],
                          data_0: Array,
                          data: Array,
                          axis: int,
                          grid: TPUGrid) -> Array:
  """Recursively sum the result of `accum_fn` applied to pairwise displacements.

  This function recursively goes through the spatial axes and applies the
  seperable convolution to compute displacements along that axis. After applying
  the convolution, this function then iterates over the neighboring particles
  at that level then descends one more level. At the final level of the
  recursion, the function applies the `accum_fn` and then sums the results.

  Args:
    accum_fn: A function that takes displacement vectors and pairs of particle
      IDs.
    data_0: Positions of the central particle that are used to compute
      displacements.
    data: Transformed data from neighboring particles that are used to compute
      offsets from the center of the central pixel / voxel.
    axis: The current axis that we are computing displacements with respect
      to.
    grid: A TPUGrid containing information about the system.

  Returns:
    The result of summing `accum_fn` over batches of neighboring particles.
  """
  num_dims = grid.num_dims
  max_grid_distance = grid.max_grid_distance

  axes = CONV_AXES_STRING[num_dims - 1]
  axis_name = axes[axis]

  kernel_width = max_grid_distance * 2 + 1

  # Compute differences along the current axis.
  w, b = _generate_offset_kernel_channel_last(axis_name, grid)
  dimension_numbers = ('N' + axes + 'C', 'IO' + axes, 'N' + axes + 'C')

  data = lax.conv_general_dilated(data, w, (1,) * num_dims, 'VALID',
                                  dimension_numbers=dimension_numbers,
                                  precision=lax.Precision.HIGHEST)
  data += b
  data = data.reshape(data.shape[:-1] + (kernel_width, num_dims + 1))

  # TODO(jaschasd, schsam): if we have an upper bound on number of neighbors, we
  # could use jax.lax.top_k to throw away entries in L corresponding to no
  # particle, while still remaining jit-able.

  # If we are at the final axis, then apply the `accum_fn`.
  if axis == num_dims-1:
    # Compute the displacement from the neighbors to the particle rather than
    # the center of its grid cell.
    displacements = data_0[..., :num_dims] - data[..., :num_dims]
    accum = accum_fn(displacements, data_0[..., -1], data[..., -1])
    return np.sum(accum, axis=(num_dims+1))

  # Otherwise loop over the neighbors in the current step and recurse.
  else:
    # save memory by performing in serial, and recomputing gradient
    fn = lambda x: _accumulate_recursion(accum_fn, data_0, x, axis + 1, grid)

    def f(i, accum):
      accum += fn(data[..., i, :])
      return accum

    accum = np.zeros(jax.eval_shape(fn, data[..., 0, :]).shape)
    accum = lax.fori_loop(0, kernel_width, f, accum)

    return accum


# Folding and Unfolding utilities.


def _folded_pad(data: Array,
                max_grid_distance: int,
                factors: Tuple[int, ...]) -> Array:
  """Pad an array along all of its axes. Assumes the array is folded."""

  # this is called outside of the pmap

  # we need this to be accurate one space into the halo, so that we accurately
  # capture particles moving one square outside of or into the bulk in
  # `update_grid_locations`.
  pad_size = max_grid_distance

  num_dims = len(factors)
  for i in range(num_dims):
    pad_shp = list(data.shape)
    pad_shp[num_dims + i] = pad_size
    p = np.ones(pad_shp) * np.nan
    data = np.concatenate([p, data, p], axis=num_dims + i)

  return data


def _folded_unpad(data: Array, max_grid_distance: int) -> Array:
  """Unpad an array along all of its axes. Assumes the array is folded."""
  # We need this to be accurate one space into the halo, so that we accurately
  # capture particles moving one square outside of or into the bulk in
  # `update_grid_locations`.
  pad_size = max_grid_distance

  num_dims = data.ndim - 2
  idx = (slice(data.shape[0]),) + (slice(pad_size, -pad_size),) * num_dims
  return data[idx]


def _fold_factors(batch_size: int,
                  grid_shape: Tuple[int, ...],
                  max_grid_distance: int) -> Tuple[int, ...]:
  """Greedily compute fold factors, trying to target a given batch size."""
  num_dims = len(grid_shape)
  max_folds = onp.log(batch_size) / onp.log(2)

  folds = onp.zeros((num_dims,), onp.int32)

  for i in range(int(max_folds)*num_dims):
    # Here we use folds[dim]+2, rather than +1, since we need an even number of
    # cells for pairwise exchange in update_grid_locations.
    dms = [divmod(grid_shape[dim], 2**(folds[dim] + 2))
           for dim in range(num_dims)]
    valid_dms = [(i, dm[0]) for i, dm in enumerate(dms) if dm[1] == 0]

    if len(valid_dms) == 0:
      break

    dim = sorted(valid_dms, key=lambda idm: idm[1], reverse=True)[0][0]

    width, _ = divmod(grid_shape[dim], 2**(folds[dim] + 1))
    _, remain = divmod(grid_shape[dim], 2**(folds[dim] + 2))
    if remain == 0 and width >= max_grid_distance:
      folds[dim] += 1
    else:
      raise ValueError(f'Failed fold. Current folds {folds}, target fold '
                       f'dimension {dim}, grid_shape {grid_shape}, '
                       f'width {width}, remain {remain}.')
    if onp.sum(folds) == max_folds:
      break
  factors = 2 ** folds

  if onp.sum(folds) < max_folds:
    msg = (f'Folds {folds} have sum smaller than target of '
           f'{max_folds}. This corresponds to a batch size of '
           f'{onp.prod(factors)} rather than the target of '
           f'{batch_size}. The grid_shape is {grid_shape}.')
    raise ValueError(msg)

  return factors


def _order_grid_by_factors(num_dims: int) -> Tuple[int, ...]:
  """Produce an ordering dimensions that will bring the factors to the front."""
  front_order = ()
  back_order = ()

  for i in range(num_dims):
    front_order += (2 * i,)
    back_order += (2 * i + 1,)

  return front_order + back_order + (2 * num_dims,)


def _fold_grid(cell_data: Array,
               max_grid_distance: int,
               batch_size: Optional[int] = None,
               factors: Optional[Tuple[int, ...]] = None,
               inner_fold: bool = True) -> Array:
  """Takes data from a contiguous grid and folds it into patches.

  This function takes data in a grid of shape (X, Y, Z, C) and folds it into
  batches of smaller patches whose shape is (fx, fy, fz, x, y, z, C) such that
  fx * x = X, fy * y = Y, fz * z = Z, and fx * fy * fz = batch_size. Sometimes
  it is not possible to find fx, fy, fz that satisfy these constraints exactly
  and so it will sometimes choose a smaller batch size.

  Args:
    data: An array of contiguous data to be folded.
    max_grid_distance: An integer dictating the amount of padding needed so that
      operations will be consistent across patches.
    batch_size: The target batch size, will usually be 128.
    factors: The factors fx, fy, and fz. If none are provided then they will be
      inferred from the batch size.
    inner_fold: A bool dictating whether this is happening inside a `pmap`.

  Returns:
    A new folded array, as described above.
  """
  num_dims = cell_data.ndim - 1

  if factors is None:
    factors = _fold_factors(batch_size, cell_data.shape[:-1], max_grid_distance)

  data_shape = []
  for i in range(num_dims):
    _, ragged = divmod(cell_data.shape[i], factors[i])
    assert ragged == 0
    data_shape += [factors[i], cell_data.shape[i] // factors[i]]
  data_shape += [cell_data.shape[-1]]

  min_size = min([s / f for s, f in zip(cell_data.shape[:-1], factors)])
  if min_size < max_grid_distance:
    # TODO(jaschasd, schsam): How do we know this is happening when splitting
    # across devices?
    print(f'Folded grid may be too small. This failure is happening when '
          f'splitting across devices. Factors: {factors}, '
          f'cell_data.shape: {cell_data.shape}, New data_shape {data_shape}.')

  ordering = _order_grid_by_factors(num_dims)

  cell_data = np.reshape(cell_data, data_shape)
  cell_data = np.transpose(cell_data, ordering)

  if inner_fold:
    cell_data = cell_data.reshape((-1,) + cell_data.shape[num_dims:])
  else:
    cell_data = _folded_pad(cell_data, max_grid_distance, factors)

  return cell_data, factors


def _unfold_grid(cell_data: Array, grid: TPUGrid, inner_fold: bool=True
                ) -> Array:
  """Takes data from a folded grid and unfolds it into a contiguous block."""
  num_dims = grid.num_dims
  max_grid_distance = grid.max_grid_distance
  factors = grid.factors

  if not inner_fold:
    cell_data = np.reshape(cell_data, (-1,) + cell_data.shape[num_dims:])
    cell_data = _folded_unpad(cell_data, max_grid_distance)
    factors = grid.topology

  cell_data = np.reshape(cell_data, factors + cell_data.shape[1:])

  ordering = _order_grid_by_factors(num_dims)
  cell_data = np.transpose(cell_data, onp.argsort(ordering))

  data_shape = ()
  for i in range(num_dims):
    data_shape += (cell_data.shape[2 * i + 1] * factors[i],)
  cell_data = np.reshape(cell_data, data_shape + cell_data.shape[2 * num_dims:])

  return cell_data


# Utilities for synchronizing data across TPU cores.


def _send_next(x: Array, axis_name: str) -> Array:
  """Send data from one TPU core to the next one along a particular axis."""
  # Note: if some devices are omitted from the permutation, lax.ppermute
  # provides zeros instead. This gives us an easy way to apply Dirichlet
  # boundary conditions.
  device_count = lax.psum(1, axis_name)
  perm = [(i, (i + 1) % device_count) for i in range(device_count)]
  return lax.ppermute(x, perm=perm, axis_name=axis_name)


def _send_prev(x: Array, axis_name: str) -> Array:
  """Send data from one TPU core to the previous one along a particular axis."""
  device_count = lax.psum(1, axis_name)
  perm = [((i + 1) % device_count, i) for i in range(device_count)]
  return lax.ppermute(x, perm=perm, axis_name=axis_name)


def _extract_halo(grid_data_unfolded: Array, pad_size: int, axis: str) -> Array:
  """Extract the padding halo from an unfolded grid along a specific axis."""

  if axis == 'X':
    start = grid_data_unfolded[:pad_size]
    end = grid_data_unfolded[-pad_size:]
  elif axis == 'Y':
    start = grid_data_unfolded[:, :pad_size]
    end = grid_data_unfolded[:, -pad_size:]
  elif axis == 'Z':
    start = grid_data_unfolded[:, :, :pad_size]
    end = grid_data_unfolded[:, :, -pad_size:]
  else:
    raise ValueError(f'Expected axis to be X, Y, or Z found {axis}.')

  return start, end


def _mesh_transport(cell_data: Array, grid: TPUGrid) -> Array:
  """If cell data is sharded then ensure consistency across the devices."""
  if not grid.topology:
    return cell_data

  num_dims = grid.num_dims
  axes = ''.join([chr(ord('X') + i) for i in range(num_dims)])
  pad_size = grid.max_grid_distance

  # TODO(jaschasd, scham): Unfolding full image is wasteful. We could just copy
  # the parts we need.
  cell_data = _unfold_grid(cell_data, grid)

  # cut off the halo
  body_slice = tuple(slice(pad_size, -pad_size) for j in range(num_dims))
  cell_data = cell_data[body_slice]

  for axis_index, axis in enumerate(axes):
    start, end = _extract_halo(cell_data, pad_size, axis)
    new_start = _send_next(end, axis)
    new_end = _send_prev(start, axis)
    cell_data = np.concatenate((new_start, cell_data, new_end), axis=axis_index)

  cell_data = _fold_grid(cell_data,
                         grid.max_grid_distance,
                         factors=grid.factors)[0]

  return cell_data


# Dynamics utilities for updating particle cell occupancies.


def _pairwise_exchange(cell_data: Array, axis: int, grid: TPUGrid) -> Array:
  """Exchanges particles along an axis if the swap reduces frustration."""
  # `cell_data` has shape [X x Y x ... C]

  num_dims = grid.num_dims
  aux_size = cell_data.shape[-1] - num_dims

  axes = EINOPS_AXES_STRING[num_dims - 1]
  axes_pre = ' '.join([f'({a} q)' if i == axis else
                       a for i, a in enumerate(axes.split(' '))])
  axes_post = axes
  cell_data = einops.rearrange(cell_data,
                               f'{axes_pre} c -> q {axes_post} c', q=2)

  delta = onp.zeros((2, num_dims + aux_size))
  delta[0, axis] = grid.cell_size
  delta[1, axis] = -grid.cell_size
  delta = delta.reshape((2,) + (1,) * num_dims + (num_dims + aux_size,))
  data_reversed = cell_data[::-1]
  # Only add offset to occupied cells
  data_swap = data_reversed + delta * (data_reversed[..., [-1]] > 0)

  def square_displacement(dat: Array) -> Array:
    return np.sum(dat[..., :num_dims]**2 * (dat[..., [-1]] > 0),
                  axis=(0, -1), keepdims=True)

  keep_mask = (square_displacement(cell_data) <= square_displacement(data_swap))

  data_swap = einops.rearrange(data_swap, f'q {axes_post} c -> {axes_pre} c')
  keep_mask = einops.rearrange(keep_mask, f'q {axes_post} c -> {axes_pre} c')

  return data_swap, keep_mask
_pairwise_exchange = vmap(_pairwise_exchange, in_axes=(0, None, None))


def _update_grid_locations(cell_data: Array, grid: TPUGrid) -> Array:
  """Exchanges particles on the grid if the swap reduces frustration."""

  for axis in range(grid.num_dims):
    data_swapped, keep_mask = _pairwise_exchange(cell_data, axis, grid)

    # Mask is for pairs of particles, so repeat.
    keep_mask = np.repeat(keep_mask, 2, axis + 1)
    cell_data = cell_data * keep_mask + data_swapped * (~keep_mask)

    # Copy the halo, to allow particle exchange across folds.
    cell_data = _pad_axis_channel_last(cell_data, grid.factors, axis)
    data_swapped, keep_mask = _pairwise_exchange(cell_data, axis, grid)

    # Make sure keep_mask is consistent across folds of the halo, by
    # cutting the end halo off, and copying from start halo.
    base_slice = tuple(slice(cell_data.shape[j]) for j in range(axis + 1))
    body_slice = base_slice + (slice(0, -1),)
    keep_mask = _pad_axis_channel_last(keep_mask[body_slice],
                                       grid.factors, axis)
    body_slice = base_slice + (slice(1, cell_data.shape[axis + 1]),)
    keep_mask = keep_mask[body_slice]

    # mask is for pairs of particles, so repeat.
    keep_mask = np.repeat(keep_mask, 2, axis + 1)
    cell_data = cell_data * keep_mask + data_swapped * (~keep_mask)

    # cut the halo back off
    body_slice = base_slice + (slice(1, -1),)
    cell_data = cell_data[body_slice]

  return cell_data


# Auxiliary Data utilities.


def _get_aux(cell_data: Array, aux_tree: TreeDef, aux_sizes: Tuple[int, ...]
             ) -> Tuple[Array, PyTree]:
  """Extract auxiliary data from a grid and shape it into a PyTree."""
  cell_data, *flat_aux_occupancy = np.split(cell_data, aux_sizes, axis=-1)
  flat_aux, occupancy = flat_aux_occupancy[:-1], flat_aux_occupancy[-1]
  cell_data = np.concatenate([cell_data, occupancy], axis=-1)
  aux = tree_unflatten(aux_tree, flat_aux)
  return cell_data, aux


def _set_aux(cell_data: Array, aux: PyTree
             ) -> Tuple[Array, TreeDef, Tuple[int, ...]]:
  """Flattens a PyTree of auxiliary data and adds it to a grid."""
  flat_aux, aux_tree = tree_flatten(aux)
  aux_sizes = [x.shape[-1] for x in flat_aux]
  aux_sizes = onp.cumsum([cell_data.shape[-1] - 1] + aux_sizes)
  cell_data = np.concatenate([cell_data[..., :-1]] +
                             flat_aux +
                             [cell_data[..., -1:]], axis=-1)
  return cell_data, aux_tree, aux_sizes


def _get_aux_spec(num_dims: int, aux: PyTree
                  ) -> Tuple[TreeDef, Tuple[int, ...]]:
  """Extract the structure of auxiliary data."""
  flat_aux, aux_tree = tree_flatten(aux)
  aux_sizes = [x.shape[-1] for x in flat_aux]
  aux_sizes = onp.cumsum([num_dims] + aux_sizes)
  return aux_tree, aux_sizes


# Grid sizing utilities


def _outer_grid_size_to_inner_grid_size(tiled_size,
                                        topology,
                                        factors,
                                        max_grid_distance):
  if topology:
    unpadded_size = tiled_size / onp.array(topology)
    padded_size = unpadded_size + 2*max_grid_distance
  else:
    padded_size = tiled_size
  folded_size = padded_size / onp.array(factors)
  return folded_size


def _inner_grid_size_to_outer_grid_size(folded_size,
                                        topology,
                                        factors,
                                        max_grid_distance):
  padded_size = folded_size * onp.array(factors)
  if topology:
    unpadded_size = padded_size - 2*max_grid_distance
    tiled_size = unpadded_size * onp.array(topology)
  else:
    tiled_size = padded_size
  return tiled_size


# Testing utilities.


def test_nve(force_fn, dt):
  dt_2 = 0.5 * dt ** 2
  def init_fn(position: TPUGrid, velocity: Array, **kwargs) -> NVEState:
    return NVEState(position, velocity, force_fn(position))    # pytype: disable=wrong-arg-count
  def apply_fn(state):
    R = state.position
    V = state.velocity
    F = state.force

    R, (V, F) = shift(R, V * dt + F * dt_2, (V, F))  # pytype: disable=attribute-error
    F_new = force_fn(R)
    V = V + 0.5 * (F + F_new) * dt

    return NVEState(R, V, F_new)  # pytype: disable=wrong-arg-count
  return init_fn, apply_fn



