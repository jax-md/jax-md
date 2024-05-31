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

"""Tests for google3.third_party.py.jax_md.mapping."""

from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized

from functools import partial

import jax

from jax import random
import jax.numpy as np

from jax import grad

from jax import jit, vmap

from jax_md import smap, partition, space, energy, quantity
from jax_md.util import *
from jax_md import test_util

jax.config.parse_flags_with_absl()


PARTICLE_COUNT = 1000
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

if jax.config.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]


class CellListTest(test_util.JAXMDTestCase):

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype,
      } for dtype in POSITION_DTYPE))
  def test_cell_list_emplace_2d(self, dtype):
    box_size = np.array([8.65, 8.0], f32)
    cell_size = f32(1.0)

    R = np.array([
      [0.25, 0.25],
      [8.5, 1.95],
      [8.1, 1.5],
      [3.7, 7.9]
    ], dtype=dtype)

    cell_fn = partition.cell_list(box_size, cell_size)

    cell_list = cell_fn.allocate(R)
    self.assertEqual(cell_list.id_buffer.dtype, jnp.int32)

    self.assertAllClose(R[0], cell_list.position_buffer[0, 0, 0])
    self.assertAllClose(R[1], cell_list.position_buffer[1, 8, 1])
    self.assertAllClose(R[2], cell_list.position_buffer[1, 8, 0])
    self.assertAllClose(R[3], cell_list.position_buffer[7, 3, 1])

    self.assertEqual(0, cell_list.id_buffer[0, 0, 0])
    self.assertEqual(1, cell_list.id_buffer[1, 8, 1])
    self.assertEqual(2, cell_list.id_buffer[1, 8, 0])
    self.assertEqual(3, cell_list.id_buffer[7, 3, 1])

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.position_buffer, (-1, 2))

    R_out = np.zeros((5, 2), dtype)
    R_out = R_out.at[id_flat].set(R_flat)[:-1]
    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_cell_list_random_emplace(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = f32(9.0)
    cell_size = f32(1.0)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim))

    cell_fn = partition.cell_list(box_size, cell_size)
    cell_list = cell_fn.allocate(R)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.position_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim))
    R_out = R_out.at[id_flat].set(R_flat)[:-1]

    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_cell_list_random_emplace_rect(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = np.array([9.0, 3.0, 7.25]) if dim == 3 else np.array([9.0, 3.25])
    cell_size = f32(1.0)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim))

    cell_fn = partition.cell_list(box_size, cell_size)
    cell_list = cell_fn.allocate(R)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.position_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim))
    R_out = R_out.at[id_flat].set(R_flat)[:-1]
    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_cell_list_random_emplace_side_data(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = (np.array([9.0, 4.0, 7.25], f32) if dim == 3 else
                np.array([9.0, 4.25], f32))
    cell_size = f32(1.23)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    side_data_dim = 2
    side_data = random.normal(key, (PARTICLE_COUNT, side_data_dim),
                              dtype=dtype)

    cell_fn = partition.cell_list(box_size, cell_size)
    cell_list = cell_fn.allocate(R, side_data=side_data)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.position_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim), dtype)
    R_out = R_out.at[id_flat].set(R_flat)[:-1]

    side_data_flat = np.reshape(
      cell_list.kwarg_buffers['side_data'], (-1, side_data_dim))
    side_data_out = np.zeros(
      (PARTICLE_COUNT + 1, side_data_dim), dtype)
    side_data_out = side_data_out.at[id_flat].set(side_data_flat)[:-1]

    self.assertAllClose(R_out, R)
    self.assertAllClose(side_data_out, side_data)


class NeighborListTest(test_util.JAXMDTestCase):
  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_neighbor_list_build(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = (
      np.array([9.0, 4.0, 7.25], f32) if dim == 3 else
      np.array([9.0, 4.25], f32))
    cutoff = f32(1.23)

    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]
    neighbor_fn = partition.neighbor_list(
      displacement, box_size, cutoff, 0.0, 1.1)

    idx = neighbor_fn.allocate(R).idx
    R_neigh = R[idx]
    mask = idx < N

    d = vmap(vmap(metric, (None, 0)))
    dR = d(R, R_neigh)

    d_exact = space.map_product(metric)
    dR_exact = d_exact(R, R)

    dR = np.where(dR < cutoff, dR, f32(0)) * mask
    mask_exact = 1. - np.eye(dR_exact.shape[0])
    dR_exact = np.where(dR_exact < cutoff, dR_exact, f32(0)) * mask_exact

    dR = np.sort(dR, axis=1)
    dR_exact = np.sort(dR_exact, axis=1)

    for i in range(dR.shape[0]):
      dR_row = dR[i]
      dR_row = dR_row[dR_row > 0.]

      dR_exact_row = dR_exact[i]
      dR_exact_row = np.array(dR_exact_row[dR_exact_row > 0.], dtype)

      self.assertAllClose(dR_row, dR_exact_row)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_neighbor_list_build_sparse(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = (
      np.array([9.0, 4.0, 7.25], f32) if dim == 3 else
      np.array([9.0, 4.25], f32))
    cutoff = f32(1.23)

    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]
    neighbor_fn = partition.neighbor_list(
      displacement, box_size, cutoff, 0.0, 1.1, format=partition.Sparse)

    nbrs = neighbor_fn.allocate(R)
    mask = partition.neighbor_list_mask(nbrs)

    d = space.map_bond(metric)
    dR = d(R[nbrs.idx[0]], R[nbrs.idx[1]])

    d_exact = space.map_product(metric)
    dR_exact = d_exact(R, R)

    dR = np.where(dR < cutoff, dR, f32(0)) * mask
    mask_exact = 1. - np.eye(dR_exact.shape[0])
    dR_exact = np.where(dR_exact < cutoff, dR_exact, f32(0)) * mask_exact

    dR_exact = np.sort(dR_exact, axis=1)

    for i in range(N):
      dR_row = dR[nbrs.idx[0] == i]
      dR_row = dR_row[dR_row > 0.]
      dR_row = np.sort(dR_row)

      dR_exact_row = dR_exact[i]
      dR_exact_row = np.array(dR_exact_row[dR_exact_row > 0.], dtype)

      self.assertAllClose(dR_row, dR_exact_row)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_neighbor_list_build_time_dependent(self, dtype, dim):
    key = random.PRNGKey(1)

    if dim == 2:
      box_fn = lambda t: np.array(
        [[9.0, t],
         [0.0, 3.75]], f32)
    elif dim == 3:
      box_fn = lambda t: np.array(
        [[9.0, 0.0, t],
         [0.0, 4.0, 0.0],
         [0.0, 0.0, 7.25]])
    min_length = np.min(np.diag(box_fn(0.)))
    cutoff = f32(1.23)
    # TODO(schsam): Get cell-list working with anisotropic cell sizes.
    cell_size = cutoff / min_length

    displacement, _ = space.periodic_general(box_fn(0.0))
    metric = space.metric(displacement)

    R = random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]
    neighbor_list_fn = partition.neighbor_list(metric, 1., cutoff, 0.0,
                                               1.1, cell_size=cell_size,
                                               t=np.array(0.))

    idx = neighbor_list_fn.allocate(R, box=box_fn(np.array(0.25))).idx
    R_neigh = R[idx]
    mask = idx < N

    metric = partial(metric, box=box_fn(0.25))
    d = vmap(vmap(metric, (None, 0)))
    dR = d(R, R_neigh)

    d_exact = space.map_product(metric)
    dR_exact = d_exact(R, R)

    dR = np.where(dR < cutoff, dR, 0) * mask
    dR_exact = np.where(dR_exact < cutoff, dR_exact, 0)

    dR = np.sort(dR, axis=1)
    dR_exact = np.sort(dR_exact, axis=1)

    for i in range(dR.shape[0]):
      dR_row = dR[i]
      dR_row = dR_row[dR_row > 0.]

      dR_exact_row = dR_exact[i]
      dR_exact_row = dR_exact_row[dR_exact_row > 0.]

      self.assertAllClose(dR_row, dR_exact_row)

  def test_cell_list_overflow(self):
    displacement_fn, shift_fn = space.free()

    box = 100.0
    r_cutoff = 3.0
    dr_threshold = 0.0

    neighbor_list_fn = partition.neighbor_list(
      displacement_fn,
      box=box,
      r_cutoff=r_cutoff,
      dr_threshold=dr_threshold,
    )

    # all far from eachother
    R = jnp.array(
      [
        [20.0, 20.0],
        [30.0, 30.0],
        [40.0, 40.0],
        [50.0, 50.0],
    ]
    )
    neighbors = neighbor_list_fn.allocate(R)
    self.assertEqual(neighbors.idx.dtype, jnp.int32)

    # two first point are close to eachother
    R = jnp.array(
      [
        [20.0, 20.0],
        [20.0, 20.0],
        [40.0, 40.0],
        [50.0, 50.0],
      ]
    )

    neighbors = neighbors.update(R)
    self.assertTrue(neighbors.did_buffer_overflow)
    self.assertEqual(neighbors.idx.dtype, jnp.int32)

  def test_custom_mask_function(self):
    displacement_fn, shift_fn = space.free()

    box = 1.0
    r_cutoff = 3.0
    dr_threshold = 0.0
    n_particles = 10
    R = jnp.broadcast_to(jnp.zeros(3), (n_particles,3))

    def acceptable_id_pair(id1, id2):
      '''
      Don't allow particles to have an interaction when their id's
      are closer than 3 (eg disabling 1-2 and 1-3 interactions)
      '''
      return jnp.abs(id1-id2)>3

    def mask_id_based(
        idx: Array,
        ids: Array,
        mask_val: int,
        _acceptable_id_pair: Callable
      ) -> Array:
      '''
      _acceptable_id_pair mapped to act upon the neighbor list where:
          - index of particle 1 is in index in the first dimension of array
          - index of particle 2 is given by the value in the array
      '''
      @partial(vmap, in_axes=(0,0,None))
      def acceptable_id_pair(idx, id1, ids):
        id2 = ids.at[idx].get()
        return vmap(_acceptable_id_pair, in_axes=(None,0))(id1,id2)
      mask = acceptable_id_pair(idx, ids, ids)
      return jnp.where(
        mask,
        idx,
        mask_val
      )

    ids = jnp.arange(n_particles) # id is just particle index here.
    mask_val = n_particles
    custom_mask_function = partial(mask_id_based,
      ids=ids,
      mask_val=mask_val,
      _acceptable_id_pair=acceptable_id_pair
    )

    neighbor_list_fn = partition.neighbor_list(
      displacement_fn,
      box=box,
      r_cutoff=r_cutoff,
      dr_threshold=dr_threshold,
      custom_mask_function=custom_mask_function,
    )

    neighbors = neighbor_list_fn.allocate(R)
    neighbors = neighbors.update(R)
    '''
    Without masking it's 9 neighbors (with mask self) -> 90 neighbors.
    With masking -> 42.
    '''
    self.assertEqual(42, (neighbors.idx!=mask_val).sum())

  def test_issue191_1(self):
    box_vector = jnp.ones(3) * 3

    r_cut = 0.1
    _positions = jnp.linspace(0.5, 0.7, 20)
    positions = jnp.stack([_positions, _positions, _positions], axis=1)

    displacement, _ = space.periodic_general(box_vector)

    neighbor_fn = partition.neighbor_list(displacement, box_vector, r_cut,
                                          0.1 * r_cut,
                                          fractional_coordinates=True)

    neighbor2_fn = partition.neighbor_list(displacement, box_vector[0],
                                           r_cut,
                                           0.1 * r_cut,
                                           fractional_coordinates=True,
                                           disable_cell_list=True)

    nbrs = neighbor_fn.allocate(positions)
    nbrs2 = neighbor2_fn.allocate(positions)
    self.assertAllClose(jnp.sort(nbrs.idx, axis=-1),
                        jnp.sort(nbrs2.idx, axis=-1))

  @parameterized.named_parameters(test_util.cases_from_list(
    {
      'testcase_name': f'_case={i}_mask_self={ms}_format={fmt.name}',
      'r_cut': r_cut,
      'disable_cell_list': dc,
      'capacity_multiplier': cm,
      'mask_self': ms,
      'format': fmt,
    } for i, (r_cut, dc, cm) in enumerate(
      [(0.12, True, 1.5), (0.25, False, 1.5),
       (0.31, False, 1.5), (0.31, False, 1.0)]) for ms in [False, True]
      for fmt in [partition.Dense, partition.Sparse, partition.OrderedSparse]))

  def test_issue191_2(self, r_cut, disable_cell_list, capacity_multiplier,
                      mask_self, format):
    box = onp.ones(3)
    # box = 1.0
    if format is partition.Dense:
      desired_shape = (20, 19) if mask_self else (20, 20)
      _positions = jnp.ones((20,)) * 0.5
    elif format is partition.Sparse:
      desired_shape = (2, 20 * 19) if mask_self else (2, 20**2)
      _positions = jnp.ones((20,)) * 0.5
    elif format is partition.OrderedSparse:
      desired_shape = (2, 20 * 19 // 2)
      _positions = jnp.ones((20,)) * 0.5
    positions = jnp.stack([_positions, _positions, _positions], axis=1)
    displacement, _ = space.periodic(box)

    neighbor_fn = partition.neighbor_list(
      displacement, box,
      r_cut, 0.1 * r_cut,
      capacity_multiplier=capacity_multiplier,
      disable_cell_list=disable_cell_list,
      mask_self=mask_self,
      format=format)

    nbrs = neighbor_fn.allocate(positions)

    self.assertFalse(nbrs.did_buffer_overflow)
    self.assertEqual(nbrs.idx.shape, desired_shape)

    new_nbrs = nbrs.update(positions + 0.1)
    self.assertFalse(new_nbrs.did_buffer_overflow)
    self.assertEqual(new_nbrs.idx.shape, desired_shape)

  def test_general_unit_cell(self):
    N = 128
    dim = 3
    rho = 0.05

    energy_form = lambda dr, **kwargs: jnp.where(dr < 2.5, 1.0, 0.0)

    def cell(key, a, b, c, alpha, beta, gamma):
      alpha = alpha * jnp.pi / 180
      beta = beta * jnp.pi / 180
      gamma = gamma * jnp.pi / 180
      xx = a
      yy = b * jnp.sin(gamma)
      xy = b * jnp.cos(gamma)
      xz = c * jnp.cos(beta)
      yz = (b * c * jnp.cos(alpha) - xy * xz) / yy
      zz = jnp.sqrt(c**2 - xz**2 - yz**2)
      U = jnp.array([
        [xx, xy, xz],
        [0,  yy, yz],
        [0,  0,  zz]
      ])
      return U
    key = random.PRNGKey(0)

    L = quantity.box_size_at_number_density(N, rho, dim)

    for _ in range(10):
      key, cell_key, pos_key, L_key, angle_key = random.split(key, 5)
      Lx, Ly, Lz = random.uniform(L_key, (3,), minval=L * 0.5, maxval=L * 3)
      alpha, beta, gamma = random.uniform(angle_key, (3,),
                                          minval=45.0, maxval=125.0)
      cl = cell(cell_key, Lx, Ly, Lz, alpha, beta, gamma)
      if jnp.any(jnp.isnan(cl)):
        continue
      displacement, shift = space.periodic_general(cl)
      metric = space.metric(displacement)

      E_exact = smap.pair(energy_form, metric, reduce_axis=(-1,))
      E = smap.pair_neighbor_list(energy_form, metric, reduce_axis=(-1,))
      neighbor_fn = partition.neighbor_list(displacement, cl, 2.5,
                                            capacity_multiplier=1.0,
                                            fractional_coordinates=True)

      R = random.uniform(key, (N, dim))
      nbrs = neighbor_fn.allocate(R)
      E_target = E_exact(R)
      self.assertTrue(jnp.any(jnp.abs(E_target) > 0.25))
      self.assertAllClose(E_target, E(R, neighbor=nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
    {
      'testcase_name': f'_factor={int(10 * f)}',
      'factor': f
    } for f in [0.5, 2.0]))
  def test_general_unit_cell_resize(self, factor):
    N = 128
    dim = 3
    rho = 0.05

    energy_form = lambda dr, **kwargs: jnp.where(dr < 2.5, 1.0, 0.0)

    def cell(key, a, b, c, alpha, beta, gamma):
      alpha = alpha * jnp.pi / 180
      beta = beta * jnp.pi / 180
      gamma = gamma * jnp.pi / 180
      xx = a
      yy = b * jnp.sin(gamma)
      xy = b * jnp.cos(gamma)
      xz = c * jnp.cos(beta)
      yz = (b * c * jnp.cos(alpha) - xy * xz) / yy
      zz = jnp.sqrt(c**2 - xz**2 - yz**2)
      U = jnp.array([
        [xx, xy, xz],
        [0,  yy, yz],
        [0,  0,  zz]
      ])
      return U
    key = random.PRNGKey(0)

    L = quantity.box_size_at_number_density(N, rho, dim)

    for _ in range(10):
      key, cell_key, pos_key, L_key, angle_key = random.split(key, 5)
      Lx, Ly, Lz = random.uniform(L_key, (3,), minval=L * 0.5, maxval=L * 3)
      alpha, beta, gamma = random.uniform(angle_key, (3,),
                                          minval=45.0, maxval=125.0)
      cl = cell(cell_key, Lx, Ly, Lz, alpha, beta, gamma)
      if jnp.any(jnp.isnan(cl)):
        continue
      displacement, shift = space.periodic_general(cl)
      metric = space.metric(displacement)

      E_exact = smap.pair(energy_form, metric, reduce_axis=(-1,))
      E = smap.pair_neighbor_list(energy_form, metric, reduce_axis=(-1,))
      neighbor_fn = partition.neighbor_list(displacement, cl, 2.5,
                                            capacity_multiplier=1.0,
                                            fractional_coordinates=True)

      R = random.uniform(key, (N, dim))
      nbrs = neighbor_fn.allocate(R)
      nbrs = nbrs.update(R, box=cl * factor)
      E_target = E_exact(R, box=cl * factor)

      if factor > 1:
        self.assertTrue(jnp.any(jnp.abs(E_target) > 0.25))
        self.assertAllClose(E_target, E(R, neighbor=nbrs, box=cl * factor))
      else:
        self.assertTrue(jnp.any(nbrs.cell_size_too_small))


if __name__ == '__main__':
  absltest.main()
