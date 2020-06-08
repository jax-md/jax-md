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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

from functools import partial

from jax.config import config as jax_config

from jax import random
import jax.numpy as np
from jax import ops

from jax.api import grad

from jax import test_util as jtu
from jax import jit, vmap

from jax_md import smap, partition, space, energy, quantity
from jax_md.util import *

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 1000
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

if FLAGS.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]

class CellListTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
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

    cell_fn = partition.cell_list(box_size, cell_size, R)

    cell_list = cell_fn(R)

    self.assertAllClose(R[0], cell_list.R_buffer[0, 0, 0])
    self.assertAllClose(R[1], cell_list.R_buffer[1, 8, 1])
    self.assertAllClose(R[2], cell_list.R_buffer[1, 8, 0])
    self.assertAllClose(R[3], cell_list.R_buffer[7, 3, 1])

    self.assertEqual(0, cell_list.id_buffer[0, 0, 0])
    self.assertEqual(1, cell_list.id_buffer[1, 8, 1])
    self.assertEqual(2, cell_list.id_buffer[1, 8, 0])
    self.assertEqual(3, cell_list.id_buffer[7, 3, 1])

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.R_buffer, (-1, 2))

    R_out = np.zeros((5, 2), dtype)
    R_out = ops.index_update(R_out, id_flat, R_flat)[:-1]
    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(jtu.cases_from_list(
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

    cell_fn = partition.cell_list(box_size, cell_size, R)
    cell_list = cell_fn(R)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.R_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim))
    R_out = ops.index_update(R_out, id_flat, R_flat)[:-1]

    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_cell_list_random_emplace_rect(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = np.array([9.0, 3.0, 7.25]) if dim is 3 else np.array([9.0, 3.25])
    cell_size = f32(1.0)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim))

    cell_fn = partition.cell_list(box_size, cell_size, R)
    cell_list = cell_fn(R)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.R_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim))
    R_out = ops.index_update(R_out, id_flat, R_flat)[:-1]
    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_cell_list_random_emplace_side_data(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = (np.array([9.0, 4.0, 7.25], f32) if dim is 3 else
                np.array([9.0, 4.25], f32))
    cell_size = f32(1.23)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    side_data_dim = 2
    side_data = random.normal(key, (PARTICLE_COUNT, side_data_dim), dtype=dtype)

    cell_fn = partition.cell_list(box_size, cell_size, R)
    cell_list = cell_fn(R, side_data=side_data)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.R_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim), dtype)
    R_out = ops.index_update(R_out, id_flat, R_flat)[:-1]

    side_data_flat = np.reshape(
      cell_list.kwarg_buffers['side_data'], (-1, side_data_dim))
    side_data_out = np.zeros(
      (PARTICLE_COUNT + 1, side_data_dim), dtype)
    side_data_out = ops.index_update(
      side_data_out, id_flat, side_data_flat)[:-1]

    self.assertAllClose(R_out, R)
    self.assertAllClose(side_data_out, side_data)

class NeighborListTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
          'dtype': dtype,
          'dim': dim,
      } for dtype in POSITION_DTYPE for dim in SPATIAL_DIMENSION))
  def test_neighbor_list_build(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = (
      np.array([9.0, 4.0, 7.25], f32) if dim is 3 else
      np.array([9.0, 4.25], f32))
    cutoff = f32(1.23)

    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]
    neighbor_fn = partition.neighbor_list(
      displacement, box_size, cutoff, 0.0, 1.1)

    idx = neighbor_fn(R).idx
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

  @parameterized.named_parameters(jtu.cases_from_list(
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

    displacement, _ = space.periodic_general(box_fn)
    metric = space.metric(displacement)

    R = random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]
    neighbor_list_fn = partition.neighbor_list(metric, 1., cutoff, 0.0,
                                               1.1, cell_size=cell_size,
                                               t=np.array(0.))

    idx = neighbor_list_fn(R, t=np.array(0.25)).idx
    R_neigh = R[idx]
    mask = idx < N

    metric = partial(metric, t=f32(0.25))
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
if __name__ == '__main__':
  absltest.main()
