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

from jax.config import config as jax_config

from jax import random
import jax.numpy as np

from jax.api import grad

from jax import test_util as jtu
from jax import jit

from jax_md import smap, space, energy, quantity
from jax_md.util import *

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 100
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

if FLAGS.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]

class SMapTest(jtu.JaxTestCase):

  def test_get_species_parameters(self):
    species = [(0, 0), (0, 1), (1, 0), (1, 1)]
    params = np.array([[2.0, 3.0], [3.0, 1.0]])
    global_params = 3.0
    self.assertAllClose(
        smap._get_species_parameters(params, species[0]), 2.0, True)
    self.assertAllClose(
        smap._get_species_parameters(params, species[1]), 3.0, True)
    self.assertAllClose(
        smap._get_species_parameters(params, species[2]), 3.0, True)
    self.assertAllClose(
        smap._get_species_parameters(params, species[3]), 1.0, True)
    for s in species:
      self.assertAllClose(
          smap._get_species_parameters(global_params, s), 3.0, True)

  def test_get_matrix_parameters(self):
    params = np.array([1.0, 2.0])
    params_mat_test = np.array([[2.0, 3.0], [3.0, 4.0]])
    params_mat = smap._get_matrix_parameters(params)
    self.assertAllClose(params_mat, params_mat_test, True)

    params_mat_direct = np.array([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllClose(
        smap._get_matrix_parameters(params_mat_direct), params_mat_direct, True)

    params_scalar = 1.0
    self.assertAllClose(
        smap._get_matrix_parameters(params_scalar), params_scalar, True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_no_species_scalar(self, spatial_dimension, dtype):
    square = lambda dr: dr ** 2
    displacement, _ = space.free()
    metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

    mapped_square = smap.pairwise(square, metric)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      self.assertAllClose(
        mapped_square(R),
        np.array(0.5 * np.sum(square(metric(R, R))), dtype=dtype), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_no_species_vector(self, spatial_dimension, dtype):
    square = lambda dr: np.sum(dr ** 2, axis=2)
    disp, _ = space.free()

    mapped_square = smap.pairwise(square, disp)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      mapped_ref = np.array(0.5 * np.sum(square(disp(R, R))), dtype=dtype)
      self.assertAllClose(mapped_square(R), mapped_ref, True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_static_species_scalar(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * dr ** 2
    params = f32(np.array([[1.0, 2.0], [2.0, 3.0]]))

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    displacement, _ = space.free()
    metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

    mapped_square = smap.pairwise(
      square, metric, species=species, param=params)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      total = 0.0
      for i in range(2):
        for j in range(2):
          param = params[i, j]
          R_1 = R[species == i]
          R_2 = R[species == j]
          total = total + 0.5 * np.sum(square(metric(R_1, R_2), param))
      self.assertAllClose(mapped_square(R), np.array(total, dtype=dtype), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_static_species_vector(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * np.sum(dr ** 2, axis=2)
    params = f32(np.array([[1.0, 2.0], [2.0, 3.0]]))

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    disp, _ = space.free()

    mapped_square = smap.pairwise(square, disp, species=species, param=params)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      total = 0.0
      for i in range(2):
        for j in range(2):
          param = params[i, j]
          R_1 = R[species == i]
          R_2 = R[species == j]
          total = total + 0.5 * np.sum(square(disp(R_1, R_2), param))
      self.assertAllClose(mapped_square(R), np.array(total, dtype=dtype), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_dynamic_species_scalar(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * dr ** 2
    params = f32(np.array([[1.0, 2.0], [2.0, 3.0]]))

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    displacement, _ = space.free()
    metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

    mapped_square = smap.pairwise(
        square, metric, species=quantity.Dynamic, param=params)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      total = 0.0
      for i in range(2):
        for j in range(2):
          param = params[i, j]
          R_1 = R[species == i]
          R_2 = R[species == j]
          total = total + 0.5 * np.sum(square(metric(R_1, R_2), param))
      self.assertAllClose(
        mapped_square(R, species, 2), np.array(total, dtype=dtype), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}'.format(dim),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def disabled_test_pairwise_dynamic_species_vector(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * np.sum(dr ** 2, axis=2)
    params = f32(np.array([[1.0, 2.0], [2.0, 3.0]]))

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    disp, _ = space.free()

    mapped_square = smap.pairwise(
        square, disp, species=quantity.Dynamic, param=params)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      total = 0.0
      for i in range(2):
        for j in range(2):
          param = params[i, j]
          R_1 = R[species == i]
          R_2 = R[species == j]
          total = total + 0.5 * np.sum(square(disp(R_1, R_2), param))
      self.assertAllClose(
        mapped_square(R, species, 2), np.array(total, dtype=dtype), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_grid_energy(self, spatial_dimension, dtype):
    key = random.PRNGKey(1)

    box_size = f16(9.0)
    cell_size = f16(2.0)
    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)
    energy_fn = smap.pairwise(
        energy.soft_sphere, metric, quantity.Dynamic,
        reduce_axis=(1,), keepdims=True)

    R = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    grid_energy_fn = smap.grid(energy_fn, box_size, cell_size, R)
    species = np.zeros((PARTICLE_COUNT,), dtype=np.int64)
    self.assertAllClose(
      np.array(energy_fn(R, species, 1), dtype=dtype),
      grid_energy_fn(R), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_grid_force(self, spatial_dimension, dtype):
    key = random.PRNGKey(1)

    box_size = f16(9.0)
    cell_size = f16(2.0)
    displacement, _ = space.periodic(box_size)
    energy_fn = energy.soft_sphere_pairwise(displacement, quantity.Dynamic)
    force_fn = quantity.force(energy_fn)

    R = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    grid_force_fn = smap.grid(force_fn, box_size, cell_size, R)
    species = np.zeros((PARTICLE_COUNT,), dtype=np.int64)
    self.assertAllClose(
      np.array(force_fn(R, species, 1), dtype=dtype), grid_force_fn(R), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_grid_force_jit(self, spatial_dimension, dtype):
    key = random.PRNGKey(1)

    box_size = f16(9.0)
    cell_size = f16(2.0)
    displacement, _ = space.periodic(box_size)
    energy_fn = energy.soft_sphere_pairwise(displacement, quantity.Dynamic)
    force_fn = quantity.force(energy_fn)

    R = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    grid_force_fn = jit(smap.grid(force_fn, box_size, cell_size, R))
    species = np.zeros((PARTICLE_COUNT,), dtype=np.int64)
    self.assertAllClose(
      np.array(force_fn(R, species, 1), dtype=dtype), grid_force_fn(R), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_grid_force_incommensurate(self, spatial_dimension, dtype):
    key = random.PRNGKey(1)

    box_size = f32(12.1)
    cell_size = f32(3.0)
    displacement, _ = space.periodic(box_size)
    energy_fn = energy.soft_sphere_pairwise(displacement, quantity.Dynamic)
    force_fn = quantity.force(energy_fn)

    R = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    grid_force_fn = jit(smap.grid(force_fn, box_size, cell_size, R))
    species = np.zeros((PARTICLE_COUNT,), dtype=np.int64)
    self.assertAllClose(
      np.array(force_fn(R, species, 1), dtype=dtype), grid_force_fn(R), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pairwise_grid_force_nonuniform(self, spatial_dimension, dtype):
    key = random.PRNGKey(1)

    if spatial_dimension == 2:
      box_size = f32(np.array([[8.0, 10.0]]))
    else:
      box_size = f32(np.array([[8.0, 10.0, 12.0]]))

    cell_size = f32(2.0)
    displacement, _ = space.periodic(box_size)
    energy_fn = energy.soft_sphere_pairwise(displacement, quantity.Dynamic)
    force_fn = quantity.force(energy_fn)

    R = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    grid_force_fn = smap.grid(force_fn, box_size, cell_size, R)
    species = np.zeros((PARTICLE_COUNT,), dtype=np.int64)
    self.assertAllClose(
      np.array(force_fn(R, species, 1), dtype=dtype), grid_force_fn(R), True)

if __name__ == '__main__':
  absltest.main()
