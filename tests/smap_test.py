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
from jax import jit, vmap

from jax_md import smap, space, energy, quantity, partition
from jax_md.util import *
from jax_md import test_util

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS

test_util.update_test_tolerance(f32_tolerance=5e-6, f64_tolerance=1e-14)

PARTICLE_COUNT = 1000
STOCHASTIC_SAMPLES = 3
SPATIAL_DIMENSION = [2, 3]

NEIGHBOR_LIST_PARTICLE_COUNT = 100

if FLAGS.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]

class SMapTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_bond_no_type_static(self, spatial_dimension, dtype):
    harmonic = lambda dr, **kwargs: (dr - f32(1)) ** f32(2)
    disp, _ = space.free()
    metric = space.metric(disp)

    mapped = smap.bond(harmonic, metric, np.array([[0, 1], [0, 2]], i32))

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      accum = harmonic(metric(R[0], R[1])) + harmonic(metric(R[0], R[2]))

      self.assertAllClose(mapped(R), dtype(accum))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_bond_no_type_dynamic(self, spatial_dimension, dtype):
    harmonic = lambda dr, **kwargs: (dr - f32(1)) ** f32(2)
    disp, _ = space.free()
    metric = space.metric(disp)

    mapped = smap.bond(harmonic, metric)
    bonds = np.array([[0, 1], [0, 2]], i32)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      accum = harmonic(metric(R[0], R[1])) + harmonic(metric(R[0], R[2]))

      self.assertAllClose(mapped(R, bonds), dtype(accum))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_bond_type_static(self, spatial_dimension, dtype):
    harmonic = lambda dr, sigma, **kwargs: (dr - sigma) ** f32(2)
    disp, _ = space.free()
    metric = space.metric(disp)

    sigma = np.array([1.0, 2.0], f32)

    mapped = smap.bond(
      harmonic, metric,
      np.array([[0, 1], [0, 2]], i32), np.array([0, 1], i32), sigma=sigma)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      accum = harmonic(metric(R[0], R[1]), 1) + harmonic(metric(R[0], R[2]), 2)

      self.assertAllClose(mapped(R), dtype(accum))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_bond_type_dynamic(self, spatial_dimension, dtype):
    harmonic = lambda dr, sigma, **kwargs: (dr - sigma) ** f32(2)
    disp, _ = space.free()
    metric = space.metric(disp)

    sigma = np.array([1.0, 2.0], f32)

    mapped = smap.bond(harmonic, metric, sigma=sigma)
    bonds = np.array([[0, 1], [0, 2]], i32)
    bond_types = np.array([0, 1], i32)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      accum = harmonic(metric(R[0], R[1]), 1) + harmonic(metric(R[0], R[2]), 2)

      self.assertAllClose(mapped(R, bonds, bond_types), dtype(accum))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_bond_params_dynamic(self, spatial_dimension, dtype):
    harmonic = lambda dr, sigma, **kwargs: (dr - sigma) ** f32(2)
    disp, _ = space.free()
    metric = space.metric(disp)

    sigma = np.array([1.0, 2.0], f32)

    mapped = smap.bond(harmonic, metric)
    bonds = np.array([[0, 1], [0, 2]], i32)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      accum = harmonic(metric(R[0], R[1]), 1) + harmonic(metric(R[0], R[2]), 2)

      self.assertAllClose(mapped(R, bonds, sigma=sigma), dtype(accum))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_bond_per_bond_static(self, spatial_dimension, dtype):
    harmonic = lambda dr, sigma, **kwargs: (dr - sigma) ** f32(2)
    disp, _ = space.free()
    metric = space.metric(disp)

    sigma = np.array([1.0, 2.0], f32)

    mapped = smap.bond(
      harmonic, metric, np.array([[0, 1], [0, 2]], i32), sigma=sigma)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      accum = harmonic(metric(R[0], R[1]), 1) + harmonic(metric(R[0], R[2]), 2)

      self.assertAllClose(mapped(R), dtype(accum))

  def test_get_species_parameters(self):
    species = [(0, 0), (0, 1), (1, 0), (1, 1)]
    params = np.array([[2.0, 3.0], [3.0, 1.0]])
    global_params = 3.0
    self.assertAllClose(
        smap._get_species_parameters(params, species[0]), 2.0)
    self.assertAllClose(
        smap._get_species_parameters(params, species[1]), 3.0)
    self.assertAllClose(
        smap._get_species_parameters(params, species[2]), 3.0)
    self.assertAllClose(
        smap._get_species_parameters(params, species[3]), 1.0)
    for s in species:
      self.assertAllClose(
          smap._get_species_parameters(global_params, s), 3.0)

  def test_get_matrix_parameters(self):
    params = np.array([1.0, 2.0])
    params_mat_test = np.array([[1.0, 1.5], [1.5, 2.0]])
    params_mat = smap._get_matrix_parameters(params)
    self.assertAllClose(params_mat, params_mat_test)

    params_mat_direct = np.array([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllClose(
        smap._get_matrix_parameters(params_mat_direct), params_mat_direct)

    params_scalar = 1.0
    self.assertAllClose(
        smap._get_matrix_parameters(params_scalar), params_scalar)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_no_species_scalar(self, spatial_dimension, dtype):
    square = lambda dr: dr ** 2
    displacement, _ = space.free()
    metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

    mapped_square = smap.pair(square, metric)
    metric = space.map_product(metric)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      self.assertAllClose(
        mapped_square(R),
        np.array(0.5 * np.sum(square(metric(R, R))), dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_no_species_scalar_dynamic(self, spatial_dimension, dtype):
    square = lambda dr, epsilon: epsilon * dr ** 2
    displacement, _ = space.free()
    metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

    mapped_square = smap.pair(square, metric)
    metric = space.map_product(metric)

    key = random.PRNGKey(0)
    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)
      R = random.uniform(
        split1, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      epsilon = random.uniform(split2, (PARTICLE_COUNT,), dtype=dtype)
      mat_epsilon = 0.5 * (epsilon[:, np.newaxis] + epsilon[np.newaxis, :])
      self.assertAllClose(
        mapped_square(R, epsilon=epsilon),
        np.array(0.5 * np.sum(
          square(metric(R, R), mat_epsilon)), dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_no_species_vector(self, spatial_dimension, dtype):
    square = lambda dr: np.sum(dr ** 2, axis=2)
    disp, _ = space.free()

    mapped_square = smap.pair(square, disp)

    disp = space.map_product(disp)
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      mapped_ref = np.array(0.5 * np.sum(square(disp(R, R))), dtype=dtype)
      self.assertAllClose(mapped_square(R), mapped_ref)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_static_species_scalar(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * dr ** 2
    params = f32(np.array([[1.0, 2.0], [2.0, 3.0]]))

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    displacement, _ = space.free()
    metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

    mapped_square = smap.pair(
      square, metric, species=species, param=params)

    metric = space.map_product(metric)

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
      self.assertAllClose(mapped_square(R), np.array(total, dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_static_species_scalar_dynamic(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * dr ** 2

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    displacement, _ = space.free()
    metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

    mapped_square = smap.pair(square, metric, species=species)

    metric = space.map_product(metric)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)
      R = random.uniform(
        split1, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      params = random.uniform(
        split2, (2, 2), dtype=dtype)
      params = f32(0.5) * (params + params.T)
      total = 0.0
      for i in range(2):
        for j in range(2):
          param = params[i, j]
          R_1 = R[species == i]
          R_2 = R[species == j]
          total = total + 0.5 * np.sum(square(metric(R_1, R_2), param))
      self.assertAllClose(
        mapped_square(R, param=params), np.array(total, dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_scalar_dummy_arg(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=f32(1.0), **unused_kwargs: param * dr ** 2

    key, split = random.split(key)
    R = random.normal(key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    displacement, shift = space.free()

    mapped = smap.pair(square, space.metric(displacement))

    mapped(R, t=f32(0))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_static_species_vector(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * np.sum(dr ** 2, axis=2)
    params = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=f32)

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    disp, _ = space.free()

    mapped_square = smap.pair(square, disp, species=species, param=params)

    disp = space.map_product(disp)

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
      self.assertAllClose(mapped_square(R), np.array(total, dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_dynamic_species_scalar(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * dr ** 2
    params = f32(np.array([[1.0, 2.0], [2.0, 3.0]]))

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    displacement, _ = space.free()
    metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

    mapped_square = smap.pair(
        square, metric, species=quantity.Dynamic, param=params)

    metric = space.map_product(metric)

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
        mapped_square(R, species, 2), np.array(total, dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_dynamic_species_vector(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * np.sum(dr ** 2, axis=2)
    params = f32(np.array([[1.0, 2.0], [2.0, 3.0]]))

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    disp, _ = space.free()

    mapped_square = smap.pair(
        square, disp, species=quantity.Dynamic, param=params)

    disp = vmap(vmap(disp, (0, None), 0), (None, 0), 0)

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
        mapped_square(R, species, 2), np.array(total, dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_neighbor_list_scalar(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = jit(smap.pair_neighbor_list(truncated_square, d))
    mapped_square = jit(smap.pair(truncated_square, d))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0)
      nbrs = neighbor_fn(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_neighbor_list_scalar_diverging_potential(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    def potential(dr, sigma):
      return np.where(dr < sigma, dr ** -6, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = jit(smap.pair_neighbor_list(potential, d))
    mapped_square = jit(smap.pair(potential, d))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0)
      nbrs = neighbor_fn(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_neighbor_list_force_scalar_diverging_potential(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    def potential(dr, sigma):
      return np.where(dr < sigma, dr ** -6, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = jit(quantity.force(smap.pair_neighbor_list(potential, d)))
    mapped_square = jit(quantity.force(smap.pair(potential, d)))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=4.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0)
      nbrs = neighbor_fn(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_neighbor_list_scalar_params_no_species(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = jit(smap.pair_neighbor_list(truncated_square, d))
    mapped_square = jit(smap.pair(truncated_square, d))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (N,), minval=0.5, maxval=1.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, np.max(sigma), 0.)
      nbrs = neighbor_fn(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_neighbor_list_scalar_params_matrix(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = jit(smap.pair_neighbor_list(truncated_square, d))
    mapped_square = jit(smap.pair(truncated_square, d))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (N, N), minval=0.5, maxval=1.5)
      sigma = 0.5 * (sigma + sigma.T)
      neighbor_fn = partition.neighbor_list(disp, box_size, np.max(sigma), 0.)
      nbrs = neighbor_fn(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_neighbor_list_scalar_params_species(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)
    species = np.zeros((N,), np.int32)
    species = np.where(np.arange(N) > N / 3, 1, species)
    species = np.where(np.arange(N) > 2 * N / 3, 2, species)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = jit(
      smap.pair_neighbor_list(truncated_square, d, species=species))
    mapped_square = jit(smap.pair(truncated_square, d, species=species))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (3, 3), minval=0.5, maxval=1.5)
      sigma = 0.5 * (sigma + sigma.T)
      neighbor_fn = partition.neighbor_list(disp, box_size, np.max(sigma), 0.)
      nbrs = neighbor_fn(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(jtu.cases_from_list(
    {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_neighbor_list_vector(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    def truncated_square(dR, sigma):
      dr = np.reshape(space.distance(dR), dR.shape[:-1] + (1,))
      return np.where(dr < sigma, dR ** 2, f32(0.))

    N = PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)

    neighbor_square = jit(smap.pair_neighbor_list(
      truncated_square, disp, reduce_axis=(1,)))
    mapped_square = jit(smap.pair(truncated_square, disp, reduce_axis=(1,)))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=1.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.)
      nbrs = neighbor_fn(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

if __name__ == '__main__':
  absltest.main()
