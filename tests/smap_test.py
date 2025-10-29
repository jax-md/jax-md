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

from absl.testing import absltest
from absl.testing import parameterized

import jax

from jax import random
import jax.numpy as np

from jax import grad

from jax import jit, vmap

from jax_md import smap, space, energy, quantity, partition, dataclasses
from jax_md.util import *
from jax_md import test_util

jax.config.parse_flags_with_absl()

test_util.update_test_tolerance(f32_tol=5e-6, f64_tol=1e-14)

PARTICLE_COUNT = 1000
STOCHASTIC_SAMPLES = 3
SPATIAL_DIMENSION = [2, 3]
NEIGHBOR_LIST_FORMAT = [partition.Dense,
                        partition.Sparse,
                        partition.OrderedSparse]

NEIGHBOR_LIST_PARTICLE_COUNT = 100

if jax.config.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]

class SMapTest(test_util.JAXMDTestCase):

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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

    mapped = smap.bond(harmonic, metric, sigma=1.0)
    bonds = np.array([[0, 1], [0, 2]], i32)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      accum = harmonic(metric(R[0], R[1]), 1) + harmonic(metric(R[0], R[2]), 2)

      self.assertAllClose(mapped(R, bonds, sigma=sigma), dtype(accum))

  @parameterized.named_parameters(test_util.cases_from_list(
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
    params_mat = smap._get_matrix_parameters(params, lambda x, y: 0.5 * (x + y))
    self.assertAllClose(params_mat, params_mat_test)

    params_mat_direct = np.array([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllClose(
        smap._get_matrix_parameters(params_mat_direct, None), params_mat_direct)

    params_scalar = 1.0
    self.assertAllClose(
        smap._get_matrix_parameters(params_scalar, None), params_scalar)

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}',
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_pair_no_species_pytree(self, dtype):
    square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
    square_higher = lambda dr, p: p[0] * dr ** 2 + p[1]

    @dataclasses.dataclass
    class Parameter:
      scale: Array
      shift: Array

    tree_fn = lambda dr, p: p.scale * dr**2 + p.shift
    displacement, _ = space.free()
    metric = space.metric(displacement)

    p = np.array([1.0, 2.0])
    M = smap.ParameterTreeMapping
    mapped_scalar = smap.pair(square_scalar, metric, p0=p[0], p1=p[1])
    mapped_higher = smap.pair(square_higher, metric,
                              p=smap.ParameterTree(p, M.Global))

    p_tree = smap.ParameterTree(Parameter(scale=p[0], shift=p[1]), M.Global)
    mapped_tree = smap.pair(tree_fn, metric, p=p_tree)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(split, (PARTICLE_COUNT, 2), dtype=dtype)
      self.assertAllClose(mapped_scalar(R), mapped_higher(R))
      self.assertAllClose(mapped_scalar(R), mapped_tree(R))

  @parameterized.named_parameters(test_util.cases_from_list(
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

    mapped_square = smap.pair(square, metric, epsilon=1.0)
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

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}',
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_pair_no_species_pytree_per_particle(self, dtype):
    square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
    square_higher = lambda dr, p: p[..., 0] * dr ** 2 + p[..., 1]

    @dataclasses.dataclass
    class Parameter:
      scale: Array
      shift: Array
    tree_fn = lambda dr, p: p.scale * dr**2 + p.shift

    displacement, _ = space.free()
    metric = space.metric(displacement)

    p = random.uniform(random.PRNGKey(1), (PARTICLE_COUNT, 2))
    M = smap.ParameterTreeMapping
    mapped_scalar = smap.pair(square_scalar, metric, p0=p[:, 0], p1=p[:, 1])
    p_higher = smap.ParameterTree(p, M.PerParticle)
    mapped_higher = smap.pair(square_higher, metric, p=p_higher)

    p_tree = smap.ParameterTree(Parameter(scale=p[:, 0], shift=p[:, 1]),
                                M.PerParticle)
    mapped_tree = smap.pair(tree_fn, metric, p=p_tree)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(split, (PARTICLE_COUNT, 2), dtype=dtype)
      self.assertAllClose(mapped_scalar(R), mapped_higher(R))
      self.assertAllClose(mapped_scalar(R), mapped_tree(R))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}',
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_pair_no_species_pytree_order_per_bond(self, dtype):
    square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
    square_higher = lambda dr, p: p[..., 0] * dr ** 2 + p[..., 1]

    @dataclasses.dataclass
    class Parameter:
      scale: Array
      shift: Array
    tree_fn = lambda dr, p: p.scale * dr**2 + p.shift

    displacement, _ = space.free()
    metric = space.metric(displacement)

    p = random.uniform(random.PRNGKey(1), (PARTICLE_COUNT, PARTICLE_COUNT, 2))
    M = smap.ParameterTreeMapping
    mapped_scalar = smap.pair(square_scalar, metric,
                              p0=p[..., 0], p1=p[..., 1])
    mapped_higher = smap.pair(square_higher, metric,
                              p=smap.ParameterTree(p, M.PerBond))

    p_tree = smap.ParameterTree(Parameter(scale=p[..., 0], shift=p[..., 1]),
                                M.PerBond)
    mapped_tree = smap.pair(tree_fn, metric, p=p_tree)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(split, (PARTICLE_COUNT, 2), dtype=dtype)
      self.assertAllClose(mapped_scalar(R), mapped_higher(R))
      self.assertAllClose(mapped_scalar(R), mapped_tree(R))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_no_species_vector_nonadditive(self, spatial_dimension, dtype):
    square = lambda dr, params: params * np.sum(dr ** 2, axis=2)
    disp, _ = space.free()

    mapped_square = smap.pair(square, disp, params=lambda x, y: x * y)

    disp = space.map_product(disp)
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, R_key, params_key = random.split(key, 3)
      R = random.uniform(
        R_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      params = random.uniform(
        params_key, (PARTICLE_COUNT,), dtype=dtype, minval=0.1, maxval=1.5)
      pp_params = params[None, :] * params[:, None]
      mapped_ref = np.array(0.5 * np.sum(square(disp(R, R), pp_params)),
                            dtype=dtype)
      self.assertAllClose(mapped_square(R, params=params), mapped_ref)

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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

    mapped_square = smap.pair(square, metric, species=species, param=1.0)

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

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}',
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_pair_species_pytree_global(self, dtype):
    square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
    square_higher = lambda dr, p: p[..., 0] * dr ** 2 + p[..., 1]

    @dataclasses.dataclass
    class Parameter:
      scale: Array
      shift: Array
    square_tree = lambda dr, p: p.scale * dr**2 + p.shift

    displacement, _ = space.free()
    metric = space.metric(displacement)

    p = np.array([1.0, 2.0])
    M = smap.ParameterTreeMapping
    species = np.where(np.arange(PARTICLE_COUNT) < PARTICLE_COUNT // 2, 0, 1)
    mapped_scalar = smap.pair(square_scalar, metric, species=species,
                              p0=p[0], p1=p[1])
    p_h = smap.ParameterTree(p, M.Global)
    mapped_higher = smap.pair(square_higher, metric, species=species, p=p_h)

    p_tree = smap.ParameterTree(Parameter(p[0], p[1]), M.Global)
    mapped_tree = smap.pair(square_tree, metric, species=species, p=p_tree)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(split, (PARTICLE_COUNT, 2), dtype=dtype)
      self.assertAllClose(mapped_scalar(R), mapped_higher(R))
      self.assertAllClose(mapped_scalar(R), mapped_tree(R))


  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}',
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_pair_species_pytree_per_species(self, dtype):
    square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
    square_higher = lambda dr, p: p[..., 0] * dr ** 2 + p[..., 1]

    @dataclasses.dataclass
    class Parameter:
      scale: Array
      shift: Array
    square_tree = lambda dr, p: p.scale * dr**2 + p.shift

    displacement, _ = space.free()
    metric = space.metric(displacement)

    p = random.uniform(random.PRNGKey(1), (2, 2, 2))
    p = p + np.transpose(p, (1, 0, 2))
    species = np.where(np.arange(PARTICLE_COUNT) < PARTICLE_COUNT // 2, 0, 1)
    mapped_scalar = smap.pair(square_scalar, metric, species=species,
                              p0=p[..., 0], p1=p[..., 1])
    M = smap.ParameterTreeMapping
    p_h = smap.ParameterTree(p, M.PerSpecies)
    mapped_higher = smap.pair(square_higher, metric, species=species, p=p_h)

    p_tree = smap.ParameterTree(Parameter(p[..., 0], p[..., 1]), M.PerSpecies)
    mapped_tree = smap.pair(square_tree, metric, species=species, p=p_tree)

    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = random.uniform(split, (PARTICLE_COUNT, 2), dtype=dtype)
      self.assertAllClose(mapped_scalar(R), mapped_higher(R))
      self.assertAllClose(mapped_scalar(R), mapped_tree(R))

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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
    metric = space.metric(displacement)

    mapped_square = smap.pair(square, metric, species=2, param=params)

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
      self.assertAllClose(mapped_square(R, species),
                          np.array(total, dtype=dtype))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_pair_dynamic_species_vector(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    square = lambda dr, param=1.0: param * np.sum(dr ** 2, axis=-1)
    params = f32(np.array([[1.0, 2.0], [2.0, 3.0]]))

    key, split = random.split(key)
    species = random.randint(split, (PARTICLE_COUNT,), 0, 2)
    disp, _ = space.free()

    mapped_square = smap.pair(square, disp, species=2, param=params)

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
      self.assertAllClose(mapped_square(R, species),
                          np.array(total, dtype=dtype))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_scalar(self, spatial_dimension, dtype, format):
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = smap.pair_neighbor_list(truncated_square, d, sigma=1.0)
    neighbor_square = jit(neighbor_square)
    mapped_square = jit(smap.pair(truncated_square, d, sigma=1.0))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dtype={dtype.__name__}'
                            f'_format={format}'),
          'dtype': dtype,
          'format': format
      } for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_pytree(self, dtype, format):
    key = random.PRNGKey(0)
    dim = 2

    def scalar_fn(dr, sigma, shift):
      return np.where(dr < sigma, dr ** 2 + shift, f32(0.))

    def higher_order_fn(dr, p):
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      return np.where(dr < p[..., 0], dr ** 2 + p[..., 1], f32(0.))

    @dataclasses.dataclass
    class Parameter:
      sigma: Array
      shift: Array

    def tree_fn(dr, p):
      return np.where(dr < p.sigma, dr ** 2 + p.shift, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / dim)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    sigma = 1.0
    shift = 2.0
    M = smap.ParameterTreeMapping
    neighbor_scalar = smap.pair_neighbor_list(scalar_fn, d,
                                              sigma=sigma, shift=shift)
    p = smap.ParameterTree(jnp.array([sigma, shift], dtype), M.Global)
    neighbor_higher = smap.pair_neighbor_list(higher_order_fn, d, p=p)

    p_tree = smap.ParameterTree(Parameter(sigma=sigma, shift=shift), M.Global)
    neighbor_tree = smap.pair_neighbor_list(tree_fn, d, p=p_tree)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, dim), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_higher(R, nbrs))
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_tree(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dtype={dtype.__name__}'
                            f'_format={str(format).split(".")[-1]}'),
          'dtype': dtype,
          'format': format
      } for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_per_atom_pytree(self, dtype, format):
    key = random.PRNGKey(0)
    dim = 2

    def scalar_fn(dr, sigma, shift):
      return np.where(dr < sigma, dr ** 2 + shift, f32(0.))

    def higher_order_fn(dr, p):
      return np.where(dr < p[..., 0], dr ** 2 + p[..., 1], f32(0.))

    @dataclasses.dataclass
    class Parameter:
      sigma: Array
      shift: Array

    def tree_fn(dr, p):
      return np.where(dr < p.sigma, dr ** 2 + p.shift, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / dim)

    key, split, split2 = random.split(key, 3)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    sigma = random.uniform(split, (N,), minval=0.5, maxval=1.0, dtype=dtype)
    shift = random.uniform(split2, (N,), dtype=dtype)
    M = smap.ParameterTreeMapping
    neighbor_scalar = smap.pair_neighbor_list(scalar_fn, d,
                                              sigma=sigma, shift=shift)
    p = smap.ParameterTree(np.concatenate([sigma[:, None], shift[:, None]],
                                          axis=-1),
                           M.PerParticle)
    neighbor_higher = smap.pair_neighbor_list(higher_order_fn, d, p=p)

    p_tree = smap.ParameterTree(Parameter(sigma=sigma, shift=shift),
                                M.PerParticle)
    neighbor_tree = smap.pair_neighbor_list(tree_fn, d, p=p_tree)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, dim), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_higher(R, nbrs))
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_tree(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dtype={dtype.__name__}'
                            f'_format={str(format).split(".")[-1]}'),
          'dtype': dtype,
          'format': format
      } for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_per_bond_pytree(self, dtype, format):
    key = random.PRNGKey(0)
    dim = 2

    def scalar_fn(dr, sigma, shift):
      return np.where(dr < sigma, dr ** 2 + shift, f32(0.))

    def higher_order_fn(dr, p):
      return np.where(dr < p[..., 0], dr ** 2 + p[..., 1], f32(0.))

    @dataclasses.dataclass
    class Parameter:
      sigma: Array
      shift: Array

    def tree_fn(dr, p):
      return np.where(dr < p.sigma, dr ** 2 + p.shift, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / dim)

    key, split, split2 = random.split(key, 3)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    sigma = random.uniform(split, (N, N), minval=0.5, maxval=1.0, dtype=dtype)
    shift = random.uniform(split2, (N, N), dtype=dtype)
    M = smap.ParameterTreeMapping
    neighbor_scalar = smap.pair_neighbor_list(scalar_fn, d,
                                              sigma=sigma, shift=shift)
    p = smap.ParameterTree(np.concatenate([sigma[:, :, None],
                                           shift[:, :, None]],
                                          axis=-1),
                           M.PerBond)
    neighbor_higher = smap.pair_neighbor_list(higher_order_fn, d, p=p)

    p_tree = smap.ParameterTree(Parameter(sigma=sigma, shift=shift),
                                M.PerBond)
    neighbor_tree = smap.pair_neighbor_list(tree_fn, d, p=p_tree)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, dim), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_higher(R, nbrs))
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_tree(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dtype={dtype.__name__}'
                            f'_format={str(format).split(".")[-1]}'),
          'dtype': dtype,
          'format': format
      } for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_species_global_pytree(self, dtype, format):
    key = random.PRNGKey(0)
    dim = 2

    def scalar_fn(dr, sigma, shift):
      return np.where(dr < sigma, dr ** 2 + shift, f32(0.))

    def higher_order_fn(dr, p):
      return np.where(dr < p[0], dr ** 2 + p[1], f32(0.))

    @dataclasses.dataclass
    class Parameter:
      sigma: Array
      shift: Array

    def tree_fn(dr, p):
      return np.where(dr < p.sigma, dr ** 2 + p.shift, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / dim)

    key, split, split2 = random.split(key, 3)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    sigma = f32(1.5)
    shift = f32(2.0)
    species = jnp.where(jnp.arange(N) < N // 2, 0, 1)
    M = smap.ParameterTreeMapping
    neighbor_scalar = smap.pair_neighbor_list(scalar_fn, d, species=species,
                                              sigma=sigma, shift=shift)
    p = smap.ParameterTree(np.array([sigma, shift]),
                           M.Global)
    neighbor_higher = smap.pair_neighbor_list(higher_order_fn, d,
                                              species=species, p=p)

    p_tree = smap.ParameterTree(Parameter(sigma=sigma, shift=shift),
                                M.Global)
    neighbor_tree = smap.pair_neighbor_list(tree_fn, d,
                                            species=species, p=p_tree)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, dim), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_higher(R, nbrs))
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_tree(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dtype={dtype.__name__}'
                            f'_format={str(format).split(".")[-1]}'),
          'dtype': dtype,
          'format': format
      } for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_species_per_species_pytree(self, dtype, format):
    key = random.PRNGKey(0)
    dim = 2

    def scalar_fn(dr, sigma, shift):
      return np.where(dr < sigma, dr ** 2 + shift, f32(0.))

    def higher_order_fn(dr, p):
      return np.where(dr < p[..., 0], dr ** 2 + p[..., 1], f32(0.))

    @dataclasses.dataclass
    class Parameter:
      sigma: Array
      shift: Array

    def tree_fn(dr, p):
      return np.where(dr < p.sigma, dr ** 2 + p.shift, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / dim)

    key, split, split2 = random.split(key, 3)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    sigma = jnp.array([[1.0, 1.2], [1.2, 1.5]], f32)
    shift = jnp.array([[2.0, 1.5], [1.5, 3.0]], f32)
    species = jnp.where(jnp.arange(N) < N // 2, 0, 1)
    M = smap.ParameterTreeMapping
    neighbor_scalar = smap.pair_neighbor_list(scalar_fn, d, species=species,
                                              sigma=sigma, shift=shift)
    p = smap.ParameterTree(np.concatenate([sigma[..., None], shift[..., None]],
                                          axis=-1),
                           M.PerSpecies)
    neighbor_higher = smap.pair_neighbor_list(higher_order_fn, d,
                                              species=species, p=p)

    p_tree = smap.ParameterTree(Parameter(sigma=sigma, shift=shift),
                                M.PerSpecies)
    neighbor_tree = smap.pair_neighbor_list(tree_fn, d,
                                            species=species, p=p_tree)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, dim), dtype=dtype)
      neighbor_fn = partition.neighbor_list(disp, box_size, jnp.max(sigma),
                                            0.0, format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_higher(R, nbrs))
      self.assertAllClose(neighbor_scalar(R, nbrs),
                          neighbor_tree(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_scalar_diverging_potential(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(0)

    def potential(dr, sigma):
      return np.where(dr < sigma, dr ** -6, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = jit(smap.pair_neighbor_list(potential, d, sigma=1.0))
    mapped_square = jit(smap.pair(potential, d, sigma=1.0))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=2.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_force_scalar_diverging_potential(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(0)

    def potential(dr, sigma):
      return np.where(dr < sigma, dr ** -6, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 4. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = smap.pair_neighbor_list(potential, d, sigma=1.0)
    neighbor_square = jit(quantity.force(neighbor_square))
    mapped_square = jit(quantity.force(smap.pair(potential, d, sigma=1.0)))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=4.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.0,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_scalar_params_no_species(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = smap.pair_neighbor_list(truncated_square, d, sigma=1.0)
    neighbor_square = jit(neighbor_square)
    mapped_square = jit(smap.pair(truncated_square, d, sigma=1.0))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (N,), minval=0.5, maxval=1.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, np.max(sigma), 0.,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_scalar_params_matrix(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = smap.pair_neighbor_list(truncated_square, d, sigma=1.0)
    neighbor_square = jit(neighbor_square)
    mapped_square = jit(smap.pair(truncated_square, d, sigma=1.0))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (N, N), minval=0.5, maxval=1.5)
      sigma = 0.5 * (sigma + sigma.T)
      neighbor_fn = partition.neighbor_list(disp, box_size, np.max(sigma), 0.,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_scalar_params_species(
      self, spatial_dimension, dtype, format):
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

    neighbor_square = smap.pair_neighbor_list(truncated_square,
                                              d, species=species, sigma=1.0)
    neighbor_square = jit(neighbor_square)
    mapped_square = smap.pair(truncated_square, d, species=species, sigma=1.0)
    mapped_square = jit(mapped_square)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (3, 3), minval=0.5, maxval=1.5)
      sigma = 0.5 * (sigma + sigma.T)
      neighbor_fn = partition.neighbor_list(disp, box_size, np.max(sigma), 0.,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_scalar_params_species_dynamic(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma, **kwargs):
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = NEIGHBOR_LIST_PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)
    species = np.zeros((N,), np.int32)
    species = np.where(np.arange(N) > N / 3, 1, species)
    species = np.where(np.arange(N) > 2 * N / 3, 2, species)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    d = space.metric(disp)

    neighbor_square = smap.pair_neighbor_list(truncated_square, d, sigma=1.0)
    neighbor_square = jit(neighbor_square)
    mapped_square = smap.pair(truncated_square, d, species=species, sigma=1.0)
    mapped_square = jit(mapped_square)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (3, 3), minval=0.5, maxval=1.5)
      sigma = 0.5 * (sigma + sigma.T)
      neighbor_fn = partition.neighbor_list(disp, box_size, np.max(sigma), 0.,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(
        mapped_square(R, sigma=sigma),
        neighbor_square(R, nbrs, sigma=sigma, species=species))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={str(format).split(".")[-1]}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_vector(self, spatial_dimension, dtype, format):
    if format is partition.OrderedSparse:
      self.skipTest('Vector valued pair_neighbor_list not supported.')
    key = random.PRNGKey(0)

    def truncated_square(dR, sigma):
      dr = np.reshape(space.distance(dR), dR.shape[:-1] + (1,))
      return np.where(dr < sigma, dR ** 2, f32(0.))

    N = PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)

    neighbor_square = jit(smap.pair_neighbor_list(
      truncated_square, disp, sigma=1.0, reduce_axis=(1,)))
    mapped_square = jit(smap.pair(truncated_square,
                                  disp, sigma=1.0, reduce_axis=(1,)))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (), minval=0.5, maxval=1.5)
      neighbor_fn = partition.neighbor_list(disp, box_size, sigma, 0.,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_vector_nonadditive(
      self, spatial_dimension, dtype, format):

    if format is partition.OrderedSparse:
      self.skipTest('Vector valued pair_neighbor_list not supported.')

    key = random.PRNGKey(0)

    def truncated_square(dR, sigma):
      dr = space.distance(dR)
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)

    neighbor_square = jit(smap.pair_neighbor_list(
      truncated_square, disp, sigma=lambda x, y: x * y,
      reduce_axis=(1,)))
    mapped_square = jit(smap.pair(truncated_square,
                                  disp, sigma=1.0, reduce_axis=(1,)))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (N,), minval=0.5, maxval=1.5)
      sigma_pair = sigma[:, None] * sigma[None, :]
      neighbor_fn = partition.neighbor_list(disp,
                                            box_size,
                                            np.max(sigma) ** 2,
                                            0.,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma_pair),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                            f'_format={format}'),
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_neighbor_list_scalar_nonadditive(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(0)

    def truncated_square(dR, sigma):
      dr = space.distance(dR)
      return np.where(dr < sigma, dr ** 2, f32(0.))

    N = PARTICLE_COUNT
    box_size = 2. * N ** (1. / spatial_dimension)

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)

    neighbor_square = jit(smap.pair_neighbor_list(
      truncated_square, disp, sigma=lambda x, y: x * y))
    mapped_square = jit(smap.pair(truncated_square, disp, sigma=1.0))

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(
        split, (N, spatial_dimension), dtype=dtype)
      sigma = random.uniform(key, (N,), minval=0.5, maxval=1.5)
      sigma_pair = sigma[:, None] * sigma[None, :]
      neighbor_fn = partition.neighbor_list(disp,
                                            box_size,
                                            np.max(sigma) ** 2,
                                            0.,
                                            format=format)
      nbrs = neighbor_fn.allocate(R)
      self.assertAllClose(mapped_square(R, sigma=sigma_pair),
                          neighbor_square(R, nbrs, sigma=sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_triplet_no_species_scalar(self, spatial_dimension, dtype):
      key = random.PRNGKey(0)

      angle_fn = lambda dR1, dR2: np.sum(np.square(dR1) + np.square(dR2))
      square = lambda dR: np.sum(np.square(dR))
      displacement, _ = space.free()
      metric = lambda Ra, Rb, **kwargs: \
          np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)

      triplet_square = smap.triplet(angle_fn, displacement)
      metric = space.map_product(metric)

      count = PARTICLE_COUNT // 50

      for _ in range(STOCHASTIC_SAMPLES):
        key, split = random.split(key)
        R = random.uniform(
            split, (count, spatial_dimension), dtype=dtype)

        self.assertAllClose(
            triplet_square(R) / count / 2.,
            np.array(0.5 * np.sum(metric(R, R)), dtype=dtype))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_triplet_static_species_scalar(self, spatial_dimension, dtype):
      key = random.PRNGKey(0)
      angle_fn = lambda dR1, dR2, param=5.0: param * np.sum(np.square(dR1))
      square = lambda dR, param: param * np.sum(np.square(dR))
      params = f32(np.array([[[1., 1.], [2., 0.]], [[0., 2.], [1., 1.]]]))

      count = PARTICLE_COUNT // 50
      key, split = random.split(key)
      species = random.randint(split, (count,), 0, 2)
      displacement, _ = space.free()
      metric = lambda Ra, Rb, **kwargs: \
        np.sum(displacement(Ra, Rb, **kwargs) ** 2, axis=-1)
      triplet_square = smap.triplet(angle_fn,
                                    displacement,
                                    species=species,
                                    param=params,
                                    reduce_axis=None)

      metric = space.map_product(metric)
      for _ in range(STOCHASTIC_SAMPLES):
        key, split = random.split(key)
        R = random.uniform(
            split, (count, spatial_dimension), dtype=dtype)
        total = 0.
        for i in range(2):
          for j in range(2):
            R_1 = R[species == i]
            R_2 = R[species == j]
            total += 0.5 * np.sum(metric(R_1, R_2))
        self.assertAllClose(triplet_square(R) / count, np.array(total, dtype=dtype))

if __name__ == '__main__':
  absltest.main()
