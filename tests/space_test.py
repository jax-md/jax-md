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

"""Tests for jax_md.space."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
import jax.numpy as jnp

from jax import grad, jit, jacfwd

from jax_md import space, test_util, quantity, energy
from jax_md.util import *

from functools import partial
from unittest import SkipTest

test_util.update_test_tolerance(5e-5, 5e-13)

jax.config.parse_flags_with_absl()

PARTICLE_COUNT = 10
STOCHASTIC_SAMPLES = 10
SHIFT_STEPS = 10
SPATIAL_DIMENSION = [2, 3]
BOX_FORMATS = ['scalar', 'vector', 'matrix']

if jax.config.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]


def make_periodic_general_test_system(N, dim, dtype, box_format):
  assert box_format in BOX_FORMATS

  box_size = quantity.box_size_at_number_density(N, 1.0, dim)
  box = dtype(box_size)
  if box_format == 'vector':
    box = jnp.array(jnp.ones(dim) * box_size, dtype)
  elif box_format == 'matrix':
    box = jnp.array(jnp.eye(dim) * box_size, dtype)

  d, s = space.periodic(jnp.diag(box) if box_format == 'matrix' else box)
  d_gf, s_gf = space.periodic_general(box)
  d_g, s_g = space.periodic_general(box, fractional_coordinates=False)

  key = random.PRNGKey(0)

  R_f = random.uniform(key, (N, dim), dtype=dtype)
  R = space.transform(box, R_f)

  E = jit(energy.soft_sphere_pair(d))
  E_gf = jit(energy.soft_sphere_pair(d_gf))
  E_g = jit(energy.soft_sphere_pair(d_g))

  return R_f, R, box, (s, E), (s_gf, E_gf), (s_g, E_g)


# pylint: disable=invalid-name
class SpaceTest(test_util.JAXMDTestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_transform(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)

      R = random.normal(
        split1, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      T = random.normal(
        split2, (spatial_dimension, spatial_dimension), dtype=dtype)

      R_prime_exact = jnp.array(jnp.einsum('ij,kj->ki', T, R), dtype=dtype)
      R_prime = space.transform(T, R)

      self.assertAllClose(R_prime_exact, R_prime)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}'.format(dim),
          'spatial_dimension': dim
      } for dim in SPATIAL_DIMENSION))
  def test_transform_grad(self, spatial_dimension):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)

      R = random.normal(split1, (PARTICLE_COUNT, spatial_dimension))
      T = random.normal(split2, (spatial_dimension, spatial_dimension))

      R_prime = space.transform(T, R)

      energy_direct = lambda R: jnp.sum(R ** 2)
      energy_indirect = lambda T, R: jnp.sum(space.transform(T, R) ** 2)

      grad_direct = grad(energy_direct)(R_prime)
      grad_indirect = grad(energy_indirect, 1)(T, R)

      self.assertAllClose(grad_direct, grad_indirect)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_transform_inverse(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    tol = 1e-13
    if dtype is f32:
      tol = 1e-5

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)

      R = random.normal(
        split1, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      T = random.normal(
        split2, (spatial_dimension, spatial_dimension), dtype=dtype)
      T_inv = space.inverse(T)

      R_test = space.transform(T_inv, space.transform(T, R))

      self.assertAllClose(R, R_test)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_canonicalize_displacement_or_metric(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    displacement, _ = space.periodic_general(jnp.eye(spatial_dimension))
    metric = space.metric(displacement)
    test_metric = space.canonicalize_displacement_or_metric(displacement)

    metric = space.map_product(metric)
    test_metric = space.map_product(test_metric)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)

      R = random.normal(
        split1, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      self.assertAllClose(metric(R, R), test_metric(R, R))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_displacement(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)

      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      dR = space.map_product(space.pairwise_displacement)(R, R)

      dR_wrapped = space.periodic_displacement(f32(1.0), dR)

      dR_direct = dR
      dr_direct = space.distance(dR)
      dr_direct = jnp.reshape(dr_direct, dr_direct.shape + (1,))

      if spatial_dimension == 2:
        for i in range(-1, 2):
          for j in range(-1, 2):
            dR_shifted = dR + jnp.array([i, j], dtype=R.dtype)

            dr_shifted = space.distance(dR_shifted)
            dr_shifted = jnp.reshape(dr_shifted, dr_shifted.shape + (1,))

            dR_direct = jnp.where(dr_shifted < dr_direct, dR_shifted, dR_direct)
            dr_direct = jnp.where(dr_shifted < dr_direct, dr_shifted, dr_direct)
      elif spatial_dimension == 3:
        for i in range(-1, 2):
          for j in range(-1, 2):
            for k in range(-1, 2):
              dR_shifted = dR + jnp.array([i, j, k], dtype=R.dtype)

              dr_shifted = space.distance(dR_shifted)
              dr_shifted = jnp.reshape(dr_shifted, dr_shifted.shape + (1,))

              dR_direct = jnp.where(
                  dr_shifted < dr_direct, dR_shifted, dR_direct)
              dr_direct = jnp.where(
                  dr_shifted < dr_direct, dr_shifted, dr_direct)

      dR_direct = jnp.array(dR_direct, dtype=dR.dtype)
      assert dR_wrapped.dtype == dtype
      self.assertAllClose(dR_wrapped, dR_direct)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_shift(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)

      R = random.uniform(
        split1, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      dR = jnp.sqrt(f32(0.1)) * random.normal(
          split2, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      dR = jnp.where(dR > 0.49, f32(0.49), dR)
      dR = jnp.where(dR < -0.49, f32(-0.49), dR)

      R_shift = space.periodic_shift(f32(1.0), R, dR)

      assert R_shift.dtype == R.dtype
      assert jnp.all(R_shift < 1.0)
      assert jnp.all(R_shift > 0.0)

      dR_after = space.periodic_displacement(f32(1.0), R_shift - R)

      assert dR_after.dtype == R.dtype
      self.assertAllClose(dR_after, dR)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_against_periodic_general(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    tol = 1e-13
    if dtype is f32:
      tol = 1e-5

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2, split3 = random.split(key, 4)

      max_box_size = f32(10.0)
      box_size = max_box_size * random.uniform(
        split1, (spatial_dimension,), dtype=dtype)
      transform = jnp.diag(box_size)

      R = random.uniform(
        split2, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      R_scaled = R * box_size

      dR = random.normal(
        split3, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      disp_fn, shift_fn = space.periodic(box_size)
      general_disp_fn, general_shift_fn = space.periodic_general(transform)

      disp_fn = space.map_product(disp_fn)
      general_disp_fn = space.map_product(general_disp_fn)

      self.assertAllClose(disp_fn(R_scaled, R_scaled), general_disp_fn(R, R))
      assert disp_fn(R_scaled, R_scaled).dtype == dtype
      self.assertAllClose(
          shift_fn(R_scaled, dR), general_shift_fn(R, dR) * box_size)
      assert shift_fn(R_scaled, dR).dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_against_periodic_general_grad(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    tol = 1e-13
    if dtype is f32:
      tol = 1e-5

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2, split3 = random.split(key, 4)

      max_box_size = f32(10.0)
      box_size = max_box_size * random.uniform(
        split1, (spatial_dimension,), dtype=dtype)
      transform = jnp.diag(box_size)

      R = random.uniform(
        split2, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      R_scaled = R * box_size

      dR = random.normal(
        split3, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      disp_fn, shift_fn = space.periodic(box_size)
      general_disp_fn, general_shift_fn = space.periodic_general(transform)

      disp_fn = space.map_product(disp_fn)
      general_disp_fn = space.map_product(general_disp_fn)

      grad_fn = grad(lambda R: jnp.sum(disp_fn(R, R) ** 2))
      general_grad_fn = grad(lambda R: jnp.sum(general_disp_fn(R, R) ** 2))

      self.assertAllClose(grad_fn(R_scaled), general_grad_fn(R))
      assert general_grad_fn(R).dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_general_dynamic(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    eye = jnp.eye(spatial_dimension)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_T0_scale, split_T0_dT = random.split(key, 3)
      key, split_T1_scale, split_T1_dT = random.split(key, 3)
      key, split_t, split_R, split_dR = random.split(key, 4)

      size_0 = 10.0 * random.uniform(split_T0_scale, ())
      dtransform_0 = 0.5 * random.normal(
        split_T0_dT, (spatial_dimension, spatial_dimension))
      T_0 = jnp.array(size_0 * (eye + dtransform_0), dtype=dtype)

      size_1 = 10.0 * random.uniform(split_T1_scale, (), dtype=dtype)
      dtransform_1 = 0.5 * random.normal(
          split_T1_dT, (spatial_dimension, spatial_dimension), dtype=dtype)
      T_1 = jnp.array(size_1 * (eye + dtransform_1), dtype=dtype)

      disp_fn, shift_fn = space.periodic_general(T_0)
      true_disp_fn, true_shift_fn = space.periodic_general(T_1)

      disp_fn = partial(disp_fn, box=T_1)

      disp_fn = space.map_product(disp_fn)
      true_disp_fn = space.map_product(true_disp_fn)

      R = random.uniform(
        split_R, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      dR = random.normal(
        split_dR, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      self.assertAllClose(
        disp_fn(R, R), jnp.array(true_disp_fn(R, R), dtype=dtype))
      self.assertAllClose(
        shift_fn(R, dR, box=T_1), jnp.array(true_shift_fn(R, dR), dtype=dtype))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_general_wrapped_vs_unwrapped(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    eye = jnp.eye(spatial_dimension, dtype=dtype)

    tol = 1e-13
    if dtype is f32:
      tol = 2e-5

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_R, split_T = random.split(key, 3)

      dT = random.normal(
        split_T, (spatial_dimension, spatial_dimension), dtype=dtype)
      T = eye + dT + jnp.transpose(dT)

      R = random.uniform(
        split_R, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      R0 = R
      unwrapped_R = R

      displacement, shift = space.periodic_general(T)
      _, unwrapped_shift = space.periodic_general(T, wrapped=False)

      displacement = space.map_product(displacement)

      for _ in range(SHIFT_STEPS):
        key, split = random.split(key)
        dR = random.normal(
          split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
        R = shift(R, dR)
        unwrapped_R = unwrapped_shift(unwrapped_R, dR)
        self.assertAllClose(
          displacement(R, R0),
          displacement(unwrapped_R, R0))
      assert not (jnp.all(unwrapped_R > 0) and jnp.all(unwrapped_R < 1))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_box_format={box_format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'box_format': box_format
      } for dim in SPATIAL_DIMENSION
    for dtype in POSITION_DTYPE
    for box_format in BOX_FORMATS))
  def test_periodic_general_energy(self, spatial_dimension, dtype, box_format):
    N = 16
    R_f, R, box, (s, E), (s_gf, E_gf), (s_g, E_g) = \
      make_periodic_general_test_system(N, spatial_dimension, dtype, box_format)
    self.assertAllClose(E(R), E_gf(R_f))
    self.assertAllClose(E(R), E_g(R))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_box_format={box_format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'box_format': box_format
      } for dim in SPATIAL_DIMENSION
    for dtype in POSITION_DTYPE
    for box_format in BOX_FORMATS))
  def test_periodic_general_force(self, spatial_dimension, dtype, box_format):
    N = 16
    R_f, R, box, (s, E), (s_gf, E_gf), (s_g, E_g) = \
      make_periodic_general_test_system(N, spatial_dimension, dtype, box_format)
    self.assertAllClose(grad(E)(R), grad(E_gf)(R_f))
    self.assertAllClose(grad(E)(R), grad(E_g)(R))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_box_format={box_format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'box_format': box_format
      } for dim in SPATIAL_DIMENSION
    for dtype in POSITION_DTYPE
    for box_format in BOX_FORMATS))
  def test_periodic_general_shift(self, spatial_dimension, dtype, box_format):
    N = 16
    R_f, R, box, (s, E), (s_gf, E_gf), (s_g, E_g) = \
      make_periodic_general_test_system(N, spatial_dimension, dtype, box_format)

    R_new = s(R, grad(E)(R))
    R_gf_new = s_gf(R_f, grad(E_gf)(R_f))
    R_g_new = s_g(R, grad(E_g)(R))

    self.assertAllClose(R_new, space.transform(box, R_gf_new))
    self.assertAllClose(R_new, R_g_new)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_box_format={box_format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'box_format': box_format
      } for dim in SPATIAL_DIMENSION
    for dtype in POSITION_DTYPE
    for box_format in BOX_FORMATS))
  def test_periodic_general_deform(self, spatial_dimension, dtype, box_format):
    N = 16
    R_f, R, box, (s, E), (s_gf, E_gf), (s_g, E_g) = \
      make_periodic_general_test_system(N, spatial_dimension, dtype, box_format)
    deformed_box = box * 0.9
    self.assertAllClose(E_gf(R_f, box=deformed_box),
                        E_g(R, new_box=deformed_box))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_box_format={box_format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'box_format': box_format
      } for dim in SPATIAL_DIMENSION
    for dtype in POSITION_DTYPE
    for box_format in BOX_FORMATS))
  def test_periodic_general_deform_grad(self,
                                        spatial_dimension, dtype, box_format):
    N = 16
    R_f, R, box, (s, E), (s_gf, E_gf), (s_g, E_g) = \
      make_periodic_general_test_system(N, spatial_dimension, dtype, box_format)
    deformed_box = box * 0.9
    self.assertAllClose(grad(E_gf)(R_f, box=deformed_box),
                        grad(E_g)(R, new_box=deformed_box))

    self.assertAllClose(jacfwd(E_gf)(R_f, box=deformed_box),
                        jacfwd(E_g)(R, new_box=deformed_box))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_box_format={box_format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'box_format': box_format
      } for dim in SPATIAL_DIMENSION
    for dtype in POSITION_DTYPE
    for box_format in BOX_FORMATS))
  def test_periodic_general_deform_shift(self,
                                        spatial_dimension, dtype, box_format):
    N = 16
    R_f, R, box, (s, E), (s_gf, E_gf), (s_g, E_g) = \
      make_periodic_general_test_system(N, spatial_dimension, dtype, box_format)
    deformed_box = box * 0.9

    R_new = s_g(R, grad(E_g)(R), new_box=deformed_box)
    R_gf_new = space.transform(deformed_box, s_gf(R_f, grad(E_gf)(R_f)))

    self.assertAllClose(R_new, R_gf_new)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_box_format={box_format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'box_format': box_format
      } for dim in SPATIAL_DIMENSION
    for dtype in POSITION_DTYPE
    for box_format in BOX_FORMATS))
  def test_periodic_general_grad_box(self, spatial_dimension, dtype, box_format):
    if box_format == 'scalar':
      raise SkipTest('Scalar case fails due to JAX Issue #5849.')
    N = 16
    R_f, R, box, (s, E), (s_gf, E_gf), (s_g, E_g) = \
      make_periodic_general_test_system(N, spatial_dimension, dtype, box_format)

    @grad
    def box_energy_g_fn(box):
      return E_g(R, new_box=box)

    @grad
    def box_energy_gf_fn(box):
      return E_gf(R_f, box=box)

    self.assertAllClose(box_energy_g_fn(box), box_energy_gf_fn(box))


if __name__ == '__main__':
  absltest.main()
