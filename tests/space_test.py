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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

from jax.config import config as jax_config
from jax import random
import jax.numpy as np

from jax.api import grad

from jax import test_util as jtu

from jax_md import space
from jax_md.util import *

from functools import partial

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 10
STOCHASTIC_SAMPLES = 10
SHIFT_STEPS = 10
SPATIAL_DIMENSION = [2, 3]

if FLAGS.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]


# pylint: disable=invalid-name
class SpaceTest(jtu.JaxTestCase):

  def test_small_inverse(self):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      mat = random.normal(split, (2, 2))

      inv_mat = space._small_inverse(mat)
      inv_mat_real_onp = onp.linalg.inv(mat)
      inv_mat_real_np = np.linalg.inv(mat)
      self.assertAllClose(inv_mat, inv_mat_real_onp, True)
      self.assertAllClose(inv_mat, inv_mat_real_np, True)

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(jtu.cases_from_list(
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

      R_prime_exact = np.array(np.dot(R, T), dtype=dtype)
      R_prime = space.transform(T, R)

      self.assertAllClose(R_prime_exact, R_prime, True)

  @parameterized.named_parameters(jtu.cases_from_list(
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

      energy_direct = lambda R: np.sum(R ** 2)
      energy_indirect = lambda T, R: np.sum(space.transform(T, R) ** 2)

      grad_direct = grad(energy_direct)(R_prime)
      grad_indirect = grad(energy_indirect, 1)(T, R)

      self.assertAllClose(grad_direct, grad_indirect, True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_transform_inverse(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)

      R = random.normal(
        split1, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      T = random.normal(
        split2, (spatial_dimension, spatial_dimension), dtype=dtype)
      T_inv = space._small_inverse(T)

      R_test = space.transform(T_inv, space.transform(T, R))

      self.assertAllClose(R, R_test, True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_canonicalize_displacement_or_metric(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    displacement, _ = space.periodic_general(np.eye(spatial_dimension))
    metric = space.metric(displacement)
    test_metric = space.canonicalize_displacement_or_metric(displacement)

    metric = space.map_product(metric)
    test_metric = space.map_product(test_metric)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2 = random.split(key, 3)

      R = random.normal(
        split1, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      self.assertAllClose(metric(R, R), test_metric(R, R), True)

  @parameterized.named_parameters(jtu.cases_from_list(
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
      dr_direct = np.reshape(dr_direct, dr_direct.shape + (1,))

      if spatial_dimension == 2:
        for i in range(-1, 2):
          for j in range(-1, 2):
            dR_shifted = dR + np.array([i, j], dtype=R.dtype)

            dr_shifted = space.distance(dR_shifted)
            dr_shifted = np.reshape(dr_shifted, dr_shifted.shape + (1,))

            dR_direct = np.where(dr_shifted < dr_direct, dR_shifted, dR_direct)
            dr_direct = np.where(dr_shifted < dr_direct, dr_shifted, dr_direct)
      elif spatial_dimension == 3:
        for i in range(-1, 2):
          for j in range(-1, 2):
            for k in range(-1, 2):
              dR_shifted = dR + np.array([i, j, k], dtype=R.dtype)

              dr_shifted = space.distance(dR_shifted)
              dr_shifted = np.reshape(dr_shifted, dr_shifted.shape + (1,))

              dR_direct = np.where(
                  dr_shifted < dr_direct, dR_shifted, dR_direct)
              dr_direct = np.where(
                  dr_shifted < dr_direct, dr_shifted, dr_direct)

      dR_direct = np.array(dR_direct, dtype=dR.dtype)
      assert dR_wrapped.dtype == dtype
      self.assertAllClose(dR_wrapped, dR_direct, True)

  @parameterized.named_parameters(jtu.cases_from_list(
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
      dR = np.sqrt(f32(0.1)) * random.normal(
          split2, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      dR = np.where(dR > 0.49, f32(0.49), dR)
      dR = np.where(dR < -0.49, f32(-0.49), dR)

      R_shift = space.periodic_shift(f32(1.0), R, dR)

      assert R_shift.dtype == R.dtype
      assert np.all(R_shift < 1.0)
      assert np.all(R_shift > 0.0)

      dR_after = space.periodic_displacement(f32(1.0), R_shift - R)

      assert dR_after.dtype == R.dtype
      self.assertAllClose(dR_after, dR, True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_against_periodic_general(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split1, split2, split3 = random.split(key, 4)

      max_box_size = f16(10.0)
      box_size = max_box_size * random.uniform(
        split1, (spatial_dimension,), dtype=dtype)
      transform = np.diag(box_size)

      R = random.uniform(
        split2, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      R_scaled = R * box_size

      dR = random.normal(
        split3, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      disp_fn, shift_fn = space.periodic(box_size)
      general_disp_fn, general_shift_fn = space.periodic_general(transform)

      disp_fn = space.map_product(disp_fn)
      general_disp_fn = space.map_product(general_disp_fn)

      self.assertAllClose(
          disp_fn(R_scaled, R_scaled), general_disp_fn(R, R), True)
      assert disp_fn(R_scaled, R_scaled).dtype == dtype
      self.assertAllClose(
          shift_fn(R_scaled, dR), general_shift_fn(R, dR) * box_size, True)
      assert shift_fn(R_scaled, dR).dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_general_time_dependence(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    eye = np.eye(spatial_dimension)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_T0_scale, split_T0_dT = random.split(key, 3)
      key, split_T1_scale, split_T1_dT = random.split(key, 3)
      key, split_t, split_R, split_dR = random.split(key, 4)

      size_0 = 10.0 * random.uniform(split_T0_scale, ())
      dtransform_0 = 0.5 * random.normal(
        split_T0_dT, (spatial_dimension, spatial_dimension))
      T_0 = np.array(size_0 * (eye + dtransform_0), dtype=dtype)

      size_1 = 10.0 * random.uniform(split_T1_scale, (), dtype=dtype)
      dtransform_1 = 0.5 * random.normal(
          split_T1_dT, (spatial_dimension, spatial_dimension), dtype=dtype)
      T_1 = np.array(size_1 * (eye + dtransform_1), dtype=dtype)

      T = lambda t: t * T_0 + (f32(1.0) - t) * T_1

      t_g = random.uniform(split_t, (), dtype=dtype)

      disp_fn, shift_fn = space.periodic_general(T)
      true_disp_fn, true_shift_fn = space.periodic_general(T(t_g))

      disp_fn = partial(disp_fn, t=t_g)

      disp_fn = space.map_product(disp_fn)
      true_disp_fn = space.map_product(true_disp_fn)

      R = random.uniform(
        split_R, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      dR = random.normal(
        split_dR, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      self.assertAllClose(
        disp_fn(R, R),
        np.array(true_disp_fn(R, R), dtype=dtype), True)
      self.assertAllClose(
        shift_fn(R, dR, t=t_g),
        np.array(true_shift_fn(R, dR), dtype=dtype), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_periodic_general_wrapped_vs_unwrapped(
      self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    eye = np.eye(spatial_dimension, dtype=dtype)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_R, split_T = random.split(key, 3)

      dT = random.normal(
        split_T, (spatial_dimension, spatial_dimension), dtype=dtype)
      T = eye + dT + np.transpose(dT)

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
          displacement(unwrapped_R, R0), True)
      assert not (np.all(unwrapped_R > 0) and np.all(unwrapped_R < 1))

if __name__ == '__main__':
  absltest.main()
