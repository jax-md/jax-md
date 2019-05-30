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

"""Tests for google3.third_party.py.jax_md.energy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

from jax.config import config as jax_config
from jax import random
import jax.numpy as np

from jax.api import grad
from jax_md import space
from jax_md.util import *

from jax import test_util as jtu

from jax_md import energy

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS

PARTICLE_COUNT = 10
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

SOFT_SPHERE_ALPHA = [2.0, 3.0]

if FLAGS.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]

class EnergyTest(jtu.JaxTestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_alpha={}_dtype={}'.format(
            dim, alpha, dtype.__name__),
          'spatial_dimension': dim,
          'alpha': alpha,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION
    for alpha in SOFT_SPHERE_ALPHA
    for dtype in POSITION_DTYPE))
  def test_soft_sphere(self, spatial_dimension, alpha, dtype):
    key = random.PRNGKey(0)
    alpha = f32(alpha)
    for _ in range(STOCHASTIC_SAMPLES):
      key, split_dR, split_sigma, split_epsilon = random.split(key, 4)
      dR = random.normal(
          split_dR, (spatial_dimension,), dtype=dtype)
      sigma = np.array(random.uniform(
          split_sigma, (1,), minval=0.0, maxval=3.0)[0], dtype=dtype)
      dR = dR * sigma / np.sqrt(np.sum(dR ** 2, axis=1, keepdims=True))
      epsilon = np.array(
        random.uniform(split_epsilon, (1,), minval=0.0, maxval=4.0)[0],
        dtype=dtype)
      self.assertAllClose(
          energy.soft_sphere(
              f32(0.0) * dR, sigma, epsilon, alpha), epsilon / alpha, True)
      self.assertAllClose(
        energy.soft_sphere(dR, sigma, epsilon, alpha),
        np.array(0.0, dtype=dtype), True)

      if alpha == 3.0:
        grad_energy = grad(energy.soft_sphere)
        g = grad_energy(dR, sigma, epsilon, alpha)
        self.assertAllClose(g, np.zeros(g.shape, dtype=dtype), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_lennard_jones(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_dR, split_sigma, split_epsilon = random.split(key, 4)
      dR = random.normal(
          split_dR, (spatial_dimension,), dtype=dtype)
      sigma = f32(random.uniform(
          split_sigma, (1,), minval=0.5, maxval=3.0)[0])
      dR = dR * sigma / np.sqrt(np.sum(dR ** 2, keepdims=True))
      dR = dR * f32(2 ** (1.0 / 6.0))
      epsilon = f32(random.uniform(
          split_epsilon, (1,), minval=0.0, maxval=4.0)[0])
      self.assertAllClose(
        energy.lennard_jones(dR, sigma, epsilon),
        np.array(-epsilon, dtype=dtype), True)
      g = grad(energy.lennard_jones)(dR, sigma, epsilon)
      self.assertAllClose(g, np.zeros(g.shape, dtype=dtype), True)


if __name__ == '__main__':
  absltest.main()
