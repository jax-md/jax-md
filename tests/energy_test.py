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

from jax import test_util as jtu

from jax_md import energy

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS

PARTICLE_COUNT = 10
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

SOFT_SPHERE_ALPHA = [2.0, 3.0]


class EnergyTest(jtu.JaxTestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_alpha={}'.format(dim, alpha),
          'spatial_dimension': dim,
          'alpha': alpha,
      } for dim in SPATIAL_DIMENSION for alpha in SOFT_SPHERE_ALPHA))
  def test_soft_sphere(self, spatial_dimension, alpha):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_dR, split_sigma, split_epsilon = random.split(key, 4)
      dR = random.normal(
          split_dR, (spatial_dimension,), dtype=np.float64)
      sigma = random.uniform(
          split_sigma, (1,), dtype=np.float64, minval=0.0, maxval=3.0)[0]
      dR = dR * sigma / np.sqrt(np.sum(dR ** 2, axis=1, keepdims=True))
      epsilon = random.uniform(
          split_epsilon, (1,), dtype=np.float64, minval=0.0, maxval=4.0)[0]
      self.assertAllClose(
          energy.soft_sphere(
              0.0 * dR, sigma, epsilon, alpha), epsilon / alpha, True)
      self.assertAllClose(
          energy.soft_sphere(dR, sigma, epsilon, alpha), 0.0, True)

      if alpha == 3.0:
        grad_energy = grad(energy.soft_sphere)
        g = grad_energy(dR, sigma, epsilon, alpha)
        self.assertAllClose(g, np.zeros_like(g), True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}'.format(dim),
          'spatial_dimension': dim,
      } for dim in SPATIAL_DIMENSION))
  def test_lennard_jones(self, spatial_dimension):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_dR, split_sigma, split_epsilon = random.split(key, 4)
      dR = random.normal(
          split_dR, (spatial_dimension,), dtype=np.float64)
      sigma = random.uniform(
          split_sigma, (1,), dtype=np.float64, minval=0.5, maxval=3.0)[0]
      dR = dR * sigma / np.sqrt(np.sum(dR ** 2, keepdims=True))
      epsilon = random.uniform(
          split_epsilon, (1,), dtype=np.float64, minval=0.0, maxval=4.0)[0]
      self.assertAllClose(
          energy.lennard_jones(dR, sigma, epsilon), -epsilon, True)
      g = grad(energy.lennard_jones)(dR, sigma, epsilon)
      self.assertAllClose(g, np.zeros_like(g), True)


if __name__ == '__main__':
  absltest.main()
