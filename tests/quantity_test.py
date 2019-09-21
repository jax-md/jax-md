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

from jax.api import jit, grad
from jax_md import quantity
from jax_md.util import *

from jax import test_util as jtu

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 10
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]
DTYPES = [f32, f64]


class QuantityTest(jtu.JaxTestCase):

  def test_canonicalize_mass(self):
    assert quantity.canonicalize_mass(3.0) == 3.0
    assert quantity.canonicalize_mass(f16(3.0)) == f16(3.0)
    assert quantity.canonicalize_mass(f32(3.0)) == f32(3.0)
    assert quantity.canonicalize_mass(f64(3.0)) == f64(3.0)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}'.format(dim),
          'spatial_dimension': dim,
      } for dim in SPATIAL_DIMENSION))
  def test_grad_kinetic_energy(self, spatial_dimension):
    key = random.PRNGKey(0)

    @jit
    def do_fn(theta):
      key = random.PRNGKey(0)
      V = random.normal(key, (PARTICLE_COUNT, spatial_dimension), dtype=f32)

      return quantity.kinetic_energy(theta * V)

    grad(do_fn)(2.0)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype,
      } for dtype in DTYPES))
  def test_pair_correlation(self, dtype):
    displacement = lambda Ra, Rb, **kwargs: Ra - Rb
    R = np.array(
        [[1, 0],
         [0, 0],
         [0, 1]], dtype=dtype)
    rs = np.linspace(0, 2, 60)
    g = quantity.pair_correlation(displacement, rs, f32(0.1))
    gs = g(R)
    gs = np.mean(gs, axis=0)
    assert np.argmax(gs) == np.argmin((rs - 1.) ** 2)
    assert gs.dtype == dtype

if __name__ == '__main__':
  absltest.main()
