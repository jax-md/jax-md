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

"""Tests for google3.third_party.py.jax_md.dynamics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

from jax.config import config as jax_config
from jax import random
from jax import jit
import jax.numpy as np

from jax import test_util as jtu

from jax_md import space
from jax_md import minimize
from jax_md import quantity
from jax_md.util import *

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 10
OPTIMIZATION_STEPS = 10
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

if FLAGS.jax_enable_x64:
  DTYPE = [f32, f64]
else:
  DTYPE = [f32]

class DynamicsTest(jtu.JaxTestCase):
  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_gradient_descent(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split, split0 = random.split(key, 3)
      R = random.uniform(split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      R0 = random.uniform(split0, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      energy = lambda R, **kwargs: np.sum((R - R0) ** 2)
      _, shift_fn = space.free()

      opt_init, opt_apply = minimize.gradient_descent(energy, shift_fn, f32(1e-1))

      E_current = energy(R)
      dr_current = np.sum((R - R0) ** 2)

      for _ in range(OPTIMIZATION_STEPS):
        R = opt_apply(R)
        E_new = energy(R)
        dr_new = np.sum((R - R0) ** 2)
        assert E_new < E_current
        assert E_new.dtype == dtype
        assert dr_new < dr_current
        assert dr_new.dtype == dtype
        E_current = E_new
        dr_current = dr_new

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_fire_descent(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split, split0 = random.split(key, 3)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      R0 = random.uniform(
        split0, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)

      energy = lambda R, **kwargs: np.sum((R - R0) ** 2)
      _, shift_fn = space.free()

      opt_init, opt_apply = minimize.fire_descent(energy, shift_fn)

      opt_state = opt_init(R)
      E_current = energy(R)
      dr_current = np.sum((R - R0) ** 2)

      # NOTE(schsam): We add this to test to make sure we can jit through the
      # creation of FireDescentState namedtuple.
      @jit
      def three_steps(state):
        return opt_apply(opt_apply(opt_apply(state)))

      for _ in range(OPTIMIZATION_STEPS):
        opt_state = three_steps(opt_state)
        R = opt_state.position
        E_new = energy(R)
        dr_new = np.sum((R - R0) ** 2)
        assert E_new < E_current
        assert E_new.dtype == dtype
        assert dr_new < dr_current
        assert dr_new.dtype == dtype
        E_current = E_new
        dr_current = dr_new

if __name__ == '__main__':
  absltest.main()
