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

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

from jax.config import config as jax_config
from jax import random
from jax import jit
from jax import lax
import jax.numpy as np

from jax_md import space
from jax_md import minimize
from jax_md import quantity
from jax_md.util import *
from jax_md import test_util
from jax_md.energy import simple_spring_bond

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

class DynamicsTest(test_util.JAXMDTestCase):
  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_gradient_descent(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split, split0 = random.split(key, 3)
      R = random.uniform(split,
                         (PARTICLE_COUNT, spatial_dimension),
                         dtype=dtype)
      R0 = random.uniform(split0,
                          (PARTICLE_COUNT, spatial_dimension),
                          dtype=dtype)

      energy = lambda R, **kwargs: np.sum((R - R0) ** 2)
      _, shift_fn = space.free()

      opt_init, opt_apply = minimize.gradient_descent(energy,
                                                      shift_fn,
                                                      f32(1e-1))

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

  @parameterized.named_parameters(test_util.cases_from_list(
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
      # creation of FireDescentState.
      step_fn = lambda i, state: opt_apply(state)

      @jit
      def three_steps(state):
        return lax.fori_loop(0, 3, step_fn, state)

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

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_fire_descent_with_unmoving_masses(self, spatial_dimension, dtype):
      '''
      Consider three particles with masses [inf, inf, 1.0] and two springs connecting them with rest length 1.0.
      (inf) -- (inf) -- (1.0)
      The initial positions are [0.0, 0.5, 1.0]. After optimization, the positions should be [0.0, 0.5, 1.5].
      In other dimensions, the coordinates are set to 0 and should remain so.
      '''
      N = 3
      bond = jnp.array([[0, 1], [1, 2]])
      R_init = jnp.zeros(shape=(N, spatial_dimension), dtype=dtype)
      R_init = R_init.at[1, 0].set(0.5)
      R_init = R_init.at[2, 0].set(1.0)
      mass = jnp.array([jnp.inf, jnp.inf, 1.0], dtype=dtype)
      iterations = 10000

      displacement_fn, shift_fn = space.free()
      energy_fn = simple_spring_bond(displacement_fn, bond=bond, length=1.0)

      init_fn, apply_fn = minimize.fire_descent(energy_fn, shift_fn=shift_fn, dt_start=0.001, dt_max=0.1)

      state = init_fn(R_init, mass=mass)
      body_fun = lambda i, state: apply_fn(state)
      state = lax.fori_loop(0, iterations, body_fun, state)
      R = state.position

      # Check x-coordinates: should be (0, 0.5, 1.5)
      self.assertAllClose(R[:, 0], np.array([0.0, 0.5, 1.5], dtype=dtype))

      # Check other coordinates: should be all zeros
      self.assertAllClose(R[:, 1:], np.zeros(shape=(N, spatial_dimension-1), dtype=dtype))


if __name__ == '__main__':
  absltest.main()
