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

"""Tests for google3.third_party.py.jax_md.ensemble."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import absltest
from absl.testing import parameterized

from jax import jit
from jax import random
from jax import test_util as jtu
from jax import lax

from jax.config import config as jax_config
import jax.numpy as np

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md.util import *

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 1000
DYNAMICS_STEPS = 800
SHORT_DYNAMICS_STEPS = 20
STOCHASTIC_SAMPLES = 5
SPATIAL_DIMENSION = [2, 3]

LANGEVIN_PARTICLE_COUNT = 8000
LANGEVIN_DYNAMICS_STEPS = 8000

BROWNIAN_PARTICLE_COUNT = 8000
BROWNIAN_DYNAMICS_STEPS = 8000

DTYPE = [f32]
if FLAGS.jax_enable_x64:
  DTYPE += [f64]


# pylint: disable=invalid-name
class SimulateTest(jtu.JaxTestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_nve_ensemble(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    pos_key, center_key, vel_key, mass_key = random.split(key, 4)
    R = random.normal(
      pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    R0 = random.normal(
      center_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    mass = random.uniform(
      mass_key, (PARTICLE_COUNT,), minval=0.1, maxval=5.0, dtype=dtype)
    _, shift = space.free()

    E = lambda R, **kwargs: np.sum((R - R0) ** 2)

    init_fn, apply_fn = simulate.nve(E, shift, 1e-3)
    apply_fn = jit(apply_fn)

    state = init_fn(vel_key, R, mass=mass)

    E_T = lambda state: \
        E(state.position) + quantity.kinetic_energy(state.velocity, state.mass)
    E_initial = E_T(state)

    for _ in range(DYNAMICS_STEPS):
      state = apply_fn(state)
      E_total = E_T(state)
      assert np.abs(E_total - E_initial) < E_initial * 0.01
      assert state.position.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_nve_ensemble_time_dependence(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    pos_key, center_key, vel_key, mass_key = random.split(key, 4)
    R = random.normal(
      pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    R0 = random.normal(
      center_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    mass = random.uniform(
      mass_key, (PARTICLE_COUNT,), minval=0.1, maxval=5.0, dtype=dtype)
    displacement, shift = space.free()

    E = energy.soft_sphere_pair(displacement)

    init_fn, apply_fn = simulate.nve(E, shift, 1e-3)
    apply_fn = jit(apply_fn)

    state = init_fn(vel_key, R, mass=mass)

    E_T = lambda state: \
        E(state.position) + quantity.kinetic_energy(state.velocity, state.mass)
    E_initial = E_T(state)

    for t in range(SHORT_DYNAMICS_STEPS):
      state = apply_fn(state, t=t*1e-3)
      E_total = E_T(state)
      assert np.abs(E_total - E_initial) < E_initial * 0.01
      assert state.position.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_nvt_nose_hoover_ensemble(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    def invariant(T, state):
      """The conserved quantity for Nose-Hoover thermostat."""
      accum = \
          E(state.position) + quantity.kinetic_energy(state.velocity, state.mass)
      DOF = spatial_dimension * PARTICLE_COUNT
      accum = accum + (state.v_xi[0]) ** 2 * state.Q[0] * 0.5 + \
          DOF * T * state.xi[0]
      for xi, v_xi, Q in zip(
          state.xi[1:], state.v_xi[1:], state.Q[1:]):
        accum = accum + v_xi ** 2 * Q * 0.5 + T * xi
      return accum

    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, center_key, vel_key, T_key, masses_key = \
          random.split(key, 6)

      R = random.normal(
        pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      R0 = random.normal(
        center_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      _, shift = space.free()

      E = functools.partial(
          lambda R, R0, **kwargs: np.sum((R - R0) ** 2), R0=R0)

      T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=dtype)
      mass = random.uniform(
          masses_key, (PARTICLE_COUNT,), minval=0.1, maxval=10.0, dtype=dtype)
      init_fn, apply_fn = simulate.nvt_nose_hoover(E, shift, 1e-3, T)
      apply_fn = jit(apply_fn)

      state = init_fn(vel_key, R, mass=mass, T_initial=dtype(1.0))

      initial = invariant(T, state)

      for _ in range(DYNAMICS_STEPS):
        state = apply_fn(state)

      assert np.abs(
          quantity.temperature(state.velocity, state.mass) - T) < 0.1
      assert np.abs(invariant(T, state) - initial) < initial * 0.01
      assert state.position.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_nvt_langevin(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, R_key, R0_key, T_key, masses_key = random.split(key, 5)

      R = random.normal(
        R_key, (LANGEVIN_PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      R0 = random.normal(
        R0_key, (LANGEVIN_PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      _, shift = space.free()

      E = functools.partial(
          lambda R, R0, **kwargs: np.sum((R - R0) ** 2), R0=R0)

      T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=dtype)
      mass = random.uniform(
        masses_key, (LANGEVIN_PARTICLE_COUNT,), minval=0.1, maxval=10.0, dtype=dtype)
      init_fn, apply_fn = simulate.nvt_langevin(E, shift, f32(1e-2), T, gamma=f32(0.3))
      apply_fn = jit(apply_fn)

      state = init_fn(key, R, mass=mass, T_initial=dtype(1.0))

      T_list = []
      for step in range(LANGEVIN_DYNAMICS_STEPS):
        state = apply_fn(state)
        if step > 4000 and step % 100 == 0:
          T_list += [quantity.temperature(state.velocity, state.mass)]

      # TODO(schsam): It would be good to check Gaussinity of R and V in the
      # noninteracting case.
      T_emp = np.mean(np.array(T_list))
      assert np.abs(T_emp - T) < 0.1
      assert state.position.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_brownian(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    key, T_split, mass_split = random.split(key, 3)

    _, shift = space.free()
    energy_fn = lambda R, **kwargs: f32(0)

    R = np.zeros((BROWNIAN_PARTICLE_COUNT, 2), dtype=dtype)
    mass = random.uniform(
      mass_split, (), minval=0.1, maxval=10.0, dtype=dtype)
    T = random.uniform(T_split, (), minval=0.3, maxval=1.4, dtype=dtype)

    dt = f32(1e-2)
    gamma = f32(0.1)

    init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, T, gamma=gamma)
    apply_fn = jit(apply_fn)

    state = init_fn(key, R, mass)

    sim_t = f32(BROWNIAN_DYNAMICS_STEPS * dt)
    for _ in range(BROWNIAN_DYNAMICS_STEPS):
      state = apply_fn(state)

    msd = np.var(state.position)
    th_msd = dtype(2 * T / (mass * gamma) * sim_t)
    assert np.abs(msd - th_msd) / msd < 1e-2
    assert state.position.dtype == dtype

if __name__ == '__main__':
  absltest.main()
