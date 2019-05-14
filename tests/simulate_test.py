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

from jax.config import config as jax_config
import jax.numpy as np

from jax_md import quantity
from jax_md import simulate
from jax_md import space

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 1000
DYNAMICS_STEPS = 800
STOCHASTIC_SAMPLES = 5
SPATIAL_DIMENSION = [2, 3]


# pylint: disable=invalid-name
class SimulateTest(jtu.JaxTestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}'.format(dim),
          'spatial_dimension': dim
      } for dim in SPATIAL_DIMENSION))
  def test_nve_ensemble(self, spatial_dimension):
    key = random.PRNGKey(0)
    pos_key, center_key, vel_key, mass_key = random.split(key, 4)
    R = random.normal(pos_key, (PARTICLE_COUNT, spatial_dimension))
    R0 = random.normal(center_key, (PARTICLE_COUNT, spatial_dimension))
    mass = random.uniform(mass_key, (PARTICLE_COUNT,), minval=0.1, maxval=5.0)
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

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}'.format(dim),
          'spatial_dimension': dim
      } for dim in SPATIAL_DIMENSION))
  def test_nvt_nose_hoover_ensemble(self, spatial_dimension):
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

      R = random.normal(pos_key, (PARTICLE_COUNT, spatial_dimension))
      R0 = random.normal(center_key, (PARTICLE_COUNT, spatial_dimension))
      _, shift = space.free()

      E = functools.partial(
          lambda R, R0, **kwargs: np.sum((R - R0) ** 2), R0=R0)

      T = random.uniform(T_key, (), minval=0.3, maxval=1.4)
      mass = random.uniform(
          masses_key, (PARTICLE_COUNT,), minval=0.1, maxval=10.0)
      init_fn, apply_fn = simulate.nvt_nose_hoover(E, shift, 1e-3, T)
      apply_fn = jit(apply_fn)

      state = init_fn(vel_key, R, mass=mass, T_initial=1.0)

      initial = invariant(T, state)

      for _ in range(DYNAMICS_STEPS):
        state = apply_fn(state)

      assert np.abs(
          quantity.temperature(state.velocity, state.mass) - T) < 0.1
      assert np.abs(invariant(T, state) - initial) < initial * 0.01


if __name__ == '__main__':
  absltest.main()




