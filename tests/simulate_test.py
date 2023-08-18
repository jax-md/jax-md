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

"""Tests for google3.third_party.py.jax_md.simulate."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

from jax import jit
from jax import vmap
from jax import random
from jax import lax

from jax.config import config as jax_config
import jax.numpy as np

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import util
from jax_md import test_util
from jax_md import partition
from jax_md import smap
from jax_md import dataclasses
from jax_md.util import *

from functools import partial

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 1000
DYNAMICS_STEPS = 800
SHORT_DYNAMICS_STEPS = 20
STOCHASTIC_SAMPLES = 5
SPATIAL_DIMENSION = [2, 3]
COORDS = ['fractional', 'real']

LANGEVIN_PARTICLE_COUNT = 8000
LANGEVIN_DYNAMICS_STEPS = 8000

BROWNIAN_PARTICLE_COUNT = 8000
BROWNIAN_DYNAMICS_STEPS = 8000

DTYPE = [f32]
if FLAGS.jax_enable_x64:
  DTYPE += [f64]


ke_fn = lambda p, m: quantity.kinetic_energy(momentum=p, mass=m)
kT_fn = lambda p, m: quantity.temperature(momentum=p, mass=m)


# pylint: disable=invalid-name
class SimulateTest(test_util.JAXMDTestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(test_util.cases_from_list(
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

    state = init_fn(vel_key, R, kT=0.5, mass=mass)

    E_T = lambda state: E(state.position) + ke_fn(state.momentum, state.mass)
    E_initial = E_T(state)

    for _ in range(DYNAMICS_STEPS):
      state = apply_fn(state)
      E_total = E_T(state)
      assert np.abs(E_total - E_initial) < E_initial * 0.01
      assert state.position.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_nve_jammed(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic(state.box[0, 0])

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)

    init_fn, apply_fn = simulate.nve(E, shift_fn, 1e-3)
    apply_fn = jit(apply_fn)

    state = init_fn(key, state.real_position, kT=1e-3)

    E_T = lambda state: E(state.position) + ke_fn(state.momentum, state.mass)
    E_initial = E_T(state) * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_and_energy):
      state, energy = state_and_energy
      state = apply_fn(state)
      energy = energy.at[i].set(E_T(state))
      return state, energy

    Es = np.zeros((DYNAMICS_STEPS,))
    state, Es = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es))

    tol = 1e-3 if dtype is f32 else 1e-7
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_nve_jammed(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic(state.box[0, 0])

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)

    init_fn, apply_fn = simulate.nve(E, shift_fn, 1e-3)
    apply_fn = jit(apply_fn)

    state = init_fn(key, state.real_position, kT=1e-3)

    E_T = lambda state: E(state.position) + ke_fn(state.momentum, state.mass)
    E_initial = E_T(state) * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_and_energy):
      state, energy = state_and_energy
      state = apply_fn(state)
      energy = energy.at[i].set(E_T(state))
      return state, energy

    Es = np.zeros((DYNAMICS_STEPS,))
    state, Es = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es))

    tol = 1e-3 if dtype is f32 else 1e-7
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}_coordinates={coords}',
          'dtype': dtype,
          'coords': coords
      } for dtype in DTYPE for coords in COORDS))
  def test_nve_jammed_periodic_general(self, dtype, coords):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic_general(state.box,
                                                       coords == 'fractional')

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)

    init_fn, apply_fn = simulate.nve(E, shift_fn, 1e-3)
    apply_fn = jit(apply_fn)

    state = init_fn(key, getattr(state, coords + '_position'), kT=1e-3)

    E_T = lambda state: E(state.position) + ke_fn(state.momentum, state.mass)
    E_initial = E_T(state) * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_and_energy):
      state, energy = state_and_energy
      state = apply_fn(state)
      energy = energy.at[i].set(E_T(state))
      return state, energy

    Es = np.zeros((DYNAMICS_STEPS,))
    state, Es = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es))

    tol = 1e-3 if dtype is f32 else 1e-7
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_nve_neighbor_list(self, spatial_dimension, dtype):
    Nx = particles_per_side = 8
    spacing = f32(1.25)

    tol = 5e-12 if dtype == np.float64 else 5e-3

    L = Nx * spacing
    if spatial_dimension == 2:
      R = np.stack([np.array(r) for r in onp.ndindex(Nx, Nx)]) * spacing
    elif spatial_dimension == 3:
      R = np.stack([np.array(r) for r in onp.ndindex(Nx, Nx, Nx)]) * spacing

    R = np.array(R, dtype)

    displacement, shift = space.periodic(L)

    neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(displacement, L)
    exact_energy_fn = energy.lennard_jones_pair(displacement)

    init_fn, apply_fn = simulate.nve(energy_fn, shift, 1e-3)
    exact_init_fn, exact_apply_fn = simulate.nve(exact_energy_fn, shift, 1e-3)

    nbrs = neighbor_fn(R)
    state = init_fn(random.PRNGKey(0), R, kT=0.5, neighbor=nbrs)
    exact_state = exact_init_fn(random.PRNGKey(0), R, kT=0.5)

    def body_fn(i, state):
      state, nbrs, exact_state = state
      nbrs = neighbor_fn(state.position, nbrs)
      state = apply_fn(state, neighbor=nbrs)
      return state, nbrs, exact_apply_fn(exact_state)

    step = 0
    for i in range(20):
      new_state, nbrs, new_exact_state = lax.fori_loop(
        0, 100, body_fn, (state, nbrs, exact_state))
      if nbrs.did_buffer_overflow:
        nbrs = neighbor_fn(state.position)
      else:
        state = new_state
        exact_state = new_exact_state
        step += 1
    assert state.position.dtype == dtype
    self.assertAllClose(state.position, exact_state.position, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_sy_steps={sy_steps}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'sy_steps': sy_steps,
      } for dim in SPATIAL_DIMENSION
        for dtype in DTYPE
        for sy_steps in [1, 3, 5, 7]))
  def test_nvt_nose_hoover(self, spatial_dimension, dtype, sy_steps):
    key = random.PRNGKey(0)

    box_size = quantity.box_size_at_number_density(PARTICLE_COUNT,
                                                   f32(1.2),
                                                   spatial_dimension)
    displacement_fn, shift_fn = space.periodic(box_size)

    bonds_i = np.arange(PARTICLE_COUNT)
    bonds_j = np.roll(bonds_i, 1)
    bonds = np.stack([bonds_i, bonds_j])

    E = energy.simple_spring_bond(displacement_fn, bonds)

    invariant = partial(simulate.nvt_nose_hoover_invariant, E)

    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, vel_key, T_key, masses_key = random.split(key, 5)

      R = box_size * random.uniform(
        pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=dtype)
      mass = 1 + random.uniform(masses_key, (PARTICLE_COUNT,), dtype=dtype)
      init_fn, apply_fn = simulate.nvt_nose_hoover(E,
                                                   shift_fn,
                                                   1e-3,
                                                   T,
                                                   sy_steps=sy_steps)
      apply_fn = jit(apply_fn)

      state = init_fn(vel_key, R, mass=mass)

      initial = invariant(state, T)

      for _ in range(DYNAMICS_STEPS):
        state = apply_fn(state)

      T_final = kT_fn(state.momentum, state.mass)
      assert np.abs(T_final - T) / T < 0.1
      tol = 5e-4 if dtype is f32 else 1e-6
      self.assertAllClose(invariant(state, T), initial, rtol=tol)
      self.assertEqual(state.position.dtype, dtype)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dtype={dtype.__name__}_sy_steps={sy_steps}',
          'dtype': dtype,
          'sy_steps': sy_steps,
      } for dtype in DTYPE
        for sy_steps in [1, 3, 5, 7]))
  def test_nvt_nose_hoover_jammed(self, dtype, sy_steps):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic(state.box[0, 0])

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)
    invariant = partial(simulate.nvt_nose_hoover_invariant, E)

    kT = 1e-3
    init_fn, apply_fn = simulate.nvt_nose_hoover(E, shift_fn, 1e-3,
                                                 kT=kT, sy_steps=sy_steps)
    apply_fn = jit(apply_fn)

    state = init_fn(key, state.real_position)

    E_initial = invariant(state, kT) * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_and_energy):
      state, energy = state_and_energy
      state = apply_fn(state)
      energy = energy.at[i].set(invariant(state, kT))
      return state, energy

    Es = np.zeros((DYNAMICS_STEPS,))
    state, Es = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es))

    tol = 1e-3 if dtype is f32 else 1e-7
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dtype={dtype.__name__}_sy_steps={sy_steps}',
          'dtype': dtype,
          'sy_steps': sy_steps,
      } for dtype in DTYPE
        for sy_steps in [1, 3, 5, 7]))
  def test_npt_nose_hoover_jammed(self, dtype, sy_steps):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic_general(state.box)

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)
    invariant = partial(simulate.npt_nose_hoover_invariant, E)
    pressure_fn = partial(quantity.pressure, E)

    nhc_kwargs = {sy_steps: sy_steps}
    kT = 1e-3
    P = state.pressure
    init_fn, apply_fn = simulate.npt_nose_hoover(E, shift_fn, 1e-3, P, kT,
                                                 nhc_kwargs, nhc_kwargs)
    apply_fn = jit(apply_fn)

    state = init_fn(key, state.fractional_position, state.box)

    E_initial = invariant(state, P, kT) * np.ones((DYNAMICS_STEPS,))
    P_target = P * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_energy_pressure):
      state, energy, pressure = state_energy_pressure
      state = apply_fn(state)
      energy = energy.at[i].set(invariant(state, P, kT))
      box = simulate.npt_box(state)
      KE = ke_fn(state.momentum, state.mass)
      p = pressure_fn(state.position, box, KE)
      pressure = pressure.at[i].set(p)
      return state, energy, pressure

    Es = np.zeros((DYNAMICS_STEPS,))
    Ps = np.zeros((DYNAMICS_STEPS,))
    state, Es, Ps = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es, Ps))

    tol = 1e-3 if dtype is f32 else 1.2e-4
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)
    self.assertAllClose(Ps, P_target, rtol=0.05, atol=0.05)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dtype={dtype.__name__}',
          'dtype': dtype,
      } for dtype in DTYPE))
  def test_npt_nose_hoover_lammps(self, dtype):
    key = random.PRNGKey(0)

    box, pos, vel = test_util.load_lammps_npt_test_case(dtype)

    displacement, shift = space.periodic_general(box)
    dist_fun = space.metric(displacement)
    neighbor_fn, energy_fn = energy.stillinger_weber_neighbor_list(
      displacement, box)

    units = {
      'mass': 1,
      'distance': 1,
      'time': 98.22694788,
      'energy': 1,
      'velocity': 0.01018051,
      'force': 1.0,
      'torque ': 1,
      'temperature': 8.617330337217213e-05,
      'pressure': 6.241509125883258e-07
    }

    fs = 1e-3 * units['time']
    ps = units['time']

    dt = fs
    write_every = 100
    T_init = 300 * units['temperature']
    P_init = 0.0 * units['pressure']
    Mass = 28.0855 * units['mass']
    key = random.PRNGKey(121)
    key, split = random.split(key)

    nbrs = neighbor_fn.allocate(pos, box=box, extra_capacity=8)
    init_fn, apply_fn = simulate.npt_nose_hoover(
      energy_fn, shift, dt=dt, pressure=P_init, kT=T_init)
    state = init_fn(key, pos, box=box, neighbor=nbrs)

    def step_fn(i, state_nbrs_buffers):
      state, nbrs, buffers = state_nbrs_buffers
      state = apply_fn(state, neighbor=nbrs)
      nbrs = nbrs.update(state.position)
      buffers['kT'] = buffers['kT'].at[i].set(quantity.temperature(
        momentum=state.momentum, mass=state.mass))
      KE = quantity.kinetic_energy(momentum=state.momentum, mass=Mass)
      buffers['P'] = buffers['P'].at[i].set(quantity.pressure(
        energy_fn, state.position, box=box, kinetic_energy=KE, neighbor=nbrs,
      ))
      buffers['H'] = buffers['H'].at[i].set(simulate.npt_nose_hoover_invariant(
        energy_fn, state, pressure=P_init, kT=T_init, neighbor=nbrs
      ))
      return state, nbrs, buffers

    buffers = {
      'kT': np.zeros((DYNAMICS_STEPS,)),
      'P': np.zeros((DYNAMICS_STEPS,)),
      'H': np.zeros((DYNAMICS_STEPS,))
    }

    state, nbrs, buffers = lax.fori_loop(0, DYNAMICS_STEPS, step_fn,
                                         (state, nbrs, buffers))

    kT_tol = 1e-2
    P_tol = 2e-3 if dtype == np.float64 else 2e-3

    self.assertAllClose(np.mean(buffers['kT'][-DYNAMICS_STEPS//2:]), T_init,
                        atol=kT_tol, rtol=kT_tol)
    self.assertAllClose(np.mean(buffers['P'][-DYNAMICS_STEPS//2:]), P_init,
                        atol=P_tol, rtol=P_tol)
    self.assertAllClose(buffers['H'],
                        np.ones((DYNAMICS_STEPS,), dtype=dtype) * buffers['H'])

  @parameterized.named_parameters(test_util.cases_from_list(
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
        masses_key, (LANGEVIN_PARTICLE_COUNT,),
        minval=0.1, maxval=10.0, dtype=dtype)
      init_fn, apply_fn = simulate.nvt_langevin(
        E, shift, f32(1e-2), T, gamma=f32(0.3))
      apply_fn = jit(apply_fn)

      state = init_fn(key, R, mass=mass, T_initial=dtype(1.0))

      T_list = []
      for step in range(LANGEVIN_DYNAMICS_STEPS):
        state = apply_fn(state)
        if step > 4000 and step % 100 == 0:
          T_list += [kT_fn(state.momentum, state.mass)]

      # TODO(schsam): It would be good to check Gaussinity of R and V in the
        # noninteracting case.
      T_emp = np.mean(np.array(T_list))
      assert np.abs(T_emp - T) < 0.1
      assert state.position.dtype == dtype

  def test_langevin_harmonic(self):
    alpha = 1.0
    E = lambda x: jnp.sum(0.5 * alpha * x ** 2)
    displacement, shift = space.free()

    N = 100_000
    steps = 1000
    kT = 0.25
    dt = 1e-4
    gamma = 3
    mass = 3.0
    tol = 1e-3

    X = jnp.ones((N, 1, 1))
    key = random.split(random.PRNGKey(0), N)

    init_fn, step_fn = simulate.nvt_langevin(E, shift, dt, kT, gamma, False)
    step_fn = jit(vmap(step_fn))

    state = vmap(init_fn, (0, 0, None))(key, X, mass)
    p0 = state.momentum

    for i in range(steps):
      state = step_fn(state)

    # Compare mean position and momentum autocorrelation with theoretical
    # prediction.

    d = jnp.sqrt(gamma ** 2 / 4 - alpha / mass)

    beta_1 = gamma / 2 + d
    beta_2 = gamma / 2 - d
    A = -beta_2 / (beta_1 - beta_2)
    B = beta_1 / (beta_1 - beta_2)
    exp1 = lambda t: jnp.exp(-beta_1 * t)
    exp2 = lambda t: jnp.exp(-beta_2 * t)
    Z = kT / (2 * d * mass)

    pos_fn = lambda t: A * exp1(t) + B * exp2(t)
    mom_fn = lambda t: Z * (-beta_2 * exp2(t) + beta_1 * exp1(t)) * mass**2

    t = steps * dt
    self.assertAllClose(jnp.mean(state.position),
                        pos_fn(t),
                        rtol=tol,
                        atol=tol)
    self.assertAllClose(jnp.mean(state.momentum * p0),
                        mom_fn(t),
                        rtol=tol,
                        atol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}',
          'dtype': dtype,
      } for dtype in DTYPE))
  def test_swap_mc_jammed(self, dtype):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    space_fn = space.periodic(state.box[0, 0])
    displacement_fn, shift_fn = space_fn

    sigma = np.diag(state.sigma)[state.species]

    energy_fn = lambda dr, sigma: energy.soft_sphere(dr, sigma=sigma)
    neighbor_fn = partition.neighbor_list(displacement_fn,
                                          state.box[0, 0],
                                          np.max(sigma) + 0.1,
                                          dr_threshold=0.5)

    kT = 1e-2
    t_md = 0.1
    N_swap = 10
    init_fn, apply_fn = simulate.hybrid_swap_mc(space_fn,
                                                energy_fn,
                                                neighbor_fn,
                                                1e-3,
                                                kT,
                                                t_md,
                                                N_swap)
    state = init_fn(key, state.real_position, sigma)

    Ts = np.zeros((DYNAMICS_STEPS,))

    def step_fn(i, state_and_temp):
      state, temp = state_and_temp
      state = apply_fn(state)
      temp = temp.at[i].set(kT_fn(state.md.momentum, 1.0))
      return state, temp

    state, Ts = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Ts))

    tol = 5e-4
    self.assertAllClose(Ts[10:],
                        kT * np.ones((DYNAMICS_STEPS - 10)),
                        rtol=5e-1,
                        atol=5e-3)
    self.assertAllClose(np.mean(Ts[10:]), kT, rtol=tol, atol=tol)
    self.assertTrue(not np.all(state.sigma == sigma))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}',
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION
        for dtype in DTYPE))
  def test_nvk_ensemble(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    box_size = quantity.box_size_at_number_density(PARTICLE_COUNT,
                                                   f32(1.2),
                                                   spatial_dimension)
    displacement_fn, shift_fn = space.periodic(box_size)

    bonds_i = np.arange(PARTICLE_COUNT)
    bonds_j = np.roll(bonds_i, 1)
    bonds = np.stack([bonds_i, bonds_j])

    E = energy.simple_spring_bond(displacement_fn, bonds)

    invariant = ke_fn

    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, vel_key, T_key, masses_key = random.split(key, 5)

      R = box_size * random.uniform(
        pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
      T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=dtype)
      mass = 1 + random.uniform(masses_key, (PARTICLE_COUNT,), dtype=dtype)
      init_fn, apply_fn = simulate.nvk(E,
                                       shift_fn,
                                       1e-3,
                                       T)
      apply_fn = jit(apply_fn)

      state = init_fn(vel_key, R, mass=mass)

      initial = invariant(state.momentum, state.mass)

      for _ in range(DYNAMICS_STEPS):
        state = apply_fn(state)

      tol = 5e-4 if dtype is f32 else 1e-6
      self.assertAllClose(invariant(state.momentum, state.mass), initial, rtol=tol)
      self.assertEqual(state.position.dtype, dtype)
  
  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_temp_csvr(self, spatial_dimension, dtype):
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
        masses_key, (LANGEVIN_PARTICLE_COUNT,),
        minval=0.1, maxval=10.0, dtype=dtype)
      init_fn, apply_fn = simulate.temp_csvr(
        E, shift, f32(1e-2), T, tau= 100 * f32(1e-2))
      apply_fn = jit(apply_fn)

      state = init_fn(key, R, mass=mass, T_initial=dtype(1.0))

      T_list = []
      for step in range(LANGEVIN_DYNAMICS_STEPS):
        state = apply_fn(state)
        if step > 4000 and step % 100 == 0:
          T_list += [kT_fn(state.momentum, state.mass)]

      T_emp = np.mean(np.array(T_list))
      assert np.abs(T_emp - T) < 0.1
      assert state.position.dtype == dtype

if __name__ == '__main__':
  absltest.main()
