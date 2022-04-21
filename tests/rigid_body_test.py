# Copyright 2022 Google LLC
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

import functools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

from jax import jit
from jax import vmap
from jax import random
from jax import test_util as jtu
from jax import lax

from jax.config import config as jax_config
import jax.numpy as jnp

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import test_util
from jax_md import partition
from jax_md import util
from jax_md import rigid_body

from functools import partial

jax_config.parse_flags_with_absl()

FLAGS = jax_config.FLAGS


f32 = util.f32
f64 = util.f64

PARTICLE_COUNT = 40
DYNAMICS_STEPS = 100
SHORT_DYNAMICS_STEPS = 20
STOCHASTIC_SAMPLES = 5
COORDS = ['fractional', 'real']

LANGEVIN_PARTICLE_COUNT = 8000
LANGEVIN_DYNAMICS_STEPS = 8000

BROWNIAN_PARTICLE_COUNT = 8000
BROWNIAN_DYNAMICS_STEPS = 8000

DTYPE = [f32]
if FLAGS.jax_enable_x64:
  DTYPE += [f64]


@partial(vmap, in_axes=(0, None))
def rand_quat(key, dtype):
  return rigid_body.random_quaternion(key, dtype)


# pylint: disable=invalid-name
class SimulateTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_2d_simple(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square

    energy_fn = rigid_body.energy(energy.soft_sphere_pair(displacement), shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return energy_fn(pos) + rigid_body.kinetic_energy(pos, mom, mass)

    state = init_fn(key, body, 1e-3, mass=shape.mass())
    E_initial = total_energy(state)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_2d_multi_shape_species(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.concatenate_shapes(
      rigid_body.square,
      rigid_body.trimer
    )
    species = onp.where(onp.arange(N) < PARTICLE_COUNT // 2, 0, 1)

    energy_fn = rigid_body.energy(energy.soft_sphere_pair(displacement), shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return energy_fn(pos) + rigid_body.kinetic_energy(pos, mom, mass)

    state = init_fn(key, body, 1e-3, mass=shape.mass(species))
    E_initial = total_energy(state)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_2d_multi_atom_species(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square.set(point_species=jnp.array([0, 1, 0, 1]))

    energy_fn = rigid_body.energy(energy.soft_sphere_pair(displacement), shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return energy_fn(pos) + rigid_body.kinetic_energy(pos, mom, mass)

    state = init_fn(key, body, 1e-3, mass=shape.mass())
    E_initial = total_energy(state)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_2d_neighbor_list(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(displacement,
                                                              box_size)
    neighbor_fn, energy_fn = rigid_body.energy_neighbor_list(energy_fn,
                                                             neighbor_fn,
                                                             shape)
    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state, nbrs):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return (energy_fn(pos, neighbor=nbrs) +
              rigid_body.kinetic_energy(pos, mom, mass))

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, 1e-3, mass=shape.mass(), neighbor=nbrs)
    E_initial = total_energy(state, nbrs)
    def step(i, state_nbrs):
      state, nbrs = state_nbrs
      nbrs = nbrs.update(body)
      state = step_fn(state, neighbor=nbrs)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, step, (state, nbrs))
    E_final = total_energy(state, nbrs)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_2d_neighbor_list_multi_atom_species(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square.set(point_species=jnp.array([0, 1, 0, 1]))

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(displacement,
                                                              box_size)
    neighbor_fn, energy_fn = rigid_body.energy_neighbor_list(energy_fn,
                                                             neighbor_fn,
                                                             shape)
    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state, nbrs):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return (energy_fn(pos, neighbor=nbrs) +
              rigid_body.kinetic_energy(pos, mom, mass))

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, 1e-3, mass=shape.mass(), neighbor=nbrs)
    E_initial = total_energy(state, nbrs)
    def step(i, state_nbrs):
      state, nbrs = state_nbrs
      nbrs = nbrs.update(body)
      state = step_fn(state, neighbor=nbrs)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, step, (state, nbrs))
    E_final = total_energy(state, nbrs)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_2d_neighbor_list_multi_shape_species(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.concatenate_shapes(
      rigid_body.square.set(point_species=jnp.array([0, 1, 0, 1])),
      rigid_body.trimer.set(point_species=jnp.array([0, 0, 1]))
    )
    shape_species = onp.where(onp.arange(N) < PARTICLE_COUNT // 2, 0, 1)

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
      displacement,
      box_size,
      sigma=jnp.array([[0.5, 1.0],
                       [1.0, 1.5]], dtype=dtype),
      species=2)
    neighbor_fn, energy_fn = rigid_body.energy_neighbor_list(energy_fn,
                                                             neighbor_fn,
                                                             shape,
                                                             shape_species)
    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state, nbrs):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return (energy_fn(pos, neighbor=nbrs) +
              rigid_body.kinetic_energy(pos, mom, mass))

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key,
                    body,
                    1e-3,
                    mass=shape.mass(shape_species),
                    neighbor=nbrs)
    E_initial = total_energy(state, nbrs)
    def step(i, state_nbrs):
      state, nbrs = state_nbrs
      nbrs = nbrs.update(body)
      state = step_fn(state, neighbor=nbrs)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, step, (state, nbrs))
    E_final = total_energy(state, nbrs)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_3d_simple(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, quat_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, dtype)

    body = rigid_body.RigidBody(R, quaternion)
    shape = rigid_body.rigid_body_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], dtype),
      rigid_body.tetrahedron.masses
    )

    energy_fn = rigid_body.energy(energy.soft_sphere_pair(displacement), shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return energy_fn(pos) + rigid_body.kinetic_energy(pos, mom, mass)

    state = init_fn(key, body, 1e-3, mass=shape.mass())
    E_initial = total_energy(state)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_3d_multi_shape_species(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, quat_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, dtype)

    species = onp.where(onp.arange(N) < N // 2, 0, 1)

    body = rigid_body.RigidBody(R, quaternion)
    shape = rigid_body.rigid_body_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], f32),
      rigid_body.tetrahedron.masses)
    shape = rigid_body.concatenate_shapes(rigid_body.tetrahedron, shape)

    energy_fn = rigid_body.energy(energy.soft_sphere_pair(displacement),
                                  shape,
                                  species)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return energy_fn(pos) + rigid_body.kinetic_energy(pos, mom, mass)

    state = init_fn(key, body, 1e-3, mass=shape.mass(species))
    E_initial = total_energy(state)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_3d_multi_atom_shape_species(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, quat_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, dtype)

    species = onp.where(onp.arange(N) < N // 2, 0, 1)

    body = rigid_body.RigidBody(R, quaternion)

    shape = rigid_body.rigid_body_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], f32),
      rigid_body.tetrahedron.masses)
    shape = rigid_body.concatenate_shapes(rigid_body.tetrahedron, shape)
    shape = shape.set(point_species=jnp.array([0, 1, 0, 1, 1, 0, 1, 0]))

    pair_energy_fn = energy.soft_sphere_pair(displacement,
                                             sigma=jnp.array([[0.5, 1.0],
                                                              [1.0, 1.5]],
                                                             f32),
                                             species=2)
    energy_fn = rigid_body.energy(pair_energy_fn, shape, species)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return energy_fn(pos) + rigid_body.kinetic_energy(pos, mom, mass)

    state = init_fn(key, body, 1e-3, mass=shape.mass(species))
    E_initial = total_energy(state)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nve_3d_neighbor_list(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, quat_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, dtype)

    body = rigid_body.RigidBody(R, quaternion)
    shape = rigid_body.rigid_body_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], dtype),
      rigid_body.tetrahedron.masses
    )

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(displacement,
                                                              box_size)
    neighbor_fn, energy_fn = rigid_body.energy_neighbor_list(energy_fn,
                                                             neighbor_fn,
                                                             shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state, nbrs):
      pos = state.position
      mom = state.momentum
      mass = state.mass

      return (energy_fn(pos, neighbor=nbrs) +
              rigid_body.kinetic_energy(pos, mom, mass))

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, 1e-3, mass=shape.mass(), neighbor=nbrs)
    E_initial = total_energy(state, nbrs)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state, neighbor=nbrs)
      nbrs = jit(nbrs.update)(state.position)
    E_final = total_energy(state, nbrs)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype


if __name__ == '__main__':
  absltest.main()
