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
import pytest

import numpy as onp

import jax
from jax import jit
from jax import vmap
from jax import random
from jax import lax
from jax import test_util as jtu

import jax.numpy as jnp

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import test_util
from jax_md import partition
from jax_md import util
from jax_md import rigid_body
from jax_md import minimize

from functools import partial

jax.config.parse_flags_with_absl()



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
if jax.config.jax_enable_x64:
  DTYPE += [f64]


@partial(vmap, in_axes=(0, None))
def rand_quat(key, dtype):
  return rigid_body.random_quaternion(key, dtype)

@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
  # This is a fixture that runs before and after each test.
  # This fixes issue 227 (https://github.com/jax-md/jax-md/issues/277)
  yield # this is where the testing happens
  jax.clear_caches()

# pylint: disable=invalid-name
class RigidBodyTest(test_util.JAXMDTestCase):
  @parameterized.named_parameters(test_util.cases_from_list(
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

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      return energy_fn(pos) + simulate.kinetic_energy(state)

    state = init_fn(key, body, 1e-3, mass=shape.mass())
    E_initial = total_energy(state)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
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

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      return energy_fn(pos) + simulate.kinetic_energy(state)

    state = init_fn(key, body, 1e-3, mass=shape.mass(species))
    E_initial = total_energy(state)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
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

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      return energy_fn(pos) + simulate.kinetic_energy(state)

    state = init_fn(key, body, 1e-3, mass=shape.mass())
    E_initial = total_energy(state)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
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
    neighbor_fn, energy_fn = rigid_body.point_energy_neighbor_list(energy_fn,
                                                                   neighbor_fn,
                                                                   shape)
    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state, nbrs):
      pos = state.position
      return (energy_fn(pos, neighbor=nbrs) +
              simulate.kinetic_energy(state))

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, 1e-3, mass=shape.mass(), neighbor=nbrs)
    E_initial = total_energy(state, nbrs)
    def step(i, state_nbrs):
      state, nbrs = state_nbrs
      nbrs = nbrs.update(state.position)
      state = step_fn(state, neighbor=nbrs)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, step, (state, nbrs))
    E_final = total_energy(state, nbrs)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
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
    neighbor_fn, energy_fn = rigid_body.point_energy_neighbor_list(energy_fn,
                                                                   neighbor_fn,
                                                                   shape)
    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state, nbrs):
      pos = state.position
      return (energy_fn(pos, neighbor=nbrs) +
              simulate.kinetic_energy(state))

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, 1e-3, mass=shape.mass(), neighbor=nbrs)
    E_initial = total_energy(state, nbrs)
    def step(i, state_nbrs):
      state, nbrs = state_nbrs
      nbrs = nbrs.update(state.position)
      state = step_fn(state, neighbor=nbrs)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, step, (state, nbrs))
    E_final = total_energy(state, nbrs)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
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
    neighbor_fn, energy_fn = rigid_body.point_energy_neighbor_list(
      energy_fn, neighbor_fn, shape, shape_species)
    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state, nbrs):
      pos = state.position
      return (energy_fn(pos, neighbor=nbrs) +
              simulate.kinetic_energy(state))

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key,
                    body,
                    1e-3,
                    mass=shape.mass(shape_species),
                    neighbor=nbrs)
    E_initial = total_energy(state, nbrs)
    def step(i, state_nbrs):
      state, nbrs = state_nbrs
      nbrs = nbrs.update(state.position)
      state = step_fn(state, neighbor=nbrs)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, step, (state, nbrs))
    E_final = total_energy(state, nbrs)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_3d_quaternion_derivative(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, quat_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, dtype)

    body = rigid_body.RigidBody(R, quaternion)
    shape = rigid_body.tetrahedron

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)

    F = quantity.force(energy_fn)(body)
    S = rigid_body.S(body.orientation)
    F_body = jnp.einsum('nij,ni->nj', S, F.orientation.vec)
    self.assertAllClose(F_body[:, 0], jnp.zeros_like(F_body[:, 0]))

  @parameterized.named_parameters(test_util.cases_from_list(
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
    shape = rigid_body.point_union_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], dtype),
      rigid_body.tetrahedron.masses
    )

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      return energy_fn(pos) + simulate.kinetic_energy(state)

    state = init_fn(key, body, 1e-3, mass=shape.mass())
    E_initial = total_energy(state)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
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
    shape = rigid_body.point_union_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], f32),
      rigid_body.tetrahedron.masses)
    shape = rigid_body.concatenate_shapes(rigid_body.tetrahedron, shape)

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape,
                                        species)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      return energy_fn(pos) + simulate.kinetic_energy(state)

    state = init_fn(key, body, 1e-3, mass=shape.mass(species))
    E_initial = total_energy(state)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
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

    shape = rigid_body.point_union_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], f32),
      jnp.array([1.0, 2.0, 3.0, 4.0], f32))
    shape = rigid_body.concatenate_shapes(rigid_body.tetrahedron, shape)
    shape = shape.set(point_species=jnp.array([0, 1, 0, 1, 1, 0, 1, 0]))

    pair_energy_fn = energy.soft_sphere_pair(displacement,
                                             sigma=jnp.array([[0.5, 1.0],
                                                              [1.0, 1.5]],
                                                              f32),
                                             species=2)
    energy_fn = rigid_body.point_energy(pair_energy_fn, shape, species)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state):
      pos = state.position
      return energy_fn(pos) + simulate.kinetic_energy(state)

    state = init_fn(key, body, 1e-3, mass=shape.mass(species))
    E_initial = total_energy(state)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = total_energy(state)

    tol = 5e-5
    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
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
    shape = rigid_body.point_union_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], dtype),
      rigid_body.tetrahedron.masses
    )

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(displacement,
                                                              box_size)
    neighbor_fn, energy_fn = rigid_body.point_energy_neighbor_list(energy_fn,
                                                                   neighbor_fn,
                                                                   shape)

    init_fn, step_fn = simulate.nve(energy_fn, shift)

    step_fn = jit(step_fn)

    @jit
    def total_energy(state, nbrs):
      pos = state.position
      return (energy_fn(pos, neighbor=nbrs) +
              simulate.kinetic_energy(state))

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, 1e-3, mass=shape.mass(), neighbor=nbrs)
    E_initial = total_energy(state, nbrs)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state, neighbor=nbrs)
      nbrs = jit(neighbor_fn.update)(state.position, nbrs)
    E_final = total_energy(state, nbrs)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  def test_shape_derivative_2d(self):
    def shape_energy_fn(points):
      body = rigid_body.RigidBody(
        jnp.array([[0.0, 0.0],
                  [1.5, 0.0]]),
        jnp.array([0.0, 0.0])
      )

      shape = rigid_body.point_union_shape(points, 1.0)

      displacement, shift = space.free()
      energy_fn = energy.soft_sphere_pair(displacement)
      energy_fn = rigid_body.point_energy(energy_fn, shape)

      return energy_fn(body)

    shape = rigid_body.trimer
    jtu.check_grads(shape_energy_fn, (shape.points,), 1, atol=5e-4, rtol=5e-4)

  def test_shape_derivative_3d(self):
    # This test currently fails on CPU due to Issue #10877
    # https://github.com/google/jax/issues/10877
    if jax.default_backend() == 'cpu':
      self.skipTest('Shape derivatives are broken on CPU in three-dimensions.')
    def shape_energy_fn(points):
      body = rigid_body.RigidBody(
        jnp.array([[0.0, 0.0, 0.0],
                   [0.5, 0.25, 0.15]]),
        rigid_body.Quaternion(
          jnp.array([[1.0, 0.0, 0.0, 0.0],
                     [1.0, 0.1, 0.0, 0.0]]))
      )

      # Right now, if we call rigid body shape, inside the function
      # we are taking the gradient of, we get NaNs. This is presumably
      # from the eigh command. If possible, we should make this safer.
      # Possibly this is because the matrix has degenerate eigenvalues.
      # If we allow the eigenvalues to not be degenerate then we don't
      # get NaNs anymore.
      shape = rigid_body.point_union_shape(points, jnp.ones((len(points),)))

      displacement, shift = space.free()
      energy_fn = energy.soft_sphere_pair(displacement)
      energy_fn = rigid_body.point_energy(energy_fn, shape)

      return energy_fn(body)
    points = jnp.array([[-0.5, -0.5, -0.5],
                        [-0.5, -0.5,  0.5],
                        [ 0.5, -0.5, -0.5],
                        [ 0.5, -0.5,  0.5],
                        [-0.5,  0.5, -0.5],
                        [-0.5,  0.5,  0.5],
                        [ 0.5,  0.5, -0.5],
                        [ 0.5,  0.5,  0.5]]) * jnp.array([[1.0, 1.1, 1.2]])
    jtu.check_grads(shape_energy_fn, (points,), 1, modes='rev')

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nvt_2d_simple(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=shape.mass())
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nvt_2d_multi_shape_species(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.concatenate_shapes(rigid_body.square, rigid_body.trimer)
    species = onp.where(onp.arange(N) < N // 2, 0, 1)

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape,
                                        species)

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=shape.mass(species))
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nvt_2d_multi_atom_species(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square.set(point_species=jnp.array([0, 1, 0, 1]))

    energy_fn = energy.soft_sphere_pair(displacement,
                                        sigma=jnp.array([[0.5, 1.0],
                                                         [1.0, 1.5]], f32),
                                        species=2)

    energy_fn = rigid_body.point_energy(energy_fn, shape)

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=shape.mass())
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nvt_2d_neighbor_list(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
      displacement,
      box_size)
    neighbor_fn, energy_fn = rigid_body.point_energy_neighbor_list(energy_fn,
                                                                   neighbor_fn,
                                                                   shape)

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, mass=shape.mass(), neighbor=nbrs)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT,
                                                   neighbor=nbrs)
    def sim_fn(i, state_nbrs):
      state, nbrs = state_nbrs
      state = step_fn(state, neighbor=nbrs)
      nbrs = nbrs.update(state.position)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, sim_fn, (state, nbrs))
    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT,
                                                 neighbor=nbrs)
    self.assertFalse(nbrs.did_buffer_overflow)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nvt_2d_multi_atom_species_neighbor_list(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 2)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square.set(point_species=jnp.array([0, 1, 0, 1]))

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
      displacement,
      box_size,
      sigma=jnp.array([[0.5, 1.0],
                       [1.0, 1.5]], f32),
      species=2)
    neighbor_fn, energy_fn = rigid_body.point_energy_neighbor_list(energy_fn,
                                                                   neighbor_fn,
                                                                   shape)

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, mass=shape.mass(), neighbor=nbrs)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT,
                                                   neighbor=nbrs)
    def sim_fn(i, state_nbrs):
      state, nbrs = state_nbrs
      state = step_fn(state, neighbor=nbrs)
      nbrs = nbrs.update(state.position)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, sim_fn, (state, nbrs))
    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT,
                                                 neighbor=nbrs)
    self.assertFalse(nbrs.did_buffer_overflow)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nvt_2d_multi_shape_species_neighbor_list(self, dtype):
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
      rigid_body.trimer.set(point_species=jnp.array([0, 0, 1])))

    shape_species = onp.where(onp.arange(N) < N // 2, 0, 1)

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
      displacement,
      box_size,
      sigma=jnp.array([[0.5, 1.0],
                       [1.0, 1.5]], f32),
      species=2)
    neighbor_fn, energy_fn = rigid_body.point_energy_neighbor_list(energy_fn,
                                                                   neighbor_fn,
                                                                   shape)

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, mass=shape.mass(shape_species), neighbor=nbrs)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT,
                                                   neighbor=nbrs)
    def sim_fn(i, state_nbrs):
      state, nbrs = state_nbrs
      state = step_fn(state, neighbor=nbrs)
      nbrs = nbrs.update(state.position)
      return state, nbrs
    state, nbrs = lax.fori_loop(0, DYNAMICS_STEPS, sim_fn, (state, nbrs))
    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT,
                                                 neighbor=nbrs)
    self.assertFalse(nbrs.did_buffer_overflow)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_nvt_3d_simple(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, quat_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, dtype)

    body = rigid_body.RigidBody(R, quaternion)
    shape = rigid_body.point_union_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], dtype),
      rigid_body.tetrahedron.masses
    )

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=shape.mass())
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)
    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    tol = 5e-8 if dtype == f64 else 5e-5

    self.assertAllClose(E_initial, E_final, rtol=tol, atol=tol)
    assert E_final.dtype == dtype


  def test_jit_shape(self):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, quat_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3))
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, jnp.float32)

    body = rigid_body.RigidBody(R, quaternion)

    @jit
    def compute_energy(body, points, masses):
      shape = rigid_body.point_union_shape(points, masses)
      energy_fn = rigid_body.point_energy(
        energy.soft_sphere_pair(displacement), shape)
      return energy_fn(body)

    s = rigid_body.tetrahedron
    compute_energy(body, s.points, s.masses)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_fire_2d(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 2), dtype=dtype)
    angle = random.uniform(angle_key, (N,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    shape = rigid_body.square

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)
    init_fn, step_fn = minimize.fire_descent(energy_fn, shift)

    state = init_fn(body, mass=shape.mass())
    state = lax.fori_loop(0, 60, lambda i, s: step_fn(s), state)

    self.assertTrue(energy_fn(state.position) < 35.0)
    self.assertTrue(state.position.center.dtype==dtype)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in DTYPE))
  def test_fire_3d_multispecies(self, dtype):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, angle_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)

    angle_key = random.split(angle_key, N)
    quat = rand_quat(angle_key, dtype)

    body = rigid_body.RigidBody(R, quat)

    shape = rigid_body.concatenate_shapes(rigid_body.tetrahedron,
                                          rigid_body.octohedron)
    shape_species = onp.where(onp.arange(N) < N / 2, 0, 1)

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                  shape,
                                  shape_species)

    init_fn, step_fn = minimize.fire_descent(energy_fn, shift,
                                             dt_start=1e-2, dt_max=4e-2)
    state = init_fn(body, mass=shape.mass(shape_species))
    state = lax.fori_loop(0, 60, lambda i, s: step_fn(s), state)

    self.assertTrue(energy_fn(state.position) < 12.0)
    self.assertTrue(state.position.center.dtype==dtype)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'kT={int(kT*1e3)}_dtype={dtype.__name__}',
          'dtype': dtype,
          'kT': kT
      } for dtype in DTYPE for kT in [1e-3, 5e-3, 1e-2, 1e-1]))
  def test_nvt_3d_simple_langevin(self, dtype, kT):
    N = PARTICLE_COUNT
    box_size = quantity.box_size_at_number_density(N, 0.1, 3)

    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, pos_key, quat_key = random.split(key, 3)

    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, dtype)

    body = rigid_body.RigidBody(R, quaternion)
    shape = rigid_body.point_union_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], dtype),
      rigid_body.tetrahedron.masses
    )

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape)

    dt = 5e-4

    gamma = rigid_body.RigidBody(0.1, 0.1)
    init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift, dt, kT, gamma)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=shape.mass())

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)

    kT_final = rigid_body.temperature(state.position, state.momentum, state.mass)

    tol = 5e-4 if kT < 2e-3 else kT / 10
    self.assertAllClose(kT_final, dtype(kT), rtol=tol, atol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'kT={int(kT * 1e3)}_dtype={dtype.__name__}',
          'dtype': dtype,
          'kT': kT
      } for dtype in DTYPE for kT in [1e-3, 5e-3, 1e-2, 1e-1]))
  def test_nvt_langevin_3d_multi_shape_species(self, dtype, kT):
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
    shape = rigid_body.point_union_shape(
      rigid_body.tetrahedron.points * jnp.array([[1.0, 2.0, 3.0]], f32),
      rigid_body.tetrahedron.masses)
    shape = rigid_body.concatenate_shapes(rigid_body.tetrahedron, shape)

    energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement),
                                        shape,
                                        species)

    dt = 5e-4
    gamma = rigid_body.RigidBody(0.1, 0.1)
    init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift, dt, kT, gamma)

    step_fn = jit(step_fn)

    state = init_fn(key, body, shape.mass(species))

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)

    kT_final = rigid_body.temperature(state.position, state.momentum, state.mass)
    tol = 5e-4 if kT < 2e-3 else kT / 8
    self.assertAllClose(kT_final, dtype(kT), rtol=tol, atol=tol)

if __name__ == '__main__':
  absltest.main()
