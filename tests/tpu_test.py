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
"""Tests for JAX MD TPU code."""

from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized

from jax_md.test_util import update_test_tolerance
from jax_md import quantity

import jax
from jax import jit, vmap, grad
from jax import lax
import jax.numpy as np
from jax import random

from jax_md import energy, space, simulate
from jax_md import test_util

import numpy as onp

from jax_md import tpu


jax.config.parse_flags_with_absl()

update_test_tolerance(5e-5, 1e-7)


def get_test_grid(rng_key, topology=None, num_dims=2, add_aux=False, ):
  # magic numbers to make the gird fold evenly, after splitting
  # across devices and padding see propose_tpu_box_size.
  cell_size = 1./4.
  interaction_distance = 0.95
  if topology:
    if num_dims == 1:
      box_size_in_cells = 636
    elif num_dims == 2:
      box_size_in_cells = 160
    elif num_dims == 3:
      box_size_in_cells = 32
      # box_size = 16.
  else:
    if num_dims == 1:
      box_size_in_cells = 512
    elif num_dims == 2:
      box_size_in_cells = 80
    elif num_dims == 3:
      box_size_in_cells = 16

  box_size_in_cells = tpu.nearest_valid_grid_size(
      box_size_in_cells,
      topology,
      int(onp.ceil(interaction_distance / cell_size) + 1),
      dimension=num_dims)

  box_size = box_size_in_cells * cell_size
  box_size_in_cells = tuple(box_size_in_cells)
  displacement_fn, shift_fn = space.periodic(
      box_size if num_dims > 1 else box_size[0])

  energy_fn = energy.soft_sphere_pair(displacement_fn)
  tpu_energy_fn, tpu_force_fn = tpu.soft_sphere(sigma=1.0)

  points = []
  for _bs in box_size:
    points.append(onp.linspace(0., _bs-1, num=int(np.ceil(_bs*2))))
  if num_dims == 3:
    X, Y, Z = onp.meshgrid(*points)
    R = onp.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1) + 0.1
  elif num_dims == 2:
    X, Y = onp.meshgrid(*points)
    R = onp.stack((X.ravel(), Y.ravel()), axis=1) + 0.1
  elif num_dims == 1:
    R = points[0].reshape((-1, 1)) + 0.1

  R += random.normal(rng_key, R.shape) * 0.1
  R = onp.array(R, onp.float64)
  if add_aux:
    # these are used as velocities
    rng_key = random.split(rng_key)[0]
    V = random.normal(rng_key, R.shape)
    R_grid, V_grid  = tpu.to_grid(R, box_size_in_cells, cell_size, interaction_distance, topology, aux=V, strategy='linear')
    print(f"R.shape {R.shape}, aux.shape {V.shape}, grid shape {V_grid.shape}, occupancy {R.shape[0]/float(onp.prod(V_grid.shape[:-1]))}")
    return ((R_grid, V_grid), tpu_energy_fn, tpu_force_fn), ((R, V), energy_fn, shift_fn)

  R_grid  = tpu.to_grid(R, box_size_in_cells, cell_size, interaction_distance, topology, strategy='linear')
  print(f"R.shape {R.shape}, grid shape {R_grid.cell_data.shape}, occupancy {R.shape[0]/float(onp.prod(R_grid.cell_data.shape[:-1]))}")
  return (R_grid, tpu_energy_fn, tpu_force_fn), (R, energy_fn, shift_fn)


SPATIAL_DIMENSIONS = [1, 2, 3]
TOPOLOGIES = [(), (2,)]


class ConvolutionalMDTest(test_util.JAXMDTestCase):

  @parameterized.named_parameters(
      test_util.cases_from_list({
          'testcase_name': '_numdims={}_topology={}'.format(num_dims, topology),
          'num_dims': num_dims,
          'topology': topology
      } for num_dims in SPATIAL_DIMENSIONS for topology in TOPOLOGIES))
  def test_position_recovery(self, num_dims, topology):
    if topology:
      if jax.device_count() == 1:
        self.skipTest('Skipping non-trivial topology; only one device detected.')
      topology = topology + (1,) * (num_dims - 1)

    key = random.PRNGKey(0)
    sim_tpu, sim_cpu = get_test_grid(key, topology, num_dims)

    (R_grid, tpu_energy_fn, tpu_force_fn) = sim_tpu
    (R, energy_fn, shift_fn) = sim_cpu
    grid_positions = tpu.from_grid(R_grid)

    displacement_fn, _ = space.periodic(onp.array(R_grid.box_size_in_cells) * R_grid.cell_size)
    dr = space.distance(space.map_bond(displacement_fn)(grid_positions, R))

    self.assertAllClose(np.zeros_like(dr), dr)

  @parameterized.named_parameters(
      test_util.cases_from_list({
          'testcase_name': '_numdims={}_topology={}'.format(num_dims, topology),
          'num_dims': num_dims,
          'topology': topology
      } for num_dims in SPATIAL_DIMENSIONS for topology in TOPOLOGIES))
  def test_position_and_aux_recovery(self, num_dims, topology):
    if topology:
      if jax.device_count() == 1:
        self.skipTest('Skipping non-trivial topology; only one device detected.')
      topology = topology + (1,) * (num_dims - 1)

    key = random.PRNGKey(0)
    sim_tpu, sim_cpu = get_test_grid(key, topology, num_dims, True)

    ((R_grid, V_grid), tpu_energy_fn, tpu_force_fn) = sim_tpu
    ((R, V), energy_fn, shift_fn) = sim_cpu

    grid_positions, grid_aux = tpu.from_grid(R_grid, V_grid)

    displacement_fn, _ = space.periodic(onp.array(R_grid.box_size_in_cells) * R_grid.cell_size)
    dr = space.distance(space.map_bond(displacement_fn)(grid_positions, R))

    self.assertAllClose(np.zeros_like(dr), dr)
    self.assertAllClose(grid_aux, V)

  @parameterized.named_parameters(
      test_util.cases_from_list({
          'testcase_name': '_numdims={}_topology={}'.format(num_dims, topology),
          'num_dims': num_dims,
          'topology': topology
      } for num_dims in SPATIAL_DIMENSIONS for topology in TOPOLOGIES))
  def test_forces(self, num_dims, topology):
    if topology:
      if jax.device_count() == 1:
        self.skipTest('Skipping non-trivial topology; only one device detected.')
      topology = topology + (1,) * (num_dims - 1)

    key = random.PRNGKey(0)
    sim_tpu, sim_cpu = get_test_grid(key, topology, num_dims)

    (R_grid, tpu_energy_fn, tpu_force_fn) = sim_tpu
    (R, energy_fn, shift_fn) = sim_cpu

    tpu_force_fn = jit(tpu_force_fn)
    tpu_force_fn = tpu.parallelize(tpu_force_fn, topology)
    forces = tpu_force_fn(R_grid)
    exact_forces = -jit(grad(energy_fn), backend='cpu')(R)

    _, tpu_forces = tpu.from_grid(R_grid, forces)
    self.assertAllClose(tpu_forces, exact_forces)

  @parameterized.named_parameters(
      test_util.cases_from_list({
          'testcase_name': '_numdims={}_topology={}'.format(num_dims, topology),
          'num_dims': num_dims,
          'topology': topology
      } for num_dims in SPATIAL_DIMENSIONS for topology in TOPOLOGIES))
  def test_nve(self, num_dims, topology):
    if topology:
      if jax.device_count() == 1:
        self.skipTest('Skipping non-trivial topology; only one device detected.')
      topology = topology + (1,) * (num_dims - 1)

    key = random.PRNGKey(0)
    sim_tpu, sim_cpu = get_test_grid(key, topology, num_dims, True)

    ((R_grid, V_grid), tpu_energy_fn, tpu_force_fn) = sim_tpu
    ((R, V), energy_fn, shift_fn) = sim_cpu

    step_size = 1e-3
    steps = 50

    # CNN-MD
    init_fn, apply_fn = tpu.test_nve(tpu_force_fn, step_size)
    tpu_state = tpu.parallelize(init_fn, topology)(R_grid, V_grid)
    @jit
    def sim(state):
      def do_sim(i, state):
        return apply_fn(state)
      return lax.fori_loop(0, steps, do_sim, state)
    sim = tpu.parallelize(sim, topology)
    new_state = sim(tpu_state)

    ## JAX-MD baseline
    jmd_init_fn, jmd_apply_fn = simulate.nve(energy_fn, shift_fn, step_size)
    force_fn = quantity.force(energy_fn)
    mass = 1.0

    jmd_state = simulate.NVEState(R, V, force_fn(R), mass)
    def jmd_step_fn(state, i):
      return jmd_apply_fn(state), i
    jax.config.update('jax_numpy_rank_promotion', 'warn')
    new_jmd_state, _ = lax.scan(jmd_step_fn, jmd_state, np.arange(steps))
    jax.config.update('jax_numpy_rank_promotion', 'raise')

    # compare outputs
    grid_positions, grid_aux = tpu.from_grid(new_state.position, new_state.velocity)

    tol = 1e-5

    self.assertAllClose(grid_positions, new_jmd_state.position, atol=tol, rtol=tol)
    self.assertAllClose(grid_aux, new_jmd_state.velocity, atol=tol, rtol=tol)


if __name__ == '__main__':
  absltest.main()

