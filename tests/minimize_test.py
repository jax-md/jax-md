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

"""Tests for jax_md.minimize."""

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
from jax import jit
from jax import lax
import jax.numpy as jnp

from jax_md import space
from jax_md import energy
from jax_md import minimize
from jax_md import quantity
from jax_md import partition
from jax_md.util import *
from jax_md import test_util
from jax_md.custom_partition import (
  neighbor_list_multi_image,
  estimate_max_neighbors_from_box,
)
from jax_md.custom_smap import pair_neighbor_list_multi_image

jax.config.parse_flags_with_absl()

PARTICLE_COUNT = 10
OPTIMIZATION_STEPS = 10
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

if jax.config.jax_enable_x64:
  DTYPE = [f32, f64]
else:
  DTYPE = [f32]


class DynamicsTest(test_util.JAXMDTestCase):
  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_gradient_descent(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split, split0 = random.split(key, 3)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )
      R0 = random.uniform(
        split0, (PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )

      energy = lambda R, **kwargs: jnp.sum((R - R0) ** 2)
      _, shift_fn = space.free()

      opt_init, opt_apply = minimize.gradient_descent(
        energy, shift_fn, f32(1e-1)
      )

      E_current = energy(R)
      dr_current = jnp.sum((R - R0) ** 2)

      for _ in range(OPTIMIZATION_STEPS):
        R = opt_apply(R)
        E_new = energy(R)
        dr_new = jnp.sum((R - R0) ** 2)
        assert E_new < E_current
        assert E_new.dtype == dtype
        assert dr_new < dr_current
        assert dr_new.dtype == dtype
        E_current = E_new
        dr_current = dr_new

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_fire_descent(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split, split0 = random.split(key, 3)
      R = random.uniform(
        split, (PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )
      R0 = random.uniform(
        split0, (PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )

      energy = lambda R, **kwargs: jnp.sum((R - R0) ** 2)
      _, shift_fn = space.free()

      opt_init, opt_apply = minimize.fire_descent(energy, shift_fn)

      opt_state = opt_init(R)
      E_current = energy(R)
      dr_current = jnp.sum((R - R0) ** 2)

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
        dr_new = jnp.sum((R - R0) ** 2)
        assert E_new < E_current
        assert E_new.dtype == dtype
        assert dr_new < dr_current
        assert dr_new.dtype == dtype
        E_current = E_new
        dr_current = dr_new


class FireDescentBoxTest(test_util.JAXMDTestCase):
  """Tests for fire_descent and fire_descent_box with multi-image NL."""

  def setUp(self):
    super().setUp()
    dtype = jnp.float64
    R_frac, cubic_box = test_util.make_fcc_fractional(
      n_cells=1, a=1.55, dtype=dtype
    )
    strain = jnp.array(
      [[1.03, 0.01, 0.005], [0.01, 0.98, 0.005], [0.005, 0.005, 1.01]],
      dtype=dtype,
    )
    self.box = strain @ cubic_box
    key = random.PRNGKey(42)
    self.R = jnp.mod(
      R_frac + random.normal(key, R_frac.shape, dtype=dtype) * 0.002, 1.0
    )
    self.N = len(self.R)
    self.dtype = dtype

    disp_fn, self.shift_fn = space.periodic_general(
      self.box, fractional_coordinates=True
    )
    r_cutoff = 2.5
    max_nbrs = estimate_max_neighbors_from_box(
      self.box, r_cutoff, n_atoms=self.N, safety_factor=3.0
    )
    self.neighbor_fn, raw_energy_fn = energy.lennard_jones_neighbor_list(
      disp_fn,
      self.box,
      r_cutoff=r_cutoff,
      r_onset=2.0,
      dr_threshold=0.0,
      fractional_coordinates=True,
      neighbor_list_fn=neighbor_list_multi_image,
      pair_neighbor_list_fn=pair_neighbor_list_multi_image,
      max_neighbors=max_nbrs,
      format=partition.Sparse,
    )

    def energy_fn(R, box=None, neighbor=None, **kwargs):
      if box is not None and neighbor is not None:
        neighbor = neighbor.set(
          box=box, shifts=neighbor.shifts.astype(box.dtype)
        )
      return raw_energy_fn(R, neighbor=neighbor, **kwargs)

    self.energy_fn = energy_fn
    self.nbrs = self.neighbor_fn.allocate(self.R, box=self.box)

  def test_fire_descent_relaxation(self):
    """Position-only FIRE must reduce energy."""
    nbrs = self.nbrs
    E_init = float(self.energy_fn(self.R, box=self.box, neighbor=nbrs))

    init_fn, apply_fn = minimize.fire_descent(
      self.energy_fn,
      self.shift_fn,
      dt_start=0.001,
      dt_max=0.01,
    )
    state = init_fn(self.R, box=self.box, neighbor=nbrs)

    for _ in range(100):
      state = apply_fn(state, box=self.box, neighbor=nbrs)
      nbrs = nbrs.update(state.position, box=self.box)

    E_final = float(self.energy_fn(state.position, box=self.box, neighbor=nbrs))
    self.assertLess(E_final, E_init, 'Energy must decrease.')

  def test_fire_descent_box_relaxation(self):
    """Box FIRE must reduce energy and change the box."""
    nbrs = self.nbrs
    E_init = float(self.energy_fn(self.R, box=self.box, neighbor=nbrs))

    init_fn, apply_fn = minimize.fire_descent_box(
      self.energy_fn,
      self.shift_fn,
      dt_start=0.001,
      dt_max=0.01,
    )
    state = init_fn(self.R, self.box, neighbor=nbrs)

    for _ in range(100):
      state = apply_fn(state, neighbor=nbrs)
      nbrs = nbrs.update(state.position, box=state.box)

    E_final = float(
      self.energy_fn(state.position, box=state.box, neighbor=nbrs)
    )
    self.assertLess(E_final, E_init, 'Energy must decrease.')
    self.assertGreater(
      float(jnp.linalg.norm(state.box - self.box)),
      1e-3,
      'Box should have changed.',
    )

  def test_fire_descent_box_with_threshold(self):
    """Box FIRE with dr_threshold > 0 must still reduce energy."""
    dtype = self.dtype
    R_frac, cubic_box = test_util.make_fcc_fractional(
      n_cells=1, a=1.55, dtype=dtype
    )
    strain = jnp.array(
      [[1.03, 0.01, 0.005], [0.01, 0.98, 0.005], [0.005, 0.005, 1.01]],
      dtype=dtype,
    )
    box = strain @ cubic_box
    key = random.PRNGKey(42)
    R = jnp.mod(
      R_frac + random.normal(key, R_frac.shape, dtype=dtype) * 0.002, 1.0
    )
    N = len(R)

    disp_fn, shift_fn = space.periodic_general(box, fractional_coordinates=True)
    r_cutoff = 2.5
    max_nbrs = estimate_max_neighbors_from_box(
      box, r_cutoff, n_atoms=N, safety_factor=3.0
    )
    neighbor_fn, raw_energy_fn = energy.lennard_jones_neighbor_list(
      disp_fn,
      box,
      r_cutoff=r_cutoff,
      r_onset=2.0,
      dr_threshold=0.5,
      fractional_coordinates=True,
      neighbor_list_fn=neighbor_list_multi_image,
      pair_neighbor_list_fn=pair_neighbor_list_multi_image,
      max_neighbors=max_nbrs,
      format=partition.Sparse,
    )

    def energy_fn(R, box=None, neighbor=None, **kwargs):
      if box is not None and neighbor is not None:
        neighbor = neighbor.set(
          box=box, shifts=neighbor.shifts.astype(box.dtype)
        )
      return raw_energy_fn(R, neighbor=neighbor, **kwargs)

    nbrs = neighbor_fn.allocate(R, box=box)
    E_init = float(energy_fn(R, box=box, neighbor=nbrs))

    init_fn, apply_fn = minimize.fire_descent_box(
      energy_fn,
      shift_fn,
      dt_start=0.001,
      dt_max=0.01,
    )
    state = init_fn(R, box, neighbor=nbrs)

    for _ in range(100):
      state = apply_fn(state, neighbor=nbrs)
      nbrs = nbrs.update(state.position, box=state.box)

    E_final = float(energy_fn(state.position, box=state.box, neighbor=nbrs))
    self.assertLess(E_final, E_init, 'Energy must decrease with threshold.')
    self.assertGreater(
      float(jnp.linalg.norm(state.box - box)),
      1e-3,
      'Box should have changed.',
    )


if __name__ == '__main__':
  absltest.main()
