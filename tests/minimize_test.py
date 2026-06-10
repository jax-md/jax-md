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
)
from jax_md.custom_smap import pair_neighbor_list_multi_image

jax.config.parse_flags_with_absl()

PARTICLE_COUNT = 10
OPTIMIZATION_STEPS = 10
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

if jax.config.jax_enable_x64:  # ty: ignore[possibly-unbound-attribute]
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

  def test_precon_fire_descent_armijo_rejects_uphill_step(self):
    R = jnp.array([[1.0]], dtype=f32)
    energy = lambda R, **kwargs: f32(0.5) * jnp.sum(R**2)
    _, shift_fn = space.free()

    init_fn, apply_fn = minimize.precon_fire_descent(
      energy,
      shift_fn,
      dt_start=0.1,
      f_dec=0.5,
      alpha_start=0.1,
      use_armijo=True,
    )
    state = init_fn(R)
    state = state.set(
      velocity=jnp.array([[10.0]], dtype=f32),
      n_pos=jnp.array(3, dtype=jnp.int32),
      initialized=jnp.array(True),
    )

    next_state = apply_fn(state)

    expected_dt = state.dt * 0.5
    expected_velocity = expected_dt * state.force
    expected_position = state.position + expected_dt * expected_velocity

    self.assertAllClose(next_state.position, expected_position)
    self.assertAllClose(next_state.velocity, expected_velocity)
    self.assertAllClose(
      next_state.force, quantity.force(energy)(expected_position)
    )
    self.assertAlmostEqual(float(next_state.dt), float(expected_dt))
    self.assertAlmostEqual(float(next_state.alpha), 0.1, delta=1e-3)
    self.assertAllClose(next_state.n_pos, jnp.array(0, dtype=jnp.int32))

  def test_c1_preconditioner_dot(self):
    displacement_fn, _ = space.free()
    _, preconditioner_dot = minimize.c1_preconditioner(
      displacement_fn,
      r_cut=1.5,
      r_NN=1.0,
      mu=2.0,
      c_stab=0.25,
    )

    R = jnp.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=f32)
    X = jnp.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=f32)
    Y = jnp.array([[13.0, 17.0], [19.0, 23.0], [29.0, 31.0]], dtype=f32)

    weights = jnp.array(
      [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      dtype=f32,
    )
    degree = jnp.sum(weights, axis=1)
    PY = 2.0 * ((degree[:, None] + 0.25) * Y - weights @ Y)
    expected = jnp.sum(X * PY)

    self.assertAllClose(preconditioner_dot(R, X, Y), expected)

  def test_exp_preconditioner_a_zero_matches_c1(self):
    displacement_fn, _ = space.free()
    _, exp_dot = minimize.exp_preconditioner(
      displacement_fn,
      r_cut=1.5,
      r_NN=1.0,
      A=0.0,
      mu=2.0,
      c_stab=0.25,
    )
    _, c1_dot = minimize.c1_preconditioner(
      displacement_fn,
      r_cut=1.5,
      r_NN=1.0,
      mu=2.0,
      c_stab=0.25,
    )

    R = jnp.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=f32)
    X = jnp.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=f32)
    Y = jnp.array([[13.0, 17.0], [19.0, 23.0], [29.0, 31.0]], dtype=f32)

    self.assertAllClose(exp_dot(R, X, Y), c1_dot(R, X, Y))

  def test_exp_preconditioner_standard_neighbor_list(self):
    displacement_fn, _ = space.free()
    neighbor_fn = partition.neighbor_list(
      displacement_fn,
      box=10.0,
      r_cutoff=1.5,
      format=partition.Dense,
    )
    R = jnp.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=f32)
    X = jnp.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=f32)
    Y = jnp.array([[13.0, 17.0], [19.0, 23.0], [29.0, 31.0]], dtype=f32)
    neighbor = neighbor_fn.allocate(R)
    _, preconditioner_dot = minimize.c1_preconditioner(
      displacement_fn,
      r_cut=1.5,
      r_NN=1.0,
      mu=2.0,
      c_stab=0.25,
    )

    weights = jnp.array(
      [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      dtype=f32,
    )
    degree = jnp.sum(weights, axis=1)
    PY = 2.0 * ((degree[:, None] + 0.25) * Y - weights @ Y)
    expected = jnp.sum(X * PY)

    self.assertAllClose(
      preconditioner_dot(R, X, Y, neighbor=neighbor), expected
    )

  def test_exp_preconditioner_ordered_sparse_multi_image(self):
    box = jnp.eye(2, dtype=f32)
    displacement_fn, _ = space.periodic_general(
      box, fractional_coordinates=True
    )
    sparse_neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      r_cutoff=0.6,
      fractional_coordinates=True,
      format=partition.Sparse,
    )
    ordered_neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      r_cutoff=0.6,
      fractional_coordinates=True,
      format=partition.OrderedSparse,
    )
    R = jnp.array([[0.1, 0.1], [0.4, 0.1], [0.8, 0.8]], dtype=f32)
    X = jnp.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=f32)
    Y = jnp.array([[13.0, 17.0], [19.0, 23.0], [29.0, 31.0]], dtype=f32)
    sparse_neighbor = sparse_neighbor_fn.allocate(R)
    ordered_neighbor = ordered_neighbor_fn.allocate(R)
    _, preconditioner_dot = minimize.c1_preconditioner(
      displacement_fn,
      r_cut=0.6,
      r_NN=0.3,
      mu=2.0,
      c_stab=0.25,
    )

    sparse_dot = preconditioner_dot(R, X, Y, neighbor=sparse_neighbor)
    ordered_dot = preconditioner_dot(R, X, Y, neighbor=ordered_neighbor)

    self.assertAllClose(ordered_dot, sparse_dot)

  def test_estimate_exp_mu(self):
    displacement_fn, _ = space.free()
    R = jnp.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.7], [3.0, 1.1]], dtype=f32)
    stiffness = jnp.array([2.0, 5.0], dtype=f32)
    energy_fn = lambda R, **kwargs: f32(0.5) * jnp.sum(stiffness * R**2)
    force_fn = quantity.force(energy_fn)

    mu = minimize.estimate_exp_mu(
      energy_fn,
      displacement_fn,
      R,
      r_cut=1.6,
      r_NN=1.0,
      A=3.0,
      c_stab=0.25,
      min_mu=0.0,
    )

    _, unit_dot = minimize.exp_preconditioner(
      displacement_fn,
      r_cut=1.6,
      r_NN=1.0,
      A=3.0,
      mu=1.0,
      c_stab=0.25,
    )
    amplitude = f32(1e-2)
    L = jnp.max(R, axis=0) - jnp.min(R, axis=0)
    V = amplitude * jnp.sin(R / L)
    expected = jnp.sum((force_fn(R) - force_fn(R + V)) * V) / unit_dot(R, V, V)

    self.assertAllClose(mu, expected)

  def test_precon_fire_descent_matches_ase_trajectory(self):
    if not jax.config.jax_enable_x64:  # ty: ignore[possibly-unbound-attribute]
      self.skipTest('ASE trajectory parity requires x64.')

    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize.precon.fire import PreconFIRE
    from ase.optimize.precon.precon import Exp

    class HarmonicCalculator(Calculator):
      implemented_properties = ['energy', 'free_energy', 'forces']

      def __init__(self, R0, stiffness):
        super().__init__()
        self.R0 = onp.asarray(R0)
        self.stiffness = onp.asarray(stiffness)

      def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
      ):
        super().calculate(atoms, properties, system_changes)
        R = self.atoms.get_positions()
        dR = R - self.R0
        energy_value = 0.5 * onp.sum(self.stiffness * dR**2)
        self.results['energy'] = float(energy_value)
        self.results['free_energy'] = float(energy_value)
        self.results['forces'] = -self.stiffness * dR

    R = jnp.array(
      [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.1, 1.0]],
      dtype=jnp.float64,
    )
    R0 = jnp.array(
      [
        [0.05, -0.02, 0.01],
        [0.95, 0.0, 0.02],
        [0.0, 0.95, -0.01],
        [0.02, 0.0, 0.95],
      ],
      dtype=jnp.float64,
    )
    stiffness = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

    energy_fn = lambda R, **kwargs: (
      f32(0.5) * jnp.sum(stiffness * (R - R0) ** 2)
    )
    displacement_fn, shift_fn = space.free()
    precon, precon_dot = minimize.exp_preconditioner(
      displacement_fn,
      r_cut=1.5,
      r_NN=1.0,
      A=3.0,
      mu=2.0,
      c_stab=0.25,
      solver='dense',
      reference_position=R,
    )
    init_fn, apply_fn = minimize.precon_fire_descent(
      energy_fn,
      shift_fn,
      dt_start=0.01,
      dt_max=0.04,
      max_move=0.2,
      preconditioner=precon,
      preconditioner_dot=precon_dot,
    )
    state = init_fn(R)

    atoms = Atoms(
      ['Ar'] * len(R),
      positions=onp.asarray(R),
      cell=[20.0, 20.0, 20.0],
      pbc=False,
    )
    atoms.calc = HarmonicCalculator(onp.asarray(R0), onp.asarray(stiffness))
    opt = PreconFIRE(
      atoms,
      logfile=None,
      trajectory=None,
      dt=0.01,
      dtmax=0.04,
      maxmove=0.2,
      precon=Exp(
        A=3.0,
        r_cut=1.5,
        r_NN=1.0,
        mu=2.0,
        dim=3,
        c_stab=0.25,
        solver='direct',
        force_stab=True,
      ),
      use_armijo=True,
    )
    opt.initialize()

    for _ in range(12):
      opt.step()
      state = apply_fn(state)

      self.assertAllClose(state.position, atoms.get_positions(), atol=1e-10)
      self.assertAllClose(state.velocity, opt.v, atol=1e-10)
      self.assertAllClose(state.force, atoms.get_forces(), atol=1e-10)
      self.assertAlmostEqual(float(state.dt), float(opt.dt), delta=1e-12)
      self.assertAlmostEqual(float(state.alpha), float(opt.a), delta=1e-12)
      self.assertEqual(int(state.n_pos), int(opt.Nsteps))


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
    self.neighbor_fn, raw_energy_fn = energy.lennard_jones_neighbor_list(
      disp_fn,
      self.box,
      r_cutoff=r_cutoff,
      r_onset=2.0,
      dr_threshold=0.0,
      fractional_coordinates=True,
      neighbor_list_fn=neighbor_list_multi_image,
      pair_neighbor_list_fn=pair_neighbor_list_multi_image,
      format=partition.Sparse,
    )

    def energy_fn(R, box=None, neighbor=None, **kwargs):
      if box is not None and neighbor is not None:
        neighbor = neighbor.set(
          box=box, shifts=neighbor.shifts.astype(box.dtype)
        )
      # The neighbor-list energy function accepts `neighbor` as a keyword,
      # but its declared Callable return type erases parameter names.
      return raw_energy_fn(R, neighbor=neighbor, **kwargs)  # ty: ignore[missing-argument, unknown-argument]

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
    neighbor_fn, raw_energy_fn = energy.lennard_jones_neighbor_list(
      disp_fn,
      box,
      r_cutoff=r_cutoff,
      r_onset=2.0,
      dr_threshold=0.5,
      fractional_coordinates=True,
      neighbor_list_fn=neighbor_list_multi_image,
      pair_neighbor_list_fn=pair_neighbor_list_multi_image,
      format=partition.Sparse,
    )

    def energy_fn(R, box=None, neighbor=None, **kwargs):
      if box is not None and neighbor is not None:
        neighbor = neighbor.set(
          box=box, shifts=neighbor.shifts.astype(box.dtype)
        )
      # The neighbor-list energy function accepts `neighbor` as a keyword,
      # but its declared Callable return type erases parameter names.
      return raw_energy_fn(R, neighbor=neighbor, **kwargs)  # ty: ignore[missing-argument, unknown-argument]

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

  def test_precon_fire_descent_box_reduces_energy_and_changes_box(self):
    dtype = self.R.dtype
    R = jnp.array([[0.15, -0.05], [0.9, 0.2], [0.25, 0.85]], dtype=dtype)
    R0 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    box = jnp.array([[1.3, 0.1], [0.0, 0.9]], dtype=dtype)
    box_target = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)

    def energy_fn(R, box=None, perturbation=None, **kwargs):
      del kwargs
      box_eff = box
      if perturbation is not None:
        box_eff = perturbation @ box
      return f32(0.5) * (
        jnp.sum((R - R0) ** 2) + jnp.sum((box_eff - box_target) ** 2)
      )

    _, shift_fn = space.free()
    init_fn, apply_fn = minimize.precon_fire_descent_box(
      energy_fn,
      shift_fn,
      dt_start=0.01,
      dt_max=0.05,
      max_move=0.2,
    )

    state = init_fn(R, box)
    initial_energy = energy_fn(state.position, box=state.box)

    for _ in range(60):
      state = apply_fn(state)

    final_energy = energy_fn(state.position, box=state.box)
    self.assertLess(final_energy, initial_energy)
    self.assertGreater(float(jnp.linalg.norm(state.box - box)), 1e-4)

  def test_precon_fire_descent_box_armijo_rejects_uphill_step(self):
    dtype = self.R.dtype
    R = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    box = jnp.eye(2, dtype=dtype)
    box_target = f32(0.9) * jnp.eye(2, dtype=dtype)

    def energy_fn(R, box=None, perturbation=None, **kwargs):
      del kwargs
      box_eff = box
      if perturbation is not None:
        box_eff = perturbation @ box
      return f32(0.5) * (jnp.sum(R**2) + jnp.sum((box_eff - box_target) ** 2))

    _, shift_fn = space.free()
    init_fn, apply_fn = minimize.precon_fire_descent_box(
      energy_fn,
      shift_fn,
      dt_start=0.1,
      f_dec=0.5,
      alpha_start=0.1,
      use_armijo=True,
    )
    state = init_fn(R, box)
    state = state.set(
      velocity=-10.0 * state.force,
      box_velocity=-10.0 * state.box_force,
      n_pos=jnp.array(3, dtype=jnp.int32),
      initialized=jnp.array(True),
    )

    next_state = apply_fn(state)

    expected_dt = state.dt * 0.5
    expected_velocity = expected_dt * state.force
    expected_box_velocity = expected_dt * state.box_force
    expected_position = state.position + expected_dt * expected_velocity
    expected_box_position = (
      state.box_position + expected_dt * expected_box_velocity
    )
    expected_box = (
      state.reference_box @ (expected_box_position / state.box_factor).T
    )

    self.assertAllClose(next_state.position, expected_position)
    self.assertAllClose(next_state.velocity, expected_velocity)
    self.assertAllClose(next_state.box_position, expected_box_position)
    self.assertAllClose(next_state.box_velocity, expected_box_velocity)
    self.assertAllClose(next_state.box, expected_box)
    self.assertAlmostEqual(float(next_state.dt), float(expected_dt))
    self.assertAlmostEqual(float(next_state.alpha), 0.1, delta=1e-3)
    self.assertAllClose(next_state.n_pos, jnp.array(0, dtype=jnp.int32))

  def test_precon_fire_descent_box_identity_preconditioner_jit(self):
    dtype = self.R.dtype
    R = jnp.array([[0.2, 0.0], [0.8, 0.25], [0.1, 0.75]], dtype=dtype)
    R0 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    box = jnp.array([[1.2, 0.05], [0.0, 0.95]], dtype=dtype)
    box_target = jnp.eye(2, dtype=dtype)

    def energy_fn(R, box=None, perturbation=None, **kwargs):
      del kwargs
      box_eff = box
      if perturbation is not None:
        box_eff = perturbation @ box
      return f32(0.5) * (
        jnp.sum((R - R0) ** 2) + jnp.sum((box_eff - box_target) ** 2)
      )

    def preconditioner(R, F, **kwargs):
      del R, kwargs
      return F

    def preconditioner_dot(R, X, Y, **kwargs):
      del R, kwargs
      return jnp.sum(X * Y)

    _, shift_fn = space.free()
    init_fn, apply_fn = minimize.precon_fire_descent_box(
      energy_fn,
      shift_fn,
      dt_start=0.01,
      dt_max=0.05,
      max_move=0.2,
      preconditioner=preconditioner,
      preconditioner_dot=preconditioner_dot,
    )

    state = init_fn(R, box)
    initial_energy = energy_fn(state.position, box=state.box)

    @jit
    def five_steps(state):
      for _ in range(5):
        state = apply_fn(state)
      return state

    state = five_steps(state)
    final_energy = energy_fn(state.position, box=state.box)

    self.assertTrue(bool(state.initialized))
    self.assertLess(final_energy, initial_energy)

  def test_precon_fire_descent_box_requires_metric(self):
    energy_fn = lambda R, box=None, **kwargs: jnp.sum(R**2)
    _, shift_fn = space.free()
    preconditioner = lambda R, F, **kwargs: F

    with self.assertRaises(ValueError):
      minimize.precon_fire_descent_box(
        energy_fn, shift_fn, preconditioner=preconditioner
      )

  def test_precon_fire_descent_box_matches_ase_trajectory(self):
    if not getattr(jax.config, 'jax_enable_x64', False):
      self.skipTest('ASE UnitCellFilter trajectory parity requires x64.')

    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.filters import UnitCellFilter
    from ase.optimize.precon.fire import PreconFIRE
    from ase.optimize.precon.precon import Exp
    from ase.stress import full_3x3_to_voigt_6_stress

    class HarmonicBoxCalculator(Calculator):
      implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

      def __init__(self, R0, stiffness, box_target):
        super().__init__()
        self.R0 = onp.asarray(R0)
        self.stiffness = onp.asarray(stiffness)
        self.box_target = onp.asarray(box_target)

      def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
      ):
        super().calculate(atoms, properties, system_changes)
        R = self.atoms.get_positions()
        box = self.atoms.cell.array
        dR = R - self.R0
        dbox = box - self.box_target
        energy_value = 0.5 * (
          onp.sum(self.stiffness * dR**2) + onp.sum(dbox**2)
        )
        self.results['energy'] = float(energy_value)
        self.results['free_energy'] = float(energy_value)
        self.results['forces'] = -self.stiffness * dR
        volume = self.atoms.get_volume()
        stress_full = box @ dbox.T / volume
        self.results['stress'] = full_3x3_to_voigt_6_stress(stress_full)

    R = jnp.array(
      [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.1, 1.0, 0.0]],
      dtype=jnp.float64,
    )
    R0 = jnp.array(
      [[0.05, -0.02, 0.01], [0.95, 0.0, 0.02], [0.0, 0.95, -0.01]],
      dtype=jnp.float64,
    )
    stiffness = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    box = jnp.diag(jnp.array([4.0, 4.2, 4.4], dtype=jnp.float64))
    box_target = jnp.diag(jnp.array([3.95, 4.1, 4.3], dtype=jnp.float64))

    def energy_fn(R, box=None, perturbation=None, **kwargs):
      del kwargs
      box_eff = box
      if perturbation is not None:
        box_eff = perturbation @ box
      return f32(0.5) * (
        jnp.sum(stiffness * (R - R0) ** 2)
        + jnp.sum((box_eff - box_target) ** 2)
      )

    displacement_fn, shift_fn = space.free()
    precon, precon_dot = minimize.exp_preconditioner(
      displacement_fn,
      r_cut=1.5,
      r_NN=1.0,
      A=3.0,
      mu=2.0,
      c_stab=0.25,
      solver='dense',
      reference_position=R,
    )
    init_fn, apply_fn = minimize.precon_fire_descent_box(
      energy_fn,
      shift_fn,
      dt_start=0.01,
      dt_max=0.04,
      max_move=0.2,
      preconditioner=precon,
      preconditioner_dot=precon_dot,
      cell_preconditioner=1.0,
    )
    state = init_fn(R, box, box_factor=jnp.asarray(float(len(R))))

    atoms = Atoms(
      ['Ar'] * len(R),
      positions=onp.asarray(R),
      cell=onp.asarray(box),
      pbc=True,
    )
    atoms.calc = HarmonicBoxCalculator(
      onp.asarray(R0), onp.asarray(stiffness), onp.asarray(box_target)
    )
    ucf = UnitCellFilter(atoms, cell_factor=float(len(R)))
    opt = PreconFIRE(
      ucf,
      logfile=None,
      trajectory=None,
      dt=0.01,
      dtmax=0.04,
      maxmove=0.2,
      precon=Exp(
        A=3.0,
        r_cut=1.5,
        r_NN=1.0,
        mu=2.0,
        mu_c=1.0,
        dim=3,
        c_stab=0.25,
        solver='direct',
        force_stab=True,
      ),
      use_armijo=True,
    )
    opt.initialize()

    for _ in range(12):
      opt.step()
      state = apply_fn(state)

      self.assertAllClose(
        jnp.vstack([state.position, state.box_position]),
        ucf.get_positions(),
        atol=1e-10,
      )
      self.assertAllClose(
        jnp.vstack([state.force, state.box_force]),
        ucf.get_forces(),
        atol=1e-10,
      )
      opt_velocity = opt.v
      assert opt_velocity is not None
      self.assertAllClose(state.velocity, opt_velocity[: len(R)], atol=1e-10)
      self.assertAllClose(
        state.box_velocity, opt_velocity[len(R) :], atol=1e-10
      )
      self.assertAllClose(state.box, atoms.cell.array, atol=1e-10)
      self.assertAlmostEqual(float(state.dt), float(opt.dt), delta=1e-12)
      self.assertAlmostEqual(float(state.alpha), float(opt.a), delta=1e-12)
      self.assertEqual(int(state.n_pos), int(opt.Nsteps))


if __name__ == '__main__':
  absltest.main()
