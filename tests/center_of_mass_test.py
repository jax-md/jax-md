# Copyright 2026 Google LLC
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

"""Fixed-COM constraints and their finite-system statistical distributions."""

from typing import Any

from absl.testing import absltest, parameterized
from jax import grad, jit, lax, random, vmap
import jax.numpy as jnp
import numpy as np

from jax_md import (
  energy,
  quantity,
  simulate,
  space,
  test_util,
  util,
)

DTYPES = [jnp.float32, jnp.float64] if util.x64_enabled() else [jnp.float32]


class CenterOfMassTest(test_util.JAXMDTestCase):
  def test_constraint_only_changes_momentum(self):
    """The COM constraint leaves positions and even nonzero net forces alone."""
    state = simulate.NVEState(
      position=jnp.arange(6.0).reshape(3, 2),
      momentum=jnp.arange(6.0).reshape(3, 2) + 1,
      force=jnp.ones((3, 2)),
      mass=jnp.arange(1.0, 4.0)[:, None],
      fix_center_of_mass=True,
    )
    constrained = simulate.constrain_center_of_mass(state)
    self.assertAllClose(constrained.position, state.position, atol=0, rtol=0)
    self.assertAllClose(constrained.force, state.force, atol=0, rtol=0)
    self.assertAllClose(jnp.sum(constrained.momentum, axis=0), jnp.zeros(2))
    self.assertAllClose(
      constrained.velocity[1:] - constrained.velocity[:-1],
      state.velocity[1:] - state.velocity[:-1],
    )

  @parameterized.parameters(
    'nve', 'nvt_langevin', 'temp_csvr', 'npt_csvr', 'npt_nose_hoover'
  )
  def test_wrapped_neighbor_list(self, integrator):
    """A close pair across the periodic boundary must survive COM initialization."""
    npt = integrator.startswith('npt')
    fractional = integrator == 'npt_csvr'
    R = jnp.array([[9.99, 1.0, 1.0], [0.98, 1.0, 1.0], [4.03, 4.0, 4.0]])
    if fractional:
      R /= 10
    displacement, shift = space.periodic_general(
      10.0, fractional_coordinates=fractional
    )
    pair = energy.soft_sphere_pair(displacement, sigma=1.0)
    neighbor_fn, E = energy.soft_sphere_neighbor_list(
      displacement,
      10.0,
      sigma=1.0,
      dr_threshold=0,
      fractional_coordinates=fractional,
      disable_cell_list=npt and not fractional,
    )
    neighbor = neighbor_fn.allocate(R)
    kwargs: dict[str, Any] = dict(dt=0.001, fix_center_of_mass=True)
    if integrator != 'nve':
      kwargs['kT'] = 1.0
    if integrator == 'temp_csvr':
      kwargs['tau'] = 0.1
    if npt:
      kwargs['pressure'] = 0.1
    if integrator == 'npt_csvr':
      kwargs.update(tau_p=1.0, tau_t=0.1)
    # Bind the initial list so force inference need not trace its metadata.
    energy_fn = lambda R, neighbor=neighbor, **kw: E(R, neighbor=neighbor, **kw)
    init, step = getattr(simulate, integrator)(energy_fn, shift, **kwargs)
    state = init(
      random.PRNGKey(9),
      R,
      mass=jnp.arange(1.0, 4.0),
      **({'box': 10.0} if npt else {'kT': 1.0}),
    )
    self.assertAllClose(state.position, R, atol=0, rtol=0)
    self.assertAllClose(state.force, -grad(E)(R, neighbor=neighbor), atol=1e-12)
    self.assertGreater(pair(R), 0)

    @jit
    def advance(state, neighbor):
      kwargs = {'box': state.box} if npt else {}
      neighbor = neighbor.update(state.position, **kwargs)
      state = step(state, neighbor=neighbor)
      kwargs = {'box': state.box} if npt else {}
      neighbor = neighbor.update(state.position, **kwargs)
      return state, neighbor

    for _ in range(10):
      state, neighbor = advance(state, neighbor)
      kwargs = {'box': state.box} if npt else {}
      self.assertFalse(neighbor.did_buffer_overflow)
      self.assertAllClose(
        E(state.position, neighbor=neighbor, **kwargs),
        pair(state.position, **kwargs),
        atol=1e-12,
      )
      self.assertAllClose(
        -grad(E)(state.position, neighbor=neighbor, **kwargs),
        -grad(pair)(state.position, **kwargs),
        atol=1e-12,
      )

  @parameterized.product(
    integrator=[
      'nve',
      'nvt_nose_hoover',
      'nvt_langevin',
      'temp_csvr',
      'temp_berendsen',
      'temp_rescale',
      'nvk',
    ],
    dim=[2, 3],
    dtype=DTYPES,
    supplied=[False, True],
  )
  def test_constraint(self, integrator, dim, dtype, supplied):
    """Momentum-based integrators keep COM fixed for unequal masses."""
    R = jnp.arange(4 * dim, dtype=dtype).reshape(4, dim) / 10
    mass = jnp.arange(1, 5, dtype=dtype)
    E = lambda R, **kw: jnp.sum((R - jnp.mean(R, axis=0)) ** 2) / 10
    kwargs: dict[str, Any] = dict(dt=0.001, fix_center_of_mass=True)
    if integrator != 'nve':
      kwargs['kT'] = 1.0
    if integrator in ('temp_csvr', 'temp_berendsen'):
      kwargs['tau'] = 0.1
    if integrator == 'temp_rescale':
      kwargs.update(window=0.1, fraction=0.5)
    init, step = getattr(simulate, integrator)(E, space.free()[1], **kwargs)
    kwargs = {'kT': 1.0} if integrator == 'nve' else {}
    if supplied:
      kwargs['momenta'] = R + 1
    state = init(random.PRNGKey(1), R, mass=mass, **kwargs)
    self.assertAllClose(state.position, R, atol=0, rtol=0)
    initial_center = jnp.sum(mass[:, None] * R, axis=0) / jnp.sum(mass)

    def advance(state, _):
      state = step(state)
      center = jnp.sum(state.mass * state.position, axis=0) / jnp.sum(
        state.mass
      )
      total_p = jnp.sum(state.momentum, axis=0)
      return state, (center, total_p)

    state, (centers, momenta) = jit(
      lambda s: lax.scan(advance, s, None, length=100)
    )(state)
    tol = 2e-5 if dtype == jnp.float32 else 1e-12
    self.assertAllClose(
      centers, jnp.broadcast_to(initial_center, centers.shape), atol=tol
    )
    self.assertAllClose(momenta, jnp.zeros_like(momenta), atol=tol)
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(
      simulate.temperature(state),
      2 * simulate.kinetic_energy(state) / (3 * dim),
    )
    self.assertEqual(state.momentum.dtype, dtype)

  @parameterized.product(
    source=['initialization', 'langevin', 'csvr'], dim=[2, 3]
  )
  def test_maxwell_distribution(self, source, dim):
    """Check the constrained Maxwell covariance and chi-squared kinetic energy."""
    R = jnp.zeros((4, dim))
    mass = jnp.arange(1, 5, dtype=R.dtype)
    E = lambda R, **kw: jnp.sum(R * 0)
    if source == 'csvr':
      init, step = simulate.temp_csvr(
        E, space.free()[1], 10.0, 1.0, 0.1, fix_center_of_mass=True
      )
    else:
      init, step = simulate.nvt_langevin(
        E, space.free()[1], 10.0, 1.0, gamma=10.0, fix_center_of_mass=True
      )

    def sample(key):
      state = init(key, R, mass=mass)
      if source != 'initialization':
        state = step(state)
      return state.momentum, simulate.kinetic_energy(state)

    momentum, kinetic = jit(vmap(sample))(
      random.split(random.PRNGKey(2), 20000)
    )
    dof = 3 * dim
    self.assertAllClose(jnp.mean(kinetic), dof / 2, rtol=0.025)
    self.assertAllClose(jnp.var(kinetic), dof / 2, rtol=0.04)
    expected = jnp.diag(mass) - mass[:, None] * mass[None, :] / jnp.sum(mass)
    self.assertAllClose(
      jnp.cov(momentum[:, :, 0].T), expected, atol=0.07, rtol=0.04
    )

  @parameterized.product(integrator=['csvr', 'nhc'], dtype=DTYPES)
  def test_npt_invariant(self, integrator, dtype):
    """NPT preserves COM momentum while allowing the usual affine box dilation."""
    R = jnp.array(
      [[0, 0, 0], [1.3, 0, 0], [0, 1.3, 0], [0, 0, 1.3]], dtype=dtype
    )
    box = dtype(5.0)
    fractional = integrator == 'csvr'
    displacement, shift = space.periodic_general(
      box, fractional_coordinates=fractional, wrapped=False
    )
    E = energy.lennard_jones_pair(displacement, r_onset=1.8, r_cutoff=2.0)
    if fractional:
      R /= box
    if integrator == 'csvr':
      init, step = simulate.npt_csvr(
        E,
        shift,
        dtype(0.0002),
        0.4,
        1.0,
        1.0,
        float('inf'),
        fix_center_of_mass=True,
      )
      invariant = lambda s: s.conserved_quantity
    else:
      init, step = simulate.npt_nose_hoover(
        E,
        shift,
        dtype(0.0002),
        0.4,
        1.0,
        thermostat_kwargs={'tau': 0.1},
        barostat_kwargs={'tau': 1.0},
        fix_center_of_mass=True,
      )
      invariant = lambda s: simulate.npt_nose_hoover_invariant(E, s, 0.4, 1.0)
    state = init(
      random.PRNGKey(4),
      R,
      box=jnp.eye(3, dtype=dtype) * box,
      mass=jnp.arange(1, 5, dtype=dtype),
    )
    self.assertAllClose(state.position, R, atol=0, rtol=0)
    initial_center = jnp.sum(state.mass * R, axis=0) / jnp.sum(state.mass)
    initial = invariant(state)

    def advance(state, _):
      state = step(state)
      center = jnp.sum(state.mass * state.position, axis=0) / jnp.sum(
        state.mass
      )
      if not fractional:
        center /= (quantity.volume(3, state.box) / box**3) ** (1 / 3)
      return state, (invariant(state), center, jnp.sum(state.momentum, axis=0))

    state, (invariants, centers, momenta) = jit(
      lambda s: lax.scan(advance, s, None, length=1000)
    )(state)
    tol = 0.002 if dtype == jnp.float32 else 0.0001
    self.assertAllClose(
      invariants, jnp.full_like(invariants, initial), atol=tol, rtol=0
    )
    self.assertAllClose(
      centers, jnp.broadcast_to(initial_center, centers.shape), atol=tol, rtol=0
    )
    self.assertAllClose(momenta, jnp.zeros_like(momenta), atol=tol, rtol=0)
    self.assertTrue(jnp.all(jnp.isfinite(invariants)))

  @parameterized.parameters('csvr', 'nhc')
  def test_npt_ideal_gas(self, integrator):
    """For N=4, standard NPT gives p(V) proportional to V**4 * exp(-V)."""
    R = jnp.zeros((4, 3))
    E = lambda R, **kw: jnp.sum(R * 0)
    _, shift = space.periodic_general(
      5 ** (1 / 3), fractional_coordinates=integrator == 'csvr', wrapped=False
    )
    if integrator == 'csvr':
      init, step = simulate.npt_csvr(
        E, shift, 0.004, 1.0, 1.0, 0.5, 0.1, fix_center_of_mass=True
      )
    else:
      init, step = simulate.npt_nose_hoover(
        E,
        shift,
        0.004,
        1.0,
        1.0,
        barostat_kwargs={'tau': 0.5},
        thermostat_kwargs={'tau': 0.1},
        fix_center_of_mass=True,
      )
    states = jit(
      vmap(lambda k: init(k, R, box=5 ** (1 / 3), mass=jnp.arange(1.0, 5.0)))
    )(random.split(random.PRNGKey(5), 32))
    step = vmap(step)
    states = jit(lambda s: lax.fori_loop(0, 5000, lambda i, s: step(s), s))(
      states
    )

    def sample(states, _):
      states = lax.fori_loop(0, 20, lambda i, s: step(s), states)
      volume = vmap(lambda s: quantity.volume(3, s.box))(states)
      return states, (volume, vmap(simulate.kinetic_energy)(states))

    _, (volumes, kinetic) = jit(
      lambda s: lax.scan(sample, s, None, length=1500)
    )(states)
    # Independent trajectories, rather than correlated frames, set the errors.
    for samples, expected_mean, expected_var in (
      (volumes, 5.0, 5.0),
      (kinetic, 4.5, 4.5),
    ):
      means = np.mean(samples, axis=0)
      variances = np.var(samples, axis=0)
      for estimates, expected in (
        (means, expected_mean),
        (variances, expected_var),
      ):
        error = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        self.assertLess(abs(np.mean(estimates) - expected), 6 * error + 0.02)
    # Restore the eliminated COM's mean kinetic pressure, kT / V.
    pressure_means = np.mean((2 * kinetic / 3 + 1) / volumes, axis=0)
    error = np.std(pressure_means, ddof=1) / np.sqrt(len(pressure_means))
    self.assertLess(abs(np.mean(pressure_means) - 1), 6 * error + 0.01)

  def test_nonuniform_langevin_friction_is_rejected(self):
    E = lambda R, **kw: jnp.sum(R * 0)
    init, step = simulate.nvt_langevin(
      E,
      space.free()[1],
      0.01,
      1.0,
      gamma=jnp.ones((4, 1)),
      fix_center_of_mass=True,
    )
    state = init(random.PRNGKey(0), jnp.zeros((4, 3)))
    with self.assertRaisesRegex(ValueError, 'scalar translational friction'):
      step(state)

  @parameterized.parameters('nve', 'nvt_nose_hoover', 'temp_csvr')
  def test_multiple_arrays_share_one_constraint(self, integrator):
    """Species represented by separate arrays must share one global COM."""
    R = {'a': jnp.array([[1.0, 2.0]]), 'b': jnp.array([[3.0, 4.0], [5.0, 6.0]])}
    mass = {'a': jnp.array([1.0]), 'b': jnp.array([2.0, 3.0])}
    E = lambda R, **kw: jnp.sum(R['a'] * 0) + jnp.sum(R['b'] * 0)
    kwargs = dict(dt=0.001, fix_center_of_mass=True)
    if integrator != 'nve':
      kwargs['kT'] = 1.0
    if integrator == 'temp_csvr':
      kwargs['tau'] = 0.1
    init, step = getattr(simulate, integrator)(E, space.free()[1], **kwargs)
    state = init(
      random.PRNGKey(8),
      R,
      mass=mass,
      **({'kT': 1.0} if integrator == 'nve' else {}),
    )
    state = jit(step)(state)
    total_p = jnp.sum(state.momentum['a'], axis=0) + jnp.sum(
      state.momentum['b'], axis=0
    )
    self.assertAllClose(total_p, jnp.zeros(2), atol=1e-6)
    self.assertFalse(jnp.allclose(state.momentum['a'], 0))
    self.assertEqual(quantity.count_dof(R, True), 4)
    self.assertAllClose(
      simulate.temperature(state), simulate.kinetic_energy(state) / 2
    )

  def test_single_particle_constraint_is_rejected(self):
    init, _ = simulate.nve(
      lambda R, **kw: jnp.sum(R * 0), space.free()[1], fix_center_of_mass=True
    )
    with self.assertRaisesRegex(ValueError, 'internal degree of freedom'):
      init(random.PRNGKey(0), jnp.zeros((1, 3)), kT=1.0)


if __name__ == '__main__':
  absltest.main()
