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

"""Tests for google3.third_party.py.jax_md.energy."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
import jax.numpy as np

from jax import jit, grad, vmap
from jax_md import space
from jax_md import quantity
from jax_md import test_util
from jax_md import energy
from jax_md import partition
from jax_md import smap
from jax_md.util import *


jax.config.parse_flags_with_absl()

PARTICLE_COUNT = 10
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

NEIGHBOR_LIST_FORMAT = [partition.Dense,
                        partition.Sparse,
                        partition.OrderedSparse]

DTYPES = [f32, f64] if jax.config.jax_enable_x64 else [f32]
COORDS = ['fractional', 'real']


class QuantityTest(test_util.JAXMDTestCase):

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}'.format(dim),
          'spatial_dimension': dim,
      } for dim in SPATIAL_DIMENSION))
  def test_grad_kinetic_energy(self, spatial_dimension):
    key = random.PRNGKey(0)

    @jit
    def do_fn(theta):
      key = random.PRNGKey(0)
      V = random.normal(key, (PARTICLE_COUNT, spatial_dimension), dtype=f32)

      return quantity.kinetic_energy(velocity=theta * V)

    grad(do_fn)(2.0)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype,
      } for dtype in DTYPES))
  def test_cosine_angles(self, dtype):
    displacement, _ = space.free()
    displacement = space.map_product(displacement)
    R = np.array(
        [[0, 0],
         [0, 1],
         [1, 1]], dtype=dtype)
    dR = displacement(R, R)
    cangles = quantity.cosine_angles(dR)
    c45 = 1 / np.sqrt(dtype(2))
    true_cangles = np.array(
        [[[0, 0, 0],
          [0, 1, c45],
          [0, c45, 1]],
         [[1, 0, 0],
          [0, 0, 0],
          [0, 0, 1]],
         [[1, c45, 0],
          [c45, 1, 0],
          [0, 0, 0]]], dtype=dtype)
    tol = 3e-7
    self.assertAllClose(cangles, true_cangles, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype,
      } for dtype in DTYPES))
  def test_cosine_angles_neighbors(self, dtype):
    displacement, _ = space.free()
    displacement = vmap(vmap(displacement, (None, 0)), 0)

    R = np.array(
        [[0, 0],
         [0, 1],
         [1, 1]], dtype=dtype)
    R_neigh = np.array(
        [[[0, 1], [1, 1]],
         [[0, 0], [0, 0]],
         [[0, 0], [0, 0]]], dtype=dtype)

    dR = displacement(R, R_neigh)

    cangles = quantity.cosine_angles(dR)
    c45 = 1 / np.sqrt(2)
    true_cangles = np.array(
        [[[1, c45], [c45, 1]],
         [[1, 1], [1, 1]],
         [[1, 1], [1, 1]]], dtype=dtype)
    tol = 3e-7
    self.assertAllClose(cangles, true_cangles, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}',
          'dtype': dtype
      } for dtype in DTYPES))
  def test_pressure_jammed_periodic(self, dtype):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic(jnp.diag(state.box))

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)
    pos = state.real_position

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(quantity.pressure(E, pos, state.box), state.pressure,
                        atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}_coordinates={coords}',
          'dtype': dtype,
          'coords': coords
      } for dtype in DTYPES for coords in COORDS))
  def test_pressure_jammed_periodic_general(self, dtype, coords):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic_general(state.box,
                                                       coords == 'fractional')
    print(state.pressure)
    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)
    pos = getattr(state, coords + '_position')

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(quantity.pressure(E, pos, state.box), state.pressure,
                        atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dim={dim}_dtype={dtype.__name__}',
          'dim': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in DTYPES))
  def test_pressure_non_minimized_free(self, dim, dtype):
    key = random.PRNGKey(0)
    N = 64

    box = quantity.box_size_at_number_density(N, 0.8, dim)
    displacement_fn, _ = space.free()

    pos = random.uniform(key, (N, dim)) * box

    energy_fn = energy.soft_sphere_pair(displacement_fn)

    def exact_stress(R):
      dR = space.map_product(displacement_fn)(R, R)
      dr = space.distance(dR)
      g = jnp.vectorize(grad(energy.soft_sphere), signature='()->()')
      V = quantity.volume(dim, box)
      dUdr = 0.5 * g(dr)[:, :, None, None]
      dr = (dr + jnp.eye(N))[:, :, None, None]
      return jnp.sum(dUdr * dR[:, :, None, :] * dR[:, :, :, None] / (V * dr),
                     axis=(0, 1))

    exact_pressure = -1 / dim * jnp.trace(exact_stress(pos))
    ad_pressure = quantity.pressure(energy_fn, pos, box)

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(exact_pressure, ad_pressure, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dim={dim}_dtype={dtype.__name__}',
          'dim': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in DTYPES))
  def test_pressure_non_minimized_periodic(self, dim, dtype):
    key = random.PRNGKey(0)
    N = 64

    box = quantity.box_size_at_number_density(N, 0.8, dim)
    displacement_fn, _ = space.periodic(box)

    pos = random.uniform(key, (N, dim)) * box

    energy_fn = energy.soft_sphere_pair(displacement_fn)

    def exact_stress(R):
      dR = space.map_product(displacement_fn)(R, R)
      dr = space.distance(dR)
      g = jnp.vectorize(grad(energy.soft_sphere), signature='()->()')
      V = quantity.volume(dim, box)
      dUdr = 0.5 * g(dr)[:, :, None, None]
      dr = (dr + jnp.eye(N))[:, :, None, None]
      return jnp.sum(dUdr * dR[:, :, None, :] * dR[:, :, :, None] / (V * dr),
                     axis=(0, 1))

    exact_pressure = -1 / dim * jnp.trace(exact_stress(pos))
    ad_pressure = quantity.pressure(energy_fn, pos, box)

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(exact_pressure, ad_pressure, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'dim={dim}_dtype={dtype.__name__}'
                            f'_coordinates={coords}'),
          'dim': dim,
          'dtype': dtype,
          'coords': coords
      } for dim in SPATIAL_DIMENSION for dtype in DTYPES for coords in COORDS))
  def test_pressure_non_minimized_periodic_general(self, dim, dtype, coords):
    key = random.PRNGKey(0)
    N = 64

    box = quantity.box_size_at_number_density(N, 0.8, dim)
    displacement_fn, _ = space.periodic_general(box, coords == 'fractional')

    pos = random.uniform(key, (N, dim))
    pos = pos if coords == 'fractional' else pos * box

    energy_fn = energy.soft_sphere_pair(displacement_fn)

    def exact_stress(R):
      dR = space.map_product(displacement_fn)(R, R)
      dr = space.distance(dR)
      g = jnp.vectorize(grad(energy.soft_sphere), signature='()->()')
      V = quantity.volume(dim, box)
      dUdr = 0.5 * g(dr)[:, :, None, None]
      dr = (dr + jnp.eye(N))[:, :, None, None]
      return jnp.sum(dUdr * dR[:, :, None, :] * dR[:, :, :, None] / (V * dr),
                     axis=(0, 1))

    exact_pressure = -1 / dim * jnp.trace(exact_stress(pos))
    ad_pressure = quantity.pressure(energy_fn, pos, box)

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(exact_pressure, ad_pressure, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dim={dim}_dtype={dtype.__name__}',
          'dim': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in DTYPES))
  def test_stress_non_minimized_free(self, dim, dtype):
    key = random.PRNGKey(0)
    N = 64

    box = quantity.box_size_at_number_density(N, 0.8, dim)
    displacement_fn, _ = space.free()

    pos = random.uniform(key, (N, dim)) * box

    energy_fn = energy.soft_sphere_pair(displacement_fn)

    def exact_stress(R):
      dR = space.map_product(displacement_fn)(R, R)
      dr = space.distance(dR)
      g = jnp.vectorize(grad(energy.soft_sphere), signature='()->()')
      V = quantity.volume(dim, box)
      dUdr = 0.5 * g(dr)[:, :, None, None]
      dr = (dr + jnp.eye(N))[:, :, None, None]
      return -jnp.sum(dUdr * dR[:, :, None, :] * dR[:, :, :, None] / (V * dr),
                      axis=(0, 1))

    exact_stress = exact_stress(pos)
    ad_stress = quantity.stress(energy_fn, pos, box)

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(exact_stress, ad_stress, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dim={dim}_dtype={dtype.__name__}',
          'dim': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in DTYPES))
  def test_stress_non_minimized_periodic(self, dim, dtype):
    key = random.PRNGKey(0)
    N = 64

    box = quantity.box_size_at_number_density(N, 0.8, dim)
    displacement_fn, _ = space.periodic(box)

    pos = random.uniform(key, (N, dim)) * box

    energy_fn = energy.soft_sphere_pair(displacement_fn)

    def exact_stress(R):
      dR = space.map_product(displacement_fn)(R, R)
      dr = space.distance(dR)
      g = jnp.vectorize(grad(energy.soft_sphere), signature='()->()')
      V = quantity.volume(dim, box)
      dUdr = 0.5 * g(dr)[:, :, None, None]
      dr = (dr + jnp.eye(N))[:, :, None, None]
      return -jnp.sum(dUdr * dR[:, :, None, :] * dR[:, :, :, None] / (V * dr),
                      axis=(0, 1))

    exact_stress = exact_stress(pos)
    ad_stress = quantity.stress(energy_fn, pos, box)

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(exact_stress, ad_stress, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'dim={dim}_dtype={dtype.__name__}_'
                            f'coords={coords}'),
          'dim': dim,
          'dtype': dtype,
          'coords': coords
      } for dim in SPATIAL_DIMENSION for dtype in DTYPES for coords in COORDS))
  def test_stress_non_minimized_periodic_general(self, dim, dtype, coords):
    key = random.PRNGKey(0)
    N = 64

    box = quantity.box_size_at_number_density(N, 0.8, dim)
    displacement_fn, _ = space.periodic_general(box, coords == 'fractional')

    pos = random.uniform(key, (N, dim))
    pos = pos if coords == 'fractional' else pos * box

    energy_fn = energy.soft_sphere_pair(displacement_fn)

    def exact_stress(R):
      dR = space.map_product(displacement_fn)(R, R)
      dr = space.distance(dR)
      g = jnp.vectorize(grad(energy.soft_sphere), signature='()->()')
      V = quantity.volume(dim, box)
      dUdr = 0.5 * g(dr)[:, :, None, None]
      dr = (dr + jnp.eye(N))[:, :, None, None]
      return -jnp.sum(dUdr * dR[:, :, None, :] * dR[:, :, :, None] / (V * dr),
                      axis=(0, 1))

    exact_stress = exact_stress(pos)
    ad_stress = quantity.stress(energy_fn, pos, box)

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(exact_stress, ad_stress, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dim={dim}_dtype={dtype.__name__}_',
          'dim': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPES))
  def test_stress_lammps_periodic_general(self, dim, dtype):
    key = random.PRNGKey(0)
    N = 64

    (box, R, V), (E, C) = test_util.load_lammps_stress_data(dtype)

    displacement_fn, _ = space.periodic_general(box)
    energy_fn = smap.pair(
      lambda dr, **kwargs: jnp.where(dr < f32(2.5),
                                     energy.lennard_jones(dr),
                                     f32(0.0)),
      space.canonicalize_displacement_or_metric(displacement_fn))

    ad_stress = quantity.stress(energy_fn, R, box, velocity=V)

    tol = 5e-5

    self.assertAllClose(energy_fn(R) / len(R), E, atol=tol, rtol=tol)
    self.assertAllClose(C, ad_stress, atol=tol, rtol=tol)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype,
      } for dtype in DTYPES))
  def test_pair_correlation(self, dtype):
    displacement = lambda Ra, Rb, **kwargs: Ra - Rb
    R = np.array(
        [[1, 0],
         [0, 0],
         [0, 1]], dtype=dtype)
    rs = np.linspace(0, 2, 60, dtype=dtype)
    g = quantity.pair_correlation(displacement, rs, f32(0.1))
    gs = g(R)
    gs = np.mean(gs, axis=0)
    assert np.argmax(gs) == np.argmin((rs - 1.) ** 2)
    assert gs.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype,
      } for dtype in DTYPES))
  def test_pair_correlation_species(self, dtype):
    displacement = lambda Ra, Rb, **kwargs: Ra - Rb
    R = np.array(
        [[1, 0],
         [0, 0],
         [10, 1],
         [10, 3]], dtype=dtype)
    species = np.array([0, 0, 1, 1])
    rs = np.linspace(0, 2, 60, dtype=dtype)
    g = quantity.pair_correlation(displacement, rs, f32(0.1), species)
    g_0, g_1 = g(R)
    g_0 = np.mean(g_0, axis=0)
    g_1 = np.mean(g_1, axis=0)
    self.assertAllClose(np.argmax(g_0), np.argmin((rs - 1.) ** 2))
    self.assertAllClose(np.argmax(g_1), np.argmin((rs - 2.) ** 2))
    assert g_0.dtype == dtype
    assert g_1.dtype == dtype

  @parameterized.named_parameters(test_util.cases_from_list(
      {
        'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                          f'_format={str(format).split(".")[-1]}'),
        'dim': dim,
        'dtype': dtype,
        'format': format
      } for dim in SPATIAL_DIMENSION
    for dtype in DTYPES
    for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_correlation_neighbor_list_species(self, dim, dtype, format):
    if format is partition.OrderedSparse:
      self.skipTest('OrderedSparse not supported for pair correlation '
                    'function.')

    N = 100
    L = 10.
    displacement, _ = space.periodic(L)
    R = random.uniform(random.PRNGKey(0), (N, dim), dtype=dtype)
    species = np.where(np.arange(N) < N // 2, 0, 1)
    rs = np.linspace(0, 2, 60, dtype=dtype)
    g = quantity.pair_correlation(displacement, rs, f32(0.1), species)
    nbr_fn, g_neigh = quantity.pair_correlation_neighbor_list(displacement,
                                                              L,
                                                              rs,
                                                              f32(0.1),
                                                              species,
                                                              format=format)
    nbrs = nbr_fn.allocate(R)

    g_0, g_1 = g(R)
    g_0 = np.mean(g_0, axis=0)
    g_1 = np.mean(g_1, axis=0)

    g_0_neigh, g_1_neigh = g_neigh(R, neighbor=nbrs)
    g_0_neigh = np.mean(g_0_neigh, axis=0)
    g_1_neigh = np.mean(g_1_neigh, axis=0)
    self.assertAllClose(g_0, g_0_neigh)
    self.assertAllClose(g_1, g_1_neigh)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
        'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'
                          f'_format={str(format).split(".")[-1]}'),
        'dim': dim,
        'dtype': dtype,
        'format': format
      } for dim in SPATIAL_DIMENSION
    for dtype in DTYPES
    for format in NEIGHBOR_LIST_FORMAT))
  def test_pair_correlation_neighbor_list(self, dim, dtype, format):
    if format is partition.OrderedSparse:
      self.skipTest('OrderedSparse not supported for pair correlation '
                    'function.')
    N = 100
    L = 10.
    displacement, _ = space.periodic(L)
    R = random.uniform(random.PRNGKey(0), (N, dim), dtype=dtype)
    rs = np.linspace(0, 2, 60, dtype=dtype)
    g = quantity.pair_correlation(displacement, rs, f32(0.1))
    nbr_fn, g_neigh = quantity.pair_correlation_neighbor_list(displacement,
                                                              L,
                                                              rs,
                                                              f32(0.1),
                                                              format=format)
    nbrs = nbr_fn.allocate(R)

    g_0 = g(R)
    g_0 = np.mean(g_0, axis=0)

    g_0_neigh = g_neigh(R, neighbor=nbrs)
    g_0_neigh = np.mean(g_0_neigh, axis=0)

    self.assertAllClose(g_0, g_0_neigh)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
        'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'),
        'dim': dim,
        'dtype': dtype
      } for dim in SPATIAL_DIMENSION
    for dtype in DTYPES))
  def test_pair_correlation_average(self, dim, dtype):
    if format is partition.OrderedSparse:
      self.skipTest('OrderedSparse not supported for pair correlation '
                    'function.')
    N = 100
    L = 10.
    displacement, _ = space.periodic(L)
    R = random.uniform(random.PRNGKey(0), (N, dim), dtype=dtype)
    rs = np.linspace(0, 2, 60, dtype=dtype)
    g = quantity.pair_correlation(displacement, rs, f32(0.1))
    g_0 = g(R)
    g_a = quantity.average_pair_correlation_results(g_0)
    g_0 = np.mean(g_0, axis=0)
    self.assertAllClose(g_0, g_a)

  @parameterized.named_parameters(test_util.cases_from_list(
      {
        'testcase_name': (f'_dim={dim}_dtype={dtype.__name__}'),
        'dim': dim,
        'dtype': dtype,
      } for dim in SPATIAL_DIMENSION
    for dtype in DTYPES))
  def test_pair_correlation_agerage_species(self, dim, dtype):
    if format is partition.OrderedSparse:
      self.skipTest('OrderedSparse not supported for pair correlation '
                    'function.')

    N = 100
    L = 10.
    displacement, _ = space.periodic(L)
    R = random.uniform(random.PRNGKey(0), (N, dim), dtype=dtype)
    species = np.where(np.arange(N) < N // 2, 0, 1)
    rs = np.linspace(0, 2, 60, dtype=dtype)
    g = quantity.pair_correlation(displacement, rs, f32(0.1), species)

    g_results = g(R)
    g_average = quantity.average_pair_correlation_results(g_results, species)

    gg = quantity.pair_correlation(displacement, rs, f32(0.1))
    g_00 = gg(R[species==0])
    g_11 = gg(R[species==1])

    g_00 = np.mean(g_00, axis=0)
    g_11 = np.mean(g_11, axis=0)

    self.assertAllClose(g_00, g_average[0][0])
    self.assertAllClose(g_11, g_average[1][1])

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_window={window}',
          'spatial_dim': dim,
          'dtype': dtype,
          'window': window
      } for dim in SPATIAL_DIMENSION
        for dtype in DTYPES
        for window in [10, 20]))
  def test_phop(self, spatial_dim, dtype, window):
    Ra = np.ones((1, spatial_dim), dtype=dtype)
    Rb = 2. * np.ones((1, spatial_dim), dtype=dtype)
    half_window = window // 2

    displacement_fn = lambda Ra, Rb: Ra - Rb

    init_fn, push_fn = quantity.phop(displacement_fn, window)

    phop_state = init_fn(Ra)

    # E_A[R] = 1
    # E_B[R] = 1 + i / half_window
    # E_A[(R - E_B[R]) ** 2] = (i / half_window) ** 2
    # E_B[(R - E_A[R]) ** 2] = (i / half_window)
    # phop = (i / half_window) ** (3 / 2)

    for i in range(half_window):
      phop_state = push_fn(phop_state, Rb)
      self.assertAllClose(
        phop_state.phop,
        np.array([(float(i) / half_window) ** (3. / 2)], dtype=dtype))


  def test_maybe_downcast(self):
    if not jax.config.jax_enable_x64:
      self.skipTest('Maybe downcast only works for float32 mode.')

    x = np.array([1, 2, 3], np.float64)
    x = maybe_downcast(x)
    self.assertEqual(x.dtype, np.float64)

  def test_clipped_force(self):
    N = 10
    dim = 3
    def U(r):
      return np.sum(1 / np.linalg.norm(r, axis=-1) ** 2)
    force_fn = quantity.clipped_force(U, 1.5)
    R = random.normal(random.PRNGKey(0), (N, dim))
    self.assertTrue(np.all(np.linalg.norm(force_fn(R), axis=-1) <= 1.5))

if __name__ == '__main__':
  absltest.main()
