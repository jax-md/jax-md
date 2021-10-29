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

from jax.config import config as jax_config
from jax import random
import jax.numpy as np

from jax import jit, grad, vmap
from jax_md import space, quantity, test_util, energy
from jax_md import partition
from jax_md.util import *

from jax import test_util as jtu

jax_config.parse_flags_with_absl()
jax_config.enable_omnistaging()
FLAGS = jax_config.FLAGS

PARTICLE_COUNT = 10
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

NEIGHBOR_LIST_FORMAT = [partition.Dense,
                        partition.Sparse,
                        partition.OrderedSparse]

DTYPES = [f32, f64] if FLAGS.jax_enable_x64 else [f32]
COORDS = ['fractional', 'real']

class QuantityTest(jtu.JaxTestCase):

  def test_canonicalize_mass(self):
    assert quantity.canonicalize_mass(3.0) == 3.0
    assert quantity.canonicalize_mass(f32(3.0)) == f32(3.0)
    assert quantity.canonicalize_mass(f64(3.0)) == f64(3.0)

  @parameterized.named_parameters(jtu.cases_from_list(
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

      return quantity.kinetic_energy(theta * V)

    grad(do_fn)(2.0)

  @parameterized.named_parameters(jtu.cases_from_list(
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

  @parameterized.named_parameters(jtu.cases_from_list(
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': f'_dtype={dtype.__name__}_coordinates={coords}',
          'dtype': dtype,
          'coords': coords
      } for dtype in DTYPES for coords in COORDS))
  def test_pressure_jammed(self, dtype, coords):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic_general(state.box,
                                                       coords == 'fractional')

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)
    pos = getattr(state, coords + '_position')

    tol = 1e-7 if dtype is f64 else 2e-5

    self.assertAllClose(quantity.pressure(E, pos, state.box), state.pressure,
                        atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
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

  @parameterized.named_parameters(jtu.cases_from_list(
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

  @parameterized.named_parameters(jtu.cases_from_list(
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

  @parameterized.named_parameters(jtu.cases_from_list(
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

  @parameterized.named_parameters(jtu.cases_from_list(
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
    if not FLAGS.jax_enable_x64:
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
