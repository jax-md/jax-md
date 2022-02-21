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

"""Tests for jax_md.energy."""

from absl.testing import absltest
from absl.testing import parameterized

import os
from jax.config import config as jax_config
from jax import random
from jax import jit, vmap
import optax
import jax.numpy as np

import numpy as onp

from jax import grad
from jax_md import space
from jax_md.util import *
from jax_md import test_util
from jax_md import quantity

from jax import test_util as jtu

from jax_md import energy
from jax_md import partition
from jax_md.interpolate import spline

jax_config.parse_flags_with_absl()
jax_config.enable_omnistaging()
FLAGS = jax_config.FLAGS

PARTICLE_COUNT = 100
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]
UNIT_CELL_SIZE = [7, 8]

SOFT_SPHERE_ALPHA = [2.0, 2.5, 3.0]
N_TYPES_TO_TEST = [1, 2]

if FLAGS.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]

NEIGHBOR_LIST_FORMAT = [partition.Dense,
                        partition.Sparse,
                        partition.OrderedSparse]

test_util.update_test_tolerance(2e-5, 1e-10)


def lattice_repeater(small_cell_pos, latvec, no_rep):
  dtype = small_cell_pos.dtype
  pos = onp.copy(small_cell_pos).tolist()
  for atom in small_cell_pos:
    for i in range(no_rep):
      for j in range(no_rep):
        for k in range(no_rep):
          if not i == j == k == 0:
            repeated_atom = atom + latvec[0] * i + latvec[1] * j + latvec[2] * k
            pos.append(onp.array(repeated_atom).tolist())
  return np.array(pos, dtype), f32(latvec*no_rep)


def make_eam_test_splines():
  cutoff = 6.28721

  num_spline_points = 21
  dr = np.arange(0, num_spline_points) * (cutoff / num_spline_points)
  dr = np.array(dr, f32)

  drho = np.arange(0, 2, 2. / num_spline_points)
  drho = np.array(drho, f32)

  density_data = np.array([2.78589606e-01, 2.02694937e-01, 1.45334053e-01,
                           1.06069912e-01, 8.42517168e-02, 7.65140344e-02,
                           7.76263116e-02, 8.23214224e-02, 8.53322309e-02,
                           8.13915861e-02, 6.59095390e-02, 4.28915711e-02,
                           2.27910928e-02, 1.13713167e-02, 6.05020311e-03,
                           3.65836583e-03, 2.60587564e-03, 2.06750708e-03,
                           1.48749693e-03, 7.40019174e-04, 6.21225205e-05],
                          np.float64)

  embedding_data = np.array([1.04222211e-10, -1.04142633e+00, -1.60359806e+00,
                             -1.89287637e+00, -2.09490167e+00, -2.26456628e+00,
                             -2.40590322e+00, -2.52245359e+00, -2.61385603e+00,
                             -2.67744693e+00, -2.71053295e+00, -2.71110418e+00,
                             -2.69287013e+00, -2.68464527e+00, -2.69204083e+00,
                             -2.68976209e+00, -2.66001244e+00, -2.60122024e+00,
                             -2.51338548e+00, -2.39650817e+00, -2.25058831e+00],
                            np.float64)

  pairwise_data = np.array([6.27032242e+01, 3.49638589e+01, 1.79007014e+01,
                            8.69001383e+00, 4.51545250e+00, 2.83260884e+00,
                            1.93216616e+00, 1.06795515e+00, 3.37740836e-01,
                            1.61087890e-02, -6.20816372e-02, -6.51314297e-02,
                            -5.35210341e-02, -5.20950200e-02, -5.51709524e-02,
                            -4.89093894e-02, -3.28051688e-02, -1.13738785e-02,
                            2.33833655e-03, 4.19132033e-03, 1.68600692e-04],
                           np.float64)

  charge_fn = spline(density_data, dr[1] - dr[0])
  embedding_fn = spline(embedding_data, drho[1] - drho[0])
  pairwise_fn = spline(pairwise_data, dr[1] - dr[0])
  return charge_fn, embedding_fn, pairwise_fn, cutoff


def lattice(R_unit_cell, copies, lattice_vectors):
  # Given a cell of positions, tile it.
  lattice_vectors = onp.array(lattice_vectors, f32)

  N, d = R_unit_cell.shape
  if isinstance(copies, int):
    copies = (copies,) * d

  if lattice_vectors.ndim == 0 or lattice_vectors.ndim == 1:
    cartesian = True
    L = onp.eye(d) * lattice_vectors[onp.newaxis, ...]
  elif lattice_vectors.ndim == 2:
    assert lattice_vectors.shape[0] == lattice_vectors.shape[1]
    cartesian = False
    L = onp.eye(d) / onp.array(copies)[onp.newaxis, ...]
    R_unit_cell /= onp.array(copies)[onp.newaxis, ...]
  else:
    raise ValueError()

  Rs = []
  for indices in onp.ndindex(copies):
    dR = 0.
    for idx, i in enumerate(indices):
      dR += i * L[idx]
    R = R_unit_cell + dR[onp.newaxis, :]
    Rs += [R]
  return np.array(onp.concatenate(Rs))


@jtu.with_config(jax_numpy_rank_promotion="allow")
class EnergyTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION
    for dtype in POSITION_DTYPE))
  def test_simple_spring(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    disp, _ = space.free()
    if spatial_dimension == 2:
      R = np.array([[0., 0.], [1., 1.]], dtype=dtype)
      dist = np.sqrt(2.)
    elif spatial_dimension == 3:
      R = np.array([[0., 0., 0.], [1., 1., 1.]], dtype=dtype)
      dist = np.sqrt(3.)
    bonds = np.array([[0, 1]], np.int32)
    for _ in range(STOCHASTIC_SAMPLES):
      key, l_key, a_key = random.split(key, 3)
      length = random.uniform(key, (), minval=0.1, maxval=3.0, dtype=dtype)
      alpha = random.uniform(key, (), minval=2., maxval=4., dtype=dtype)
      E = energy.simple_spring_bond(disp, bonds, length=length, alpha=alpha)
      E_exact = dtype((dist - length) ** alpha / alpha)
      self.assertAllClose(E(R), E_exact)

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_alpha={}_dtype={}'.format(
            dim, alpha, dtype.__name__),
          'spatial_dimension': dim,
          'alpha': alpha,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION
    for alpha in SOFT_SPHERE_ALPHA
    for dtype in POSITION_DTYPE))
  def test_soft_sphere(self, spatial_dimension, alpha, dtype):
    key = random.PRNGKey(0)
    alpha = f32(alpha)
    for _ in range(STOCHASTIC_SAMPLES):
      key, split_sigma, split_epsilon = random.split(key, 3)
      sigma = np.array(random.uniform(
          split_sigma, (1,), minval=0.0, maxval=3.0)[0], dtype=dtype)
      epsilon = np.array(
        random.uniform(split_epsilon, (1,), minval=0.0, maxval=4.0)[0],
        dtype=dtype)
      self.assertAllClose(
          energy.soft_sphere(
            dtype(0), sigma, epsilon, alpha), epsilon / alpha)
      self.assertAllClose(
        energy.soft_sphere(dtype(sigma), sigma, epsilon, alpha),
        np.array(0.0, dtype=dtype))
      self.assertAllClose(
        grad(energy.soft_sphere)(dtype(2 * sigma), sigma, epsilon, alpha),
        np.array(0.0, dtype=dtype))

      if alpha > 2.0:
        grad_energy = grad(energy.soft_sphere)
        g = grad_energy(dtype(sigma), sigma, epsilon, alpha)
        self.assertAllClose(g, np.array(0, dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_lennard_jones(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_sigma, split_epsilon = random.split(key, 3)
      sigma = dtype(random.uniform(
          split_sigma, (1,), minval=0.5, maxval=3.0)[0])
      epsilon = dtype(random.uniform(
          split_epsilon, (1,), minval=0.0, maxval=4.0)[0])
      dr = dtype(sigma * 2 ** (1.0 / 6.0))
      self.assertAllClose(
        energy.lennard_jones(dr, sigma, epsilon),
        np.array(-epsilon, dtype=dtype))
      g = grad(energy.lennard_jones)(dr, sigma, epsilon)
      self.assertAllClose(g, np.array(0, dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': 'dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_gupta(self, dtype):
      displacement, shift = space.free()
      pos = np.array([[0, 0, 0], [0, 0, 2.9], [0, 2.9, 2.9]])
      energy_fn = energy.gupta_gold55(displacement)
      self.assertAllClose(-5.4632421255957135, energy_fn(pos))


  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': 'dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_bks(self, dtype):
      LATCON = 3.5660930663857577e+01
      displacement, shift = space.periodic(LATCON)
      dist_fun = space.metric(displacement)
      species = np.tile(np.array([0, 1, 1]), 1000)
      R_f = test_util.load_silica_data()
      energy_fn = energy.bks_silica_pair(dist_fun, species=species)
      self.assertAllClose(-857939.528386092, energy_fn(R_f))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': f'dtype={dtype.__name__}_format={format}',
          'dtype': dtype,
          'format': format
      } for dtype in POSITION_DTYPE for format in NEIGHBOR_LIST_FORMAT))
  def test_bks_neighbor_list(self, dtype, format):
    LATCON = 3.5660930663857577e+01
    displacement, shift = space.periodic(LATCON)
    dist_fun = space.metric(displacement)
    species = np.tile(np.array([0, 1, 1]), 1000)
    R_f = test_util.load_silica_data()
    neighbor_fn, energy_nei = energy.bks_silica_neighbor_list(
      dist_fun, LATCON, species=species, format=format)
    nbrs = neighbor_fn.allocate(R_f)
    self.assertAllClose(-857939.528386092, energy_nei(R_f, nbrs))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': (f'dtype={dtype.__name__}_'
                            f'num_repetitions={num_repetitions}'),
          'dtype': dtype,
          'num_repetitions': num_repetitions,
      } for dtype in POSITION_DTYPE for num_repetitions in [2, 3]))
  def test_stillinger_weber(self, dtype, num_repetitions):
    lattice_vectors = lattice_vectors = np.array([[0, .5, .5],
                                                  [.5, 0, .5],
                                                  [.5, .5, 0]]) * 5.428
    positions = np.array([[0,0,0], [0.25, 0.25, 0.25]])
    positions = lattice(positions, num_repetitions, lattice_vectors)
    lattice_vectors *= num_repetitions
    displacement, shift = space.periodic_general(lattice_vectors)
    energy_fn = jit(energy.stillinger_weber(displacement))
    N = positions.shape[0]
    self.assertAllClose(energy_fn(positions) / N, -4.336503155764325)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
        'testcase_name': (f'dtype={dtype.__name__}'
                          f'_num_repetitions={num_repetitions}'
                          f'_format={str(format).split(".")[-1]}'),
        'dtype': dtype,
        'num_repetitions': num_repetitions,
        'format': format
      } for dtype in POSITION_DTYPE
    for num_repetitions in [3, 4]
    for format in NEIGHBOR_LIST_FORMAT
  ))
  def test_stillinger_weber_neighbor_list(self, dtype, num_repetitions,
                                          format):
    if format in [partition.OrderedSparse, partition.Sparse]:
      self.skipTest(f'{format} not supported for Stillinger-Weber.')
    lattice_vectors = np.array([[0, .5, .5],
                                [.5, 0, .5],
                                [.5, .5, 0]]) * 5.428
    positions = np.array([[0,0,0], [0.25, 0.25, 0.25]])
    positions = lattice(positions, num_repetitions, lattice_vectors)
    lattice_vectors *= num_repetitions
    displacement, shift = space.periodic_general(lattice_vectors)
    box_size =  np.linalg.det(lattice_vectors) ** (1/3) * num_repetitions
    neighbor_fn, energy_fn = \
      energy.stillinger_weber_neighbor_list(displacement, box_size,
                                            fractional_coordinates=True,
                                            format=format)
    nbrs = neighbor_fn.allocate(positions)
    N = positions.shape[0]
    self.assertAllClose(energy_fn(positions, neighbor=nbrs) / N,
                        -4.336503155764325)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_morse(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_sigma, split_epsilon, split_alpha = random.split(key, 4)
      sigma = dtype(random.uniform(
          split_sigma, (1,), minval=0., maxval=3.0)[0])
      epsilon = dtype(random.uniform(
          split_epsilon, (1,), minval=0.0, maxval=4.0)[0])
      alpha = dtype(random.uniform(
          split_alpha, (1,), minval=1.0, maxval=30.0)[0])
      dr = dtype(sigma)
      self.assertAllClose(
        energy.morse(dr, sigma, epsilon, alpha),
        np.array(-epsilon, dtype=dtype))
      g = grad(energy.morse)(dr, sigma, epsilon, alpha)
      self.assertAllClose(g, np.array(0, dtype=dtype))

    # if dr = a/alpha + sigma, then V_morse(dr, sigma, epsilon, alpha)/epsilon
    #   should be independent of sigma, epsilon, and alpha, depending only on a.
    key, split_sigma, split_epsilon, split_alpha = random.split(key, 4)
    sigmas = random.uniform(
        split_sigma, (STOCHASTIC_SAMPLES,), minval=0., maxval=3.0)
    epsilons = random.uniform(
        split_epsilon, (STOCHASTIC_SAMPLES,), minval=0.1, maxval=4.0)
    alphas = random.uniform(
        split_alpha, (STOCHASTIC_SAMPLES,), minval=1.0, maxval=30.0)
    for sigma,epsilon,alpha in zip(sigmas,epsilons,alphas):
      a = np.linspace(max(-2.5, -alpha * sigma), 8.0, 100)
      dr = np.array(a / alpha + sigma, dtype=dtype)
      U = energy.morse(dr, sigma, epsilon, alpha)/dtype(epsilon)
      Ucomp = np.array((dtype(1) - np.exp(-a)) ** dtype(2) - dtype(1),
                       dtype=dtype)
      self.assertAllClose(U, Ucomp)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_isotropic_cutoff(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, split_rs, split_rl, split_sigma, split_epsilon = random.split(key, 5)
      sigma = f32(random.uniform(
          split_sigma, (1,), minval=0.5, maxval=3.0)[0])
      epsilon = f32(random.uniform(
          split_epsilon, (1,), minval=0.0, maxval=4.0)[0])
      r_small = random.uniform(
          split_rs, (10,), minval=0.0, maxval=2.0 * sigma, dtype=dtype)
      r_large = random.uniform(
          split_rl, (10,), minval=2.5 * sigma, maxval=3.0 * sigma, dtype=dtype)

      r_onset = f32(2.0 * sigma)
      r_cutoff = f32(2.5 * sigma)

      E = energy.multiplicative_isotropic_cutoff(
        energy.lennard_jones, r_onset, r_cutoff)

      self.assertAllClose(
        E(r_small, sigma, epsilon),
        energy.lennard_jones(r_small, sigma, epsilon))
      self.assertAllClose(
        E(r_large, sigma, epsilon), np.zeros_like(r_large, dtype=dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_soft_sphere_neighbor_list_energy(self, spatial_dimension, dtype,
                                            format):
    key = random.PRNGKey(1)

    box_size = f32(15.0)
    displacement, _ = space.periodic(box_size)
    exact_energy_fn = energy.soft_sphere_pair(displacement)

    R = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
      displacement, box_size, format=format)

    nbrs = neighbor_fn.allocate(R)

    self.assertAllClose(
      np.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_lennard_jones_cell_neighbor_list_energy(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(1)

    box_size = f32(15)
    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)
    exact_energy_fn = energy.lennard_jones_pair(displacement)

    R = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
      displacement, box_size, format=format)

    nbrs = neighbor_fn.allocate(R)
    self.assertAllClose(
      np.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_morse_cell_neighbor_list_energy(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(1)

    box_size = f32(15)
    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)
    exact_energy_fn = energy.morse_pair(displacement)

    R = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    neighbor_fn, energy_fn = energy.morse_neighbor_list(
      displacement, box_size, format=format)

    nbrs = neighbor_fn.allocate(R)
    self.assertAllClose(
      np.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_morse_small_neighbor_list_energy(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(1)

    box_size = f32(5.0)
    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)
    exact_energy_fn = energy.morse_pair(displacement)

    R = box_size * random.uniform(
      key, (10, spatial_dimension), dtype=dtype)
    neighbor_fn, energy_fn = energy.morse_neighbor_list(
      displacement, box_size, format=format)

    nbrs = neighbor_fn.allocate(R)
    self.assertAllClose(
      np.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_lennard_jones_small_neighbor_list_energy(
      self, spatial_dimension, dtype, format):
    key = random.PRNGKey(1)

    box_size = f32(5.0)
    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)
    exact_energy_fn = energy.lennard_jones_pair(displacement)

    R = box_size * random.uniform(
      key, (10, spatial_dimension), dtype=dtype)
    neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
      displacement, box_size, format=format)

    nbrs = neighbor_fn.allocate(R)
    self.assertAllClose(
      np.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_lennard_jones_neighbor_list_force(self, spatial_dimension, dtype,
                                             format):
    key = random.PRNGKey(1)

    box_size = f32(15.0)
    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)
    exact_force_fn = quantity.force(energy.lennard_jones_pair(displacement))

    r = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
      displacement, box_size, format=format)
    force_fn = quantity.force(energy_fn)

    nbrs = neighbor_fn.allocate(r)
    if dtype == f32 and format is partition.OrderedSparse:
      self.assertAllClose(
        np.array(exact_force_fn(r), dtype=dtype),
        force_fn(r, nbrs), atol=5e-5, rtol=5e-5)
    else:
      self.assertAllClose(
        np.array(exact_force_fn(r), dtype=dtype),
        force_fn(r, nbrs))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_N_types={}_dtype={}'.format(N_types, dtype.__name__),
          'N_types': N_types,
          'dtype': dtype,
      } for N_types in N_TYPES_TO_TEST for dtype in POSITION_DTYPE))
  def test_behler_parrinello_network(self, N_types, dtype):
    key = random.PRNGKey(1)
    R = np.array([[0,0,0], [1,1,1], [1,1,0]], dtype)
    species = np.array([1, 1, N_types]) if N_types > 1 else None
    box_size = f32(1.5)
    displacement, _ = space.periodic(box_size)
    nn_init, nn_apply = energy.behler_parrinello(displacement, species)
    params = nn_init(key, R)
    nn_force_fn = grad(nn_apply, argnums=1)
    nn_force = jit(nn_force_fn)(params, R)
    nn_energy = jit(nn_apply)(params, R)
    self.assertAllClose(np.any(np.isnan(nn_energy)), False)
    self.assertAllClose(np.any(np.isnan(nn_force)), False)
    self.assertAllClose(nn_force.shape, [3,3])

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_N_types={N_types}_dtype={dtype.__name__}_format={str(format).split(".")[-1]}',
          'N_types': N_types,
          'dtype': dtype,
          'format': format,
      } for N_types in N_TYPES_TO_TEST
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_behler_parrinello_network_neighbor_list(self, N_types, dtype,
                                                   format):
    if format is partition.OrderedSparse:
      self.skipTest('OrderedSparse format incompatible with Behler-Parrinello '
                    'force field.')
    key = random.PRNGKey(1)
    R = np.array([[0,0,0], [1,1,1], [1,1,0]], dtype)
    species = np.array([1, 1, N_types]) if N_types > 1 else None
    box_size = f32(1.5)
    displacement, _ = space.periodic(box_size)
    neighbor_fn, nn_init, nn_apply = energy.behler_parrinello_neighbor_list(
      displacement, box_size, species, format=format)

    nbrs = neighbor_fn.allocate(R)
    params = nn_init(key, R, nbrs)
    nn_force_fn = grad(nn_apply, argnums=1)
    nn_force = jit(nn_force_fn)(params, R, nbrs)
    nn_energy = jit(nn_apply)(params, R, nbrs)
    self.assertAllClose(np.any(np.isnan(nn_energy)), False)
    self.assertAllClose(np.any(np.isnan(nn_force)), False)
    self.assertAllClose(nn_force.shape, [3,3])

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={format}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_morse_neighbor_list_force(self, spatial_dimension, dtype, format):
    key = random.PRNGKey(1)

    box_size = f32(15.0)
    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)
    exact_force_fn = quantity.force(energy.morse_pair(displacement))

    r = box_size * random.uniform(
      key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    neighbor_fn, energy_fn = energy.morse_neighbor_list(
      displacement, box_size, format=format)
    force_fn = quantity.force(energy_fn)

    nbrs = neighbor_fn.allocate(r)
    if dtype == f32 and format is partition.OrderedSparse:
      self.assertAllClose(
        np.array(exact_force_fn(r), dtype=dtype),
        force_fn(r, nbrs), atol=5e-5, rtol=5e-5)
    else:
      self.assertAllClose(
        np.array(exact_force_fn(r), dtype=dtype),
        force_fn(r, nbrs))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_num_reps={}_dtype={}'.format(
            num_repetitions, dtype.__name__),
          'num_repetitions': num_repetitions,
          'dtype': dtype,
      } for num_repetitions in UNIT_CELL_SIZE for dtype in POSITION_DTYPE))
  def test_eam(self, num_repetitions, dtype):
    latvec = np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=dtype) * f32(4.05 / 2)
    atoms = np.array([[0, 0, 0]], dtype=dtype)
    atoms_repeated, latvec_repeated = lattice_repeater(
        atoms, latvec, num_repetitions)
    inv_latvec = np.array(onp.linalg.inv(onp.array(latvec_repeated)),
                          dtype=dtype)
    displacement, _ = space.periodic_general(latvec_repeated)
    charge_fn, embedding_fn, pairwise_fn, _ = make_eam_test_splines()
    assert charge_fn(np.array(1.0, dtype)).dtype == dtype
    assert embedding_fn(np.array(1.0, dtype)).dtype == dtype
    assert pairwise_fn(np.array(1.0, dtype)).dtype == dtype
    eam_energy = energy.eam(displacement, charge_fn, embedding_fn, pairwise_fn)
    E = eam_energy(np.dot(atoms_repeated, inv_latvec)) / num_repetitions ** 3
    if dtype is f64:
      self.assertAllClose(E, dtype(-3.3633387837793505), atol=1e-8, rtol=1e-8)
    else:
      self.assertAllClose(E, dtype(-3.3633387837793505))

  @parameterized.named_parameters(jtu.cases_from_list(
      {
        'testcase_name': (f'_num_reps={num_repetitions}'
                          f'_dtype={dtype.__name__}'
                          f'_format={str(format).split(".")[-1]}'),
        'num_repetitions': num_repetitions,
        'dtype': dtype,
        'format': format
      } for num_repetitions in UNIT_CELL_SIZE
    for dtype in POSITION_DTYPE
    for format in NEIGHBOR_LIST_FORMAT))
  def test_eam_neighbor_list(self, num_repetitions, dtype, format):
    if format is partition.OrderedSparse:
      self.skipTest('OrderedSparse neighbor lists not supported for EAM '
                    'potential.')
    latvec = np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=dtype) * f32(4.05 / 2)
    atoms = np.array([[0, 0, 0]], dtype=dtype)
    atoms_repeated, latvec_repeated = lattice_repeater(
        atoms, latvec, num_repetitions)
    inv_latvec = np.array(onp.linalg.inv(onp.array(latvec_repeated)),
                          dtype=dtype)
    R = np.dot(atoms_repeated, inv_latvec)
    displacement, _ = space.periodic_general(latvec_repeated)
    box_size = np.linalg.det(latvec_repeated) ** (1 / 3)
    neighbor_fn, energy_fn = energy.eam_neighbor_list(displacement, box_size,
                                                      *make_eam_test_splines(),
                                                      format=format)
    nbrs = neighbor_fn.allocate(R)
    E = energy_fn(R, nbrs) / num_repetitions ** 3
    if dtype is f64:
      self.assertAllClose(E, dtype(-3.3633387837793505), atol=1e-8, rtol=1e-8)
    else:
      self.assertAllClose(E, dtype(-3.3633387837793505))


  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_graph_network_shape_dtype(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    R = random.uniform(key, (32, spatial_dimension), dtype=dtype)

    d, _ = space.free()

    cutoff = 0.2

    init_fn, energy_fn = energy.graph_network(d, cutoff)
    params = init_fn(key, R)

    E_out = energy_fn(params, R)

    assert E_out.shape == ()
    assert E_out.dtype == dtype

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={str(format).split(".")[-1]}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_graph_network_neighbor_list(self, spatial_dimension, dtype, format):
    if format is partition.OrderedSparse:
      self.skipTest('OrderedSparse format incompatible with GNN '
                    'force field.')

    key = random.PRNGKey(0)

    R = random.uniform(key, (32, spatial_dimension), dtype=dtype)

    d, _ = space.free()

    cutoff = 0.2

    init_fn, energy_fn = energy.graph_network(d, cutoff)
    params = init_fn(key, R)

    neighbor_fn, _, nl_energy_fn = \
      energy.graph_network_neighbor_list(d, 1.0, cutoff, 0.0, format=format)

    nbrs = neighbor_fn.allocate(R)
    if format is partition.Dense:
      self.assertAllClose(energy_fn(params, R), nl_energy_fn(params, R, nbrs))
    else:
      self.assertAllClose(energy_fn(params, R), nl_energy_fn(params, R, nbrs),
                          rtol=2e-4, atol=2e-4)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name':
        f'_dim={dim}_dtype={dtype.__name__}_format={str(format).split(".")[-1]}',
          'spatial_dimension': dim,
          'dtype': dtype,
          'format': format
      } for dim in SPATIAL_DIMENSION
        for dtype in POSITION_DTYPE
        for format in NEIGHBOR_LIST_FORMAT))
  def test_graph_network_neighbor_list_moving(self,
                                              spatial_dimension,
                                              dtype,
                                              format):
    if format is partition.OrderedSparse:
      self.skipTest('OrderedSparse format incompatible with GNN '
                    'force field.')

    key = random.PRNGKey(0)

    R = random.uniform(key, (32, spatial_dimension), dtype=dtype)

    d, _ = space.free()

    cutoff = 0.3
    dr_threshold = 0.1

    init_fn, energy_fn = energy.graph_network(d, cutoff)
    params = init_fn(key, R)

    neighbor_fn, _, nl_energy_fn = \
      energy.graph_network_neighbor_list(d, 1.0, cutoff,
                                         dr_threshold, format=format)

    nbrs = neighbor_fn.allocate(R)
    key = random.fold_in(key, 1)
    R = R + random.uniform(key, (32, spatial_dimension),
                           minval=-0.05, maxval=0.05, dtype=dtype)
    if format is partition.Dense:
      self.assertAllClose(energy_fn(params, R), nl_energy_fn(params, R, nbrs))
    else:
      self.assertAllClose(energy_fn(params, R), nl_energy_fn(params, R, nbrs),
                          rtol=2e-4, atol=2e-4)


  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype,
      } for dim in SPATIAL_DIMENSION for dtype in POSITION_DTYPE))
  def test_graph_network_learning(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    R_key, dr0_key, params_key = random.split(key, 3)

    d, _ = space.free()

    R = random.uniform(R_key, (6, 3, spatial_dimension), dtype=dtype)
    dr0 = random.uniform(dr0_key, (6, 3, 3), dtype=dtype)
    E_gt = vmap(
      lambda R, dr0: \
      np.sum((space.distance(space.map_product(d)(R, R)) - dr0) ** 2))

    cutoff = 0.2

    init_fn, energy_fn = energy.graph_network(d, cutoff)
    params = init_fn(params_key, R[0])

    @jit
    def loss(params, R):
      return np.mean((vmap(energy_fn, (None, 0))(params, R) - E_gt(R, dr0)) ** 2)

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))


    @jit
    def update(params, opt_state, R):
      updates, opt_state = opt.update(grad(loss)(params, R),
                                      opt_state)
      return optax.apply_updates(params, updates), opt_state

    opt_state = opt.init(params)

    l0 = loss(params, R)
    for i in range(4):
      params, opt_state = update(params, opt_state, R)

    assert loss(params, R) < l0 * 0.95

if __name__ == '__main__':
  absltest.main()

