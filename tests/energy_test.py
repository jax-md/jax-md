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

from jax import random
import jax
from jax import jit, vmap, grad
from jax.tree_util import tree_map
import jax.numpy as jnp
from scipy.io import loadmat

from jax_md import space
from jax_md.util import *
from jax_md import test_util
from jax_md import quantity

from jax_md import energy
from jax_md import partition
from jax_md.interpolate import spline

jax.config.parse_flags_with_absl()

PARTICLE_COUNT = 100
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]
UNIT_CELL_SIZE = [7, 8]

SOFT_SPHERE_ALPHA = [2.0, 2.5, 3.0]
N_TYPES_TO_TEST = [1, 2]

if jax.config.x64_enabled:
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
  return jnp.array(pos, dtype), f32(latvec*no_rep)


def make_eam_test_splines():
  cutoff = 6.28721

  num_spline_points = 21
  dr = jnp.arange(0, num_spline_points) * (cutoff / num_spline_points)
  dr = jnp.array(dr, f32)

  drho = jnp.arange(0, 2, 2. / num_spline_points)
  drho = jnp.array(drho, f32)

  density_data = jnp.array([2.78589606e-01, 2.02694937e-01, 1.45334053e-01,
                           1.06069912e-01, 8.42517168e-02, 7.65140344e-02,
                           7.76263116e-02, 8.23214224e-02, 8.53322309e-02,
                           8.13915861e-02, 6.59095390e-02, 4.28915711e-02,
                           2.27910928e-02, 1.13713167e-02, 6.05020311e-03,
                           3.65836583e-03, 2.60587564e-03, 2.06750708e-03,
                           1.48749693e-03, 7.40019174e-04, 6.21225205e-05],
                          jnp.float64)

  embedding_data = jnp.array([1.04222211e-10, -1.04142633e+00, -1.60359806e+00,
                             -1.89287637e+00, -2.09490167e+00, -2.26456628e+00,
                             -2.40590322e+00, -2.52245359e+00, -2.61385603e+00,
                             -2.67744693e+00, -2.71053295e+00, -2.71110418e+00,
                             -2.69287013e+00, -2.68464527e+00, -2.69204083e+00,
                             -2.68976209e+00, -2.66001244e+00, -2.60122024e+00,
                             -2.51338548e+00, -2.39650817e+00, -2.25058831e+00],
                            jnp.float64)

  pairwise_data = jnp.array([6.27032242e+01, 3.49638589e+01, 1.79007014e+01,
                            8.69001383e+00, 4.51545250e+00, 2.83260884e+00,
                            1.93216616e+00, 1.06795515e+00, 3.37740836e-01,
                            1.61087890e-02, -6.20816372e-02, -6.51314297e-02,
                            -5.35210341e-02, -5.20950200e-02, -5.51709524e-02,
                            -4.89093894e-02, -3.28051688e-02, -1.13738785e-02,
                            2.33833655e-03, 4.19132033e-03, 1.68600692e-04],
                           jnp.float64)

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
  return jnp.array(onp.concatenate(Rs))


def tile_silicon(tile_count, position, box, *extra_data):
  tiles = jnp.arange(tile_count)
  dx, dy, dz = jnp.meshgrid(tiles, tiles, tiles)
  dR = jnp.stack((dx, dy, dz), axis=-1)
  dR = jnp.reshape(dR, (-1, 3))

  position = (position[:, None, :] + dR[None, :, :]) / tile_count
  position = jnp.reshape(position, (-1, 3))
  box *= tile_count

  tiled_data = []
  for ex in extra_data:
    assert ex.ndim == 2
    ex = jnp.broadcast_to(ex[:, None, :],
                          (ex.shape[0], len(dR), ex.shape[1]))
    ex = jnp.reshape(ex, (-1, ex.shape[-1]))
    tiled_data += [ex]

  return [position, box] + tiled_data


def cell(a, b, c, alpha, beta, gamma):
  alpha = alpha * jnp.pi / 180
  beta = beta * jnp.pi / 180
  gamma = gamma * jnp.pi / 180
  xx = a
  yy = b * jnp.sin(gamma)
  xy = b * jnp.cos(gamma)
  xz = c * jnp.cos(beta)
  yz = (b * c * jnp.cos(alpha) - xy * xz) / yy
  zz = jnp.sqrt(c**2 - xz**2 - yz**2)
  return jnp.array([
      [xx, xy, xz],
      [0,  yy, yz],
      [0,  0,  zz]
  ])


class EnergyTest(test_util.JAXMDTestCase):

  @parameterized.named_parameters(test_util.cases_from_list(
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
      R = jnp.array([[0., 0.], [1., 1.]], dtype=dtype)
      dist = jnp.sqrt(2.)
    elif spatial_dimension == 3:
      R = jnp.array([[0., 0., 0.], [1., 1., 1.]], dtype=dtype)
      dist = jnp.sqrt(3.)
    bonds = jnp.array([[0, 1]], jnp.int32)
    for _ in range(STOCHASTIC_SAMPLES):
      key, l_key, a_key = random.split(key, 3)
      length = random.uniform(key, (), minval=0.1, maxval=3.0, dtype=dtype)
      alpha = random.uniform(key, (), minval=2., maxval=4., dtype=dtype)
      E = energy.simple_spring_bond(disp, bonds, length=length, alpha=alpha)
      E_exact = dtype(jnp.abs(dist - length) ** alpha / alpha)
      self.assertAllClose(E(R), E_exact)

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(test_util.cases_from_list(
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
      sigma = jnp.array(random.uniform(
          split_sigma, (1,), minval=0.0, maxval=3.0)[0], dtype=dtype)
      epsilon = jnp.array(
        random.uniform(split_epsilon, (1,), minval=0.0, maxval=4.0)[0],
        dtype=dtype)
      self.assertAllClose(
          energy.soft_sphere(
            dtype(0), sigma, epsilon, alpha), epsilon / alpha)
      self.assertAllClose(
        energy.soft_sphere(dtype(sigma), sigma, epsilon, alpha),
        jnp.array(0.0, dtype=dtype))
      self.assertAllClose(
        grad(energy.soft_sphere)(dtype(2 * sigma), sigma, epsilon, alpha),
        jnp.array(0.0, dtype=dtype))

      if alpha > 2.0:
        grad_energy = grad(energy.soft_sphere)
        g = grad_energy(dtype(sigma), sigma, epsilon, alpha)
        self.assertAllClose(g, jnp.array(0, dtype=dtype))

  @parameterized.named_parameters(test_util.cases_from_list(
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
        jnp.array(-epsilon, dtype=dtype))
      g = grad(energy.lennard_jones)(dr, sigma, epsilon)
      self.assertAllClose(g, jnp.array(0, dtype=dtype))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': 'dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_gupta(self, dtype):
      displacement, shift = space.free()
      pos = jnp.array([[0, 0, 0], [0, 0, 2.9], [0, 2.9, 2.9]])
      energy_fn = energy.gupta_gold55(displacement)
      self.assertAllClose(-5.4632421255957135, energy_fn(pos))


  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': 'dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_bks(self, dtype):
      LATCON = 3.5660930663857577e+01
      displacement, shift = space.periodic(LATCON)
      dist_fun = space.metric(displacement)
      species = jnp.tile(jnp.array([0, 1, 1]), 1000)
      R_f = test_util.load_silica_data().astype(dtype)
      energy_fn = energy.bks_silica_pair(dist_fun, species=species)
      assert energy_fn(R_f).dtype == dtype
      if dtype == f64:
          self.assertAllClose(dtype(-857939.528386092), energy_fn(R_f), atol=1e-5, rtol=2e-8)
      else:
          self.assertAllClose(dtype(-857939.528386092), energy_fn(R_f))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': f'dtype={dtype.__name__}_format={format}',
          'dtype': dtype,
          'format': format
      } for dtype in POSITION_DTYPE for format in NEIGHBOR_LIST_FORMAT))
  def test_bks_neighbor_list(self, dtype, format):
      LATCON = 3.5660930663857577e+01
      displacement, shift = space.periodic(LATCON)
      dist_fun = space.metric(displacement)
      species = jnp.tile(jnp.array([0, 1, 1]), 1000)
      R_f = test_util.load_silica_data().astype(dtype)
      neighbor_fn, energy_nei = energy.bks_silica_neighbor_list(
          dist_fun, LATCON, species=species, format=format)
      nbrs = neighbor_fn.allocate(R_f)
      assert energy_nei(R_f, nbrs).dtype == dtype
      if dtype == f64:
          self.assertAllClose(dtype(-857939.528386092), energy_nei(R_f, nbrs), atol=1e-5, rtol=2e-8)
      else:
          self.assertAllClose(dtype(-857939.528386092), energy_nei(R_f, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': (f'dtype={dtype.__name__}_'
                            f'num_repetitions={num_repetitions}'),
          'dtype': dtype,
          'num_repetitions': num_repetitions,
      } for dtype in POSITION_DTYPE for num_repetitions in [2, 3]))
  def test_stillinger_weber(self, dtype, num_repetitions):
    lattice_vectors = jnp.array([[0, .5, .5],
                                                  [.5, 0, .5],
                                                  [.5, .5, 0]]) * 5.428
    positions = jnp.array([[0,0,0], [0.25, 0.25, 0.25]])
    positions = lattice(positions, num_repetitions, lattice_vectors)
    lattice_vectors *= num_repetitions
    displacement, shift = space.periodic_general(lattice_vectors)
    energy_fn = jit(energy.stillinger_weber(displacement))
    N = positions.shape[0]
    self.assertAllClose(energy_fn(positions) / N, -4.336503155764325)

  @parameterized.named_parameters(test_util.cases_from_list(
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
    lattice_vectors = jnp.array([[0, .5, .5],
                                [.5, 0, .5],
                                [.5, .5, 0]]) * 5.428
    positions = jnp.array([[0,0,0], [0.25, 0.25, 0.25]])
    positions = lattice(positions, num_repetitions, lattice_vectors)
    lattice_vectors *= num_repetitions
    displacement, shift = space.periodic_general(lattice_vectors)
    box_size =  jnp.linalg.det(lattice_vectors) ** (1/3) * num_repetitions
    neighbor_fn, energy_fn = \
      energy.stillinger_weber_neighbor_list(displacement, box_size,
                                            fractional_coordinates=True,
                                            format=format)
    nbrs = neighbor_fn.allocate(positions)
    N = positions.shape[0]
    self.assertAllClose(energy_fn(positions, neighbor=nbrs) / N,
                        -4.336503155764325)

  @parameterized.named_parameters(test_util.cases_from_list(
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
        jnp.array(-epsilon, dtype=dtype))
      g = grad(energy.morse)(dr, sigma, epsilon, alpha)
      self.assertAllClose(g, jnp.array(0, dtype=dtype))

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
      a = jnp.linspace(max(-2.5, -alpha * sigma), 8.0, 100)
      dr = jnp.array(a / alpha + sigma, dtype=dtype)
      U = energy.morse(dr, sigma, epsilon, alpha)/dtype(epsilon)
      Ucomp = jnp.array((dtype(1) - jnp.exp(-a)) ** dtype(2) - dtype(1),
                       dtype=dtype)
      self.assertAllClose(U, Ucomp)

  @parameterized.named_parameters(test_util.cases_from_list(
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
        E(r_large, sigma, epsilon), jnp.zeros_like(r_large, dtype=dtype))

  @parameterized.named_parameters(test_util.cases_from_list(
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
      jnp.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
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
      jnp.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
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
      jnp.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
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
      jnp.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
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
      jnp.array(exact_energy_fn(R), dtype=dtype),
      energy_fn(R, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
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
        jnp.array(exact_force_fn(r), dtype=dtype),
        force_fn(r, nbrs), atol=5e-5, rtol=5e-5)
    else:
      self.assertAllClose(
        jnp.array(exact_force_fn(r), dtype=dtype),
        force_fn(r, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_N_types={}_dtype={}'.format(N_types, dtype.__name__),
          'N_types': N_types,
          'dtype': dtype,
      } for N_types in N_TYPES_TO_TEST for dtype in POSITION_DTYPE))
  def test_behler_parrinello_network(self, N_types, dtype):
    key = random.PRNGKey(1)
    R = jnp.array([[0,0,0], [1,1,1], [1,1,0]], dtype)
    species = jnp.array([1, 1, N_types]) if N_types > 1 else None
    box_size = f32(1.5)
    displacement, _ = space.periodic(box_size)
    nn_init, nn_apply = energy.behler_parrinello(displacement, species)
    params = nn_init(key, R)
    nn_force_fn = grad(nn_apply, argnums=1)
    nn_force = jit(nn_force_fn)(params, R)
    nn_energy = jit(nn_apply)(params, R)
    self.assertAllClose(jnp.any(jnp.isnan(nn_energy)), False)
    self.assertAllClose(jnp.any(jnp.isnan(nn_force)), False)
    self.assertAllClose(nn_force.shape, [3,3])

  @parameterized.named_parameters(test_util.cases_from_list(
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
    R = jnp.array([[0,0,0], [1,1,1], [1,1,0]], dtype)
    species = jnp.array([1, 1, N_types]) if N_types > 1 else None
    box_size = f32(1.5)
    displacement, _ = space.periodic(box_size)
    neighbor_fn, nn_init, nn_apply = energy.behler_parrinello_neighbor_list(
      displacement, box_size, species, format=format)

    nbrs = neighbor_fn.allocate(R)
    params = nn_init(key, R, nbrs)
    nn_force_fn = grad(nn_apply, argnums=1)
    nn_force = jit(nn_force_fn)(params, R, nbrs)
    nn_energy = jit(nn_apply)(params, R, nbrs)
    self.assertAllClose(jnp.any(jnp.isnan(nn_energy)), False)
    self.assertAllClose(jnp.any(jnp.isnan(nn_force)), False)
    self.assertAllClose(nn_force.shape, [3,3])

  @parameterized.named_parameters(test_util.cases_from_list(
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
        jnp.array(exact_force_fn(r), dtype=dtype),
        force_fn(r, nbrs), atol=5e-5, rtol=5e-5)
    else:
      self.assertAllClose(
        jnp.array(exact_force_fn(r), dtype=dtype),
        force_fn(r, nbrs))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_num_reps={}_dtype={}'.format(
            num_repetitions, dtype.__name__),
          'num_repetitions': num_repetitions,
          'dtype': dtype,
      } for num_repetitions in UNIT_CELL_SIZE for dtype in POSITION_DTYPE))
  def test_eam(self, num_repetitions, dtype):
    latvec = jnp.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=dtype) * f32(4.05 / 2)
    atoms = jnp.array([[0, 0, 0]], dtype=dtype)
    atoms_repeated, latvec_repeated = lattice_repeater(
        atoms, latvec, num_repetitions)
    inv_latvec = jnp.array(onp.linalg.inv(onp.array(latvec_repeated)),
                          dtype=dtype)
    displacement, _ = space.periodic_general(latvec_repeated)
    charge_fn, embedding_fn, pairwise_fn, _ = make_eam_test_splines()
    assert charge_fn(jnp.array(1.0, dtype)).dtype == dtype
    assert embedding_fn(jnp.array(1.0, dtype)).dtype == dtype
    assert pairwise_fn(jnp.array(1.0, dtype)).dtype == dtype
    eam_energy = energy.eam(displacement, charge_fn, embedding_fn, pairwise_fn)
    E = eam_energy(jnp.dot(atoms_repeated, inv_latvec)) / num_repetitions ** 3
    if dtype is f64:
      self.assertAllClose(E, dtype(-3.3633387837793505), atol=1e-8, rtol=1e-8)
    else:
      self.assertAllClose(E, dtype(-3.3633387837793505))

  @parameterized.named_parameters(test_util.cases_from_list(
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
    latvec = jnp.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=dtype) * f32(4.05 / 2)
    atoms = jnp.array([[0, 0, 0]], dtype=dtype)
    atoms_repeated, latvec_repeated = lattice_repeater(
        atoms, latvec, num_repetitions)
    inv_latvec = jnp.array(onp.linalg.inv(onp.array(latvec_repeated)),
                          dtype=dtype)
    R = jnp.dot(atoms_repeated, inv_latvec)
    displacement, _ = space.periodic_general(latvec_repeated)
    box_size = jnp.linalg.det(latvec_repeated) ** (1 / 3)
    neighbor_fn, energy_fn = energy.eam_neighbor_list(displacement, box_size,
                                                      *make_eam_test_splines(),
                                                      format=format)
    nbrs = neighbor_fn.allocate(R)
    E = energy_fn(R, nbrs) / num_repetitions ** 3
    if dtype is f64:
      self.assertAllClose(E, dtype(-3.3633387837793505), atol=1e-8, rtol=1e-8)
    else:
      self.assertAllClose(E, dtype(-3.3633387837793505))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_tersoff(self, dtype):
    lattice_vectors = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype) * 5.431
    atoms = jnp.array(
      [[0.00, 0.00, 0.00],
       [0.25, 0.25, 0.25],
       [0.00, 0.50, 0.50],
       [0.25, 0.75, 0.75],
       [0.50, 0.00, 0.50],
       [0.75, 0.25, 0.75],
       [0.50, 0.50, 0.00],
       [0.75, 0.75, 0.25]], dtype=dtype)
    atoms = lattice(atoms, 2, lattice_vectors)
    if dtype == f32:
      atoms = f32(atoms)
    lattice_vectors *= 2
    displacement, _ = space.periodic_general(lattice_vectors,
                                             fractional_coordinates=True)
    box_size = jnp.linalg.det(lattice_vectors) ** (1 / 3)
    with open('tests/data/Si.tersoff', 'r') as fh:
      tersoff_parameters = energy.load_lammps_tersoff_parameters(fh)
    if dtype == f32:
      tersoff_parameters = tree_map(lambda x: f32(x) if isinstance(x, Array) else x, tersoff_parameters)
    energy_fn = energy.tersoff(displacement, tersoff_parameters)
    E = energy_fn(atoms)
    if dtype is f64:
      self.assertAllClose(E, dtype(-296.3463784635968), atol=1e-5, rtol=2e-8)
    else:
      self.assertAllClose(E, dtype(-296.3463784635968))

    self.assertAllClose(quantity.force(energy_fn)(atoms), jnp.zeros_like(atoms))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_tersoff_neighbor_list(self, dtype):
    lattice_vectors = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype) * 5.431
    atoms = jnp.array(
      [[0.00, 0.00, 0.00],
       [0.25, 0.25, 0.25],
       [0.00, 0.50, 0.50],
       [0.25, 0.75, 0.75],
       [0.50, 0.00, 0.50],
       [0.75, 0.25, 0.75],
       [0.50, 0.50, 0.00],
       [0.75, 0.75, 0.25]], dtype=dtype)
    atoms = lattice(atoms, 2, lattice_vectors)
    if dtype == f32:
      atoms = f32(atoms)
    lattice_vectors *= 2
    displacement, _ = space.periodic_general(lattice_vectors,
                                             fractional_coordinates=True)
    box_size = jnp.linalg.det(lattice_vectors) ** (1 / 3)
    with open('tests/data/Si.tersoff', 'r') as fh:
      tersoff_parameters = energy.load_lammps_tersoff_parameters(fh)
    if dtype == f32:
      tersoff_parameters = tree_map(lambda x: f32(x) if isinstance(x, Array) else x, tersoff_parameters)
    neighbor_fn, energy_fn = energy.tersoff_neighbor_list(displacement,
                                                          box_size,
                                                          tersoff_parameters)
    nbrs = neighbor_fn.allocate(atoms)
    E = energy_fn(atoms, nbrs)
    if dtype is f64:
      self.assertAllClose(E, dtype(-296.3463784635968), atol=1e-5, rtol=2e-8)
    else:
      self.assertAllClose(E, dtype(-296.3463784635968))

    self.assertAllClose(quantity.force(energy_fn)(atoms, nbrs),
                        jnp.zeros_like(atoms))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_edip(self, dtype):
    lattice_vectors = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype) * 5.430
    atoms = jnp.array(
        [[0.25, 0.75, 0.25],
         [0.00, 0.00, 0.50],
         [0.25, 0.25, 0.75],
         [0.00, 0.50, 0.00],
         [0.75, 0.75, 0.75],
         [0.50, 0.00, 0.00],
         [0.75, 0.25, 0.25],
         [0.50, 0.50, 0.50]], dtype=dtype)
    atoms = lattice(atoms, 2, lattice_vectors)
    if dtype == f32:
      atoms = f32(atoms)
    lattice_vectors *= 2
    displacement, _ = space.periodic_general(lattice_vectors,
                                             fractional_coordinates=True)
    box_size = jnp.linalg.det(lattice_vectors) ** (1 / 3)
    energy_fn = energy.edip(displacement)
    E = energy_fn(atoms)

    if dtype is f64:
      self.assertAllClose(E, dtype(-297.597013492761), atol=1e-5, rtol=2e-8)
    else:
      self.assertAllClose(E, dtype(-297.597013492761))

    self.assertAllClose(quantity.force(energy_fn)(atoms), jnp.zeros_like(atoms))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype
      } for dtype in POSITION_DTYPE))
  def test_edip_neighbor_list(self, dtype):
    lattice_vectors = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype) * 5.430
    atoms = jnp.array(
        [[0.25, 0.75, 0.25],
         [0.00, 0.00, 0.50],
         [0.25, 0.25, 0.75],
         [0.00, 0.50, 0.00],
         [0.75, 0.75, 0.75],
         [0.50, 0.00, 0.00],
         [0.75, 0.25, 0.25],
         [0.50, 0.50, 0.50]], dtype=dtype)
    atoms = lattice(atoms, 2, lattice_vectors)
    if dtype == f32:
      atoms = f32(atoms)
    lattice_vectors *= 2
    displacement, _ = space.periodic_general(lattice_vectors,
                                             fractional_coordinates=True)
    box_size = jnp.linalg.det(lattice_vectors) ** (1 / 3)
    neighbor_fn, energy_fn = energy.edip_neighbor_list(displacement,
                                                          box_size)
    nbrs = neighbor_fn.allocate(atoms)
    E = energy_fn(atoms, nbrs)
    if dtype is f64:
      self.assertAllClose(E, dtype(-297.597013492761), atol=1e-5, rtol=2e-8)
    else:
      self.assertAllClose(E, dtype(-297.597013492761))

    self.assertAllClose(quantity.force(energy_fn)(atoms, nbrs),
                        jnp.zeros_like(atoms))

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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

  @parameterized.named_parameters(test_util.cases_from_list(
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
      jnp.sum((space.distance(space.map_product(d)(R, R)) - dr0) ** 2))

    cutoff = 0.2

    init_fn, energy_fn = energy.graph_network(d, cutoff)
    params = init_fn(params_key, R[0])

    @jit
    def loss(params, R):
      return jnp.mean((vmap(energy_fn, (None, 0))(params, R) - E_gt(R, dr0)) ** 2)

    # For some reason, importing optax at the top level causes flags to clash with
    # `jax_md.test_util`.
    import optax
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

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dtype={}'.format(dtype.__name__),
          'dtype': dtype,
      } for dtype in POSITION_DTYPE))
  def test_nequip_silicon(self, dtype):
    position = jnp.array([[0.262703, 0.752304, 0.243743],
                          [0.018137, 0.002302, 0.491184],
                          [0.248363, 0.237012, 0.776354],
                          [0.991889, 0.496844, 0.005244],
                          [0.773055, 0.742292, 0.755667],
                          [0.488418, 0.001649, 0.993714],
                          [0.7502, 0.232352, 0.236175],
                          [0.52915, 0.488468, 0.512127],], dtype=dtype)
    n = len(position)

    gt_forces = jnp.array(
      [[ -6.2008705, -12.834937,    5.787031],
       [  3.9308796,   8.419766,   -3.6275134],
       [ -4.3150625,  -6.7310944,   2.2780294],
       [  5.800577,  11.102457,   -3.3206246],
       [ -4.4260826,  -8.080319,    3.398284],
       [  6.7915206,  11.888205,   -5.344664],
       [ -5.6801977, -10.894533,    3.8579397],
       [  4.0992346,   7.1304564,  -3.0284817]], dtype=dtype)

    box = jnp.array([[ 4.707482, -0.92776, 0],
                     [ 0, 5.658054,  0.,],
                     [ 2.14774, 1.224906, 6.006509]], dtype=dtype)

    atoms = jnp.zeros((n, 94), dtype=dtype).at[:, 13].set(1)

    position, box, atoms, gt_forces = tile_silicon(5, position, box, atoms,
                                                   gt_forces)

    displacement_fn, _ = space.periodic_general(box)

    neighbor_fn, energy_fn = energy.load_gnome_model_neighbor_list(
      displacement_fn,
      box,
      'tests/data/nequip_silicon_test/',
      atoms,
      fractional_coordinates=True,
      disable_cell_list=True)

    nbrs = neighbor_fn.allocate(position)

    e, g = jax.value_and_grad(energy_fn)(position, nbrs)
    etol = 5e-3
    self.assertAllClose(e / len(position), dtype(14.0278845 / 8),
                        rtol=etol, atol=etol)

    fatol = 5e-2
    frtol = 5e-3

    self.assertAllClose(-g, gt_forces, rtol=frtol, atol=fatol)

  def test_coulomb_cubeions(self):
    dat = loadmat('tests/data/pme/cubeions.mat')
    xyzq = dat['crdq']
    gdim = dat['gdim']

    L = gdim # Angstroms

    displacement, shift = space.periodic(L[0, 0])

    R = xyzq[:, :3]
    Q = xyzq[:, -1]

    Q /= 0.05487686461

    R = jnp.mod(R, L[0, 0])

    energy_fn = energy.coulomb(displacement, L[0, 0], Q, 96, alpha=0.3488)
    F = quantity.force(energy_fn)(R)

    tol = 1e-2
    self.assertAllClose(F.reshape((-1,)),
                        (dat['fdirAMB'] + dat['frecAMB']).reshape((-1,)),
                        atol=tol, rtol=tol)


  def test_coulomb_octions(self):
    dat = loadmat('tests/data/pme/octions.mat')
    xyzq = dat['crdq']
    gdim = dat['gdim']

    box = cell(*gdim[0]) # Angstroms

    displacement, shift = space.periodic_general(box)

    R = xyzq[:, :3]
    Q = xyzq[:, -1]

    Q /= 0.05487686461

    ibox = space.inverse(box)
    R_frac = space.transform(ibox, R)
    R_frac = jnp.mod(R_frac, 1.0)

    neighbor_fn, energy_fn = energy.coulomb_neighbor_list(
      displacement, box, jnp.array(Q), 96, alpha=0.3488,
      fractional_coordinates=True)
    energy_fn = jit(energy_fn)

    nbrs = neighbor_fn.allocate(R_frac)
    F = quantity.force(energy_fn)(R_frac, nbrs)

    tol = 1e-2
    # NOTE: This test case has some very large forces O(20000) which is a bit
    # unphysical and numerical errors can lead to large absolute differences
    # which nonetheless represent a 1e-2 fractional difference. There are also
    # some very small forces which lead to large fractional differences but
    # very small absolute differences. Here we ensure that the relative
    # difference is small for points whose forces are sufficiently large.
    self.assertAllClose(F.reshape((-1,)) + 0.1,
                        (dat['fdirAMB'] + dat['frecAMB']).reshape((-1,)) + 0.1,
                        atol=10, rtol=tol)


  def test_coulomb_direct_octions(self):
    dat = loadmat('tests/data/pme/octions.mat')
    xyzq = dat['crdq']
    gdim = dat['gdim']

    box = cell(*gdim[0]) # Angstroms

    displacement, shift = space.periodic_general(box)

    R = xyzq[:, :3]
    Q = xyzq[:, -1]

    Q /= 0.05487686461

    ibox = space.inverse(box)
    R_frac = space.transform(ibox, R)
    R_frac = jnp.mod(R_frac, 1.0)

    neighbor_fn, energy_fn = energy.coulomb_direct_neighbor_list(
      displacement, box, jnp.array(Q), alpha=0.3488,
      fractional_coordinates=True)
    energy_fn = jit(energy_fn)

    nbrs = neighbor_fn.allocate(R_frac)
    F = quantity.force(energy_fn)(R_frac, nbrs)

    tol = 1e-2
    self.assertAllClose(F.reshape((-1,)) + 0.1,
                        dat['fdirAMB'].reshape((-1,)) + 0.1,
                        atol=10.0, rtol=tol)


  def test_coulomb_recip_octions(self):
    dat = loadmat('tests/data/pme/octions.mat')
    xyzq = dat['crdq']
    gdim = dat['gdim']

    box = cell(*gdim[0]) # Angstroms

    displacement, shift = space.periodic_general(box)

    R = xyzq[:, :3]
    Q = xyzq[:, -1]

    Q /= 0.05487686461

    ibox = space.inverse(box)
    R_frac = space.transform(ibox, R)
    R_frac = jnp.mod(R_frac, 1.0)

    energy_fn = energy.coulomb_recip_pme(
      jnp.array(Q),
      box,
      96,
      alpha=0.3488,
      fractional_coordinates=True)
    energy_fn = jit(energy_fn)

    F = quantity.force(energy_fn)(R_frac)

    tol = 1e-2
    self.assertAllClose(F.reshape((-1,)) + 0.1,
                        dat['frecAMB'].reshape((-1,)) + 0.1,
                        atol=10, rtol=tol)


if __name__ == '__main__':
  absltest.main()
