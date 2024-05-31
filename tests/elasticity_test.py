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

"""Tests for google3.third_party.py.jax_md.elasticity."""

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
from jax import jit
from jax import lax
from jax import grad
import jax.numpy as np

from jax_md import space
from jax_md import energy
from jax_md import minimize
from jax_md import quantity
from jax_md import elasticity
from jax_md import test_util
from jax_md.util import *

jax.config.parse_flags_with_absl()

PARTICLE_COUNT = 64
NUM_SAMPLES = 2
SPATIAL_DIMENSION = [2, 3]
LOWPRESSURE = [True, False]

if jax.config.jax_enable_x64:
  DTYPE = [f32, f64]
else:
  DTYPE = [f32]

def run_minimization_while(energy_fn,
                           R_init,
                           shift,
                           max_grad_thresh = 1e-12,
                           max_num_steps=1000000,
                           **kwargs):
  init,apply=minimize.fire_descent(jit(energy_fn), shift, **kwargs)
  apply = jit(apply)

  @jit
  def get_maxgrad(state):
    return jnp.amax(jnp.abs(state.force))

  @jit
  def cond_fn(val):
    state, i = val
    return jnp.logical_and(get_maxgrad(state) > max_grad_thresh,
                           i<max_num_steps)

  @jit
  def body_fn(val):
    state, i = val
    return apply(state), i+1

  state = init(R_init)
  state, num_iterations = lax.while_loop(cond_fn, body_fn, (state, 0))

  return state.position, get_maxgrad(state), num_iterations


class ElasticityTest(test_util.JAXMDTestCase):
  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}_lowpressure={}'.format(
            dim, dtype.__name__, low_pressure),
          'spatial_dimension': dim,
          'dtype': dtype,
          'low_pressure': low_pressure
      } for dim in SPATIAL_DIMENSION
        for dtype in DTYPE
        for low_pressure in LOWPRESSURE))
  def test_EMT_from_db_basic(self, spatial_dimension, dtype, low_pressure):
    if spatial_dimension == 2:
      N = 64
    else:
      N = 128

    if dtype == jnp.float32:
      max_grad_thresh = 1e-5
      atol = 1e-4
      rtol = 1e-3
    else:
      max_grad_thresh = 1e-10
      atol = 1e-8
      rtol = 1e-5

    for index in range(NUM_SAMPLES):
      cijkl, R, sigma, box = test_util.load_elasticity_test_data(
          spatial_dimension, low_pressure, dtype, index)
      R = space.transform(box, R)
      box = box[0,0]

      displacement, shift = space.periodic(box)
      energy_fn = energy.soft_sphere_pair(displacement, sigma=sigma)
      assert( jnp.max(jnp.abs(grad(energy_fn)(R))) < max_grad_thresh )

      EMT_fn = jit(elasticity.athermal_moduli(energy_fn,check_convergence=True))
      C, converged = EMT_fn(R,box)
      assert( C.dtype == dtype )
      assert( C.shape == (spatial_dimension,spatial_dimension,
        spatial_dimension,spatial_dimension) )
      if converged:
        self.assertAllClose(cijkl,elasticity._extract_elements(C,False),
            atol=atol, rtol=rtol)

        #make sure the symmetries are there
        self.assertAllClose(C, jnp.einsum("ijkl->jikl", C))
        self.assertAllClose(C, jnp.einsum("ijkl->ijlk", C))
        self.assertAllClose(C, jnp.einsum("ijkl->lkij", C))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}_lowpressure={}'.format(dim,
            dtype.__name__, low_pressure),
          'spatial_dimension': dim,
          'dtype': dtype,
          'low_pressure': low_pressure
      } for dim in SPATIAL_DIMENSION
        for dtype in DTYPE
        for low_pressure in LOWPRESSURE))
  def test_EMT_from_db_pgeneral(self, spatial_dimension, dtype, low_pressure):
    if spatial_dimension == 2:
      N = 64
    else:
      N = 128

    if dtype == jnp.float32:
      max_grad_thresh = 1e-5
      atol = 1e-4
      rtol = 1e-3
    else:
      max_grad_thresh = 1e-10
      atol = 1e-8
      rtol = 1e-5

    for index in range(NUM_SAMPLES):
      cijkl, R, sigma, box = test_util.load_elasticity_test_data(
          spatial_dimension, low_pressure, dtype, index)
      R = space.transform(box, R)

      displacement, shift = space.periodic_general(box,
          fractional_coordinates=False)
      energy_fn = energy.soft_sphere_pair(displacement, sigma=sigma)
      assert( jnp.max(jnp.abs(grad(energy_fn)(R))) < max_grad_thresh )

      EMT_fn = jit(elasticity.athermal_moduli(energy_fn,check_convergence=True))
      C, converged = EMT_fn(R,box)
      assert( C.dtype == dtype )
      assert( C.shape == (spatial_dimension,spatial_dimension,
        spatial_dimension,spatial_dimension) )
      if converged:
        self.assertAllClose(cijkl,elasticity._extract_elements(C,False),
            atol=atol, rtol=rtol)

        #make sure the symmetries are there
        self.assertAllClose(C, jnp.einsum("ijkl->jikl", C))
        self.assertAllClose(C, jnp.einsum("ijkl->ijlk", C))
        self.assertAllClose(C, jnp.einsum("ijkl->lkij", C))

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}_lowpressure={}'.format(dim,
            dtype.__name__, low_pressure),
          'spatial_dimension': dim,
          'dtype': dtype,
          'low_pressure': low_pressure
      } for dim in SPATIAL_DIMENSION
        for dtype in DTYPE
        for low_pressure in LOWPRESSURE))
  def test_EMT_from_db_fraccoord(self, spatial_dimension, dtype, low_pressure):
    if spatial_dimension == 2:
      N = 64
    else:
      N = 128

    if dtype == jnp.float32:
      max_grad_thresh = 1e-5
      atol = 1e-4
      rtol = 1e-3
    else:
      max_grad_thresh = 1e-10
      atol = 1e-8
      rtol = 1e-5

    for index in range(NUM_SAMPLES):
      cijkl, R, sigma, box = test_util.load_elasticity_test_data(
          spatial_dimension, low_pressure, dtype, index)

      displacement, shift = space.periodic_general(box,
          fractional_coordinates=True)
      energy_fn = energy.soft_sphere_pair(displacement, sigma=sigma)
      assert( jnp.max(jnp.abs(grad(energy_fn)(R))) < max_grad_thresh )

      EMT_fn = jit(elasticity.athermal_moduli(energy_fn,check_convergence=True))
      C, converged = EMT_fn(R,box)
      assert( C.dtype == dtype )
      assert( C.shape == (spatial_dimension,spatial_dimension,
        spatial_dimension,spatial_dimension) )
      if converged:
        self.assertAllClose(cijkl,elasticity._extract_elements(C,False),
            atol=atol, rtol=rtol)

        #make sure the symmetries are there
        self.assertAllClose(C, jnp.einsum("ijkl->jikl", C))
        self.assertAllClose(C, jnp.einsum("ijkl->ijlk", C))
        self.assertAllClose(C, jnp.einsum("ijkl->lkij", C))


  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}_lowpressure={}'.format(dim,
            dtype.__name__, low_pressure),
          'spatial_dimension': dim,
          'dtype': dtype,
          'low_pressure': low_pressure
      } for dim in SPATIAL_DIMENSION
        for dtype in DTYPE
        for low_pressure in LOWPRESSURE))
  def test_EMT_from_db_dynamic(self, spatial_dimension, dtype, low_pressure):
    if spatial_dimension == 2:
      N = 64
    else:
      N = 128

    if dtype == jnp.float32:
      max_grad_thresh = 1e-5
      atol = 1e-4
      rtol = 1e-3
    else:
      max_grad_thresh = 1e-10
      atol = 1e-8
      rtol = 1e-5

    for index in range(NUM_SAMPLES):
      cijkl, R, sigma, box = test_util.load_elasticity_test_data(
          spatial_dimension, low_pressure, dtype, index)
      R = space.transform(box, R)
      box = box[0,0]

      displacement, shift = space.periodic(box)
      #Below we use the wrong sigma, so we must pass it dynamically
      energy_fn = energy.soft_sphere_pair(displacement, sigma=1.0)
      maxgrad = jnp.max(jnp.abs(grad(energy_fn)(R, sigma=sigma)))
      assert( maxgrad < max_grad_thresh )

      EMT_fn = jit(elasticity.athermal_moduli(energy_fn,check_convergence=True))
      C, converged = EMT_fn(R,box,sigma=sigma)
      assert( C.dtype == dtype )
      assert( C.shape == (spatial_dimension,spatial_dimension,
        spatial_dimension,spatial_dimension) )
      if converged:
        self.assertAllClose(cijkl,elasticity._extract_elements(C,False),
            atol=atol, rtol=rtol)

        #make sure the symmetries are there
        self.assertAllClose(C, jnp.einsum("ijkl->jikl", C))
        self.assertAllClose(C, jnp.einsum("ijkl->ijlk", C))
        self.assertAllClose(C, jnp.einsum("ijkl->lkij", C))


  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}_lowpressure={}'.format(dim,
            dtype.__name__, low_pressure),
          'spatial_dimension': dim,
          'dtype': dtype,
          'low_pressure': low_pressure
      } for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
      for low_pressure in LOWPRESSURE))
  def test_EMT_from_db_nbrlist(self, spatial_dimension, dtype, low_pressure):
    if spatial_dimension == 2:
      N = 64
    else:
      N = 128

    if dtype == jnp.float32:
      max_grad_thresh = 1e-5
      atol = 1e-4
      rtol = 1e-3
    else:
      max_grad_thresh = 1e-10
      atol = 1e-8
      rtol = 1e-5

    for index in range(NUM_SAMPLES):
      cijkl, R, sigma, box = test_util.load_elasticity_test_data(
          spatial_dimension, low_pressure, dtype, index)

      displacement, shift = space.periodic_general(box,
          fractional_coordinates=True)
      neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
          displacement, box, sigma=sigma, fractional_coordinates=True)
      nbrs = neighbor_fn.allocate(R)
      assert( jnp.max(jnp.abs(grad(energy_fn)(R, nbrs))) < max_grad_thresh )

      EMT_fn = jit(elasticity.athermal_moduli(energy_fn,check_convergence=True))
      C, converged = EMT_fn(R,box,neighbor=nbrs)
      assert( C.dtype == dtype )
      assert( C.shape == (spatial_dimension,spatial_dimension,
        spatial_dimension,spatial_dimension) )
      if converged:
        self.assertAllClose(cijkl,elasticity._extract_elements(C,False),
            atol=atol, rtol=rtol)

        #make sure the symmetries are there
        self.assertAllClose(C, jnp.einsum("ijkl->jikl", C))
        self.assertAllClose(C, jnp.einsum("ijkl->ijlk", C))
        self.assertAllClose(C, jnp.einsum("ijkl->lkij", C))


  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION for dtype in DTYPE))
  def test_mandel(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(NUM_SAMPLES):
      key, split = random.split(key, 2)

      if spatial_dimension == 2:
        C = jnp.array(
            [[[[ 1.13172135,  0.00230386],
               [ 0.00230386,  0.55599847]],

              [[ 0.00230386,  0.39732675],
               [ 0.39732675, -0.01848658]]],


             [[[ 0.00230386,  0.39732675],
               [ 0.39732675, -0.01848658]],

              [[ 0.55599847, -0.01848658],
               [-0.01848658,  1.14191721]]]],
              dtype = dtype)
      else:
        C = jnp.array(
            [[[[ 6.02440072e-01, -3.81092776e-02, -5.43015909e-02],
               [-3.81092776e-02,  5.29322888e-01, -2.06230517e-02],
               [-5.43015909e-02, -2.06230517e-02,  2.66616858e-01]],

              [[-3.81092776e-02,  1.27745712e-01,  8.22006563e-03],
               [ 1.27745712e-01,  3.85350327e-02, -1.06514552e-02],
               [ 8.22006563e-03, -1.06514552e-02, -3.45478902e-02]],

              [[-5.43015909e-02,  8.22006563e-03,  1.12683315e-01],
               [ 8.22006563e-03,  3.89981105e-02,  3.45578275e-02],
               [ 1.12683315e-01,  3.45578275e-02,  1.15577485e-02]]],


             [[[-3.81092776e-02,  1.27745712e-01,  8.22006563e-03],
               [ 1.27745712e-01,  3.85350327e-02, -1.06514552e-02],
               [ 8.22006563e-03, -1.06514552e-02, -3.45478902e-02]],

              [[ 5.29322888e-01,  3.85350327e-02,  3.89981105e-02],
               [ 3.85350327e-02,  6.67240460e-01, -3.73405057e-04],
               [ 3.89981105e-02, -3.73405057e-04,  5.27227690e-01]],

              [[-2.06230517e-02, -1.06514552e-02,  3.45578275e-02],
               [-1.06514552e-02, -3.73405057e-04,  2.17405756e-01],
               [ 3.45578275e-02,  2.17405756e-01, -1.62693657e-02]]],


             [[[-5.43015909e-02,  8.22006563e-03,  1.12683315e-01],
               [ 8.22006563e-03,  3.89981105e-02,  3.45578275e-02],
               [ 1.12683315e-01,  3.45578275e-02,  1.15577485e-02]],

              [[-2.06230517e-02, -1.06514552e-02,  3.45578275e-02],
               [-1.06514552e-02, -3.73405057e-04,  2.17405756e-01],
               [ 3.45578275e-02,  2.17405756e-01, -1.62693657e-02]],

              [[ 2.66616858e-01, -3.45478902e-02,  1.15577485e-02],
               [-3.45478902e-02,  5.27227690e-01, -1.62693657e-02],
               [ 1.15577485e-02, -1.62693657e-02,  5.45653623e-01]]]],
              dtype = dtype)

      e = random.uniform(split, (spatial_dimension,spatial_dimension),
          minval=-1, maxval=1, dtype=dtype)
      e = (e + e.T) / 2.0

      e_m = elasticity.tensor_to_mandel(e)
      C_m = elasticity.tensor_to_mandel(C)

      sum_m = jnp.einsum('i,ij,j->',e_m, C_m, e_m)
      sum_t = jnp.einsum('ij,ijkl,kl->',e, C, e)

      self.assertAllClose(sum_m, sum_t)

      self.assertAllClose(e, elasticity.mandel_to_tensor(e_m))
      self.assertAllClose(C, elasticity.mandel_to_tensor(C_m))


if __name__ == '__main__':
  absltest.main()
