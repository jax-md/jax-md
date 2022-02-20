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

"""Defines testing utility functions."""

import netCDF4 as nc

import jax.test_util as jtu
import jax.numpy as jnp
import numpy as onp
from jax.config import config as jax_config

from jax_md import dataclasses

FLAGS = jax_config.FLAGS

f32 = jnp.float32


def update_test_tolerance(f32_tolerance=None, f64_tolerance=None):
  if f32_tolerance is not None:
    jtu._default_tolerance[onp.dtype(onp.float32)] = f32_tolerance
  if f64_tolerance is not None:
    jtu._default_tolerance[onp.dtype(onp.float64)] = f64_tolerance
  def default_tolerance():
    if jtu.device_under_test() != 'tpu':
      return jtu._default_tolerance
    tol = jtu._default_tolerance.copy()
    tol[onp.dtype(onp.float32)] = 5e-2
    return tol
  jtu.default_tolerance = default_tolerance


def _load_silica_data(filename: str) -> jnp.ndarray:
  filename = FLAGS.test_srcdir + filename
  with open(filename, 'rb') as f:
    return jnp.array(onp.load(f))


def load_silica_data() -> jnp.ndarray:
  try:
    filename = 'tests/data/silica_positions.npy'
    return _load_silica_data(filename)
  except FileNotFoundError:
    filename = '/google3/third_party/py/jax_md/tests/data/silica_positions.npy'
    return _load_silica_data(filename)


def _load_elasticity_test_data(filename, spatial_dimension, dtype, index):
  filename = FLAGS.test_srcdir + filename
  ds = nc.Dataset(filename)
  cijkl = jnp.array(ds.variables['Cijkl'][index], dtype=dtype)
  R = jnp.array(ds.variables['pos'][index], dtype=dtype)
  R = jnp.reshape(R, (R.shape[0]//spatial_dimension,spatial_dimension))
  sigma = 2. * jnp.array(ds.variables['rad'][index], dtype=dtype)
  box = jnp.array(ds.variables['box'][index], dtype=dtype)
  box = jnp.reshape(box, (spatial_dimension,spatial_dimension))
  return cijkl, R, sigma, box


def load_elasticity_test_data(spatial_dimension, low_pressure, dtype, index):
  try:
    if low_pressure:
      if spatial_dimension == 2:
        fn = 'tests/data/2d_polyuniform_N64_Lp-4.0.nc'
      else:
        fn = 'tests/data/3d_bi_N128_Lp-4.0.nc'
    else:
      if spatial_dimension == 2:
        fn = 'tests/data/2d_polyuniform_N64_Lp-1.0.nc'
      else:
        fn = 'tests/data/3d_bi_N128_Lp-1.0.nc'
    return _load_elasticity_test_data(fn, spatial_dimension, dtype, index)
  except FileNotFoundError:
    if low_pressure:
      if spatial_dimension == 2:
        fn = '/google3/third_party/py/jax_md/tests/data/2d_polyuniform_N64_Lp-4.0.nc'
      else:
        fn = '/google3/third_party/py/jax_md/tests/data/3d_bi_N128_Lp-4.0.nc'
    else:
      if spatial_dimension == 2:
        fn = '/google3/third_party/py/jax_md/tests/data/2d_polyuniform_N64_Lp-1.0.nc'
      else:
        fn = '/google3/third_party/py/jax_md/tests/data/3d_bi_N128_Lp-1.0.nc'
    return _load_elasticity_test_data(fn, spatial_dimension, dtype, index)


@dataclasses.dataclass
class JammedTestState:
  fractional_position: jnp.ndarray
  real_position: jnp.ndarray
  species: jnp.ndarray
  sigma: jnp.ndarray
  box: jnp.ndarray
  energy: jnp.ndarray
  pressure: jnp.ndarray


def _load_jammed_state(filename: str, dtype) -> JammedTestState:
  filename = FLAGS.test_srcdir + filename
  with open(filename, 'rb') as f:
    return JammedTestState(
        fractional_position=onp.load(f).astype(dtype),
        real_position=onp.load(f).astype(dtype),
        species=onp.load(f),
        sigma=onp.load(f).astype(dtype),
        box=onp.load(f).astype(dtype),
        energy=onp.load(f).astype(dtype),
        pressure=onp.load(f).astype(dtype),  # pytype: disable=wrong-keyword-args
    )


def load_jammed_state(filename: str, dtype) -> JammedTestState:
  try:
    full_filename = f'tests/data/{filename}'
    return _load_jammed_state(full_filename, dtype)
  except FileNotFoundError:
    full_filename = f'/google3/third_party/py/jax_md/tests/data/{filename}'
    return _load_jammed_state(full_filename, dtype)


def load_lammps_stress_data(dtype):
  def parse_state(filename):
    filename = FLAGS.test_srcdir + filename
    with open(filename) as f:
      data = f.read()
    data = data.split('\n')
    t = int(data[1])
    n = int(data[3])
    box = float(data[5].split(' ')[-1])
    R = []
    V = []
    for l in data[9:-1]:
      R += [[float(xx) for xx in l.split(' ')[:3]]]
      V += [[float(xx) for xx in l.split(' ')[3:]]]
    return f32(box), jnp.array(R, dtype), jnp.array(V, dtype)

  def parse_results(filename):
    filename = FLAGS.test_srcdir + filename
    with open(filename) as f:
      data = f.read()
    data = [[float(dd) for dd in d.split(' ') if dd != ' ' and dd != '']
            for d in data.split('\n')]
    step = jnp.array([int(d[0]) for d in data if len(d) > 0])
    Es = jnp.array([d[1] for d in data if len(d) > 0], dtype)
    C = jnp.array([d[2:] for d in data if len(d) > 0], dtype)
    C = jnp.array([[C[0, 0], C[0, 3], C[0, 4]],
                   [C[0, 3], C[0, 1], C[0, 5]],
                   [C[0, 4], C[0, 5], C[0, 2]]], dtype)
    return Es[0], C

  try:
    return (parse_state('tests/data/lammps_lj_stress_test_states'),
            parse_results('tests/data/lammps_lj_stress_test'))
  except FileNotFoundError:
    return (parse_state('/google3/third_party/py/jax_md/tests/data/'
                        'lammps_lj_stress_test_states'),
            parse_results('/google3/third_party/py/jax_md/tests/data/'
                          'lammps_lj_stress_test'))
