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

from absl import flags
from absl.testing import parameterized

import os

from typing import Dict, Sequence

import netCDF4 as nc

import jax.numpy as jnp
import numpy as onp
from jax.config import config

import jax
from jax import dtypes as _dtypes
from jax import jit
from jax import vmap

from jax_md import dataclasses

flags.DEFINE_string(
    'jax_test_dut',
    '',
    help=
    'Describes the device under test in case special consideration is required.'
)


FLAGS = flags.FLAGS


f32 = jnp.float32


# Utility functions forked from jax._src.public_test_util


_python_scalar_dtypes : dict = {
    bool: onp.dtype('bool'),
    int: onp.dtype('int64'),
    float: onp.dtype('float64'),
    complex: onp.dtype('complex128'),
}


def _dtype(x):
  if hasattr(x, 'dtype'):
    return x.dtype
  elif type(x) in _python_scalar_dtypes:
    return onp.dtype(_python_scalar_dtypes[type(x)])
  else:
    return onp.asarray(x).dtype


def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True


def device_under_test():
  return getattr(FLAGS, 'jax_test_dut', None) or jax.default_backend()


_DEFAULT_TOLERANCE = {
    onp.dtype(onp.bool_): 0,
    onp.dtype(onp.int32): 0,
    onp.dtype(onp.int64): 0,
    onp.dtype(onp.float32): 1e-6,
    onp.dtype(onp.float64): 1e-15,
}


def _default_tolerance():
  if device_under_test() != 'tpu':
    return _DEFAULT_TOLERANCE
  tol = _DEFAULT_TOLERANCE.copy()
  tol[onp.dtype(onp.float32)] = 5e-2
  tol[onp.dtype(onp.complex64)] = 5e-2
  return tol


def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=''):
  if a.dtype == b.dtype == _dtypes.float0:
    onp.testing.assert_array_equal(a, b, err_msg=err_msg)
    return
  a = a.astype(onp.float32) if a.dtype == _dtypes.bfloat16 else a
  b = b.astype(onp.float32) if b.dtype == _dtypes.bfloat16 else b
  kw = {}
  if atol: kw['atol'] = atol
  if rtol: kw['rtol'] = rtol
  with onp.errstate(invalid='ignore'):
    # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
    # value errors. It should not do that.
    onp.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


def _tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {onp.dtype(key): value for key, value in tol.items()}
  dtype = _dtypes.canonicalize_dtype(onp.dtype(dtype))
  return tol.get(dtype, _default_tolerance()[dtype])


_CACHED_INDICES: Dict[int, Sequence[int]] = {}


def cases_from_list(xs):
  xs = list(xs)
  return xs


class JAXMDTestCase(parameterized.TestCase):
  """Testing helper class forked from JaxTestCase."""

  def _assertAllClose(self, x, y, *, check_dtypes=True, atol=None, rtol=None,
                      canonicalize_dtypes=True, err_msg=''):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self._assertAllClose(x[k], y[k], check_dtypes=check_dtypes, atol=atol,
                             rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                             err_msg=err_msg)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(is_sequence(y) and not hasattr(y, '__array__'))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self._assertAllClose(x_elt, y_elt, check_dtypes=check_dtypes, atol=atol,
                             rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                             err_msg=err_msg)
    elif hasattr(x, '__array__') or onp.isscalar(x):
      self.assertTrue(hasattr(y, '__array__') or onp.isscalar(y))
      if check_dtypes:
        self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
      x = onp.asarray(x)
      y = onp.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
                                err_msg=err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assertArraysAllClose(self, x, y, *, check_dtypes=True, atol=None,
                           rtol=None, err_msg=''):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(_tolerance(_dtype(x), atol), _tolerance(_dtype(y), atol))
    rtol = max(_tolerance(_dtype(x), rtol), _tolerance(_dtype(y), rtol))
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    if not config.x64_enabled and canonicalize_dtypes:
      self.assertEqual(_dtypes.canonicalize_dtype(_dtype(x)),
                       _dtypes.canonicalize_dtype(_dtype(y)))
    else:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(
      self,
      x,
      y,
      *,
      check_dtypes=True,
      atol=None,
      rtol=None,
      canonicalize_dtypes=True,
      err_msg=''):
    def is_finite(x):
      self.assertTrue(jnp.all(jnp.isfinite(x)))

    jax.tree_map(is_finite, x)
    jax.tree_map(is_finite, y)

    def assert_close(x, y):
      self._assertAllClose(
          x, y,
          check_dtypes=check_dtypes,
          atol=atol,
          rtol=rtol,
          canonicalize_dtypes=canonicalize_dtypes,
          err_msg=err_msg,
      )

    if dataclasses.is_dataclass(x):
      self.assertIs(type(y), type(x))
      for field in dataclasses.fields(x):  # pytype: disable=module-attr
        key = field.name
        x_value, y_value = getattr(x, key), getattr(y, key)
        is_pytree_node = field.metadata.get('pytree_node', True)
        if is_pytree_node:
          assert_close(x_value, y_value)
        else:
          self.assertEqual(x_value, y_value, key)
    else:
      assert_close(x, y)


# JAX MD specific utilities.


def update_test_tolerance(f32_tol=5e-3, f64_tol=1e-5):
  _DEFAULT_TOLERANCE[onp.dtype(onp.float32)] = f32_tol
  _DEFAULT_TOLERANCE[onp.dtype(onp.float64)] = f64_tol


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
