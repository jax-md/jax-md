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

import jax.test_util as jtu
import jax.numpy as jnp
import numpy as onp

from jax_md import dataclasses

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



@dataclasses.dataclass
class SimulationTestState:
  fractional_position: jnp.ndarray
  real_position: jnp.ndarray
  species: jnp.ndarray
  sigma: jnp.ndarray
  box: jnp.ndarray
  energy: jnp.ndarray
  pressure: jnp.ndarray


def load_test_state(filename: str, dtype) -> SimulationTestState:
  filename = f'tests/data/{filename}'
  with open(filename, 'rb') as f:
    return SimulationTestState(
      fractional_position=onp.load(f).astype(dtype),
      real_position=onp.load(f).astype(dtype),
      species=onp.load(f),
      sigma=onp.load(f).astype(dtype),
      box=onp.load(f).astype(dtype),
      energy=onp.load(f).astype(dtype),
      pressure=onp.load(f).astype(dtype),
    )
