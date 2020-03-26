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
import jax.numpy as np

def update_test_tolerance(f32_tolerance=None, f64_tolerance=None):
  if f32_tolerance is not None:
    jtu._default_tolerance[np.onp.dtype(np.onp.float32)] = f32_tolerance
  if f64_tolerance is not None:
    jtu._default_tolerance[np.onp.dtype(np.onp.float64)] = f64_tolerance
  def default_tolerance():
    if jtu.device_under_test() != 'tpu':
      return jtu._default_tolerance
    tol = jtu._default_tolerance.copy()
    tol[np.onp.dtype(np.onp.float32)] = 5e-2
    return tol
  jtu.default_tolerance = default_tolerance
