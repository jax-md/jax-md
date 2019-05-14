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

"""Utilities for constructing various interpolating functions.

This code was adapted from the way learning rate schedules are are built in JAX.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np


def constant(f):
  def schedule(unused_t):
    return f
  return schedule


def canonicalize(scalar_or_schedule_fun):
  if callable(scalar_or_schedule_fun):
    return scalar_or_schedule_fun
  elif np.ndim(scalar_or_schedule_fun) == 0:
    return constant(scalar_or_schedule_fun)
  else:
    raise TypeError(type(scalar_or_schedule_fun))
