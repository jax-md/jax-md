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
from scipy.interpolate import splrep, PPoly

from jax_md import util


# Typing

f32 = util.f32
f64 = util.f64

#


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


def spline(y, dx, degree=3):
  """Spline fit a given scalar function.

  Args:
    y: The values of the scalar function evaluated on points starting at zero
    with the interval dx.
    dx: The interval at which the scalar function is evaluated.
    degree: Polynomial degree of the spline fit.

  Returns:
    A function that computes the spline function.
  """
  num_points = len(y)
  dx = f32(dx)
  x = np.arange(num_points, dtype=f32) * dx
  # Create a spline fit using the scipy function.
  fn = splrep(x, y, s=0, k=degree)  # Turn off smoothing by setting s to zero.
  params = PPoly.from_spline(fn)
  # Store the coefficients of the spline fit to an array.
  coeffs = np.array(params.c)
  def spline_fn(x):
    """Evaluates the spline fit for values of x."""
    ind = np.array(x / dx, dtype=np.int64)
    # The spline is defined for x values between 0 and largest value of y. If x
    # is outside this domain, truncate its ind value to within the domain.
    truncated_ind = np.array(
        np.where(ind < num_points, ind, num_points - 1), np.int64)
    truncated_ind = np.array(
        np.where(truncated_ind >= 0, truncated_ind, 0), np.int64)
    result = np.array(0, x.dtype)
    dX = x - np.array(ind, np.float32) * dx
    for i in range(degree + 1):  # sum over the polynomial terms up to degree.
      result = result + np.array(coeffs[degree - i, truncated_ind + 2], 
                                 x.dtype) * dX ** np.array(i, x.dtype)
    # For x values that are outside the domain of the spline fit, return zeros.
    result = np.where(ind < num_points, result, np.array(0.0, x.dtype))
    return result
  return spline_fn
