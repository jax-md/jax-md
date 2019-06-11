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

"""Code to minimize the energy of a system.

  This file contains a number of different methods that can be used to find the
  nearest minimum (inherent structure) to some initial system described by a
  position R.

  In general, minimization code follows the same overall structure as optimizers
  in JAX. Optimizers return two functions:
    init_fn: function that initializes the  state of an optimizer. Should take
      positions as an ndarray of shape [n, output_dimension]. Returns a state
      which will be a namedtuple.
    apply_fn: function that takes a state and produces a new state after one
      step of optimization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import jax.numpy as np

from jax_md import quantity
from jax_md.util import *


# pylint: disable=invalid-name
def gradient_descent(
    energy_or_force, shift_fn, step_size, quant=quantity.Energy):
  """Defines gradient descent minimization.

    This is the simplest optimization strategy that moves particles down their
    gradient to the nearest minimum. Generally, gradient descent is slower than
    other methods and is included mostly for its simplicity.

    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        [n, spatial_dimension].
      shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
      step_size: A floating point specifying the size of each step.
      quant: Either a quantity.Energy or a quantity.Force specifying whether
        energy_or_force is an energy or force respectively.

    Returns:
      See above.
  """
  force = quantity.canonicalize_force(energy_or_force, quant)
  def init_fun(R, **unused_kwargs):
    return R
  def apply_fun(R, **kwargs):
    R = shift_fn(R, step_size * force(R, **kwargs), **kwargs)
    return R
  return init_fun, apply_fun


class FireDescentState(namedtuple(
    'FireDescentState',
    ['position', 'velocity', 'force', 'dt', 'alpha', 'n_pos'])):
  """A tuple containing state information for the Fire Descent minimizer.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape [n, spatial_dimension].
    velocity: The current velocity of particles. An ndarray of floats
      with shape [n, spatial_dimension].
    force: The current force on particles. An ndarray of floats
      with shape [n, spatial_dimension].
    dt: A float specifying the current step size.
    alpha: A float specifying the current momentum.
    n_pos: The number of steps in the right direction, so far.
  """

  def __new__(cls, position, velocity, force, dt, alpha, n_pos):
    return super(FireDescentState, cls).__new__(
        cls, position, velocity, force, dt, alpha, n_pos)
register_pytree_namedtuple(FireDescentState)


def fire_descent(
    energy_or_force, shift_fn, quant=quantity.Energy, dt_start=0.1,
    dt_max=0.4, n_min=5, f_inc=1.1, f_dec=0.5, alpha_start=0.1, f_alpha=0.99):
  """Defines FIRE minimization.

  This code implements the "Fast Inertial Relaxation Engine" from [1].

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    quant: Either a quantity.Energy or a quantity.Force specifying whether
      energy_or_force is an energy or force respectively.
    dt_start: The initial step size during minimization as a float.
    dt_max: The maximum step size during minimization as a float.
    n_min: An integer specifying the minimum number of steps moving in the
      correct direction before dt and f_alpha should be updated.
    f_inc: A float specifying the fractional rate by which the step size
      should be increased.
    f_dec: A float specifying the fractional rate by which the step size
      should be decreased.
    alpha_start: A float specifying the initial momentum.
    f_alpha: A float specifying the fractional change in momentum.
  Returns:
    See above.

  [1] Bitzek, Erik, Pekka Koskinen, Franz Gahler, Michael Moseler,
      and Peter Gumbsch. "Structural relaxation made simple."
      Physical review letters 97, no. 17 (2006): 170201.
  """

  dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha = static_cast(
    dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha)

  force = quantity.canonicalize_force(energy_or_force, quant)
  def init_fun(R, **kwargs):
    V = np.zeros_like(R)
    return FireDescentState(
      R, V, force(R, **kwargs), dt_start, alpha_start, f32(0))
  def apply_fun(state, **kwargs):
    R, V, F_old, dt, alpha, n_pos = state

    R = shift_fn(R, dt * V + dt ** f32(2) * F_old, **kwargs)

    F = force(R, **kwargs)

    V = V + dt * f32(0.5) * (F_old + F)

    # NOTE(schsam): This will be wrong if F_norm ~< 1e-8.
    # TODO(schsam): We should check for forces below 1e-6. @ErrorChecking
    F_norm = np.sqrt(np.sum(F ** f32(2)) + f32(1e-6))
    V_norm = np.sqrt(np.sum(V ** f32(2)))

    P = np.array(np.dot(np.reshape(F, (-1)), np.reshape(V, (-1))))

    V = V + alpha * (F * V_norm / F_norm - V)

    # NOTE(schsam): Can we clean this up at all?
    n_pos = np.where(P >= 0, n_pos + f32(1.0), f32(0))
    dt_choice = np.array([dt * f_inc, dt_max])
    dt = np.where(
        P > 0, np.where(n_pos > n_min, np.min(dt_choice), dt), dt)
    dt = np.where(P < 0, dt * f_dec, dt)
    alpha = np.where(
        P > 0, np.where(n_pos > n_min, alpha * f_alpha, alpha), alpha)
    alpha = np.where(P < 0, alpha_start, alpha)
    V = (P < 0) * np.zeros_like(V) + (P >= 0) * V

    return FireDescentState(R, V, F, dt, alpha, n_pos)
  return init_fun, apply_fun
