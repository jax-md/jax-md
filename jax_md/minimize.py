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

  Minimization code follows the same overall structure as optimizers in JAX.
  Optimizers return two functions:
    init_fn:
      Function that initializes the  state of an optimizer. Should take
      positions as an ndarray of shape `[n, output_dimension]`. Returns a state
      which will be a namedtuple.
    apply_fn:
      Function that takes a state and produces a new state after one
      step of optimization.
"""

from collections import namedtuple

from typing import TypeVar, Callable, Tuple, Union, Any

import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce

from jax_md import quantity
from jax_md import dataclasses
from jax_md import util
from jax_md import space
from jax_md import simulate

# Types

PyTree = Any
Array = util.Array
f32 = util.f32
f64 = util.f64

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Minimizer = Tuple[InitFn, ApplyFn]


def gradient_descent(energy_or_force: Callable[..., Array],
                     shift_fn: ShiftFn,
                     step_size: float) -> Minimizer[Array]:
  """Defines gradient descent minimization.

    This is the simplest optimization strategy that moves particles down their
    gradient to the nearest minimum. Generally, gradient descent is slower than
    other methods and is included mostly for its simplicity.

    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        `[n, spatial_dimension]`.
      shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both `R`
        and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
      step_size: A floating point specifying the size of each step.

    Returns:
      See above.
  """
  force = quantity.canonicalize_force(energy_or_force)
  def init_fn(R: Array, **unused_kwargs) -> Array:
    return R
  def apply_fn(R: Array, **kwargs) -> Array:
    R = shift_fn(R, step_size * force(R, **kwargs), **kwargs)
    return R
  return init_fn, apply_fn


@dataclasses.dataclass
class FireDescentState:
  """A dataclass containing state information for the Fire Descent minimizer.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    velocity: The current velocity of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    dt: A float specifying the current step size.
    alpha: A float specifying the current momentum.
    n_pos: The number of steps in the right direction, so far.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array
  dt: float
  alpha: float
  n_pos: int


def fire_descent(energy_or_force: Callable[..., Array],
                 shift_fn: ShiftFn,
                 dt_start: float=0.1,
                 dt_max: float=0.4,
                 n_min: float=5,
                 f_inc: float=1.1,
                 f_dec: float=0.5,
                 alpha_start: float=0.1,
                 f_alpha: float=0.99) -> Minimizer[FireDescentState]:
  """Defines FIRE minimization.

  This code implements the "Fast Inertial Relaxation Engine" from Bitzek et
  al. [#bitzek]_

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    quant: Either a quantity.Energy or a quantity. Force specifying whether
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

  .. rubric:: References
  .. [#bitzek] Bitzek, Erik, Pekka Koskinen, Franz Gahler, Michael Moseler,
      and Peter Gumbsch. "Structural relaxation made simple."
      Physical review letters 97, no. 17 (2006): 170201.
  """
  dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha = util.static_cast(
    dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha)

  nve_init_fn, nve_step_fn = simulate.nve(energy_or_force, shift_fn, dt_start)
  force = quantity.canonicalize_force(energy_or_force)

  def init_fn(R: PyTree, mass: Array=1.0, **kwargs) -> FireDescentState:
    P = tree_map(lambda x: jnp.zeros_like(x), R)
    n_pos = jnp.zeros((), jnp.int32)
    F = force(R, **kwargs)
    state = FireDescentState(R, P, F, mass, dt_start, alpha_start, n_pos)  # pytype: disable=wrong-arg-count
    return simulate.canonicalize_mass(state)

  def apply_fn(state: FireDescentState, **kwargs) -> FireDescentState:
    state = nve_step_fn(state, dt=state.dt, **kwargs)
    R, P, F, M, dt, alpha, n_pos = dataclasses.unpack(state)

    # NOTE(schsam): This will be wrong if F_norm ~< 1e-8.
    # TODO(schsam): We should check for forces below 1e-6. @ErrorChecking
    F_norm = jnp.sqrt(tree_reduce(lambda accum, f:
                                  accum + jnp.sum(f ** 2) + 1e-6, F, 0.0))
    P_norm = jnp.sqrt(tree_reduce(lambda accum, p:
                                  accum + jnp.sum(p ** 2), P, 0.0))

    # NOTE: In the original FIRE algorithm, the quantity that determines when
    # to reset the momenta is F.V rather than F.P. However, all of the JAX MD
    # simulations are in momentum space for easier agreement with prior work /
    # rigid body physics. We only use the sign of F.P here, which shouldn't
    # differ from F.V, however if there are regressions then we should
    # reconsider this choice.
    F_dot_P = tree_reduce(
        lambda accum, f_dot_p: accum + f_dot_p,
        tree_map(lambda f, p: jnp.sum(f * p), F, P))
    P = tree_map(lambda p, f: p + alpha * (f * P_norm / F_norm - p), P, F)

    # NOTE(schsam): Can we clean this up at all?
    n_pos = jnp.where(F_dot_P >= 0, n_pos + 1, 0)
    dt_choice = jnp.array([dt * f_inc, dt_max])
    dt = jnp.where(F_dot_P > 0,
                   jnp.where(n_pos > n_min,
                             jnp.min(dt_choice),
                             dt),
                   dt)
    dt = jnp.where(F_dot_P < 0, dt * f_dec, dt)
    alpha = jnp.where(F_dot_P > 0,
                      jnp.where(n_pos > n_min,
                                alpha * f_alpha,
                                alpha),
                      alpha)
    alpha = jnp.where(F_dot_P < 0, alpha_start, alpha)
    P = tree_map(lambda p: (F_dot_P >= 0) * p, P)

    return FireDescentState(R, P, F, M, dt, alpha, n_pos)  # pytype: disable=wrong-arg-count
  return init_fn, apply_fn
