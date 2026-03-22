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
  which will be a dataclass.

apply_fn:
  Function that takes a state and produces a new state after one
  step of optimization.
"""

from typing import TypeVar, Callable, Tuple, Union, Any

import jax
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


def gradient_descent(
  energy_or_force: Callable[..., Array], shift_fn: ShiftFn, step_size: float
) -> Minimizer[Array]:
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
    momentum: The current momentum of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    mass: The mass of particles. A float or an ndarray of floats
      with shape `[n]`.
    dt: A float specifying the current step size.
    alpha: A float specifying the current FIRE mixing parameter.
    n_pos: The number of consecutive steps with positive power.
  """

  position: Array
  momentum: Array
  force: Array
  mass: Array
  dt: float
  alpha: float
  n_pos: int


def fire_descent(
  energy_or_force: Callable[..., Array],
  shift_fn: ShiftFn,
  dt_start: float = 0.1,
  dt_max: float = 0.4,
  n_min: float = 5,
  f_inc: float = 1.1,
  f_dec: float = 0.5,
  alpha_start: float = 0.1,
  f_alpha: float = 0.99,
) -> Minimizer[FireDescentState]:
  """Defines FIRE minimization.

  This code implements the "Fast Inertial Relaxation Engine" from Bitzek et
  al. [#bitzek]_

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
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
  dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha = (
    util.static_cast(
      dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha
    )
  )

  nve_init_fn, nve_step_fn = simulate.nve(energy_or_force, shift_fn, dt_start)
  force = quantity.canonicalize_force(energy_or_force)

  def init_fn(R: PyTree, mass: Array = 1.0, **kwargs) -> FireDescentState:
    P = tree_map(lambda x: jnp.zeros_like(x), R)
    n_pos = jnp.zeros((), jnp.int32)
    F = force(R, **kwargs)
    state = FireDescentState(
      R, P, F, mass, dt_start, alpha_start, n_pos
    )  # pytype: disable=wrong-arg-count
    return simulate.canonicalize_mass(state)

  def apply_fn(state: FireDescentState, **kwargs) -> FireDescentState:
    state = nve_step_fn(state, dt=state.dt, **kwargs)
    R, P, F, M, dt, alpha, n_pos = dataclasses.unpack(state)

    # NOTE(schsam): This will be wrong if F_norm ~< 1e-8.
    # TODO(schsam): We should check for forces below 1e-6. @ErrorChecking
    F_norm = jnp.sqrt(
      tree_reduce(lambda accum, f: accum + jnp.sum(f**2) + 1e-6, F, 0.0)
    )
    P_norm = jnp.sqrt(
      tree_reduce(lambda accum, p: accum + jnp.sum(p**2), P, 0.0)
    )

    # NOTE: In the original FIRE algorithm, the quantity that determines when
    # to reset the momenta is F.V rather than F.P. However, all of the JAX MD
    # simulations are in momentum space for easier agreement with prior work /
    # rigid body physics. We only use the sign of F.P here, which shouldn't
    # differ from F.V, however if there are regressions then we should
    # reconsider this choice.
    F_dot_P = tree_reduce(
      lambda accum, f_dot_p: accum + f_dot_p,
      tree_map(lambda f, p: jnp.sum(f * p), F, P),
    )
    P = tree_map(lambda p, f: p + alpha * (f * P_norm / F_norm - p), P, F)

    # NOTE(schsam): Can we clean this up at all?
    n_pos = jnp.where(F_dot_P >= 0, n_pos + 1, 0)
    dt_choice = jnp.array([dt * f_inc, dt_max])
    dt = jnp.where(
      F_dot_P > 0, jnp.where(n_pos > n_min, jnp.min(dt_choice), dt), dt
    )
    dt = jnp.where(F_dot_P < 0, dt * f_dec, dt)
    alpha = jnp.where(
      F_dot_P > 0, jnp.where(n_pos > n_min, alpha * f_alpha, alpha), alpha
    )
    alpha = jnp.where(F_dot_P < 0, alpha_start, alpha)
    P = tree_map(lambda p: (F_dot_P >= 0) * p, P)

    return FireDescentState(
      R, P, F, M, dt, alpha, n_pos
    )  # pytype: disable=wrong-arg-count

  return init_fn, apply_fn


# Box optimization


@dataclasses.dataclass
class FireBoxDescentState:
  """State for the combined atom + box FIRE minimizer.

  Box degrees of freedom are parameterized by the deformation gradient
  ``F`` relative to a reference box, following Tadmor et al. (1999).
  The current box is reconstructed as ``F @ reference_box``.

  The ``box_factor`` (default ``N``) scales
  the deformation-gradient DOFs to be comparable to atomic positions,
  acting as a preconditioner.

  Attributes:
    position: Atomic fractional positions, shape ``(N, dim)``.
    momentum: Atomic momenta, shape ``(N, dim)``.
    force: Atomic forces, shape ``(N, dim)``.
    mass: Atomic masses.
    box: Current simulation box (``F @ reference_box``), shape ``(dim, dim)``.
    reference_box: Initial box (constant), shape ``(dim, dim)``.
    box_position: ``box_factor * F``, shape ``(dim, dim)``.
    box_momentum: Momentum in deformation-gradient space, ``(dim, dim)``.
    box_force: Force in deformation-gradient space, ``(dim, dim)``.
    box_mass: Box mass (scalar).
    box_factor: Scaling preconditioner (scalar, default ``N``).
    dt: Current FIRE step size.
    alpha: Current FIRE momentum-mixing parameter.
    n_pos: Number of consecutive steps with positive power.
  """

  position: Array
  momentum: Array
  force: Array
  mass: Array
  box: Array
  reference_box: Array
  box_position: Array
  box_momentum: Array
  box_force: Array
  box_mass: Array
  box_factor: Array
  dt: float
  alpha: float
  n_pos: int


def fire_descent_box(
  energy_fn: Callable[..., Array],
  shift_fn: ShiftFn,
  dt_start: float = 0.1,
  dt_max: float = 0.4,
  n_min: float = 5,
  f_inc: float = 1.1,
  f_dec: float = 0.5,
  alpha_start: float = 0.1,
  f_alpha: float = 0.99,
  scalar_pressure: float = 0.0,
  hydrostatic_strain: bool = False,
  constant_volume: bool = False,
  mask: Array = None,
) -> Minimizer[FireBoxDescentState]:
  """FIRE minimization of both atomic positions and the simulation box.

  Args:
    energy_fn: Energy function taking ``(R, box=box, **kwargs)``.
    shift_fn: Shift function from :func:`~jax_md.space.periodic_general`
      (``fractional_coordinates=True``).
    dt_start: Initial FIRE step size.
    dt_max: Maximum FIRE step size.
    n_min: Minimum positive-power steps before increasing dt.
    f_inc: Factor to increase dt.
    f_dec: Factor to decrease dt on overshoot.
    alpha_start: Initial FIRE mixing parameter.
    f_alpha: Factor to decrease alpha.
    scalar_pressure: Target external pressure.
    hydrostatic_strain: Constrain box to isotropic deformation.
    constant_volume: Project out volume changes.
    mask: Strain-component mask, ``(dim, dim)``.

  Returns:
    ``(init_fn, apply_fn)`` pair.
  """
  dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha = (
    util.static_cast(
      dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha
    )
  )
  force_fn = quantity.canonicalize_force(energy_fn)

  def virial_fn(R, box, **kwargs):
    """Compute the constrained virial for the box force.

    The virial is ``W = -dE/deps - P_ext * V * I`` where ``dE/deps`` is
    the strain derivative obtained via autodiff.  The sign follows
    JAX-MD's stress convention (``quantity.stress`` returns
    ``(1/V)(-dU/deps)`` with positive = tension).  ``-dE/deps`` points
    in the direction of energy decrease, matching the simple
    ``cell_step`` approach (``box += dt * stress``).

    Optional constraint projections (hydrostatic, mask, constant-volume)
    are applied before returning.
    """
    dim = R.shape[1]
    I_d = jnp.eye(dim, dtype=box.dtype)
    zero = jnp.zeros((dim, dim), dtype=box.dtype)

    def U(eps):
      return energy_fn(R, box=box, perturbation=(I_d + eps), **kwargs)

    dUdeps = jax.grad(U)(zero)

    v = -dUdeps
    if scalar_pressure != 0.0:
      vol = quantity.volume(dim, box)
      v = v - scalar_pressure * vol * I_d

    if hydrostatic_strain:
      tr = jnp.trace(v)
      v = jnp.eye(dim, dtype=v.dtype) * (tr / dim)

    if mask is not None:
      v = v * jnp.asarray(mask, dtype=v.dtype)

    if constant_volume:
      tr = jnp.trace(v)
      v = v - jnp.eye(dim, dtype=v.dtype) * (tr / dim)

    return v

  def box_force_fn(virial, F_deform, bf):
    """Transform virial to deformation-gradient space and scale.

    Maps the virial from box-space to F-space via the Jacobian
    ``F^{-T}``: ``virial_F = solve(F, virial^T)^T``.  This matches
    ASE ``UnitCellFilter.get_forces`` and TorchSim
    ``compute_cell_forces``.  At ``F = I`` the transformation is
    identity.
    """
    W_F = jnp.linalg.solve(F_deform, virial.T).T
    return W_F / bf

  def init_fn(
    R: Array,
    box: Array,
    mass: Array = 1.0,
    box_mass: Array = None,
    box_factor: Array = None,
    **kwargs,
  ) -> FireBoxDescentState:
    N = R.shape[0]
    dim = R.shape[1]
    if box_mass is None:
      box_mass = f32(1.0)
    if box_factor is None:
      box_factor = f32(N)
    F_atom = force_fn(R, box=box, **kwargs)
    I_d = jnp.eye(dim, dtype=box.dtype)
    virial = virial_fn(R, box, **kwargs)
    F_b = box_force_fn(virial, I_d, box_factor)
    state = FireBoxDescentState(
      R,
      jnp.zeros_like(R),
      F_atom,
      mass,
      box,
      box,
      box_factor * I_d,
      jnp.zeros_like(box),
      F_b,
      box_mass,
      box_factor,
      dt_start,
      alpha_start,
      jnp.zeros((), jnp.int32),
    )  # pytype: disable=wrong-arg-count
    return simulate.canonicalize_mass(state)

  def apply_fn(
    state: FireBoxDescentState,
    **kwargs,
  ) -> FireBoxDescentState:
    (R, P, F, M, _, ref_box, X_b, P_b, F_b, M_b, bf, dt, alpha, n_pos) = (
      dataclasses.unpack(state)
    )

    dt_2 = f32(dt / 2)

    # TODO(ag): Add different integration schemes following:
    # https://doi.org/10.1016/j.commatsci.2020.109584
    # velocity Verlet: half-step momenta
    P = P + dt_2 * F
    P_b = P_b + dt_2 * F_b

    # position update
    F_deform = X_b / bf
    box = F_deform @ ref_box
    R = shift_fn(R, dt * P / M, box=box, **kwargs)
    X_b = X_b + dt * P_b / M_b
    F_deform = X_b / bf
    box = F_deform @ ref_box

    # recompute forces at new (R, box)
    F = force_fn(R, box=box, **kwargs)
    virial = virial_fn(R, box, **kwargs)
    F_b = box_force_fn(virial, F_deform, bf)

    # velocity Verlet: half-step momenta
    P = P + dt_2 * F
    P_b = P_b + dt_2 * F_b

    # FIRE: combined power check
    F_dot_P = jnp.sum(F * P) + jnp.sum(F_b * P_b)

    F_norm = jnp.sqrt(jnp.sum(F**2) + jnp.sum(F_b**2) + 1e-6)
    P_norm = jnp.sqrt(jnp.sum(P**2) + jnp.sum(P_b**2))

    # FIRE: momentum mixing
    P = P + alpha * (F * P_norm / F_norm - P)
    P_b = P_b + alpha * (F_b * P_norm / F_norm - P_b)

    # FIRE: adaptive dt / alpha
    n_pos = jnp.where(F_dot_P >= 0, n_pos + 1, 0)
    dt_choice = jnp.array([dt * f_inc, dt_max])
    dt = jnp.where(
      F_dot_P > 0,
      jnp.where(n_pos > n_min, jnp.min(dt_choice), dt),
      dt,
    )
    dt = jnp.where(F_dot_P < 0, dt * f_dec, dt)
    alpha = jnp.where(
      F_dot_P > 0,
      jnp.where(n_pos > n_min, alpha * f_alpha, alpha),
      alpha,
    )
    alpha = jnp.where(F_dot_P < 0, alpha_start, alpha)

    # reset momenta on overshoot
    keep = f32(F_dot_P >= 0)
    P = keep * P
    P_b = keep * P_b

    return FireBoxDescentState(
      R,
      P,
      F,
      M,
      box,
      ref_box,
      X_b,
      P_b,
      F_b,
      M_b,
      bf,
      dt,
      alpha,
      n_pos,
    )  # pytype: disable=wrong-arg-count

  return init_fn, apply_fn
