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
from jax.scipy.sparse.linalg import cg
from jax.tree_util import tree_leaves, tree_map, tree_reduce

from jax_md import quantity
from jax_md import dataclasses
from jax_md import partition
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


@dataclasses.dataclass
class PreconFireDescentState:
  """State for the preconditioned FIRE minimizer.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    velocity: The current optimizer velocity. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current raw force on particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    dt: A float specifying the current step size.
    alpha: A float specifying the current FIRE mixing parameter.
    n_pos: The number of consecutive steps with positive power.
    initialized: Whether the first-step velocity initialization has happened.
    preconditioner_position: Reference positions for cached graph
      preconditioners.
    preconditioner_previous_position: Previous positions used for
      preconditioner rebuild checks.
    preconditioner_previous_initialized: Whether
      ``preconditioner_previous_position`` has been initialized.
    momentum: Alias for ``velocity`` for compatibility with momentum-based
      minimizer states.
  """

  position: Array
  velocity: Array
  force: Array
  dt: float
  alpha: float
  n_pos: int
  initialized: bool
  preconditioner_position: PyTree
  preconditioner_previous_position: PyTree
  preconditioner_previous_initialized: bool

  @property
  def momentum(self) -> Array:
    return self.velocity


def exp_preconditioner(
  displacement_fn: space.DisplacementFn,
  r_cut: Array | float | None = None,
  r_NN: Array | float | None = None,
  A: float = 3.0,
  mu: float = 1.0,
  c_stab: float = 0.1,
  solve_tol: float = 1e-5,
  maxiter: int | None = None,
  reference_position: Array | None = None,
  solver: str = 'cg',
) -> Tuple[Callable[..., Array], Callable[..., Array]]:
  """Builds exponential graph preconditioner callables.

  This implements the position-only universal preconditioner from Packwood
  et al., J. Chem. Phys. 144, 164109 (2016), in a matrix-free, JAX-friendly
  form. The scalar atom graph is expanded over Cartesian components as
  ``P = L kron I_d``.

  Args:
    displacement_fn: Function returning pair displacements.
    r_cut: Neighbor cutoff. If ``None``, uses ``2 * r_NN``.
    r_NN: Nearest-neighbor distance. If ``None``, estimated from ``R``.
    A: Exponential decay parameter. ``A=0`` gives constant neighbor weights.
    mu: Energy scale multiplying the preconditioner.
    c_stab: Diagonal stabilization coefficient.
    solve_tol: Conjugate-gradient solve tolerance.
    maxiter: Optional maximum CG iterations.
    reference_position: Optional positions used to build a fixed preconditioner.
      If ``None``, the graph is rebuilt from the current ``R`` on every call.
    solver: Linear solver to use. ``'cg'`` is matrix-free; ``'dense'`` builds
      the scalar atom matrix and uses a direct dense solve.

  Returns:
    ``(preconditioner, preconditioner_dot)`` callables suitable for
    :func:`precon_fire_descent`.
  """

  def pair_distances(R: Array, **kwargs) -> Array:
    d = jax.vmap(
      jax.vmap(
        lambda Ra, Rb: displacement_fn(Ra, Rb, **kwargs), in_axes=(None, 0)
      ),
      in_axes=(0, None),
    )(R, R)
    return jnp.sqrt(jnp.sum(d**2, axis=-1))

  def estimate_r_NN(dist: Array) -> Array:
    N = dist.shape[0]
    dist_no_self = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, dist)
    return jnp.max(jnp.min(dist_no_self, axis=1))

  def graph_weights(R: Array, **kwargs) -> Array:
    R_graph = kwargs.get('preconditioner_position', reference_position)
    R_graph = R if R_graph is None else R_graph
    N = R_graph.shape[0]
    A_value = jnp.asarray(A, dtype=R_graph.dtype)
    if r_NN is None:
      r_NN_value = estimate_r_NN(pair_distances(R_graph, **kwargs))
    else:
      r_NN_value = jnp.asarray(r_NN, dtype=R_graph.dtype)
    r_cut_value = (
      2.0 * r_NN_value
      if r_cut is None
      else jnp.asarray(r_cut, dtype=R_graph.dtype)
    )

    neighbor = kwargs.get('neighbor')
    if neighbor is not None:
      if hasattr(neighbor, 'shifts'):
        if neighbor.format is partition.Dense:
          idx = neighbor.idx
          senders = idx.reshape(-1)
          receivers = jnp.repeat(jnp.arange(N), idx.shape[1])
          shifts = neighbor.shifts.reshape((-1, R_graph.shape[-1]))
        else:
          receivers, senders = neighbor.idx
          shifts = neighbor.shifts
        valid = jnp.logical_and(receivers < N, senders < N)
        send_safe = jnp.clip(senders, 0, N - 1)
        recv_safe = jnp.clip(receivers, 0, N - 1)
        shift_cart = shifts.astype(R_graph.dtype) @ neighbor.box.T
        dR = R_graph[send_safe] + shift_cart - R_graph[recv_safe]
        dist = jnp.sqrt(jnp.sum(dR**2, axis=-1))
        weights_edge = jnp.exp(-A_value * (dist / r_NN_value - 1.0))
        weights_edge = jnp.where(valid, weights_edge, 0.0)
        weights = (
          jnp.zeros((N, N), dtype=R_graph.dtype)
          .at[recv_safe, send_safe]
          .add(weights_edge)
        )
        if neighbor.format is partition.OrderedSparse:
          weights = weights.at[send_safe, recv_safe].add(weights_edge)
        return weights

      if neighbor.format is partition.Dense:
        senders = neighbor.idx.reshape(-1)
        receivers = jnp.repeat(jnp.arange(N), neighbor.idx.shape[1])
      else:
        receivers, senders = neighbor.idx
      valid = jnp.logical_and(receivers < N, senders < N)
      send_safe = jnp.clip(senders, 0, N - 1)
      recv_safe = jnp.clip(receivers, 0, N - 1)
      dR = jax.vmap(
        lambda r_recv, r_send: displacement_fn(r_recv, r_send, **kwargs)
      )(R_graph[recv_safe], R_graph[send_safe])
      dist = jnp.sqrt(jnp.sum(dR**2, axis=-1))
      weights_edge = jnp.exp(-A_value * (dist / r_NN_value - 1.0))
      weights_edge = jnp.where(valid, weights_edge, 0.0)
      weights = (
        jnp.zeros((N, N), dtype=R_graph.dtype)
        .at[recv_safe, send_safe]
        .add(weights_edge)
      )
      if neighbor.format is partition.OrderedSparse:
        weights = weights.at[send_safe, recv_safe].add(weights_edge)
      return weights

    dist = pair_distances(R_graph, **kwargs)
    mask = jnp.logical_and(dist < r_cut_value, ~jnp.eye(N, dtype=bool))
    weights = jnp.exp(-A_value * (dist / r_NN_value - 1.0))
    return jnp.where(mask, weights, 0.0)

  def graph_matrix(R: Array, **kwargs) -> Array:
    weights = graph_weights(R, **kwargs)
    degree = jnp.sum(weights, axis=1)
    R_graph = kwargs.get('preconditioner_position', reference_position)
    R_graph = R if R_graph is None else R_graph
    mu_value = jnp.asarray(mu, dtype=R_graph.dtype)
    c_stab_value = jnp.asarray(c_stab, dtype=R_graph.dtype)
    return mu_value * (jnp.diag(degree + c_stab_value) - weights)

  def graph_matvec(R: Array, X: Array, **kwargs) -> Array:
    weights = graph_weights(R, **kwargs)
    degree = jnp.sum(weights, axis=1)
    mu_value = jnp.asarray(mu, dtype=X.dtype)
    c_stab_value = jnp.asarray(c_stab, dtype=X.dtype)
    return mu_value * ((degree[:, None] + c_stab_value) * X - weights @ X)

  def preconditioner(R: Array, F: Array, **kwargs) -> Array:
    if solver == 'dense':
      return jnp.linalg.solve(graph_matrix(R, **kwargs), F)
    solve = lambda X: graph_matvec(R, X, **kwargs)
    return cg(solve, F, tol=solve_tol, maxiter=maxiter)[0]

  def preconditioner_dot(R: Array, X: Array, Y: Array, **kwargs) -> Array:
    return jnp.sum(X * graph_matvec(R, Y, **kwargs))

  return preconditioner, preconditioner_dot


def c1_preconditioner(
  displacement_fn: space.DisplacementFn,
  r_cut: Array | float | None = None,
  r_NN: Array | float | None = None,
  mu: float = 1.0,
  c_stab: float = 0.1,
  solve_tol: float = 1e-5,
  maxiter: int | None = None,
  reference_position: Array | None = None,
  solver: str = 'cg',
) -> Tuple[Callable[..., Array], Callable[..., Array]]:
  """Builds C1 graph preconditioner callables.

  This is the constant-weight special case of :func:`exp_preconditioner`.
  """
  return exp_preconditioner(
    displacement_fn,
    r_cut=r_cut,
    r_NN=r_NN,
    A=0.0,
    mu=mu,
    c_stab=c_stab,
    solve_tol=solve_tol,
    maxiter=maxiter,
    reference_position=reference_position,
    solver=solver,
  )


def estimate_exp_mu(
  energy_or_force: Callable[..., Array],
  displacement_fn: space.DisplacementFn,
  R: Array,
  r_cut: Array | float | None = None,
  r_NN: Array | float | None = None,
  A: float = 3.0,
  c_stab: float = 0.1,
  min_mu: float = 1.0,
  **kwargs,
) -> Array:
  """Estimate the Exp preconditioner energy scale.

  The estimate matches the finite-difference curvature equation from
  Packwood et al., J. Chem. Phys. 144, 164109 (2016):

  ``(grad E(R + V) - grad E(R)) . V = mu * V^T P_{mu=1} V``.

  Args:
    energy_or_force: Function producing either an energy or a force.
    displacement_fn: Function returning pair displacements.
    R: Reference positions.
    r_cut: Neighbor cutoff. If ``None``, uses ``2 * r_NN``.
    r_NN: Nearest-neighbor distance. If ``None``, estimated from ``R``.
    A: Exponential decay parameter for the unit preconditioner.
    c_stab: Diagonal stabilization coefficient.
    min_mu: Lower bound applied to the estimate, matching ASE's cap at 1.
    **kwargs: Extra arguments forwarded to ``energy_or_force`` and
      ``displacement_fn``.

  Returns:
    Scalar estimate for ``mu``.
  """
  force = quantity.canonicalize_force(energy_or_force)

  def pair_distances(R: Array) -> Array:
    d = jax.vmap(
      jax.vmap(
        lambda Ra, Rb: displacement_fn(Ra, Rb, **kwargs), in_axes=(None, 0)
      ),
      in_axes=(0, None),
    )(R, R)
    return jnp.sqrt(jnp.sum(d**2, axis=-1))

  def estimate_r_NN_from_R(R: Array) -> Array:
    dist = pair_distances(R)
    N = dist.shape[0]
    dist_no_self = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, dist)
    return jnp.max(jnp.min(dist_no_self, axis=1))

  r_NN_value = estimate_r_NN_from_R(R) if r_NN is None else r_NN
  amplitude = 1e-2 * r_NN_value
  L = jnp.max(R, axis=0) - jnp.min(R, axis=0)
  safe_L = jnp.where(L == 0, 1.0, L)
  mode = jnp.where(L == 0, 0.0, jnp.sin(R / safe_L))
  V = amplitude * mode

  _, unit_dot = exp_preconditioner(
    displacement_fn,
    r_cut=r_cut,
    r_NN=r_NN_value,
    A=A,
    mu=1.0,
    c_stab=c_stab,
  )
  F = force(R, **kwargs)
  F_plus = force(R + V, **kwargs)
  lhs = jnp.sum((F - F_plus) * V)
  rhs = unit_dot(R, V, V, **kwargs)
  return jnp.maximum(lhs / rhs, min_mu)


def precon_fire_descent(
  energy_or_force: Callable[..., Array],
  shift_fn: ShiftFn,
  dt_start: float = 0.1,
  dt_max: float = 1.0,
  max_move: float = 0.2,
  n_min: float = 5,
  f_inc: float = 1.1,
  f_dec: float = 0.5,
  alpha_start: float = 0.1,
  f_alpha: float = 0.99,
  theta: float = 0.1,
  armijo_tol: float = 0.0,
  use_armijo: bool = True,
  preconditioner_update_threshold: float | None = None,
  preconditioner: Callable[..., PyTree] | None = None,
  preconditioner_dot: Callable[..., Array] | None = None,
) -> Minimizer[PreconFireDescentState]:
  """Defines preconditioned FIRE minimization.

  This optimizer uses the graph-metric preconditioning idea of Packwood et al.,
  J. Chem. Phys. 144, 164109 (2016). Unlike :func:`fire_descent`, it stores
  velocity rather than momentum and applies the preconditioned force direction
  directly.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt_start: The initial step size during minimization as a float.
    dt_max: The maximum step size during minimization as a float.
    max_move: The maximum Euclidean norm of the displacement in one step.
    n_min: An integer specifying the minimum number of positive-power steps
      before ``dt`` and ``alpha`` should be updated.
    f_inc: A float specifying the fractional rate by which the step size
      should be increased.
    f_dec: A float specifying the fractional rate by which the step size
      should be decreased.
    alpha_start: A float specifying the initial FIRE mixing parameter.
    f_alpha: A float specifying the fractional change in ``alpha``.
    theta: Armijo sufficient-decrease parameter.
    armijo_tol: Numerical tolerance for the Armijo comparison. Trials within
      this tolerance of the sufficient-decrease boundary are rejected. The
      default ``0.0`` matches ASE's strict comparison.
    use_armijo: Whether to use ASE's Armijo trial-step rejection. If ``True``,
      ``energy_or_force`` must be an energy function.
    preconditioner_update_threshold: Optional maximum absolute displacement
      threshold for updating cached preconditioner reference positions. ASE
      uses ``0.5 * r_NN``.
    preconditioner: Optional callable that returns the preconditioned force
      direction ``H^{-1} F``. It should take ``(R, F, **kwargs)`` and return a
      PyTree with the same structure as ``F``.
    preconditioner_dot: Optional callable for the preconditioner metric. When
      ``preconditioner`` is supplied, this must take ``(R, X, Y, **kwargs)``
      and return ``X^T H Y``.

  Returns:
    See above.
  """
  if preconditioner is not None and preconditioner_dot is None:
    raise ValueError(
      'preconditioner_dot must be supplied when preconditioner is supplied.'
    )

  force = quantity.canonicalize_force(energy_or_force)

  def tree_dot(X: PyTree, Y: PyTree) -> Array:
    return tree_reduce(
      lambda accum, x_dot_y: accum + x_dot_y,
      tree_map(lambda x, y: jnp.sum(x * y), X, Y),
      0.0,
    )

  def preconditioned_force(R: PyTree, F: PyTree, **kwargs) -> PyTree:
    if preconditioner is None:
      return F
    return preconditioner(R, F, **kwargs)

  def velocity_metric_dot(R: PyTree, X: PyTree, Y: PyTree, **kwargs) -> Array:
    if preconditioner is None:
      return tree_dot(X, Y)
    metric = preconditioner_dot
    if metric is None:
      raise ValueError(
        'preconditioner_dot must be supplied when preconditioner is supplied.'
      )
    return metric(R, X, Y, **kwargs)

  def shift_position(R: PyTree, dR: PyTree, **kwargs) -> PyTree:
    s_fn = shift_fn
    if isinstance(s_fn, Callable):
      s_fn = tree_map(lambda r: shift_fn, R)
    return tree_map(lambda s, r, dr: s(r, dr, **kwargs), s_fn, R, dR)

  def init_fn(R: PyTree, **kwargs) -> PreconFireDescentState:
    dtype = tree_leaves(R)[0].dtype
    V = tree_map(lambda x: jnp.zeros_like(x), R)
    F = force(R, **kwargs)
    return PreconFireDescentState(
      R,
      V,
      F,
      jnp.asarray(dt_start, dtype=dtype),
      jnp.asarray(alpha_start, dtype=dtype),
      jnp.zeros((), jnp.int32),
      jnp.array(False),
      R,
      R,
      jnp.array(False),
    )  # pytype: disable=wrong-arg-count

  def apply_fn(
    state: PreconFireDescentState, **kwargs
  ) -> PreconFireDescentState:
    (
      R,
      V,
      F,
      dt,
      alpha,
      n_pos,
      initialized,
      precon_R,
      precon_previous_R,
      precon_previous_initialized,
    ) = dataclasses.unpack(state)

    def tree_max_abs(X: PyTree) -> Array:
      return tree_reduce(
        lambda accum, x: jnp.maximum(accum, jnp.max(jnp.abs(x))),
        X,
        0.0,
      )

    if preconditioner_update_threshold is not None:
      max_old_disp = tree_max_abs(
        tree_map(lambda r, r0: r - r0, R, precon_previous_R)
      )
      update_precon = jnp.logical_and(
        initialized,
        jnp.logical_and(
          precon_previous_initialized,
          max_old_disp >= preconditioner_update_threshold,
        ),
      )
      precon_R = tree_map(
        lambda r, r0: jnp.where(update_precon, r, r0), R, precon_R
      )
      set_old = initialized
      precon_previous_R = tree_map(
        lambda r, r0: jnp.where(set_old, r, r0), R, precon_previous_R
      )
      precon_previous_initialized = jnp.logical_or(
        precon_previous_initialized, initialized
      )

    precon_kwargs = dict(kwargs)
    precon_kwargs['preconditioner_position'] = precon_R
    G = preconditioned_force(R, F, **precon_kwargs)

    def finish_step(V_bar, dt_bar, alpha_bar, n_pos_bar):
      V_new = tree_map(lambda v, g: v + dt_bar * g, V_bar, G)
      dR_raw = tree_map(lambda v: dt_bar * v, V_new)
      dR_norm = jnp.sqrt(tree_dot(dR_raw, dR_raw))
      dR_scale = jnp.where(dR_norm > max_move, max_move / dR_norm, 1.0)
      dR = tree_map(lambda dr: dR_scale * dr, dR_raw)

      R_new = shift_position(R, dR, **kwargs)
      F_new = force(R_new, **kwargs)

      return PreconFireDescentState(
        R_new,
        V_new,
        F_new,
        dt_bar,
        alpha_bar,
        n_pos_bar,
        jnp.array(True),
        precon_R,
        precon_previous_R,
        precon_previous_initialized,
      )  # pytype: disable=wrong-arg-count

    F_dot_V = tree_dot(F, V)
    is_first = jnp.logical_not(initialized)
    is_positive = jnp.logical_and(initialized, F_dot_V > 0)

    V_zero = tree_map(jnp.zeros_like, V)
    V_norm = jnp.sqrt(velocity_metric_dot(R, V, V, **precon_kwargs))
    F_norm = jnp.sqrt(tree_dot(F, G))
    V_positive = tree_map(
      lambda v, g: (1.0 - alpha) * v + alpha * (V_norm / F_norm) * g,
      V,
      G,
    )

    grow = n_pos > n_min
    dt_choice = jnp.array([dt * f_inc, dt_max])
    dt_positive = jnp.where(grow, jnp.min(dt_choice), dt)
    alpha_positive = jnp.where(grow, alpha * f_alpha, alpha)
    n_pos_positive = n_pos + 1

    V_bar = tree_map(
      lambda v_pos, v_zero: jnp.where(is_positive, v_pos, v_zero),
      V_positive,
      V_zero,
    )
    dt_bar = jnp.where(is_positive, dt_positive, dt * f_dec)
    alpha_bar = jnp.where(is_positive, alpha_positive, alpha_start)
    n_pos_bar = jnp.where(is_positive, n_pos_positive, jnp.zeros((), jnp.int32))

    V_bar = tree_map(
      lambda v_first, v_later: jnp.where(is_first, v_first, v_later),
      V_zero,
      V_bar,
    )
    dt_bar = jnp.where(is_first, dt, dt_bar)
    alpha_bar = jnp.where(is_first, alpha, alpha_bar)
    n_pos_bar = jnp.where(is_first, n_pos, n_pos_bar)

    armijo_fail = jnp.array(False)
    if use_armijo:
      V_test = tree_map(lambda v, g: v + dt * g, V, G)
      dR_test = tree_map(lambda v: dt * v, V_test)
      R_test = shift_position(R, dR_test, **kwargs)
      E = energy_or_force(R, **kwargs)
      E_test = energy_or_force(R_test, **kwargs)
      armijo_rhs = E - theta * dt * tree_dot(V_test, F)
      failed_decrease = E_test > armijo_rhs - armijo_tol
      armijo_fail = jnp.logical_and(
        initialized, jnp.logical_or(failed_decrease, ~jnp.isfinite(E_test))
      )

    V_bar = tree_map(
      lambda v: jnp.where(armijo_fail, jnp.zeros_like(v), v), V_bar
    )
    dt_bar = jnp.where(armijo_fail, dt * f_dec, dt_bar)
    alpha_bar = jnp.where(armijo_fail, alpha_start, alpha_bar)
    n_pos_bar = jnp.where(armijo_fail, jnp.zeros((), jnp.int32), n_pos_bar)

    return finish_step(V_bar, dt_bar, alpha_bar, n_pos_bar)

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


@dataclasses.dataclass
class PreconFireBoxDescentState:
  """State for the atom + box preconditioned FIRE minimizer."""

  position: Array
  velocity: Array
  force: Array
  box: Array
  reference_box: Array
  box_position: Array
  box_velocity: Array
  box_force: Array
  box_factor: Array
  dt: float
  alpha: float
  n_pos: int
  initialized: bool
  preconditioner_position: PyTree
  preconditioner_previous_position: PyTree
  preconditioner_previous_initialized: bool

  @property
  def momentum(self) -> Array:
    return self.velocity

  @property
  def box_momentum(self) -> Array:
    return self.box_velocity


def precon_fire_descent_box(
  energy_fn: Callable[..., Array],
  shift_fn: ShiftFn,
  dt_start: float = 0.1,
  dt_max: float = 1.0,
  max_move: float = 0.2,
  n_min: float = 5,
  f_inc: float = 1.1,
  f_dec: float = 0.5,
  alpha_start: float = 0.1,
  f_alpha: float = 0.99,
  theta: float = 0.1,
  armijo_tol: float = 0.0,
  use_armijo: bool = True,
  preconditioner: Callable[..., PyTree] | None = None,
  preconditioner_dot: Callable[..., Array] | None = None,
  cell_preconditioner: Array | float = 1.0,
  preconditioner_update_threshold: float | None = None,
  scalar_pressure: float = 0.0,
  hydrostatic_strain: bool = False,
  constant_volume: bool = False,
  mask: Array = None,
) -> Minimizer[PreconFireBoxDescentState]:
  """Preconditioned FIRE minimization of atoms and the simulation box."""
  del shift_fn

  if preconditioner is not None and preconditioner_dot is None:
    raise ValueError(
      'preconditioner_dot must be supplied when preconditioner is supplied.'
    )

  force_fn = quantity.canonicalize_force(energy_fn)

  def tree_dot(X: PyTree, Y: PyTree) -> Array:
    return tree_reduce(
      lambda accum, x_dot_y: accum + x_dot_y,
      tree_map(lambda x, y: jnp.sum(x * y), X, Y),
      0.0,
    )

  def preconditioned_force(R: PyTree, F: PyTree, **kwargs) -> PyTree:
    if preconditioner is None:
      return F
    return preconditioner(R, F, **kwargs)

  def velocity_metric_dot(R: PyTree, X: PyTree, Y: PyTree, **kwargs) -> Array:
    if preconditioner is None:
      return tree_dot(X, Y)
    metric = preconditioner_dot
    if metric is None:
      raise ValueError(
        'preconditioner_dot must be supplied when preconditioner is supplied.'
      )
    return metric(R, X, Y, **kwargs)

  def tree_max_abs(X: PyTree) -> Array:
    return tree_reduce(
      lambda accum, x: jnp.maximum(accum, jnp.max(jnp.abs(x))),
      X,
      0.0,
    )

  def virial_fn(R, box, **kwargs):
    dim = R.shape[1]
    I_d = jnp.eye(dim, dtype=box.dtype)
    zero = jnp.zeros((dim, dim), dtype=box.dtype)

    def U(eps):
      return energy_fn(R, box=box, perturbation=(I_d + eps), **kwargs)

    dUdeps = jax.grad(U)(zero)
    v = -dUdeps
    v = (v + v.T) / 2.0
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

  def actual_box(ref_box, F_deform):
    return ref_box @ F_deform.T

  def actual_position(q_R, F_deform):
    return q_R @ F_deform.T

  def generalized_position(R, F_deform):
    return jnp.linalg.solve(F_deform, R.T).T

  def generalized_force(F_real, F_deform):
    return F_real @ F_deform

  def box_force_fn(virial, F_deform, bf):
    W_F = jnp.linalg.solve(F_deform, virial.T).T
    return W_F / bf

  def init_fn(
    R: Array,
    box: Array,
    box_factor: Array = None,
    **kwargs,
  ) -> PreconFireBoxDescentState:
    dtype = tree_leaves(R)[0].dtype
    N = R.shape[0]
    dim = R.shape[1]
    if box_factor is None:
      box_factor = jnp.asarray(N, dtype=dtype)
    I_d = jnp.eye(dim, dtype=box.dtype)
    F_deform = I_d
    q_R = generalized_position(R, F_deform)
    R_actual = actual_position(q_R, F_deform)
    F_real = force_fn(R_actual, box=box, **kwargs)
    F = generalized_force(F_real, F_deform)
    F_b = box_force_fn(virial_fn(R_actual, box, **kwargs), F_deform, box_factor)
    return PreconFireBoxDescentState(  # type: ignore[call-arg]
      q_R,  # type: ignore[call-arg]
      tree_map(jnp.zeros_like, q_R),
      F,
      box,
      box,
      box_factor * I_d,
      jnp.zeros_like(box),
      F_b,
      box_factor,
      jnp.asarray(dt_start, dtype=dtype),
      jnp.asarray(alpha_start, dtype=dtype),
      jnp.zeros((), jnp.int32),
      jnp.array(False),
      R,
      R,
      jnp.array(False),
    )  # pytype: disable=wrong-arg-count

  def apply_fn(
    state: PreconFireBoxDescentState, **kwargs
  ) -> PreconFireBoxDescentState:
    (
      R,
      V,
      F,
      _,
      ref_box,
      X_b,
      V_b,
      F_b,
      bf,
      dt,
      alpha,
      n_pos,
      initialized,
      precon_R,
      precon_previous_R,
      precon_previous_initialized,
    ) = dataclasses.unpack(state)

    F_deform = X_b / bf
    box = actual_box(ref_box, F_deform)
    R_actual = actual_position(R, F_deform)

    if preconditioner_update_threshold is not None:
      max_old_disp = tree_max_abs(
        tree_map(lambda r, r0: r - r0, R, precon_previous_R)
      )
      update_precon = jnp.logical_and(
        initialized,
        jnp.logical_and(
          precon_previous_initialized,
          max_old_disp >= preconditioner_update_threshold,
        ),
      )
      precon_R = tree_map(
        lambda r, r0: jnp.where(update_precon, r, r0), R, precon_R
      )
      precon_previous_R = tree_map(
        lambda r, r0: jnp.where(initialized, r, r0), R, precon_previous_R
      )
      precon_previous_initialized = jnp.logical_or(
        precon_previous_initialized, initialized
      )

    precon_kwargs = dict(kwargs)
    precon_kwargs['box'] = box
    precon_kwargs['preconditioner_position'] = precon_R
    G = preconditioned_force(R, F, **precon_kwargs)
    cell_precon = jnp.asarray(cell_preconditioner, dtype=F_b.dtype)
    G_b = F_b if preconditioner is None else F_b / cell_precon

    F_dot_V = tree_dot(F, V) + jnp.sum(F_b * V_b)
    is_first = jnp.logical_not(initialized)
    is_positive = jnp.logical_and(initialized, F_dot_V > 0)

    V_zero = tree_map(jnp.zeros_like, V)
    V_b_zero = jnp.zeros_like(V_b)
    box_velocity_norm = (
      jnp.sum(V_b**2)
      if preconditioner is None
      else cell_precon * jnp.sum(V_b**2)
    )
    V_norm = jnp.sqrt(
      velocity_metric_dot(R, V, V, **precon_kwargs) + box_velocity_norm
    )
    F_norm = jnp.sqrt(tree_dot(F, G) + jnp.sum(F_b * G_b))
    V_positive = tree_map(
      lambda v, g: (1.0 - alpha) * v + alpha * (V_norm / F_norm) * g,
      V,
      G,
    )
    V_b_positive = (1.0 - alpha) * V_b + alpha * (V_norm / F_norm) * G_b

    grow = n_pos > n_min
    dt_choice = jnp.array([dt * f_inc, dt_max])
    dt_positive = jnp.where(grow, jnp.min(dt_choice), dt)
    alpha_positive = jnp.where(grow, alpha * f_alpha, alpha)
    n_pos_positive = n_pos + 1

    V_bar = tree_map(
      lambda v_pos, v_zero: jnp.where(is_positive, v_pos, v_zero),
      V_positive,
      V_zero,
    )
    V_b_bar = jnp.where(is_positive, V_b_positive, V_b_zero)
    dt_bar = jnp.where(is_positive, dt_positive, dt * f_dec)
    alpha_bar = jnp.where(is_positive, alpha_positive, alpha_start)
    n_pos_bar = jnp.where(is_positive, n_pos_positive, jnp.zeros((), jnp.int32))

    V_bar = tree_map(
      lambda v_first, v_later: jnp.where(is_first, v_first, v_later),
      V_zero,
      V_bar,
    )
    V_b_bar = jnp.where(is_first, V_b_zero, V_b_bar)
    dt_bar = jnp.where(is_first, dt, dt_bar)
    alpha_bar = jnp.where(is_first, alpha, alpha_bar)
    n_pos_bar = jnp.where(is_first, n_pos, n_pos_bar)

    armijo_fail = jnp.array(False)
    if use_armijo:
      V_test = tree_map(lambda v, g: v + dt * g, V, G)
      V_b_test = V_b + dt * G_b
      dR_test = tree_map(lambda v: dt * v, V_test)
      X_b_test = X_b + dt * V_b_test
      box_test = actual_box(ref_box, X_b_test / bf)
      q_R_test = R + dR_test
      E = energy_fn(R_actual, box=box, **kwargs)
      E_test = energy_fn(
        actual_position(q_R_test, X_b_test / bf), box=box_test, **kwargs
      )
      power_test = tree_dot(V_test, F) + jnp.sum(V_b_test * F_b)
      armijo_rhs = E - theta * dt * power_test
      failed_decrease = E_test > armijo_rhs - armijo_tol
      armijo_fail = jnp.logical_and(
        initialized, jnp.logical_or(failed_decrease, ~jnp.isfinite(E_test))
      )

    V_bar = tree_map(
      lambda v: jnp.where(armijo_fail, jnp.zeros_like(v), v), V_bar
    )
    V_b_bar = jnp.where(armijo_fail, V_b_zero, V_b_bar)
    dt_bar = jnp.where(armijo_fail, dt * f_dec, dt_bar)
    alpha_bar = jnp.where(armijo_fail, alpha_start, alpha_bar)
    n_pos_bar = jnp.where(armijo_fail, jnp.zeros((), jnp.int32), n_pos_bar)

    V_new = tree_map(lambda v, g: v + dt_bar * g, V_bar, G)
    V_b_new = V_b_bar + dt_bar * G_b
    dR_raw = tree_map(lambda v: dt_bar * v, V_new)
    dX_b_raw = dt_bar * V_b_new
    dR_norm = jnp.sqrt(tree_dot(dR_raw, dR_raw) + jnp.sum(dX_b_raw**2))
    dR_scale = jnp.where(dR_norm > max_move, max_move / dR_norm, 1.0)
    dR = tree_map(lambda dr: dR_scale * dr, dR_raw)
    dX_b = dR_scale * dX_b_raw

    R_new = R + dR
    X_b_new = X_b + dX_b
    F_deform_new = X_b_new / bf
    box_new = actual_box(ref_box, F_deform_new)
    R_actual_new = actual_position(R_new, F_deform_new)
    F_real_new = force_fn(R_actual_new, box=box_new, **kwargs)
    F_new = generalized_force(F_real_new, F_deform_new)
    F_b_new = box_force_fn(
      virial_fn(R_actual_new, box_new, **kwargs), F_deform_new, bf
    )

    return PreconFireBoxDescentState(  # type: ignore[call-arg]
      R_new,  # type: ignore[call-arg]
      V_new,
      F_new,
      box_new,
      ref_box,
      X_b_new,
      V_b_new,
      F_b_new,
      bf,
      dt_bar,
      alpha_bar,
      n_pos_bar,
      jnp.array(True),
      precon_R,
      precon_previous_R,
      precon_previous_initialized,
    )  # pytype: disable=wrong-arg-count

  return init_fn, apply_fn
