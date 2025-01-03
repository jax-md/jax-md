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

"""Code to simulate systems in various statistical ensembles.

  This file contains a number of different methods that can be used to
  simulate systems in a variety of ensembles.

  In general, simulation code follows the same overall structure as optimizers
  in JAX. Simulations are tuples of two functions:

    init_fn:
      Function that initializes the  state of a system. Should take
      positions as an ndarray of shape `[n, output_dimension]`. Returns a state
      which will be a namedtuple.
    apply_fn:
      Function that takes a state and produces a new state after one
      step of optimization.

  One question that we need to think about is whether the simulations should
  also return a function that computes the invariant for that ensemble. This
  can be used for testing purposes, but is not often used otherwise.
"""

from collections import namedtuple

from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import functools

from brainunit import Quantity
import brainunit as u
from jax import grad
from jax import jit
from jax import random
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten

from jax_md import quantity
from jax_md import util
from jax_md import space
from jax_md import dataclasses
from jax_md import partition
from jax_md import smap

static_cast = util.static_cast


# Types


Array = util.Array
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]


"""Dispatch By State Code.

JAX MD allows for simulations to be extensible using a dispatch strategy where
functions are dispatched to specific cases based on the type of state provided.
In particular, we make decisions about which function to call based on the type
of the position argument. For those familiar with C / C++, our dispatch code is
essentially function overloading based on the type of the positions.

If you are interested in setting up a simulation using a different type of
system you can do so in a relatively light weight manner by introducing a new
type for storing the state that is compatible with the JAX PyTree system
(we usually choose a dataclass) and then overriding the functions below.

These extensions allow a range of simulations to be run by just changing the
type of the position argument. There are essentially two types of functions to
be overloaded. Functions that compute physical quantities, such as the kinetic
energy, and functions that evolve a state according to the Suzuki-Trotter
decomposition. Specifically, one might want to override the position step,
momentum step for deterministic and stochastic simulations or the
`stochastic_step` for stochastic simulations (e.g Langevin).
"""


class dispatch_by_state:
  """Wrap a function and dispatch based on the type of positions."""
  def __init__(self, fn):
    self._fn = fn
    self._registry = {}

  def __call__(self, state, *args, **kwargs):
    if type(state.position) in self._registry:
      return self._registry[type(state.position)](state, *args, **kwargs)
    return self._fn(state, *args, **kwargs)

  def register(self, oftype):
    def register_fn(fn):
      self._registry[oftype] = fn
    return register_fn


@dispatch_by_state
def canonicalize_mass(state: T) -> T:
  """Reshape mass vector for broadcasting with positions."""
  def canonicalize_fn(mass):
    # if isinstance(mass, Quantity):
    #   mass = mass.to_decimal(u.atomic_mass)
    if isinstance(mass, float):
      return mass
    if mass.ndim == 2 and mass.shape[1] == 1:
      return mass
    elif mass.ndim == 1:
      return u.math.reshape(mass, (mass.shape[0], 1))
    elif mass.ndim == 0:
      return mass
    msg = (
      'Expected mass to be either a floating point number or a one-dimensional'
      'ndarray. Found {}.'.format(mass)
    )
    raise ValueError(msg)
  return state.set(mass=tree_map(canonicalize_fn, state.mass))

@dispatch_by_state
def initialize_momenta(state: T, key: Array, kT: float) -> T:
  """Initialize momenta with the Maxwell-Boltzmann distribution."""
  R, mass = state.position, state.mass
  if isinstance(R, Quantity) and isinstance(mass, Quantity) and isinstance(kT, Quantity):
    R = R.to_decimal(u.angstrom)
    mass = mass.to_decimal(u.atomic_mass)
    kT = kT.to_decimal(u.fsecond)
    return_quantity = True
  elif isinstance(R, Quantity) or isinstance(mass, Quantity) or isinstance(kT, Quantity):
    raise ValueError('r, m, and kT must all be Quantities or none of them.')
  else:
    return_quantity = False

  R, treedef = tree_flatten(R)
  mass, _ = tree_flatten(mass)
  keys = random.split(key, len(R))

  def initialize_fn(k, r, m):
    p = jnp.sqrt(m * kT) * random.normal(k, r.shape, dtype=r.dtype)
    # If simulating more than one particle, center the momentum.
    if r.shape[0] > 1:
      p = p - jnp.mean(p, axis=0, keepdims=True)
    return p

  P = [initialize_fn(k, r, m) for k, r, m in zip(keys, R, mass)]

  return state.set(momentum=tree_unflatten(treedef, P) * u.IMF * u.angstrom if return_quantity else tree_unflatten(treedef, P))


@dispatch_by_state
def momentum_step(state: T, dt: float) -> T:
  """Apply a single step of the time evolution operator for momenta."""
  assert hasattr(state, 'momentum')
  if isinstance(state.momentum, Quantity) and isinstance(state.force, Quantity) and isinstance(dt, Quantity):
    dt = dt.to_decimal(dt.unit)
    new_momentum = tree_map(lambda p, f: p + dt * f,
                            state.momentum.to_decimal(u.IMF * u.angstrom),
                            state.force.to_decimal(u.IMF)) * u.IMF * u.angstrom
  elif isinstance(state.momentum, Quantity) or isinstance(state.mass, Quantity) or isinstance(dt, Quantity):
    raise ValueError('state.momentum, state.mass, and dt must all be Quantities or none of them.')
  else:
    new_momentum = tree_map(lambda p, f: p + dt * f,
                            state.momentum,
                            state.force)
  return state.set(momentum=new_momentum)


@dispatch_by_state
def position_step(state: T, shift_fn: Callable, dt: float, **kwargs) -> T:
  """Apply a single step of the time evolution operator for positions."""
  if isinstance(shift_fn, Callable):
    shift_fn = tree_map(lambda r: shift_fn, state.position.mantissa if isinstance(state.position, Quantity) else state.position)
  if isinstance(state.position, Quantity) and isinstance(state.momentum, Quantity) and isinstance(state.mass, Quantity) and isinstance(dt, Quantity):
    dt = dt.to_decimal(dt.unit)
    dr = tree_map(lambda p, m: dt * p / m,
                        state.momentum.to_decimal(u.IMF * u.angstrom),
                        state.mass.to_decimal(u.atomic_mass)) * u.angstrom
    new_position = tree_map(lambda s_fn, r, d: s_fn(r, d, **kwargs),
                        shift_fn,
                        state.position,
                        dr)
  else:
    new_position = tree_map(lambda s_fn, r, p, m: s_fn(r, dt * p / m, **kwargs),
                            shift_fn,
                            state.position,
                            state.momentum,
                            state.mass)
  return state.set(position=new_position)


@dispatch_by_state
def kinetic_energy(state: T) -> Array:
  """Compute the kinetic energy of a state."""
  return quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)


@dispatch_by_state
def temperature(state: T) -> Array:
  """Compute the temperature of a state."""
  return quantity.temperature(momentum=state.momentum, mass=state.mass)


"""Deterministic Simulations

JAX MD includes integrators for deterministic simulations of the NVE, NVT, and
NPT ensembles. For a qualitative description of statistical physics ensembles
see the wikipedia article here:
en.wikipedia.org/wiki/Statistical_ensemble_(mathematical_physics)

Integrators are based direct translation method outlined in the paper,

"A Liouville-operator derived measure-preserving integrator for molecular
dynamics simulations in the isothermal–isobaric ensemble"

M. E. Tuckerman, J. Alejandre, R. López-Rendón, A. L Jochim, and G. J. Martyna
J. Phys. A: Math. Gen. 39 5629 (2006)

As such, we define several primitives that are generically useful in describing
simulations of this type. Namely, the velocity-Verlet integration step that is
used in the NVE and NVT simulations. We also define a general Nose-Hoover chain
primitive that is used to couple components of the system to a chain that
regulates the temperature. These primitives can be combined to construct more
interesting simulations that involve e.g. temperature gradients.
"""


def velocity_verlet(force_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    state: T,
                    **kwargs) -> T:
  """Apply a single step of velocity Verlet integration to a state."""
  return_quantity = False
  if isinstance(dt, Quantity):
    dt = f32(dt.to_decimal(dt.unit)) * dt.unit
    dt_2 = f32(dt.to_decimal(dt.unit) / 2) * dt.unit
    return_quantity = True
  else:
    dt = f32(dt)
    dt_2 = f32(dt / 2)

  state = momentum_step(state, dt_2)
  state = position_step(state, shift_fn, dt, **kwargs)
  state = state.set(force=force_fn(state.position, **kwargs) * u.IMF if return_quantity else force_fn(state.position, **kwargs))
  state = momentum_step(state, dt_2)

  return state


# Constant Energy Simulations


@dataclasses.dataclass
class NVEState:
  """A struct containing the state of an NVE simulation.

  This tuple stores the state of a simulation that samples from the
  microcanonical ensemble in which the (N)umber of particles, the (V)olume, and
  the (E)nergy of the system are held fixed.

  Attributes:
    position: An ndarray of shape `[n, spatial_dimension]` storing the position
      of particles.
    momentum: An ndarray of shape `[n, spatial_dimension]` storing the momentum
      of particles.
    force: An ndarray of shape `[n, spatial_dimension]` storing the force
      acting on particles from the previous step.
    mass: A float or an ndarray of shape `[n]` containing the masses of the
      particles.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass


# pylint: disable=invalid-name
def nve(energy_or_force_fn, shift_fn, dt=1e-3, **sim_kwargs):
  """Simulates a system in the NVE ensemble.

  Samples from the microcanonical ensemble in which the number of particles
  (N), the system volume (V), and the energy (E) are held constant. We use a
  standard velocity Verlet integration scheme.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
  Returns:
    See above.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  @jit
  def init_fn(key, R, kT, mass=f32(1.0), **kwargs):
    force = force_fn(R, **kwargs)
    state = NVEState(R, None, force, mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, key, kT)

  @jit
  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    return velocity_verlet(force_fn, shift_fn, _dt, state, **kwargs)

  return init_fn, step_fn


# Constant Temperature Simulations


# Suzuki-Yoshida weights for integrators of different order.
# These are copied from OpenMM at
# https://github.com/openmm/openmm/blob/master/openmmapi/src/NoseHooverChain.cpp


SUZUKI_YOSHIDA_WEIGHTS = {
    1: [1],
    3: [0.828981543588751, -0.657963087177502, 0.828981543588751],
    5: [0.2967324292201065, 0.2967324292201065, -0.186929716880426,
        0.2967324292201065, 0.2967324292201065],
    7: [0.784513610477560, 0.235573213359357, -1.17767998417887,
        1.31518632068391, -1.17767998417887, 0.235573213359357,
        0.784513610477560]
}


@dataclasses.dataclass
class NoseHooverChain:
  """State information for a Nose-Hoover chain.

  Attributes:
    position: An ndarray of shape `[chain_length]` that stores the position of
      the chain.
    momentum: An ndarray of shape `[chain_length]` that stores the momentum of
      the chain.
    mass: An ndarray of shape `[chain_length]` that stores the mass of the
      chain.
    tau: The desired period of oscillation for the chain. Longer periods result
      is better stability but worse temperature control.
    kinetic_energy: A float that stores the current kinetic energy of the
      system that the chain is coupled to.
    degrees_of_freedom: An integer specifying the number of degrees of freedom
      that the chain is coupled to.
  """
  position: Array
  momentum: Array
  mass: Array
  tau: Array
  kinetic_energy: Array
  degrees_of_freedom: int=dataclasses.static_field()


@dataclasses.dataclass
class NoseHooverChainFns:
  initialize: Callable
  half_step: Callable
  update_mass: Callable


def nose_hoover_chain(dt: float,
                      chain_length: int,
                      chain_steps: int,
                      sy_steps: int,
                      tau: float
                      ) -> NoseHooverChainFns:
  """Helper function to simulate a Nose-Hoover Chain coupled to a system.

  This function is used in simulations that sample from thermal ensembles by
  coupling the system to one, or more, Nose-Hoover chains. We use the direct
  translation method outlined in Martyna et al. [#martyna92]_ and the
  Nose-Hoover chains are updated using two half steps: one at the beginning of
  a simulation step and one at the end. The masses of the Nose-Hoover chains
  are updated automatically to enforce a specific period of oscillation, `tau`.
  Larger values of `tau` will yield systems that reach the target temperature
  more slowly but are also more stable.

  As described in Martyna et al. [#martyna92]_, the Nose-Hoover chain often
  evolves on a faster timescale than the rest of the simulation. Therefore, it
  sometimes necessary
  to integrate the chain over several substeps for each step of MD. To do this
  we follow the Suzuki-Yoshida scheme. Specifically, we subdivide our chain
  simulation into :math:`n_c` substeps. These substeps are further subdivided
  into :math:`n_sy` steps. Each :math:`n_sy` step has length
  :math:`\delta_i = \Delta t w_i / n_c` where :math:`w_i` are constants such
  that :math:`\sum_i w_i = 1`. See the table of Suzuki-Yoshida weights above
  for specific values. The number of substeps and the number of Suzuki-Yoshida
  steps are set using the `chain_steps` and `sy_steps` arguments.

  Consequently, the Nose-Hoover chains are described by three functions: an
  `init_fn` that initializes the state of the chain, a `half_step_fn` that
  updates the chain for one half-step, and an `update_chain_mass_fn` that
  updates the masses of the chain to enforce the correct period of oscillation.

  Note that a system can have many Nose-Hoover chains coupled to it to produce,
  for example, a temperature gradient. We also note that the NPT ensemble
  naturally features two chains: one that couples to the thermal degrees of
  freedom and one that couples to the barostat.

  Attributes:
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number :math:`n_c` of outer substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either `1`, `3`, `5`, or `7`.
    tau: A floating point timescale over which temperature equilibration occurs.
      Measured in units of `dt`. The performance of the Nose-Hoover chain
      thermostat can be quite sensitive to this choice.
  Returns:
    A triple of functions that initialize the chain, do a half step of
    simulation, and update the chain masses respectively.
  """
  if isinstance(dt, Quantity) and isinstance(tau, Quantity):
    dt = dt.to_decimal(u.fsecond)
    tau = tau.to_decimal(u.fsecond)
    return_quantity = True
  elif isinstance(dt, Quantity) or isinstance(tau, Quantity):
    raise ValueError('dt and tau must both be Quantities or neither of them.')
  else:
    return_quantity = False

  def init_fn(degrees_of_freedom, KE, kT):
    if isinstance(KE, Quantity) and isinstance(kT, Quantity):
      KE = KE.to_decimal(u.eV)
      kT = kT.to_decimal(u.fsecond)
    elif isinstance(KE, Quantity) or isinstance(kT, Quantity):
      raise ValueError('KE and kT must both be Quantities or neither of them.')
    xi = jnp.zeros(chain_length, KE.dtype)
    p_xi = jnp.zeros(chain_length, KE.dtype)

    Q = kT * tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
    Q = Q.at[0].multiply(degrees_of_freedom)
    if return_quantity:
      return NoseHooverChain(
        position=xi * u.angstrom,
        momentum=p_xi * u.IMF * u.angstrom,
        mass=Q * u.atomic_mass,
        tau=tau * u.fsecond,
        kinetic_energy=KE * u.eV,
        degrees_of_freedom=degrees_of_freedom
      )
    else:
      return NoseHooverChain(
        position=xi,
        momentum=p_xi,
        mass=Q,
        tau=tau,
        kinetic_energy=KE,
        degrees_of_freedom=degrees_of_freedom
      )

  def substep_fn(delta, P, state, kT):
    """Apply a single update to the chain parameters and rescales velocity."""
    xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)
    if isinstance(xi, Quantity):
      xi_unit = xi.unit
      xi = xi.to_decimal(u.angstrom)
    if isinstance(p_xi, Quantity):
      p_xi_unit = p_xi.unit
      p_xi = p_xi.to_decimal(u.IMF * u.angstrom)
    if isinstance(Q, Quantity):
      Q_unit = Q.unit
      Q = Q.to_decimal(u.atomic_mass)
    if isinstance(_tau, Quantity):
      _tau_unit = _tau.unit
      _tau = _tau.to_decimal(u.fsecond)
    if isinstance(KE, Quantity):
      KE_unit = KE.unit
      KE = KE.to_decimal(u.eV)
    if isinstance(kT, Quantity):
      kT_unit = kT.unit
      kT = kT.to_decimal(u.fsecond)
    if isinstance(P, Quantity):
      P_unit = P.unit
      P = P.to_decimal(u.eV)

    delta_2 = delta   / f32(2.0)
    delta_4 = delta_2 / f32(2.0)
    delta_8 = delta_4 / f32(2.0)

    M = chain_length - 1

    G = (p_xi[M - 1] ** f32(2) / Q[M - 1] - kT)
    p_xi = p_xi.at[M].add(delta_4 * G)

    def backward_loop_fn(p_xi_new, m):
      G = p_xi[m - 1] ** 2 / Q[m - 1] - kT
      scale = jnp.exp(-delta_8 * p_xi_new / Q[m + 1])
      p_xi_new = scale * (scale * p_xi[m] + delta_4 * G)
      return p_xi_new, p_xi_new
    idx = jnp.arange(M - 1, 0, -1)
    _, p_xi_update = lax.scan(backward_loop_fn, p_xi[M], idx, unroll=2)
    p_xi = p_xi.at[idx].set(p_xi_update)

    G = f32(2.0) * KE - DOF * kT
    scale = jnp.exp(-delta_8 * p_xi[1] / Q[1])
    p_xi = p_xi.at[0].set(scale * (scale * p_xi[0] + delta_4 * G))

    scale = jnp.exp(-delta_2 * p_xi[0] / Q[0])
    KE = KE * scale ** f32(2)
    P = tree_map(lambda p: p * scale, P)

    xi = xi + delta_2 * p_xi / Q

    G = f32(2) * KE - DOF * kT
    def forward_loop_fn(G, m):
      scale = jnp.exp(-delta_8 * p_xi[m + 1] / Q[m + 1])
      p_xi_update = scale * (scale * p_xi[m] + delta_4 * G)
      G = p_xi_update ** 2 / Q[m] - kT
      return G, p_xi_update
    idx = jnp.arange(M)
    G, p_xi_update = lax.scan(forward_loop_fn, G, idx, unroll=2)
    p_xi = p_xi.at[idx].set(p_xi_update)
    p_xi = p_xi.at[M].add(delta_4 * G)

    if return_quantity:
      return P * u.IMF * u.angstrom, NoseHooverChain(xi * u.angstrom, p_xi * u.IMF * u.angstrom, Q * u.atomic_mass, _tau * u.fsecond, KE * u.eV, DOF), kT * u.fsecond
    else:
      return P, NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF), kT

  def half_step_chain_fn(P, state, kT):
    if chain_steps == 1 and sy_steps == 1:
      P, state, _ = substep_fn(dt, P, state, kT)
      return P, state

    delta = dt / chain_steps
    ws = jnp.array(SUZUKI_YOSHIDA_WEIGHTS[sy_steps])
    def body_fn(cs, i):
      d = f32(delta.to_decimal(u.fsecond) if isinstance(delta, Quantity) else delta * ws[i % sy_steps])
      return (substep_fn(d, *cs), 0) if return_quantity else (substep_fn(d, *cs), 0)
    P, state, _ = lax.scan(body_fn,
                           (P, state, kT),
                           jnp.arange(chain_steps * sy_steps))[0]
    return P, state

  def update_chain_mass_fn(state, kT):
    xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)
    if isinstance(xi, Quantity):
      xi_unit = xi.unit
      xi = xi.to_decimal(u.angstrom)
    if isinstance(p_xi, Quantity):
      p_xi_unit = p_xi.unit
      p_xi = p_xi.to_decimal(u.IMF * u.angstrom)
    if isinstance(Q, Quantity):
      Q_unit = Q.unit
      Q = Q.to_decimal(u.atomic_mass)
    if isinstance(_tau, Quantity):
      _tau_unit = _tau.unit
      _tau = _tau.to_decimal(u.fsecond)
    if isinstance(KE, Quantity):
      KE_unit = KE.unit
      KE = KE.to_decimal(u.eV)
    if isinstance(kT, Quantity):
      kT_unit = kT.unit
      kT = kT.to_decimal(u.fsecond)

    Q = kT * _tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
    Q = Q.at[0].multiply(DOF)

    if return_quantity:
      return NoseHooverChain(xi * u.angstrom, p_xi * u.IMF * u.angstrom, Q * u.atomic_mass, _tau * u.fsecond, KE * u.eV, DOF)
    else:
      return NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF)

  return NoseHooverChainFns(init_fn, half_step_chain_fn, update_chain_mass_fn)


def default_nhc_kwargs(tau: float, overrides: Dict) -> Dict:
  default_kwargs = {
      'chain_length': 3,
      'chain_steps': 2,
      'sy_steps': 3,
      'tau': tau
  }

  if overrides is None:
    return default_kwargs

  return {
      key: overrides.get(key, default_kwargs[key])
      for key in default_kwargs
  }


@dataclasses.dataclass
class NVTNoseHooverState:
  """State information for an NVT system with a Nose-Hoover chain thermostat.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on the particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape `[n]`.
    chain: The variables describing the Nose-Hoover chain.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array
  chain: NoseHooverChain

  @property
  def velocity(self):
    return self.momentum / self.mass


def nvt_nose_hoover(energy_or_force_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    kT: float,
                    chain_length: int=5,
                    chain_steps: int=2,
                    sy_steps: int=3,
                    tau: Optional[float]=None,
                    **sim_kwargs) -> Simulator:
  """Simulation in the NVT ensemble using a Nose Hoover Chain thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. We use a
  Nose Hoover Chain (NHC) thermostat described in [#martyna92]_ [#martyna98]_
  [#tuckerman]_. We follow the direct translation method outlined in
  Tuckerman et al. [#tuckerman]_ and the interested reader might want to look
  at that paper as a reference.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number, :math:`n_c`, of outer
      substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either `1`, `3`, `5`, or `7`.
    tau: A floating point timescale over which temperature equilibration
      occurs. Measured in units of `dt`. The performance of the Nose-Hoover
      chain thermostat can be quite sensitive to this choice.
  Returns:
    See above.

  .. rubric:: References
  .. [#martyna92] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
    "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
    The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
  .. [#martyna98] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
    "Explicit reversible integrators for extended systems dynamics."
    Molecular Physics 87. (1998) 1117-1157.
  .. [#tuckerman] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
    Andrea L. Jochim, and Glenn J. Martyna.
    "A Liouville-operator derived measure-preserving integrator for molecular
    dynamics simulations in the isothermal-isobaric ensemble."
    Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  return_quantity = False
  if isinstance(dt, Quantity):
    dt = f32(dt.to_decimal(u.fsecond)) * u.fsecond
    return_quantity = True
    if tau is None:
      tau = dt * 100
  else:
    dt = f32(dt)
    dt_2 = f32(dt / 2)
    if tau is None:
      tau = dt * 100
    tau = f32(tau)

  thermostat = nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']
    if isinstance(R, Quantity):
        mass = mass * u.atomic_mass

    dof = quantity.count_dof(R)

    state = NVTNoseHooverState(R, None, force_fn(R, **kwargs) * u.IMF if return_quantity else force_fn(R, **kwargs), mass, None)
    state = canonicalize_mass(state)
    state = initialize_momenta(state, key, _kT)
    KE = kinetic_energy(state)
    return state.set(chain=thermostat.initialize(dof, KE, _kT))

  @jit
  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    chain = state.chain

    chain = thermostat.update_mass(chain, _kT)

    p, chain = thermostat.half_step(state.momentum, chain, _kT)
    state = state.set(momentum=p)

    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)

    chain = chain.set(kinetic_energy=kinetic_energy(state))

    p, chain = thermostat.half_step(state.momentum, chain, _kT)
    state = state.set(momentum=p, chain=chain)

    return state
  return init_fn, apply_fn


def nvt_nose_hoover_invariant(energy_fn: Callable[..., Array],
                              state: NVTNoseHooverState,
                              kT: float,
                              **kwargs) -> float:
  """The conserved quantity for the NVT ensemble with a Nose-Hoover thermostat.

  This function is normally used for debugging the Nose-Hoover thermostat.

  Arguments:
    energy_fn: The energy function of the Nose-Hoover system.
    state: The current state of the system.
    kT: The current goal temperature of the system.

  Returns:
    The Hamiltonian of the extended NVT dynamics.
  """
  PE = energy_fn(state.position, **kwargs)
  KE = kinetic_energy(state)
  return_quantity = False
  if isinstance(kT, Quantity):
    kT = kT.to_decimal(u.fsecond)
    return_quantity = True

  DOF = quantity.count_dof(state.position)
  E = u.get_mantissa(PE) + u.get_mantissa(KE)

  c = state.chain

  E += u.get_mantissa(c.momentum[0]) ** 2 / (2 * u.get_mantissa(c.mass[0])) + DOF * kT * u.get_mantissa(c.position[0])
  for r, p, m in zip(u.get_mantissa(c.position[1:]), u.get_mantissa(c.momentum[1:]), u.get_mantissa(c.mass[1:])):
    E += p ** 2 / (2 * m) + kT * r
  return E * u.eV if return_quantity else E


@dataclasses.dataclass
class NPTNoseHooverState:
  """State information for an NPT system with Nose-Hoover chain thermostats.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    momentum: The velocity of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on the particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape `[n]`.
    reference_box: A box used to measure relative changes to the simulation
      environment.
    box_position: A positional degree of freedom used to describe the current
      box. box_position is parameterized as `box_position = (1/d)log(V/V_0)`
      where `V` is the current volume, `V_0` is the reference volume, and `d`
      is the spatial dimension.
    box_velocity: A velocity degree of freedom for the box.
    box_mass: The mass assigned to the box.
    barostat: The variables describing the Nose-Hoover chain coupled to the
      barostat.
    thermostsat: The variables describing the Nose-Hoover chain coupled to the
      thermostat.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array

  reference_box: Box

  box_position: Array
  box_momentum: Array
  box_mass: Array

  barostat: NoseHooverChain
  thermostat: NoseHooverChain

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass

  @property
  def box(self) -> Array:
    """Get the current box from an NPT simulation."""
    dim = self.position.shape[1]
    ref = self.reference_box
    V_0 = quantity.volume(dim, ref)
    V = V_0 * jnp.exp(dim * self.box_position)
    return (V / V_0) ** (1 / dim) * ref


def _npt_box_info(state: NPTNoseHooverState
                  ) -> Tuple[float, Callable[[float], float]]:
  """Gets the current volume and a function to compute the box from volume."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * jnp.exp(dim * state.box_position.to_decimal(u.angstrom) if isinstance(state.box_position, Quantity) else state.box_position)
  return V, lambda V: (V / V_0) ** (1 / dim) * ref


def npt_box(state: NPTNoseHooverState) -> Box:
  """Get the current box from an NPT simulation."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * jnp.exp(dim * state.box_position.to_decimal(u.angstrom) if isinstance(state.box_position, Quantity) else state.box_position)
  return (V / V_0) ** (1 / dim) * ref


def npt_nose_hoover(energy_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    pressure: float,
                    kT: float,
                    barostat_kwargs: Optional[Dict]=None,
                    thermostat_kwargs: Optional[Dict]=None) -> Simulator:
  """Simulation in the NPT ensemble using a pair of Nose Hoover Chains.

  Samples from the canonical ensemble in which the number of particles (N),
  the system pressure (P), and the temperature (T) are held constant.
  We use a pair of Nose Hoover Chains (NHC) described in
  [#martyna92]_ [#martyna98]_ [#tuckerman]_ coupled to the
  barostat and the thermostat respectively. We follow the direct translation
  method outlined in Tuckerman et al. [#tuckerman]_ and the interested reader
  might want to look at that paper as a reference.

  Args:
    energy_fn: A function that produces either an energy from a set of particle
      positions specified as an ndarray of shape `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    pressure: Floating point number specifying the target pressure. To update
      the pressure dynamically during a simulation one should pass `pressure`
      as a keyword argument to the step function.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    barostat_kwargs: A dictionary of keyword arguments passed to the barostat
      NHC. Any parameters not set are drawn from a relatively robust default
      set.
    thermostat_kwargs: A dictionary of keyword arguments passed to the
      thermostat NHC. Any parameters not set are drawn from a relatively robust
      default set.

  Returns:
    See above.

  """

  t = f32(dt)
  dt_2 = f32(dt / 2)

  force_fn = quantity.force(energy_fn)

  barostat_kwargs = default_nhc_kwargs(1000 * dt, barostat_kwargs)
  barostat = nose_hoover_chain(dt, **barostat_kwargs)

  thermostat_kwargs = default_nhc_kwargs(100 * dt, thermostat_kwargs)
  thermostat = nose_hoover_chain(dt, **thermostat_kwargs)

  def init_fn(key, R, box, mass=f32(1.0), **kwargs):
    N, dim = R.shape

    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    # The box position is defined via pos = (1 / d) log V / V_0.
    zero = jnp.zeros((), dtype=R.dtype)
    one = jnp.ones((), dtype=R.dtype)
    box_position = zero
    box_momentum = zero
    box_mass = dim * (N + 1) * kT * barostat_kwargs['tau'] ** 2 * one
    KE_box = quantity.kinetic_energy(momentum=box_momentum, mass=box_mass)

    if jnp.isscalar(box) or box.ndim == 0:
      # TODO(schsam): This is necessary because of JAX issue #5849.
      box = jnp.eye(R.shape[-1]) * box

    state = NPTNoseHooverState(
      R, None, force_fn(R, box=box, **kwargs),
      mass, box, box_position, box_momentum, box_mass,
      barostat.initialize(1, KE_box, _kT),
      None)  # pytype: disable=wrong-arg-count
    state = canonicalize_mass(state)
    state = initialize_momenta(state, key, _kT)
    KE = kinetic_energy(state)
    return state.set(
      thermostat=thermostat.initialize(quantity.count_dof(R), KE, _kT))

  def update_box_mass(state, kT):
    N, dim = state.position.shape
    dtype = state.position.dtype
    box_mass = jnp.array(dim * (N + 1) * kT * state.barostat.tau ** 2, dtype)
    return state.set(box_mass=box_mass)

  def box_force(alpha, vol, box_fn, position, momentum, mass, force, pressure,
                **kwargs):
    N, dim = position.shape

    def U(eps):
      return energy_fn(position, box=box_fn(vol), perturbation=(1 + eps),
                       **kwargs)

    dUdV = grad(U)
    KE2 = util.high_precision_sum(momentum ** 2 / mass)

    return alpha * KE2 - dUdV(0.0) - pressure * vol * dim

  def sinhx_x(x):
    """Taylor series for sinh(x) / x as x -> 0."""
    return (1 + x ** 2 / 6 + x ** 4 / 120 + x ** 6 / 5040 +
            x ** 8 / 362_880 + x ** 10 / 39_916_800)

  def exp_iL1(box, R, V, V_b, **kwargs):
    x = V_b * dt
    x_2 = x / 2
    sinhV = sinhx_x(x_2)  # jnp.sinh(x_2) / x_2
    return shift_fn(R, R * (jnp.exp(x) - 1) + dt * V * jnp.exp(x_2) * sinhV,
                    box=box, **kwargs)  # pytype: disable=wrong-keyword-args

  def exp_iL2(alpha, P, F, V_b):
    x = alpha * V_b * dt_2
    x_2 = x / 2
    sinhP = sinhx_x(x_2)  # jnp.sinh(x_2) / x_2
    return P * jnp.exp(-x) + dt_2 * F * sinhP * jnp.exp(-x_2)

  def inner_step(state, **kwargs):
    _pressure = kwargs.pop('pressure', pressure)

    R, P, M, F = state.position, state.momentum, state.mass, state.force
    R_b, P_b, M_b = state.box_position, state.box_momentum, state.box_mass

    N, dim = R.shape

    vol, box_fn = _npt_box_info(state)

    alpha = 1 + 1 / N
    G_e = box_force(alpha, vol, box_fn, R, P, M, F, _pressure, **kwargs)
    P_b = P_b + dt_2 * G_e
    P = exp_iL2(alpha, P, F, P_b / M_b)

    R_b = R_b + P_b / M_b * dt
    state = state.set( box_position=R_b)

    vol, box_fn = _npt_box_info(state)

    box = box_fn(vol)
    R = exp_iL1(box, R, P / M, P_b / M_b)
    F = force_fn(R, box=box, **kwargs)

    P = exp_iL2(alpha, P, F, P_b / M_b)
    G_e = box_force(alpha, vol, box_fn, R, P, M, F, _pressure, **kwargs)
    P_b = P_b + dt_2 * G_e

    return state.set(position=R, momentum=P, mass=M, force=F,
                     box_position=R_b, box_momentum=P_b, box_mass=M_b)

  def apply_fn(state, **kwargs):
    S = state
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    bc = barostat.update_mass(S.barostat, _kT)
    tc = thermostat.update_mass(S.thermostat, _kT)
    S = update_box_mass(S, _kT)

    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)
    P, tc = thermostat.half_step(S.momentum, tc, _kT)

    S = S.set(momentum=P, box_momentum=P_b)
    S = inner_step(S, **kwargs)

    KE = quantity.kinetic_energy(momentum=S.momentum, mass=S.mass)
    tc = tc.set(kinetic_energy=KE)

    KE_box = quantity.kinetic_energy(momentum=S.box_momentum, mass=S.box_mass)
    bc = bc.set(kinetic_energy=KE_box)

    P, tc = thermostat.half_step(S.momentum, tc, _kT)
    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)

    S = S.set(thermostat=tc, barostat=bc, momentum=P, box_momentum=P_b)

    return S
  return init_fn, apply_fn


def npt_nose_hoover_invariant(energy_fn: Callable[..., Array],
                              state: NPTNoseHooverState,
                              pressure: float,
                              kT: float,
                              **kwargs) -> float:
  """The conserved quantity for the NPT ensemble with a Nose-Hoover thermostat.

  This function is normally used for debugging the NPT simulation.

  Arguments:
    energy_fn: The energy function of the system.
    state: The current state of the system.
    pressure: The current goal pressure of the system.
    kT: The current goal temperature of the system.

  Returns:
    The Hamiltonian of the extended NPT dynamics.
  """
  volume, box_fn = _npt_box_info(state)
  PE = energy_fn(state.position, box=box_fn(volume), **kwargs)
  KE = kinetic_energy(state)

  DOF = state.position.size
  E = PE + KE

  c = state.thermostat
  E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
  for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
    E += p ** 2 / (2 * m) + kT * r

  c = state.barostat
  for r, p, m in zip(c.position, c.momentum, c.mass):
    E += p ** 2 / (2 * m) + kT * r

  E += pressure * volume
  E += state.box_momentum ** 2 / (2 * state.box_mass)

  return E


"""Stochastic Simulations

JAX MD includes integrators for stochastic simulations of Langevin dynamics and
Brownian motion for systems in the NVT ensemble with a solvent.
"""


@dataclasses.dataclass
class Normal:
  """A simple normal distribution."""
  mean: jnp.ndarray
  var: jnp.ndarray

  def sample(self, key):
    mu, sigma = self.mean, jnp.sqrt(self.var)
    return mu + sigma * random.normal(key, mu.shape ,dtype=mu.dtype)

  def log_prob(self, x):
    return (-0.5 * jnp.log(2 * jnp.pi * self.var) -
            1 / (2 * self.var) * (x - self.mean)**2)


@dataclasses.dataclass
class NVTLangevinState:
  """A struct containing state information for the Langevin thermostat.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    force: The (non-stochastic) force on particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape `[n]`.
    rng: The current state of the random number generator.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array
  rng: Array

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass


@dispatch_by_state
def stochastic_step(state: NVTLangevinState, dt:float, kT: float, gamma: float):
  """A single stochastic step (the `O` step)."""
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(kT * (1 - c1**2))
  momentum_dist = Normal(c1 * state.momentum, c2**2 * state.mass)
  key, split = random.split(state.rng)
  return state.set(momentum=momentum_dist.sample(split), rng=key)


def nvt_langevin(energy_or_force_fn: Callable[..., Array],
                 shift_fn: ShiftFn,
                 dt: float,
                 kT: float,
                 gamma: float=0.1,
                 center_velocity: bool=True,
                 **sim_kwargs) -> Simulator:
  """Simulation in the NVT ensemble using the BAOAB Langevin thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. Langevin
  dynamics are stochastic and it is supposed that the system is interacting
  with fictitious microscopic degrees of freedom. An example of this would be
  large particles in a solvent such as water. Thus, Langevin dynamics are a
  stochastic ODE described by a friction coefficient and noise of a given
  covariance.

  Our implementation follows the paper [#davidcheck] by Davidchack, Ouldridge,
  and Tretyakov.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
    center_velocity: A boolean specifying whether or not the center of mass
      position should be subtracted.
  Returns:
    See above.

  .. rubric:: References
  .. [#carlon] R. L. Davidchack, T. E. Ouldridge, and M. V. Tretyakov.
    "New Langevin and gradient thermostats for rigid body dynamics."
    The Journal of Chemical Physics 142, 144114 (2015)
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    force = force_fn(R, **kwargs)
    state = NVTLangevinState(R, None, force, mass, key)
    state = canonicalize_mass(state)
    return initialize_momenta(state, split, _kT)

  @jit
  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    _kT = kwargs.pop('kT', kT)
    dt_2 = _dt / 2

    state = momentum_step(state, dt_2)
    state = position_step(state, shift_fn, dt_2, **kwargs)
    state = stochastic_step(state, _dt, _kT, gamma)
    state = position_step(state, shift_fn, dt_2, **kwargs)
    state = state.set(force=force_fn(state.position, **kwargs))
    state = momentum_step(state, dt_2)

    return state

  return init_fn, step_fn


@dataclasses.dataclass
class BrownianState:
  """A tuple containing state information for Brownian dynamics.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape `[n]`.
    rng: The current state of the random number generator.
  """
  position: Array
  mass: Array
  rng: Array


def brownian(energy_or_force: Callable[..., Array],
             shift: ShiftFn,
             dt: float,
             kT: float,
             gamma: float=0.1) -> Simulator:
  """Simulation of Brownian dynamics.

  Simulates Brownian dynamics which are synonymous with the overdamped
  regime of Langevin dynamics. However, in this case we don't need to take into
  account velocity information and the dynamics simplify. Consequently, when
  Brownian dynamics can be used they will be faster than Langevin. As in the
  case of Langevin dynamics our implementation follows Carlon et al. [#carlon]_

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.

  Returns:
    See above.
  """

  force_fn = quantity.canonicalize_force(energy_or_force)

  dt, gamma = static_cast(dt, gamma)

  def init_fn(key, R, mass=f32(1)):
    state = BrownianState(R, mass, key)
    return canonicalize_mass(state)

  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    R, mass, key = dataclasses.astuple(state)

    key, split = random.split(key)

    F = force_fn(R, **kwargs)
    xi = random.normal(split, R.shape, R.dtype)

    nu = f32(1) / (mass * gamma)

    dR = F * dt * nu + jnp.sqrt(f32(2) * _kT * dt * nu) * xi
    R = shift(R, dR, **kwargs)

    return BrownianState(R, mass, key)  # pytype: disable=wrong-arg-count

  return init_fn, apply_fn


"""Experimental Simulations.


Below are simulation environments whose implementation is somewhat
experimental / preliminary. These environments might not be as ergonomic
as the more polished environments above.
"""


@dataclasses.dataclass
class SwapMCState:
  """A struct containing state information about a Hybrid Swap MC simulation.

  Attributes:
    md: A NVTNoseHooverState containing continuous molecular dynamics data.
    sigma: An `[n,]` array of particle radii.
    key: A JAX PRGNKey used for random number generation.
    neighbor: A NeighborList for the system.
  """
  md: NVTNoseHooverState
  sigma: Array
  key: Array
  neighbor: partition.NeighborList


# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
def hybrid_swap_mc(space_fns: space.Space,
                   energy_fn: Callable[[Array, Array], Array],
                   neighbor_fn: partition.NeighborFn,
                   dt: float,
                   kT: float,
                   t_md: float,
                   N_swap: int,
                   sigma_fn: Optional[Callable[[Array], Array]]=None
                   ) -> Simulator:
  """Simulation of Hybrid Swap Monte-Carlo.

  This code simulates the hybrid Swap Monte Carlo algorithm introduced in
  Berthier et al. [#berthier]_
  Here an NVT simulation is performed for `t_md` time and then `N_swap` MC
  moves are performed that swap the radii of randomly chosen particles. The
  random swaps are accepted with Metropolis-Hastings step. Each call to the
  step function runs molecular dynamics for `t_md` and then performs the swaps.

  Note that this code doesn't feature some of the convenience functions in the
  other simulations. In particular, there is no support for dynamics keyword
  arguments and the energy function must be a simple callable of two variables:
  the distance between adjacent particles and the diameter of the particles.
  If you want support for a better notion of potential or dynamic keyword
  arguments, please file an issue!

  Args:
    space_fns: A tuple of a displacement function and a shift function defined
      in `space.py`.
    energy_fn: A function that computes the energy between one pair of
      particles as a function of the distance between the particles and the
      diameter. This function should not have been passed to `smap.xxx`.
    neighbor_fn: A function to construct neighbor lists outlined in
      `partition.py`.
    dt: The timestep used for the continuous time MD portion of the simulation.
    kT: The temperature of heat bath that the system is coupled to during MD.
    t_md: The time of each MD block.
    N_swap: The number of swapping moves between MD blocks.
    sigma_fn: An optional function for combining radii if they are to be
      non-additive.

  Returns:
    See above.

  .. rubric:: References
  .. [#berthier] L. Berthier, E. Flenner, C. J. Fullerton, C. Scalliet, and M. Singh.
    "Efficient swap algorithms for molecular dynamics simulations of
    equilibrium supercooled liquids", J. Stat. Mech. (2019) 064004
  """
  displacement_fn, shift_fn = space_fns
  metric_fn = space.metric(displacement_fn)
  nbr_metric_fn = space.map_neighbor(metric_fn)

  md_steps = int(t_md // dt)

  # Canonicalize the argument names to be dr and sigma.
  wrapped_energy_fn = lambda dr, sigma: energy_fn(dr, sigma)
  if sigma_fn is None:
    sigma_fn = lambda si, sj: 0.5 * (si + sj)
  nbr_energy_fn = smap.pair_neighbor_list(wrapped_energy_fn,
                                          metric_fn,
                                          sigma=sigma_fn)

  nvt_init_fn, nvt_step_fn = nvt_nose_hoover(nbr_energy_fn,
                                             shift_fn,
                                             dt,
                                             kT=kT,
                                             chain_length=3)
  def init_fn(key, position, sigma, nbrs=None):
    key, sim_key = random.split(key)
    nbrs = neighbor_fn(position, nbrs)  # pytype: disable=wrong-arg-count
    md_state = nvt_init_fn(sim_key, position, neighbor=nbrs, sigma=sigma)
    return SwapMCState(md_state, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def md_step_fn(i, state):
    md, sigma, key, nbrs = dataclasses.unpack(state)
    md = nvt_step_fn(md, neighbor=nbrs, sigma=sigma)  # pytype: disable=wrong-keyword-args
    nbrs = neighbor_fn(md.position, nbrs)
    return SwapMCState(md, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def swap_step_fn(i, state):
    md, sigma, key, nbrs = dataclasses.unpack(state)

    N = md.position.shape[0]

    # Swap a random pair of particle radii.
    key, particle_key, accept_key = random.split(key, 3)
    ij = random.randint(particle_key, (2,), jnp.array(0), jnp.array(N))
    new_sigma = sigma.at[ij].set([sigma[ij[1]], sigma[ij[0]]])

    # Collect neighborhoods around the two swapped particles.
    nbrs_ij = nbrs.idx[ij]
    R_ij = md.position[ij]
    R_neigh = md.position[nbrs_ij]

    sigma_ij = sigma[ij][:, None]
    sigma_neigh = sigma[nbrs_ij]

    new_sigma_ij = new_sigma[ij][:, None]
    new_sigma_neigh = new_sigma[nbrs_ij]

    dR = nbr_metric_fn(R_ij, R_neigh)

    # Compute the energy before the swap.
    energy = energy_fn(dR, sigma_fn(sigma_ij, sigma_neigh))
    energy = jnp.sum(energy * (nbrs_ij < N))

    # Compute the energy after the swap.
    new_energy = energy_fn(dR, sigma_fn(new_sigma_ij, new_sigma_neigh))
    new_energy = jnp.sum(new_energy * (nbrs_ij < N))

    # Accept or reject with a metropolis probability.
    p = random.uniform(accept_key, ())
    accept_prob = jnp.minimum(1, jnp.exp(-(new_energy - energy) / kT))
    sigma = jnp.where(p < accept_prob, new_sigma, sigma)

    return SwapMCState(md, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def block_fn(state):
    state = lax.fori_loop(0, md_steps, md_step_fn, state)
    state = lax.fori_loop(0, N_swap, swap_step_fn, state)
    return state

  return init_fn, block_fn
# pytype: enable=wrong-arg-count
# pytype: enable=wrong-keyword-args


def temp_rescale(energy_or_force_fn: Callable[..., Array],
                 shift_fn: ShiftFn,
                 dt: float,
                 kT: float,
                 window: float,
                 fraction: float,
                 **sim_kwargs) -> Simulator:
  """Simulation using explicit velocity rescaling.

  Rescale the velocities of atoms explicitly so that the desired temperature is
  reached.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    window: Floating point number specifying the temperature window outside which 
      rescaling is performed. Measured in units of `kT`.
    fraction: Floating point number which determines the amount of rescaling 
      applied to the velocities. Takes values from 0.0-1.0.
  Returns:
    See above.

  .. rubric:: References
  .. [#berendsen84] Woodcock, L. V.
    "ISOTHERMAL MOLECULAR DYNAMICS CALCULATIONS FOR LIQUID SALTS."
    Chem. Phys. Lett. 1971, 10, 257–261.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)

  def velocity_rescale(state, window, fraction, kT):
    """Rescale the momentum if the the difference between current and target
    temperature is more than the window"""
    kT_current = temperature(state)
    cond = jnp.abs(kT_current - kT) > window
    kT_target = kT_current - fraction*(kT_current - kT)
    lam = jnp.where(cond, jnp.sqrt(kT_target / kT_current), 1)
    new_momentum = tree_map(lambda p: p * lam, state.momentum)
    return state.set(momentum = new_momentum)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    # Reuse the NVEState dataclass
    state = NVEState(R, None, force_fn(R, **kwargs), mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, key, kT)

  def apply_fn(state, **kwargs):
    state = velocity_rescale(state, window, fraction, kT)
    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)
    return state
  return init_fn, apply_fn


def temp_berendsen(energy_or_force_fn: Callable[..., Array],
                   shift_fn: ShiftFn,
                   dt: float,
                   kT: float,
                   tau: float,
                   **sim_kwargs) -> Simulator:
  """Simulation using the Berendsen thermostat.

  Berendsen (weak coupling) thermostat rescales the velocities of atoms such
  that the desired temperature is reached. This rescaling is performed at each
  timestep (dt) and the rescaling factor is calculated using
  Eq.10 Berendsen et al. [#berendsen84]_.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    tau: A floating point number determining how fast the temperature
      is relaxed during the simulation. Measured in units of `dt`.
  Returns:
    See above.

  .. rubric:: References
  .. [#berendsen84] H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola, J. R. Haak.
    "Molecular dynamics with coupling to an external bath."
    J. Chem. Phys. 15 October 1984; 81 (8): 3684-3690.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)

  def berendsen_update(state, tau, kT, dt):
    """Rescaling the momentum of the particle by the factor lam."""
    _kT = temperature(state)
    lam = jnp.sqrt(1 + ((dt/tau) * ((kT/_kT) - 1)))
    new_momentum = tree_map(lambda p: p * lam, state.momentum)
    return state.set(momentum=new_momentum)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    # Reuse the NVEState dataclass
    state = NVEState(R, None, force_fn(R, **kwargs), mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, key, kT)

  def apply_fn(state, **kwargs):
    state = berendsen_update(state, tau, kT, dt)
    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)
    return state
  return init_fn, apply_fn


def nvk(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float,
        kT: float,
        **sim_kwargs) -> Simulator:
  """Simulation in the NVK (isokinetic) ensemble using the Gaussian thermostat.

  Samples from the isokinetic ensemble in which the number of particles (N),
  the system volume (V), and the kinetic energy (K) are held constant. A 
  Gaussian thermostat is used for the integration and the kinetic energy is 
  held constant during the simulation. The implementation follows the steps 
  described in [#minary2003]_ and [#zhang97]_. See section 4(B) equation 
  4.12-4.17 in [#minary2003]_ for detailed description.     

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
  Returns:
    See above.

  .. rubric:: References
  .. [#minary2003] Minary, Peter and Martyna, Glenn J. and Tuckerman, Mark E.
    "Algorithms and novel applications based on the isokinetic ensemble. I. 
    Biophysical and path integral molecular dynamics"
    J. Chem. Phys., Vol. 118, No. 6, 8 February 2003.
  .. [#zhang97] Zhang, Fei.
    "Operator-splitting integrators for constant-temperature molecular dynamics"
    J. Chem. Phys. 106, 6102–6106 (1997).
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)
  dt_2 = f32(dt / 2)

  def momentum_update(state, KE):
    # eps to avoid edge cases when forces are zero
    eps = 1e-16

    # Equation 4.13 to compute a and b
    update_fn = (lambda f, p, m: f * p / m)
    a = util.high_precision_sum(update_fn(state.force, state.momentum, state.mass)) + eps
    b = util.high_precision_sum(update_fn(state.force, state.force, state.mass)) + eps
    a /= (2.0 * KE)
    b /= (2.0 * KE)

    # Equation 4.12 to compute s(t) and s_dot(t)
    b_sqrt = jnp.sqrt(b)
    s_t = ((a / b) * (jnp.cosh(dt_2 * b_sqrt) - 1.0)) + jnp.sinh(dt_2 * b_sqrt) / b_sqrt
    s_dot_t = (b_sqrt * (a / b) * jnp.sinh(dt_2 * b_sqrt)) + jnp.cosh(dt_2 * b_sqrt)

    # Get the new momentum using Equation 4.15  
    new_momentum = tree_map(lambda p, f, s, sdot: (p + f * s) / sdot,
                            state.momentum,
                            state.force,
                            s_t,
                            s_dot_t)
    return state.set(momentum=new_momentum)

  def position_update(state, shift_fn, **kwargs):
    if isinstance(shift_fn, Callable):
      shift_fn = tree_map(lambda r: shift_fn, state.position)
    # Get the new positions using Equation 4.16 (Should read r = r + dt * p / m)
    new_position = tree_map(lambda s_fn, r, v: s_fn(r, dt * v, **kwargs),
                            shift_fn,
                            state.position,
                            state.velocity)
    return state.set(position=new_position)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    # Reuse the NVEState dataclass
    state = NVEState(R, None, force_fn(R, **kwargs), mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, split, _kT)

  def apply_fn(state, **kwargs):
    _KE = kinetic_energy(state)
    state = momentum_update(state, _KE)
    state = position_update(state, shift_fn)
    state = state.set(force=force_fn(state.position, **kwargs))
    state = momentum_update(state, _KE)
    return state
  return init_fn, apply_fn


def temp_csvr(energy_or_force_fn: Callable[..., Array],
              shift_fn: ShiftFn,
              dt: float,
              kT: float,
              tau: float,
              **sim_kwargs) -> Simulator:
  """Simulation using the canonical sampling through velocity rescaling (CSVR) thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. CSVR
  algorithmn samples the canonical distribution by rescaling the velocities
  by a appropritely chosen random factor. At each timestep (dt) the rescaling
  takes place and the rescaling factor is calculated using
  A7 Bussi et al. [#bussi2007]_. CSVR updates to the velocity are stochastic in
  nature and unlike the Berendsen thermostat it samples the true canonical
  distribution [#Braun2018]_.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    tau: A floating point number determining how fast the temperature
      is relaxed during the simulation. Measured in units of `dt`.
  Returns:
    See above.

  .. rubric:: References
  .. [#bussi2007] Bussi G, Donadio D, Parrinello M.
    "Canonical sampling through velocity rescaling."
    The Journal of chemical physics, 126(1), 014101.
  .. [#Braun2018] Efrem Braun, Seyed Mohamad Moosavi, and Berend Smit.
    "Anomalous Effects of Velocity Rescaling Algorithms: The Flying Ice Cube Effect Revisited."
    Journal of Chemical Theory and Computation 2018 14 (10), 5262-5272.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)

  def sum_noises(state, key):
    """Sum of N independent gaussian noises squared.
    Adapted from https://github.com/GiovanniBussi/StochasticVelocityRescaling
    For more details see Eq.A7 Bussi et al. [#bussi2007]_"""
    dof = quantity.count_dof(state.position) - 1
    _dtype = state.position.dtype

    if dof == 0:
      """If there are no terms return zero."""
      return 0

    elif dof == 1:
      """For a single noise term, directly calculate the square of the Gaussian
      noise value."""
      rr = random.normal(key, dtype=_dtype)
      return rr * rr

    elif dof % 2 == 0:
      """For an even number of noise terms, use the gamma-distributed random
      number generator"""
      return 2.0 * random.gamma(key, dof // 2, dtype=_dtype)

    else:
      """For an odd number of noise terms, sum two terms: one from the
      gamma-distributed generator and another from the square of a
      Gaussian-distributed random number."""
      rr = random.normal(key, dtype=_dtype)
      return 2.0 * random.gamma(key, (dof - 1) // 2, dtype=_dtype) + (rr * rr)

  def csvr_update(state, tau, kT, dt):
    """Update the momentum by an scaling factor as described by
    Eq.A7 Bussi et al. [#bussi2007]_"""
    key, split = random.split(state.rng)
    dof = quantity.count_dof(state.position)

    _kT = temperature(state)

    KE_old = dof * _kT / 2
    KE_new = dof * kT / 2

    r1 = random.normal(key, dtype=state.position.dtype)
    r2 = sum_noises(state, key)

    c1 = jnp.exp(-dt / tau)
    c2 = (1 - c1) * KE_new / KE_old / dof

    scale = c1 + (c2*((r1 * r1) + r2)) + (2 * r1 * jnp.sqrt(c1 * c2))
    lam = jnp.sqrt(scale)

    new_momentum = tree_map(lambda p: p * lam, state.momentum)
    return state.set(momentum=new_momentum, rng=key)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    # Reuse the NVTLangevinState dataclass
    state = NVTLangevinState(R, None, force_fn(R, **kwargs), mass, key)
    state = canonicalize_mass(state)
    return initialize_momenta(state, split, _kT)

  def apply_fn(state, **kwargs):
    state = csvr_update(state, tau, kT, dt)
    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)
    return state
  return init_fn, apply_fn