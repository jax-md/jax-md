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
# TODO: Get a proper type for UpdateFn.
UpdateFn = Callable

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]


"""Single Dispatch Code.

JAX MD allows for simulations to be extensible using single
dispatch. For those familiar with C / C++, single dispatch is essentially
function overloading based on the type of the first argument. For those
unfamiliar, single dispatch chooses which version of a function to call by
looking up the type of the first argument. If you are interested in setting up
a simulation using a different type of system you can do so in a relatively
light weight manner by introducing a new type for storing the state that is
compatible with the JAX PyTree system (we usually choose a dataclass) and then
overriding the four functions below.

These extensions allow a range of simulations to be run by just changing the
type of the position argument. In general, the most complicated function will
be the `inner_update_fn` which should be induced by the Suzuki-Trotter
decomposition for a given simulation environment. At the moment this set of
extensible functions works with deterministic NVE and NVT simulation
environments but not NPT or stochatic ones (like Langevin or Brownian). If that
use-case is important to you, please raise an issue. It is likely there will be
an additional function lookup needed in those cases.
"""


@functools.singledispatch
def canonicalize_mass(mass: Union[Array, float]) -> Union[Array, float]:
  """Reshape mass vector for broadcasting with positions."""
  if isinstance(mass, float):
    return mass
  if mass.ndim == 2 and mass.shape[1] == 1:
    return mass
  elif mass.ndim == 1:
    return jnp.reshape(mass, (mass.shape[0], 1))
  elif mass.ndim == 0:
    return mass
  msg = (
      'Expected mass to be either a floating point number or a one-dimensional'
      'ndarray. Found {}.'.format(mass)
      )
  raise ValueError(msg)


@functools.singledispatch
def initialize_momenta(R: Array, mass: Array, key: Array, kT: float):
  """Initialize momenta with the Maxwell-Boltzmann distribution."""
  P = jnp.sqrt(mass * kT) * random.normal(key, R.shape, dtype=R.dtype)
  # If simulating more than one particle, center the momentum.
  if R.shape[0] > 1:
    P = P - jnp.mean(P, axis=0, keepdims=True)
  return P


@functools.singledispatch
def inner_update_fn(R: Array, shift_fn: Callable[..., Array],
                       **unused_kwargs
                       ) -> UpdateFn:
  """Perform the inner update in the Suzuki-Trotter decomposition.

  This function assumes that the simulation employs a factorization that looks
  like :math:`e^{iL_{other}dt}e^{iL_vdt/2}e^{iL_{inner}dt}e^{iL_vdt/2}e^{iL_{other}dt}`
  where it is assumed that :math:`e^{iL_{other}}` does not update the velocity
  or the position, :math:`e^{iL_vdt/2}` only updates the velocity, and
  :math:`e^{iL_{inner}dt}` is arbitrary. This setting covers the NVE and NVT
  ensemble, but not the NPT.

  Since the inner update function is arbitrary, it takes positions, momenta,
  forces, and masses as input and returns a new set of positions, momenta,
  forces, and masses. In general, this function should be chosen to implement
  a factorization of the total Liouville operator for a given simulation
  environment given the total form above.
  """
  def update_fn(R: Array, P: Array, F: Array, M: Array, dt: float, **kwargs
               ) -> Tuple[Array, ...]:
    return (shift_fn(R, dt * P / M, **kwargs), P, F, M)
  return update_fn


@functools.singledispatch
def kinetic_energy(R: Array, P: Array, M: Array):
  """Compute the kinetic energy, possibly as a function of position."""
  return quantity.kinetic_energy(momentum=P, mass=M)


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
                    update_fn: UpdateFn,
                    dt: float,
                    state: T,
                    **kwargs) -> T:
  """Apply a single step of velocity Verlet integration to a state."""
  dt = f32(dt)
  dt_2 = f32(dt / 2)
  dt2_2 = f32(dt ** 2 / 2)

  R, P, F, M = state.position, state.momentum, state.force, state.mass

  # First half step.
  P = tree_map(lambda p, f: p + dt_2 * f, P, F)

  # Update positional degrees of freedom.
  R, P, F, M = update_fn(R, P, F, M, dt, **kwargs)

  # Second half step.
  F = force_fn(R, **kwargs)
  P = tree_map(lambda p, f: p + dt_2 * f, P, F)

  return dataclasses.replace(state,
                             position=R,
                             momentum=P,
                             force=F)


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
    mass = canonicalize_mass(mass)
    P = initialize_momenta(R, mass, key, kT)
    force = force_fn(R, **kwargs)
    return NVEState(R, P, force, mass)

  @jit
  def step_fn(state, **kwargs):
    _dt = dt
    if 'dt' in kwargs:
      _dt = kwargs['dt']
      del kwargs['dt']
    update_fn = inner_update_fn(state.position, shift_fn, **sim_kwargs)
    return velocity_verlet(force_fn, update_fn, _dt, state, **kwargs)

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

  def init_fn(degrees_of_freedom, KE, kT):
    xi = jnp.zeros(chain_length, KE.dtype)
    p_xi = jnp.zeros(chain_length, KE.dtype)

    Q = kT * tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
    Q = Q.at[0].multiply(degrees_of_freedom)
    return NoseHooverChain(xi, p_xi, Q, tau, KE, degrees_of_freedom)

  def substep_fn(delta, P, state, kT):
    """Apply a single update to the chain parameters and rescales velocity."""
    xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

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

    return P, NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF), kT

  def half_step_chain_fn(P, state, kT):
    if chain_steps == 1 and sy_steps == 1:
      P, state, _ = substep_fn(dt, P, state, kT)
      return P, state

    delta = dt / chain_steps
    ws = jnp.array(SUZUKI_YOSHIDA_WEIGHTS[sy_steps])
    def body_fn(cs, i):
      d = f32(delta * ws[i % sy_steps])
      return substep_fn(d, *cs), 0
    P, state, _ = lax.scan(body_fn,
                           (P, state, kT),
                           jnp.arange(chain_steps * sy_steps))[0]
    return P, state

  def update_chain_mass_fn(state, kT):
    xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

    Q = kT * _tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
    Q = Q.at[0].multiply(DOF)

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
  dt = f32(dt)
  dt_2 = f32(dt / 2)
  if tau is None:
    tau = dt * 100
  tau = f32(tau)

  chain_fns = nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    mass = canonicalize_mass(mass)

    P = initialize_momenta(R, mass, key, _kT)
    KE = kinetic_energy(R, P, mass)
    dof = quantity.count_dof(R)
    return NVTNoseHooverState(R, P, force_fn(R, **kwargs), mass,
                              chain_fns.initialize(dof, KE, _kT))

  @jit
  def apply_fn(state, **kwargs):
    update_fn = inner_update_fn(state.position, shift_fn, **sim_kwargs)

    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    chain = state.chain

    chain = chain_fns.update_mass(chain, _kT)

    p, chain = chain_fns.half_step(state.momentum, chain, _kT)
    state = dataclasses.replace(state, momentum=p)

    state = velocity_verlet(force_fn, update_fn, dt, state, **kwargs)

    KE = kinetic_energy(state.position, state.momentum, state.mass)
    chain = dataclasses.replace(chain, kinetic_energy=KE)

    p, chain = chain_fns.half_step(state.momentum, chain, _kT)
    state = dataclasses.replace(state, momentum=p, chain=chain)

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
  KE = kinetic_energy(state.position, state.momentum, state.mass)

  DOF = quantity.count_dof(state.position)
  E = PE + KE

  c = state.chain

  E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
  for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
    E += p ** 2 / (2 * m) + kT * r
  return E


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
  V = V_0 * jnp.exp(dim * state.box_position)
  return V, lambda V: (V / V_0) ** (1 / dim) * ref


def npt_box(state: NPTNoseHooverState) -> Box:
  """Get the current box from an NPT simulation."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * jnp.exp(dim * state.box_position)
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

    mass = canonicalize_mass(mass)
    P = initialize_momenta(R, mass, key, kT)
    KE = quantity.kinetic_energy(momentum=P, mass=mass)

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

    return NPTNoseHooverState(R, P, force_fn(R, box=box, **kwargs), mass, box,
                              box_position, box_momentum, box_mass,
                              barostat.initialize(1, KE_box, _kT),
                              thermostat.initialize(R.size, KE, _kT))  # pytype: disable=wrong-arg-count

  def update_box_mass(state, kT):
    N, dim = state.position.shape
    dtype = state.position.dtype
    box_mass = jnp.array(dim * (N + 1) * kT * state.barostat.tau ** 2, dtype)
    return dataclasses.replace(state, box_mass=box_mass)

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

  def exp_iL2(alpha, P, F, P_b):
    x = alpha * P_b * dt_2
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
    P = exp_iL2(alpha, P, F, P_b)

    R_b = R_b + P_b / M_b * dt
    state = dataclasses.replace(state, box_position=R_b)

    vol, box_fn = _npt_box_info(state)

    box = box_fn(vol)
    R = exp_iL1(box, R, P / M, P_b / M_b)
    F = force_fn(R, box=box, **kwargs)

    P = exp_iL2(alpha, P, F, P_b)
    G_e = box_force(alpha, vol, box_fn, R, P, M, F, _pressure, **kwargs)
    P_b = P_b + dt_2 * G_e

    return dataclasses.replace(state,
                               position=R, momentum=P, mass=M, force=F,
                               box_position=R_b, box_momentum=P_b, box_mass=M_b)

  def apply_fn(state, **kwargs):
    S = state
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    bc = barostat.update_mass(S.barostat, _kT)
    tc = thermostat.update_mass(S.thermostat, _kT)
    S = update_box_mass(S, _kT)

    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)
    P, tc = thermostat.half_step(S.momentum, tc, _kT)

    S = dataclasses.replace(S, momentum=P, box_momentum=P_b)
    S = inner_step(S, **kwargs)

    KE = quantity.kinetic_energy(momentum=S.momentum, mass=S.mass)
    tc = dataclasses.replace(tc, kinetic_energy=KE)

    KE_box = quantity.kinetic_energy(momentum=S.box_momentum, mass=S.box_mass)
    bc = dataclasses.replace(bc, kinetic_energy=KE_box)

    P, tc = thermostat.half_step(S.momentum, tc, _kT)
    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)

    S = dataclasses.replace(S,
                            thermostat=tc, barostat=bc,
                            momentum=P, box_momentum=P_b)

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
  KE = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)

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

TODO: Make Langevin and Brownian simulations work with new `tree_map`
formalism.
"""

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


def nvt_langevin(energy_or_force: Callable[..., Array],
                 shift: ShiftFn,
                 dt: float,
                 kT: float,
                 gamma: float=0.1,
                 center_velocity: bool=True) -> Simulator:
  """Simulation in the NVT ensemble using the Langevin thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. Langevin
  dynamics are stochastic and it is supposed that the system is interacting
  with fictitious microscopic degrees of freedom. An example of this would be
  large particles in a solvent such as water. Thus, Langevin dynamics are a
  stochastic ODE described by a friction coefficient and noise of a given
  covariance.

  Our implementation follows the excellent set of lecture notes by Carlon,
  Laleman, and Nomidis [#carlon]_ .

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
  .. [#carlon] E. Carlon, M. Laleman, S. Nomidis. "Molecular Dynamics Simulation."
    http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
    Accessed on 06/05/2019.
  """

  force_fn = quantity.canonicalize_force(energy_or_force)

  dt_2 = f32(dt / 2)
  dt2 = f32(dt ** 2 / 2)
  dt32 = f32(dt ** (3.0 / 2.0) / 2)

  kT = f32(kT)

  gamma = f32(gamma)

  def init_fn(key, R, mass=f32(1), **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']
    mass = canonicalize_mass(mass)

    key, split = random.split(key)

    P = initialize_momenta(R, mass, key, kT)
    return NVTLangevinState(R, P, force_fn(R, **kwargs), mass, key)  # pytype: disable=wrong-arg-count

  def apply_fn(state, **kwargs):
    R, P, F, mass, key = dataclasses.astuple(state)

    _kT = kT if 'kT' not in kwargs else kwargs['kT']
    N, dim = R.shape

    key, xi_key, theta_key = random.split(key, 3)
    xi = random.normal(xi_key, (N, dim), dtype=R.dtype)
    sqrt3 = f32(jnp.sqrt(3))
    theta = random.normal(theta_key, (N, dim), dtype=R.dtype) / sqrt3

    # NOTE(schsam): We really only need to recompute sigma if the temperature
    # is nonconstant. @Optimization
    # TODO(schsam): Check that this is really valid in the case that the masses
    # are non identical for all particles.
    sigma = jnp.sqrt(f32(2) * _kT * gamma * mass)
    C = dt2 * (F - gamma * P) + sigma * dt32 * (xi + theta)

    R = shift(R, (dt * P + C) / mass, **kwargs)
    F_new = force_fn(R, **kwargs)
    P = (f32(1) - dt * gamma) * P + dt_2 * (F_new + F)
    P = P + sigma * jnp.sqrt(dt) * xi - gamma * C

    return NVTLangevinState(R, P, F_new, mass, key)  # pytype: disable=wrong-arg-count
  return init_fn, apply_fn


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
    mass = canonicalize_mass(mass)

    return BrownianState(R, mass, key)  # pytype: disable=wrong-arg-count

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
