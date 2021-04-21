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
    init_fn: function that initializes the  state of a system. Should take
      positions as an ndarray of shape [n, output_dimension]. Returns a state
      which will be a namedtuple.
    apply_fn: function that takes a state and produces a new state after one
      step of optimization.

  One question that we need to think about is whether the simulations should
  also return a function that computes the invariant for that ensemble. This
  can be used for testing purposes, but is not often used otherwise.
"""

from collections import namedtuple

from typing import Callable, TypeVar, Union, Tuple, Dict

from jax.api import grad
from jax import ops
from jax import random
import jax.numpy as np
from jax import lax

from jax_md import quantity, interpolate, util, space, dataclasses

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
simulations of this type. Namely, the velocity-verlet integration step that is
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
  """Apply a single step of velocity verlet integration to a state."""
  dt = f32(dt)
  dt_2 = f32(dt / 2)
  dt2_2 = f32(dt ** 2 / 2)

  R, V, F, M = state.position, state.velocity, state.force, state.mass

  Minv = 1 / M

  R = shift_fn(R, V * dt + F * dt2_2 * Minv, **kwargs)
  F_new = force_fn(R, **kwargs)
  V += (F + F_new) * dt_2 * Minv
  return dataclasses.replace(state,
                             position=R,
                             velocity=V,
                             force=F_new)


# Constant Energy Simulations


@dataclasses.dataclass
class NVEState:
  """A struct containing the state of an NVE simulation.

  This tuple stores the state of a simulation that samples from the
  microcanonical ensemble in which the (N)umber of particles, the (V)olume, and
  the (E)nergy of the system are held fixed.

  Attributes:
    position: An ndarray of shape [n, spatial_dimension] storing the position
      of particles.
    velocity: An ndarray of shape [n, spatial_dimension] storing the velocity
      of particles.
    force: An ndarray of shape [n, spatial_dimension] storing the force acting
      on particles from the previous step.
    mass: A float or an ndarray of shape [n] containing the masses of the
      particles.
  """
  position: Array
  velocity: Array
  force: Array
  mass: Array


# pylint: disable=invalid-name
def nve(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float) -> Simulator:
  """Simulates a system in the NVE ensemble.

  Samples from the microcanonical ensemble in which the number of particles (N),
  the system volume (V), and the energy (E) are held constant. We use a standard
  velocity verlet integration scheme.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    quant: Either a quantity.Energy or a quantity.Force specifying whether
      energy_or_force is an energy or force respectively.
  Returns:
    See above.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  def init_fn(key, R, kT, mass=f32(1.0), **kwargs):
    mass = quantity.canonicalize_mass(mass)
    V = np.sqrt(kT / mass) * random.normal(key, R.shape, dtype=R.dtype)
    V = V - np.mean(V, axis=0, keepdims=True)
    return NVEState(R, V, force_fn(R, **kwargs), mass)

  def step_fn(state, **kwargs):
    return velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)

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
    position: An ndarray of shape [chain_length] that stores the position of
      the chain.
    velocity: An ndarray of shape [chain_length] that stores the velocity of
      the chain.
    mass: An ndarray of shape [chain_length] that stores the mass of the
      chain.
    tau: The desired period of oscillation for the chain. Longer periods result
      is better stability but worse temperature control.
    kinetic_energy: A float that stores the current kinetic energy of the
      system that the chain is coupled to.
    degrees_of_freedom: An integer specifying the number of degrees of freedom
      that the chain is coupled to.
  """
  position: Array
  velocity: Array
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
  translation method outlined in [1] and the Nose-Hoover chains are updated
  using two half steps: one at the beginning of a simulation step and one at
  the end. The masses of the Nose-Hoover chains are updated automatically to
  enforce a specific period of oscillation, `tau`. Larger values of `tau` will
  yield systems that reach the target temperature more slowly but are also more
  stable.

  As described in [1], the Nose-Hoover chain often evolves on a faster
  timescale than the rest of the simulation. Therefore, it sometimes necessary
  to integrate the chain over several substeps for each step of MD. To do this
  we follow the Suzuki-Yoshida scheme. Specifically, we subdivide our chain
  simulation into $n_c$ substeps. These substeps are further subdivided into
  $n_sy$ steps. Each $n_sy$ step has length $\delta_i = \Delta t w_i / n_c$
  where $w_i$ are constants such that $\sum_i w_i = 1$. See the table of
  Suzuki_Yoshida weights above for specific values. The number of substeps
  and the number of Suzuki-Yoshida steps are set using the `chain_steps` and
  `sy_steps` arguments.

  Consequently, the Nose-Hoover chains are described by three functions: an
  `init_fn` that initializes the state of the chian, a `half_step_fn` that
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
    chain_steps: An integer specifying the number, $n_c$, of outer substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either 1, 3, 5, or 7.
    tau: A floating point timescale over which temperature equilibration occurs.
      Measured in units of dt. The performance of the Nose-Hoover chain
      thermostat can be quite sensitive to this choice.
  Returns:
    A triple of functions that initialize the chain, do a half step of
    simulation, and update the chain masses respectively.
  """

  def init_fn(degrees_of_freedom, KE, kT):
    xi = np.zeros(chain_length, KE.dtype)
    v_xi = np.zeros(chain_length, KE.dtype)

    Q = kT * tau ** f32(2) * np.ones(chain_length, dtype=f32)
    Q = ops.index_update(Q, 0, Q[0] * degrees_of_freedom)
    return NoseHooverChain(xi, v_xi, Q, tau, KE, degrees_of_freedom)

  def substep_fn(delta, V, state, kT):
    """Apply a single update to the chain parameters and rescales velocity."""

    xi, v_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

    delta_2 = delta   / f32(2.0)
    delta_4 = delta_2 / f32(2.0)
    delta_8 = delta_4 / f32(2.0)

    M = chain_length - 1

    G = (v_xi[M - 1] ** f32(2) * Q[M - 1] - kT) / Q[M]
    v_xi = ops.index_add(v_xi, M, delta_4 * G)

    def backward_loop_fn(v_xi_new, m): 
      G = (v_xi[m - 1] ** 2 * Q[m - 1] - kT) / Q[m]
      scale = np.exp(-delta_8 * v_xi_new)
      v_xi_new = scale * (scale * v_xi[m] + delta_4 * G)
      return v_xi_new, v_xi_new
    idx = np.arange(M - 1, 0, -1)
    _, v_xi_update = lax.scan(backward_loop_fn, v_xi[M], idx, unroll=2)
    v_xi = ops.index_update(v_xi, idx, v_xi_update)

    G = (f32(2.0) * KE - DOF * kT) / Q[0]
    scale = np.exp(-delta_8 * v_xi[1])
    v_xi = ops.index_update(v_xi, 0, scale * (scale * v_xi[0] + delta_4 * G))

    scale = np.exp(-delta_2 * v_xi[0])
    KE = KE * scale ** f32(2)
    V = V * scale

    xi = xi + delta_2 * v_xi

    G = (f32(2) * KE - DOF * kT) / Q[0]
    def forward_loop_fn(G, m):
      scale = np.exp(-delta_8 * v_xi[m + 1])
      v_xi_update = scale * (scale * v_xi[m] + delta_4 * G)
      G = (v_xi_update ** 2 * Q[m] - kT) / Q[m + 1]
      return G, v_xi_update
    idx = np.arange(M)
    G, v_xi_update = lax.scan(forward_loop_fn, G, idx, unroll=2)
    v_xi = ops.index_update(v_xi, idx, v_xi_update)
    v_xi = ops.index_add(v_xi, M, delta_4 * G)

    return V, NoseHooverChain(xi, v_xi, Q, _tau, KE, DOF), kT

  def half_step_chain_fn(V, state, kT):
    if chain_steps == 1 and sy_steps == 1:
      V, state, _ = substep_fn(dt, V, state, kT)
      return P, state

    delta = dt / chain_steps
    ws = np.array(SUZUKI_YOSHIDA_WEIGHTS[sy_steps], dtype=V.dtype)
    def body_fn(cs, i):
      d = f32(delta * ws[i % sy_steps])
      return substep_fn(d, *cs), 0
    V, state, _ = lax.scan(body_fn,
                           (V, state, kT),
                           np.arange(chain_steps * sy_steps))[0]
    return V, state

  def update_chain_mass_fn(state, kT):
    xi, v_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

    Q = kT * _tau ** f32(2) * np.ones(chain_length, dtype=f32)
    Q = ops.index_update(Q, 0, Q[0] * DOF)

    return NoseHooverChain(xi, v_xi, Q, _tau, KE, DOF)

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
      with shape [n, spatial_dimension].
    velocity: The velocity of particles. An ndarray of floats
      with shape [n, spatial_dimension].
    force: The current force on the particles. An ndarray of floats with shape
      [n, spatial_dimension].
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape [n].
    chain: The variables describing the Nose-Hoover chain.
  """
  position: Array
  velocity: Array
  force: Array
  mass: Array
  chain: NoseHooverChain


def nvt_nose_hoover(energy_or_force_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    kT: float,
                    chain_length: int=5,
                    chain_steps: int=2,
                    sy_steps: int=3,
                    tau: float=None) -> Simulator:
  """Simulation in the NVT ensemble using a Nose Hoover Chain thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. We use a
  Nose Hoover Chain (NHC) thermostat described in [1, 2, 3]. We follow the
  direct translation method outlined in [3] and the interested reader might
  want to look at that paper as a reference.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature inunits of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number, $n_c$, of outer substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either 1, 3, 5, or 7.
    tau: A floating point timescale over which temperature equilibration occurs.
      Measured in units of dt. The performance of the Nose-Hoover chain
      thermostat can be quite sensitive to this choice.
  Returns:
    See above.

  [1] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
      "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
      The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
  [2] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
      "Explicit reversible integrators for extended systems dynamics."
      Molecular Physics 87. (1998) 1117-1157.
  [3] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
      Andrea L. Jochim, and Glenn J. Martyna.
      "A Liouville-operator derived measure-preserving integrator for molecular
      dynamics simulations in the isothermal-isobaric ensemble."
      Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
  """
  dt = f32(dt)
  if tau is None:
    tau = dt * 100
  tau = f32(tau)

  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  chain_fns = nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    mass = quantity.canonicalize_mass(mass)
    V = np.sqrt(_kT / mass) * random.normal(key, R.shape, dtype=R.dtype)
    V = V - np.mean(V, axis=0, keepdims=True)
    KE = quantity.kinetic_energy(V, mass)

    return NVTNoseHooverState(R, V, force_fn(R, **kwargs), mass, 
                              chain_fns.initialize(R.size, KE, _kT))

  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    chain = state.chain

    chain = chain_fns.update_mass(chain, _kT)

    v, chain = chain_fns.half_step(state.velocity, chain, _kT)
    state = dataclasses.replace(state, velocity=v)

    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)

    KE = quantity.kinetic_energy(state.velocity, state.mass)
    chain = dataclasses.replace(chain, kinetic_energy=KE)

    v, chain = chain_fns.half_step(state.velocity, chain, _kT)
    state = dataclasses.replace(state, velocity=v, chain=chain)

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
  KE = quantity.kinetic_energy(state.velocity, state.mass)

  DOF = state.position.size
  E = PE + KE

  c = state.chain

  E += c.mass[0] * c.velocity[0] ** 2 / 2 + DOF * kT * c.position[0]
  for r, v, m in zip(c.position[1:], c.velocity[1:], c.mass[1:]):
    E += m * v ** 2 / 2 + kT * r
  return E


@dataclasses.dataclass
class NPTNoseHooverState:
  """State information for an NPT system with Nose-Hoover chain thermostats.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape [n, spatial_dimension].
    velocity: The velocity of particles. An ndarray of floats
      with shape [n, spatial_dimension].
    force: The current force on the particles. An ndarray of floats with shape
      [n, spatial_dimension].
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape [n].
    reference_box: A box used to measure relative changes to the simulation
      environment.
    box_position: A positional degree of freedom used to describe the current
      box. The box_position is parameterized as `box_position = (1/d)log(V/V_0)`
      where `V` is the current volume, `V_0` is the reference volume and `d` is
      the spatial dimension.
    box_velocity: A velocity degree of freedom for the box.
    box_mass: The mass assigned to the box.
    barostat: The variables describing the Nose-Hoover chain coupled to the
      barostat.
    thermostsat: The variables describing the Nose-Hoover chain coupled to the
      thermostat.
  """
  position: Array
  velocity: Array
  force: Array
  mass: Array

  reference_box: Box

  box_position: Array
  box_velocity: Array
  box_mass: Array

  barostat: NoseHooverChain
  thermostat: NoseHooverChain


def _npt_box_info(state: NPTNoseHooverState) -> Tuple[float,
                                                      Callable[[float], float]]:
  """Gets the current volume and a function to compute the box from volume."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * np.exp(dim * state.box_position)
  return V, lambda V: (V / V_0) ** (1 / dim) * ref


def npt_box(state: NPTNoseHooverState) -> Box:
  """Get the current box from an NPT simulation."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * np.exp(dim * state.box_position)
  return (V / V_0) ** (1 / dim) * ref


def npt_nose_hoover(energy_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    pressure: float,
                    kT: float,
                    barostat_kwargs: Dict=None,
                    thermostat_kwargs: Dict=None) -> Simulator:
  """Simulation in the NPT ensemble using a pair of Nose Hoover Chains.

  Samples from the canonical ensemble in which the number of particles (N),
  the system pressure (P), and the temperature (T) are held constant. We use a
  pair of Nose Hoover Chains (NHC) described in [1, 2, 3] coupled to the
  barostat and the thermostat respectively. We follow the direct translation
  method outlined in [3] and the interested reader might want to look at that
  paper as a reference.

  Args:
    energy_fn: A function that produces either an energy from a set of particle
      positions specified as an ndarray of shape [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
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

  [1] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
      "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
      The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
  [2] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
      "Explicit reversible integrators for extended systems dynamics."
      Molecular Physics 87. (1998) 1117-1157.
  [3] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
      Andrea L. Jochim, and Glenn J. Martyna.
      "A Liouville-operator derived measure-preserving integrator for molecular
      dynamics simulations in the isothermal-isobaric ensemble."
      Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
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

    V = np.sqrt(_kT / mass) * random.normal(key, R.shape, dtype=R.dtype)
    V = V - np.mean(V, axis=0, keepdims=True)
    KE = quantity.kinetic_energy(V, mass)

    # The box position is defined via pos = (1 / d) log V / V_0.
    zero = np.zeros((), dtype=R.dtype)
    one = np.ones((), dtype=R.dtype)
    box_position = zero
    box_velocity = zero
    box_mass = dim * (N + 1) * kT * barostat_kwargs['tau'] ** 2 * one
    KE_box = quantity.kinetic_energy(box_velocity, box_mass)

    if np.isscalar(box) or box.ndim == 0:
      # TODO(schsam): This is necessary because of JAX issue #5849.
      box = np.eye(R.shape[-1]) * box

    return NPTNoseHooverState(R, V, force_fn(R, box=box, **kwargs), mass, box,
                              box_position, box_velocity, box_mass,
                              barostat.initialize(1, KE_box, _kT),
                              thermostat.initialize(R.size, KE, _kT))

  def update_box_mass(state, kT):
    N, dim = state.position.shape
    dtype = state.position.dtype
    box_mass = np.array(dim * (N + 1) * kT * state.barostat.tau ** 2, dtype)
    return dataclasses.replace(state, box_mass=box_mass)

  def box_force(alpha, vol, box_fn, position, velocity, mass, force, pressure,
                **kwargs):
    N, dim = position.shape

    def U(vol):
      return energy_fn(position, box=box_fn(vol), **kwargs)

    dUdV = grad(U)
    KE2 = util.high_precision_sum(velocity ** 2 * mass)
    R = space.transform(box_fn(vol), position)
    RdotF = util.high_precision_sum(R * force)

    return alpha * KE2 + RdotF - dim * vol * dUdV(vol) - pressure * vol * dim

  def sinhx_x(x):
    """Taylor series for sinh(x) / x as x -> 0."""
    return 1 + x ** 2 / 6 + x ** 4 / 120

  def exp_iL1(box, R, V, V_b, **kwargs):
    x = V_b * dt
    x_2 = x / 2
    sinhV = sinhx_x(x_2)  # np.sinh(x_2) / x_2
    return shift_fn(R * np.exp(x), dt * V * np.exp(x_2) * sinhV, box=box,
                    **kwargs)

  def exp_iL2(alpha, V, A, V_b):
    x = alpha * V_b * dt_2
    x_2 = x / 2
    sinhV = sinhx_x(x_2)  # np.sinh(x_2) / x_2
    return V * np.exp(-x) + dt_2 * A * sinhV * np.exp(-x_2)

  def inner_step(state, **kwargs):
    _pressure = kwargs.pop('pressure', pressure)

    R, V, M, F = state.position, state.velocity, state.mass, state.force
    R_b, V_b, M_b = state.box_position, state.box_velocity, state.box_mass

    N, dim = R.shape

    vol, box_fn = _npt_box_info(state)

    alpha = 1 + 1 / N
    G_e = box_force(alpha, vol, box_fn, R, V, M, F, _pressure, **kwargs)
    V_b = V_b + dt_2 * G_e / M_b
    V = exp_iL2(alpha, V, F / M, V_b)

    R_b = R_b + V_b * dt
    state = dataclasses.replace(state, box_position=R_b)

    vol, box_fn = _npt_box_info(state)

    box = box_fn(vol)
    R = exp_iL1(box, R, V, V_b)
    F = force_fn(R, box=box, **kwargs)

    V = exp_iL2(alpha, V, F / M, V_b)
    G_e = box_force(alpha, vol, box_fn, R, V, M, F, _pressure, **kwargs)
    V_b = V_b + dt_2 * G_e / M_b

    return dataclasses.replace(state,
                               position=R, velocity=V, mass=M, force=F,
                               box_position=R_b, box_velocity=V_b, box_mass=M_b)

  def apply_fn(state, **kwargs):
    S = state
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    bc = barostat.update_mass(S.barostat, _kT)
    tc = thermostat.update_mass(S.thermostat, _kT)
    S = update_box_mass(S, _kT)

    V_b, bc = barostat.half_step(S.box_velocity, bc, _kT)
    V, tc = thermostat.half_step(S.velocity, tc, _kT)

    S = dataclasses.replace(S, velocity=V, box_velocity=V_b)
    S = inner_step(S, **kwargs)

    KE = quantity.kinetic_energy(S.velocity, S.mass)
    tc = dataclasses.replace(tc, kinetic_energy=KE)

    KE_box = quantity.kinetic_energy(S.box_velocity, S.box_mass)
    bc = dataclasses.replace(bc, kinetic_energy=KE_box)

    V, tc = thermostat.half_step(S.velocity, tc, _kT)
    V_b, bc = barostat.half_step(S.box_velocity, bc, _kT)

    S = dataclasses.replace(S,
                            thermostat=tc, barostat=bc,
                            velocity=V, box_velocity=V_b)

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
  KE = quantity.kinetic_energy(state.velocity, state.mass)

  DOF = state.position.size
  E = PE + KE

  c = state.thermostat
  E += c.mass[0] * c.velocity[0] ** 2 / 2 + DOF * kT * c.position[0]
  for r, v, m in zip(c.position[1:], c.velocity[1:], c.mass[1:]):
    E += m * v ** 2 / 2 + kT * r

  c = state.barostat
  for r, v, m in zip(c.position, c.velocity, c.mass):
    E += m * v ** 2 / 2 + kT * r

  E += pressure * volume
  E += state.box_mass * state.box_velocity ** 2 / 2

  return E


"""Stochastic Simulations

JAX MD includes integrators for stochastic simulations of Langevin dynamics and
Brownian motion for systems in the NVT ensemble with a solvent.
"""

@dataclasses.dataclass
class NVTLangevinState:
  """A struct containing state information for the Langevin thermostat.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape [n, spatial_dimension].
    velocity: The velocity of particles. An ndarray of floats with shape
      [n, spatial_dimension].
    force: The (non-stochistic) force on particles. An ndarray of floats with
      shape [n, spatial_dimension].
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape [n].
    rng: The current state of the random number generator.
  """
  position: Array
  velocity: Array
  force: Array
  mass: Array
  rng: Array

def nvt_langevin(energy_or_force: Callable[..., Array],
                 shift: ShiftFn,
                 dt: float,
                 kT: float,
                 gamma: float=0.1) -> Simulator:
  """Simulation in the NVT ensemble using the Langevin thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. Langevin
  dynamics are stochastic and it is supposed that the system is interacting with
  fictitious microscopic degrees of freedom. An example of this would be large
  particles in a solvent such as water. Thus, Langevin dynamics are a stochastic
  ODE described by a friction coefficient and noise of a given covariance.

  Our implementation follows the excellent set of lecture notes by Carlon,
  Laleman, and Nomidis [1].

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature inunits of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
  Returns:
    See above.

    [1] E. Carlon, M. Laleman, S. Nomidis. "Molecular Dynamics Simulation."
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
    mass = quantity.canonicalize_mass(mass)

    key, split = random.split(key)

    V = np.sqrt(_kT / mass) * random.normal(split, R.shape, dtype=R.dtype)
    V = V - np.mean(V, axis=0, keepdims=True)

    return NVTLangevinState(R, V, force_fn(R, **kwargs), mass, key)  # pytype: disable=wrong-arg-count

  def apply_fn(state, **kwargs):
    R, V, F, mass, key = dataclasses.astuple(state)

    _kT = kT if 'kT' not in kwargs else kwargs['kT']
    N, dim = R.shape

    key, xi_key, theta_key = random.split(key, 3)
    xi = random.normal(xi_key, (N, dim), dtype=R.dtype)
    theta = random.normal(theta_key, (N, dim), dtype=R.dtype) / np.sqrt(f32(3))

    # NOTE(schsam): We really only need to recompute sigma if the temperature
    # is nonconstant. @Optimization
    # TODO(schsam): Check that this is really valid in the case that the masses
    # are non identical for all particles.
    sigma = np.sqrt(f32(2) * _kT * gamma / mass)
    C = dt2 * (F - gamma * V) + sigma * dt32 * (xi + theta)

    R = shift(R, dt * V + C, **kwargs)
    F_new = force_fn(R, **kwargs)
    V = (f32(1) - dt * gamma) * V + dt_2 * (F_new + F)
    V = V + sigma * np.sqrt(dt) * xi - gamma * C

    return NVTLangevinState(R, V, F_new, mass, key)  # pytype: disable=wrong-arg-count
  return init_fn, apply_fn


@dataclasses.dataclass
class BrownianState:
  """A tuple containing state information for Brownian dynamics.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape [n, spatial_dimension].
    mass: The mmass of particles. Will either be a float or an ndarray of floats
      with shape [n].
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

  This code simulates Brownian dynamics which are synonymous with the overdamped
  regime of Langevin dynamics. However, in this case we don't need to take into
  account velocity information and the dynamics simplify. Consequently, when
  Brownian dynamics can be used they will be faster than Langevin. As in the
  case of Langevin dynamics our implementation follows [1].

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature inunits of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.

  Returns:
    See above.

    [1] E. Carlon, M. Laleman, S. Nomidis. "Molecular Dynamics Simulation."
        http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
        Accessed on 06/05/2019.
  """

  force_fn = quantity.canonicalize_force(energy_or_force)

  dt, gamma = static_cast(dt, gamma)

  def init_fn(key, R, mass=f32(1)):
    mass = quantity.canonicalize_mass(mass)

    return BrownianState(R, mass, key)  # pytype: disable=wrong-arg-count

  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    R, mass, key = dataclasses.astuple(state)

    key, split = random.split(key)

    F = force_fn(R, **kwargs)
    xi = random.normal(split, R.shape, R.dtype)

    nu = f32(1) / (mass * gamma)

    dR = F * dt * nu + np.sqrt(f32(2) * _kT * dt * nu) * xi
    R = shift(R, dR, **kwargs)

    return BrownianState(R, mass, key)  # pytype: disable=wrong-arg-count

  return init_fn, apply_fn
