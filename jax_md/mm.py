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

"""Handler of molecular mechanics energy handling and aggregation"""

from functools import wraps, partial

from typing import Callable, Tuple, TextIO, Dict, Any, Optional, Iterable, NamedTuple

import jax
import jax.numpy as jnp
from jax import ops
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk
from jax_md import space, smap, partition, nn, quantity, interpolate, util, dataclasses, energy

maybe_downcast = util.maybe_downcast

# Types


f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
MetricFn = space.MetricFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat


# MM Parameter Trees


class HarmonicBondParameters(NamedTuple):
  """A tuple containing parameter information for `HarmonicBondEnergyFn`.

  Attributes:
    particles: The particle index tuples. An ndarray of floats with
      shape `[n_bonds, 2]`.
    k: spring constant in kJ/(mol * nm**2). An ndarray of floats with shape `[n_bonds,]`
    r0 : spring equilibrium lengths in nm. an ndarray of floats with shape `[nbonds,]`
  """
  particles: Array
  k: Array
  r0: Array

class HarmonicAngleParameters(NamedTuple):
  """A tuple containing parameter information for `HarmonicAngleEnergyFn`.

  Attributes:
    particles: The particle index tuples. An ndarray of floats with
      shape `[n_angles, 3]`.
    k: spring constant in kJ/(mol * deg**2). An ndarray of floats with shape `[n_angles,]`
    theta0 : spring equilibrium lengths in deg. an ndarray of floats with shape `[n_angles,]`
  """
  particles: Array
  k: Array
  theta0: Array

class PeriodicTorsionParameters(NamedTuple):
  """A tuple containing parameter information for `PeriodicTorsionEnergyFn`.

  Attributes:
    particles: The particle index tuples. An ndarray of floats with
      shape `[n_torsions, 4]`.
    k: amplitude in kJ/(mol). An ndarray of floats with shape `[n_torsions,]`
    periodicity: periodicity of angle (unitless). An ndarray of floats with shape `[n_torsions,]`
    phase : angle phase shift in deg. an ndarray of floats with shape `[n_torsions,]`
  """
  particles: Array
  k: Array
  periodicity: Array
  phase: Array

class NonbondedParameters(NamedTuple):
  """A tuple containing parameter information for `NonbondedEnergyFn`.

  Attributes:
    charge : charge in e on each particle. An ndarray of floats with shape `[n_particles,]`
    sigma : lennard_jones sigma term in nm. An ndarray of floats with shape `[n_particles,]`
    epsilon : lennard_jones epsilon in kJ/mol. An ndarray of floats with shape `[n_particles,]`
    nonbonded_exception_pairs : pairs of particle exception indices. An ndarray
      of floats with shape `[n_exceptions,]`
    nonbonded_exception_chargeProd : chargeprod in e**2 on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
    nonbonded_exception_sigma : exception sigma in nm on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
    nonbonded_exception_epsilon : exception epsilon in kJ/mol on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
  """



# Valence Energy Functions


def HarmonicBondEnergyFn(R: Array,
                         parameter_tree: PyTree,
                         perturbation: Array,
                         metric_fn: MetricFn) -> Array:
    r1, r2 = R[parameter_tree.particles]
    dr = metric_fn(r1, r2, perturbation = perturbation)
    return energy.simple_spring(dr = dr, length = parameter_tree.r0,
                                epsilon = parameter_tree.k, alpha = 2)

def HarmonicAngleEnergyFn(R: Array,
                          parameter_tree: PyTree,
                          perturbation: Array,
                          displacement_fn: DisplacementFn) -> Array:
  # NOTE(dominicrufa): check that this computing the right angle within omm tolerance
  r1, r2, r3 = R[parameter_tree.particles]
  dR_12 = displacement_fn(r1, r2, perturbation)
  dR_32 = displacement_fn(r3, r2, perturbation)
  return simple_spring(dr = jnp.arccos(cosine_angle_between_two_vectors(dR_12, dR_32)),
                         length = parameter_tree.theta0,
                         epsilon = parameter_tree.k, alpha=2)

def PeriodicTorsionEnergyFn(R: Array,
                            parameter_tree: PyTree,
                            perturbation: Array,
                            displacement_fn: DisplacementFn) -> Array:
  r1, r2, r3, r4 = R[parameter_tree.particles]
  dR_12 = displacement_fn(r2, r1, perturbation=perturbation)
  dR_32 = displacement_fn(r2, r3, perturbation=perturbation)
  dR_34 = displacement_fn(r4, r3, perturbation=perturbation)
  torsion_angle = angle_between_two_half_planes(dR_12, dR_32, dR_34)
  return energy.periodic_torsion(torsion_angle = torsion_angle,
                                 amplitude = parameter_tree.k,
                                 periodicity = parameter_tree.periodicity,
                                 phase = parameter_tree.phase)


# Nonbonded Energy Functions
def NonbondedForce(R : Array,
                   neighbor_list_idx : Array,
                   parameter_tree: PyTree,
                   metric_fn : MetricFn,
                   charge_mixing_fn : Optional[Callable[[Array, Array], Array]]=lambda q1, q2: q1*q2,
                   sigma_mixiing_fn : Optional[Callable[[Array, Array], Array]]=lambda sig1, sig2: (sig1+sig2)/2,
                   epsilon_mixing_fn: Optional[Callable[[Array, Array], Array]]=lambda eps1, eps2: jnp.sqrt(eps1*eps2),
                   **nonbonded_kwargs,
                   ) -> Array:
  """ 'standard' nonbondedForce that takes a masked neighbor list idx, computes the neighbor distance_sq matrix,
      and computes lennard-jones and electrostatic interactions
  """

 def NonbondedExceptionForce(R: )
