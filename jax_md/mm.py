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
MaskFn = Callable[[Array], Array]


# MM Parameter Trees


# NOTE(dominicrufa): standardize naming convention; we typically use `OpenMM` force definitions, but this need not be the case
CANONICAL_MM_FORCENAMES = ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce']
CANONICAL_MM_BONDFORCENAMES = ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']



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

class StandardNonbondedParameters(NamedTuple):
  """A tuple containing parameter information for `StandardNonbondedEnergyFn`.

  Attributes:
    charge : charge in e on each particle. An ndarray of floats with shape `[n_particles,]`
    sigma : lennard_jones sigma term in nm. An ndarray of floats with shape `[n_particles,]`
    epsilon : lennard_jones epsilon in kJ/mol. An ndarray of floats with shape `[n_particles,]`
  """
  charge: Array
  sigma: Array
  epsilon: Array


class ExceptionNonbondedParameters(NamedTuple):
  """A tuple containing parameter information for `ExceptionNonbondedEnergyFn`.

  Attributes:
    nonbonded_exception_pair : pairs of particle exception indices. An ndarray
      of floats with shape `[n_exceptions,]`
    nonbonded_exception_chargeProd : chargeprod in e**2 on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
    nonbonded_exception_sigma : exception sigma in nm on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
    nonbonded_exception_epsilon : exception epsilon in kJ/mol on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
  """
  nonbonded_exception_pair: Array
  nonbonded_exception_chargeProd: Array
  nonbonded_exception_sigma: Array
  nonbonded_exception_epsilon: Array

class NonbondedParameters(NamedTuple):
    """A tuple containing parameter information for each `_NonbondedParameters` NamedTuple which the `NonbondedEnergyFn` can query
    """
    standard_nonbonded_parameters : NamedTuple
    exception_nonbonded_parameters: NamedTuple



class MMEnergyFnParameters(NamedTuple):
  """A tuple containing parameter information for each `Parameters` NamedTuple which each `EnergyFn` can query

  Attributes:
    harmonic_bond_parameters : HarmonicBondParameters
    harmonic_angle_parameters : HarmonicAngleParameters
    periodic_torsion_parameters : PeriodicTorsionParameters
    nonbonded_parameters : NonbondedParameters
  """
  harmonic_bond_parameters: NamedTuple
  harmonic_angle_parameters: NamedTuple
  periodic_torsion_parameters: NamedTuple
  nonbonded_parameters: NamedTuple



# EnergyFn utilities


def get_bond_geometry_handler_fns(displacement_fn: DisplacementFn,**kwargs) -> Dict[str, Callable]:
  """each of the CANONICAL_MM_BONDFORCENAMES has a different `geometry_handler_fn` for `smap.bond`;
  return a dict that
     "HarmonicBondForce" is defaulted, so we can omit this
  """
  def angle_handler_fn(R: Array, bonds: Array, **_dynamic_kwargs):
    r1s, r2s, r3s = [R[bonds[:,i]] for i in range(3)]
    d = vmap(partial(displacement_fn, **_dynamic_kwargs), 0, 0)
    r21s, r23s = d(r1s, r2s), d(r3s, r2s)
    return = (vmap(lambda _r1, _r2: jnp.arccos(cosine_angle_between_two_vectors(_r1, _r2)), 0, 0)(r21s, r23s),)

  def torsion_handler_fn(R: Array, bonds: Array, **_dynamic_kwargs):
    r1s, r2s, r3s, r4s = [R[bonds[:,i] for i in range(4)]
    d = vmap(partial(displacement_fn, **_dynamic_kwargs), 0, 0)
    dR_12s, dR_32s, dR_34s = d(r2s, r1s), d(r2s, r3s), d(r4s, r3s)
    return  = (vmap(angle_between_two_half_planes, 0, 0, 0)(dR_12, dR_32, dR_34),)

  return {'HarmonicBondForce': None, 'HarmonicAngleForce': angle_handler_fn, 'PeriodicTorsionForce': torsion_handler_fn}


def get_exception_match(idx : Array, pair_exception : Array):
  """simple utility to return the exception match of a target `idx` from an exception pair;
     if the `pair_exception` doesn't contain the idx, return -1"""
  are_matches_bool = jnp.where(pair_exception == idx, True, False)
  non_matches = jnp.argwhere(idx != pair_exception, size=1)
  exception_idx = jax.lax.cond(jnp.any(are_matches_bool), lambda _x: pair_exception[_x[0]], lambda _x: _x[0]-1, non_matches)
  return exception_idx

def query_idx_in_pair_exceptions(indices, pair_exceptions):
  """query the pair exceptions via vmapping and generate a padded [n_particles, max_exceptions] of exceptions corresponding to the leading axis idx;
     the output is used as the querying array for the `custom_mask_function` of the `neighbor_list`"""
  all_exceptions = vmap(vmap(get_exception_match, in_axes=(None, 0)), in_axes=(0,None))(indices, pair_exceptions).squeeze()
  all_exceptions_list = onp.array(all_exceptions).tolist()
  unique_exceptions = [set(_entry).difference({-1}) for _entry in all_exceptions_list]
  max_unique_exceptions = max([len(_entry) for _entry in unique_exceptions])
  safe_padded_exceptions = [list(_entry) + [-1]*(max_unique_exceptions - len(_entry)) for _entry in unique_exceptions]
  return jnp.array(safe_padded_exceptions)

def acceptable_id_pair(id1, id2, exception_array):
  """the index pair is acceptable if the id1-th entry of the `pair_lookup_array` does not contain any matches with the query idx id2"""
  return jnp.all(pair_lookup_array[id1] != id2)

def nonbonded_exception_mask_fn(n_particles, padded_exception_array) -> MaskFn:
  """generate a `MaskFn` custom mask function for the `neighbor_list` that omits entries in the `neighbor_list` which appear in the `padded_exception_array`.
     This makes the masking complexity O(max_exceptions_per_particle).
  """
  # NOTE(dominicrufa): need to benchmark this against a more naive strategy? use `Sparse` neighborlist and remove matches to exception array?
  def mask_id_based(idx, ids, mask_val, _acceptable_id_pair):
    # NOTE(dominicrufa): this is taken from the test for `custom_mapping_function`. since we are using it again, maybe we can abstract it a bit to avoid duplication
    @partial(vmap, in_axes=(0,0,None))
    def acceptable_id_pair(idx, id1, ids):
      id2 = ids.at[idx].get()
      return vmap(_acceptable_id_pair, in_axes=(None,0))(id1, id2)
    mask=acceptable_id_pair(idx, ids, ids)
    return jnp.where(mask, idx, mask_val)
  ids = jnp.arange(n_particles)
  mask_val = n_particles
  custom_mask_function = partial(mask_id_based, ids=ids, mask_val=mask_val, _acceptable_id_pair=partial(acceptable_id_pair, exception_array=padded_exception_array))


# Valence Energy Functions

def HarmonicBondEnergyFn(R: Array,
                         parameters : HarmonicBondParameters,
                         metric_fn: MetricFn,
                         **kwargs) -> Array:

    r1, r2 = R[parameter_tree.particles]
    dr = metric_fn(r1, r2, **kwargs)
    return energy.simple_spring(dr = dr, length = parameter_tree.r0,
                                epsilon = parameter_tree.k, alpha = 2)

def HarmonicAngleEnergyFn(R: Array,
                          parameter_tree: HarmonicAngleParameters,
                          box: Array,
                          displacement_fn: DisplacementFn) -> Array:
  # NOTE(dominicrufa): check that this computing the right angle within omm tolerance
  r1, r2, r3 = R[parameter_tree.particles]
  dR_12 = displacement_fn(r1, r2, box)
  dR_32 = displacement_fn(r3, r2, box)
  return simple_spring(dr = jnp.arccos(cosine_angle_between_two_vectors(dR_12, dR_32)),
                         length = parameter_tree.theta0,
                         epsilon = parameter_tree.k, alpha=2)

def PeriodicTorsionEnergyFn(R: Array,
                            parameter_tree: PeriodicTorsionParameters,
                            box: Array,
                            displacement_fn: DisplacementFn) -> Array:
  r1, r2, r3, r4 = R[parameter_tree.particles]
  dR_12 = displacement_fn(r2, r1, box=box)
  dR_32 = displacement_fn(r2, r3, box=box)
  dR_34 = displacement_fn(r4, r3, box=box)
  torsion_angle = angle_between_two_half_planes(dR_12, dR_32, dR_34)
  return energy.periodic_torsion(torsion_angle = torsion_angle,
                                 amplitude = parameter_tree.k,
                                 periodicity = parameter_tree.periodicity,
                                 phase = parameter_tree.phase)


# Nonbonded Energy Functions


def NonbondedStandardEnergyFn(R : Array,
                   neighbor_list_idx : Array,
                   parameter_tree: StandardNonbondedParameters,
                   box: Array,
                   charge_mixing_fn : Optional[Callable[[Array, Array], Array]]=lambda q1, q2: q1*q2,
                   sigma_mixiing_fn : Optional[Callable[[Array, Array], Array]]=lambda sig1, sig2: (sig1+sig2)/2,
                   epsilon_mixing_fn: Optional[Callable[[Array, Array], Array]]=lambda eps1, eps2: jnp.sqrt(eps1*eps2),
                   **unused_kwargs,
                   ) -> Array:
  """'standard' nonbonded function that takes a masked neighbor list idx to compute non-exception pairwise interactions;
     this is the only `EnergyFn` that is implicitly vmapped.
  """
  # NOTE(dominicrufa): we need to partial out more vectorizable functions based on which kind of nonbonded implementation we are using;
  # e.g. rxn field, pme, lj w/ dispersion, etc

def NonbondedExceptionEnergyFn(R: Array,
                               parameter_tree: ExceptionNonbondedParameters,
                               box: Array,
                               metric_fn: MetricFn,
                              **unused_kwargs,
                               ) -> Array:
  """ vmappable 'exception' nonbonded function that takes a precomputed dr_array of shape `[n_exceptions, 2]` and computes a total energy of
  pairwise coulombic and lennard_jones interactions"""
  # NOTE(dominicrufa): the "canonical" way of handling these is with vacuum electrostatics and lj terms sans cutoff or any other modifications
  # NOTE(dominicrufa): have to modify energy.dsf_coulomb for proper units.
  # NOTE(dominicrufa): have to
  r1, r2 = R[parameter_tree.nonbonded_exception_pair]
  dr = metric_fn(r1, r2, box)
  lj_term = energy.lennard_jones(dr = dr, sigma = parameter_tree.nonbonded_exception_sigma, epsilon = parameter_tree.nonbonded_exception_epsilon)
  return


def NonbondedEnergyFn(R: Array,
                      neighbor_list_idx : Array,
                      parameter_tree: NonbondedParameters,
                      box: Array,
                      metric_fn: MetricFn,
                      **kwargs) -> Array:
  """ 'complete' canonical nonbonded mixing function
  """
  exception_energies = util.high_precision_sum(
                                               vmap(NonbondedExceptionEnergyFn, in_axes=(None,0,None))(R,
                                                                                                       parameter_tree.exception_nonbonded_parameters,
                                                                                                       vmap(metric_fn, in_axes=(0,0,None)))
                                              )
  standard_energies = util.high_precision_sum() # standard nonbonded here
  return exception_energies + standard_energies

  # Aggregate All Energy Functions for full MM-type energy fn.


def exception_mask_fn(nonbonded_exception_pairs: Array) -> MaskFn:
  """create a `MaskFn` for the `NonbondedStandardEnergyFn` that will remove particle pair exceptions with a mask
  """
  pass




def get_final_energy_fn(displacement, **builder_kwargs):
    # make with/without neighbor list # smap.pair_neighbor_list
    # merge the dynamic kwargs
    # https://github.com/google/jax-md/blob/8633cd5ce50ac13f3ebbcc99f4670cfcbed4841e/jax_md/energy.py#L228

    def final_energy_fn(R, neighbor_list, displacement, **dynamic_kwargs):
        ...
        return energy

    return partial(final_energy_fn, displacement = displacement)
