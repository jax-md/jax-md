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
EnergyFn = Callable[..., Array]


# MM Parameter Trees


class HarmonicBondParameters(NamedTuple):
  """A tuple containing parameter information for `HarmonicBondEnergyFn`.

  Attributes:
    particles: The particle index tuples. An ndarray of floats with
      shape `[n_bonds, 2]`.
    epsilon: spring constant in kJ/(mol * nm**2). An ndarray of floats with shape `[n_bonds,]`
    length : spring equilibrium lengths in nm. an ndarray of floats with shape `[nbonds,]`
  """
  particles: Optional[Array] = None
  epsilon: Optional[Array] = None
  length: Optional[Array] = None

class HarmonicAngleParameters(NamedTuple):
  """A tuple containing parameter information for `HarmonicAngleEnergyFn`.

  Attributes:
    particles: The particle index tuples. An ndarray of floats with
      shape `[n_angles, 3]`.
    epsilon: spring constant in kJ/(mol * deg**2). An ndarray of floats with shape `[n_angles,]`
    length: spring equilibrium lengths in deg. an ndarray of floats with shape `[n_angles,]`
  """
  particles: Optional[Array] = None
  epsilon: Optional[Array] = None
  length: Optional[Array] = None

class PeriodicTorsionParameters(NamedTuple):
  """A tuple containing parameter information for `PeriodicTorsionEnergyFn`.

  Attributes:
    particles: The particle index tuples. An ndarray of floats with
      shape `[n_torsions, 4]`.
    amplitude: amplitude in kJ/(mol). An ndarray of floats with shape `[n_torsions,]`
    periodicity: periodicity of angle (unitless). An ndarray of floats with shape `[n_torsions,]`
    phase : angle phase shift in deg. an ndarray of floats with shape `[n_torsions,]`
  """
  particles: Optional[Array] = None
  amplitude: Optional[Array] = None
  periodicity: Optional[Array] = None
  phase: Optional[Array] = None

class NonbondedExceptionParameters(NamedTuple):
  """A tuple containing parameter information for `NonbondedExceptionEnergyFn`.

  Attributes:
    particles : pairs of particle exception indices. An ndarray
      of floats with shape `[n_exceptions,]`
    Q_sq : chargeprod in e**2 on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
    sigma : exception sigma in nm on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
    epsilon : exception epsilon in kJ/mol on each exception.
      An ndarray of floats with shape `[n_exceptions,]`
    """
    particles: Optional[Array] = None
    Q_sq: Optional[Array] = None
    sigma: Optional[Array] = None
    epsilon: Optional[Array] = None

class NonbondedParameters(NamedTuple):
  """A tuple containing parameter information for `NonbondedForce`.

  Attributes:
    charge : charge in e on each particle. An ndarray of floats with shape `[n_particles,]`
    sigma : lennard_jones sigma term in nm. An ndarray of floats with shape `[n_particles,]`
    epsilon : lennard_jones epsilon in kJ/mol. An ndarray of floats with shape `[n_particles,]`
  """
  charge: Optional[Array] = None
  sigma: Optional[Array] = None
  epsilon: Optional[Array] = None



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
  nonbonded_exception_parameters: NamedTuple
  nonbonded_parameters: NamedTuple

# NOTE(dominicrufa): standardize naming convention; we typically use `OpenMM` force definitions, but this need not be the case
CANONICAL_MM_FORCENAMES = ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce']
CANONICAL_MM_BOND_PARAMETER_PARTICLE_ALLOWABLES = {
                                     _tup.__class__.__name__: i for _tup, i in zip([HarmonicBondParameters(),
                                     HarmonicAngleParameters(),
                                     PeriodicTorsionParameters(),
                                     NonbondedExceptionParameters()], [2, 3, 4, 2])
                                     }
CANONICAL_MM_BOND_PARAMETER_NAMES = [_key for _key in CANONICAL_MM_BOND_PARAMETER_PARTICLE_ALLOWABLES.keys()]
CANONICAL_MM_NONBONDED_PARAMETER_NAMES = [_tup.__class__.__name__ for _tup in [NonbondedParameters()]]


# EnergyFn utilities

def camel_to_snake(_str, **unused_kwargs): -> str
   return ''.join(['_'+i.lower() if i.isupper() else i for i in _str]).lstrip('_')

def get_bond_fns(displacement_fn: DisplacementFn, **unused_kwargs) -> Dict[str, Callable]:
  """each of the CANONICAL_MM_BONDFORCENAMES has a different `geometry_handler_fn` for `smap.bond`;
  return a dict that
     "harmonic_bond_parameters" is defaulted, so we can omit this
  """
  def angle_handler_fn(R: Array, bonds: Array, **_dynamic_kwargs):
    r1s, r2s, r3s = [R[bonds[:,i]] for i in range(3)]
    d = vmap(partial(displacement_fn, **_dynamic_kwargs), 0, 0)
    r21s, r23s = d(r1s, r2s), d(r3s, r2s)
    return = (vmap(lambda _r1, _r2: jnp.arccos(quantity.cosine_angle_between_two_vectors(_r1, _r2)))(r21s, r23s),)

  def torsion_handler_fn(R: Array, bonds: Array, **_dynamic_kwargs):
    r1s, r2s, r3s, r4s = [R[bonds[:,i] for i in range(4)]
    d = vmap(partial(displacement_fn, **_dynamic_kwargs), 0, 0)
    dR_12s, dR_32s, dR_34s = d(r2s, r1s), d(r2s, r3s), d(r4s, r3s)
    return  = (vmap(quantity.angle_between_two_half_planes)(dR_12, dR_32, dR_34),)

  bond_fn_dict = {'harmonic_bond_parameters': {'geometry_handler_fn': None,
                                               'fn': energy.simple_spring, alpha=2},
                  'harmonic_angle_parameters': {'geometry_handler_fn': angle_handler_fn,
                                                'fn': energy.simple_spring},
                  'periodic_torsion_parameters': {'geometry_handler_fn': torsion_handler_fn,
                                                  'fn': energy.periodic_torsion},
                  'nonbonded_exception_parameters': {'geometry_handler_fn': None,
                                                     'fn': lambda *args, **kwargs: energy.lennard_jones(*args, **kwargs) + energy.coulomb(*args, **kwargs)
                                                     }
                 }
  return bond_fn_dict

def nonbonded_neighbor_list(displacement_or_metric : DisplacementOrMetricFn,
                     nonbonded_parameters : NonbondedParameters,
                     use_neighbor_list : bool,
                     use_multiplicative_isotropic_cutoff : bool,
                     use_dsf_coulomb : bool,
                     multiplicative_isotropic_cutoff_kwargs,
                     neighbor_kwargs,
                     **unused_kwargs) -> Tuple[EnergyFn, NeighborFn]:
  """each of the nonbonded forces are handled here;
     space assertions are made with `check_support`, so we can omit them here.
     If `use_neighbor_list`, then `use_dsf_coulomb`, `use_multiplicative_isotropic_cutoff` assert True;

     TODO:
       - check `r_cutoff` is less than half box size if `use_neighbor_list`?
       - support `per_particle` computation? (how do we handle this with bonded interactions?)
       - support generalization for more nonbonded forces
       - throw warnings instead of errors if `assert`s fail?
       - do we want to call `maybe_downcast`for default parameters?
  """
  # NOTE(dominicrufa): we may want to make assertions about nonbonded cutoff if periodic and the value of alpha
  neighbor_kwargs = util.merge_dicts(multiplicative_isotropic_cutoff_kwargs, neighbor_kwargs)
  coulomb_energy_fn = energy.coulomb if not use_dsf_coulomb else energy.dsf_coulomb
  if use_neighbor_list:
    assert use_multiplicative_isotropic_cutoff # we require this for neighbor lists
    assert use_dsf_coulomb # we require this for neighbor lists
    coulomb_energy_fn = energy.dsf_coulomb # do we want to use the `multiplicative_isotropic_cutoff` for this, as well? need to check
    def pair_nonbonded_fn(*args, **kwargs):
      lj_e = energy.multiplicative_isotropic_cutoff(energy.lennard_jones, **neighbor_kwargs)(*args, **kwargs)
      coulomb_e = coulomb_energy_fn(*args, **kwargs)
      return lj_e + coulomb_e
    neighbor_fn = partition.neighbor_list(displacement_or_metric, **neighbor_kwargs)
    energy_fn = smap.pair_neighbor_list(
      pair_nonbonded_fn,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      **neighbor_kwargs,
      **nonbonded_parameters._asdict()
      )
  else:
    lj_energy_fn = energy.multiplicative_isotropic_cutoff(energy.lennard_jones, **neighbor_kwargs) if use_multiplicative_isotropic_cutoff else energy.lennard_jones
    pair_nonbonded_fn = lambda *args, **kwargs: lj_e(*args, **kwargs) + coulomb_energy_fn(*args, **kwargs)
    energy_fn = smap.pair(
      pair_nonbonded_fn,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      **nonbonded_parameters._asdict()
      )
    neighbor_fn = None
  return energy_fn, neighbor_fn

def get_exception_match(idx : Array, pair_exception : Array, **unused_kwargs):
  """simple utility to return the exception match of a target `idx` from an exception pair;
     if the `pair_exception` doesn't contain the idx, return -1"""
  are_matches_bool = jnp.where(pair_exception == idx, True, False)
  non_matches = jnp.argwhere(idx != pair_exception, size=1)
  exception_idx = jax.lax.cond(jnp.any(are_matches_bool), lambda _x: pair_exception[_x[0]], lambda _x: _x[0]-1, non_matches)
  return exception_idx

def query_idx_in_pair_exceptions(indices, pair_exceptions, **unused_kwargs):
  """query the pair exceptions via vmapping and generate a padded [n_particles, max_exceptions] of exceptions corresponding to the leading axis idx;
     the output is used as the querying array for the `custom_mask_function` of the `neighbor_list`"""
  all_exceptions = vmap(vmap(get_exception_match, in_axes=(None, 0)), in_axes=(0,None))(indices, pair_exceptions).squeeze()
  all_exceptions_list = onp.array(all_exceptions).tolist()
  unique_exceptions = [set(_entry).difference({-1}) for _entry in all_exceptions_list]
  max_unique_exceptions = max([len(_entry) for _entry in unique_exceptions])
  safe_padded_exceptions = [list(_entry) + [-1]*(max_unique_exceptions - len(_entry)) for _entry in unique_exceptions]
  return jnp.array(safe_padded_exceptions)

def acceptable_id_pair(id1, id2, exception_array, **unused_kwargs):
  """the index pair is acceptable if the id1-th entry of the `pair_lookup_array` does not contain any matches with the query idx id2"""
  return jnp.all(pair_lookup_array[id1] != id2)

def nonbonded_exception_mask_fn(n_particles, padded_exception_array, **unused_kwargs) -> MaskFn:
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
  return custom_mask_function


def check_support(space_shape, use_neighbor_list, box_size, **kwargs):
  """`mm_energy_fn` supports all spaces with neighbor_lists;
    `space.periodic` requires an initial box_size
  """
  if space_shape == space.free:
    assert box_size is None
  else:
    assert box_size is not None
    assert use_neighbor_list


def check_parameters(parameter_tree : MMEnergyFnParameters) -> bool:
    bond_parameter_particle_allowables = {camel_to_snake(_key): val for _key, val in CANONICAL_MM_BOND_PARAMETER_PARTICLE_ALLOWABLES.items()}
    for _parameter_tuple_name in parameter_tree._fields: # iterate over each parameter object
      is_bonded = _parameter_tuple_name in [camel_to_snake(_item for _item in CANONICAL_MM_BOND_PARAMETER_NAMES]
      if is_bonded: # `particles` must be an entry`
        nested_tuple = getattr(parameter_tree, _parameter_tuple_name)
        nested_tuple_fields = nested_tuple._fields
        if 'particles' not in nested_tuple_fields:
          raise ValueError(f"""retrieved bonded parameters {_parameter_tuple_name} from parameter_tree,
                            but 'particles' was not in 'fields' ({nested_tuple_fields})}""")
        assert util.is_array(nested_tuple.particles)
        particles_shape = nested_tuple.particles
        assert len(particles_shape) == 2
        num_bonds, bonds_shape = particles_shape
        if not bonds_shape == bond_parameter_particle_allowables[_parameter_tuple_name]:
          raise ValueError(f"""bonds shape of {bonds_shape} of parameter name {_parameter_tuple_name}
                            does not match the allowed dictionary entry of {bond_parameter_particle_allowables[_parameter_tuple_name]}""")
        for _field_name in nested_tuple_fields:
          if _field_name == 'particles':
            pass # handled
          _param = getattr(nested_tuple, _field_name)
          assert util.is_array(_param), f"bond parameter {_field_name} is not an array."
          assert len(_param.shape) == 1
          assert _param.shape[0] == num_bonds
      else:
        assert _parameter_tuple_name in [camel_to_snake(_entry) for _entry in CANONICAL_MM_NONBONDED_PARAMETER_NAMES]
        parameter_shapes = []
        for entry in getattr(parameter_tree, _parameter_tuple_name):
          assert util.is_array(entry)
          assert len(entry.shape) == 1
          parameter_lengths.append(entry.shape[0])
        assert all(x == parameter_lengths[0] for x in parameter_lengths), f"all entries of nonbonded force must be consistent shapes"


# Energy Functions


def mm_energy_fn(displacement_fn : DisplacementFn,
                 parameters : MMEnergyFnParameters,
                 space_shape : Union[space.free, space.periodic, space.periodic_general] = space.periodic,
                 use_neighbor_list : Optional[bool] = True,
                 box_size: Optional[Box] = 1.,
                 use_multiplicative_isotropic_cutoff: Optional[bool]=True,
                 use_dsf_coulomb: Optional[bool]=True,
                 neighbor_kwargs: Optional[Dict[str, Any]]=None,
                 multiplicative_isotropic_cutoff_kwargs: Optional[Dict[str, Any]]=None,
                 **unused_kwargs,
                 ) -> Union[EnergyFn, NeighborListFns]:
  """
  generator of a canonical molecular mechanics-like `EnergyFn`;

  TODO :
    - partial vacuum neighbor list
    - make `nonbonded_exception_parameters.particles` dynamic (requires changing `custom_mask_fn` handler)
    -

  Args:
    displacement_fn: A `DisplacementFn`.
    parameters: An `MMEnergyFnParameters` containing all of the parameters of the model;
      While these are dynamic, `mm_energy_fn` does effectively make some of these static, namely the `particles` parameter
    space_shape : A generalized `jax_md.space`
    use_neighbor_list : whether to use a neighbor list for `NonbondedParameters`
    box_size : size of box for `space.periodic` or `space.periodic_general`; omitted for `space.free`
  Returns:
    An `EnergyFn` taking positions R (an ndarray of shape [n_particles, 3]), parameters,
      (optionally) a `NeighborList`, and optional kwargs
    A `neighbor_fn` for allocating and updating a neighbor_list
  """
  check_support(space_shape, use_neighbor_list, box_size)

  # bonded energy fns
  bond_fns = get_bond_fns(displacement_fn) # get geometry handlers dict
  check_parameters(parameters) # just make sure that parameters
  bonded_energy_fns = {}
  for parameter_field in parameters._fields:
    if parameter_field in list(bond_fns.keys()): # then it is bonded
      bond_kwargs = getattr(parameters, parameter_field)._asdict()
      mapped_bonded_energy_fn = smap.bond(displacement_or_metric=displacement_fn,
                                     **bond_fns[parameter_field], # `geometry_handler_fn` and `fn`
                                     **bond_kwargs)
      bonded_energy_fns[parameter_field] = mapped_bonded_energy_fn
    elif parameter_field in [camel_to_snake(_entry) for _entry in CANONICAL_MM_NONBONDED_PARAMETER_NAMES]: # nonbonded
      nonbonded_parameters = getattr(parameters, parameter_field)
      if 'nonbonded_exception_parameters' in parameters._fields: # handle custom nonbonded mask
        n_particles=nonbonded_parameters.charges.shape[0]
        padded_exception_array = query_idx_in_pair_exceptions(indices=jnp.arange(n_particles), pair_exceptions=getattr(getattr(parameters, 'nonbonded_exception_parameters'), 'particles'))
        custom_mask_fn = nonbonded_exception_mask_fn(n_particles=n_particles, padded_exception_array=padded_exception_array)
        neighbor_kwargs = util.merge_dicts({'custom_mask_fn': custom_mask_fn}, neighbor_kwargs)
      else:
        nonbonded_energy_fn, neighbor_fn = nonbonded_neighbor_list(displacement_or_metric=displacement_fn,
                                 nonbonded_parameters=getattr(parameters, parameter_field),
                                 use_neighbor_list=use_neighbor_list,
                                 use_multiplicative_isotropic_cutoff=use_multiplicative_isotropic_cutoff,
                                 use_dsf_coulomb=use_dsf_coulomb,
                                 multiplicative_isotropic_cutoff_kwargs=multiplicative_isotropic_cutoff_kwargs,
                                 neighbor_kwargs=neighbor_kwargs)
    else:
      raise NotImplementedError(f"""parameter name {parameter_field} is not currently supported by
      `CANONICAL_MM_BOND_PARAMETER_NAMES` or `CANONICAL_MM_NONBONDED_PARAMETER_NAMES`""")

  def bond_handler(parameters, **unused_kwargs):
    """a simple function to easily `tree_map` bond functions"""
      bonds = {_key: getattr(parameters, _key).particles for _key in parameters._fields if _key in [bond_fns.keys()]}
      bond_types = {_key: parameters[_key]._asdict() for _key in bonds.keys()}
      # this is a deprecated version of the above code. (remove this before merge)
      # bond_types = {_key: {__key: getattr(getattr(parameters, _key), __key) for __key in [_field for _field in _key._fields if _field != 'particles']}
      #                     for _key in parameters._fields if _key in [bond_fns.keys()]}
      return bonds, bond_types

  def energy_fn(R: Array, **dynamic_kwargs) -> Array:
    accum = f32(0)
    bonds, bond_types = bond_handler(**dynamic_kwargs)
    bonded_energies = jax.tree_util.tree_map(lambda _f, _bonds, _bond_types : _f(R, _bonds, _bond_types, **dynamic_kwargs),
                                             bonded_energy_fns, (bonds, bond_types))
    accum = accum + util.high_precision_sum(bonded_energies)

    # nonbonded
    accum = accum + nonbonded_energy_fn(R, **dynamic_kwargs) # handle if/not in
    return accum

  return energy_fn, neighbor_fn
