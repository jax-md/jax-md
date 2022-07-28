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

"""
Handler of molecular mechanics energy handling and aggregation;
NOTE : all I/O units are in units of `openmm.unit.md_unit_system` (see http://docs.openmm.org/latest/userguide/theory/01_introduction.html#units)
"""

from functools import wraps, partial

from typing import Callable, Tuple, TextIO, Dict, Any, Optional, Iterable, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
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
  charge: Optional[Array] = None # this throws problems as the `energy.coulomb` requires `Q_sq`
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
  harmonic_bond_parameters: Optional[HarmonicBondParameters] = None
  harmonic_angle_parameters: Optional[HarmonicAngleParameters] = None
  periodic_torsion_parameters: Optional[PeriodicTorsionParameters] = None
  nonbonded_exception_parameters: Optional[NonbondedExceptionParameters] = None
  nonbonded_parameters: Optional[NonbondedParameters] = None

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

def camel_to_snake(_str, **unused_kwargs) -> str:
   return ''.join(['_'+i.lower() if i.isupper() else i for i in _str]).lstrip('_')

def get_bond_fns(displacement_fn: DisplacementFn,
                 **unused_kwargs) -> Dict[str, Callable]:
  """each of the CANONICAL_MM_BONDFORCENAMES has a different `geometry_handler_fn` for `smap.bond`;
  return a dict that
     "harmonic_bond_parameters" is defaulted, so we can omit this
  """
  def angle_handler_fn(R: Array, bonds: Array, **_dynamic_kwargs):
    r1s, r2s, r3s = [R[bonds[:,i]] for i in range(3)]
    d = vmap(partial(displacement_fn, **_dynamic_kwargs), 0, 0)
    r21s, r23s = d(r1s, r2s), d(r3s, r2s)
    return (vmap(lambda _r1, _r2: jnp.arccos(quantity.cosine_angle_between_two_vectors(_r1, _r2)))(r21s, r23s),)

  def torsion_handler_fn(R: Array, bonds: Array, **_dynamic_kwargs):
    r1s, r2s, r3s, r4s = [R[bonds[:,i]] for i in range(4)]
    d = vmap(partial(displacement_fn, **_dynamic_kwargs), 0, 0)
    dR_12s, dR_32s, dR_34s = d(r2s, r1s), d(r2s, r3s), d(r4s, r3s)
    return (vmap(quantity.angle_between_two_half_planes)(dR_12s, dR_32s, dR_34s),)

  bond_fn_dict = {'harmonic_bond_parameters': {'geometry_handler_fn': None,
                                               'fn': energy.simple_spring},
                  'harmonic_angle_parameters': {'geometry_handler_fn': angle_handler_fn,
                                                'fn': energy.simple_spring},
                  'periodic_torsion_parameters': {'geometry_handler_fn': torsion_handler_fn,
                                                  'fn': energy.periodic_torsion},
                  'nonbonded_exception_parameters': {'geometry_handler_fn': None,
                                                     'fn': lambda *args, **kwargs: energy.lennard_jones(*args, **kwargs) + energy.coulomb(*args, **kwargs)
                                                     }
                 }
  return bond_fn_dict

COMBINATOR_DICT = {'charge': lambda _q1, _q2: _q1*_q2,
                   'sigma': lambda _s1, _s2: f32(0.5)*(_s1 + _s2),
                   'epsilon': lambda _e1, _e2: jnp.sqrt(_e1*_e2)
                  }

def nonbonded_neighbor_list(displacement_or_metric : DisplacementOrMetricFn,
                     nonbonded_parameters : NonbondedParameters,
                     use_neighbor_list : bool,
                     use_multiplicative_isotropic_cutoff : bool,
                     use_dsf_coulomb : bool,
                     multiplicative_isotropic_cutoff_kwargs : Dict[str, Any],
                     neighbor_kwargs: Dict[str, Any],
                     particle_exception_indices: Array=None,
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
  # define a nonbonded modifier for combinators
  def nonbonded_parameters_combinator_mod(nonbonded_parameters_dict, **unused_kwargs):
    out_dict = jax.tree_util.tree_map(lambda _combinator, _params: (_combinator, _params),
                                      COMBINATOR_DICT,
                                      nonbonded_parameters_dict)
    out_dict['Q_sq'] = out_dict['charge']
    return out_dict

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
    pair_nonbonded_fn = lambda *_args, **_kwargs: lj_energy_fn(*_args, **_kwargs) + coulomb_energy_fn(*_args, **_kwargs)
    # print(f"particle_exception_indices: {particle_exception_indices}")
    energy_fn = smap.pair(
      pair_nonbonded_fn,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      use_custom_mask = True if particle_exception_indices is not None else False,
      mask_indices = particle_exception_indices,
      **nonbonded_parameters_combinator_mod(nonbonded_parameters._asdict())
      )
    neighbor_fn = None
  def wrapped_energy_fn(R: Array, nonbonded_parameters_dict: Dict, **unused_kwargs) -> Array:
    # return energy_fn(R, **nonbonded_parameters_combinator_mod(nonbonded_parameters_dict))
    return energy_fn(R, **nonbonded_parameters_dict)
  return wrapped_energy_fn, neighbor_fn

def get_exception_match(idx : Array, pair_exception : Array, **unused_kwargs):
  """simple utility to return the exception match of a target `idx` from an exception pair;
     if the `pair_exception` doesn't contain the idx, return -1"""
  are_matches_bool = jnp.where(pair_exception == idx, True, False)
  non_matches = jnp.argwhere(idx != pair_exception, size=1)
  exception_idx = jax.lax.cond(jnp.any(are_matches_bool), lambda _x: pair_exception[_x[0]], lambda _x: _x[0]-1, non_matches)
  return exception_idx

def dense_to_sparse(idx):
  N = idx.shape[0]
  sender_idx = jnp.broadcast_to(jnp.arange(N)[:, None], idx.shape) # (N, N_neigh)
  sender_idx = jnp.reshape(sender_idx, (-1,))
  receiver_idx = jnp.reshape(idx, (-1,))
  return jnp.stack((sender_idx, receiver_idx), axis=0)

def sparse_to_dense(N, max_count, idx):
  senders, receivers = idx

  offset = jnp.tile(jnp.arange(max_count), N)[:len(senders)]
  hashes = senders * max_count + offset
  dense_idx = N * jnp.ones(((N + 1) * max_count,), i32)
  dense_idx = dense_idx.at[hashes].set(receivers).reshape((N + 1, max_count))
  return dense_idx[:-1]

def sparse_mask_to_dense_mask(sparse_mask_fn):
  def dense_mask_fn(idx, **kwargs): # idx shape (N, N_neigh)
    N, max_count = idx.shape
    sparse_idx = dense_to_sparse(idx)
    sparse_masked_idx = sparse_mask_fn(sparse_idx, **kwargs)
    return sparse_to_dense(sparse_masked_idx)
  return dense_mask_fn

# spse mask_pair shape (2, N_max) then let's say (2, N) were real and then (N_max - N, 2) were just
# (-1, -1)

def custom_mask_pairs(pairs):
  def mask_fn(idx, **kwargs):
    _mask_pairs = kwargs.get('mask_pairs', pairs)
    @partial(vmap, axis_in=(None, 0))
    def mask_single(idx, mask_pair):
      return jnp.where(idx != mask_pair, idx, mask_val)
    return mask_single(idx, _mask_pairs)


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

def register_bonded_parameter(parameter: NamedTuple, **unused_kwargs) -> bool:
  """
  check if a leaf of `MMEnergyFnParameters` is properly parameterized
  and lives in `CANONICAL_MM_BOND_PARAMETER_NAMES`.
  """
  register_parameter = False
  _parameter_tuple_name = parameter.__class__.__name__
  is_bonded = _parameter_tuple_name in [camel_to_snake(_item) for _item in CANONICAL_MM_BOND_PARAMETER_NAMES]
  if not is_bonded: # if it is not bonded, do not register
    return register_parameter
  # else query it to make sure it is valid
  bond_parameter_fields = parameter._fields
  if 'particles' not in bond_parameter_fields:
    raise ValueError(f"""retrieved bonded parameters {_parameter_tuple_name}
                      from parameter,
                      but 'particles' was not in 'fields'
                       ({bond_parameter_fields})""")
  if bond_parameters.particles is None:
    return register_parameter # this bond parameter is empty

  particles_shape = parameter.particles.shape
  num_bonds, bonds_shape = particles_shape
  allowable_bonds_shape = CANONICAL_MM_BOND_PARAMETER_PARTICLE_ALLOWABLES[_parameter_tuple_name]
  if bonds_shape != allowable_bonds_shape:
    raise ValueError(f"""parameter {_parameter_tuple_name} bonds shape of
                       {bonds_shape} does not match the allowed shape of
                       {allowable_bonds_shape}"""
                       )
  for nested_parameter_name, nested_parameter in parameters._asdict().items():
    if not util.is_array(nested_parameter): # each entry must be a parameter
      raise ValueError(f"""retrieved bonded parameters {_parameter_tuple_name}'s
                       nested parameter {nested_parameter_name}
                       but was not an array:
                       ({nested_parameter})""")
    nested_parameter_shape = nested_parameter.shape
    if (len(nested_parameter_shape) != 1) or (nested_parameter_shape[0] != num_bonds):
      raise ValueError(f"""retrieved bonded parameters {_parameter_tuple_name}'s
                       nested parameter {nested_parameter_name}
                       but entry was not an array of dim 1 and shape
                       {num_bonds}: ({nested_parameter_shape})""")
  # all passed, return True
  register_parameter = True
  return register_parameter

def register_nonbonded_parameter(parameter,
                                 allowed_nonbonded_
                                 **unused_kwargs):
  """
  check if a leaf of `MMEnergyFnParameters` is properly parameterized
  """

def register_parameters(parameters: MMEnergyFnParameters,
                        **unused_kwargs) -> Dict[str, Dict[str, None]]:
    """
    iterate over each parameter object in MMEnergyFnParameters,
    make appropriate assertions for bonded and nonbonded parameters;
    return a parameter_template to initialize bond calculations fns.
    """
    bonded_parameter_template, not_bonded_parameter_names = {}
    # query each field of each entry in the parameter tree
    for snake_parameter_name, parameter in parameters._asdict().items():
      # check if is a valid bonded parameter
      is_bonded = register_bonded_parameter(parameter)
      if is_bonded: # register the template, omitting `particles`
        bonded_parameter_template[snake_parameter_name] = {_key: None for _key in parameter._fields if _key != 'particles'}
      else: # it _must_ be nonbonded
        if not parameter.__class__.__name__ in CANONICAL_MM_NONBONDED_PARAMETER_NAMES:
          raise ValueError(f"""
          """)
        assert parameter.__class__.__name__ in CANONICAL_MM_NONBONDED_PARAMETER_NAMES
        parameter_lengths = []
        for entry in getattr(parameter_tree, _parameter_tuple_name):
          assert util.is_array(entry)
          assert len(entry.shape) == 1
          parameter_lengths.append(entry.shape[0])
        assert all(x == parameter_lengths[0] for x in parameter_lengths), f"all entries of nonbonded force must be consistent shapes"
    return parameter_template

def mm_energy_fn(displacement_fn : DisplacementFn,
                 parameters : MMEnergyFnParameters,
                 space_shape : Union[space.free, space.periodic, space.periodic_general] = space.periodic,
                 use_neighbor_list : Optional[bool] = True,
                 box_size: Optional[Box] = 1.,
                 use_multiplicative_isotropic_cutoff: Optional[bool]=True,
                 use_dsf_coulomb: Optional[bool]=True,
                 neighbor_kwargs: Optional[Dict[str, Any]]=None,
                 multiplicative_isotropic_cutoff_kwargs: Optional[Dict[str, Any]]={},
                 **unused_kwargs,
                 ) -> Union[EnergyFn, partition.NeighborListFns]:
  """
  generator of a canonical molecular mechanics-like `EnergyFn`;

  TODO :
    - render `nonbonded_exception_parameters.particles` static (requires changing `custom_mask_fn` handler)
    - retrieve standard nonbonded energy (already coded)
    - retrieve nonbonded energy per particle
    - render `.particles` parameters static (may affect jit compilation speeds)

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

  Example (vacuum from `openmm`):
  >>> pdb = app.PDBFile('alanine-dipeptide-explicit.pdb')
  >>> ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
  >>> mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False, removeCMMotion=False)
  >>> model = Modeller(pdb.topology, pdb.positions)
  >>> model.deleteWater()
  >>> mmSystem = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
  >>> context = openmm.Context(mmSystem, openmm.VerletIntegrator(1.*unit.femtoseconds))
  >>> context.setPositions(model.getPositions())
  >>> omm_state = context.getState(getEnergy=True, getPositions=True)
  >>> positions = jnp.array(omm_state.getPositions(asNumpy=True).value_in_unit_system(unit.md_unit_system))
  >>> energy = omm_state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
  >>> from jax_md import mm_utils
  >>> params = mm_utils.parameters_from_openmm_system(mmSystem)
  >>> displacement_fn, shift_fn = space.free()
  >>> energy_fn, neighbor_list = mm_energy_fn(displacement_fn=displacement_fn,
                                           parameters = params,
                                           space_shape=space.free,
                                           use_neighbor_list=False,
                                           box_size=None,
                                           use_multiplicative_isotropic_cutoff=False,
                                           use_dsf_coulomb=False,
                                           neighbor_kwargs={},
                                           )
  >>> out_energy = energy_fn(positions, parameters = params) # retrieve potential energy in units of `openmm.unit.md_unit_system` (kJ/mol)
  """
  check_support(space_shape, use_neighbor_list, box_size)

  # bonded energy fns
  bond_fns = get_bond_fns(displacement_fn) # get geometry handlers dict
  parameter_template = check_parameters(parameters) # just make sure that parameters
  for _key in parameter_template['bonded']:
    bond_fns[_key] = util.merge_dicts(bond_fns[_key], parameter_template[_key])
  bonded_energy_fns = {}
  for parameter_field in parameters._fields:
    if parameter_field in list(bond_fns.keys()): # then it is bonded
      mapped_bonded_energy_fn = smap.bond(
                                     displacement_or_metric=displacement_fn,
                                     **bond_fns[parameter_field], # `geometry_handler_fn` and `fn`
                                     )
      bonded_energy_fns[parameter_field] = mapped_bonded_energy_fn
    elif parameter_field in [camel_to_snake(_entry) for _entry in CANONICAL_MM_NONBONDED_PARAMETER_NAMES]: # nonbonded
      nonbonded_parameters = getattr(parameters, parameter_field)
      are_nonbonded_exception_parameters = True if 'nonbonded_exception_parameters' in parameters._fields else False
      if are_nonbonded_exception_parameters: # handle custom nonbonded mask
        n_particles=nonbonded_parameters.charge.shape[0] # query the number of particles
        padded_exception_array = query_idx_in_pair_exceptions(indices=jnp.arange(n_particles), pair_exceptions=getattr(getattr(parameters, 'nonbonded_exception_parameters'), 'particles'))
        custom_mask_fn = nonbonded_exception_mask_fn(n_particles=n_particles, padded_exception_array=padded_exception_array)
        neighbor_kwargs = util.merge_dicts({'custom_mask_fn': custom_mask_fn}, neighbor_kwargs)
      nonbonded_energy_fn, neighbor_fn = nonbonded_neighbor_list(displacement_or_metric=displacement_fn,
                             nonbonded_parameters=getattr(parameters, parameter_field),
                             use_neighbor_list=use_neighbor_list,
                             use_multiplicative_isotropic_cutoff=use_multiplicative_isotropic_cutoff,
                             use_dsf_coulomb=use_dsf_coulomb,
                             multiplicative_isotropic_cutoff_kwargs=multiplicative_isotropic_cutoff_kwargs,
                             particle_exception_indices=parameters.nonbonded_exception_parameters.particles if are_nonbonded_exception_parameters else None,
                             neighbor_kwargs=neighbor_kwargs)
    else:
      raise NotImplementedError(f"""parameter name {parameter_field} is not currently supported by
      `CANONICAL_MM_BOND_PARAMETER_NAMES` or `CANONICAL_MM_NONBONDED_PARAMETER_NAMES`""")

  def bond_handler(parameters, **unused_kwargs):
    """a simple function to easily `tree_map` bond functions"""
    bonds = {_key: getattr(parameters, _key).particles for _key in parameters._fields if _key in bond_fns.keys()}
    bond_types = {_key: getattr(parameters, _key)._replace(particles=None)._asdict() for _key in bonds.keys()}
    return bonds, bond_types

  def nonbonded_handler(parameters, **unused_kwargs) -> Dict:
    if 'nonbonded_parameters' in parameters._fields:
      return parameters.nonbonded_parameters._asdict()
    else:
      return {}

  def energy_fn(R: Array, **dynamic_kwargs) -> Array:
    accum = f32(0)
    bonds, bond_types = bond_handler(**dynamic_kwargs)
    bonded_energies = jax.tree_util.tree_map(lambda _f, _bonds, _bond_types : _f(R, _bonds, _bond_types),
                                             bonded_energy_fns, bonds, bond_types)
    accum = accum + util.high_precision_sum(jnp.array(list(bonded_energies.values())))

    # nonbonded
    nonbonded_parameters = nonbonded_handler(**dynamic_kwargs)
    nonbonded_energy = nonbonded_energy_fn(R, nonbonded_parameters_dict=nonbonded_parameters)
    accum = accum + nonbonded_energy # handle if/not in
    return accum

  return energy_fn, neighbor_fn
