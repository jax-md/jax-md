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
NOTE : all I/O units are in units of `openmm.unit.md_unit_system`
(see http://docs.openmm.org/latest/userguide/theory/01_introduction.html#units)
"""

from functools import wraps, partial

from typing import (Callable, Tuple, TextIO, Dict,
                    Any, Optional, Iterable, NamedTuple, Union)

import jax
import jax.numpy as jnp
import numpy as onp
from jax import ops
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk
from jax_md import (space, smap, partition, nn,
                    quantity, interpolate, util, dataclasses, energy)
maybe_downcast = util.maybe_downcast

# Types


i32 = util.i32
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
    epsilon: spring constant in kJ/(mol * nm**2).
        An ndarray of floats with shape `[n_bonds,]`
    length : spring equilibrium lengths in nm.
        An ndarray of floats with shape `[nbonds,]`
  """
  particles: Optional[Array] = None
  epsilon: Optional[Array] = None
  length: Optional[Array] = None

class HarmonicAngleParameters(NamedTuple):
  """A tuple containing parameter information for `HarmonicAngleEnergyFn`.

  Attributes:
    particles: The particle index tuples. An ndarray of floats with
      shape `[n_angles, 3]`.
    epsilon: spring constant in kJ/(mol * deg**2).
        An ndarray of floats with shape `[n_angles,]`
    length: spring equilibrium lengths in deg.
        An ndarray of floats with shape `[n_angles,]`
  """
  particles: Optional[Array] = None
  epsilon: Optional[Array] = None
  length: Optional[Array] = None

class PeriodicTorsionParameters(NamedTuple):
  """A tuple containing parameter information for `PeriodicTorsionEnergyFn`.

  Attributes:
    particles: The particle index tuples. An ndarray of floats with
      shape `[n_torsions, 4]`.
    amplitude: amplitude in kJ/(mol).
        An ndarray of floats with shape `[n_torsions,]`
    periodicity: periodicity of angle (unitless).
        An ndarray of floats with shape `[n_torsions,]`
    phase : angle phase shift in deg.
        An ndarray of floats with shape `[n_torsions,]`
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
    charge : charge in e on each particle.
        An ndarray of floats with shape `[n_particles,]`
    sigma : lennard_jones sigma term in nm.
        An ndarray of floats with shape `[n_particles,]`
    epsilon : lennard_jones epsilon in kJ/mol.
        An ndarray of floats with shape `[n_particles,]`
  """
  charge: Optional[Array] = None # this throws problems as the `energy.coulomb` requires `Q_sq`
  sigma: Optional[Array] = None
  epsilon: Optional[Array] = None



class MMEnergyFnParameters(NamedTuple):
  """A tuple containing parameter information for each
  `Parameters` NamedTuple which each `EnergyFn` can query

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

class Dummy(NamedTuple):
    """dummy namedtuple"""
    pass

# NOTE(dominicrufa): standardize naming convention; we typically use `OpenMM`
# force definitions, but this need not be the case
CANONICAL_MM_FORCENAMES = ['HarmonicBondForce',
                           'HarmonicAngleForce',
                           'PeriodicTorsionForce',
                           'NonbondedForce']
CANONICAL_MM_BOND_PARAMETER_PARTICLE_ALLOWABLES = {
                                _tup.__class__.__name__: i for _tup, i \
                                in zip([HarmonicBondParameters(),
                                HarmonicAngleParameters(),
                                PeriodicTorsionParameters(),
                                NonbondedExceptionParameters()], [2, 3, 4, 2])
                                }
CANONICAL_MM_BOND_PARAMETER_NAMES = [_key for _key in \
    CANONICAL_MM_BOND_PARAMETER_PARTICLE_ALLOWABLES.keys()]
CANONICAL_MM_NONBONDED_PARAMETER_NAMES = [_tup.__class__.__name__ for \
    _tup in [NonbondedParameters()]]

NONBONDED_COMBINATOR_DICT = {
                   'charge': lambda _q1, _q2: _q1*_q2,
                   'sigma': lambda _s1, _s2: f32(0.5)*(_s1 + _s2),
                   'epsilon': lambda _e1, _e2: jnp.sqrt(_e1*_e2)
                  }

NONBONDED_MOD_DICT_CONVERTER = {
    'charge': 'Q_sq',
    'sigma': 'sigma',
    'epsilon': 'epsilon'
}


# EnergyFn utilities

def camel_to_snake(_str, **unused_kwargs) -> str:
  return ''.join(['_'+i.lower() if i.isupper() else i for i in _str]).lstrip('_')

def snake_to_camel(_str, **unused_kwargs) -> str:
  return ''.join(word.title() for word in _str.split('_'))

def pair_parameter_combinator_mod(
    nonbonded_parameters_dict: Dict[str, Array],
    combinator_dict: Dict[str, Callable]=NONBONDED_COMBINATOR_DICT,
    key_mod_dict_converter: Dict[str, str]=NONBONDED_MOD_DICT_CONVERTER,
):
    out_dict = jax.tree_util.tree_map(
        lambda _combinator, _params: (_combinator, _params),
        combinator_dict,
        nonbonded_parameters_dict)
    mod_out_dict = {
        key_mod_dict_converter[key]: val for key, val in out_dict.items()}
    return mod_out_dict

def symmetrize_and_transpose(indices, **unused_kwargs):
    """
    convert an [N,2] array into [2,2*N] (symmetrized)
    """
    out = jnp.hstack((jnp.transpose(indices), \
    jnp.vstack((indices[:,1], indices[:,0]))))
    return out

def row_operation(row_number: int,
                  senders_receivers: Array,
                  max_neighbors: int,
                  fill_value: int) -> Array:
    """
    make a vmappable row operation for `get_dense_exception_mask_jax`
    """
    senders, receivers = senders_receivers
#     row_match = jnp.where(senders == row_number,
#                           receivers,
#                           fill_value)
    row_match = jnp.argwhere(senders == row_number,
                             size=max_neighbors,
                             fill_value=-1)
    out = jnp.where(row_match != -1, receivers[row_match], fill_value)
    return out.flatten()

def get_dense_exception_mask(num_particles: int,
                             extra_padding: int,
                             exceptions: Array) -> Array:
    """
    non-jittable function to return a dense mask passable to
    `smap.pair_neighbor_list`'s `custom_mask_function`

    This uses list comprehension;
    TODO : write this in canonical jax?
    """
    template_list = [[] for i in range(num_particles)]
    for exception_pair in exceptions:
        i,j = exception_pair
        template_list[i].append(j)
        template_list[j].append(i)
    max_counts = [len(i) for i in template_list]
    max_count = max(max_counts) + extra_padding

    padded_template = [_lst + [-1]*((max_count - len(_lst))) \
                      for _lst in template_list]
    return jnp.array(padded_template, dtype=jnp.int32)

def get_dense_exception_mask_jax(num_particles: int,
                                 max_neighbors: int,
                                 exceptions: int) -> Array:
    """
    try to build the same exception dict as `get_dense_exception_mask`,
    but with array comprehension only
    """
    symm_exceptions = symmetrize_and_transpose(exceptions)
    return jax.vmap(row_operation, in_axes=(0,None,None,None))(
        jnp.arange(num_particles),
        symm_exceptions,
        max_neighbors,
        -1)

def get_neighbor_custom_mask_function(
    n_particles: int,
    default_exceptions: Array,
    max_neighbors: int=None) -> Tuple[Callable, Array]:
    """
    mask particles appropriately;
    if `max_neighbors` is None, will compute the max number of exceptions
    to a single particle use that as `max_neighbors`.

    Returns:
        custom_mask_fn: a fn that takes the neighbor.idx as an input
            as well as a potential `dense_mask` (if not default)
            and masks the input neighbor.idx appropriately
        default_dense_mask: the default dense mask array
    """
    symm_transp_pair_exceptions = symmetrize_and_transpose(default_exceptions)
    if max_neighbors is None:
        unique, counts = onp.unique(default_exceptions.flatten(), return_counts=True)
        counts_dict = dict(zip(unique, counts))
        max_neighbors = max(counts_dict.values())

    default_dense_mask = get_dense_exception_mask_jax(
                                 num_particles=n_particles,
                                 max_neighbors=max_neighbors,
                                 exceptions=default_exceptions)

    def row_masking_fn(idx_row: Array, dense_mask_row, **unused_kwargs):
        return jnp.where(jnp.isin(idx_row, dense_mask_row), n_particles, idx_row)

    def custom_mask_fn(idx: Array, **dynamic_kwargs):
        in_dense_mask = dynamic_kwargs.get("dense_mask", default_dense_mask)
        masked_idx = jax.vmap(row_masking_fn, in_axes=(0,0))(idx, in_dense_mask)
        return masked_idx

    return custom_mask_fn, default_dense_mask

def get_bond_prereq_fns(
    displacement_fn: DisplacementFn,
    auxiliary_bond_fns_dict : Dict[str, Dict[str, Dict[str, Callable]]],
    **unused_kwargs) -> Dict[str, Callable]:
  """each of the CANONICAL_MM_BONDFORCENAMES has a different
    `geometry_handler_fn` for `smap.bond`;
  return a dict that each `bond` parameter can query.
     "harmonic_bond_parameters" is defaulted, so we can omit this
  """
  def angle_handler_fn(R: Array, bonds: Array, **_dynamic_kwargs):
    r1s, r2s, r3s = [R[bonds[:,i]] for i in range(3)]
    d = vmap(partial(displacement_fn, **_dynamic_kwargs), 0, 0)
    r21s, r23s = d(r1s, r2s), d(r3s, r2s)
    return (vmap(lambda _r1, _r2: jnp.arccos(
        quantity.cosine_angle_between_two_vectors(_r1, _r2)))(r21s, r23s),)

  def torsion_handler_fn(R: Array, bonds: Array, **_dynamic_kwargs):
    r1s, r2s, r3s, r4s = [R[bonds[:,i]] for i in range(4)]
    d = vmap(partial(displacement_fn, **_dynamic_kwargs), 0, 0)
    dR_12s, dR_32s, dR_34s = d(r2s, r1s), d(r2s, r3s), d(r4s, r3s)
    return (vmap(quantity.angle_between_two_half_planes)(dR_12s, dR_32s, dR_34s),)

  bond_fn_dict = {'HarmonicBondParameters':
                        {'geometry_handler_fn': None,
                         'singular_fn': energy.simple_spring},
                  'HarmonicAngleParameters':
                        {'geometry_handler_fn': angle_handler_fn,
                         'singular_fn': energy.simple_spring},
                  'PeriodicTorsionParameters':
                        {'geometry_handler_fn': torsion_handler_fn,
                         'singular_fn': energy.periodic_torsion},
                  'NonbondedExceptionParameters':
                        {'geometry_handler_fn': None,
                         'singular_fn': lambda *args, **kwargs: \
                                        energy.lennard_jones(*args, **kwargs) \
                                        + energy.coulomb(*args, **kwargs)
                                                     }
                 }
  bond_fn_dict.update(auxiliary_bond_fns_dict)
  return bond_fn_dict

def bonded_energy_handler(
                        displacement_fn,
                        default_parameters,
                        geometry_handler_fn,
                        singular_fn,
                        per_term=False,
                        bond_capture_kwargs=None,
                        **unused_kwargs):
    """
    create a smap.bond fn
    """
    camel_parameter_name = default_parameters.__class__.__name__
    snake_parameter_name = camel_to_snake(camel_parameter_name)
    parameter_template_dict = {key: None for key in default_parameters._fields if key != 'particles'}
    bond_fn = smap.bond(
        fn = singular_fn,
        geometry_handler_fn=geometry_handler_fn,
        displacement_or_metric=displacement_fn,
        per_term=per_term,
        capture_kwargs=bond_capture_kwargs,
        **parameter_template_dict)

    def energy_fn(R: Array,
                  parameters,
                  **dynamic_kwargs):
        bonds = parameters.particles
        bond_types = dict(parameters._asdict())
        _ = bond_types.pop('particles') # remove for redundancy
        out = bond_fn(R, bonds, bond_types, **dynamic_kwargs)
        return out

    def wrapped_energy_fn(R: Array,
                          parent_parameters_dict,
                          **dynamic_kwargs):
        energy_specific_parameters = \
            parent_parameters_dict.get(snake_parameter_name, default_parameters)
        return energy_fn(R, energy_specific_parameters, **dynamic_kwargs)

    return wrapped_energy_fn

def nonbonded_energy_handler(
    displacement_or_metric,
    default_parameters,
    neighbor_kwargs={}, # default empty dict triggers no neighbor update fn.
    default_particle_exception_indices=None,
    singular_nb_fn=None,
    per_term=False,
    combinator_dict = NONBONDED_COMBINATOR_DICT,
    key_mod_dict_converter=NONBONDED_MOD_DICT_CONVERTER):
    """
    handle the neighbor-compatible and neighbor-incompatible nonbonded fn;
    TODO : do we really want to
        `space.canonicalize_displacement_or_metric(displacement_fn)`
    TODO : allow for changing the `nonbonded_exception_indices`;
        currently, these are hardcoded if `use_neighbor_list`

    WARNING: at present, neighbor_list mode
        (the update of which occurs outside `mm_energy_fn`)
        does not allow for OTF updates of the `custom_mask_function`
        mask indices.
    """
    camel_parameter_name = default_parameters.__class__.__name__
    snake_parameter_name = camel_to_snake(camel_parameter_name)

    use_neighbor_list=False if neighbor_kwargs in [None, {}] else True

    # canonicalize the nb parameters
    canonicalized_nb_parameters = pair_parameter_combinator_mod(
        dict(default_parameters._asdict()),
        combinator_dict=combinator_dict,
        key_mod_dict_converter=key_mod_dict_converter
    )

    # query the singular nonbonded function
    if singular_nb_fn is None:
        singular_nb_fn = lambda *_args, **_kwargs: \
                         energy.lennard_jones(*_args, **_kwargs) \
                         + energy.coulomb(*_args, **_kwargs)

    # default the custom mask function, then query to define it
    custom_mask_function=None
    if default_particle_exception_indices is not None:
        # make the appropriate custom mask fn
        if not use_neighbor_list:
            custom_mask_function = smap.get_default_custom_mask_function(
                default_mask_indices = default_particle_exception_indices
                )
        else:
            num_particles = default_parameters[0].shape[0] # lead_axis=n_paricle
            custom_mask_function, _\
                =get_neighbor_custom_mask_function(
                num_particles,
                default_particle_exception_indices)

    # make the callable
    if not use_neighbor_list:
        # energy function is given by `smap.pair`
        energy_fn = smap.pair(
            singular_nb_fn,
            space.canonicalize_displacement_or_metric(displacement_or_metric),
            custom_mask_function=custom_mask_function,
            **canonicalized_nb_parameters
        )
        neighbor_list_fns = None
    else:
        energy_fn = smap.pair_neighbor_list(
            singular_nb_fn,
            space.canonicalize_displacement_or_metric(displacement_or_metric),
            **canonicalized_nb_parameters
        )
        neighbor_list_fns = partition.neighbor_list(
            custom_mask_function = custom_mask_function,
            **neighbor_kwargs,
        )


    def wrapped_energy_fn(R: Array,
                          parent_parameters_dict,
                          **dynamic_kwargs):
        energy_specific_parameters = parent_parameters_dict.get(
            snake_parameter_name,
            default_parameters)

        # query nonbonded exception particle indices
        nonbonded_exception_parameters = parent_parameters_dict.get(
            'nonbonded_exception_parameters',
            Dummy())
        nonbonded_exception_indices = \
            dict(nonbonded_exception_parameters._asdict()).get(
            'particles',
            default_particle_exception_indices)

        # mod the names of specified energy-specific parameters
        energy_specific_parameters =  {key_mod_dict_converter[key]: val for \
            key, val in dict(energy_specific_parameters._asdict()).items()
        }

        # query the nonbonded exception indices
        if not use_neighbor_list:
            energy_specific_parameters['mask_indices'] = nonbonded_exception_indices
        else:
            # default a `default_padded_exception_array`
            # otherwise, make a `padded_exception_array`
            # and add it to `energy_specific_parameters`
            pass
        merged_kwargs = util.merge_dicts(energy_specific_parameters,
          dynamic_kwargs)
        return energy_fn(R, **merged_kwargs)

    return wrapped_energy_fn, neighbor_list_fns

def mm_energy_fn(
    displacement_fn: DisplacementFn,
    default_mm_parameters: MMEnergyFnParameters,
    neighbor_kwargs: Dict[str, Any]={},
    auxiliary_bond_prereq_fns: Dict[str, Any]={},
    nonbonded_energy_handler_kwargs: Dict[str, Any]={},
    bond_capture_kwargs: Iterable[str]=None,
    ):
    """
    retrieve a energy function and a `NeighborListFns` if `neighbor_kwargs` is
    not an empty dict.

    Args:
        displacement_fn: A function `d(R_a, R_b)` that computes the displacement
            between pairs of points.
        default_mm_parameters: A `MMEnergyFnParameters` `NamedTuple`
        neighbor_kwargs: A Dict of parameters to pass to the
            `partition.neighbor_fn` when creating a `smap.pair_neighbor_list`
            fn. Default as `{}` to omit neighbor list construction and instead
            use `smap.pair`.
            Default will also return a `None` for the `neighbor_list_fns`
        auxiliary_bond_prereq_fns: A Dict of kwargs to pass to to
            `get_bond_prereq_fns`; default is {}
        nonbonded_energy_handler_kwargs: A Dict of kwargs to pass to the
            `nonbonded_energy_handler` fn when creating `nonbonded_parameters`
            energy_fn.
        bond_capture_kwargs: an iterable of strings that are passed to
          `smap.bond` as `capture_kwargs`
    Returns:
        energy_fn: a function to pass positions R and kwargs to return
            a floating point energy. see annotations below.
        neighbor_list_fns: A NeighborListFns object that contains a
            method to allocate a new neighbor list and a method to update an
            existing neighbor list; default is `None` since no neighbor kwargs
            is passed
    """
    assert default_mm_parameters.__class__.__name__ == 'MMEnergyFnParameters'
    # get bonded prerequisites
    bond_prereq_fns = get_bond_prereq_fns(displacement_fn,
                                          auxiliary_bond_prereq_fns)
    # query
    mm_energy_fns = {}
    neighbor_list_fns = None

    # query the `MMEnergyFnParameters` (snake case forces); attempt to register
    for parameter in default_mm_parameters:
        if parameter is None: # omit registration if the parameter is `None`
            continue
        parameter_name = parameter.__class__.__name__
        is_bonded = parameter_name \
        in list(CANONICAL_MM_BOND_PARAMETER_PARTICLE_ALLOWABLES.keys())
        if is_bonded and parameter.particles is None: # this is an empty default
            continue

        is_nonbonded = parameter_name \
        in list(CANONICAL_MM_NONBONDED_PARAMETER_NAMES)
        if is_bonded:
            energy_fn = bonded_energy_handler(
                displacement_fn=displacement_fn,
                default_parameters=parameter,
                per_term=False,
                geometry_handler_fn=bond_prereq_fns[parameter_name]\
                    ['geometry_handler_fn'],
                singular_fn=bond_prereq_fns[parameter_name]['singular_fn'],
                bond_capture_kwargs=bond_capture_kwargs,
            )
        elif is_nonbonded:
            if parameter[0] is None: # this is an empty default
                continue
            # first query exceptions
            exception_indices = None
            if ('nonbonded_exception_parameters' in \
                default_mm_parameters._fields):
                nonbonded_exception_parameters = \
                    default_mm_parameters.nonbonded_exception_parameters
                if nonbonded_exception_parameters.particles is not None:
                    exception_indices = nonbonded_exception_parameters.particles
            energy_fn, neighbor_list_fns = nonbonded_energy_handler(
                displacement_or_metric = displacement_fn,
                default_parameters = parameter,
                neighbor_kwargs=neighbor_kwargs,
                default_particle_exception_indices = exception_indices,
                **nonbonded_energy_handler_kwargs
                )
        else:
            raise NotImplementedError(f"""parameter {parameter_name}
                is not currently supported""")

        # don't allow duplicate forces
        assert parameter_name not in list(mm_energy_fns.keys())
        mm_energy_fns[parameter_name] = energy_fn

    # create callable
    def energy_fn(R: Array,
                  **dynamic_kwargs):
        """
        compute the molecular mechanics type energy.
        Args:
            R: input positions array of shape [N,3]
            dynamic_kwargs: dynamic kwargs that are passed
                explicitly to the tree_mapped bonded (multiple) and nonbonded
                (singular) functions. If `parameters` is passed, it is expected
                to be a `MMEnergyFnParameters` with the same nesting as the
                `default_mm_parameters`; if it is not passed, will default
                parameters to `default_mm_parameters`.
                Other dynamic_kwargs may include `neighbor` in the case neighbor
                lists are being used (specifically for the `NonbondedParameters`
                fn passable). All extra `dynamic_kwargs` are passed explicitly
                to the nonbonded `smap.pair` or `smap.pair_neighbor_list` fn
                since the bond fns have explicitly defined templating that
                should be unaffected by changes in the environment.
                TODO: fix this functionality for `space.periodic_general` since
                we need to pass `box` to the displacement fn
        Returns:
            energy: float of energy in kJ/mol

        """
        mm_parameters = dynamic_kwargs.get('parameters',
                                             default_mm_parameters)
        energies = jax.tree_util.tree_map(lambda e_fn:
                                            e_fn(R,
                                            parent_parameters_dict=\
                                                dict(mm_parameters._asdict()),
                                            **dynamic_kwargs),
                                mm_energy_fns)
        # do we want to return a dict or a singular float?
        accum = f32(0)
        accum = accum + util.high_precision_sum(
                            jnp.array(list(energies.values()))
                            )
        return accum
    return energy_fn, neighbor_list_fns
