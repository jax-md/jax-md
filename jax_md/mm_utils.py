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

"""`mm.py` utilities"""

from functools import wraps, partial

from typing import Callable, Tuple, TextIO, Dict, Any, Optional, Iterable, NamedTuple

import jax
import jax.numpy as jnp
from jax import ops
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk
from jax_md import space, smap, partition, mm, quantity, interpolate, util, dataclasses, energy


try:
  import openmm
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.message)
except Exception as exception:
    print(exception, False)
    print(exception.__class__.__name__ + ": " + exception.message)

from openmm import unit


maybe_downcast = util.maybe_downcast

# Types


f32 = util.f32
f64 = util.f64
i32 = util.i32
Array = jnp.array


# `openmm.System` to `mm.MMEnergyFnParameters` converter utilities


def bond_and_angle_parameter_retrieval_fn(force, **unused_kwargs):
  """retrieve bond and angle parameter namedtuple; put these together because their omm syntax is very similar"""
  force_name = force.__class__.__name__
  is_bond = force_name == 'HarmonicBondForce'
  if not is_bond: # is Angle
    assert force_name == 'HarmonicAngleForce', f"""retrieved force name may only be
    'HarmonicBondForce' or 'HarmonicAngleForce', but the force name received is {force_name}"""
    particle_query_fn = lambda _parameter_list: _parameter_list[:3]
    num_params = force.getNumAngles()
    param_query_fn = lambda _idx: force.getAngleParameters(_idx)
    param_tuple = mm.HarmonicAngleParameters
  else: # is Bond
    particle_query_fn = lambda _parameter_list: _parameter_list[:2]
    num_params = force.getNumBonds()
    param_query_fn = lambda _idx: force.getBondParameters(_idx)
    param_tuple = mm.HarmonicBondParameters
  particles, lengths, ks = [], [], []
  for idx in range(num_params):
    _params = param_query_fn(idx)
    particles.append(particle_query_fn(_params))
    lengths.append(_params[-2].value_in_unit_system(unit.md_unit_system))
    ks.append(_params[-1].value_in_unit_system(unit.md_unit_system))
  out_parameters = param_tuple(
    particles = Array(particles, dtype=i32),
    epsilon = Array(ks),
    length = Array(lengths)
    )
  return out_parameters

def torsion_parameter_retrieval_fn(force, **unused_kwargs):
  """retrieve periodic torsion parameter namedtuple"""
  particles, per, phase, k = [], [], [], []
  for idx in range(force.getNumTorsions()):
    _params = force.getTorsionParameters(idx)
    particles.append(_params[:4])
    per.append(_params[4])
    phase.append(_params[5].value_in_unit_system(unit.md_unit_system))
    k.append(_params[6].value_in_unit_system(unit.md_unit_system))
  out_parameters = mm.PeriodicTorsionParameters(
    particles = Array(particles, dtype=i32),
    amplitude = Array(k),
    periodicity = Array(per),
    phase = Array(phase)
  )
  return out_parameters

def nonbonded_exception_parameter_retrieval_fn(force, **unused_kwargs):
  """retrieve the nonbonded exceptions"""
  particles, Q_sq, sigma, epsilon = [], [], [], []
  for idx in range(force.getNumExceptions()):
    _params = force.getExceptionParameters(idx)
    particles.append(_params[:2])
    Q_sq.append(_params[2].value_in_unit_system(unit.md_unit_system))
    sigma.append(_params[3].value_in_unit_system(unit.md_unit_system))
    epsilon.append(_params[4].value_in_unit_system(unit.md_unit_system))
  out_parameters = mm.NonbondedExceptionParameters(
    particles = Array(particles, dtype=i32),
    Q_sq = Array(Q_sq),
    sigma = Array(sigma),
    epsilon = Array(epsilon)
  )
  return out_parameters

def nonbonded_parameter_retrieval_fn(force, **unused_kwargs):
  """retrieve `NonbondedParameters`"""
  charge, sigma, epsilon = [], [], []
  for idx in range(force.getNumParticles()):
    _params = force.getParticleParameters(idx)
    for _idx, _lst in zip([0,1,2], [charge, sigma, epsilon]):
      _lst.append(_params[_idx].value_in_unit_system(unit.md_unit_system))
  out_parameters = mm.NonbondedParameters(
    charge = Array(charge),
    sigma = Array(sigma),
    epsilon = Array(epsilon)
  )
  return out_parameters

def assert_supported_force(forces : Iterable[openmm.Force]) -> None:
  force_name_retrieval_dict = {}
  for idx, force in enumerate(forces):
    force_name = force.__class__.__name__
    if force_name not in mm.CANONICAL_MM_FORCENAMES:
      raise NotImplementedError(f"""force {idx} with name {force_name} is not currently supported;
      current supported forces are {mm.CANONICAL_MM_FORCENAMES}""")

def get_full_retrieval_fn_dict(**unused_kwargs) -> Dict[str, Callable[[openmm.Force, ...], NamedTuple]]:
  """get a dictionary to retrieve the entries of `mm.MMEnergyFnParameters`"""
  retrieval_dict = {
    'harmonic_bond_parameters': bond_and_angle_parameter_retrieval_fn,
    'harmonic_angle_parameters': bond_and_angle_parameter_retrieval_fn,
    'periodic_torsion_parameters': torsion_parameter_retrieval_fn,
    'nonbonded_exception_parameters': nonbonded_exception_parameter_retrieval_fn,
    'nonbonded_parameters': nonbonded_parameter_retrieval_fn,
    }
  retrieval_dict_key_set = set(list(retrieval_dict.keys()))
  energy_fn_namedtuple_key_set = set(list(mm.MMEnergyFnParameters()._asdict().keys()))
  if retrieval_dict_key_set != energy_fn_namedtuple_key_set:
    raise Exception(f"""
      There is an inconsistency in the parameterization.
      You are attempting to generate {retrieval_dict_key_set}
      but `mm.EnergyFnParameters` supports {energy_fn_namedtuple_key_set}
      """)
  return retrieval_dict

def parameters_from_openmm_system(system : openmm.System,
                                   **unused_kwargs) -> mm.MMEnergyFnParameters:
  """retrieve all parameters from an `openmm.System`

  Example:
  >>> from openmm import app, unit
  >>> pdb = app.PDBFile('alanine-dipeptide-explicit.pdb')
  >>> ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
  >>> mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False, removeCMMotion=False)
  >>> parameters = parameters_from_openmm_system(mmSystem)
  """
  forces = system.getForces()
  parameter_dict = {}
  assert_supported_force(forces)
  retrieval_dict = get_full_retrieval_fn_dict(**unused_kwargs)
  for force in forces:
    force_name = force.__class__.__name__
    if force_name == 'HarmonicBondForce':
      parameter_dict['harmonic_bond_parameters'] = retrieval_dict['harmonic_bond_parameters'](force, **unused_kwargs)
    elif force_name == 'HarmonicAngleForce':
      parameter_dict['harmonic_angle_parameters'] = retrieval_dict['harmonic_angle_parameters'](force, **unused_kwargs)
    elif force_name == 'PeriodicTorsionForce':
      parameter_dict['periodic_torsion_parameters'] = retrieval_dict['periodic_torsion_parameters'](force, **unused_kwargs)
    elif force_name == 'NonbondedForce':
      parameter_dict['nonbonded_parameters'] = retrieval_dict['nonbonded_parameters'](force, **unused_kwargs)
      parameter_dict['nonbonded_exception_parameters'] = retrieval_dict['nonbonded_exception_parameters'](force, **unused_kwargs)
  return mm.MMEnergyFnParameters(**parameter_dict)
