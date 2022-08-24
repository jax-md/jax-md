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

from typing import (Callable, Tuple, TextIO, Dict,
                    Any, Optional, Iterable, NamedTuple)

from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
from jax import ops
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk
from jax_md import (space, smap, partition, mm,
                    quantity, interpolate, util, dataclasses, energy)


try:
  import openmm
except ImportError as error:
    print(error)
except Exception as exception:
    print(exception, False)
    print(exception)

from openmm import unit


maybe_downcast = util.maybe_downcast

# Types


f32 = util.f32
f64 = util.f64
i32 = util.i32
Array = jnp.array


# `openmm` general conversion utilities

def get_box_vectors_from_vec3s(
    vec3s : Tuple[unit.Quantity, unit.Quantity, unit.Quantity]) -> Array:
    """
    query a tuple object of vec3s to get a box array (for pbc-enabled nonbonded functions)
    Example:
    >>> a,b,c = system.getDefaultPeriodicBoxVectors()
    >>> bvs = get_box_vectors_from_vec3s((a,b,c))
    """
    rank = []
    enumers = [0,1,2]
    for idx, i in enumerate(vec3s):
        rank.append(i[idx].value_in_unit_system(unit.md_unit_system))
        lessers = [q for q in enumers if q != idx]
        for j in lessers:
            assert jnp.isclose(i[j].value_in_unit_system(unit.md_unit_system), \
                0.), f"vec3({i,j} is nonzero. vec3 is not a cube)"
    return Array(rank)


# `openmm.System` to `mm.MMEnergyFnParameters` converter utilities


def bond_and_angle_parameter_retrieval_fn(force, **unused_kwargs):
  """retrieve bond and angle parameter namedtuple;
  put these together because their omm syntax is very similar"""
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

def nonbonded_exception_parameter_retrieval_fn(
    force,
    parameter_template = mm.NonbondedExceptionParameters,
    **unused_kwargs):
  """retrieve the nonbonded exceptions"""
  particles, Q_sq, sigma, epsilon = [], [], [], []
  for idx in range(force.getNumExceptions()):
    _params = force.getExceptionParameters(idx)
    particles.append(_params[:2])
    Q_sq.append(_params[2].value_in_unit_system(unit.md_unit_system))
    sigma.append(_params[3].value_in_unit_system(unit.md_unit_system))
    epsilon.append(_params[4].value_in_unit_system(unit.md_unit_system))
  out_parameters = parameter_template(
    particles = Array(particles, dtype=i32),
    Q_sq = Array(Q_sq),
    sigma = Array(sigma),
    epsilon = Array(epsilon)
  )
  return out_parameters

def rf_nonbonded_exception_parameter_retrieval_fn(force, **unused_kwargs):
  """retrieve nonbonded exceptions in the reaction field regime"""
  class NonbondedExceptionParameters(NamedTuple):
    particles: jnp.array = None
    Q_sq: jnp.array = None
    sigma: jnp.array = None
    epsilon: jnp.array = None
    aux_Q_sq: jnp.array = None
  out_parameters = nonbonded_exception_parameter_retrieval_fn(
    force,
    NonbondedExceptionParameters)
  aux_Q_sqs = []
  for idx in range(force.getNumExceptions()):
    j, k, chargeprod, sigma, epsilon = force.getExceptionParameters(idx)
    ch1, _, _ = force.getParticleParameters(j)
    ch2, _, _ = force.getParticleParameters(k)
    aux_Q_sqs.append((ch1*ch2).value_in_unit_system(unit.md_unit_system))
  aux_Q_sqs = Array(aux_Q_sqs)
  out_parameters = out_parameters._replace(aux_Q_sq=aux_Q_sqs)
  del NonbondedExceptionParameters
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
      raise NotImplementedError(f"""force {idx} with name {force_name} \
        is not currently supported;
      current supported forces are {mm.CANONICAL_MM_FORCENAMES}""")

def get_full_retrieval_fn_dict(use_rf_exceptions: bool=False,
                               **unused_kwargs) -> \
    Dict[str, Callable[[openmm.Force], NamedTuple]]:
  """get a dictionary to retrieve the entries of `mm.MMEnergyFnParameters`"""
  retrieval_dict = {
    'harmonic_bond_parameters': bond_and_angle_parameter_retrieval_fn,
    'harmonic_angle_parameters': bond_and_angle_parameter_retrieval_fn,
    'periodic_torsion_parameters': torsion_parameter_retrieval_fn,
    'nonbonded_exception_parameters': \
        nonbonded_exception_parameter_retrieval_fn if not use_rf_exceptions else \
        rf_nonbonded_exception_parameter_retrieval_fn,
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
                                  **kwargs) -> mm.MMEnergyFnParameters:
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
  retrieval_dict = get_full_retrieval_fn_dict(**kwargs)
  for force in forces:
    force_name = force.__class__.__name__
    if force_name == 'HarmonicBondForce':
      parameter_dict['harmonic_bond_parameters'] = retrieval_dict['harmonic_bond_parameters'](force, **kwargs)
    elif force_name == 'HarmonicAngleForce':
      parameter_dict['harmonic_angle_parameters'] = retrieval_dict['harmonic_angle_parameters'](force, **kwargs)
    elif force_name == 'PeriodicTorsionForce':
      parameter_dict['periodic_torsion_parameters'] = retrieval_dict['periodic_torsion_parameters'](force, **kwargs)
    elif force_name == 'NonbondedForce':
      parameter_dict['nonbonded_parameters'] = retrieval_dict['nonbonded_parameters'](force, **kwargs)
      parameter_dict['nonbonded_exception_parameters'] = retrieval_dict['nonbonded_exception_parameters'](force, **kwargs)
  return mm.MMEnergyFnParameters(**parameter_dict)


# `openmm.NonbondedForce` to reaction field converter


class ReactionFieldConverter(object):
    """
    convert a canonical `openmm.System` object's `openmm.NonbondedForce`
    to a `openmm.CustomNonbondedForce` that treats electrostatics with reaction
    field;
    see: 10.1039/d0cp03835k;
    adapted from
    https://github.com/rinikerlab/reeds/blob/\
    52882d7e009b5393df172dd4b703323f1d84dabb/reeds/openmm/reeds_openmm.py#L265
    """
    def __init__(self,
                 system : openmm.System,
                 cutoff: float=1.2,
                 eps_rf: float=78.5,
                 ONE_4PI_EPS0: float=138.93545764438198,
                 **unused_kwargs,
                 ):
        """
        It is assumed that the nonbonded force is "canonical"
        in that it contains N particles and N_e exceptions
        without further decorating attrs.

        Args:
            system : openmm.System
            cutoff : float=1.2 (cutoff in nm)
            eps_rf : float=78.5; dielectric constant of solvent
        """
        import copy

        nbfs = [f for f in system.getForces() if f.__class__.__name__ \
            == "NonbondedForce"]
        assert len(nbfs) == 1, f"{len(nbfs)} nonbonded forces were found"

        self._nbf = nbfs[0]
        self._system = system
        self._cutoff = cutoff
        self._eps_rf = eps_rf
        self.ONE_4PI_EPS0 = ONE_4PI_EPS0

        pair_nbf = self.handle_nonbonded_pairs()
        exception_bf = self.handle_nb_exceptions()
        self_bf = self.handle_self_term()

    @property
    def rf_system(self):
        import copy
        new_system = copy.deepcopy(self._system)

        pair_nbf = self.handle_nonbonded_pairs()
        exception_bf = self.handle_nb_exceptions()
        self_bf = self.handle_self_term()

        # remove the nbf altogether
        for idx, force in enumerate(self._system.getForces()):
            if force.__class__.__name__ == 'NonbondedForce':
                break
        new_system.removeForce(idx)

        for force in [pair_nbf, exception_bf, self_bf]:
            new_system.addForce(force)
        return new_system

    def handle_nonbonded_pairs(self):
        energy_fn = self._get_energy_fn()
        energy_fn += f"chargeprod_ = charge1 * charge2;"

        custom_nb_force = openmm.CustomNonbondedForce(energy_fn)
        custom_nb_force.addPerParticleParameter('charge') # always add

        custom_nb_force.addPerParticleParameter('sigma')
        custom_nb_force.addPerParticleParameter('epsilon')
        custom_nb_force.setNonbondedMethod(openmm.CustomNonbondedForce.\
            CutoffPeriodic) # always
        custom_nb_force.setCutoffDistance(self._cutoff)
        custom_nb_force.setUseLongRangeCorrection(False) # for lj, never

        # add particles
        for idx in range(self._nbf.getNumParticles()):
            c, s, e = self._nbf.getParticleParameters(idx)
            custom_nb_force.addParticle([c, s, e])

        # add exclusions from nbf exceptions
        for idx in range(self._nbf.getNumExceptions()):
            j, k, _, _, _ = self._nbf.getExceptionParameters(idx)
            custom_nb_force.addExclusion(j,k)
        return custom_nb_force

    def handle_nb_exceptions(self):
        energy_fn = self._get_energy_fn(exception=True)
        custom_b_force = openmm.CustomBondForce(energy_fn)
        # add terms separately so we need not reimplement the energy fn
        for _param in ['chargeprod', 'sigma', 'epsilon', 'chargeprod_']:
            custom_b_force.addPerBondParameter(_param)

        # copy exceptions
        for idx in range(self._nbf.getNumExceptions()):
            j, k , chargeprod, mix_sigma, mix_epsilon = self._nbf.\
                getExceptionParameters(idx)

            # now query charges, sigma, epsilon
            c1, _, _ = self._nbf.getParticleParameters(j)
            c2, _, _ = self._nbf.getParticleParameters(k)

            custom_b_force.addBond(j, k,
                                [chargeprod, mix_sigma, mix_epsilon, c1*c2])

        return custom_b_force

    def handle_self_term(self):
        (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf) = self._get_rf_terms()

        crf_self_term = f"0.5 * ONE_4PI_EPS0 * chargeprod_ * (-crf);"
        crf_self_term += "ONE_4PI_EPS0 = {:f};".format(self.ONE_4PI_EPS0)
        crf_self_term += "crf = {:f};".format(crf)

        force_crf_self_term = openmm.CustomBondForce(crf_self_term)
        force_crf_self_term.addPerBondParameter('chargeprod_')
        force_crf_self_term.setUsesPeriodicBoundaryConditions(True)

        for i in range(self._nbf.getNumParticles()):
            ch1, _, _ = self._nbf.getParticleParameters(i)
            force_crf_self_term.addBond(i, i, [ch1*ch1])
        return force_crf_self_term

    def _get_rf_terms(self):
        cutoff, eps_rf = self._cutoff, self._eps_rf
        krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff**3)
        mrf = 4
        nrf = 6
        arfm = (3 * cutoff**(-(mrf+1))/(mrf*(nrf - mrf)))* \
            ((2*eps_rf+nrf-1)/(1+2*eps_rf))
        arfn = (3 * cutoff**(-(nrf+1))/(nrf*(mrf - nrf)))* \
            ((2*eps_rf+mrf-1)/(1+2*eps_rf))
        crf = ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + arfm * \
            cutoff**mrf + arfn * cutoff ** nrf
        return (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf)

    def _get_energy_fn(self, exception=False):
        """
        see https://github.com/rinikerlab/reeds/blob/\
        b8cf6895d08f3a85a68c892ad7d873ec129dd2c3/reeds/openmm/\
        reeds_openmm.py#L265
        """
        (cutoff, eps_rf, krf, mrf, nrf, arfm, arfn, crf) = self._get_rf_terms()

        # define additive energy terms
        #total_e = f"elec_e + lj_e;"
        total_e = "lj_e + elec_e;"
        # total_e += "elec_e = ONE_4PI_EPS0*chargeprod*(1/r + krf*r2 + arfm*r4 + arfn*r6 - crf);"
        total_e += f"elec_e = ONE_4PI_EPS0*( chargeprod*(1/r) + chargeprod_*(krf*r2 + arfm*r4 + arfn*r6 - crf));"
        total_e += f"lj_e = 4*epsilon*(sigma_over_r12 - sigma_over_r6);"
        total_e += "krf = {:f};".format(krf)
        total_e += "crf = {:f};".format(crf)
        total_e += "r6 = r2*r4;"
        total_e += "r4 = r2*r2;"
        total_e += "r2 = r*r;"
        total_e += "arfm = {:f};".format(arfm)
        total_e += "arfn = {:f};".format(arfn)
        total_e += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
        total_e += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
        total_e += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
        total_e += "sigma_over_r = sigma/r;"
        if not exception:
            total_e += "epsilon = sqrt(epsilon1*epsilon2);"
            total_e += "sigma = 0.5*(sigma1+sigma2);"
            total_e += "chargeprod = charge1*charge2;"
        total_e += "ONE_4PI_EPS0 = {:f};".format(self.ONE_4PI_EPS0)
        return total_e
