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

"""Tests for google3.third_party.py.jax_md.mm."""

from absl.testing import absltest
from absl.testing import parameterized

from jax.config import config as jax_config

from jax import random
import jax.numpy as np

from jax import grad

from jax import jit, vmap

from jax_md import smap, space, energy, quantity, partition, dataclasses, mm, mm_utils
from jax_md.util import *
from jax_md import test_util

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS

test_util.update_test_tolerance(f32_tol=1e-3, f64_tol=1e-5) # reduced temporarily from default

NEIGHBOR_LIST_FORMAT = [partition.Dense,
                        partition.Sparse,
                        partition.OrderedSparse]

PDB_FILENAMES = ['alanine-dipeptide-explicit.pdb']
PBCS_BOOLEAN = [False] # omit True until dsf/PME is sorted

if FLAGS.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]

class MMTest(test_util.JAXMDTestCase):
    # TODO : make a class to handle omm loading utilities
    # TODO : place each omm.Force into a different ForceGroup to make energy assertions on a Force-specific basis (this functionality is in `perses`)
    # TODO : write a test to check jit recompilation for different `particle` entries in parameters
    # TODO : write a test to check energy change assertions for non `particle` parameter adjustement (nothing should be `partialed`)
    # TODO : check energy error (this is likely a consequence of f32 use)
    # TODO : allow for **parameters in the energy computation
    # TODO : assert that not passing `parameters` to energy fn uses `parameters` that the `mm_energy_fn` was initiated with.

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_pdb_filename={}_pbc={}_dtype={}'.format(pdb_filename, pbcs, dtype.__name__),
          'pdb_filename': pdb_filename,
          'pbcs': pbcs,
          'dtype': dtype
      } for pdb_filename in PDB_FILENAMES for pbcs in PBCS_BOOLEAN for dtype in POSITION_DTYPE))
  def test_mm_vacuum(self, pdb_filename, pbcs, dtype):
    """assert that the vacuum energy of a solvent-stripped `openmm.System` object matches that in `jax_md`"""
    pdb = app.PDBFile('alanine-dipeptide-explicit.pdb')
    ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    model = Modeller(pdb.topology, pdb.positions)
    model.deleteWater()
    mmSystem = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
    context = openmm.Context(mmSystem, openmm.VerletIntegrator(1.*unit.femtoseconds))
    context.setPositions(model.getPositions())
    omm_state = context.getState(getEnergy=True, getPositions=True)
    positions = jnp.array(omm_state.getPositions(asNumpy=True).value_in_unit_system(unit.md_unit_system))
    omm_energy = omm_state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system)

    params = mm_utils.parameters_from_openmm_system(mmSystem)
    displacement_fn, shift_fn = space.free()
    energy_fn, neighbor_list = mm.mm_energy_fn(displacement_fn=displacement_fn,
                                           parameters = params,
                                           space_shape=space.free,
                                           use_neighbor_list=False,
                                           box_size=None,
                                           use_multiplicative_isotropic_cutoff=False,
                                           use_dsf_coulomb=False,
                                           neighbor_kwargs={},
                                           )
    jax_energy = energy_fn(positions, parameters = params)
    self.assertAllClose(jax_energy, omm_energy)
