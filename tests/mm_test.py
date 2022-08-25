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

PDB_FILENAMES = [
  'alanine-dipeptide-explicit.pdb',
  '1yi6-minimize.pdb'
                 ]
PBCS_BOOLEAN = [False] # omit True until dsf/PME is sorted

if FLAGS.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]

try:
  import openmm
  from openmm import app, unit
except ImportError as error:
  print(error)
except Exception as exception:
  print(exception, False)
  print(exception)

# utilities

def render_mm_objs_from_pdb(
  pdb_path: str,
  ff_files: Iterable[str] = ['amber14-all.xml',
                             'amber14/tip3pfb.xml'],
  delete_waters: bool=False,
  create_system_kwargs: Dict={'nonbondedMethod': app.NoCutoff,
                              'constraints': None,
                              'rigidWater': False,
                              'removeCMMotion': False}) -> Tuple[
                              openmm.System, Array, openmm.Topology
                              ]:
  """render an `openmm.System`, positions, and `openmm.Topology`"""
  pdb = app.PDBFile(pdb_path)
  ff = app.ForceField(ff_files)
  model = app.Modeller(pdb.topology, pdb.positions)
  if delete_waters:
      model.deleteWater()
  mmSystem = ff.createSystem(
    **create_system_kwargs
    )
  return mmSystem, model.positions, model.topology


class MMTest(test_util.JAXMDTestCase):

  @parameterized.named_parameters(test_util.cases_from_list(
      {
          'testcase_name': '_pdb_filename={}_pbc={}_dtype={}'.format(pdb_filename, pbcs, dtype.__name__),
          'pdb_filename': pdb_filename,
          'pbcs': pbcs,
          'dtype': dtype
      } for pdb_filename in PDB_FILENAMES for pbcs in PBCS_BOOLEAN \
        for dtype in POSITION_DTYPE))
  def test_mm_vacuum(self, pdb_filename, pbcs, dtype):
    """assert that the vacuum energy of a solvent-stripped
        `openmm.System` object matches that in `jax_md`
    """
    pdb_filepath = (f"data/{pdb_filename}")
    mmSystem, positions, topology = render_mm_objs_from_pdb(
      pdb_path=pdb_filepath,
      delete_waters=True)
    context = openmm.Context(mmSystem,
        openmm.VerletIntegrator(1.*unit.femtoseconds))
    context.setPositions(positions)
    omm_state = context.getState(getEnergy=True, getPositions=True)
    positions = jnp.array(omm_state.getPositions(asNumpy=True).\
        value_in_unit_system(unit.md_unit_system))
    omm_energy = omm_state.getPotentialEnergy().\
        value_in_unit_system(unit.md_unit_system)
    params = mm_utils.parameters_from_openmm_system(mmSystem)
    displacement_fn, shift_fn = space.free()
    energy_fn, neighbor_list_fn = mm.mm_energy_fn(
        displacement_fn=displacement_fn,
        default_mm_parameters = params)

    jax_energy = energy_fn(positions)
    self.assertAllClose(jax_energy, omm_energy)

    @parameterized.named_parameters(test_util.cases_from_list(
        {
            'testcase_name': '_pdb_filename={}_dtype={}'.format(pdb_filename, dtype.__name__),
            'pdb_filename': pdb_filename,
            'dtype': dtype
        } for pdb_filename in PDB_FILENAMES \
          for dtype in POSITION_DTYPE))
    def test_dense_exception_mask(self, pdb_filename, dtype):
      """assert that the jax dense mask renders the same mask as
      the manual list comprehension mask
      """
      mmSystem, positions, topology = render_mm_objs_from_pdb(
        pdb_path=pdb_filepath,
        delete_waters=True) # remove waters for speed
      params = mm_utils.parameters_from_openmm_system(
        mmSystem)
      num_particles = mmSystem.getNumParticles()
      exceptions = params.nonbonded_exception_parameters.particles

      #manual
      max_count, dense_mask = mm.get_dense_exception_mask(
        num_particles,
        0,
        exceptions)

      #jax
      symm_exceptions = mm.symmetrize_and_transpose(
        exceptions,
        )
      jax_dense_mask = mm.symm_exceptions_to_dense(num_particles, exceptions)

      # assert equivalent
      assert mm.assert_equiv_masks(dense_mask, jax_dense_mask)

if __name__ == '__main__':
  absltest.main()
