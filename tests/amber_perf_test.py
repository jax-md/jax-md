import os
import sys
from typing import Any

import jax
from absl.testing import absltest, parameterized

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy.optimize as jsp_opt
import numpy as np

from jax_md import partition, quantity, simulate, space
from jax_md import energy as jax_energy
from jax_md import test_util as jtu
from jax_md.mm_forcefields.amber import constraints as amber_constraints
from jax_md.mm_forcefields.amber import energy as amber_energy
from jax_md.mm_forcefields.io.openmm import (
  convert_openmm_system,
  load_amber_system,
  load_charmm_system,
  virtual_site_apply_positions,
  virtual_site_fix_state,
)
from jax_md.mm_forcefields.nonbonded import electrostatics

jax.config.parse_flags_with_absl()

try:
  import openmm
  import openmm.app as app
  import openmm.unit as unit
except ImportError:
  openmm = None
  app = None
  unit = None

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'amber_data')

# Constants
_KB_KCAL_MOL_K = (
  unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
).value_in_unit(unit.kilocalorie_per_mole / unit.kelvin)
_AKMA_TO_FS = (
  1.0 * (unit.dalton * unit.angstrom**2 / unit.kilocalorie_per_mole) ** 0.5
).value_in_unit(unit.femtosecond)
_FS_TO_AKMA = 1.0 / _AKMA_TO_FS

def _make_coulomb_handler(
  nb_method: Any,
  r_cut: float,
  recip_alpha: Any,
  recip_grid: Any,
) -> electrostatics.CoulombHandler:
  """Pick a coulomb handler consistent with how we built the OpenMM system."""
  if nb_method == app.PME:
    return electrostatics.PMECoulomb(
      r_cut=r_cut, alpha=recip_alpha, grid_size=recip_grid
    )
  if nb_method == app.Ewald:
    return electrostatics.EwaldCoulomb(r_cut=r_cut)
  return electrostatics.CutoffCoulomb(r_cut=r_cut)

class AMBERPerfTest(jtu.JAXMDTestCase, parameterized.TestCase):
  @parameterized.product(
    system_size=[13, 14, 15, 16, 17, 18, 19],
    mode=['omm', 'jax'],
    precision=['double'],
    dense_format=[False],
  )
  def test_amber_scaling(self, system_size, mode, precision, dense_format):
    import time

    dt_base = 1.0
    dt = dt_base * _FS_TO_AKMA
    n_steps = 10000
    n_steps_inner = 1000
    iter_count = n_steps // n_steps_inner
    target_temp = 300.0
    kT = target_temp * _KB_KCAL_MOL_K
    r_cut = 8.0

    n_water = 2 ** system_size
    n_atoms = n_water * 3
    forcefield = app.ForceField('tip3p.xml')
    omm_modeller = app.Modeller(app.Topology(), [])
    omm_modeller.addSolvent(forcefield, model='tip3p', numAdded=n_water)
    omm_system = forcefield.createSystem(omm_modeller.topology,
                                     nonbondedMethod=app.PME,
                                     constraints=None,
                                     rigidWater=False,
                                     removeCMMotion=False,
                                     nonbondedCutoff=r_cut*unit.angstrom)
    integrator = openmm.VerletIntegrator(dt_base * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName('OpenCL')
    properties = {'Precision': precision}
    simulation = app.Simulation(omm_modeller.topology, omm_system, integrator, platform, properties)
    simulation.context.setPositions(omm_modeller.getPositions())
    simulation.minimizeEnergy()

    omm_state = simulation.context.getState(positions=True, enforcePeriodicBox=True)
    omm_topology = omm_modeller.topology
    omm_positions = omm_state.getPositions(asNumpy=True)
    omm_box_vectors = omm_state.getPeriodicBoxVectors(asNumpy=True)
    phi = n_atoms/omm_state.getPeriodicBoxVolume().value_in_unit(unit.angstrom**3)

    if mode == 'jax':
      print("JAX simulation")
      nb_method = app.PME
      mm = convert_openmm_system(
        omm_system,
        omm_topology,
        omm_positions,
        omm_box_vectors,
        r_cut=r_cut,
        dr_threshold=2.0,
        format=partition.NeighborListFormat.OrderedSparse,
      )

      coulomb = _make_coulomb_handler(
        nb_method, mm.nb_options.r_cut, mm.recip_alpha, mm.recip_grid
      )

      energy_fn, neighbor_fn, disp_fn, shift_fn = amber_energy(
        mm.params,
        mm.topology,
        mm.box_vectors,
        coulomb_options=coulomb,
        nb_options=mm.nb_options,
        # recip_alpha=mm.recip_alpha,
        dense_mask_format=dense_format,
      )

      nbrs = neighbor_fn.allocate(mm.positions)

      def scalar_energy_fn(pos, nbr_list, **kwargs):
        return energy_fn(pos, nbr_list, **kwargs)['etotal']

      energy_fn_jit = jax.jit(scalar_energy_fn)

      init_fn, apply_fn = simulate.nve(energy_fn_jit, shift_fn, dt)
      state = init_fn(
        jax.random.PRNGKey(0), mm.positions, mass=mm.masses, kT=kT, nbr_list=nbrs
      )

      def body_fn(i, state):
        state, nbrs = state
        nbrs = nbrs.update(state.position)
        state = apply_fn(state, nbr_list=nbrs)
        return state, nbrs

      # burn in
      new_state, _ = jax.lax.fori_loop(0, n_steps_inner, body_fn, (state, nbrs))
      new_state.position.block_until_ready()

      step = 0
      print("step , PE , KE , TotalE - in kJ/mol, kT")
      pE = energy_fn_jit(state.position, nbr_list=nbrs, params=mm.params)
      kE = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
      temp = (
        quantity.temperature(momentum=state.momentum, mass=state.mass)
        / _KB_KCAL_MOL_K
      )
      jax.debug.print(
        "{step}, {pE}, {kE}, {pEkE}, {temp}",
        step=step,
        pE=pE * 4.184,
        kE=kE * 4.184,
        pEkE=(pE + kE) * 4.184,
        temp=temp,
      )

      t0 = time.perf_counter()

      for i in range(iter_count):
        new_state, nbrs = jax.lax.fori_loop(0, n_steps_inner, body_fn, (state, nbrs))
        if jnp.any(nbrs.did_buffer_overflow):
          print('Neighbor list overflowed, reallocating.')
          nbrs = neighbor_fn.allocate(state.position)
        else:
          state = new_state
          step += n_steps_inner
        pE = energy_fn_jit(state.position, nbr_list=nbrs, params=mm.params)
        kE = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
        temp = (
          quantity.temperature(momentum=state.momentum, mass=state.mass)
          / _KB_KCAL_MOL_K
        )
        jax.debug.print(
          "{step}, {pE}, {kE}, {pEkE}, {temp}",
          step=step,
          pE=pE * 4.184,
          kE=kE * 4.184,
          pEkE=(pE + kE) * 4.184,
          temp=temp,
        )

      new_state.position.block_until_ready()

      t1 = time.perf_counter()
      t_diff = t1 - t0
      sps = n_steps / t_diff
    else:
      simulation.context.setVelocitiesToTemperature(target_temp * unit.kelvin)
      simulation.reporters.append(
        app.StateDataReporter(
          sys.stdout,
          n_steps_inner,
          step=True,
          potentialEnergy=True,
          kineticEnergy=True,
          totalEnergy=True,
          temperature=True,
        )
      )

      omm_state = simulation.context.getState(energy=True, forces=True)
      omm_pe = omm_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
      )
      omm_ke = omm_state.getKineticEnergy().value_in_unit(
        unit.kilojoules_per_mole
      )
      print("step 0", omm_pe, omm_ke)

      t0_omm = time.perf_counter()
      simulation.step(n_steps)
      t1_omm = time.perf_counter()
      sps = n_steps / (t1_omm - t0_omm)

    print(
      f'[{mode}]  '
      f'n_atoms={n_atoms}  '
      f'phi={phi}  '
      f'n_steps={n_steps}  '
      f'ms/step={1000.0 / sps:.3f}  '
    )

    return

if __name__ == '__main__':
  absltest.main()
