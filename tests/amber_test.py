"""
Tests for AMBER forcefield interoperability with OpenMM

Running
  - Quiet (default): python jax-md/tests/amber_test.py
  - Verbose: VERBOSE_TESTS=1 python jax-md/tests/amber_test.py

High-level goals
  - Validate openmm.py conversion -> JAX params/topology/nb settings
  - Validate energy + force parity against OpenMM w/ Reference platform
  - Validate stability / API for PME, NPT, constraints/virtual sites

General TODO
  - More explicit handling of single vs double precision tests
  - Expand coverage of OpenMM nonbonded methods:
      - NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME
  - Standardize tolerances:
      - Use same per-observable tolerances (energy, force, pressure, volume)
      - Prefer atol for magnitude-invariant quantities (positions, constraint
        error) and rtol for extensive quantities (energies)
  - Add a shared simulation driver helper
      - Neighbor list update + overflow handling
      - Correct ordering for virtual sites / constraints (when enabled)
      - Optional logging (E/T/P/volume + max/RMS force stats)
  - Add a smoke/slow toggle:
      - Short CI vs longer locally-run trajectory checks
  - Consider multi-platform comparison (OpenMM Reference vs OpenCL/CUDA double)
    for CPU vs GPU / single vs double precision
  - Derive constants from OpenMM/CODATA where possible (kB, time units, pressure)
  - Future: alchemical lambda / softcore terms

Per-test TODO
  - test_energy_force:
      - Compare per-term energy breakdowns (force-group parity) where possible
      - Use a more stable force-difference metric, e.g. OMM platform to platform
        2*||dF||/(||F1||+||F2||+eps) plus max abs
      - Include GB tests from Amber20 suite once implemented.
  - test_virtual_sites:
      - Verify force redistribution onto parent atoms via chain rule
      - Verify thermodynamic DOF handling in temperature/pressure utilities
      - Create wrapper that applies update order consistently
        constraints -> virtual sites -> neighbor list -> forces
  - test_nve (gold-standard variant):
      - Start from minimized/equilibrated restarts
      - Compare drift (slope + RMS) JAX vs OpenMM using identical initial vels
      - Include constrained solvated systems
  - test_nvt:
      - Control RNG where possible (OpenMM randomSeed) and compare summary
        statistics (mean/RMS temperature) after burn-in
  - test_npt:
      - More comprehensive single point comparison of thermodynamic quantities
      - Consider tracking volume mean/RMS on an equilibrated snapshot
  - test_geometry_optimization:
      - Keep cross-evaluation checks (OpenMM energy at JAX-minimized coords and
        vice versa) as a robust parity signal
  - test_ions:
      - Add periodic + PME variant (small neutral lattice) for reciprocal stress
      - Optional: separate LJ-only and Coulomb-only reference checks
  - test_constraints:
      - Add velocity constraint parity checks (applyVelocityConstraints)
      - Add short constrained dynamics test once the integrator loop
        applies constraints at the correct substeps

"""

import os
import sys
import time
from types import SimpleNamespace
from functools import partial
from typing import Optional
from absl.testing import absltest, parameterized

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.optimize as jsp_opt
import numpy as np

from jax_md import partition, util, simulate, quantity, energy
from jax_md import space
from jax_md import minimize
from jax_md.mm_forcefields.oplsaa import energy as opls_energy
from jax_md.mm_forcefields.amber import energy as amber_energy
from jax_md.mm_forcefields.amber import constraints as amber_constraints
from jax_md.mm_forcefields.io.openmm import load_amber_system, load_charmm_system, convert_openmm_system, virtual_site_apply_positions, virtual_site_fix_state
from jax_md.mm_forcefields.nonbonded import electrostatics
from jax_md import test_util as jtu

jax.config.parse_flags_with_absl()

try:
  import openmm
  import openmm.app as app
  import openmm.unit as unit
except ImportError:
  openmm = None
  app = None
  unit = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "amber_data")
# Keep per-test summaries behind an env var so CI logs stay readable
_VERBOSE_TESTS = os.environ.get("VERBOSE_TESTS", "").strip().lower() not in ("", "0", "false", "no")


def _vprint(*args, **kwargs):
  if _VERBOSE_TESTS:
    print(*args, **kwargs)


# Constants
_KB_KCAL_MOL_K = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
  unit.kilocalorie_per_mole / unit.kelvin
)
_AKMA_TO_FS = (1.0 * (unit.dalton * unit.angstrom**2 / unit.kilocalorie_per_mole) ** 0.5).value_in_unit(
  unit.femtosecond
)
_FS_TO_AKMA = 1.0 / _AKMA_TO_FS
# atmospheres -> (kcal/mol)/A^3:
#   1 atm = 101325 Pa = 101325 J/m^3
#   1 m^3 = 1e30 A^3
#   1 (kcal/mol) = 4184 J / N_A  => 1 J = (N_A/4184) (kcal/mol)
# So:
#   1 atm = 101325 * (N_A/4184) / 1e30  (kcal/mol)/A^3
_ATM_TO_KCAL_MOL_ANG3 = 101325.0 * 6.02214076e23 / 4184.0 / 1.0e30


# TODO determine procedure to generate restarts for nve, nvt, npt
# constrained? 300k final temp? min -> heat?
def _min_and_heat():
  return


def _openmm_energy_force(system, positions, platform_name: str = "Reference"):
  """Compute OpenMM potential energy, broken down by Force.

  NOTE .setIncludeDirectSpace and .setReciprocalSpaceForceGroup can be used
  to separate the Ewald terms somewhat. In addition, for inputs converted in
  charmmpsffile.py, UREY_BRADLEY_FORCE_GROUP = 3 by default.

  Returns:
      energy_terms: dict[str, float] mapping force keys -> kcal/mol, plus a
        total key for the full potential energy.
      forces: numpy array of forces in kcal/(mol*A).
  """
  energy_terms = {}

  for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)

  integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
  platform = openmm.Platform.getPlatformByName(platform_name)

  context = openmm.Context(system, integrator, platform)
  context.setPositions(positions)

  for i, f in enumerate(system.getForces()):
    state = context.getState(getEnergy=True, groups=(1 << i))
    key = f.getName()
    nrg = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    energy_terms[key] = np.float64(nrg)

  state = context.getState(getEnergy=True, getForces=True)
  energy_terms['etotal'] = np.float64(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
  forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole/unit.angstrom)

  return energy_terms, np.array(forces)


def _make_coulomb_handler(nb_method, r_cut, recip_alpha, recip_grid):
  """Pick a coulomb handler consistent with how we built the OpenMM system."""
  if nb_method == app.PME:
    return electrostatics.PMECoulomb(r_cut=r_cut, alpha=recip_alpha, grid_size=recip_grid)
  if nb_method == app.Ewald:
    return electrostatics.EwaldCoulomb(r_cut=r_cut)
  return electrostatics.CutoffCoulomb(r_cut=r_cut)


# copied from autogenerated CHARMM-GUI script omm_readparams.py
def _charmm_read_params(filename):
  base_path = os.path.dirname(filename)
  parFiles = ()
  with open(filename, 'r') as f:
    for line in f:
      if '!' in line: line = line.split('!')[0]
      parfile = line.strip()
      if len(parfile) != 0: parFiles += ( os.path.join(base_path, parfile), )

  return parFiles


class AMBEREnergyTest(jtu.JAXMDTestCase, parameterized.TestCase):
  @parameterized.product(
    system_info=[
      # RAMP1 system generated with a variety of solvent types
      ("RAMP1_AMBER/RAMP1_gas.prmtop", "RAMP1_AMBER/RAMP1_gas.inpcrd", app.NoCutoff),
      ("RAMP1_AMBER/RAMP1_opc_box.prmtop", "RAMP1_AMBER/RAMP1_opc_box.inpcrd", app.PME),
      ("RAMP1_AMBER/RAMP1_opc_oct.prmtop", "RAMP1_AMBER/RAMP1_opc_oct.inpcrd", app.PME),
      ("RAMP1_AMBER/RAMP1_spce_box.prmtop", "RAMP1_AMBER/RAMP1_spce_box.inpcrd", app.PME),
      ("RAMP1_AMBER/RAMP1_tip3p_box.prmtop", "RAMP1_AMBER/RAMP1_tip3p_box.inpcrd", app.PME),
      #("RAMP1_AMBER/RAMP1_tip3p_box.prmtop", "RAMP1_AMBER/RAMP1_tip3p_box.inpcrd", app.CutoffPeriodic),
      #("RAMP1_AMBER/RAMP1_tip3p_box.prmtop", "RAMP1_AMBER/RAMP1_tip3p_box.inpcrd", app.Ewald),
      ("RAMP1_AMBER/RAMP1_tip3p_oct.prmtop", "RAMP1_AMBER/RAMP1_tip3p_oct.inpcrd", app.PME),
      ("RAMP1_AMBER/RAMP1_tip4pew_box.prmtop", "RAMP1_AMBER/RAMP1_tip4pew_box.inpcrd", app.PME),

      # GAFF molecule in vacuum
      ("sustiva.prmtop", "sustiva.rst7", app.NoCutoff),

      # AMBER benchmark systems from Amber20_Benchmark_Suite
      # https://ambermd.org/GPUPerformance.php
      ("Amber20_Benchmark_Suite/PME/Topologies/JAC.prmtop", "Amber20_Benchmark_Suite/PME/Coordinates/JAC.inpcrd", app.PME),
      #("Amber20_Benchmark_Suite/PME/Topologies/FactorIX.prmtop", "Amber20_Benchmark_Suite/PME/Coordinates/FactorIX.inpcrd", app.PME), # incompatible with omm due to negative periodicity
      #("Amber20_Benchmark_Suite/PME/Topologies/Cellulose.prmtop", "Amber20_Benchmark_Suite/PME/Coordinates/Cellulose.inpcrd", app.PME), # OOM issues
      #("Amber20_Benchmark_Suite/PME/Topologies/STMV.prmtop", "Amber20_Benchmark_Suite/PME/Coordinates/STMV.inpcrd", app.PME), # OOM issues

      # RAMP1 in solvent - CHARMM system generated with CHARMM-GUI
      ("RAMP1_CHARMM/openmm/step3_input.crd", "RAMP1_CHARMM/openmm/step3_input.psf", "RAMP1_CHARMM/openmm/toppar.str", "RAMP1_CHARMM/openmm/sysinfo.dat", app.PME, False, False),
      ("RAMP1_CHARMM/openmm/step3_input.crd", "RAMP1_CHARMM/openmm/step3_input.psf", "RAMP1_CHARMM/openmm/toppar.str", "RAMP1_CHARMM/openmm/sysinfo.dat", app.PME, True, True),
    ]
  )
  def test_energy_force(self, system_info):
    #neighbor_format = partition.NeighborListFormat.OrderedSparse

    if len(system_info) == 3:
      prmtop_name, inpcrd_name, nb_method = system_info
      prmtop_file = os.path.join(DATA_DIR, prmtop_name)
      inpcrd_file = os.path.join(DATA_DIR, inpcrd_name)
      omm_system, omm_topology, omm_positions, omm_box_vectors = load_amber_system(prmtop_file, inpcrd_file, nb_method)
      use_switch = False
      use_lrc = False
    elif len(system_info) == 7:
      crd_name, psf_name, param_name, sys_info, nb_method, use_switch, use_lrc = system_info
      crd_file = os.path.join(DATA_DIR, crd_name)
      psf_file = os.path.join(DATA_DIR, psf_name)
      param_file = os.path.join(DATA_DIR, param_name)
      param_files = _charmm_read_params(param_file)
      if sys_info is not None:
        sys_info_file = os.path.join(DATA_DIR, sys_info)
      omm_system, omm_topology, omm_positions, omm_box_vectors = load_charmm_system(crd_file, psf_file, param_files, nb_method, sys_info_file)

    # If switching is selected for a CHARMM system, include it in the NonbondedForce
    if use_switch and nb_method is not app.NoCutoff:
      switch_distance = .5 * unit.nanometer
      for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce) or isinstance(force, openmm.CustomNonbondedForce):
          r_cut = force.getCutoffDistance()
          if switch_distance.value_in_unit(unit.nanometer) >= r_cut.value_in_unit(unit.nanometer):
            raise ValueError('switchDistance is too large compared to the cutoff!')
          force.setUseSwitchingFunction(True)
          force.setSwitchingDistance(switch_distance)

    # NOTE further testing is needed to determine when dispersion correction
    # should be turned on for cases where nbfix terms are present
    # if use_lrc and nb_method is not app.NoCutoff:
    #   for force in omm_system.getForces():
    #     if isinstance(force, openmm.NonbondedForce):
    #       force.setUseDispersionCorrection(True)
    #     if isinstance(force, openmm.CustomNonbondedForce) and force.getNumTabulatedFunctions() != 1:
    #       force.setUseLongRangeCorrection(True)

    # Convert system to JAX structures
    mm = convert_openmm_system(omm_system, omm_topology, omm_positions, omm_box_vectors, format=partition.NeighborListFormat.OrderedSparse)

    coulomb = _make_coulomb_handler(nb_method, mm.nb_options.r_cut, mm.recip_alpha, mm.recip_grid)

    energy_fn, neighbor_fn, _, _ = amber_energy(
      mm.params,
      mm.topology,
      mm.box_vectors,
      coulomb_options=coulomb,
      nb_options=mm.nb_options,
    )

    nlist = neighbor_fn.allocate(mm.positions)
    energy_dict = energy_fn(mm.positions, nlist)
    ref_terms, omm_frcs = _openmm_energy_force(omm_system, omm_positions)

    # print("\njax energy dict", energy_dict)
    # print("\nomm energy dict", ref_terms)
    # print("\njax energy", energy_dict["etotal"], energy_dict["etotal"]*4.184)
    # print("\nomm energy", ref_terms["etotal"], ref_terms["etotal"]*4.184)

    self.assertAllClose(energy_dict["etotal"], ref_terms["etotal"], rtol=1e-4, atol=0.0)

    def total_energy_fn(pos, nlist):
      E = energy_fn(pos, nlist)
      return E['etotal']
    grad_fn = jax.grad(total_energy_fn)
    gradients = -grad_fn(mm.positions, nlist)

    self.assertEqual(gradients.shape, mm.positions.shape)
    self.assertTrue(jnp.all(jnp.isfinite(gradients)))
    #self.assertAllClose(gradients, omm_frcs, rtol=1e-4, atol=0.0)

    # Optional per-case summary to help spot regressions at a glance.
    label = str(system_info[0])
    if len(system_info) == 3:
      label = f"AMBER {prmtop_name} ({nb_method})"
    elif len(system_info) == 7:
      label = f"CHARMM {crd_name} switch={use_switch} lrc={use_lrc} ({nb_method})"

    e_jax = float(jax.device_get(energy_dict["etotal"]))
    e_omm = float(ref_terms["etotal"])
    dE = e_jax - e_omm

    frc = jnp.asarray(omm_frcs, dtype=gradients.dtype)
    df = gradients - frc
    max_abs = float(jnp.max(jnp.abs(df)))
    rms_abs = float(jnp.sqrt(jnp.mean(df**2)))
    rms_rel = float(rms_abs / (jnp.sqrt(jnp.mean(frc**2)) + 1e-12))

    _vprint(
      "[E/F] "
      f"{label}  "
      f"E_jax={e_jax:.6f} E_omm={e_omm:.6f} dE={dE:.3e} kcal/mol  "
      f"F_err: max_abs={max_abs:.3e} rms_abs={rms_abs:.3e} rms_rel={rms_rel:.3e}"
    )


  """
  Notes:
  - In OpenMM, forces are explicitly redistributed onto parent atoms.
    This doesn't take place until computeVirtualSites is called, at
    which point the forces on the virtual sites will still remain.
    Reference: https://github.com/openmm/openmm/issues/1106
  - Virtual sites don't correspond to a degree of freedom in the
    system. Therefore, thermodynamic quantities will be wrong unless
    this is specifically accounted for.
  - Projecting virtual site positions can invalidate the neighbor list,
    but also must be done in any function called in jax.grad to ensure
    correct forces under AD.
  """
  @parameterized.product(
    system_info=[
      ("RAMP1_AMBER/RAMP1_opc_box.prmtop", "RAMP1_AMBER/RAMP1_opc_box.inpcrd", app.PME),
      ("RAMP1_AMBER/RAMP1_opc_oct.prmtop", "RAMP1_AMBER/RAMP1_opc_oct.inpcrd", app.PME),
    ],
  )
  def test_virtual_sites(self, system_info):
    dt = 1.0 * _FS_TO_AKMA
    n_steps = 10
    n_steps_inner = 1
    iter_count = n_steps // n_steps_inner
    target_temp = 300.0
    kT = target_temp * _KB_KCAL_MOL_K

    prmtop_name, inpcrd_name, nb_method = system_info

    prmtop_file = os.path.join(DATA_DIR, prmtop_name)
    inpcrd_file = os.path.join(DATA_DIR, inpcrd_name)

    load_kwargs = {'constraints':None, 'removeCMMotion':False, 'rigidWater':False}

    omm_system, omm_topology, omm_positions, omm_box_vectors = load_amber_system(
      prmtop_file, inpcrd_file, nb_method, **load_kwargs
    )

    if nb_method == app.NoCutoff:
      r_cut = None
    else:
      r_cut = 8.0

    for force in omm_system.getForces():
      if isinstance(force, openmm.NonbondedForce):
        force.setCutoffDistance(r_cut * unit.angstrom)

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
    )

    nbrs = neighbor_fn.allocate(mm.positions)

    def scalar_energy_fn(pos, nbr_list, **kwargs):
      # Ensures that virtual site forces are distributed to parent sites
      pos = virtual_site_apply_positions(
        pos,
        mm.virtual_sites,
        displacement_fn=disp_fn,
        shift_fn=shift_fn,
        box=mm.box_vectors,
        use_periodic_general=mm.nb_options.use_periodic_general,
      )
      return energy_fn(pos, nbr_list, **kwargs)['etotal']

    energy_fn_jit = jax.jit(scalar_energy_fn)
    grad_fn = jax.grad(scalar_energy_fn)

    init_fn, apply_fn = simulate.nve(energy_fn_jit, shift_fn, dt)
    state = init_fn(jax.random.PRNGKey(0), mm.positions, mass=mm.masses, kT=kT, nbr_list=nbrs)
    state = virtual_site_fix_state(
      state,
      mm.virtual_sites,
      displacement_fn=disp_fn,
      shift_fn=shift_fn,
      box=mm.box_vectors,
      use_periodic_general=mm.nb_options.use_periodic_general,
    )
    nbrs = nbrs.update(state.position)
    vs_pos_jax = state.position

    # neighbor list is stale after virtual site update
    first_step_jax = energy_fn_jit(vs_pos_jax, nbr_list=nbrs, params=mm.params)*4.184
    first_step_jax_grads = -grad_fn(vs_pos_jax, nbr_list=nbrs, params=mm.params)

    # NOTE example of how dynamics loop has to work to avoid discontinuities
    # this isn't particularly clean or ergonomic

    # def body_fn(i, state):
    #   state, nbrs = state
    #   nbrs = nbrs.update(state.position)
    #   state = apply_fn(state, nbr_list=nbrs)
    #   state = virtual_site_fix_state(
    #     state,
    #     mm.virtual_sites,
    #     displacement_fn=disp_fn,
    #     shift_fn=shift_fn,
    #     box=mm.box_vectors,
    #     use_periodic_general=mm.nb_options.use_periodic_general,
    #   )
    #   return state, nbrs

    # step = 0
    # print("step , PE , KE , TotalE - in kJ/mol, kT")
    # pE = energy_fn_jit(state.position, nbr_list=nbrs, params=mm.params)
    # kE = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
    # temp = quantity.temperature(momentum=state.momentum, mass=state.mass)/_KB_KCAL_MOL_K
    # jax.debug.print("{step}, {pE}, {kE}, {pEkE}, {temp}", step=step, pE=pE*4.184, kE=kE*4.184, pEkE=(pE+kE)*4.184, temp=temp)
    
    # for i in range(iter_count):
    #   new_state, nbrs = jax.lax.fori_loop(0, n_steps_inner, body_fn, (state, nbrs))
    #   if jnp.any(nbrs.did_buffer_overflow):
    #     print('Neighbor list overflowed, reallocating.')
    #     nbrs = neighbor_fn.allocate(state.position)
    #   else:
    #     state = new_state
    #     step += n_steps_inner
    #   pE = energy_fn_jit(state.position, nbr_list=nbrs, params=mm.params)
    #   kE = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
    #   temp = quantity.temperature(momentum=state.momentum, mass=state.mass)/_KB_KCAL_MOL_K
    #   jax.debug.print("{step}, {pE}, {kE}, {pEkE}, {temp}", step=step, pE=pE*4.184, kE=kE*4.184, pEkE=(pE+kE)*4.184, temp=temp)

    platform = openmm.Platform.getPlatformByName('Reference')
    properties = {}
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    sim = app.Simulation(omm_topology, omm_system, integrator, platform, properties)
    sim.context.setPositions(omm_positions)
    
    # NOTE Virtual site positions aren't automatically updated
    # this is required to get good 0th step agreement
    sim.context.computeVirtualSites()
    vs_pos_omm = sim.context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    # If fractional coordinates are used, convert OMM positions
    if mm.nb_options.fractional_coordinates:
      vs_pos_omm = space.transform(space.inverse(mm.box_vectors), vs_pos_omm)
      vs_pos_omm = jnp.mod(vs_pos_omm, 1.0)  # wrap into [0,1)

    self.assertAllClose(vs_pos_jax, vs_pos_omm, rtol=1e-10, atol=0.0)

    # NOTE will not work as expected due to extra DoF with virtual sites
    sim.context.setVelocitiesToTemperature(target_temp * unit.kelvin)
    sim.reporters.append(app.StateDataReporter(sys.stdout, n_steps_inner, 
                                              step=True,
                                              potentialEnergy=True,
                                              kineticEnergy=True,
                                              totalEnergy=True,
                                              temperature=True,
                                              ))

    omm_state = sim.context.getState(getEnergy=True, getForces=True)
    first_step_omm = omm_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    first_step_omm_grads = omm_state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole/unit.angstrom)

    self.assertAllClose(first_step_jax, first_step_omm, rtol=1e-4, atol=0.0)
    # NOTE basic force comparisons need to get worked out first, but keep in
    # mind that OpenMM doesn't zero out virtual site forces
    # self.assertAllClose(first_step_jax_grads * ~mm.virtual_sites.is_virtual_site.reshape(-1, 1), first_step_omm_grads * ~mm.virtual_sites.is_virtual_site.reshape(-1, 1), rtol=1e-5, atol=0.0)

    # Summary diagnostics (virtual site systems are especially sensitive).
    pos_delta = vs_pos_jax - jnp.asarray(vs_pos_omm, dtype=mm.positions.dtype)
    max_abs_pos = float(jnp.max(jnp.abs(pos_delta)))
    rms_abs_pos = float(jnp.sqrt(jnp.mean(pos_delta**2)))

    frc = jnp.asarray(first_step_omm_grads, dtype=first_step_jax_grads.dtype)
    df = first_step_jax_grads - frc
    max_abs_f = float(jnp.max(jnp.abs(df)))
    rms_abs_f = float(jnp.sqrt(jnp.mean(df**2)))
    rms_rel_f = float(rms_abs_f / (jnp.sqrt(jnp.mean(frc**2)) + 1e-12))

    _vprint(
      "[vsites] "
      f"{prmtop_name}  "
      f"pos_err: max_abs={max_abs_pos:.3e} rms_abs={rms_abs_pos:.3e}  "
      f"E0_jax={first_step_jax:.6f} E0_omm={first_step_omm:.6f} kJ/mol  "
      f"F0_err: max_abs={max_abs_f:.3e} rms_abs={rms_abs_f:.3e} rms_rel={rms_rel_f:.3e}"
    )


  def test_nve(self):
    """NVE stability / energy drift check for a small vacuum system."""
    # if openmm is None:
    #   self.skipTest('OpenMM is not installed.')

    dt = 1.0 # fs
    dt_jax = dt * _FS_TO_AKMA
    dt_omm = dt * unit.femtosecond
    n_steps = 10000
    target_temp = 300.0
    kT = target_temp * _KB_KCAL_MOL_K

    prmtop_file = os.path.join(DATA_DIR, "sustiva.prmtop")
    inpcrd_file = os.path.join(DATA_DIR, "sustiva.rst7")

    # NOTE unconstrained reference system to avoid deleting bond force for O-H
    load_kwargs = {'constraints': None, 'removeCMMotion': False, 'rigidWater': False}
    omm_system, omm_topology, omm_positions, omm_box_vectors = load_amber_system(
      prmtop_file, inpcrd_file, app.NoCutoff, **load_kwargs
    )

    mm = convert_openmm_system(
      omm_system,
      omm_topology,
      omm_positions,
      omm_box_vectors,
      format=partition.NeighborListFormat.OrderedSparse,
    )

    coulomb = _make_coulomb_handler(app.NoCutoff, mm.nb_options.r_cut, mm.recip_alpha, mm.recip_grid)
    energy_fn, neighbor_fn, _, shift_fn = amber_energy(
      mm.params, mm.topology, mm.box_vectors, coulomb_options=coulomb, nb_options=mm.nb_options
    )

    nlist = neighbor_fn.allocate(mm.positions)

    def scalar_energy_fn(pos, nbr_list, **kwargs):
      return energy_fn(pos, nbr_list, **kwargs)['etotal']

    energy_fn_jit = jax.jit(scalar_energy_fn)
    init_fn, apply_fn = simulate.nve(energy_fn_jit, shift_fn, dt_jax)
    state = init_fn(
      jax.random.PRNGKey(0),
      mm.positions,
      mass=mm.masses,
      kT=kT,
      nbr_list=nlist,
      params=mm.params,
    )

    def step(carry, _):
      state, nlist = carry
      nlist = nlist.update(state.position)
      state = apply_fn(state, nbr_list=nlist, params=mm.params)
      pe = energy_fn_jit(state.position, nbr_list=nlist, params=mm.params)
      ke = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
      etot = pe + ke
      return (state, nlist), etot

    (_, _), etot_series = jax.lax.scan(step, (state, nlist), xs=None, length=n_steps)
    etot_series = np.asarray(jax.device_get(etot_series))
    self.assertTrue(np.all(np.isfinite(etot_series)))

    jax_delta = etot_series - etot_series[0]
    jax_drift_abs = float(np.max(np.abs(jax_delta)))
    jax_drift_rms = float(np.sqrt(np.mean(jax_delta**2)))
    self.assertLess(jax_drift_abs, 5.0)

    integrator = openmm.VerletIntegrator(dt_omm)
    platform = openmm.Platform.getPlatformByName("Reference")
    sim = app.Simulation(omm_topology, omm_system, integrator, platform)
    sim.context.setPositions(omm_positions)
    sim.context.setVelocitiesToTemperature(target_temp * unit.kelvin, 0)

    omm_etot = np.empty((n_steps + 1,), dtype=np.float64)
    state0 = sim.context.getState(getEnergy=True)
    omm_etot[0] = (
      state0.getPotentialEnergy() + state0.getKineticEnergy()
    ).value_in_unit(unit.kilocalories_per_mole)
    for i in range(1, n_steps + 1):
      sim.step(1)
      st = sim.context.getState(getEnergy=True)
      omm_etot[i] = (
        st.getPotentialEnergy() + st.getKineticEnergy()
      ).value_in_unit(unit.kilocalories_per_mole)

    self.assertTrue(np.all(np.isfinite(omm_etot)))
    omm_delta = omm_etot - omm_etot[0]
    omm_drift_abs = float(np.max(np.abs(omm_delta)))
    omm_drift_rms = float(np.sqrt(np.mean(omm_delta**2)))
    self.assertLess(omm_drift_abs, 5.0)

    _vprint(
      "[NVE drift] "
      f"JAX abs={jax_drift_abs:.6f} rms={jax_drift_rms:.6f}  "
      f"OpenMM abs={omm_drift_abs:.6f} rms={omm_drift_rms:.6f}  "
    )

  def test_nvt(self):
    """NVT thermostat sanity test (JAX only).

    This test checks that a Langevin thermostat produces a reasonable mean
    temperature over a short trajectory.
    """
    if openmm is None:
      self.skipTest('OpenMM is not installed.')

    dt = 1.0 # fs
    dt_jax = dt * _FS_TO_AKMA
    dt_omm = dt * unit.femtosecond
    burn_in = 200 # TODO remove
    n_steps = 10000
    target_temp = 300.0
    kT = target_temp * _KB_KCAL_MOL_K

    prmtop_file = os.path.join(DATA_DIR, "sustiva.prmtop")
    inpcrd_file = os.path.join(DATA_DIR, "sustiva.rst7")

    load_kwargs = {'constraints': None, 'removeCMMotion': False, 'rigidWater': False}
    omm_system, omm_topology, omm_positions, omm_box_vectors = load_amber_system(
      prmtop_file, inpcrd_file, app.NoCutoff, **load_kwargs
    )

    mm = convert_openmm_system(
      omm_system,
      omm_topology,
      omm_positions,
      omm_box_vectors,
      format=partition.NeighborListFormat.OrderedSparse,
    )

    coulomb = _make_coulomb_handler(app.NoCutoff, mm.nb_options.r_cut, mm.recip_alpha, mm.recip_grid)
    energy_fn, neighbor_fn, _, shift_fn = amber_energy(
      mm.params, mm.topology, mm.box_vectors, coulomb_options=coulomb, nb_options=mm.nb_options
    )

    nlist = neighbor_fn.allocate(mm.positions)

    def scalar_energy_fn(pos, nbr_list, **kwargs):
      return energy_fn(pos, nbr_list, **kwargs)['etotal']

    energy_fn_jit = jax.jit(scalar_energy_fn)

    gamma = 0.5
    init_fn, apply_fn = simulate.nvt_langevin(energy_fn_jit, shift_fn, dt_jax, kT=kT, gamma=gamma)
    state = init_fn(
      jax.random.PRNGKey(0),
      mm.positions,
      mass=mm.masses,
      nbr_list=nlist,
      params=mm.params,
    )

    def step(carry, _):
      state, nlist = carry
      nlist = nlist.update(state.position)
      state = apply_fn(state, nbr_list=nlist, params=mm.params)
      temp_kT = quantity.temperature(momentum=state.momentum, mass=state.mass)
      temp_K = temp_kT / _KB_KCAL_MOL_K
      return (state, nlist), temp_K

    (_, _), temps = jax.lax.scan(step, (state, nlist), xs=None, length=n_steps)
    temps = np.asarray(jax.device_get(temps))
    self.assertTrue(np.all(np.isfinite(temps)))

    mean_T = float(np.mean(temps[burn_in:]))
    rms_T = float(np.sqrt(np.mean((temps[burn_in:] - mean_T) ** 2)))
    self.assertGreater(mean_T, 0.0)
    self.assertLess(abs(mean_T - target_temp), 75.0)

    # This is intentionally a loose comparison because the stochastic dynamics
    # will not match step-by-step and OpenMM/JAX use different RNG
    friction = gamma / ((unit.dalton * unit.angstrom**2 / unit.kilocalorie_per_mole) ** 0.5)
    integrator = openmm.LangevinIntegrator(target_temp * unit.kelvin, friction, dt_omm)
    platform = openmm.Platform.getPlatformByName("Reference")
    sim = app.Simulation(omm_topology, omm_system, integrator, platform)
    sim.context.setPositions(omm_positions)
    sim.context.setVelocitiesToTemperature(target_temp * unit.kelvin, 0)

    n_particles = omm_system.getNumParticles()
    dof = 3 * n_particles
    R_kcal = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilocalorie_per_mole / unit.kelvin)

    sample_stride = 10
    n_samples = n_steps // sample_stride
    burn_in_samples = burn_in // sample_stride
    temps_omm = np.empty((n_samples,), dtype=np.float64)

    for i in range(n_samples):
      sim.step(sample_stride)
      st = sim.context.getState(getEnergy=True)
      ke = st.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
      temps_omm[i] = 2.0 * ke / (dof * R_kcal)

    self.assertTrue(np.all(np.isfinite(temps_omm)))
    mean_T_omm = float(np.mean(temps_omm[burn_in_samples:]))
    rms_T_omm = float(np.sqrt(np.mean((temps_omm[burn_in_samples:] - mean_T_omm) ** 2)))
    self.assertGreater(mean_T_omm, 0.0)
    self.assertLess(abs(mean_T_omm - target_temp), 75.0)

    self.assertLess(abs(mean_T - mean_T_omm), 100.0)
    _vprint(
      "[NVT temp] "
      f"JAX mean={mean_T:.3f}K rms={rms_T:.3f}K  "
      f"OpenMM mean={mean_T_omm:.3f}K rms={rms_T_omm:.3f}K  "
      f"(dt={dt:.3f} fs, stride={sample_stride}, friction={friction.value_in_unit(unit.picosecond**-1):.6f} 1/ps)"
    )

  def test_npt(self):
    """
    Notes:
    - it looks like "box" is internally expected in some places, this will probably
      only work with periodic general
    - energy function has to support perturbation and consistently pass box through
      for any references to displacement or shift
    """
    if openmm is None:
      self.skipTest('OpenMM is not installed.')

    # Use a solvated, periodic Amber system with PME.
    prmtop_file = os.path.join(DATA_DIR, "RAMP1_AMBER/RAMP1_tip3p_box.prmtop")
    inpcrd_file = os.path.join(DATA_DIR, "RAMP1_AMBER/RAMP1_tip3p_box.inpcrd")

    load_kwargs = {'constraints': None, 'removeCMMotion': False, 'rigidWater': False}
    omm_system, omm_topology, omm_positions, omm_box_vectors = load_amber_system(
      prmtop_file, inpcrd_file, app.PME, **load_kwargs
    )

    mm = convert_openmm_system(
      omm_system,
      omm_topology,
      omm_positions,
      omm_box_vectors,
      format=partition.NeighborListFormat.OrderedSparse,
    )
    if mm.box_vectors is None:
      raise ValueError("NPT test requires a periodic system with box vectors.")

    # JAX-MD NPT currently expects periodic_general and uses
    # fractional coordinates in the unit cube.
    box0 = mm.box_vectors
    if box0.ndim == 1:
      box0 = jnp.diag(box0)
    pos0_cart = mm.positions
    pos0_frac = space.transform(space.inverse(box0), pos0_cart)
    pos0_frac = jnp.mod(pos0_frac, 1.0)

    nb_options = mm.nb_options._replace(use_periodic_general=True, fractional_coordinates=True)
    mm = mm._replace(positions=pos0_frac, box_vectors=box0, nb_options=nb_options)

    coulomb = _make_coulomb_handler(app.PME, mm.nb_options.r_cut, mm.recip_alpha, mm.recip_grid)
    energy_fn, neighbor_fn, _, shift_fn = amber_energy(
      mm.params, mm.topology, mm.box_vectors, coulomb_options=coulomb, nb_options=mm.nb_options
    )

    nbrs = neighbor_fn.allocate(mm.positions, box=mm.box_vectors)

    def scalar_energy_fn(pos, *, nbr_list, **kwargs):
      return energy_fn(pos, nbr_list, **kwargs)['etotal']

    energy_fn_jit = jax.jit(scalar_energy_fn)

    # the perturbation pathway used by simulate.npt_nose_hoover
    # must influence the energy so dU/dV is nonzero and finite
    def U(eps):
      return energy_fn_jit(
        mm.positions,
        nbr_list=nbrs,
        params=mm.params,
        box=mm.box_vectors,
        perturbation=(1.0 + eps),
      )

    dUdeps = jax.grad(U)(0.0)
    self.assertTrue(jnp.isfinite(dUdeps))
    self.assertNotEqual(float(dUdeps), 0.0)

    dt_fs = 1.0
    dt_jax = dt_fs * _FS_TO_AKMA
    target_temp = 300.0
    kT = target_temp * _KB_KCAL_MOL_K

    pressure = 1.0 * _ATM_TO_KCAL_MOL_ANG3

    init_fn, apply_fn = simulate.npt_nose_hoover(energy_fn_jit, shift_fn, dt_jax, pressure, kT)
    state = init_fn(
      jax.random.PRNGKey(0),
      mm.positions,
      mm.box_vectors,
      mass=mm.masses,
      nbr_list=nbrs,
      params=mm.params,
    )

    n_steps = 50

    def step(carry, _):
      state, nbrs = carry
      box = simulate.npt_box(state)
      nbrs = nbrs.update(state.position, box=box)
      overflow = jnp.any(nbrs.did_buffer_overflow)

      state = apply_fn(state, pressure=pressure, kT=kT, nbr_list=nbrs, params=mm.params)

      box_new = simulate.npt_box(state)
      vol = jnp.linalg.det(box_new)
      temp_kT = quantity.temperature(momentum=state.momentum, mass=state.mass)
      return (state, nbrs), (vol, temp_kT, overflow)

    (_, nbrs_final), (vols, temps_kT, overflows) = jax.lax.scan(
      step, (state, nbrs), xs=None, length=n_steps
    )

    vols = np.asarray(jax.device_get(vols))
    temps = np.asarray(jax.device_get(temps_kT / _KB_KCAL_MOL_K))
    overflows = np.asarray(jax.device_get(overflows))

    # volume and temp must be finite
    # neighbor list must not be stale as well
    self.assertTrue(np.all(np.isfinite(vols)))
    self.assertTrue(np.all(vols > 0.0))
    self.assertTrue(np.all(np.isfinite(temps)))
    self.assertFalse(bool(np.any(overflows)))

    # very loose sanity checks due to small timescale
    vol0 = float(np.linalg.det(np.asarray(jax.device_get(mm.box_vectors))))
    self.assertLess(float(np.max(vols)), 10.0 * vol0)
    self.assertGreater(float(np.min(vols)), 0.1 * vol0)

    _vprint(
      "[NPT sanity] "
      f"dUdeps={float(jax.device_get(dUdeps)):.6e}  "
      f"V0={vol0:.6e}A^3  "
      f"V[min,max]=[{float(np.min(vols)):.6e},{float(np.max(vols)):.6e}]A^3  "
      f"T[mean,rms]=[{float(np.mean(temps)):.3f},{float(np.sqrt(np.mean((temps-np.mean(temps))**2))):.3f}]K  "
      f"overflow={bool(np.any(overflows))}"
    )

  def test_geometry_optimization(self):
    """Geometry optimization comparison vs OpenMM."""

    prmtop_file = os.path.join(DATA_DIR, "sustiva.prmtop")
    inpcrd_file = os.path.join(DATA_DIR, "sustiva.rst7")

    load_kwargs = {'constraints': None, 'removeCMMotion': False, 'rigidWater': False}
    omm_system, omm_topology, omm_positions, omm_box_vectors = load_amber_system(
      prmtop_file, inpcrd_file, app.NoCutoff, **load_kwargs
    )

    mm = convert_openmm_system(
      omm_system,
      omm_topology,
      omm_positions,
      omm_box_vectors,
      format=partition.NeighborListFormat.OrderedSparse,
    )

    coulomb = _make_coulomb_handler(app.NoCutoff, mm.nb_options.r_cut, mm.recip_alpha, mm.recip_grid)
    energy_fn, neighbor_fn, _, shift_fn = amber_energy(
      mm.params, mm.topology, mm.box_vectors, coulomb_options=coulomb, nb_options=mm.nb_options
    )

    tol_omm = 10.0 * unit.kilojoule_per_mole / unit.nanometer
    tol_component = float(
      tol_omm.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    )

    # OpenMM reference minimization.
    sim = app.Simulation(
      omm_topology,
      omm_system,
      openmm.VerletIntegrator(1.0 * unit.femtosecond),
      openmm.Platform.getPlatformByName("Reference"),
    )
    sim.context.setPositions(omm_positions)
    sim.minimizeEnergy(tolerance=tol_omm, maxIterations=200)
    st_omm = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
    pos_omm_min = np.asarray(st_omm.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    e_omm_min = float(st_omm.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

    f_omm = np.asarray(
      st_omm.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    )
    f_omm_rms_component = float(np.sqrt(np.mean(f_omm**2)))
    f_omm_rms_magnitude = float(np.sqrt(np.mean(np.sum(f_omm**2, axis=1))))

    # check that JAX energies are the same 
    pos_omm_min_jnp = jnp.asarray(pos_omm_min, dtype=mm.positions.dtype)

    # start from OpenMM coordinates to avoid falling into another local minimum
    nlist = neighbor_fn.allocate(pos_omm_min_jnp)

    def jax_energy(pos: jnp.ndarray) -> jnp.ndarray:
      return energy_fn(pos, nlist, params=mm.params)["etotal"]

    def obj(x: jnp.ndarray) -> jnp.ndarray:
      return jax_energy(jnp.reshape(x, mm.positions.shape))

    x0 = jnp.reshape(pos_omm_min_jnp, (-1,))
    n_particles = omm_system.getNumParticles()
    gtol = float(np.sqrt(3.0 * n_particles) * tol_component)
    res = jsp_opt.minimize(
      obj,
      x0,
      method="BFGS",
      options={"maxiter": 200, "gtol": gtol, "norm": 2},
    )

    success = bool(np.asarray(jax.device_get(res.success)))
    self.assertTrue(success)

    x_opt = res.x
    pos_jax_min = np.asarray(jax.device_get(jnp.reshape(x_opt, mm.positions.shape)))
    e_jax_min = float(jax.device_get(res.fun))

    # Cross-evaluate: OpenMM energy at the JAX-minimized coordinates.
    sim.context.setPositions(pos_jax_min * unit.angstrom)
    st_at_jax = sim.context.getState(getEnergy=True)
    e_omm_at_jax = float(st_at_jax.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    self.assertAlmostEqual(e_jax_min, e_omm_at_jax, places=2)

    f_jax = np.asarray(jax.device_get(res.jac))
    f_jax_rms_component = float(np.sqrt(np.mean(f_jax**2)))
    self.assertLess(f_jax_rms_component, tol_component * 2.0)

    _vprint(
      "[MIN] "
      f"OpenMM={e_omm_min:.6f} kcal/mol  "
      f"JAX={e_jax_min:.6f} kcal/mol  "
      f"RMS(F_comp): JAX={f_jax_rms_component:.6f} OpenMM={f_omm_rms_component:.6f}  "
      f"RMS(|F_i|): OpenMM={f_omm_rms_magnitude:.6f}  "
      f"tol_comp={tol_component:.6f}"
    )


  def test_ions(self):
    """
    Test for an artificial system that consists of a lattice of ions.

    This ensures that:
      - convert_openmm_system can handle systems with only a NonbondedForce.
      - The JAX energy function returns finite energies with bonded terms zero.
      - Energies and forces are relatively close for an OpenMM system
    """
    nb_method = app.NoCutoff
    lattice_constant_nm = 0.5
    n_rep = 10  # an even value for n_rep is required for charge neutrality

    # mass is exact, other parameters probably aren't physically reasonable
    ion_params = {
      "Na": {"q": +1.0, "mass_da": 22.99, "sigma_nm": 0.25, "eps_kj_mol": 0.10},
      "Cl": {"q": -1.0, "mass_da": 35.45, "sigma_nm": 0.40, "eps_kj_mol": 0.10},
    }

    system = openmm.System()
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    top = app.Topology()
    chain = top.addChain()

    pos_nm = []
    for i in range(n_rep):
      for j in range(n_rep):
        for k in range(n_rep):
          sym = "Na" if ((i + j + k) % 2 == 0) else "Cl"
          p = ion_params[sym]

          system.addParticle(p["mass_da"])
          force.addParticle(
            p["q"] * unit.elementary_charge,
            p["sigma_nm"] * unit.nanometer,
            p["eps_kj_mol"] * unit.kilojoule_per_mole,
          )

          res = top.addResidue(sym, chain)
          element = app.Element.getBySymbol(sym)
          top.addAtom(sym, element, res)

          pos_nm.append(openmm.Vec3(i, j, k) * lattice_constant_nm)

    system.addForce(force)
    positions = unit.Quantity(pos_nm, unit.nanometer) # conversion function expects quantities

    mm = convert_openmm_system(
      system,
      top,
      positions,
      box_vectors=None,
      format=partition.NeighborListFormat.OrderedSparse,
    )

    coulomb = _make_coulomb_handler(nb_method, mm.nb_options.r_cut, mm.recip_alpha, mm.recip_grid)
    energy_fn, neighbor_fn, _, _ = amber_energy(
      mm.params,
      mm.topology,
      mm.box_vectors,
      coulomb_options=coulomb,
      nb_options=mm.nb_options,
    )

    nlist = neighbor_fn.allocate(mm.positions)
    energy_dict = energy_fn(mm.positions, nlist)

    def etotal_fn(pos):
      return energy_fn(pos, nlist)["etotal"]

    jax_forces = -jax.grad(etotal_fn)(mm.positions)

    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    omm_pe = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    omm_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    omm_forces = jnp.asarray(np.asarray(omm_forces), dtype=mm.positions.dtype)

    self.assertTrue(jnp.isfinite(energy_dict["etotal"]))
    self.assertTrue(jnp.all(jnp.isfinite(jax_forces)))
    self.assertAllClose(energy_dict["etotal"], omm_pe, rtol=1e-5, atol=0.0)
    self.assertAllClose(jax_forces, omm_forces, rtol=1e-5, atol=0.0)

    df = jax_forces - omm_forces
    max_abs = float(jnp.max(jnp.abs(df)))
    rms_abs = float(jnp.sqrt(jnp.mean(df**2)))
    rms_rel = float(rms_abs / (jnp.sqrt(jnp.mean(omm_forces**2)) + 1e-12))
    _vprint(
      "[ions] "
      f"n={int(mm.positions.shape[0])} lattice_nm={lattice_constant_nm} rep={n_rep}  "
      f"E_jax={float(jax.device_get(energy_dict['etotal'])):.6f} "
      f"E_omm={float(omm_pe):.6f} kcal/mol  "
      f"F_err: max_abs={max_abs:.3e} rms_abs={rms_abs:.3e} rms_rel={rms_rel:.3e}"
    )

  def test_constraints(self):
    if openmm is None:
      self.skipTest('OpenMM is not installed.')

    # protein + rigid water + HBond constraints.
    prmtop_file = os.path.join(DATA_DIR, "RAMP1_AMBER/RAMP1_tip3p_box.prmtop")
    inpcrd_file = os.path.join(DATA_DIR, "RAMP1_AMBER/RAMP1_tip3p_box.inpcrd")

    load_kwargs = {'constraints': app.HBonds, 'removeCMMotion': False, 'rigidWater': True}
    omm_system, omm_topology, omm_positions, omm_box_vectors = load_amber_system(
      prmtop_file, inpcrd_file, app.PME, **load_kwargs
    )

    # OpenMM reference constrained positions
    tol = 1e-6
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    sim = app.Simulation(omm_topology, omm_system, integrator, platform)
    sim.context.setPositions(omm_positions)
    sim.context.applyConstraints(tol)
    pos_omm_constrained = np.asarray(
      sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    )

    mm = convert_openmm_system(
      omm_system,
      omm_topology,
      omm_positions,
      omm_box_vectors,
      format=partition.NeighborListFormat.OrderedSparse,
    )

    # NOTE this only works in the case of an orthorhombic box
    disp_fn, shift_fn = space.periodic(mm.box_vectors)

    settle_data, ccma_data = amber_constraints.prepare_settle_ccma(
      np.asarray(jax.device_get(mm.constraint_idx)),
      np.asarray(jax.device_get(mm.constraint_dist)),
      np.asarray(jax.device_get(mm.masses)),
    )

    pos0 = mm.positions
    pos_jax = amber_constraints.settle(
      pos0,
      pos0,
      settle_data,
      mm.masses,
      disp_fn,
      shift_fn,
      box=mm.box_vectors,
      use_periodic_general=mm.nb_options.use_periodic_general,
    )
    pos_jax = amber_constraints.ccma(
      pos_jax,
      ccma_data,
      mm.masses,
      disp_fn,
      shift_fn,
      box=mm.box_vectors,
      use_periodic_general=mm.nb_options.use_periodic_general,
      tolerance=tol,
      max_iters=150,
    )

    # Constraint satisfaction: compare max distance violation.
    idx = mm.constraint_idx
    target = mm.constraint_dist
    i = idx[:, 0]
    j = idx[:, 1]

    dist_fn = jax.vmap(lambda a, b: jnp.sqrt(jnp.sum(disp_fn(a, b) ** 2)))
    pos_omm_init = np.asarray(omm_positions.value_in_unit(unit.angstrom))
    d_omm_init = dist_fn(
      jnp.asarray(pos_omm_init, dtype=pos0.dtype)[i],
      jnp.asarray(pos_omm_init, dtype=pos0.dtype)[j],
    )
    d_omm = dist_fn(
      jnp.asarray(pos_omm_constrained, dtype=pos0.dtype)[i],
      jnp.asarray(pos_omm_constrained, dtype=pos0.dtype)[j],
    )
    d_jax_init = dist_fn(pos0[i], pos0[j])
    d_jax = dist_fn(pos_jax[i], pos_jax[j])

    max_viol_omm = float(jnp.max(jnp.abs(d_omm - target)))
    max_viol_jax = float(jnp.max(jnp.abs(d_jax - target)))

    self.assertTrue(np.isfinite(max_viol_omm))
    self.assertTrue(np.isfinite(max_viol_jax))

    self.assertLess(max_viol_omm, 1e-6)
    self.assertLess(max_viol_jax, 1e-6)

    # Compare bond distances after projection
    self.assertAllClose(d_jax, d_omm, rtol=1e-6, atol=1e-4)

    def _abs_err_stats(d):
      abs_err = jnp.abs(d - target)
      return (
        float(jnp.max(abs_err)),
        float(jnp.mean(abs_err)),
        float(jnp.sqrt(jnp.mean(abs_err**2))),
      )

    max_omm_init, mean_omm_init, rms_omm_init = _abs_err_stats(d_omm_init)
    max_omm, mean_omm, rms_omm = _abs_err_stats(d_omm)
    max_jax_init, mean_jax_init, rms_jax_init = _abs_err_stats(d_jax_init)
    max_jax, mean_jax, rms_jax = _abs_err_stats(d_jax)

    n_constraints = int(idx.shape[0])
    n_settle = int(settle_data.atom1.shape[0])
    n_ccma = int(ccma_data.idx.shape[0])
    _vprint(
      "[constraints] "
      f"n_constraints={n_constraints} n_settle={n_settle} n_ccma={n_ccma} tol={tol:g}"
    )
    _vprint(
      "[constraints] "
      "OpenMM abs|d-d0|: "
      f"init max={max_omm_init:.3e} mean={mean_omm_init:.3e} rms={rms_omm_init:.3e}  "
      f"after max={max_omm:.3e} mean={mean_omm:.3e} rms={rms_omm:.3e}"
    )
    _vprint(
      "[constraints] "
      "JAX abs|d-d0|: "
      f"init max={max_jax_init:.3e} mean={mean_jax_init:.3e} rms={rms_jax_init:.3e}  "
      f"after max={max_jax:.3e} mean={mean_jax:.3e} rms={rms_jax:.3e}"
    )
    pos_delta = pos_jax - jnp.asarray(pos_omm_constrained, dtype=pos0.dtype)
    _vprint(
      "[constraints] "
      f"pos diff vs OpenMM: max_abs={float(jnp.max(jnp.abs(pos_delta))):.3e} "
      f"rms_abs={float(jnp.sqrt(jnp.mean(pos_delta**2))):.3e}"
    )


if __name__ == '__main__':
  absltest.main()
