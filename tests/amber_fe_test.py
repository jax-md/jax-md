from __future__ import annotations

from pathlib import Path
import os

import jax
from absl.testing import absltest, parameterized

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from jax_md import partition
from jax_md import test_util as jtu
import jax_md.mm_forcefields.amber as amber_energy
from jax_md.mm_forcefields.io.openmm import convert_openmm_system
from jax_md.mm_forcefields.nonbonded import electrostatics

try:
  import openmm as mm
  import openmm.app as app
  import openmm.unit as u
except ImportError:
  mm = None
  app = None
  u = None

# TODO change to files in data directory
_GRO = Path('data/amber_data/fe_test_data/mobley_3053621.gro')
_TOP = Path('data/amber_data/fe_test_data/mobley_3053621.top')
_INCLUDE_DIR = Path('data/amber_data/fe_test_data/gromacs_ffs')
_XVG_DIR = Path('data/amber_data/fe_test_data/sp_alch')

# copied from prod.X.mdp files in FreeSolv repository
# NOTE some options from the original FreeSolv setup were changed to be
# compatible with newer versions of gromacs and/or openmm; computed HFE
# values with pymbar are close, but obtaining identical results to the
# original publication may require further changes to the test setup
_LAMBDA_Q_SCHEDULE = [
  1.00, 0.75, 0.50, 0.25, 0.00,
  0.00, 0.00, 0.00, 0.00, 0.00,
  0.00, 0.00, 0.00, 0.00, 0.00,
  0.00, 0.00, 0.00, 0.00, 0.00,
]
_LAMBDA_LJ_SCHEDULE = [
  1.00, 1.00, 1.00, 1.00, 1.00,
  0.95, 0.90, 0.80, 0.70, 0.60,
  0.50, 0.40, 0.35, 0.30, 0.25,
  0.20, 0.15, 0.10, 0.05, 0.00,
]

_OMM_GMX_TOL_KJ = 0.3
_JAX_GMX_TOL_KJ = 0.3
_JAX_OMM_TOL_KJ = 0.001
_VERBOSE_TESTS = os.environ.get('VERBOSE_TESTS', '').strip().lower() not in (
  '',
  '0',
  'false',
  'no',
)


def _get_nonbonded_force(system: mm.System) -> mm.NonbondedForce:
  for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
      return force
  raise RuntimeError('No NonbondedForce found in system.')


def _configure_nbforce_for_jax_conversion(
  system: mm.System,
  *,
  use_switch: bool | None,
  switch_nm: float | None,
  use_dispersion_correction: bool | None,
) -> mm.System:
  nbforce = _get_nonbonded_force(system)
  if use_switch is not None:
    nbforce.setUseSwitchingFunction(bool(use_switch))
    if use_switch:
      nbforce.setSwitchingDistance(float(switch_nm) * u.nanometer)
  if use_dispersion_correction is not None:
    nbforce.setUseDispersionCorrection(bool(use_dispersion_correction))
  return system


def _scale_solute_charges_in_context(
  nbforce: mm.NonbondedForce,
  context: mm.Context,
  solute_indices: list[int],
  full_solute_charges: list[u.Quantity],
  lambda_q: float,
) -> None:
  for particle_index, full_charge in zip(solute_indices, full_solute_charges):
    _, sigma, epsilon = nbforce.getParticleParameters(particle_index)
    nbforce.setParticleParameters(particle_index, lambda_q * full_charge, sigma, epsilon)
  nbforce.updateParametersInContext(context)


def compute_alchemical_energy(
  system: mm.System,
  context: mm.Context,
  nbforce: mm.NonbondedForce,
  solute_indices: list[int],
  full_solute_charges: list[u.Quantity],
  lambda_q: float,
  lambda_lj: float,
) -> u.Quantity:
  context.setParameter('lambdaQ', lambda_q)
  context.setParameter('lambdaLJ', lambda_lj)
  _scale_solute_charges_in_context(
    nbforce, context, solute_indices, full_solute_charges, lambda_q=lambda_q
  )
  return context.getState(getEnergy=True).getPotentialEnergy()


def read_gromacs_potential_from_xvg(xvg_path: Path) -> float:
  series_legend: dict[int, str] = {}
  data_rows: list[list[float]] = []
  with xvg_path.open() as handle:
    for raw_line in handle:
      line = raw_line.strip()
      if not line:
        continue
      if line.startswith('@'):
        parts = line.split()
        if len(parts) >= 4 and parts[1].startswith('s') and parts[2] == 'legend':
          try:
            series_idx = int(parts[1][1:])
          except ValueError:
            continue
          legend = line.split('legend', 1)[1].strip().strip('"')
          series_legend[series_idx] = legend
        continue
      if line.startswith('#'):
        continue
      data_rows.append([float(token) for token in line.split()])
  potential_series_idx = None
  for idx, name in series_legend.items():
    if name.strip().lower() == 'potential':
      potential_series_idx = idx
      break
  potential_col = potential_series_idx + 1
  return data_rows[-1][potential_col]


# Adapted from https://github.com/choderalab/openmmtools/issues/376
# The core idea for the implementation used in gromacs involves dividing
# interactions into softcore (i.e. solute) and non-softcore groups and
# consistently applying tail corrections, lambda scaling, and exclusions
# NOTE solute larger than the main cutoff, charged molecules, and/or other
# combinations of options in gromacs may lead to discrepancies
def modify_alchemical_system(
  topology: app.Topology, system: mm.System, solute_resname: str = 'MOL'
) -> tuple[mm.System, list[int], list[u.Quantity]]:
  solute_indices: list[int] = []
  for res in topology.residues():
    if res.name == solute_resname:
      for atom in res.atoms():
        solute_indices.append(atom.index)

  nbforce = _get_nonbonded_force(system)
  alchemical_particles = set(solute_indices)
  chemical_particles = set(range(system.getNumParticles())) - alchemical_particles

  softcore_fn = '4.0*lambdaLJ*epsilon*x*(x-1.0); x=(1.0/reff_sterics);'
  softcore_fn += 'reff_sterics=(0.5*(1.0-lambdaLJ)+((r/sigma)^6));'
  softcore_fn += 'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)'
  softcore_force = mm.CustomNonbondedForce(softcore_fn)
  softcore_force.addGlobalParameter('lambdaLJ', 1.0)
  softcore_force.addPerParticleParameter('sigma')
  softcore_force.addPerParticleParameter('epsilon')

  one_4pi_eps0 = 138.935456
  solute_coul_fn = '(1.0-(lambdaQ^2))*ONE_4PI_EPS0*charge/r;'
  solute_coul_fn += f'ONE_4PI_EPS0 = {one_4pi_eps0:.16e};'
  solute_coul_fn += 'charge = charge1*charge2'
  solute_coul_force = mm.CustomNonbondedForce(solute_coul_fn)
  solute_coul_force.addGlobalParameter('lambdaQ', 1.0)
  solute_coul_force.addPerParticleParameter('charge')

  solute_lj_fn = '4.0*epsilon*x*(x-1.0); x=(sigma/r)^6;'
  solute_lj_fn += 'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)'
  solute_lj_force = mm.CustomNonbondedForce(solute_lj_fn)
  solute_lj_force.addPerParticleParameter('sigma')
  solute_lj_force.addPerParticleParameter('epsilon')

  index_to_solute = {idx: i for i, idx in enumerate(solute_indices)}
  full_solute_charges = [0.0 * u.elementary_charge] * len(solute_indices)

  for ind in range(system.getNumParticles()):
    charge, sigma, epsilon = nbforce.getParticleParameters(ind)
    newsigma = 0.3 * u.nanometer if sigma / u.nanometer == 0.0 else sigma
    softcore_force.addParticle([newsigma, epsilon])
    solute_coul_force.addParticle([charge])
    solute_lj_force.addParticle([sigma, epsilon])
    if ind in alchemical_particles:
      nbforce.setParticleParameters(ind, charge, sigma, epsilon * 0.0)
      full_solute_charges[index_to_solute[ind]] = charge

  for ind in range(nbforce.getNumExceptions()):
    p1, p2, _, _, _ = nbforce.getExceptionParameters(ind)
    softcore_force.addExclusion(p1, p2)
    solute_coul_force.addExclusion(p1, p2)
    solute_lj_force.addExclusion(p1, p2)

  softcore_force.addInteractionGroup(alchemical_particles, chemical_particles)
  solute_coul_force.addInteractionGroup(alchemical_particles, alchemical_particles)
  solute_lj_force.addInteractionGroup(alchemical_particles, alchemical_particles)

  softcore_force.setCutoffDistance(10.0 * u.angstroms)
  softcore_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  softcore_force.setUseSwitchingFunction(True)
  softcore_force.setSwitchingDistance(9.0 * u.angstroms)
  softcore_force.setUseLongRangeCorrection(True)

  solute_coul_force.setCutoffDistance(10.0 * u.angstroms)
  solute_coul_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  solute_coul_force.setUseLongRangeCorrection(False)

  solute_lj_force.setCutoffDistance(10.0 * u.angstroms)
  solute_lj_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  solute_lj_force.setUseSwitchingFunction(True)
  solute_lj_force.setSwitchingDistance(9.0 * u.angstroms)
  solute_lj_force.setUseLongRangeCorrection(False)

  system.addForce(softcore_force)
  system.addForce(solute_coul_force)
  system.addForce(solute_lj_force)
  return system, solute_indices, full_solute_charges


def build_jax_alchemical_components(
  *,
  base_system: mm.System,
  omm_topology: app.Topology,
  omm_positions,
  omm_box_vectors,
  solute_indices: list[int],
  use_switch: bool,
  switch_nm: float,
  use_dispersion_correction: bool,
  use_softcore_lrc: bool,
  softcore_lrc_points: int,
):
  # trick to deep copy the system to avoid conflicting changes
  conversion_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
  _configure_nbforce_for_jax_conversion(
    conversion_system,
    use_switch=use_switch,
    switch_nm=switch_nm,
    use_dispersion_correction=use_dispersion_correction,
  )

  converted = convert_openmm_system(
    conversion_system,
    omm_topology,
    omm_positions,
    omm_box_vectors,
    format=partition.NeighborListFormat.OrderedSparse,
    precision='double',
  )

  if converted.recip_alpha is not None and converted.recip_grid is not None:
    coul = electrostatics.PMECoulomb(
      r_cut=converted.nb_options.r_cut,
      alpha=converted.recip_alpha,
      grid_size=converted.recip_grid,
    )
  elif converted.nb_options.r_cut is not None:
    coul = electrostatics.CutoffCoulomb(r_cut=converted.nb_options.r_cut)
  else:
    coul = electrostatics.CutoffCoulomb(r_cut=1.0e6)

  sc_mask = jnp.zeros((converted.topology.n_atoms,), dtype=bool)
  sc_mask = sc_mask.at[jnp.asarray(solute_indices, dtype=jnp.int32)].set(True)

  fe_opts = amber_energy.FEOptions(
    vdw_scaling='softcore',
    coul_scaling='linear',
    sc_mask=sc_mask,
    softcore_lrc=bool(use_softcore_lrc),
    softcore_lrc_points=int(softcore_lrc_points),
  )

  energy_fn, neighbor_fn, _, _ = amber_energy.energy(
    converted.params,
    converted.topology,
    converted.box_vectors,
    coulomb_options=coul,
    nb_options=converted.nb_options,
    fe_options=fe_opts,
    dense_mask_format=False,
  )
  nbr = neighbor_fn.allocate(converted.positions)
  return converted, energy_fn, neighbor_fn, nbr, sc_mask


def compute_alchemical_energy_jax(
  *,
  lambda_q: float,
  lambda_lj: float,
  converted,
  energy_fn,
  neighbor_fn,
  nbr,
  sc_mask,
):
  nbr = neighbor_fn.update(converted.positions, nbr)

  params = converted.params
  charges = params.nonbonded.charges
  scaled_charges = jnp.where(sc_mask, charges * lambda_q, charges)
  nonbonded_scaled = params.nonbonded._replace(
    charges=scaled_charges,
    exc_charge_prod=params.nonbonded.exc_charge_prod,
  )
  params_scaled = params._replace(nonbonded=nonbonded_scaled)

  cl_lambda = 1.0 - lambda_lj
  terms = energy_fn(
    converted.positions,
    nbr,
    cl_lambda=cl_lambda,
    lambda_q=lambda_q,
    params=params_scaled,
  )
  return float(terms['etotal'] * 4.184), nbr


class AMBERFETest(jtu.JAXMDTestCase, parameterized.TestCase):

  def test_amber_fe_schedule(self):
    lambda_q_schedule = _LAMBDA_Q_SCHEDULE
    lambda_lj_schedule = _LAMBDA_LJ_SCHEDULE

    gro = app.GromacsGroFile(str(_GRO))
    top = app.GromacsTopFile(
      str(_TOP),
      includeDir=str(_INCLUDE_DIR),
      periodicBoxVectors=gro.getPeriodicBoxVectors(),
    )

    base_system = top.createSystem(
      nonbondedMethod=app.PME,
      nonbondedCutoff=1.0 * u.nanometer,
      constraints=app.HBonds,
      rigidWater=True,
      ewaldErrorTolerance=1.0e-5,
      switchDistance=0.9 * u.nanometer,
    )
    _configure_nbforce_for_jax_conversion(
      base_system,
      use_switch=None,
      switch_nm=None,
      use_dispersion_correction=True,
    )

    # trick to deep copy system to avoid conflicting changes with alchemical modifications
    alchemical_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
    alchemical_system, solute_indices, full_solute_charges = modify_alchemical_system(
      top.topology, alchemical_system
    )
    alchemical_nbforce = _get_nonbonded_force(alchemical_system)

    platform = mm.Platform.getPlatformByName("Reference")
    omm_integrator = mm.VerletIntegrator(1.0 * u.femtoseconds)
    omm_context = mm.Context(alchemical_system, omm_integrator, platform)
    omm_context.setPositions(gro.getPositions())

    converted, energy_fn, neighbor_fn, nbr, sc_mask = build_jax_alchemical_components(
      base_system=base_system,
      omm_topology=top.topology,
      omm_positions=gro.getPositions(),
      omm_box_vectors=gro.getPeriodicBoxVectors(),
      solute_indices=solute_indices,
      use_switch=True,
      switch_nm=0.9,
      use_dispersion_correction=True,
      use_softcore_lrc=True,
      softcore_lrc_points=96,
    )

    if _VERBOSE_TESTS:
      print('\nWindow comparison:')
      print('window  lambdaQ  lambdaLJ      openmm_kj         jax_kj     gromacs_kj    omm-gmx    jax-gmx    jax-omm')

    for window, (lambda_q, lambda_lj) in enumerate(zip(lambda_q_schedule, lambda_lj_schedule)):
      omm_kj = compute_alchemical_energy(
        alchemical_system,
        omm_context,
        alchemical_nbforce,
        solute_indices,
        full_solute_charges,
        lambda_q=lambda_q,
        lambda_lj=lambda_lj,
      ).value_in_unit(u.kilojoule_per_mole)

      jax_kj, nbr = compute_alchemical_energy_jax(
        lambda_q=lambda_q,
        lambda_lj=lambda_lj,
        converted=converted,
        energy_fn=energy_fn,
        neighbor_fn=neighbor_fn,
        nbr=nbr,
        sc_mask=sc_mask,
      )

      xvg_path = _XVG_DIR / f'sp_all_terms_window_{window:02d}.xvg'
      gmx_kj = read_gromacs_potential_from_xvg(xvg_path)

      d_omm_gmx = omm_kj - gmx_kj
      d_jax_gmx = jax_kj - gmx_kj
      d_jax_omm = jax_kj - omm_kj

      if _VERBOSE_TESTS:
        print(
          f'{window:6d}  {lambda_q:7.3f}  {lambda_lj:8.3f}  '
          f'{omm_kj:13.6f}  {jax_kj:13.6f}  {gmx_kj:13.6f}  '
          f'{d_omm_gmx:9.6f}  {d_jax_gmx:9.6f}  {d_jax_omm:9.6f}'
        )

      self.assertLessEqual(abs(d_omm_gmx), _OMM_GMX_TOL_KJ)
      self.assertLessEqual(abs(d_jax_gmx), _JAX_GMX_TOL_KJ)
      self.assertLessEqual(abs(d_jax_omm), _JAX_OMM_TOL_KJ)


if __name__ == '__main__':
  absltest.main()
