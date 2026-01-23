"""OpenMM-based loader that converts MM inputs into `mm_forcefields` types.

This module intentionally treats OpenMM objects as an I/O layer: we extract
values into NumPy/JAX arrays and then operate purely in JAX downstream.
"""

import os
from functools import partial
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

# TODO convert these to use sentinel variable to avoid type checking issues
try:
  import openmm  # type: ignore
  import openmm.app as app  # type: ignore
  import openmm.unit as unit  # type: ignore
except ImportError:  # pragma: no cover
  openmm = None
  app = None
  unit = None

try:
  from openmmforcefields.generators import SystemGenerator  # type: ignore
except ImportError:  # pragma: no cover
  SystemGenerator = None

try:
  import parmed  # type: ignore
except ImportError:  # pragma: no cover
  parmed = None

from jax_md import partition, simulate, space, util
from jax_md.mm_forcefields.base import (
  BondedParameters,
  NonbondedOptions,
  NonbondedParameters,
  Topology,
)
from jax_md.mm_forcefields.oplsaa.params import Parameters

# Types and constants

f32 = util.f32
f64 = util.f64
Array = util.Array
NonbondedMethod = Union[
  app.NoCutoff,
  app.CutoffNonPeriodic,
  app.CutoffPeriodic,
  app.Ewald,
  app.PME,
]

_KCAL_TO_KJ = 4.184
_KJ_TO_KCAL = 1.0 / _KCAL_TO_KJ
_NM_TO_ANG = 10.0
_ANG_TO_NM = 1.0 / _NM_TO_ANG

class VirtualSiteData(NamedTuple):
  """Fixed-shape virtual site metadata for per-step position reconstruction.

  All arrays are per-atom and padded so they are JAX-friendly and vmappable.

  Notes:
    - For non-virtual-site atoms, entries are dummy values (type=0, parents=-1).
    - For massless sites (common in 4-site water models like OPC), masses[i] == 0
      and is_virtual_site[i] == True. Integrators should generally not advance
      these DOFs; instead, recompute their positions from the parent atoms.
  """

  is_virtual_site: jnp.ndarray  # (N,) bool
  vsite_type: (
    jnp.ndarray
  )  # (N,) int32: 0 none, 1 two-avg, 2 three-avg, 3 oop, 4 local
  parent_idx: jnp.ndarray  # (N, 3) int32, padded with -1
  weights: jnp.ndarray  # (N, 4) float, interpretation depends on type
  origin_weights: jnp.ndarray  # (N, 3) float (LocalCoordinatesSite)
  x_weights: jnp.ndarray  # (N, 3) float (LocalCoordinatesSite)
  y_weights: jnp.ndarray  # (N, 3) float (LocalCoordinatesSite)
  local_position: jnp.ndarray  # (N, 3) float, Angstrom (LocalCoordinatesSite)


# TODO consider if this can be cleaned up or if using Optional/Any is good practice
class OpenMMSystem(NamedTuple):
  positions: Array
  topology: Topology
  params: Parameters
  box_vectors: Optional[Array]
  nb_options: NonbondedOptions
  recip_alpha: Optional[float]
  recip_grid: Optional[Array]
  ewald_error_tolerance: Optional[float]
  masses: Optional[Array]
  virtual_sites: Optional[VirtualSiteData]
  constraint_idx: Optional[Array]
  constraint_dist: Optional[Array]


# TODO test more rigorously to make sure these error messages occur as expected
def _maybe_import_omm() -> None:
  """Raise an ImportError if OpenMM is unavailable."""
  if openmm is None or app is None or unit is None:
    raise ImportError('openmm is required to use the OpenMM loader.')


def _maybe_import_ommff() -> None:
  """Raise an ImportError if openmmforcefields is unavailable."""
  if SystemGenerator is None:
    raise ImportError('openmmforcefields is required to use the OpenMM loader.')


def _maybe_import_parmed() -> None:
  """Raise an ImportError if ParmEd is unavailable."""
  if parmed is None:
    raise ImportError('parmed is required to use the ParmEd loader.')


def load_amber_system(
  prmtop_path: str,
  inpcrd_path: str,
  nb_method: Optional[NonbondedMethod] = None,
  **kwargs: Any,
) -> tuple[Any, Any, Any, Any]:
  """Load Amber inputs and build an OpenMM System."""
  _maybe_import_omm()
  if not (os.path.exists(prmtop_path) and os.path.exists(inpcrd_path)):
    raise FileNotFoundError(f'Missing input files {prmtop_path}/{inpcrd_path}')
  inpcrd = app.AmberInpcrdFile(inpcrd_path)
  positions = inpcrd.getPositions(asNumpy=True)
  prmtop = app.AmberPrmtopFile(
    prmtop_path, periodicBoxVectors=inpcrd.boxVectors
  )
  if nb_method is None:
    nb_method = app.PME if inpcrd.boxVectors is not None else app.NoCutoff
  system = prmtop.createSystem(nonbondedMethod=nb_method, **kwargs)
  # TODO also maybe return velocities if present? possibly also base omm objects?
  return system, prmtop.topology, positions, inpcrd.boxVectors


# also from omm_readparams.py
def _charmm_read_box(psf: Any, filename: str) -> Any:
  with open(filename, 'r') as f:
    try:
      import json

      sysinfo = json.load(f)
      boxlx, boxly, boxlz = map(float, sysinfo['dimensions'][:3])
    except:
      for line in f:
        segments = line.split('=')
        if segments[0].strip() == 'BOXLX':
          boxlx = float(segments[1])
        if segments[0].strip() == 'BOXLY':
          boxly = float(segments[1])
        if segments[0].strip() == 'BOXLZ':
          boxlz = float(segments[1])
  psf.setBox(
    boxlx * unit.angstroms, boxly * unit.angstroms, boxlz * unit.angstroms
  )
  return psf


def load_charmm_system(
  crd_path: str,
  psf_path: str,
  param_paths: Sequence[str],
  nb_method: Optional[NonbondedMethod] = None,
  sys_info: Optional[str] = None,
  **kwargs: Any,
) -> tuple[Any, Any, Any, Any]:
  # param_paths is list of files with extensions such as par, prm, top, rtf, inp, and str
  _maybe_import_omm()
  crd_extension = os.path.splitext(crd_path)[-1]
  if crd_extension == '.pdb':
    crd = app.PDBFile(crd_path)
    positions = crd.getPositions(asNumpy=True)
  elif crd_extension == '.crd':
    crd = app.CharmmCrdFile(crd_path)
    positions = crd.positions
  elif crd_extension == '.rst':
    crd = app.CharmmRstFile(crd_path)
    positions = crd.positions
  else:
    raise ValueError('CHARMM coordinate files must end in .pdb, .crd, or .rst')
  psf = app.CharmmPsfFile(psf_path)
  if sys_info is not None:
    psf = _charmm_read_box(psf, sys_info)
  params = app.CharmmParameterSet(*param_paths)
  if nb_method is None:
    nb_method = app.PME if psf.boxVectors is not None else app.NoCutoff
  system = psf.createSystem(params, nonbondedMethod=nb_method, **kwargs)
  return system, psf.topology, positions, psf.boxVectors


def load_gromacs_system(
  gro_path: str,
  top_path: str,
  nb_method: Optional[NonbondedMethod] = None,
  **kwargs: Any,
) -> tuple[Any, Any, Any, Any]:
  _maybe_import_omm()
  gro = app.GromacsGroFile(gro_path)
  positions = gro.getPositions(asNumpy=True)
  top = app.GromacsTopFile(
    top_path,
    periodicBoxVectors=gro.getPeriodicBoxVectors(),
    includeDir='/usr/local/gromacs/share/gromacs/top',
  )
  if nb_method is None:
    nb_method = app.PME if gro.boxVectors is not None else app.NoCutoff
  system = top.createSystem(nonbondedMethod=nb_method, **kwargs)
  return system, top.topology, positions, gro.boxVectors


# TODO is this necessary/will amoeba be supported?
def load_tinker_system() -> None:
  return


# following is based on examples from:
# https://github.com/openmm/openmmforcefields
# https://docs.openmm.org/latest/userguide/application/02_running_sims.html#force-fields
# TODO this needs a look, especially due to the complex parameter passing behavior
# also need to look at openff.toolkit to reason about the most general way to do this
# https://github.com/openmm/openmmforcefields/blob/main/openmmforcefields/
# generators/system_generators.py
def load_generator_system(
  topology: Any,
  generator: Any = None,
  nb_method: Optional[NonbondedMethod] = None,
  molecules: Optional[Sequence[Any]] = None,
  forcefield_files: Optional[Sequence[str]] = None,
  small_molecule_forcefield: str = 'gaff-2.11',
  generator_kwargs: Optional[Mapping[str, Any]] = None,
  create_system_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
  """Load a system using OpenMM SystemGenerator (e.g. OpenMMForceFields/OpenFF).

  Args:
      topology: OpenMM Topology object.
      generator: Optional prebuilt SystemGenerator. If None, one is created.
      nb_method: Optional OpenMM nonbonded method. If None, inferred from box.
      molecules: Optional list of small-molecule objects passed to the generator.
      forcefield_files: XML files for protein/water forcefields when building a generator.
      small_molecule_forcefield: Name of small-molecule template (e.g., GAFF/SMIRNOFF).
      generator_kwargs: Extra kwargs for SystemGenerator constructor.
      create_system_kwargs: Extra kwargs for generator.createSystem.

  Returns:
      (system, boxVectors)
  """
  _maybe_import_ommff()
  box = topology.getPeriodicBoxVectors()
  if nb_method is None:
    nb_method = app.PME if box is not None else app.NoCutoff

  if generator is None:
    if SystemGenerator is None:  # pragma: no cover
      raise ImportError(
        'openmmforcefields is required for generator-based loading.'
      )
    generator_kwargs = generator_kwargs or {}
    generator = SystemGenerator(
      forcefields=forcefield_files or [],
      small_molecule_forcefield=small_molecule_forcefield,
      molecules=molecules,
      **generator_kwargs,
    )

  create_system_kwargs = create_system_kwargs or {}
  system = generator.createSystem(
    topology,
    molecules=molecules,
    nonbondedMethod=nb_method,
    **create_system_kwargs,
  )
  return system, box


def load_parmed_system(
  structure: Any,
  param_files: Optional[Sequence[str]] = None,
  nb_method: Optional[NonbondedMethod] = None,
  **kwargs: Any,
) -> tuple[Any, Any, Any, Any]:
  """Load a system via ParmEd and convert to OpenMM.

  Args:
      structure: Either a ParmEd Structure object, or a path accepted by
        `parmed.load_file(...)`.
      nb_method: Optional OpenMM nonbonded method. If None, inferred from box.
      kwargs: Extra kwargs forwarded to Structure.createSystem.

  Returns:
      (system, topology, positions, boxVectors)
  """
  _maybe_import_parmed()

  # Normalize to a ParmEd Structure.
  if isinstance(structure, str):
    structure = parmed.load_file(structure, param_files)  # type: ignore

  box_vectors = getattr(structure, 'box_vectors', None)
  if nb_method is None:
    nb_method = app.PME if box_vectors is not None else app.NoCutoff

  system = structure.createSystem(nonbondedMethod=nb_method, **kwargs)
  topology = structure.topology
  positions = structure.positions

  return system, topology, positions, box_vectors


# TODO it may be wise to keep everything as ONP structs and then only convert if needed
# TODO break up into more modular function
def convert_openmm_system(
  system: Any,
  topology: Any,
  positions: Any,
  box_vectors: Any,
  r_cut: Optional[float] = None,
  dr_threshold: float = 0.0,
  format: partition.NeighborListFormat = partition.NeighborListFormat.OrderedSparse,
  precision: str = 'double',
) -> OpenMMSystem:
  """Convert an OpenMM System/Topology into mm_forcefields structures.

  Args:
      system: OpenMM System with forces already constructed.
      topology: OpenMM Topology corresponding to the system.
      positions: Positions (OpenMM Quantity in nm or ndarray in nm).
      box_vectors: Box vectors for the system # TODO should be removed and extracted from system?
      dr_threshold: Neighbor list skin distance (A).
      precision: "single" or "double" to choose dtype of returned array.
        In general, floats are stored as float32/64 while ints are int32

  Returns:
      OpenMMSystem tuple with positions (A), Topology, Parameters, NonbondedOptions,
      and per-pair 1-4 scaling information.
  """
  _maybe_import_omm()

  if precision == 'double':
    wp_float = np.float64
    wp_int = np.int32
  elif precision == 'single':
    wp_float = np.float32
    wp_int = np.int32
  else:
    raise ValueError('Invalid option provided for precision')

  pos = np.asarray(positions.value_in_unit(unit.angstrom), dtype=wp_float)

  n_atoms = system.getNumParticles()

  # TODO Excessive use of empty arrays for rarely used terms isn't performant

  # Get masses in AMU and record virtual site information if relevant
  masses = np.zeros((n_atoms,), dtype=wp_float)
  is_virtual_site = np.zeros((n_atoms,), dtype=np.bool_)
  vsite_type = np.zeros((n_atoms,), dtype=wp_int)
  vsite_parent_idx = np.full((n_atoms, 3), -1, dtype=wp_int)
  vsite_weights = np.zeros((n_atoms, 4), dtype=wp_float)
  vsite_origin_weights = np.zeros((n_atoms, 3), dtype=wp_float)
  vsite_x_weights = np.zeros((n_atoms, 3), dtype=wp_float)
  vsite_y_weights = np.zeros((n_atoms, 3), dtype=wp_float)
  vsite_local_position = np.zeros((n_atoms, 3), dtype=wp_float)

  two_avg_cls = getattr(openmm, 'TwoParticleAverageSite', None)
  three_avg_cls = getattr(openmm, 'ThreeParticleAverageSite', None)
  oop_cls = getattr(openmm, 'OutOfPlaneSite', None)
  local_cls = getattr(openmm, 'LocalCoordinatesSite', None)
  # TODO Also implement symmetry site later if applicable

  def _vec3_to_ang(vec3):
    if hasattr(vec3, 'value_in_unit'):
      return np.asarray(vec3.value_in_unit(unit.angstrom), dtype=wp_float)
    return _NM_TO_ANG * np.asarray(vec3, dtype=wp_float)

  def _vec3_to_arr(vec3):
    if hasattr(vec3, 'value_in_unit'):
      return np.asarray(vec3.value_in_unit(unit.dimensionless), dtype=wp_float)
    return np.asarray(vec3, dtype=wp_float)

  # TODO move to utility function
  vs_types = []
  vs_count = 0
  for idx in range(n_atoms):
    masses[idx] = system.getParticleMass(idx).value_in_unit(unit.dalton)
    if system.isVirtualSite(idx):
      vs_count += 1
      is_virtual_site[idx] = True
      virtual_site = system.getVirtualSite(idx)
      if type(virtual_site) not in vs_types:
        vs_types.append(type(virtual_site))
      if two_avg_cls is not None and isinstance(virtual_site, two_avg_cls):
        vsite_type[idx] = 1
        vsite_parent_idx[idx, 0] = virtual_site.getParticle(0)
        vsite_parent_idx[idx, 1] = virtual_site.getParticle(1)
        if hasattr(virtual_site, 'getWeight'):
          vsite_weights[idx, 0] = float(virtual_site.getWeight(0))
          vsite_weights[idx, 1] = float(virtual_site.getWeight(1))
        else:
          vsite_weights[idx, 0] = float(virtual_site.getWeight1())
          vsite_weights[idx, 1] = float(virtual_site.getWeight2())
      elif three_avg_cls is not None and isinstance(
        virtual_site, three_avg_cls
      ):
        vsite_type[idx] = 2
        vsite_parent_idx[idx, 0] = virtual_site.getParticle(0)
        vsite_parent_idx[idx, 1] = virtual_site.getParticle(1)
        vsite_parent_idx[idx, 2] = virtual_site.getParticle(2)
        if hasattr(virtual_site, 'getWeight'):
          vsite_weights[idx, 0] = float(virtual_site.getWeight(0))
          vsite_weights[idx, 1] = float(virtual_site.getWeight(1))
          vsite_weights[idx, 2] = float(virtual_site.getWeight(2))
        else:
          vsite_weights[idx, 0] = float(virtual_site.getWeight1())
          vsite_weights[idx, 1] = float(virtual_site.getWeight2())
          vsite_weights[idx, 2] = float(virtual_site.getWeight3())
      elif oop_cls is not None and isinstance(virtual_site, oop_cls):
        vsite_type[idx] = 3
        vsite_parent_idx[idx, 0] = virtual_site.getParticle(0)
        vsite_parent_idx[idx, 1] = virtual_site.getParticle(1)
        vsite_parent_idx[idx, 2] = virtual_site.getParticle(2)
        # OpenMM OutOfPlaneSite uses weights: w12, w13, wCross.
        if hasattr(virtual_site, 'getWeight12'):
          vsite_weights[idx, 0] = float(virtual_site.getWeight12())
          vsite_weights[idx, 1] = float(virtual_site.getWeight13())
          vsite_weights[idx, 2] = float(virtual_site.getWeightCross())
        else:
          vsite_weights[idx, 0] = float(virtual_site.getWeight(0))
          vsite_weights[idx, 1] = float(virtual_site.getWeight(1))
          vsite_weights[idx, 2] = float(virtual_site.getWeight(2))
      elif local_cls is not None and isinstance(virtual_site, local_cls):
        vsite_type[idx] = 4
        # Local coordinate frame defined by (origin, x, y) particles.
        vsite_parent_idx[idx, 0] = int(virtual_site.getOriginParticle())
        vsite_parent_idx[idx, 1] = int(virtual_site.getXParticle())
        vsite_parent_idx[idx, 2] = int(virtual_site.getYParticle())
        vsite_origin_weights[idx] = _vec3_to_arr(
          virtual_site.getOriginWeights()
        )
        vsite_x_weights[idx] = _vec3_to_arr(virtual_site.getXWeights())
        vsite_y_weights[idx] = _vec3_to_arr(virtual_site.getYWeights())
        vsite_local_position[idx] = _vec3_to_ang(
          virtual_site.getLocalPosition()
        )
      else:
        raise ValueError(f'Unknown virtual type site {type(virtual_site)}')

  if vs_types:
    print(
      '[WARNING] Virtual site support is incomplete and can result in incorrect energies, forces, and physical quantities'
    )

  # TODO might be better to load arrays into dict as they're created
  # and then have a function that constructs the dataclasses with appropriate
  # defaults as needed, or bake the defaults into the class signature

  # Topology placeholders
  bonds_arr = np.zeros((0, 2), dtype=wp_int)
  angles_arr = np.zeros((0, 3), dtype=wp_int)
  torsions_arr = np.zeros((0, 4), dtype=wp_int)
  impropers_arr = np.zeros((0, 4), dtype=wp_int)
  cmap_atoms_arr = np.zeros((0, 8), dtype=wp_int)
  cmap_map_id_arr = np.zeros((0,), dtype=wp_int)
  exc_pairs_arr = np.zeros((0, 2), dtype=wp_int)
  nbfix_atom_type_arr = np.zeros((0,), dtype=wp_int)
  constraint_idx_arr = np.zeros((0, 2), dtype=wp_int)
  constraint_dist_arr = np.zeros((0,), dtype=wp_float)

  # Bonded placeholders
  bond_k = np.zeros((0,), dtype=wp_float)
  bond_r0 = np.zeros((0,), dtype=wp_float)
  angle_k = np.zeros((0,), dtype=wp_float)
  angle_theta0 = np.zeros((0,), dtype=wp_float)
  torsion_k = np.zeros((0,), dtype=wp_float)
  torsion_n = np.zeros((0,), dtype=wp_int)
  torsion_phase = np.zeros((0,), dtype=wp_float)
  improper_k = np.zeros((0,), dtype=wp_float)
  improper_theta0 = np.zeros((0,), dtype=wp_float)
  cmap_maps_arr = np.zeros((0, 0, 0), dtype=wp_float)

  # Nonbonded placeholders
  charges_arr = np.zeros((0,), dtype=wp_float)
  sigma_arr = np.zeros((0,), dtype=wp_float)
  epsilon_arr = np.zeros((0,), dtype=wp_float)
  exc_charge_prod_arr = np.zeros((0,), dtype=wp_float)
  exc_sigma_arr = np.zeros((0,), dtype=wp_float)
  exc_epsilon_arr = np.zeros((0,), dtype=wp_float)
  nbfix_acoef_table = np.zeros((0, 0), dtype=wp_float)
  nbfix_bcoef_table = np.zeros((0, 0), dtype=wp_float)

  # NBOption placeholders
  use_soft_lj = False
  lj_cap = None
  use_shift_lj = False
  scale_14_lj = None
  scale_14_coul = None
  use_pbc = False
  use_periodic_general = False
  fractional_coordinates = False
  disp_coef = 0.0
  r_switch = None

  # PME parameter placeholders
  recip_alpha = None
  recip_grid = None
  ewald_error_tolerance = None

  # TODO it may not be wise to extract bond, angle, dihedral, etc from forces
  # charmmpsffile.py adds 2 bond forces, one for covalent bonds
  # and one for Urey-Bradley terms, consider separating later
  # the easiest way to detect the UB force is to use getForceGroup
  # and check for UREY_BRADLEY_FORCE_GROUP which is usually 3
  for force in system.getForces():
    if isinstance(force, openmm.HarmonicBondForce):
      num_bonds = force.getNumBonds()
      bonds_this = np.zeros((num_bonds, 2), dtype=wp_int)
      bond_r0_this = np.zeros((num_bonds,), dtype=wp_float)
      bond_k_this = np.zeros((num_bonds,), dtype=wp_float)
      for idx in range(num_bonds):
        i, j, r0, k = force.getBondParameters(idx)
        bonds_this[idx] = (i, j)
        bond_r0_this[idx] = r0.value_in_unit(unit.angstrom)
        bond_k_this[idx] = k.value_in_unit(
          unit.kilocalorie_per_mole / unit.angstrom**2
        )
      # Ensures Urey-Bradley terms are handled correctly
      bonds_arr = np.concatenate((bonds_arr, bonds_this), axis=0)
      bond_r0 = np.concatenate((bond_r0, bond_r0_this), axis=0)
      bond_k = np.concatenate((bond_k, bond_k_this), axis=0)
    elif isinstance(force, openmm.HarmonicAngleForce):
      num_angles = force.getNumAngles()
      angles_arr = np.zeros((num_angles, 3), dtype=wp_int)
      angle_theta0 = np.zeros((num_angles,), dtype=wp_float)
      angle_k = np.zeros((num_angles,), dtype=wp_float)
      for idx in range(num_angles):
        i, j, k, theta0, k_val = force.getAngleParameters(idx)
        angles_arr[idx] = (i, j, k)
        angle_theta0[idx] = theta0.value_in_unit(unit.radian)
        angle_k[idx] = k_val.value_in_unit(
          unit.kilocalorie_per_mole / unit.radian**2
        )
    elif isinstance(force, openmm.PeriodicTorsionForce):
      num_torsions = force.getNumTorsions()
      torsions_arr = np.zeros((num_torsions, 4), dtype=wp_int)
      torsion_n = np.zeros((num_torsions,), dtype=wp_int)
      torsion_phase = np.zeros((num_torsions,), dtype=wp_float)
      torsion_k = np.zeros((num_torsions,), dtype=wp_float)
      for idx in range(force.getNumTorsions()):
        i, j, k, l, periodicity, phase, k_val = force.getTorsionParameters(idx)
        torsions_arr[idx] = (i, j, k, l)
        torsion_n[idx] = periodicity
        torsion_phase[idx] = phase.value_in_unit(unit.radian)
        torsion_k[idx] = k_val.value_in_unit(unit.kilocalorie_per_mole)
    elif isinstance(force, openmm.NonbondedForce):
      # Relevant options from force object
      nb_method = force.getNonbondedMethod()
      if force.getNonbondedMethod() is not openmm.NonbondedForce.NoCutoff:
        if r_cut is None:
          r_cut = force.getCutoffDistance().value_in_unit(unit.angstrom)
      if force.getUseSwitchingFunction():
        r_switch = force.getSwitchingDistance().value_in_unit(unit.angstrom)

      # Per-atom nonbonded params
      charges_arr = np.zeros((n_atoms,), dtype=wp_float)
      sigma_arr = np.zeros((n_atoms,), dtype=wp_float)
      epsilon_arr = np.zeros((n_atoms,), dtype=wp_float)
      for idx in range(n_atoms):
        charge, sigma, eps = force.getParticleParameters(idx)
        charges_arr[idx] = charge.value_in_unit(unit.elementary_charge)
        sigma_arr[idx] = sigma.value_in_unit(unit.angstrom)
        epsilon_arr[idx] = eps.value_in_unit(unit.kilocalorie_per_mole)

      # Exception params
      num_exceptions = force.getNumExceptions()
      exc_pairs_arr = np.zeros((num_exceptions, 2), dtype=wp_int)
      exc_charge_prod_arr = np.zeros((num_exceptions,), dtype=wp_float)
      exc_sigma_arr = np.zeros((num_exceptions,), dtype=wp_float)
      exc_epsilon_arr = np.zeros((num_exceptions,), dtype=wp_float)
      for idx in range(force.getNumExceptions()):
        i, j, charge_prod, sigma_exc, eps_exc = force.getExceptionParameters(
          idx
        )
        exc_pairs_arr[idx] = (i, j)
        exc_charge_prod_arr[idx] = charge_prod.value_in_unit(
          unit.elementary_charge**2
        )
        exc_sigma_arr[idx] = sigma_exc.value_in_unit(unit.angstrom)
        exc_epsilon_arr[idx] = eps_exc.value_in_unit(unit.kilocalorie_per_mole)

      # PME / Ewald parameters.
      # TODO
      # In OpenMM, these can be explicitly set on the NonbondedForce, or computed
      # automatically at context creation time. When automatic, getParameters()
      # often returns zeros, so this falls back to getParametersInContext()
      # Some codepaths may not use the parameters this returns though
      # This probably also isn't safe for non-constant volume simulations (NPT)
      # so this behavior for OpenMM needs to be investigated more.
      # Alternatively, the function that determines the grid size could be copied
      # with extra constraints for small boxes or oddly shaped triclinic cells
      if nb_method in (openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald):
        ewald_error_tolerance = float(force.getEwaldErrorTolerance())
        alpha, nx, ny, nz = force.getPMEParameters()
        recip_alpha = alpha.value_in_unit(unit.angstrom**-1)
        recip_grid = np.asarray([int(nx), int(ny), int(nz)], dtype=wp_int)

        if (
          (recip_alpha == 0.0)
          or (int(nx) == 0)
          or (int(ny) == 0)
          or (int(nz) == 0)
        ):
          integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
          platform = openmm.Platform.getPlatformByName('Reference')
          context = openmm.Context(system, integrator, platform)
          context.setPositions(positions)
          alpha_ctx, nx_ctx, ny_ctx, nz_ctx = force.getPMEParametersInContext(
            context
          )
          recip_alpha = float(alpha_ctx) / _NM_TO_ANG
          recip_grid = np.asarray(
            [int(nx_ctx), int(ny_ctx), int(nz_ctx)], dtype=wp_int
          )

      if nb_method is not openmm.NonbondedForce.NoCutoff:
        use_pbc = True
        # OpenMM supports triclinic periodic boxes (see 22.1 in the OMM manual)
        # have to convert lower triangular to upper triangular matrix e.g.
        # [[Lx, Ly, Lz], [0, Ly, Lz], [0, 0, Lz]]
        box_vectors = np.array(box_vectors.value_in_unit(unit.angstrom)).T

        # If matrix is diagonal, then unit cell is orthorhombic, otherwise
        # it is assumed to be triclinic according to OpenMM's conventions
        # positions must be wrapped back into the unit cell if taken from OpenMM
        if np.array_equal(box_vectors, np.diag(np.diag(box_vectors))):
          box_vectors = np.diag(box_vectors)
          pos = np.mod(pos, box_vectors)
        else:
          # NOTE cell list neighboring with 3x3 box and no fractional coordinates
          # seems to have issues, so fractional coordinates are required for non
          # orthorhombic boxes, also requiring the positions to be converted
          pos = space.transform(space.inverse(box_vectors), pos)
          pos = np.mod(pos, 1.0)  # wrap into [0,1)
          use_periodic_general = True
          fractional_coordinates = True

      # NOTE charmpsffile.py turns off dispersion correction on the main force,
      # but does not add any long range correction to the CustomNonbondedForce.
      #
      # CustomNonbondedForceImpl.cpp uses numeric integration to support
      # arbitrary potentials (not necessarily 12-6 LJ), whereas the standard
      # NonbondedForce uses an analytic LJ long-range correction.
      #
      # Related: https://github.com/openmm/openmm/issues/3162
      #
      # For now, assume NBFIX and an analytic dispersion correction are not
      # enabled at the same time, and that the correction we compute (below) is
      # only for a 12-6 potential.
      if (
        force.getUseDispersionCorrection()
        and nb_method != openmm.NonbondedForce.NoCutoff
      ):
        disp_coefs = np.stack([sigma_arr, epsilon_arr], axis=1)
        values, count = np.unique(disp_coefs, axis=0, return_counts=True)
        sigma2 = values[:, 0] * values[:, 0]
        sigma6 = sigma2 * sigma2 * sigma2
        count_s = count * (count + 1) / 2
        sum1 = np.sum(count_s * values[:, 1] * sigma6 * sigma6)
        sum2 = np.sum(count_s * values[:, 1] * sigma6)

        sig_mesh = np.triu(
          np.array(np.meshgrid(values[:, 0], values[:, 0])), k=1
        ).T.reshape(-1, 2)
        eps_mesh = np.triu(
          np.array(np.meshgrid(values[:, 1], values[:, 1])), k=1
        ).T.reshape(-1, 2)
        count_mesh = np.triu(
          np.array(np.meshgrid(count, count)), k=1
        ).T.reshape(-1, 2)

        sig_c = 0.5 * np.sum(sig_mesh, axis=1)
        eps_c = np.sqrt(eps_mesh[:, 0] * eps_mesh[:, 1])
        count_c = np.prod(count_mesh, axis=1)

        sigma2 = sig_c * sig_c
        sigma6 = sigma2 * sigma2 * sigma2

        sum1 = sum1 + np.sum(count_c * eps_c * sigma6 * sigma6)
        sum2 = sum2 + np.sum(count_c * eps_c * sigma6)

        sum3 = 0.0

        denom = (n_atoms * (n_atoms + 1)) / 2
        sum1 = sum1 / denom
        sum2 = sum2 / denom
        sum3 = sum3 / denom

        disp_coef = (
          8
          * n_atoms
          * n_atoms
          * np.pi
          * (
            sum1 / (9 * np.power(r_cut, 9))
            - sum2 / (3 * np.power(r_cut, 3))
            + sum3
          )
        )
    elif isinstance(force, openmm.CMAPTorsionForce):
      num_maps = force.getNumMaps()
      num_cmap_torsions = force.getNumTorsions()
      map_size = force.getMapParameters(0)[0]

      # Extract energy maps
      cmap_maps_arr = np.zeros((num_maps, map_size, map_size))
      for idx in range(num_maps):
        size, energies = force.getMapParameters(idx)
        if size != map_size:
          raise ValueError('All CMAP maps must have the same size')
        cmap_maps_arr[idx] = np.array(
          energies.value_in_unit(unit.kilocalorie_per_mole), dtype=wp_float
        ).reshape((size, size))

      # Extract map indices and phi/psi pairs
      cmap_map_id_arr = np.zeros((num_cmap_torsions,), dtype=wp_int)
      cmap_atoms_arr = np.zeros((num_cmap_torsions, 8), dtype=wp_int)
      for t in range(num_cmap_torsions):
        (
          map_idx,
          a1,
          a2,
          a3,
          a4,
          b1,
          b2,
          b3,
          b4,
        ) = force.getTorsionParameters(t)
        cmap_map_id_arr[t] = map_idx
        cmap_atoms_arr[t] = (a1, a2, a3, a4, b1, b2, b3, b4)
    elif isinstance(force, openmm.CustomTorsionForce):
      energy_function = force.getEnergyFunction()
      # TODO probably a better way to search for custom force definitions
      if 'min(dtheta' not in energy_function or 'theta0' not in energy_function:
        raise NotImplementedError(
          f'Unsupported CustomTorsionForce energy function: {energy_function}'
        )

      num_impropers = force.getNumTorsions()
      impropers_arr = np.zeros((num_impropers, 4), dtype=wp_int)
      improper_k = np.zeros((num_impropers,), dtype=wp_float)
      improper_theta0 = np.zeros((num_impropers,), dtype=wp_float)
      for idx in range(num_impropers):
        i, j, k, l, prm = force.getTorsionParameters(idx)
        k_val, theta0 = prm
        impropers_arr[idx] = (i, j, k, l)
        improper_k[idx] = k_val * _KJ_TO_KCAL
        improper_theta0[idx] = theta0
    elif isinstance(force, openmm.CustomNonbondedForce):
      energy_function = force.getEnergyFunction()
      if (
        'acoef(type1, type2)' not in energy_function
        or 'bcoef(type1, type2)' not in energy_function
      ):
        raise NotImplementedError(
          f'Unsupported CustomTorsionForce energy function: {energy_function}'
        )

      # Extract relevant options from force object
      # TODO there should probably be better error handling if there
      # is a mismatch between the normal and custom nb forces for
      # cutoff, switching, or lrc
      if force.getUseSwitchingFunction():
        r_switch = force.getSwitchingDistance().value_in_unit(unit.angstrom)
      if force.getNonbondedMethod() is not openmm.NonbondedForce.NoCutoff:
        if r_cut is None:
          r_cut = force.getCutoffDistance().value_in_unit(unit.angstrom)
      if force.getUseLongRangeCorrection():
        raise NotImplementedError(
          'General long range corrections for CustomNonbondedForce is not yet supported'
        )

      # Extract per-particle type indices
      num_particles = force.getNumParticles()
      nbfix_atom_type_arr = np.zeros((num_particles,), dtype=wp_int)
      for idx in range(num_particles):
        per_particle = force.getParticleParameters(idx)
        nbfix_atom_type_arr[idx] = per_particle[0]

      # Extract tabulated function tables
      for fi in range(force.getNumTabulatedFunctions()):
        name = force.getTabulatedFunctionName(fi)
        fn = force.getTabulatedFunction(fi)
        xsize, ysize, values = fn.getFunctionParameters()
        mat = np.asarray(values, dtype=wp_float).reshape(
          (int(xsize), int(ysize)), order='F'
        )
        if name == 'acoef':
          nbfix_acoef_table = mat * (10**6) * np.sqrt(_KJ_TO_KCAL)
        else:
          nbfix_bcoef_table = mat * (10**6) * _KJ_TO_KCAL
    elif isinstance(force, openmm.CMMotionRemover):
      # TODO decide whether to add this as a parsed option or ignore it.
      pass
    else:
      raise NotImplementedError(f'{force} is not yet supported.')

  # Extract holonomic distance constraints.
  #
  # OpenMM represents all constraints (including rigid water) as explicit
  # pairwise distance constraints stored on the System.
  #

  # TODO refactor this a bit to use dummy arrays
  num_constraints = system.getNumConstraints()
  if num_constraints > 0:
    c_idx = []
    c_dist = []
    for ci in range(num_constraints):
      i, j, dist = system.getConstraintParameters(ci)
      i = int(i)
      j = int(j)
      if masses[i] == 0.0 and masses[j] == 0.0:
        continue
      dist_ang = float(dist.value_in_unit(unit.angstrom))
      c_idx.append((i, j))
      c_dist.append(dist_ang)
    if c_idx:
      constraint_idx_arr = np.asarray(c_idx, dtype=wp_int)
      constraint_dist_arr = np.asarray(c_dist, dtype=wp_float)
    else:
      constraint_idx_arr = np.zeros((0, 2), dtype=wp_int)
      constraint_dist_arr = np.zeros((0,), dtype=wp_float)

  # TODO it's probably best to add explicit type checking to these structures
  # either here or in the energy generator function
  topology = Topology(
    n_atoms=n_atoms,
    bonds=jnp.asarray(bonds_arr),
    angles=jnp.asarray(angles_arr),
    torsions=jnp.asarray(torsions_arr),
    impropers=jnp.asarray(impropers_arr),
    exclusion_mask=None,
    pair_14_mask=None,
    molecule_id=None,
    cmap_atoms=jnp.asarray(cmap_atoms_arr),
    cmap_map_idx=jnp.asarray(cmap_map_id_arr),
    exc_pairs=jnp.asarray(exc_pairs_arr),
    nbfix_atom_type=jnp.asarray(nbfix_atom_type_arr),
  )

  bonded = BondedParameters(
    bond_k=jnp.asarray(bond_k),
    bond_r0=jnp.asarray(bond_r0),
    angle_k=jnp.asarray(angle_k),
    angle_theta0=jnp.asarray(angle_theta0),
    torsion_k=jnp.asarray(torsion_k),
    torsion_n=jnp.asarray(torsion_n),
    torsion_gamma=jnp.asarray(torsion_phase),
    improper_k=jnp.asarray(improper_k),
    improper_n=None,
    improper_gamma=jnp.asarray(improper_theta0),
    cmap_maps=jnp.asarray(cmap_maps_arr),
  )

  nonbonded = NonbondedParameters(
    charges=jnp.asarray(charges_arr),
    sigma=jnp.asarray(sigma_arr),
    epsilon=jnp.asarray(epsilon_arr),
    exc_charge_prod=jnp.asarray(exc_charge_prod_arr),
    exc_sigma=jnp.asarray(exc_sigma_arr),
    exc_epsilon=jnp.asarray(exc_epsilon_arr),
    nbfix_acoef=jnp.asarray(nbfix_acoef_table),
    nbfix_bcoef=jnp.asarray(nbfix_bcoef_table),
  )

  params = Parameters(bonded=bonded, nonbonded=nonbonded)

  # TODO jax types?
  nb_options = NonbondedOptions(
    r_cut=r_cut,
    dr_threshold=dr_threshold,
    use_soft_lj=use_soft_lj,
    lj_cap=lj_cap,
    use_shift_lj=use_shift_lj,
    scale_14_lj=scale_14_lj,
    scale_14_coul=scale_14_coul,
    use_pbc=use_pbc,
    nb_format=format,
    use_periodic_general=use_periodic_general,
    fractional_coordinates=fractional_coordinates,
    disp_coef=disp_coef,
    r_switch=r_switch,
  )

  return OpenMMSystem(
    jnp.asarray(pos, dtype=jnp.float64),
    topology,
    params,
    jnp.asarray(box_vectors, dtype=jnp.float64)
    if box_vectors is not None
    else None,
    nb_options,
    jnp.asarray(recip_alpha, dtype=jnp.float64)
    if recip_alpha is not None
    else None,
    jnp.asarray(recip_grid, dtype=jnp.int32)
    if recip_grid is not None
    else None,
    jnp.asarray(ewald_error_tolerance, dtype=jnp.float64)
    if ewald_error_tolerance is not None
    else None,
    jnp.asarray(masses, dtype=jnp.float64),
    VirtualSiteData(
      is_virtual_site=jnp.asarray(is_virtual_site),
      vsite_type=jnp.asarray(vsite_type, dtype=jnp.int32),
      parent_idx=jnp.asarray(vsite_parent_idx, dtype=jnp.int32),
      weights=jnp.asarray(vsite_weights, dtype=jnp.float64),
      origin_weights=jnp.asarray(vsite_origin_weights, dtype=jnp.float64),
      x_weights=jnp.asarray(vsite_x_weights, dtype=jnp.float64),
      y_weights=jnp.asarray(vsite_y_weights, dtype=jnp.float64),
      local_position=jnp.asarray(vsite_local_position, dtype=jnp.float64),
    ),
    jnp.asarray(constraint_idx_arr, dtype=jnp.int32),
    jnp.asarray(constraint_dist_arr, dtype=jnp.float64),
  )


def get_ewald_parameters() -> None:
  return None


def virtual_site_apply_positions(
  pos: Array,
  virtual_sites: Optional[VirtualSiteData],
  displacement_fn: Any,
  shift_fn: Any,
  box: Optional[Array] = None,
  use_periodic_general: bool = False,
) -> Array:
  """Overwrite virtual-site particle positions from parent atoms.

  This is intended as a lightweight, JAX-friendly helper for test-time MD loops.
  It supports the fixed-shape representation produced by convert_openmm_system
  in mm.virtual_sites.

  Notes:
    - This operates in the same coordinate system as pos (Å or fractional).
    - When use_periodic_general=True, this assumes the displacement_fn and
      shift_fn follow space.periodic_general semantics: positions may be
      fractional, but displacements are in real space.
    - box is passed through to space.periodic_general via box=... so the
      caller does not need to manage box kwargs.
  """
  if virtual_sites is None:
    return pos

  box_kwargs = {'box': box} if use_periodic_general else {}

  is_vs = virtual_sites.is_virtual_site
  # Fast path: no virtual sites.
  if isinstance(is_vs, (bool, np.bool_)):
    return pos

  # parent_idx padded with -1; clip for safe gathers (weights for unused parents
  # are expected to be 0).
  parent_idx = jnp.clip(virtual_sites.parent_idx, 0, pos.shape[0] - 1)
  p0 = pos[parent_idx[:, 0]]
  p1 = pos[parent_idx[:, 1]]
  p2 = pos[parent_idx[:, 2]]

  t = virtual_sites.vsite_type
  out = pos

  # Use displacements to ensure minimum-image consistency for wrapped fractional
  # coordinates (e.g., triclinic cells with fractional_coordinates=True).
  # TODO canonicalize?
  disp_fn = jax.vmap(partial(displacement_fn, **box_kwargs), 0, 0)
  d01 = disp_fn(p1, p0)
  d02 = disp_fn(p2, p0)

  # TwoParticleAverageSite: r = p0 + w1*(p1 - p0)
  w = virtual_sites.weights
  r_two = shift_fn(p0, w[:, 1:2] * d01, **box_kwargs)
  out = jnp.where((t == 1)[:, None], r_two, out)

  # ThreeParticleAverageSite: r = p0 + w1*(p1 - p0) + w2*(p2 - p0)
  r_three = shift_fn(p0, w[:, 1:2] * d01 + w[:, 2:3] * d02, **box_kwargs)
  out = jnp.where((t == 2)[:, None], r_three, out)

  # OutOfPlaneSite:
  #   r = p0 + w12*d01 + w13*d02 + wC*cross(d01, d02)
  r_oop = shift_fn(
    p0,
    w[:, 0:1] * d01 + w[:, 1:2] * d02 + w[:, 2:3] * jnp.cross(d01, d02),
    **box_kwargs,
  )
  out = jnp.where((t == 3)[:, None], r_oop, out)

  # LocalCoordinatesSite:
  #   origin = sum_i origin_w[i]*p_i
  #   xdef = sum_i x_w[i]*p_i
  #   ydef = sum_i y_w[i]*p_i
  #   xdir = xdef - origin
  #   ydir = ydef - origin
  #   xhat = normalize(xdir)
  #   yhat = normalize(ydir - xhat*dot(ydir,xhat))
  #   zhat = cross(xhat, yhat)
  #   r = origin + local.x*xhat + local.y*yhat + local.z*zhat
  #
  # Formed using minimum-image displacements from p0 so the resulting
  # frame is stable even if positions are stored wrapped in fractional coords.
  ow = virtual_sites.origin_weights
  xw = virtual_sites.x_weights
  yw = virtual_sites.y_weights

  # Assumes each weight triple sums to 1 (typical for OpenMM LocalCoordinatesSite)
  origin_delta = ow[:, 1:2] * d01 + ow[:, 2:3] * d02
  xdef_delta = xw[:, 1:2] * d01 + xw[:, 2:3] * d02
  ydef_delta = yw[:, 1:2] * d01 + yw[:, 2:3] * d02
  xdir = xdef_delta - origin_delta
  ydir = ydef_delta - origin_delta

  eps = jnp.asarray(1e-12, dtype=pos.dtype)
  xnorm = jnp.sqrt(jnp.sum(xdir * xdir, axis=1, keepdims=True) + eps)
  xhat = xdir / xnorm
  yproj = jnp.sum(ydir * xhat, axis=1, keepdims=True)
  yperp = ydir - yproj * xhat
  ynorm = jnp.sqrt(jnp.sum(yperp * yperp, axis=1, keepdims=True) + eps)
  yhat = yperp / ynorm
  zhat = jnp.cross(xhat, yhat)

  local = virtual_sites.local_position
  r_local = (
    origin_delta
    + local[:, 0:1] * xhat
    + local[:, 1:2] * yhat
    + local[:, 2:3] * zhat
  )
  r_local = shift_fn(p0, r_local, **box_kwargs)
  out = jnp.where((t == 4)[:, None], r_local, out)

  # Only overwrite virtual-site entries
  return jnp.where(is_vs[:, None], out, pos)


def virtual_site_fix_state(
  state: simulate.NVEState,
  virtual_sites: Optional[VirtualSiteData],
  displacement_fn: Any,
  shift_fn: Any,
  box: Optional[Array] = None,
  use_periodic_general: bool = False,
) -> simulate.NVEState:
  """Make an NVEState safe to integrate with massless virtual sites.

  - Ensures virtual site masses are nonzero (avoid division by 0 in position_step).
  - Zeros momentum/force for virtual sites (they are not independent DOFs).
  - Updates virtual site positions from their parent atoms.
  """
  if virtual_sites is None:
    return state
  is_vs = virtual_sites.is_virtual_site

  pos = virtual_site_apply_positions(
    state.position,
    virtual_sites,
    displacement_fn=displacement_fn,
    shift_fn=shift_fn,
    box=box,
    use_periodic_general=use_periodic_general,
  )

  # simulate.nve canonicalizes mass to shape (N, 1) for broadcasting.
  mass = state.mass
  if hasattr(mass, 'ndim') and mass.ndim == 2:
    mask = is_vs[:, None]
  else:
    mask = is_vs

  # Any positive dummy mass works here since we zero momentum for virtual sites.
  safe_mass = jnp.where(mask, jnp.ones_like(mass), mass)
  safe_momentum = jnp.where(
    mask, jnp.zeros_like(state.momentum), state.momentum
  )
  safe_force = jnp.where(mask, jnp.zeros_like(state.force), state.force)

  return state.set(
    position=pos, mass=safe_mass, momentum=safe_momentum, force=safe_force
  )
