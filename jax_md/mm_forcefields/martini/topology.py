"""Utilities for parsing Martini topologies.
Parts adapted from OpenMM's gromacstopfile.py. CMAP implementation based on
OpenMM CMAPTorsionForceImpl.

This module parses a :class:`GromacsTopFile` into a
:class:`MartiniTopology`, which stores the precomputed arrays required by
the force field and energy functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import jax.numpy as jnp

from jax_md.util import Array

from jax_md.mm_forcefields.martini.top_file_parser import (
  Angle,
  GromacsTopFile,
  Atom,
)
from jax_md.mm_forcefields.martini.vsites import _parse_vsites_from_topology


def _arr(lst, dtype=jnp.float64):
  """Convert a list to a JAX array, returning an empty array if the list is falsy.

  Args:
    lst: Input list to convert.
    dtype: Desired JAX dtype for the output array.

  Returns:
    JAX array of the given dtype, or a zero-length array if lst is falsy.
  """
  return jnp.array(lst, dtype=dtype) if lst else jnp.zeros(0)


def _idx(lst):
  """Convert a list to a JAX int32 index array, returning None if the list is falsy.

  Args:
    lst: Input list to convert.

  Returns:
    JAX int32 array, or None if lst is falsy.
  """
  return jnp.array(lst, dtype=jnp.int32) if lst else None


def deg_to_rad():
  """Return the conversion factor from degrees to radians (π / 180).

  Returns:
    Scalar conversion factor as a JAX float.
  """
  return jnp.pi / 180.0


@dataclass
class MartiniTopology:
  """Precomputed topology data for a Martini system.

  This dataclass stores all bonded, nonbonded, CMAP, and virtual-site
  parameters needed to evaluate the system energy. It is constructed by
  :func:`create_topology` and consumed by
  :func:`jax_md.mm_forcefields.martini.energy.energy_fn`.

  Attributes
  ----------
  atoms : list of Atom
    Flat list of all atoms (all molecule copies).
  masses : Array, shape (N,)
    Per-atom masses in a.m.u.

  Bond arrays
  -----------
  bond_indices : Array, shape (B, 2), B = Number of Bonds
  bond_rs : Array, shape (B,)  — equilibrium distances [nm]
  bond_ks : Array, shape (B,)  — force constants [kJ/mol/nm²]

  Angle arrays (harmonic)
  -----------------------
  harm_angle_indices : Array, shape (A, 3), A = Number of Harmonic Angles
  harm_angle_theta0s : Array, shape (A,)   — [rad]
  harm_angle_ks : Array, shape (A,)   - [kJ/mol/rad^2]

  Angle arrays (G96)
  ------------------
  g96_angle_indices : Array, shape (A, 3), A = Number of G96 Angles
  g96_angle_cos_theta0s : Array, shape (A,)
  g96_angle_ks : Array, shape (A,)  - [kJ/mol]

  Angle arrays (restricted)
  -------------------------
  rest_angle_indices : Array, shape (A, 3), A = Number of Restricted Angles
  rest_angle_cos_theta0s : Array, shape (A,)
  rest_angle_ks : Array, shape (A,)   - [kJ/mol]

  Dihedral arrays (periodic)
  --------------------------
  per_dihedral_indices : Array, shape (D, 4), D = Number of Periodic Dihedrals
  per_dihedral_ns : Array, shape (D,)
  per_dihedral_ks : Array, shape (D,) - [kJ/mol]
  per_dihedral_phi0s : Array, shape (D,)   — [rad]

  Dihedral arrays (harmonic/improper)
  ------------------------------------
  harm_dihedral_indices : Array, shape (D, 4), D = Number of harmonic/improper Dihedrals
  harm_dihedral_ks : Array, shape (D,)  - [kJ/mol/rad^2]
  harm_dihedral_phi0s : Array, shape (D,)  — [rad]

  Dihedral arrays (Ryckaert-Bellemans / Fourier), D = Number of RB/Fourier Dihedrals
  -----------------------------------------------
  rb_dihedral_indices : Array, shape (D, 4)
  rb_dihedral_cs : Array, shape (D, 6)  - [kJ/mol]

  Dihedral arrays (CBT), D = Number of CBT Dihedrals
  ----------------------
  cbt_indices : Array, shape (D, 4)
  cbt_a_coeffs : Array, shape (D, 5)
  cbt_ks : Array, shape (D,)  - [kJ/mol]

  CMAP arrays
  -----------
  cmap_indices : Array, shape (C, 5)
  cmap_map_ids : Array, shape (C,)
  cmap_grids : Array, shape (M, S, S)
  cmap_coeffs : Array, shape (M, S, S, 16)

  Non-bonded / LJ arrays
  ----------------------
  n_atoms : int
  atom_type_indices : Array, shape (N,)
  charges : Array, shape (N,) - [electron]
  C6_table : Array, shape (T, T), T = Number of Types
  C12_table : Array, shape (T, T)
  exclusion_pairs : Array, shape (E, 2), E = Number of Exclusion Pairs
  excl_mask : Array, shape (N, N)  — boolean
  exception_pairs : Array, shape (X, 2), X = Number of Exception Pairs
  exception_q : Array, shape (X,)
  exception_C6 : Array, shape (X,)
  exception_C12 : Array, shape (X,)
  epsilon_r : float
  r_cut : float   - [nm]

  Virtual-site callable
  ----------------------
  apply_vsites : Callable
  """

  # Meta
  atoms: List[Atom]
  masses: Array

  # Bonds
  bond_indices: Array
  bond_rs: Array
  bond_ks: Array

  # Harmonic angles
  harm_angle_indices: Array
  harm_angle_theta0s: Array
  harm_angle_ks: Array

  # G96 angles
  g96_angle_indices: Array
  g96_angle_cos_theta0s: Array
  g96_angle_ks: Array

  # Restricted angles
  rest_angle_indices: Array
  rest_angle_cos_theta0s: Array
  rest_angle_ks: Array

  # Periodic dihedrals
  per_dihedral_indices: Array
  per_dihedral_ns: Array
  per_dihedral_ks: Array
  per_dihedral_phi0s: Array

  # Harmonic (improper) dihedrals
  harm_dihedral_indices: Array
  harm_dihedral_ks: Array
  harm_dihedral_phi0s: Array

  # RB / Fourier dihedrals
  rb_dihedral_indices: Array
  rb_dihedral_cs: Array

  # CBT dihedrals
  cbt_indices: Array
  cbt_a_coeffs: Array
  cbt_ks: Array

  # CMAP
  cmap_indices: Array
  cmap_map_ids: Array
  cmap_grids: Array
  cmap_coeffs: Array

  # Non-bonded
  n_atoms: int
  atom_type_indices: Array
  charges: Array
  C6_table: Array
  C12_table: Array
  exclusion_pairs: Array
  excl_mask: Array
  exception_pairs: Array
  exception_q: Array
  exception_C6: Array
  exception_C12: Array
  epsilon_r: float
  r_cut: float

  # Virtual sites
  apply_vsites: Callable


def create_topology(
  top_file: GromacsTopFile,
  nonbonded_cutoff: float = 1.1,
  epsilon_r: float = 15.0,
) -> MartiniTopology:
  """Create a MartiniTopology from a parsed GROMACS topology.

  Args:
    top_file: Parsed GROMACS topology.
    nonbonded_cutoff: Nonbonded cutoff distance in nm.
    epsilon_r: Reaction field dielectric constant.

  Returns:
    A MartiniTopology containing all precomputed parameters needed for
    energy evaluation.
  """
  atoms, masses = _build_atom_and_mass_list(top_file)
  all_bonded_types_mol, all_atom_types_mol = _get_all_bonded_types_mol(top_file)
  bond_indices, bond_rs, bond_ks = _parse_bonds(top_file, all_bonded_types_mol)
  (
    harm_angle_indices,
    harm_angle_theta0s,
    harm_angle_ks,
    g96_angle_indices,
    g96_angle_cos_theta0s,
    g96_angle_ks,
    rest_angle_indices,
    rest_angle_cos_theta0s,
    rest_angle_ks,
  ) = _parse_angles(top_file, all_bonded_types_mol)
  (
    per_dihedral_indices,
    per_dihedral_ns,
    per_dihedral_ks,
    per_dihedral_phi0s,
    harm_dihedral_indices,
    harm_dihedral_ks,
    harm_dihedral_phi0s,
    rb_dihedral_indices,
    rb_dihedral_cs,
    cbt_indices,
    cbt_a_coeffs,
    cbt_ks,
  ) = _parse_dihedrals(top_file, all_bonded_types_mol)
  (
    cmap_indices,
    cmap_map_ids,
    cmap_grids,
    cmap_coeffs,
  ) = _parse_cmaps(top_file, all_bonded_types_mol)
  # Virtual-site parsing also collects excluded pairs.
  excl_pairs: List[Tuple[int, int]] = []
  apply_vsites, excl_pairs = _parse_vsites_from_topology(top_file, excl_pairs)
  (
    n_atoms,
    atom_type_indices,
    charges,
    C6_table,
    C12_table,
    exclusion_pairs,
    excl_mask,
    exception_pairs,
    exception_q,
    exception_C6,
    exception_C12,
  ) = _parse_lj(top_file, all_atom_types_mol, excl_pairs)

  return MartiniTopology(
    atoms=atoms,
    masses=masses,
    # bonds
    bond_indices=bond_indices,
    bond_rs=bond_rs,
    bond_ks=bond_ks,
    # harmonic angles
    harm_angle_indices=harm_angle_indices,
    harm_angle_theta0s=harm_angle_theta0s,
    harm_angle_ks=harm_angle_ks,
    # G96 angles
    g96_angle_indices=g96_angle_indices,
    g96_angle_cos_theta0s=g96_angle_cos_theta0s,
    g96_angle_ks=g96_angle_ks,
    # restricted angles
    rest_angle_indices=rest_angle_indices,
    rest_angle_cos_theta0s=rest_angle_cos_theta0s,
    rest_angle_ks=rest_angle_ks,
    # periodic dihedrals
    per_dihedral_indices=per_dihedral_indices,
    per_dihedral_ns=per_dihedral_ns,
    per_dihedral_ks=per_dihedral_ks,
    per_dihedral_phi0s=per_dihedral_phi0s,
    # harmonic dihedrals
    harm_dihedral_indices=harm_dihedral_indices,
    harm_dihedral_ks=harm_dihedral_ks,
    harm_dihedral_phi0s=harm_dihedral_phi0s,
    # RB dihedrals
    rb_dihedral_indices=rb_dihedral_indices,
    rb_dihedral_cs=rb_dihedral_cs,
    # CBT dihedrals
    cbt_indices=cbt_indices,
    cbt_a_coeffs=cbt_a_coeffs,
    cbt_ks=cbt_ks,
    # CMAP
    cmap_indices=cmap_indices,
    cmap_map_ids=cmap_map_ids,
    cmap_grids=cmap_grids,
    cmap_coeffs=cmap_coeffs,
    # non-bonded
    n_atoms=n_atoms,
    atom_type_indices=atom_type_indices,
    charges=charges,
    C6_table=C6_table,
    C12_table=C12_table,
    exclusion_pairs=exclusion_pairs,
    excl_mask=excl_mask,
    exception_pairs=exception_pairs,
    exception_q=exception_q,
    exception_C6=exception_C6,
    exception_C12=exception_C12,
    epsilon_r=epsilon_r,
    r_cut=nonbonded_cutoff,
    # vsites
    apply_vsites=apply_vsites,
  )


def _build_atom_and_mass_list(
  top: GromacsTopFile,
) -> Tuple[List[Atom], Array]:
  """Build the flattened atom and mass arrays.

  Args:
    top: Parsed GROMACS topology.

  Returns:
    Tuple containing the flattened atom list and per-atom masses.
  """
  masses: List[float] = []
  atoms: List[Atom] = []
  for mol in top._molecules:
    mol_type = top._moleculeTypes[mol.name]
    for _ in range(mol.count):
      for atom in mol_type.atoms:
        atom_type = top._atomTypes[atom.type]
        mass = atom.mass if atom.mass is not None else atom_type.mass
        masses.append(mass)
        atoms.append(atom)
  return atoms, jnp.array(masses)


def _get_all_bonded_types_mol(
  top_file: GromacsTopFile,
) -> Tuple[List[List[str]], List[List[str]]]:
  """Get list of atom types and their bonded tyes for each molecule.

  Args:
    top_file: Parsed GROMACS topology.

  Returns:
    Tuple containing the atom types and bonded atom types for
    each molecule.
  """
  all_bonded_types_mol = []
  all_atom_types_mol = []
  for mol in top_file._molecules:
    mol_type = top_file._moleculeTypes[mol.name]
    atom_types_mol = [atom.type for atom in mol_type.atoms]
    try:
      bonded_types_mol = [
        top_file._atomTypes[t].bonded_type for t in atom_types_mol
      ]
    except KeyError as e:
      raise ValueError('Unknown atom type: ' + str(e))
    bonded_types_mol = [
      b if b is not None else a
      for a, b in zip(atom_types_mol, bonded_types_mol)
    ]
    all_bonded_types_mol.append(bonded_types_mol)
    all_atom_types_mol.append(atom_types_mol)
  return all_bonded_types_mol, all_atom_types_mol


def _parse_bonds(
  top: GromacsTopFile,
  all_bonded_types_mol: List[List[str]],
) -> Tuple[Array, Array, Array]:
  """Parse bond parameters from the topology.

  Args:
    top: Parsed GROMACS topology.
    all_bonded_types_mol: Bonded atom types for each molecule.

  Returns:
    Tuple containing the bond indices, equilibrium bond lengths, and
    force constants.
  """
  bond_idx: List[List[int]] = []
  bond_l: List[float] = []
  bond_k: List[float] = []

  base_atom_index = 0
  for mol_idx, mol in enumerate(top._molecules):
    mol_type = top._moleculeTypes[mol.name]
    n_atoms = len(mol_type.atoms)
    bonded_types_mol = all_bonded_types_mol[mol_idx]

    for _ in range(mol.count):
      for bond in mol_type.bonds:
        atom_types = tuple(bonded_types_mol[i] for i in bond.atoms)
        types_rev = atom_types[::-1] + (bond.func_type,)
        types = atom_types + (bond.func_type,)

        if bond.params is not None:
          b0, k = bond.params.b0, bond.params.k
        elif types in top._bondTypes:
          bt = top._bondTypes[types]
          b0, k = bt.b0, bt.k
        elif types_rev in top._bondTypes:
          bt = top._bondTypes[types_rev]
          b0, k = bt.b0, bt.k
        else:
          raise ValueError(
            f'No bond params for atoms {bond.atoms[0]}, {bond.atoms[1]}'
          )

        bond_idx.append(
          [base_atom_index + bond.atoms[0], base_atom_index + bond.atoms[1]]
        )
        bond_l.append(b0)
        bond_k.append(k)

      base_atom_index += n_atoms

  return (
    _idx(bond_idx) if bond_idx else jnp.zeros((0, 2), dtype=jnp.int32),
    _arr(bond_l),
    _arr(bond_k),
  )


def _parse_angles(
  top: GromacsTopFile,
  all_bonded_types_mol: List[List[str]],
):
  """Parse angle parameters from the topology.

  Args:
    top: Parsed GROMACS topology.
    all_bonded_types_mol: Bonded atom types for each molecule.

  Returns:
    Tuple containing the harmonic, G96, and restricted angle indices
    and parameters.
  """
  base_atom_index = 0

  harm_ang_idx: List[List[int]] = []
  harm_ang_th: List[float] = []
  harm_ang_k: List[float] = []

  g96_ang_idx: List[List[int]] = []
  g96_ang_cos_th: List[float] = []
  g96_ang_k: List[float] = []

  rest_ang_idx: List[List[int]] = []
  rest_ang_cos_th: List[float] = []
  rest_ang_k: List[float] = []

  for mol_idx, mol in enumerate(top._molecules):
    mol_type = top._moleculeTypes[mol.name]
    n_atoms = len(mol_type.atoms)
    bonded_types_mol = all_bonded_types_mol[mol_idx]

    for _ in range(mol.count):
      for angle in mol_type.angles:
        if angle.func_type == '1':
          atms, th, k = _get_angle_params(top, angle, bonded_types_mol)
          harm_ang_idx.append([base_atom_index + a for a in atms])
          harm_ang_th.append(th)
          harm_ang_k.append(k)
        elif angle.func_type == '2':
          atms, th, k = _get_angle_params(top, angle, bonded_types_mol)
          g96_ang_idx.append([base_atom_index + a for a in atms])
          g96_ang_cos_th.append(np.cos(th))
          g96_ang_k.append(k)
        elif angle.func_type == '10':
          atms, th, k = _get_angle_params(top, angle, bonded_types_mol)
          rest_ang_idx.append([base_atom_index + a for a in atms])
          rest_ang_cos_th.append(np.cos(th))
          rest_ang_k.append(k)

      base_atom_index += n_atoms

  return (
    _idx(harm_ang_idx) if harm_ang_idx else jnp.zeros((0, 3), dtype=jnp.int32),
    _arr(harm_ang_th),
    _arr(harm_ang_k),
    _idx(g96_ang_idx) if g96_ang_idx else jnp.zeros((0, 3), dtype=jnp.int32),
    _arr(g96_ang_cos_th),
    _arr(g96_ang_k),
    _idx(rest_ang_idx) if rest_ang_idx else jnp.zeros((0, 3), dtype=jnp.int32),
    _arr(rest_ang_cos_th),
    _arr(rest_ang_k),
  )


def _parse_dihedrals(
  top: GromacsTopFile,
  all_bonded_types_mol: List[List[str]],
):
  """Parse dihedral parameters from the topology.

  Args:
    top: Parsed GROMACS topology.
    all_bonded_types_mol: Bonded atom types for each molecule.

  Returns:
    Tuple containing the periodic, harmonic, Ryckaert-Bellemans, and
    combined bending-torsion dihedral parameters.
  """
  dihedral_type_table, wildcard_dihedral_types = _build_dihedral_table(top)

  per_idx: List[List[int]] = []
  per_phi0: List[float] = []
  per_k: List[float] = []
  per_n: List[int] = []

  harm_idx: List[List[int]] = []
  harm_phi0: List[float] = []
  harm_k: List[float] = []

  rb_idx: List[List[int]] = []
  rb_cs: List[List[float]] = []

  cbt_idx: List[List[int]] = []
  cbt_k: List[float] = []
  cbt_a: List[List[float]] = []

  base_atom_index = 0
  for mol_idx, mol in enumerate(top._molecules):
    mol_type = top._moleculeTypes[mol.name]
    n_atoms = len(mol_type.atoms)
    bonded_types_mol = all_bonded_types_mol[mol_idx]

    for _ in range(mol.count):
      for dihedral in mol_type.dihedrals:
        atoms = dihedral.atoms
        dihedral_type = dihedral.func_type

        params_list = _get_dihedral_params(
          top,
          dihedral,
          bonded_types_mol,
          dihedral_type_table,
          wildcard_dihedral_types,
        )

        for prm in params_list:
          idxs = [base_atom_index + a for a in atoms]
          if dihedral_type in ('1', '4', '9'):
            k = prm.k
            if k != 0:
              per_idx.append(idxs)
              per_phi0.append(prm.phi0 * deg_to_rad())
              per_k.append(k)
              per_n.append(prm.n)
          elif dihedral_type == '2':
            k = prm.k
            if k != 0:
              phi0 = prm.phi0
              phi0 = phi0 - 360 if phi0 > 180 else phi0
              harm_idx.append(idxs)
              harm_phi0.append(phi0 * deg_to_rad())
              harm_k.append(k)
          elif dihedral_type == '11':
            cbt_idx.append(idxs)
            cbt_k.append(prm.k)
            cbt_a.append(prm.a_array)
          elif dihedral_type in ('3', '5'):
            c = prm.c_array
            if any(x != 0 for x in c):
              if dihedral_type == '5':
                c = [
                  c[1] + 0.5 * (c[0] + c[2]),
                  0.5 * (-c[0] + 3 * c[2]),
                  -c[1] + 4 * c[3],
                  -2 * c[2],
                  -4 * c[3],
                  0.0,
                ]
              rb_idx.append(idxs)
              rb_cs.append(c)

      base_atom_index += n_atoms

  return (
    _idx(per_idx) if per_idx else jnp.zeros((0, 4), dtype=jnp.int32),
    jnp.array(per_n, dtype=jnp.int32)
    if per_n
    else jnp.zeros(0, dtype=jnp.int32),
    _arr(per_k),
    _arr(per_phi0),
    _idx(harm_idx) if harm_idx else jnp.zeros((0, 4), dtype=jnp.int32),
    _arr(harm_k),
    _arr(harm_phi0),
    _idx(rb_idx) if rb_idx else jnp.zeros((0, 4), dtype=jnp.int32),
    jnp.array(rb_cs, dtype=jnp.float64) if rb_cs else jnp.zeros((0, 6)),
    _idx(cbt_idx) if cbt_idx else jnp.zeros((0, 4), dtype=jnp.int32),
    jnp.array(cbt_a, dtype=jnp.float64) if cbt_a else jnp.zeros((0, 5)),
    _arr(cbt_k),
  )


def _build_dihedral_table(
  top: GromacsTopFile,
) -> Tuple[Dict, List]:
  """Build lookup tables for dihedral parameter matching.

  Args:
    top: Parsed GROMACS topology.

  Returns:
    Tuple containing the dihedral lookup table and wildcard dihedral
    definitions.
  """
  dihedral_type_table: Dict = {}
  for key in top._dihedralTypes:
    if key[1] != 'X' and key[2] != 'X':
      if (key[1], key[2]) not in dihedral_type_table:
        dihedral_type_table[(key[1], key[2])] = []
      dihedral_type_table[(key[1], key[2])].append(key)
      if (key[2], key[1]) not in dihedral_type_table:
        dihedral_type_table[(key[2], key[1])] = []
      dihedral_type_table[(key[2], key[1])].append(key)

  wildcard_dihedral_types = []
  for key in top._dihedralTypes:
    if key[1] == 'X' or key[2] == 'X':
      wildcard_dihedral_types.append(key)
      for types in dihedral_type_table.values():
        types.append(key)

  return dihedral_type_table, wildcard_dihedral_types


def _parse_cmaps(
  top: GromacsTopFile,
  all_bonded_types_mol: List[List[str]],
) -> Tuple[Array, Array, Array, Array]:
  """Parse CMAP parameters from the topology.

  Args:
    top: Parsed GROMACS topology.
    all_bonded_types_mol: Bonded atom types for each molecule.

  Returns:
    Tuple containing the CMAP indices, map IDs, unique energy grids,
    and bicubic interpolation coefficients.
  """

  cmap_idx: List[List[int]] = []
  cmap_map_ids: List[int] = []
  cmap_grids_dict: Dict[tuple, int] = {}
  cmap_grid_list: List[np.ndarray] = []

  base_atom_index = 0
  for mol_idx, mol in enumerate(top._molecules):
    mol_type = top._moleculeTypes[mol.name]
    n_atoms = len(mol_type.atoms)
    bonded_types_mol = all_bonded_types_mol[mol_idx]

    for _ in range(mol.count):
      for cmap in mol_type.cmaps:
        atoms = cmap.atoms
        types = tuple(bonded_types_mol[i] for i in atoms)

        if cmap.params is not None:
          x_size, y_size, cmap_grid = cmap.x_size, cmap.y_size, cmap.grid
        elif types in top._cmapTypes:
          params = top._cmapTypes[types]
          x_size, y_size, cmap_grid = (
            params.x_size,
            params.y_size,
            params.grid,
          )
        elif types[::-1] in top._cmapTypes:
          params = top._cmapTypes[types[::-1]]
          x_size, y_size, cmap_grid = (
            params.x_size,
            params.y_size,
            params.grid,
          )
        else:
          raise ValueError(
            f'No parameters specified for cmap: '
            f'{cmap.atoms[0]}, {cmap.atoms[1]}, {cmap.atoms[2]}, {cmap.atoms[3]}, {cmap.atoms[4]}'
          )

        map_size = x_size
        if map_size != y_size:
          raise ValueError('Non-square CMAPs are not supported')

        grid = np.zeros((map_size, map_size), dtype=np.float64)
        for i in range(map_size):
          for j in range(map_size):
            grid[i, j] = float(
              cmap_grid[
                map_size * ((i + map_size // 2) % map_size)
                + ((j + map_size // 2) % map_size)
              ]
            )

        grid_key = tuple(grid.flatten())
        if grid_key not in cmap_grids_dict:
          cmap_grids_dict[grid_key] = len(cmap_grid_list)
          cmap_grid_list.append(grid)

        cmap_idx.append([base_atom_index + a for a in atoms])
        cmap_map_ids.append(cmap_grids_dict[grid_key])

      base_atom_index += n_atoms

  if cmap_grid_list:
    grids_array = jnp.array(np.stack(cmap_grid_list, axis=0), dtype=jnp.float64)
    coeffs = jnp.stack(
      [jnp.array(_compute_bicubic_coeffs(g)) for g in grids_array]
    )
  else:
    grids_array = jnp.zeros((0, 1, 1), dtype=jnp.float64)
    coeffs = jnp.zeros((0, 1, 1, 16), dtype=jnp.float64)

  return (
    _idx(cmap_idx) if cmap_idx else jnp.zeros((0, 5), dtype=jnp.int32),
    _idx(cmap_map_ids) if cmap_map_ids else jnp.zeros(0, dtype=jnp.int32),
    grids_array,
    coeffs,
  )


@staticmethod
def _compute_bicubic_coeffs(grid: np.ndarray) -> np.ndarray:
  """Compute bicubic interpolation coefficients for a periodic CMAP grid.
  Given a (S, S) periodic energy grid, return a (S, S, 16) array of
  bicubic spline coefficients matching OpenMM's calcMapDerivatives.
  Grid layout: grid[i, j] where i = phi index, j = psi index.

  Args:
    grid: Periodic CMAP energy grid with shape (S, S).

  Returns:
    Bicubic interpolation coefficients with shape (S, S, 16).
  """
  S = grid.shape[0]
  d1 = np.zeros((S, S))
  d2 = np.zeros((S, S))
  d12 = np.zeros((S, S))
  x = np.arange(S + 1) * (2 * np.pi / S)

  for i in range(S):
    y = np.append(grid[i, :], grid[i, 0])
    deriv = _create_periodic_spline(x, y)
    for j in range(S):
      d1[i, j] = _eval_spline_derivative(x, y, deriv, x[j])

  for j in range(S):
    y = np.append(grid[:, j], grid[0, j])
    deriv = _create_periodic_spline(x, y)
    for i in range(S):
      d2[i, j] = _eval_spline_derivative(x, y, deriv, x[i])

  for i in range(S):
    y = np.append(d2[i, :], d2[i, 0])
    deriv = _create_periodic_spline(x, y)
    for j in range(S):
      d12[i, j] = _eval_spline_derivative(x, y, deriv, x[j])

  wt = np.array(
    [
      [1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4],
      [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4],
      [0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4],
      [0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2],
      [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2],
      [0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2],
      [0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2],
      [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2],
      [0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1],
      [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1],
    ],
    dtype=np.float64,
  )

  delta = 2 * np.pi / S
  coeffs = np.zeros((S, S, 16))
  for i in range(S):
    for j in range(S):
      ni = (i + 1) % S
      nj = (j + 1) % S
      # OpenMM: energy[i + j*size], so flat index = i + j*S
      # matches our grid[i, j]
      e = [grid[j, i], grid[j, ni], grid[nj, ni], grid[nj, i]]
      e1 = [d1[j, i], d1[j, ni], d1[nj, ni], d1[nj, i]]
      e2 = [d2[j, i], d2[j, ni], d2[nj, ni], d2[nj, i]]
      e12 = [d12[j, i], d12[j, ni], d12[nj, ni], d12[nj, i]]
      rhs = np.array(
        e
        + [v * delta for v in e1]
        + [v * delta for v in e2]
        + [v * delta * delta for v in e12]
      )
      coeffs[i, j] = wt.T @ rhs
  return coeffs


def _create_periodic_spline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Construct a periodic cubic spline.

  Args:
    x: Grid point coordinates.
    y: Function values at the grid points.

  Returns:
    Periodic cubic spline.
  """
  n = y.size
  if n < 3:
    raise ValueError('Input array must have at least 3 points')
  if x.size != n:
    raise ValueError('x and y must have the same length')
  if not np.allclose(y[0], y[-1]):
    raise ValueError('y must be periodic with y[0] == y[-1]')

  a = np.zeros(n - 1)
  b = np.zeros(n - 1)
  c = np.zeros(n - 1)
  rhs = np.zeros(n - 1)

  a[0] = x[n - 1] - x[n - 2]
  b[0] = 2.0 * (x[1] - x[0] + x[n - 1] - x[n - 2])
  c[0] = x[1] - x[0]
  rhs[0] = 6.0 * (
    (y[1] - y[0]) / (x[1] - x[0])
    - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
  )

  for i in range(1, n - 1):
    a[i] = x[i] - x[i - 1]
    b[i] = 2.0 * (x[i + 1] - x[i - 1])
    c[i] = x[i + 1] - x[i]
    rhs[i] = 6.0 * (
      (y[i + 1] - y[i]) / (x[i + 1] - x[i])
      - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
    )

  gamma = -b[0]
  alpha = c[n - 2]
  beta_sm = a[0]
  b[0] -= gamma
  b[n - 2] -= alpha * beta_sm / gamma

  deriv = _solve_tridiagonal(a, b, c, rhs)
  u = np.zeros(n - 1)
  u[0] = gamma
  u[n - 2] = alpha
  z = _solve_tridiagonal(a, b, c, u)

  scale = (deriv[0] + beta_sm * deriv[n - 2] / gamma) / (
    1.0 + z[0] + beta_sm * z[n - 2] / gamma
  )
  deriv -= scale * z
  return np.append(deriv, deriv[0])


def _solve_tridiagonal(
  a: np.ndarray, b: np.ndarray, c: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
  """Solve a tridiagonal linear system."""
  n = len(a)
  sol = np.zeros(n)
  gamma = np.zeros(n)
  sol[0] = rhs[0] / b[0]
  beta = b[0]
  for i in range(1, n):
    gamma[i] = c[i - 1] / beta
    beta = b[i] - a[i] * gamma[i]
    sol[i] = (rhs[i] - a[i] * sol[i - 1]) / beta
  for i in range(n - 2, -1, -1):
    sol[i] -= gamma[i + 1] * sol[i + 1]
  return sol


def _eval_spline_derivative(
  x: np.ndarray, y: np.ndarray, deriv: np.ndarray, t: float
) -> float:
  """Evaluate the derivative of a periodic cubic spline."""
  n = len(x)
  lower, upper = 0, n - 1
  while upper - lower > 1:
    middle = (upper + lower) // 2
    if x[middle] > t:
      upper = middle
    else:
      lower = middle
  dx = x[upper] - x[lower]
  a = (x[upper] - t) / dx
  b = 1.0 - a
  dadx = -1.0 / dx
  return (
    dadx * y[lower]
    - dadx * y[upper]
    + ((1.0 - 3.0 * a * a) * deriv[lower] + (3.0 * b * b - 1.0) * deriv[upper])
    * dx
    / 6.0
  )


def _parse_lj(
  top: GromacsTopFile,
  all_atom_types_mol: List[List[str]],
  excl_pairs: List[Tuple[int, int]],
):
  """Parse Lennard-Jones parameters from the topology.

  Args:
    top: Parsed GROMACS topology.
    all_atom_types_mol: Atom types for each molecule.
    excl_pairs: Existing excluded atom pairs.

  Returns a flat tuple consumed directly by :func:`create_topology`.
  """

  atom_types = []
  for mol in top._molecules:
    mol_type = top._moleculeTypes[mol.name]
    for atom in mol_type.atoms:
      atom_types.append(atom.type)

  lj_idx_list = [0 for _ in atom_types]
  atom_params = []
  num_lj_types = 0
  lj_type_list = []
  for i, atom_type in enumerate(atom_types):
    atom = top._atomTypes[atom_type]
    if lj_idx_list[i]:
      continue
    ljtype = (atom.v, atom.w)
    atom_params.append(ljtype)
    num_lj_types += 1
    lj_idx_list[i] = num_lj_types
    lj_type_list.append(atom)
    for j in range(i + 1, len(atom_types)):
      atom_type2 = atom_types[j]
      if lj_idx_list[j] > 0:
        continue
      atom2 = top._atomTypes[atom_type2]
      if atom2 is atom:
        lj_idx_list[j] = num_lj_types

  C6_table, C12_table = _build_c6_c12_tables(
    lj_type_list, atom_params, num_lj_types, top
  )

  atom_type_indices: List[int] = []
  charges: List[float] = []
  exceptions: Dict[Tuple[int, int], Tuple | None] = {}

  base_atom_index = 0
  atom_list_base_idx = 0
  for mol_idx, mol in enumerate(top._molecules):
    mol_type = top._moleculeTypes[mol.name]
    atom_types_mol = all_atom_types_mol[mol_idx]
    n_mol_atoms = len(mol_type.atoms)

    for _ in range(mol.count):
      for atom_idx, atom in enumerate(mol_type.atoms):
        atom_type = top._atomTypes[atom.type]
        q = float(atom.q) if atom.q is not None else atom_type.charge
        charges.append(q)
        atom_type_indices.append(lj_idx_list[atom_list_base_idx + atom_idx] - 1)

      exceptions = _collect_exceptions(
        top, mol_type, base_atom_index, atom_types_mol, exceptions
      )
      base_atom_index += n_mol_atoms
    atom_list_base_idx += len(mol_type.atoms)

  excl_pairs, exc_pairs, exc_q, exc_c6, exc_c12 = _process_exceptions(
    exceptions, excl_pairs
  )

  atom_type_indices_arr = _idx(atom_type_indices)
  n_atoms = len(charges)
  exclusion_pairs = (
    jnp.array(excl_pairs, dtype=jnp.int32)
    if excl_pairs
    else jnp.zeros((0, 2), dtype=jnp.int32)
  )
  excl_mask = _get_excl_mask(n_atoms, exclusion_pairs)

  return (
    n_atoms,
    atom_type_indices_arr,
    _arr(charges, dtype=jnp.float64),
    C6_table,
    C12_table,
    exclusion_pairs,
    excl_mask,
    (
      jnp.array(exc_pairs, dtype=jnp.int32)
      if exc_pairs
      else jnp.zeros((0, 2), dtype=jnp.int32)
    ),
    _arr(exc_q, dtype=np.float64),
    _arr(exc_c6, dtype=np.float64),
    _arr(exc_c12, dtype=np.float64),
  )


def _build_c6_c12_tables(lj_type_list, atom_params, num_lj_types, top):
  """Build Lennard-Jones parameter lookup tables.

  Args:
    lj_type_list: List of unique Lennard-Jones atom types.
    atom_params: Lennard-Jones parameters for each unique atom type.
    num_lj_types: Number of unique Lennard-Jones atom types.
    top: Parsed GROMACS topology.

  Returns:
    Tuple containing the C6 and C12 interaction tables.
  """
  C6_flat, C12_flat = [], []
  combination_type = top._defaults.comb_rule
  for i in range(num_lj_types):
    name_i = lj_type_list[i].name
    for j in range(num_lj_types):
      name_j = lj_type_list[j].name
      type_i, type_j = tuple(sorted([name_i, name_j]))
      if (type_i, type_j) in top._nonbondTypes:
        params = top._nonbondTypes[(type_i, type_j)]
        if combination_type in (2, 3):
          sigma, eps = params.v, params.w
          c6 = 4 * eps * sigma**6
          c12 = 4 * eps * sigma**12
        else:
          c6, c12 = params.v, params.w
      else:
        vi, wi = atom_params[i]
        vj, wj = atom_params[j]
        if combination_type == 1:
          c6 = math.sqrt(vi * vj)
          c12 = math.sqrt(wi * wj)
        else:
          sigma = (
            (vi + vj) / 2.0 if combination_type == 2 else math.sqrt(vi * vj)
          )
          epsilon = math.sqrt(wi * wj)
          c6 = 4 * epsilon * sigma**6
          c12 = 4 * epsilon * sigma**12
      C6_flat.append(c6)
      C12_flat.append(c12)

  C6_table = jnp.array(C6_flat, dtype=jnp.float64).reshape(
    num_lj_types, num_lj_types
  )
  C12_table = jnp.array(C12_flat, dtype=jnp.float64).reshape(
    num_lj_types, num_lj_types
  )
  return C6_table, C12_table


def _collect_exceptions(top, mol, base_atom_index, atom_types_mol, exceptions):
  """Collect nonbonded exceptions and exclusions for a molecule.

  Args:
    top: Parsed GROMACS topology.
    mol: Molecule definition.
    base_atom_index: Global index of the molecule's first atom.
    atom_types_mol: Bonded atom types for the molecule.
    exceptions: Dictionary mapping atom pairs to exception parameters.

  Returns:
    Updated exception dictionary containing explicit pair interactions
    and excluded atom pairs.
  """
  mol_charges = []
  for atom in mol.atoms:
    atom_type = top._atomTypes[atom.type]
    q = atom.q if atom.q is not None else float(atom_type.charge)
    mol_charges.append(q)

  def convert_params(v, w):
    if top._defaults.comb_rule == 3:
      sigma, epsilon = v, w
      return [4 * epsilon * sigma**6, 4 * epsilon * sigma**12]
    return v, w

  def get_key(base, i, j):
    return (
      min(base + i, base + j),
      max(base + i, base + j),
    )

  for pair in mol.pairs:
    atoms = pair.atoms
    types = tuple(atom_types_mol[i] for i in atoms)
    if pair.params is not None:
      v, w = pair.params.v, pair.params.w
    elif types in top._pairTypes:
      p = top._pairTypes[types]
      v, w = p.v, p.w
    elif types[::-1] in top._pairTypes:
      p = top._pairTypes[types[::-1]]
      v, w = p.v, p.w
    else:
      continue
    v, w = convert_params(v, w)
    q1, q2 = mol_charges[atoms[0]], mol_charges[atoms[1]]
    key = get_key(base_atom_index, atoms[0], atoms[1])
    exceptions[key] = (q1 * q2, v, w)

  for exclusion in mol.exclusions:
    atoms = exclusion.atoms
    for atom in atoms[1:]:
      exceptions[get_key(base_atom_index, atoms[0], atom)] = None

  for pair in mol.findExclusionsFromBonds(False):
    exceptions[get_key(base_atom_index, pair[0], pair[1])] = None

  for constraint in mol.constraints:
    exceptions[
      get_key(base_atom_index, constraint.atoms[0], constraint.atoms[1])
    ] = None

  return exceptions


def _process_exceptions(exceptions, excl_pairs):
  """Separate exclusions from explicit nonbonded exceptions.

  Args:
    exceptions: Dictionary mapping atom pairs to exception parameters.
    excl_pairs: List of excluded atom pairs.

  Returns:
    Tuple containing the updated exclusion list, exception atom pairs,
    charge products, C6 parameters, and C12 parameters.
  """

  exc_pairs, exc_q, exc_c6, exc_c12 = [], [], [], []
  for (i, j), params in exceptions.items():
    excl_pairs.append([i, j])
    if params is not None:
      q, c6, c12 = params
      exc_pairs.append([i, j])
      exc_q.append(q)
      exc_c6.append(c6)
      exc_c12.append(c12)
  return excl_pairs, exc_pairs, exc_q, exc_c6, exc_c12


def _get_excl_mask(n_atoms, exclusion_pairs):
  """Build the exclusion mask for nonbonded interactions.

  Args:
    n_atoms: Number of atoms in the system.
    exclusion_pairs: Array of excluded atom pairs.

  Returns:
    Boolean exclusion mask with shape (n_atoms, n_atoms).
  """
  excl_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)
  if exclusion_pairs.shape[0] > 0:
    excl_mask = excl_mask.at[exclusion_pairs[:, 0], exclusion_pairs[:, 1]].set(
      True
    )
    excl_mask = excl_mask.at[exclusion_pairs[:, 1], exclusion_pairs[:, 0]].set(
      True
    )
  return excl_mask


def _get_angle_params(top, angle: Angle, btyp):
  """Retrieve force-field parameters for a bonded angle interaction.

  Parameters are taken from the angle's inline definition if present;
  otherwise they are looked up in the topology's angle-type table using the
  atom types of the three participating atoms (tried in both forward and
  reverse order).

  Args:
    top: Gromacs topology object containing the ``_angleTypes`` look-up table.
    angle: ``Angle`` dataclass instance describing the interaction,
      including atom indices and optional inline parameters.
    btyp: Sequence mapping atom indices to their force-field type strings.

  Returns:
    Tuple of (atoms, theta0, k) where:
      atoms is the list of atom indices defining the angle,
      theta0 is the equilibrium angle in radians,
      k is the force constant in the topology's native units.

  Raises:
    ValueError: If no parameters can be found for the given atom-type
      combination in either direction.
  """
  typs = tuple(btyp[i] for i in angle.atoms)
  if angle.params is not None:
    k = angle.params.k
    theta0 = angle.params.theta0
  elif typs in top._angleTypes:
    angle_type = top._angleTypes[typs]
    k = angle_type.k
    theta0 = angle_type.theta0
  elif typs[::-1] in top._angleTypes:
    angle_type = top._angleTypes[typs[::-1]]
    k = angle_type.k
    theta0 = angle_type.theta0
  else:
    raise ValueError(
      f'No parameters specified for angle: {angle.atoms[0]}, {angle.atoms[1]}, {angle.atoms[2]}'
    )
  return angle.atoms, theta0 * jnp.pi / 180.0, k


def _get_dihedral_params(
  top, dihedral, btyp, dihedral_type_table, wildcard_dihedral_types
):
  """Retrieve force-field parameters for a bonded dihedral interaction.

  Parameters are taken from the dihedral's inline definition if present.
  Otherwise, the central bond pair (typs[1], typs[2]) is used to look up
  candidate types in ``dihedral_type_table``; if not found, the wildcard
  list is used instead. The first matching entry (exact matches preferred
  over wildcard 'X' entries) is returned.

  Args:
    top: Gromacs topology object containing the ``_dihedralTypes`` look-up table.
    dihedral: Dihedral dataclass instance describing the interaction,
      including atom indices, functional type, and optional inline
      parameters.
    btyp: Sequence mapping atom indices to their force-field type strings.
    dihedral_type_table: Dict mapping central-bond atom-type pairs
      ``(typs[1], typs[2])`` to lists of candidate type keys.
    wildcard_dihedral_types: Fallback list of type keys used when the
      central bond pair is not found in ``dihedral_type_table``.

  Returns:
    List of parameter objects for the matched dihedral type. Multiple
    entries may be present for Fourier-series dihedrals with several
    multiplicities.

  Raises:
    ValueError: If no matching parameters can be found for the given
      atom-type combination in either forward or reverse order.
  """
  typs = tuple(btyp[i] for i in dihedral.atoms)
  dihedral_type = dihedral.func_type
  rev_typs = typs[::-1] + (dihedral_type,)
  typs_key = typs + (dihedral_type,)
  if dihedral.params is not None:
    params_list = [dihedral.params]
  else:
    params_list = None
    if (typs[1], typs[2]) in dihedral_type_table:
      dihedral_types = dihedral_type_table[(typs[1], typs[2])]
    else:
      dihedral_types = wildcard_dihedral_types
    for key in dihedral_types:
      if all(a == b or a == 'X' for a, b in zip(key, typs_key)) or all(
        a == b or a == 'X' for a, b in zip(key, rev_typs)
      ):
        params_list = top._dihedralTypes[key]
        if 'X' not in key:
          break
    if params_list is None:
      raise ValueError(
        f'No parameters specified for dihedral: {dihedral.atoms[0]}, {dihedral.atoms[1]}, {dihedral.atoms[2]}, {dihedral.atoms[3]}'
      )
  return params_list
