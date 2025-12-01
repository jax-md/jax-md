"""Neighbor list utilities for molecular mechanics forcefields."""

import jax.numpy as jnp
from jax_md import partition
from jax_md.partition import NeighborFn
from jax_md.util import Array
from typing import Optional


def create_neighbor_list(
  displacement_fn, box, r_cut: float, dr_threshold: float = 0.5, **kwargs
) -> NeighborFn:
  """Create a neighbor list function for MM forcefields.

  Wrapper around jax_md.partition.neighbor_list with MM-appropriate defaults.

  Args:
      displacement_fn: Displacement function from jax_md.space.
      box: Simulation box (scalar or array).
      r_cut: Cutoff distance for neighbor list.
      dr_threshold: Skin distance for neighbor list updates.
      **kwargs: Additional arguments passed to partition.neighbor_list.

  Returns:
      neighbor_fn: Function to build/update neighbor lists.
  """
  # Default to masked format for MM forcefields
  if 'mask' not in kwargs:
    kwargs['mask'] = True

  return partition.neighbor_list(
    displacement_fn, box, r_cut, dr_threshold=dr_threshold, **kwargs
  )


def make_exclusion_mask(
  n_atoms: int, bonds: Array, angles: Array, molecule_id: Optional[Array] = None
) -> Array:
  """Build exclusion mask for nonbonded interactions.

  Atoms that are bonded (1-2) or in the same angle (1-3) are excluded
  from nonbonded interactions.

  Args:
      n_atoms: Number of atoms.
      bonds: Array of shape (n_bonds, 2) with bonded atom pairs.
      angles: Array of shape (n_angles, 3) with angle atom indices.
      molecule_id: Optional array of shape (n_atoms,) with molecule IDs.
          If provided, only exclude pairs within the same molecule.

  Returns:
      exclusion_mask: Boolean array of shape (n_atoms, n_atoms).
          True indicates the pair should be excluded.
  """
  exclusion_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)

  # Filter by molecule if provided
  if molecule_id is not None:
    bond_same_mol = molecule_id[bonds[:, 0]] == molecule_id[bonds[:, 1]]
    angle_same_mol = molecule_id[angles[:, 0]] == molecule_id[angles[:, 2]]
    bonds = bonds[bond_same_mol]
    angles = angles[angle_same_mol]

  # Exclude bonded pairs (1-2 interactions)
  exclusion_mask = exclusion_mask.at[bonds[:, 0], bonds[:, 1]].set(True)
  exclusion_mask = exclusion_mask.at[bonds[:, 1], bonds[:, 0]].set(True)

  # Exclude 1-3 pairs (atoms at ends of angles)
  exclusion_mask = exclusion_mask.at[angles[:, 0], angles[:, 2]].set(True)
  exclusion_mask = exclusion_mask.at[angles[:, 2], angles[:, 0]].set(True)

  return exclusion_mask


def make_14_table(
  n_atoms: int,
  torsions: Array,
  exclusion_mask: Array,
  molecule_id: Optional[Array] = None,
) -> Array:
  """Build 1-4 interaction lookup table.

  1-4 pairs are atoms at the ends of a proper dihedral that are not
  already excluded (bonded or in angle).

  Args:
      n_atoms: Number of atoms.
      torsions: Array of shape (n_torsions, 4) with torsion atom indices.
      exclusion_mask: Existing exclusion mask to check against.
      molecule_id: Optional array of shape (n_atoms,) with molecule IDs.
          If provided, only consider pairs within the same molecule.

  Returns:
      pair_14_mask: Boolean array of shape (n_atoms, n_atoms).
          True indicates a 1-4 pair.
  """
  pair_14_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)

  # Extract 1-4 pairs (first and last atoms in torsions)
  pairs_14 = jnp.stack([torsions[:, 0], torsions[:, 3]], axis=1)

  # Filter by molecule if provided
  if molecule_id is not None:
    same_mol = molecule_id[pairs_14[:, 0]] == molecule_id[pairs_14[:, 1]]
    pairs_14 = pairs_14[same_mol]

  # Check which pairs are not already excluded
  not_excluded = ~exclusion_mask[pairs_14[:, 0], pairs_14[:, 1]]
  valid_14_pairs = pairs_14[not_excluded]

  # Mark valid 1-4 pairs in the table
  if valid_14_pairs.shape[0] > 0:
    pair_14_mask = pair_14_mask.at[
      valid_14_pairs[:, 0], valid_14_pairs[:, 1]
    ].set(True)
    pair_14_mask = pair_14_mask.at[
      valid_14_pairs[:, 1], valid_14_pairs[:, 0]
    ].set(True)

  return pair_14_mask
