"""Base dataclasses and types for molecular mechanics forcefields."""

from typing import NamedTuple, Optional
from jax_md.util import Array


class Topology(NamedTuple):
  """Molecular topology defining bonded connectivity.

  Attributes:
      n_atoms: Number of atoms in the system.
      bonds: Array of shape (n_bonds, 2) with atom indices.
      angles: Array of shape (n_angles, 3) with atom indices.
      torsions: Array of shape (n_torsions, 4) with atom indices.
      impropers: Array of shape (n_impropers, 4) with atom indices.
      exclusion_mask: Boolean array of shape (n_atoms, n_atoms) indicating
          pairs that should be excluded from nonbonded interactions.
      pair_14_mask: Boolean array of shape (n_atoms, n_atoms) indicating
          1-4 pairs that get special scaling.
      molecule_id: Array of shape (n_atoms,) assigning each atom to a molecule.
  """

  n_atoms: int
  bonds: Array
  angles: Array
  torsions: Array
  impropers: Array
  exclusion_mask: Array
  pair_14_mask: Array
  molecule_id: Optional[Array] = None


class BondedParameters(NamedTuple):
  """Force constants and equilibrium values for bonded terms.

  All arrays should have shape matching the corresponding topology arrays.

  Attributes:
      bond_k: Bond force constants (kcal/mol/Å²).
      bond_r0: Equilibrium bond lengths (Å).
      angle_k: Angle force constants (kcal/mol/rad²).
      angle_theta0: Equilibrium angles (radians).
      torsion_k: Torsion force constants (kcal/mol).
      torsion_n: Torsion periodicity.
      torsion_gamma: Torsion phase (radians).
      improper_k: Improper force constants (kcal/mol).
      improper_n: Improper periodicity.
      improper_gamma: Improper phase (radians).
  """

  bond_k: Array
  bond_r0: Array
  angle_k: Array
  angle_theta0: Array
  torsion_k: Array
  torsion_n: Array
  torsion_gamma: Array
  improper_k: Array
  improper_n: Array
  improper_gamma: Array


class NonbondedParameters(NamedTuple):
  """Nonbonded interaction parameters.

  Attributes:
      charges: Partial charges for each atom (e).
      sigma: Lennard-Jones sigma parameters (Å).
      epsilon: Lennard-Jones epsilon parameters (kcal/mol).
  """

  charges: Array
  sigma: Array
  epsilon: Array


class NonbondedOptions(NamedTuple):
  """Options for nonbonded interaction calculations.

  Attributes:
      r_cut: Cutoff distance for nonbonded interactions (Å).
      dr_threshold: Neighbor list skin distance (Å).
      use_soft_lj: Whether to use soft-core LJ potential.
      lj_cap: Cap value for soft-core LJ (kcal/mol).
      use_shift_lj: Whether to shift LJ potential to zero at cutoff.
      scale_14_lj: Scaling factor for 1-4 LJ interactions (typically 0.5).
      scale_14_coul: Scaling factor for 1-4 Coulomb interactions (typically 0.5).
  """

  r_cut: float = 12.0
  dr_threshold: float = 0.5
  use_soft_lj: bool = False
  lj_cap: float = 1000.0
  use_shift_lj: bool = False
  scale_14_lj: float = 0.5
  scale_14_coul: float = 0.5
