"""Topology management for OPLSAA forcefield."""

import jax.numpy as jnp
from jax_md.mm_forcefields.base import Topology
from jax_md.mm_forcefields import neighbor
from jax_md.util import Array
from typing import Optional


def create_topology(n_atoms: int,
                   bonds: Array,
                   angles: Array,
                   torsions: Array,
                   impropers: Array,
                   molecule_id: Optional[Array] = None) -> Topology:
    """Create OPLSAA topology with automatically computed masks.
    
    Args:
        n_atoms: Number of atoms.
        bonds: Bond connectivity, shape (n_bonds, 2).
        angles: Angle connectivity, shape (n_angles, 3).
        torsions: Torsion connectivity, shape (n_torsions, 4).
        impropers: Improper connectivity, shape (n_impropers, 4).
        molecule_id: Optional molecule IDs for each atom.
    
    Returns:
        Topology object with computed exclusion and 1-4 masks.
    """
    # Build exclusion mask (1-2 and 1-3 interactions)
    exclusion_mask = neighbor.make_exclusion_mask(n_atoms, bonds, angles, molecule_id)
    
    # Build 1-4 interaction table
    pair_14_mask = neighbor.make_14_table(n_atoms, torsions, exclusion_mask, molecule_id)
    
    return Topology(
        n_atoms=n_atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        impropers=impropers,
        exclusion_mask=exclusion_mask,
        pair_14_mask=pair_14_mask,
        molecule_id=molecule_id
    )


def validate_topology(topology: Topology) -> None:
    """Validate topology data structures.
    
    Args:
        topology: Topology to validate.
    
    Raises:
        ValueError: If topology is invalid.
    """
    n = topology.n_atoms
    
    # Check array shapes
    if topology.bonds.ndim != 2 or topology.bonds.shape[1] != 2:
        raise ValueError(f"bonds must have shape (n_bonds, 2), got {topology.bonds.shape}")
    
    if topology.angles.ndim != 2 or topology.angles.shape[1] != 3:
        raise ValueError(f"angles must have shape (n_angles, 3), got {topology.angles.shape}")
    
    if topology.torsions.ndim != 2 or topology.torsions.shape[1] != 4:
        raise ValueError(f"torsions must have shape (n_torsions, 4), got {topology.torsions.shape}")
    
    if topology.impropers.ndim != 2 or topology.impropers.shape[1] != 4:
        raise ValueError(f"impropers must have shape (n_impropers, 4), got {topology.impropers.shape}")
    
    # Check mask shapes
    if topology.exclusion_mask.shape != (n, n):
        raise ValueError(f"exclusion_mask must have shape ({n}, {n}), got {topology.exclusion_mask.shape}")
    
    if topology.pair_14_mask.shape != (n, n):
        raise ValueError(f"pair_14_mask must have shape ({n}, {n}), got {topology.pair_14_mask.shape}")
    
    # Check atom indices are in valid range
    max_bond_idx = jnp.max(topology.bonds) if topology.bonds.size > 0 else -1
    if max_bond_idx >= n:
        raise ValueError(f"Bond indices exceed n_atoms: {max_bond_idx} >= {n}")
    
    max_angle_idx = jnp.max(topology.angles) if topology.angles.size > 0 else -1
    if max_angle_idx >= n:
        raise ValueError(f"Angle indices exceed n_atoms: {max_angle_idx} >= {n}")
    
    max_torsion_idx = jnp.max(topology.torsions) if topology.torsions.size > 0 else -1
    if max_torsion_idx >= n:
        raise ValueError(f"Torsion indices exceed n_atoms: {max_torsion_idx} >= {n}")
    
    max_improper_idx = jnp.max(topology.impropers) if topology.impropers.size > 0 else -1
    if max_improper_idx >= n:
        raise ValueError(f"Improper indices exceed n_atoms: {max_improper_idx} >= {n}")

