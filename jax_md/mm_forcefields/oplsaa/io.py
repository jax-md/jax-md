"""Input/Output utilities for OPLSAA forcefield.

This module provides basic support for loading CHARMM/NAMD style files (`.pdb`,
`.prm`, `.rtf`) and translating them into the OPLS-AA topology and parameter
objects used throughout `jax_md`.
"""

import jax.numpy as jnp
from typing import Tuple, Optional, List, Set
from jax_md.mm_forcefields.oplsaa.topology import create_topology
from jax_md.mm_forcefields.oplsaa.params import create_parameters, Parameters
from jax_md.mm_forcefields.base import Topology
from jax_md.mm_forcefields.io.charmm import parse_rtf, parse_prm, parse_pdb_simple


def load_charmm_system(
    pdb_file: str,
    prm_file: str,
    rtf_file: str,
    molecule_id: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Topology, Parameters]:
    """Load CHARMM/NAMD format files.

    Args:
        pdb_file: Path to PDB file with coordinates.
        prm_file: Path to CHARMM parameter file (.prm).
        rtf_file: Path to CHARMM topology file (.rtf).
        molecule_id: Optional array assigning each atom to a molecule.

    Returns:
        positions: Atomic positions, shape (n_atoms, 3) in Angstroms.
        topology: OPLSAA Topology object.
        parameters: OPLSAA Parameters object.

    Raises:
        ValueError: If files cannot be parsed.
    """

    # Parse input files
    _, rtf_atoms, rtf_atom_name_to_idx, rtf_bonds, rtf_impropers = (
        parse_rtf(rtf_file)
    )
    bond_params, angle_params, dihedral_params, nonbonded_params = parse_prm(prm_file)
    pdb_atom_names, positions = parse_pdb_simple(pdb_file)

    n_atoms = len(pdb_atom_names)

    # Check that atom counts match
    if len(rtf_atoms) != n_atoms:
        raise ValueError(
            f"Atom count mismatch: PDB has {n_atoms} atoms, RTF has {len(rtf_atoms)} atoms"
        )

    # Map atoms by position (RTF and PDB should have atoms in same order)
    atom_types = []
    charges = []
    for i in range(n_atoms):
        rtf_atom = rtf_atoms[i]
        atom_types.append(rtf_atom.type)
        charges.append(rtf_atom.charge)

    charges = jnp.array(charges)

    # Build bonds from RTF
    bonds = []
    for atom1_name, atom2_name in rtf_bonds:
        idx1 = rtf_atom_name_to_idx[atom1_name]
        idx2 = rtf_atom_name_to_idx[atom2_name]
        bonds.append([idx1, idx2])
    bonds = (
        jnp.array(bonds, dtype=jnp.int32) if bonds else jnp.zeros((0, 2), dtype=jnp.int32)
    )

    # Infer angles from bonds (all sets of 3 atoms where 1-2 and 2-3 are bonded)
    # Build adjacency list
    adjacency: List[Set[int]] = [set() for _ in range(n_atoms)]
    for idx1, idx2 in bonds:
        idx1 = int(idx1)
        idx2 = int(idx2)
        adjacency[idx1].add(idx2)
        adjacency[idx2].add(idx1)

    angles = []
    for center in range(n_atoms):
        neighbors = list(adjacency[center])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                angles.append([neighbors[i], center, neighbors[j]])
    angles = (
        jnp.array(angles, dtype=jnp.int32) if angles else jnp.zeros((0, 3), dtype=jnp.int32)
    )

    # Infer proper dihedrals from bonds (all sets of 4 atoms where 1-2, 2-3, 3-4 are bonded)
    torsions = []
    for idx2, idx3 in bonds:
        idx2 = int(idx2)
        idx3 = int(idx3)
        # idx2-idx3 is the central bond
        for idx1 in adjacency[idx2]:
            if idx1 == idx3:
                continue
            for idx4 in adjacency[idx3]:
                if idx4 == idx2:
                    continue
                torsions.append([idx1, idx2, idx3, idx4])
    torsions = (
        jnp.array(torsions, dtype=jnp.int32)
        if torsions
        else jnp.zeros((0, 4), dtype=jnp.int32)
    )

    # Build impropers from RTF
    impropers = []
    for atom1_name, atom2_name, atom3_name, atom4_name in rtf_impropers:
        idx1 = rtf_atom_name_to_idx[atom1_name]
        idx2 = rtf_atom_name_to_idx[atom2_name]
        idx3 = rtf_atom_name_to_idx[atom3_name]
        idx4 = rtf_atom_name_to_idx[atom4_name]
        impropers.append([idx1, idx2, idx3, idx4])
    impropers = (
        jnp.array(impropers, dtype=jnp.int32)
        if impropers
        else jnp.zeros((0, 4), dtype=jnp.int32)
    )

    # Create topology
    if molecule_id is None:
        # All atoms in same molecule
        molecule_id = jnp.zeros(n_atoms, dtype=jnp.int32)

    topology = create_topology(
        n_atoms=n_atoms,
        bonds=jnp.array(bonds),
        angles=jnp.array(angles),
        torsions=jnp.array(torsions),
        impropers=jnp.array(impropers),
        molecule_id=jnp.array(molecule_id),
    )

    # Extract bond parameters
    bond_k = jnp.zeros(len(bonds))
    bond_r0 = jnp.zeros(len(bonds))
    for i, (idx1, idx2) in enumerate(bonds):
        idx1 = int(idx1)
        idx2 = int(idx2)
        type1 = atom_types[idx1]
        type2 = atom_types[idx2]
        key = (type1, type2)
        if key in bond_params:
            bond_k = bond_k.at[i].set(bond_params[key].k)  # kcal/mol/Å²
            bond_r0 = bond_r0.at[i].set(bond_params[key].r0)  # Å

    # Extract angle parameters
    angle_k = jnp.zeros(len(angles))
    angle_theta0 = jnp.zeros(len(angles))
    for i, (idx1, idx2, idx3) in enumerate(angles):
        idx1 = int(idx1)
        idx2 = int(idx2)
        idx3 = int(idx3)
        type1 = atom_types[idx1]
        type2 = atom_types[idx2]
        type3 = atom_types[idx3]
        key = (type1, type2, type3)
        if key in angle_params:
            angle_k = angle_k.at[i].set(angle_params[key].k)  # kcal/mol/rad²
            angle_theta0 = angle_theta0.at[i].set(
                jnp.radians(angle_params[key].theta0)
            )  # Convert to radians

    # Extract torsion parameters
    torsion_k = jnp.zeros(len(torsions))
    torsion_n = jnp.zeros(len(torsions), dtype=jnp.int32)
    torsion_gamma = jnp.zeros(len(torsions))

    for i, (idx1, idx2, idx3, idx4) in enumerate(torsions):
        idx1 = int(idx1)
        idx2 = int(idx2)
        idx3 = int(idx3)
        idx4 = int(idx4)
        type1 = atom_types[idx1]
        type2 = atom_types[idx2]
        type3 = atom_types[idx3]
        type4 = atom_types[idx4]
        key = (type1, type2, type3, type4)

        # CHARMM can have multiple dihedral terms - we'll use the first one
        if key in dihedral_params and len(dihedral_params[key]) > 0:
            dihedral = dihedral_params[key][0]
            torsion_k = torsion_k.at[i].set(dihedral.k)
            torsion_n = torsion_n.at[i].set(dihedral.n)
            torsion_gamma = torsion_gamma.at[i].set(jnp.radians(dihedral.phase))

    # Extract improper parameters
    improper_k = jnp.zeros(len(impropers))
    improper_n = jnp.zeros(len(impropers), dtype=jnp.int32)
    improper_gamma = jnp.zeros(len(impropers))

    for i, (idx1, idx2, idx3, idx4) in enumerate(impropers):
        idx1 = int(idx1)
        idx2 = int(idx2)
        idx3 = int(idx3)
        idx4 = int(idx4)
        type1 = atom_types[idx1]
        type2 = atom_types[idx2]
        type3 = atom_types[idx3]
        type4 = atom_types[idx4]
        key = (type1, type2, type3, type4)

        if key in dihedral_params and len(dihedral_params[key]) > 0:
            dihedral = dihedral_params[key][0]
            improper_k = improper_k.at[i].set(dihedral.k)
            improper_n = improper_n.at[i].set(dihedral.n)
            improper_gamma = improper_gamma.at[i].set(jnp.radians(dihedral.phase))

    # Extract nonbonded parameters (LJ)
    # CHARMM uses Rmin/2 and epsilon
    # Convert to sigma: sigma = Rmin/2 * 2^(1/6) ≈ Rmin/2 * 1.122462
    sigma = jnp.zeros(n_atoms)
    epsilon = jnp.zeros(n_atoms)

    for i, atom_type in enumerate(atom_types):
        if atom_type in nonbonded_params:
            nb_param = nonbonded_params[atom_type]
            epsilon = epsilon.at[i].set(nb_param.epsilon)
            # Convert Rmin/2 to sigma
            sigma = sigma.at[i].set(nb_param.rmin_half * (jnp.power(2.0, 1.0 / 6.0)))

    # Create parameters
    parameters = create_parameters(
        bond_k=jnp.array(bond_k),
        bond_r0=jnp.array(bond_r0),
        angle_k=jnp.array(angle_k),
        angle_theta0=jnp.array(angle_theta0),
        torsion_k=jnp.array(torsion_k),
        torsion_n=jnp.array(torsion_n),
        torsion_gamma=jnp.array(torsion_gamma),
        improper_k=jnp.array(improper_k),
        improper_n=jnp.array(improper_n),
        improper_gamma=jnp.array(improper_gamma),
        charges=jnp.array(charges),
        sigma=jnp.array(sigma),
        epsilon=jnp.array(epsilon),
    )

    return positions, topology, parameters
