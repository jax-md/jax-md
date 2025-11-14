"""Parser for CHARMM file formats (RTF and PRM files).

This parser is generic and can be used with any force field that uses
CHARMM-format parameter files (e.g., CHARMM, OPLSAA, CGenFF, etc.).
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class AtomType:
    """CHARMM atom type definition."""
    name: str
    mass: float
    element: str
    epsilon: float = 0.0
    rmin_half: float = 0.0  # Rmin/2 in CHARMM format


@dataclass
class Atom:
    """Atom in a residue."""
    name: str
    type: str
    charge: float


@dataclass
class BondType:
    """Bond parameter."""
    atom1: str
    atom2: str
    k: float  # kcal/mol/Å²
    r0: float  # Å


@dataclass
class AngleType:
    """Angle parameter."""
    atom1: str
    atom2: str
    atom3: str
    k: float  # kcal/mol/rad²
    theta0: float  # degrees


@dataclass
class DihedralType:
    """Dihedral parameter."""
    atom1: str
    atom2: str
    atom3: str
    atom4: str
    k: float  # kcal/mol
    n: int  # periodicity
    phase: float  # degrees


def parse_rtf(rtf_file: str) -> Tuple[Dict[str, AtomType], List[Atom], Dict[str, int], List[Tuple[str, str]], List[Tuple[str, str, str, str]]]:
    """Parse CHARMM RTF (residue topology) file.
    
    Args:
        rtf_file: Path to RTF file
        
    Returns:
        atom_types: Dict of atom type name -> AtomType
        atoms: List of Atom objects (ordered as in RTF)
        atom_name_to_idx: Dict mapping RTF atom name to index
        bonds: List of (atom1_name, atom2_name) tuples
        impropers: List of (atom1, atom2, atom3, atom4) tuples
    """
    atom_types = {}
    atoms = []
    atom_name_to_idx = {}
    bonds = []
    impropers = []
    
    with open(rtf_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('*') or line.startswith('!'):
            i += 1
            continue
        
        # Parse MASS (atom type definitions)
        if line.startswith('MASS'):
            parts = line.split()
            if len(parts) >= 4:
                type_name = parts[2]
                mass = float(parts[3])
                element = parts[4] if len(parts) > 4 else type_name[0]
                atom_types[type_name] = AtomType(type_name, mass, element)
        
        # Parse RESI (residue definition)
        elif line.startswith('RESI'):
            # Parse atoms in this residue
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith('!'):
                    i += 1
                    continue
                
                if line.startswith('ATOM'):
                    parts = line.split()
                    if len(parts) >= 4:
                        atom_name = parts[1]
                        atom_type = parts[2]
                        charge = float(parts[3])
                        atom_idx = len(atoms)
                        atoms.append(Atom(atom_name, atom_type, charge))
                        atom_name_to_idx[atom_name] = atom_idx
                
                elif line.startswith('BOND'):
                    # Parse bonds - can be multiple pairs per line
                    parts = line.split()[1:]  # Skip 'BOND'
                    for j in range(0, len(parts)-1, 2):
                        if j+1 < len(parts):
                            bonds.append((parts[j], parts[j+1]))
                
                elif line.startswith('IMPR'):
                    # Parse impropers - 4 atoms
                    parts = line.split()[1:]  # Skip 'IMPR'
                    if len(parts) >= 4:
                        impropers.append((parts[0], parts[1], parts[2], parts[3]))
                
                elif line.startswith('PATCH') or line.startswith('END'):
                    break
                
                i += 1
            continue
        
        i += 1
    
    return atom_types, atoms, atom_name_to_idx, bonds, impropers


def parse_prm(prm_file: str) -> Tuple[Dict, Dict, Dict, Dict[str, AtomType]]:
    """Parse CHARMM PRM (parameter) file.
    
    Args:
        prm_file: Path to PRM file
        
    Returns:
        bond_params: Dict of (type1, type2) -> BondType
        angle_params: Dict of (type1, type2, type3) -> AngleType  
        dihedral_params: Dict of (type1, type2, type3, type4) -> list of DihedralType
        nonbonded_params: Dict of type -> AtomType (with epsilon and rmin)
    """
    bond_params = {}
    angle_params = {}
    dihedral_params = {}
    nonbonded_params = {}
    
    section = None
    
    with open(prm_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('!') or line.startswith('*'):
                continue
            
            # Detect section headers
            if line.startswith('BOND'):
                section = 'BOND'
                continue
            elif line.startswith('ANGLE'):
                section = 'ANGLE'
                continue
            elif line.startswith('DIHE') or line.startswith('TORSION'):
                section = 'DIHEDRAL'
                continue
            elif line.startswith('IMPR'):
                section = 'IMPROPER'
                continue
            elif line.startswith('NONB') or line.startswith('NBON'):
                section = 'NONBONDED'
                continue
            elif line.startswith('END'):
                break
            
            # Parse parameters based on section
            parts = line.split()
            
            if section == 'BOND' and len(parts) >= 4:
                type1, type2 = parts[0], parts[1]
                k = float(parts[2])
                r0 = float(parts[3])
                bond_params[(type1, type2)] = BondType(type1, type2, k, r0)
                bond_params[(type2, type1)] = BondType(type2, type1, k, r0)
            
            elif section == 'ANGLE' and len(parts) >= 5:
                type1, type2, type3 = parts[0], parts[1], parts[2]
                k = float(parts[3])
                theta0 = float(parts[4])
                angle_params[(type1, type2, type3)] = AngleType(type1, type2, type3, k, theta0)
                angle_params[(type3, type2, type1)] = AngleType(type3, type2, type1, k, theta0)
            
            elif section == 'DIHEDRAL' and len(parts) >= 7:
                type1, type2, type3, type4 = parts[0], parts[1], parts[2], parts[3]
                k = float(parts[4])
                n = int(float(parts[5]))
                phase = float(parts[6])
                
                # CHARMM can have multiple terms for same atom types
                key = (type1, type2, type3, type4)
                if key not in dihedral_params:
                    dihedral_params[key] = []
                dihedral_params[key].append(DihedralType(type1, type2, type3, type4, k, n, phase))
                
                # Also store reverse
                key_rev = (type4, type3, type2, type1)
                if key_rev not in dihedral_params:
                    dihedral_params[key_rev] = []
                dihedral_params[key_rev].append(DihedralType(type4, type3, type2, type1, k, n, phase))
            
            elif section == 'IMPROPER' and len(parts) >= 7:
                # Same format as dihedral
                type1, type2, type3, type4 = parts[0], parts[1], parts[2], parts[3]
                k = float(parts[4])
                n = int(float(parts[5]))
                phase = float(parts[6])
                
                key = (type1, type2, type3, type4)
                if key not in dihedral_params:
                    dihedral_params[key] = []
                dihedral_params[key].append(DihedralType(type1, type2, type3, type4, k, n, phase))
            
            elif section == 'NONBONDED' and len(parts) >= 4:
                # Skip header/option lines (they have keywords like 'nbxmod', 'cutnb', etc.)
                if parts[0] in ['nbxmod', 'cutnb', 'ctofnb', 'ctonnb', 'eps', 'e14fac', 'atom', 'cdiel', 'switch', 'vatom', 'vdistance', 'vswitch', '-']:
                    continue
                
                # Try to parse as numeric data
                try:
                    atom_type = parts[0]
                    # CHARMM NONBONDED format: atomtype ignored epsilon Rmin/2 [ignored 1-4_epsilon 1-4_Rmin/2]
                    epsilon = abs(float(parts[2]))  # kcal/mol
                    rmin_half = float(parts[3])  # Rmin/2 in Å
                    
                    # Create or update atom type
                    if atom_type in nonbonded_params:
                        nonbonded_params[atom_type].epsilon = epsilon
                        nonbonded_params[atom_type].rmin_half = rmin_half
                    else:
                        nonbonded_params[atom_type] = AtomType(atom_type, 0.0, atom_type[0], epsilon, rmin_half)
                except (ValueError, IndexError):
                    # Skip lines that don't parse as numbers
                    continue
    
    return bond_params, angle_params, dihedral_params, nonbonded_params


def parse_pdb_simple(pdb_file: str) -> Tuple[List[str], jnp.ndarray]:
    """Parse PDB file to get atom names and coordinates.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        atom_names: List of atom names
        positions: Array of shape (n_atoms, 3) with coordinates
    """
    atom_names = []
    positions = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # PDB format: columns 13-16 for atom name, 31-38, 39-46, 47-54 for x,y,z
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                atom_names.append(atom_name)
                positions.append([x, y, z])
    
    return atom_names, jnp.array(positions)
