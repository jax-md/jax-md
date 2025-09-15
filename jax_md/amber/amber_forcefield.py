"""
JAX compatible dataclass to store force field parameters
"""

from jax_md import dataclasses, util
from dataclasses import fields
import jax
import jax.numpy as jnp

Array = util.Array

# TODO change name of file to amber_dataclasses
# TODO add PME parameter dataclass
# TODO add base parameter dataclass with common indexing that all others are based on
# TODO also add structure dataclass that can be extended with custom fields

@dataclasses.dataclass
class BondRestraint(object):
    ind1: Array
    ind2: Array
    target: Array
    force1: Array
    force2: Array

@dataclasses.dataclass
class AngleRestraint(object):
    ind1: Array
    ind2: Array
    ind3: Array
    target: Array
    force1: Array
    force2: Array

@dataclasses.dataclass
class TorsionRestraint(object):
    ind1: Array
    ind2: Array
    ind3: Array
    ind4: Array
    target: Array
    force1: Array
    force2: Array

@dataclasses.dataclass
class AmberForceField(object):
    """
    A JAX transformation compatible dataclass containing the AMBER parameters.

    Attributes:
        
    """
    name: Array
    atom_count: Array # TODO change to num_atoms
    atom_types: Array
    atomic_number: Array
    positions: Array # TODO move this to other structure
    box_vectors: Array # TODO change to orthoganalization_matrix
    # orth_matrix: Array
    masses: Array # might not work for HMR
    total_charge: Array
    params_to_indices: list = dataclasses.static_field()

    bond_restraints: BondRestraint
    angle_restraints: AngleRestraint
    torsion_restraints: TorsionRestraint

    cutoff: Array

    ### PME params
    nbr_list: Array # TODO decouple this from this and move to interaction class
    grid_points: Array
    ewald_alpha: Array
    ewald_error: Array
    dr_threshold: Array
    exclusions: Array
    
    # TODO add numeric mapping scheme to populate full forcefield in JAX friendly way
    # TODO load all fields of the amber ff here for future use

    ### Bond Parameters
    bond_idx: Array
    bond_k: Array
    bond_len: Array

    ### Angle Parameters
    angle_idx: Array
    angle_k: Array
    angle_equil: Array

    ### Torsion Parameters
    torsion_idx: Array
    torsion_k: Array
    torsion_phase: Array
    torsion_period: Array

    ### Nonbonded Parameters
    pairs: Array
    sigma: Array
    epsilon: Array
    charges: Array

    ### 1-4 Parameters
    pairs_14: Array
    charges_14: Array
    sigma_14: Array
    epsilon_14: Array
    scee_14: Array
    scnb_14: Array

    ### Dispersion term
    disp_coef: Array

    ### FFQ Parameters
    # TODO remove this in future version, add to separate dataclass
    gamma: Array
    electronegativity: Array
    hardness: Array
    species: Array
    name_to_index: Array # TODO this may not be safe for vectorization due to string types
    # TODO this change is for linear response code as shape of Amat and bvec depend on this
    # this may not be a safe change in general and may need to be reexamined
    solute_cut: Array = dataclasses.static_field() # TODO change to num_solute

    # TODO populate local indices and fill off diagonal terms where applicable, may need to store original arrays
    # this helps because you can still maintain globally consistent indexing for parameter optimization
    # but have the ability to dynamically recompute the pre-indexed parameters and combinations (e.g. LJ sig/eps)
    # only as needed when parameters are updated, or never if you're just doing standard MD
    def fill_off_diag():
        return

@dataclasses.dataclass
class FFQForceField(object):
    gamma: Array
    electronegativity: Array
    hardness: Array