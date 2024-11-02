"""
JAX compatible dataclass to store force field parameters
"""

from jax_md import dataclasses, util
from dataclasses import fields
import jax
import jax.numpy as jnp

Array = util.Array

@dataclasses.dataclass
class AmberForceField(object):
    """
    A JAX transformation compatible dataclass containing the AMBER parameters.

    Attributes:
        
    """
    # name: Array
    # atom_count: Array
    # atom_types: Array
    # atomic_nums: Array
    # positions: Array
    # orth_matrix: Array

    # total_charge: Array
    # energy_minimize: Array
    # energy_minim_steps: Array
    # periodic_image_shifts: Array

    # bond_restraints: BondRestraint
    # angle_restraints: AngleRestraint
    # torsion_restraints: TorsionRestraint

    # target_e: Array
    # target_f: Array
    # target_ch: Array
    
    # TODO: Create numeric mapping scheme to populate full forcefield in JAX friendly way

    # Bond Parameters
    b_k: Array
    b_l: Array
    b_1_idx: Array
    b_2_idx: Array
    b_prm_idx: Array

    # Angle Parameters
    a_k: Array
    a_eq_ang: Array
    a_1_idx: Array
    a_2_idx: Array
    a_3_idx: Array
    a_prm_idx: Array

    # Torsion Parameters
    t_k: Array
    t_phase: Array
    t_period: Array
    t_1_idx: Array
    t_2_idx: Array
    t_3_idx: Array
    t_4_idx: Array
    t_prm_idx: Array

    # Common Nonbonded Parameters
    pairs: Array
    pairs14: Array

    # Lennard-Jones Parameters
    lj_type: Array
    sigma: Array
    epsilon: Array
    scnb: Array

    # Coulomb Parameters
    charges: Array
    scee: Array

