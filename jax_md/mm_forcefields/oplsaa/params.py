"""Parameter management for OPLSAA forcefield."""

from jax_md.mm_forcefields.base import BondedParameters, NonbondedParameters
from jax_md.util import Array
from typing import NamedTuple


class Parameters(NamedTuple):
    """Complete OPLSAA parameters.

    Attributes:
        bonded: Bonded interaction parameters.
        nonbonded: Nonbonded interaction parameters.
    """

    bonded: BondedParameters
    nonbonded: NonbondedParameters


def create_parameters(
    bond_k: Array,
    bond_r0: Array,
    angle_k: Array,
    angle_theta0: Array,
    torsion_k: Array,
    torsion_n: Array,
    torsion_gamma: Array,
    improper_k: Array,
    improper_n: Array,
    improper_gamma: Array,
    charges: Array,
    sigma: Array,
    epsilon: Array,
) -> Parameters:
    """Create OPLSAA parameters.

    Args:
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
        charges: Partial charges (e).
        sigma: LJ sigma parameters (Å).
        epsilon: LJ epsilon parameters (kcal/mol).

    Returns:
        Parameters object.
    """
    bonded = BondedParameters(
        bond_k=bond_k,
        bond_r0=bond_r0,
        angle_k=angle_k,
        angle_theta0=angle_theta0,
        torsion_k=torsion_k,
        torsion_n=torsion_n,
        torsion_gamma=torsion_gamma,
        improper_k=improper_k,
        improper_n=improper_n,
        improper_gamma=improper_gamma,
    )

    nonbonded = NonbondedParameters(charges=charges, sigma=sigma, epsilon=epsilon)

    return Parameters(bonded=bonded, nonbonded=nonbonded)


def validate_parameters(
    params: Parameters,
    n_bonds: int,
    n_angles: int,
    n_torsions: int,
    n_impropers: int,
    n_atoms: int,
) -> None:
    """Validate parameter dimensions.

    Args:
        params: Parameters to validate.
        n_bonds: Expected number of bonds.
        n_angles: Expected number of angles.
        n_torsions: Expected number of torsions.
        n_impropers: Expected number of impropers.
        n_atoms: Expected number of atoms.

    Raises:
        ValueError: If dimensions don't match.
    """
    bonded = params.bonded
    nonbonded = params.nonbonded

    # Check bonded parameters
    if bonded.bond_k.shape[0] != n_bonds:
        raise ValueError(
            f"bond_k has wrong length: {bonded.bond_k.shape[0]} != {n_bonds}"
        )
    if bonded.bond_r0.shape[0] != n_bonds:
        raise ValueError(
            f"bond_r0 has wrong length: {bonded.bond_r0.shape[0]} != {n_bonds}"
        )

    if bonded.angle_k.shape[0] != n_angles:
        raise ValueError(
            f"angle_k has wrong length: {bonded.angle_k.shape[0]} != {n_angles}"
        )
    if bonded.angle_theta0.shape[0] != n_angles:
        raise ValueError(
            f"angle_theta0 has wrong length: {bonded.angle_theta0.shape[0]} != {n_angles}"
        )

    if bonded.torsion_k.shape[0] != n_torsions:
        raise ValueError(
            f"torsion_k has wrong length: {bonded.torsion_k.shape[0]} != {n_torsions}"
        )
    if bonded.torsion_n.shape[0] != n_torsions:
        raise ValueError(
            f"torsion_n has wrong length: {bonded.torsion_n.shape[0]} != {n_torsions}"
        )
    if bonded.torsion_gamma.shape[0] != n_torsions:
        raise ValueError(
            f"torsion_gamma has wrong length: {bonded.torsion_gamma.shape[0]} != {n_torsions}"
        )

    if bonded.improper_k.shape[0] != n_impropers:
        raise ValueError(
            f"improper_k has wrong length: {bonded.improper_k.shape[0]} != {n_impropers}"
        )
    if bonded.improper_n.shape[0] != n_impropers:
        raise ValueError(
            f"improper_n has wrong length: {bonded.improper_n.shape[0]} != {n_impropers}"
        )
    if bonded.improper_gamma.shape[0] != n_impropers:
        raise ValueError(
            f"improper_gamma has wrong length: {bonded.improper_gamma.shape[0]} != {n_impropers}"
        )

    # Check nonbonded parameters
    if nonbonded.charges.shape[0] != n_atoms:
        raise ValueError(
            f"charges has wrong length: {nonbonded.charges.shape[0]} != {n_atoms}"
        )
    if nonbonded.sigma.shape[0] != n_atoms:
        raise ValueError(
            f"sigma has wrong length: {nonbonded.sigma.shape[0]} != {n_atoms}"
        )
    if nonbonded.epsilon.shape[0] != n_atoms:
        raise ValueError(
            f"epsilon has wrong length: {nonbonded.epsilon.shape[0]} != {n_atoms}"
        )
