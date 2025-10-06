import jax
import jax.numpy as jnp
import numpy as onp
from jax_md.amber.amber_helper import angle, torsion, rdndgr
from jax.scipy.linalg import svd

def amber_distance_restraint(r, r1, r2, r3, r4, k):
    """
    AMBER flat-well distance restraint with 4 control values
    potential is parabolic between r1 - r2 and r3 - r4
    and zero for distances between r2 - r3
    """
    energy = jnp.where(
        r < r2, k * (r - r2)**2,
        jnp.where(
            r > r3, k * (r - r3)**2,
            0.0
        )
    )
    return energy

def harmonic_restraint(delta, k, exponent_1=1, exponent_2=2, norm_const=1):
    """
    General harmonic restraint for bonds, angles, dihedrals.

    """
    return k * jnp.power(jnp.sum(jnp.power(delta, exponent_2))/norm_const, exponent_1)
    # TODO: this is just an idea to build all of these common operations around a single helper
    # this may be inefficient and it may be easier to just divide into harmonic, pos, best fit, etc
 
    # i think these are all just essentially different vector norms, maybe an easier way to display?
    # examples for a few cases
    # energy_bond = harmonic_restraint(r_current-r_eq, k_bond)
    # energy_angle = harmonic_restraint(angle_current-angle_eq, k_angle)
    # energy_dihedral = harmonic_restraint(dihedral_current-dihedral_eq, k_dihedral)
    # std_pos_rest = harmonic_restraint(pos_cur-pos_ref, k, 1, 2, jnp.size(pos_cur))
    # rmsg forces = harmonic_restraint(pos_cur-pos_ref, 1, .5, 2, jnp.size(pos_cur))
    # rmsd structure = harmonic_restraint(pos_cur-pos_ref, 1, .5, 2, jnp.size(pos_cur))

# TODO move some of the code into this to prevent repetition
# going to also need different restraint classes
# e.g. harmonic_bonds, reax_bonds
def reax_general_restraint(value, target, force1, force2):
    diff_sq = (value - target)**2
    energy = force1 * (1.0 - jnp.exp(-force2 * diff_sq))
    return jnp.sum(energy)

def positional_restraint_standard(positions, ref_positions, k, exponent=2, masses=1):
    """
    Standard harmonic positional restraint.
    positions: [N, 3] current atomic positions
    ref_positions: [N, 3] reference atomic positions
    k: scalar, force constant
    exponent: controls power of exponent
    masses: for mass weighting, optional
    """
    delta = positions - ref_positions
    energy = k * jnp.sum(jnp.power(delta, exponent))
    return energy

def shake(positions, constraints, lengths, tol=1e-6, max_iter=100):
    """
    SHAKE constraint algorithm
    positions: [N, 3] positions array
    constraints: [(i, j), ...] pairs of constrained atom indices
    lengths: [L1, L2, ...] target lengths for each constrained pair
    """
    def body(iteration, state):
        positions, converged = state
        max_delta = 0.0
        for (i, j), L in zip(constraints, lengths):
            rij = positions[j] - positions[i]
            dist = jnp.linalg.norm(rij)
            delta = (dist - L) / dist * 0.5
            correction = rij * delta
            positions = positions.at[i].add(correction)
            positions = positions.at[j].add(-correction)
            max_delta = jnp.maximum(max_delta, jnp.abs(dist - L))
        converged = max_delta < tol
        return (positions, converged)

    # Initialize convergence flag as False
    converged = False
    positions, converged = fori_loop(0, max_iter, body, (positions, converged))
    return positions, converged

# TODO look at openmm too, this is only the positional restraint portion of settle
# it's unclear if the velocities should always also be modified too
# equilibrium distances are the same as amber tip3p i believe, but this also warrants checking
# it's likely that there should be some behavior to change this based on the water model used
# also probably need to introduce switch for water angle
# also needs testing against reference code
def settle(water_positions, O_H_dist=0.9566, H_H_dist=1.5138):
    """
    SETTLE algorithm for rigid water (TIP3P)
    water_positions: shape [3,3] array [O, H1, H2].
    """
    # might need to revisit this
    # my understanding is that the algorithm as presented in the paper involves
    #
    O, H1, H2 = water_positions
    # TODO take real masses from force field file
    com = (16.0 * O + H1 + H2) / 18.0  # masses (O=16, H=1)

    # move COM to origin
    # TODO create convenience function for getting COM
    O_rel = O - com
    H1_rel = H1 - com
    H2_rel = H2 - com

    # set ideal geometry in XY-plane
    O_new = jnp.array([0.0, 0.0, 0.0])
    H1_new = jnp.array([O_H_dist, 0.0, 0.0])
    x = O_H_dist * jnp.cos(104.52 / 2 * jnp.pi / 180)
    y = O_H_dist * jnp.sin(104.52 / 2 * jnp.pi / 180)
    H2_new = jnp.array([-x, y, 0.0])

    ideal = jnp.stack([O_new, H1_new, H2_new])

    current = jnp.stack([O_rel, H1_rel, H2_rel])
    R = kabsch(current, ideal)

    new_positions = ideal @ R.T + com

    return new_positions

# TODO also implement lincs as outlined in gromacs documentation
def lincs():
    return

def cm_motion_removal(positions, velocities, masses):
    """
    Center of mass (CMM) motion remover
    positions: [N, 3]
    velocities: [N, 3]
    masses: [N,]
    """
    total_mass = jnp.sum(masses)
    com_velocity = jnp.sum(velocities * masses[:, None], axis=0) / total_mass
    corrected_velocities = velocities - com_velocity
    return positions, corrected_velocities

def com_harmonic_restraint(positions, masses, com_ref, k):
    """
    Harmonic restraint on center-of-mass (COM).
    positions: [N, 3]
    masses: [N,]
    com_ref: [3,] reference COM
    """
    total_mass = jnp.sum(masses)
    com = jnp.sum(positions * masses[:, None], axis=0) / total_mass
    delta = com - com_ref
    return k * jnp.sum(delta**2)

def kabsch(P, Q):
    """
    Compute optimal rotation matrix to align P to Q.
    ref: https://ieeexplore.ieee.org/document/88573
    """
    #TODO double check reference paper and find better citations
    C = P.T @ Q
    V, S, Wt = svd(C)
    d = jnp.linalg.det(V @ Wt)
    D = jnp.diag(jnp.array([1, 1, jnp.sign(d)]))
    R = V @ D @ Wt
    return R

def positional_restraint_bestfit(positions, ref_positions, k):
    """
    CHARMM BESTFIT positional restraint.
    positions: [N, 3] current atomic positions
    ref_positions: [N, 3] reference atomic positions
    k: scalar, force constant
    """
    pos_centroid = positions.mean(axis=0)
    ref_centroid = ref_positions.mean(axis=0)

    pos_centered = positions - pos_centroid
    ref_centered = ref_positions - ref_centroid

    R = kabsch(pos_centered, ref_centered)
    aligned_positions = pos_centered @ R

    delta = aligned_positions - ref_centered
    energy = k * jnp.sum(jnp.square(delta))
    return energy

def rmsd_restraint(positions, ref_positions, k):
    """
    RMSD-based restraint.
    positions: [N, 3] current atomic positions
    ref_positions: [N, 3] reference atomic positions
    k: scalar, restraint constant scaling the RMSD
    """
    pos_centroid = positions.mean(axis=0)
    ref_centroid = ref_positions.mean(axis=0)

    pos_centered = positions - pos_centroid
    ref_centered = ref_positions - ref_centroid

    R = kabsch(pos_centered, ref_centered)
    aligned_positions = pos_centered @ R

    squared_diff = jnp.square(aligned_positions - ref_centered)
    rmsd = jnp.sqrt(jnp.mean(jnp.sum(squared_diff, axis=1)))

    energy = k * rmsd
    return energy

def reax_bond_restraint(positions, idx_1, idx_2, frc_1, frc_2, targets, dist_fn):
    """
    ReaxFF style restraint for bond distance
    Erestraint= Force1*{1.0-exp(Force2*(distance-target_distance)^2}
    positions: [N, 3]
    TODO rest of parameters will likely come in data structure
    """
    cur_dists = jax.vmap(dist_fn)(positions[idx_1] - positions[idx_2])
    rest_pot = jnp.sum(mask * frc_1 *
                     (1.0 - jnp.exp(-frc_2 * (cur_dists - targets)**2)))
    return rest_pot

#TODO move to helper file
def angle_difference(angle1, angle2):
    """
    Calculate angle difference between 2 angles
    while respecting periodicity
    Ex. diff between 170 and -170 is 20 degree. 
    """

    diff = jnp.mod(angle1 - angle2, 360)
    diff = jnp.where(diff < 180, diff, 360 - diff)
    return diff

def reax_angle_restraint(positions, idx_1, idx_2, idx_3, frc_1, frc_2, targets):
    """
    ReaxFF style restraint for angle difference
    Erestraint= Force1*{1.0-exp(Force2*(distance-target_distance)^2}
    """

    dr_12 = positions[idx_1] - positions[idx_2]
    dr_32 = positions[idx_3] - positions[idx_2]

    cur_angle = jax.vmap(angle)(dr_12, dr_32)
    cur_angle = cur_angle * rdndgr

    rest_pot = jnp.sum(mask * frc_1 *
                     (1.0 - jnp.exp(-frc_2 * (cur_angle - targets)**2)))
    return rest_pot

def reax_dihedral_restraint(positions, idx_1, idx_2, idx_3, frc_1, frc_2, targets):
    """
    ReaxFF style restraint for dihedral difference
    Erestraint= Force1*{1.0-exp(Force2*(distance-target_distance)^2}
    """

    cur_angle = jax.vmap(torsion)(positions[idx_1], positions[idx_2], 
                                  positions[idx_3], positions[idx_4])
    cur_angle = cur_angle * rdndgr

    rest_pot = jnp.sum(mask * frc_1 *
                     (1.0 - jnp.exp(-frc_2 * (cur_angle - targets)**2)))
    return rest_pot

# General function to calculate restraint energies
def calculate_restraint_energy(positions, restraints, dist_fn, rest_mask, geo_method="harmonic"):
    total_energy = 0.0

    # TODO take a closer look at this:
    # https://www.charmm-gui.org/charmmdoc/cons.html
    # it seems like there should be some way of condensing all of these cases into one function
    # plus the separate rmsg function
    # also look at this
    # https://manual.gromacs.org/2024.2/reference-manual/algorithms/constraint-algorithms.html
    # and this
    # https://www.r-ccs.riken.jp/labs/cbrt/tutorials2022/tutorial-9-1/

    # rest mask has 1 for terms that are being calculated, precheck against None
    # e.g. bond/angle/torsion/rmsd/positional/bestfit/com
    # geo method either "harmonic", "reaxff", or "amber", must be static arg, look at decorator
    # this allows switching terms on and off and is still static for jit purposes

    # Bonds
    if rest_mask[0]:
        res = restraints.harmonic_bonds
        cur_dists = jax.vmap(dist_fn)(positions[res.idx_1] - positions[res.idx_2])
    
        if geo_method == "harmonic":
            total_energy += harmonic_restraint(res.value(positions), res.target, res.k)
        elif geo_method == "reaxff":
            total_energy += reax_restraint(res.value(positions), res.target, res.force1, res.force2)
        elif geo_method == "amber":
            total_energy += amber_well_restraint(res.value(positions), res.r1, res.r2, res.r3, res.r4, res.k)

    # Angles
    #if rest_mask[1]:
    res = restraints.harmonic_angle
    dr_12 = positions[res.idx_1] - positions[res.idx_2]
    dr_32 = positions[res.idx_3] - positions[res.idx_2]

    cur_angle = jax.vmap(angle)(dr_12, dr_32)
    cur_angle = cur_angle * rdndgr
    total_energy += harmonic_restraint(res.value(positions), res.target, res.k)

    # Torsions
    total_energy += harmonic_restraint(res.value(positions), res.target, res.k)

    # CHARMM constraints information: https://www.charmm-gui.org/charmmdoc/cons.html

    # CHARMM ABSOLUTE positional restraint
    aligned_pos = restraints.rmsd.align(positions)
    total_energy += rmsd_restraint(aligned_pos, restraints.rmsd.reference, restraints.rmsd.k)

    # CHARMM BESTFIT positional restraint
    aligned_pos = restraints.bestfit.align(positions)
    total_energy += bestfit_restraint(aligned_pos, restraints.bestfit.reference, restraints.bestfit.k)

    # CHARMM RELATIVE positional restraint

    # COM/centroid harmonic restraint, need to break out to harmonic
    # TODO make this a general centroid, but add a mass weighting term
    total_energy += com_harmonic_restraint(positions, restraints.com.masses, restraints.com.reference_com, restraints.com.k)
    # add reax and amber style force constants

    return total_energy