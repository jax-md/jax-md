'''
File with helper functions for loading dataclasses and handling I/O

Classes:

Functions:

Author: William Betancourt
'''
import jax
import jax.numpy as jnp
import numpy as onp
import openmm.app as app
import sys
from jax_md.reaxff.reaxff_helper import safe_sqrt
from jax_md.amber.amber_forcefield import AmberForceField, FFQForceField
from jax_md import space, util
from frozendict import frozendict
from enum import IntEnum, EnumMeta, auto

# TODO sanitize order of import statements, system -> JAX -> external libs -> internal files
# TODO add code to obtain any physical quantities of interest
# TODO add utility function to generate box vectors with buffer from system
# TODO generally sanitize this part of the code for omm vs jax
# TODO jax will be slower in any non jit scenario so only use omm
# right up until either energy calculation/md or optimization
# there is also the added benefit of no tracing for static values
# TODO there should be a flag in the dataclasses that contain their context (jnp or onp)
# TODO add dtype toggles to all of these
# e.g. if jax.config("jax_enable_x64") -> 64
# or define top level file that should be run first
# maybe amber.py and then set enums for types - look at jax md for inspiration
# TODO need to implement mixed precision accumulation and toggle
# TODO look for any missing constants of relevance and move all constant definitions to a special file
# OMM and others have lists of these that may be worth keeping on hand
# TODO not declaring data types explicitly is a solution to multi process corruption when doing imports
# there should be a way to set up lazy evaluation of constant dtypes to avoid this

KCAL_TO_KJ = 4.184
ONE_4PI_EPS = 138.935456
CHARGE_SCALE_AMBER = 18.2223

# Valid options for different runtime settings
# TODO consider turning these into enums
# nonbonded_methods = ["NoCutoff", "PME"]
# scipy_minimize_options = ["L-BFGS-B"]
# ensemble = ["NVE", "NVT", "NPT", "MIN"]
# min_method = [None, "grad", "bfgs", "dlfind"]
# charge_method = ["GAFF", "FFQ"]

class CustomEnumMeta(EnumMeta):
    special_mapping = {"n+": "n_"}  # Custom mapping

    def __getitem__(cls, name):
        name = cls.special_mapping.get(name, name)  # Translate if needed
        return super().__getitem__(name)

    def __contains__(cls, name):
        return name in cls.__members__ or name in cls.special_mapping  # Support special cases

class GAFFTYPES(IntEnum, metaclass=CustomEnumMeta):
    # AMBER General Force Field for organic molecules (Version 2.2.20, March 2021)
    c  = 0 # Sp2 C carbonyl group 
    cs = auto() # Sp2 C in c=S
    c1 = auto() # Sp C
    c2 = auto() # Sp2 C  
    c3 = auto() # Sp3 C
    ca = auto() # Sp2 C in pure aromatic systems
    cp = auto() # Head Sp2 C that connect two rings in biphenyl sys. 
    cq = auto() # Head Sp2 C that connect two rings in biphenyl sys. identical to cp 
    cc = auto() # Sp2 carbons in non-pure aromatic systems
    cd = auto() # Sp2 carbons in non-pure aromatic systems, identical to cc
    ce = auto() # Inner Sp2 carbons in conjugated systems
    cf = auto() # Inner Sp2 carbons in conjugated systems, identical to ce
    cg = auto() # Inner Sp carbons in conjugated systems
    ch = auto() # Inner Sp carbons in conjugated systems, identical to cg
    cx = auto() # Sp3 carbons in triangle systems
    cy = auto() # Sp3 carbons in square systems
    c5 = auto() # Sp3 carbons in five-memberred rings
    c6 = auto() # Sp3 carbons in six-memberred rings
    cu = auto() # Sp2 carbons in triangle systems
    cv = auto() # Sp2 carbons in square systems
    cz = auto() # Sp2 carbon in guanidine group
    h1 = auto() # H bonded to aliphatic carbon with 1 electrwd. group  
    h2 = auto() # H bonded to aliphatic carbon with 2 electrwd. group 
    h3 = auto() # H bonded to aliphatic carbon with 3 electrwd. group 
    h4 = auto() # H bonded to non-sp3 carbon with 1 electrwd. group 
    h5 = auto() # H bonded to non-sp3 carbon with 2 electrwd. group 
    ha = auto() # H bonded to aromatic carbon  
    hc = auto() # H bonded to aliphatic carbon without electrwd. group 
    hn = auto() # H bonded to nitrogen atoms
    ho = auto() # Hydroxyl group
    hp = auto() # H bonded to phosphate 
    hs = auto() # Hydrogen bonded to sulphur 
    hw = auto() # Hydrogen in water 
    hx = auto() # H bonded to C next to positively charged group  
    f  = auto() # Fluorine
    cl = auto() # Chlorine 
    br = auto() # Bromine 
    i  = auto() # Iodine 
    n  = auto() # Sp2 nitrogen in amide groups
    n1 = auto() # Sp N  
    n2 = auto() # aliphatic Sp2 N with two connected atoms 
    n3 = auto() # Sp3 N with three connected atoms
    n4 = auto() # Sp3 N with four connected atoms 
    na = auto() # Sp2 N with three connected atoms 
    nb = auto() # Sp2 N in pure aromatic systems 
    nc = auto() # Sp2 N in non-pure aromatic systems
    nd = auto() # Sp2 N in non-pure aromatic systems, identical to nc
    ne = auto() # Inner Sp2 N in conjugated systems
    nf = auto() # Inner Sp2 N in conjugated systems, identical to ne
    nh = auto() # Amine N connected one or more aromatic rings 
    no = auto() # Nitro N  
    ns = auto() # amind N, with 1 attached hydrogen atom
    nt = auto() # amide N, with 2 attached hydrogen atoms
    nx = auto() # like n4, but only has one hydrogen atom
    ny = auto() # like n4, but only has two hydrogen atoms
    nz = auto() # like n4, but only has three three hydrogen atoms
    n_ = auto() # NH4+
    nu = auto() # like nh, but only has one attached hydrogen atom
    nv = auto() # like nh, but only has two attached hydrogen atoms
    n7 = auto() # like n3, but only has one attached hydrogen atom 
    n8 = auto() # like n3, but only has two attached hydrogen atoms
    n9 = auto() # NH3eeeeeeeeeeeee
    ni = auto() # like n in RG3 
    nj = auto() # like n in RG4
    nk = auto() # like n4/nx/ny in RG3 
    nl = auto() # like n4/nx/ny in RG4
    nm = auto() # like nh in RG3 
    nn = auto() # like nh in RG4
    np = auto() # like n3 in RG3 
    nq = auto() # like n3 in RG4
    n5 = auto() # like n7 in RG3 
    n6 = auto() # like n7 in RG4
    o  = auto() # Oxygen with one connected atom
    oh = auto() # Oxygen in hydroxyl group
    op = auto() # Oxygen in RG3
    oq = auto() # Oxygen in RG4
    os = auto() # Ether and ester oxygen
    ow = auto() # Oxygen in water 
    p2 = auto() # Phosphate with two connected atoms 
    p3 = auto() # Phosphate with three connected atoms, such as PH3
    p4 = auto() # Phosphate with three connected atoms, such as O=P(CH3)2
    p5 = auto() # Phosphate with four connected atoms, such as O=P(OH)3
    pb = auto() # Sp2 P in pure aromatic systems 
    pc = auto() # Sp2 P in non-pure aromatic systems
    pd = auto() # Sp2 P in non-pure aromatic systems, identical to pc
    pe = auto() # Inner Sp2 P in conjugated systems
    pf = auto() # Inner Sp2 P in conjugated systems, identical to pe
    px = auto() # Special p4 in conjugated systems
    py = auto() # Special p5 in conjugated systems
    s  = auto() # S with one connected atom 
    s2 = auto() # S with two connected atom, involved at least one double bond  
    s4 = auto() # S with three connected atoms 
    s6 = auto() # S with four connected atoms 
    sh = auto() # Sp3 S connected with hydrogen 
    ss = auto() # Sp3 S in thio-ester and thio-ether
    sx = auto() # Special s4 in conjugated systems
    sy = auto() # Special s6 in conjugated systems
    sp = auto() # not described
    sq = auto() # not described
    
NUM_GAFF_TYPES = len(GAFFTYPES)

def load_amber_ff(inpcrd_file,
                  prmtop_file,
                  ffq_file=None,
                  nonbonded_method="NoCutoff",
                  charge_method="GAFF",
                  cutoff=.8,
                  ffq_cutoff=1.0,
                  grid_points=64,
                  ewald_alpha=.34,
                  ewald_error=5e-4,
                  dr_threshold=0.0,
                  dtype=jnp.float32):
    '''
    Load AMBER force field parameters from prmtop and preprocess as necessary

    Args:

    Returns:

    Raises:
    '''
    # TODO print all values and dtypes of things in forcefield dataclass for validation
    # TODO insert more shape/bounds/internal consistency checking

    if charge_method in ["FFQ", "FFQMM"] and nonbonded_method != "NoCutoff":
        print("FFQ is not implemented for PBCs/PME")
        #raise Exception("FFQ is not implemented for PBCs/PME")
        sys.exit()

    # TODO these should be returned separately after some thought
    if inpcrd_file == None:
        positions = jnp.zeros(1)
    else:
        inpcrd = app.AmberInpcrdFile(inpcrd_file)
        positions = jnp.array(inpcrd.getPositions(asNumpy=True))

    # TODO consider if keeping track of names to indices or other mapping is useful
    params_to_indices = dict()

    prmtop = app.AmberPrmtopFile(prmtop_file)
    prmtop = prmtop._prmtop

    atom_count = jnp.int32(prmtop.getNumAtoms())

    if nonbonded_method == "PME":
        # TODO add an option to manually provide ewald parameters
        # TODO return error estimates for pme or ewald when calculating this
        # TODO tune grid size to FFT implementation, can also try to target resolution (e.g. 1A)

        # This is a heuristic carried over from OpenMM
        ewald_alpha = 1.0/cutoff * jnp.sqrt(-jnp.log(2.0 * ewald_error))

        # Equation from NonbondedForceImpl.cpp in OpenMM reference platform
        # Heuristic from OpenMM -> round up to nearest power of 2
        grid_points = jnp.ceil(2 * ewald_alpha * box_vectors / (3*jnp.power(ewald_error, 0.2))).astype(jnp.int32)
        # OpenMM requires >= 6 points in each direction
        # grid_points = jnp.where(grid_points > 6, grid_points, 6)

        print(f"[INFO] Calculated Ewald Alpha {ewald_alpha} and grid dimensions {grid_points}")

        # Explicitly calculate list of excluded atoms to construct neighbor mask later

        # TODO this feels cumbersome, possibly only need second half and then don't append to 1-4
        exclusions = []
        exclusions14 = []
        excluded_atom_pairs = set()
        for (iAtom, lAtom, chargeProd, rMin, epsilon, iScee, iScnb) in prmtop.get14Interactions():
            exclusions14.append((iAtom, lAtom))
            excluded_atom_pairs.add(min((iAtom, lAtom), (lAtom, iAtom)))

        excluded_atoms = prmtop.getExcludedAtoms()

        for iAtom in range(atom_count):
            for jAtom in excluded_atoms[iAtom]:
                if min((iAtom, jAtom), (jAtom, iAtom)) in excluded_atom_pairs: continue
                #excluded_atom_pairs.add(min((iAtom, jAtom), (jAtom, iAtom)))
                exclusions.append((iAtom, jAtom))

        exclusions = jnp.concatenate((exclusions, pairs_14))

        # TODO a better error message should exist if non periodic file is passed with PME enabled
        # alternatively, PME is automatically toggled, not necessarily in line with AMBER md.in convention
        # TODO add extra code to ensure that periodic box is strictly greater than 2x cutoff
        box_vectors = jnp.array([v._value for v in prmtop.getBoxBetaAndDimensions()][1:4], dtype=dtype)/10
        pairs = []
    else:
        grid_points = 0
        ewald_alpha = 0.0
        ewald_error = 0.0
        exclusions = jnp.zeros((1,2), dtype=jnp.int32)
        # TODO should displace the center of mass to the center of
        # the box if a more rigorous solution is not implemented
        # at the very least, a warning if a molecule is bigger than this
        # or within a certain distance of the fake box
        box_vectors = jnp.array([999.0, 999.0, 999.0], dtype=dtype)
        pairs = get_nonbond_pairs(prmtop)

    pairs = jnp.array(pairs, dtype=jnp.int32)
    

    ### General parameter loading
    # TODO replace all [] lists with jnp.zeros() of the right shape for batching dataclasses
    # TODO create readme file with mapping for params and flesh out final params_to_indices
    # TODO handle systems with no bond, angle, torsion, pair, exclusion, etc
    # TODO consider if approach for above works for structures with exactly 1 torsion/exclusion/etc
    # TODO make sure all masking is standardized as -1 and accounted for accordingly in energy function
    # TODO better yet, create a general mask/filtering class to make this more extensible

    ### Bonded parameters

    bonds = jnp.array(prmtop.getBondsWithH() + prmtop.getBondsNoH(), dtype=dtype)

    bond_idx = bonds[:, 0:2].astype(jnp.int32)
    bond_k = bonds[:, 2]
    bond_len = bonds[:, 3]

    # for i in range(len(bond_idx)):
    #     ind1 = bond_idx[i, 0]
    #     ind2 = bond_idx[i, 1]
    #     params_to_indices[(0,i,0)] = ("bond_k", (ind1, ind2))
    #     params_to_indices[(0,i,1)] = ("bond_l", (ind1, ind2))

    ### Angle parameters

    angles = jnp.array(prmtop.getAngles(), dtype=dtype)

    angle_idx = angles[:, 0:3].astype(jnp.int32)
    angle_k = angles[:, 3]
    angle_equil = angles[:, 4]

    # for i in range(len(angle_idx)):
    #     ind1 = angle_idx[i, 0]
    #     ind2 = angle_idx[i, 1]
    #     ind3 = angle_idx[i, 2]
    #     params_to_indices[(1,i,0)] = ("angle_k", (ind1, ind2, ind3))
    #     params_to_indices[(1,i,1)] = ("angle_equil", (ind1, ind2, ind3))

    ### Torsion parameters

    dihedrals = jnp.array(prmtop.getDihedrals(), dtype=dtype)

    if dihedrals.ndim == 2:
        torsion_idx = dihedrals[:, 0:4].astype(jnp.int32)
        torsion_k = dihedrals[:, 4]
        torsion_phase = dihedrals[:, 5]
        torsion_period = dihedrals[:, 6].astype(jnp.int32)

        # for i in range(len(torsion_idx)):
        #     ind1 = torsion_idx[i, 0]
        #     ind2 = torsion_idx[i, 1]
        #     ind3 = torsion_idx[i, 2]
        #     ind4 = torsion_idx[i, 3]
        #     params_to_indices[(2,i,0)] = ("torsion_k", (ind1, ind2, ind3, ind4))
        #     params_to_indices[(2,i,1)] = ("torsion_phase", (ind1, ind2, ind3, ind4))
        #     params_to_indices[(2,i,2)] = ("torsion_period", (ind1, ind2, ind3, ind4))
    else:
        torsion_idx = jnp.zeros((1,4), dtype=jnp.int32)-1
        torsion_k = jnp.zeros((1,), dtype=dtype)
        torsion_phase = jnp.zeros((1,), dtype=dtype)
        torsion_period = jnp.zeros((1,), dtype=dtype)

    ### Nonbonded interactions

    nonbonds = jnp.array(prmtop.getNonbondTerms(), dtype=dtype)

    sigma = nonbonds[:, 0] * 2**(-1./6.) * 2.0
    epsilon = nonbonds[:, 1]

    charges = jnp.array(prmtop.getCharges(), dtype=dtype)

    # for i in range(len(nonbonds)):
    #     params_to_indices[(3,i,0)] = ("sigma", (i,))
    #     params_to_indices[(3,i,1)] = ("epsilon", (i,))

    ### 1-4 interactions

    inter_14 = jnp.array(prmtop.get14Interactions(), dtype=dtype)

    # Deal with topologies that have no 1-4 interactions
    if inter_14.ndim == 2:
        pairs_14 = inter_14[:, 0:2].astype(jnp.int32)
        charges_14 = inter_14[:, 2]
        sigma_14 = inter_14[:, 3] * 2**(-1./6.)
        epsilon_14 = inter_14[:, 4]
        scee_14 = inter_14[:, 5]
        scnb_14 = inter_14[:, 6]

        # Precomputing this for efficiency
        charges_14 = charges_14 / scee_14
        epsilon_14 = epsilon_14 / scnb_14
    else:
        # TODO maybe replace nones where there should be arrays with np.array([])
        # creates array with (0,) shape usually, or just do 0s if i need a certain shape for broadcast
        pairs_14 = jnp.zeros((1,2), dtype=jnp.int32)-1 # TODO make sure all masking is standardized as -1
        charges_14 = jnp.zeros((1,), dtype=dtype)
        sigma_14 = jnp.zeros((1,), dtype=dtype)
        epsilon_14 = jnp.zeros((1,), dtype=dtype)
        scee_14 = jnp.zeros((1,), dtype=dtype)
        scnb_14 = jnp.zeros((1,), dtype=dtype)

    # TODO in the event these are changed in the optimizer
    # need to include code to remodify the charge and epsilon values
    # this might also just be worth throwing out and integrating more tightly with the
    # direct sum the way openmm does it

    # this might just generally need to be rewritten, indexing 1/4 interactions and applying
    # scaling isn't particularly expensive to begin with but storing things in this format makes
    # parameter updates far more difficult

    # for i in range(len(inter_14)):
    #     params_to_indices[(4,i,0)] = ("scee_14", (ind1, ind2))
    #     params_to_indices[(4,i,1)] = ("scnb_14", (ind1, ind2))

    ### Miscelaneous fields to store from prmtop

    masses = jnp.array(prmtop.getMasses(), dtype=dtype)
    # AMBER_ATOM_TYPES
    # ATOM_NAME
    # ATOMIC_NUMBER
    # residues? cap info? radii?
    # urey-bradley, improper, and cmaps for charmm?

    ### Dispersion correction for periodic systems with LJ cutoff
    # TODO consider carefully if the gradient of this should be excluded
    # it doesn't look like omm considers this in the force calculation
    # TODO move this to PME helper function
    if nonbonded_method == "PME":
        # TODO might be able to simplify this using triu k=0
        # also consider if this can be done with less memory
        disp_coefs = jnp.stack([sigma, epsilon], axis=1)
        values, count = jnp.unique(disp_coefs, axis=0, return_counts=True)
        sigma2 = values[:, 0] * values[:, 0]
        sigma6 = sigma2 * sigma2 * sigma2
        count_s = count * (count + 1)/2
        sum1 = jnp.sum(count_s*values[:, 1]*sigma6*sigma6)
        sum2 = jnp.sum(count_s*values[:, 1]*sigma6)

        sig_mesh = jnp.triu(jnp.array(jnp.meshgrid(values[:, 0], values[:, 0])), k=1).T.reshape(-1, 2)
        eps_mesh = jnp.triu(jnp.array(jnp.meshgrid(values[:, 1], values[:, 1])), k=1).T.reshape(-1, 2)
        count_mesh = jnp.triu(jnp.array(jnp.meshgrid(count, count)), k=1).T.reshape(-1, 2)

        sig_c = 0.5*jnp.sum(sig_mesh, axis=1)
        eps_c = jnp.sqrt(eps_mesh[:, 0]*eps_mesh[:, 1])
        count_c = jnp.prod(count_mesh, axis=1)

        sigma2 = sig_c*sig_c
        sigma6 = sigma2*sigma2*sigma2

        sum1 = sum1 + jnp.sum(count_c*eps_c*sigma6*sigma6)
        sum2 = sum2 + jnp.sum(count_c*eps_c*sigma6)

        sum3 = 0.0

        sum1 = sum1 / ((atom_count*(atom_count+1))/2)
        sum2 = sum2 / ((atom_count*(atom_count+1))/2)
        sum3 = sum3 / ((atom_count*(atom_count+1))/2)

        disp_coef = 8*atom_count*atom_count*jnp.pi*(sum1/(9*jnp.power(cutoff, 9)) - sum2/(3*jnp.power(cutoff, 3)) + sum3)
    else:
        disp_coef = 0.0

    ### FFQ Input Parsing
    # TODO add reax forcefield support for this or extend normal prmtop format
    # and write appropriate conversion tools for input/output
    # TODO also move this to another function and return FFQ dataclass
    gamma = onp.zeros(NUM_GAFF_TYPES, dtype=dtype)
    electronegativity = onp.zeros(NUM_GAFF_TYPES, dtype=dtype)
    hardness = onp.zeros(NUM_GAFF_TYPES, dtype=dtype)
    species = []

    if charge_method in ["FFQ", "FFQMM"]:
        with open(ffq_file, 'r') as f:
            lines = f.readlines()

        for i in range(3,len(lines)):
            line = lines[i].strip()
            split_line = line.split()
            atom_name = split_line[0]
            if atom_name in GAFFTYPES.__members__:
                idx = GAFFTYPES[atom_name]
                gamma[idx] = float(split_line[1])
                electronegativity[idx] = float(split_line[2])
                hardness[idx] = float(split_line[3])

        # TODO consider changing these to canonical names - gamma, chi, eta respectively
        species = jnp.array([GAFFTYPES[name.lower()] for name in prmtop._raw_data['AMBER_ATOM_TYPE']], dtype=jnp.int32)

        for i in range(len(gamma)):
            params_to_indices[(5,i,0)] = ("gamma", (i,))
            params_to_indices[(5,i,1)] = ("electronegativity", (i,))
            params_to_indices[(5,i,2)] = ("hardness", (i,))

        gamma = jnp.array(gamma, dtype=dtype)
        electronegativity = jnp.array(electronegativity, dtype=dtype)
        hardness = jnp.array(hardness, dtype=dtype)

        num_solute_residues = 0
        for residue in prmtop._raw_data['RESIDUE_LABEL']:
            if residue != "WAT":
                num_solute_residues += 1
            # TODO consider if there can be a case where you get a pattern like MOL WAT MOL WAT
            # if so, create flag and make sure you don't accidently get the wrong number of solute atoms

        if len(prmtop._raw_data['RESIDUE_POINTER']) > 1 and charge_method == "FFQ":
            solute_cut = jnp.int32(prmtop._raw_data['RESIDUE_POINTER'][num_solute_residues])-1
        else:
            solute_cut = atom_count
    else:
        solute_cut = None

    # must be frozendict to be implicitly treated as static during JIT compilation
    # this is important for compatibility with JAX transformations
    params_to_indices = frozendict(params_to_indices)

    ### Create and return final force field objects

    #TODO convert to init_from_arg_dict rather than explicit construction
    amber_ff = AmberForceField(
                                name=0.0,
                                atom_types=0.0,
                                total_charge=0.0,
                                params_to_indices=params_to_indices,
                                bond_restraints=0.0,
                                angle_restraints=0.0,
                                torsion_restraints=0.0,
                                atom_count=atom_count,
                                positions=positions,
                                box_vectors=box_vectors,
                                masses=masses,
                                cutoff=jnp.float32(cutoff).astype(dtype),
                                nbr_list=0.0,
                                grid_points=grid_points,
                                ewald_alpha=ewald_alpha,
                                ewald_error=ewald_error,
                                dr_threshold=jnp.float32(dr_threshold).astype(dtype),
                                exclusions=exclusions,
                                bond_idx=bond_idx,
                                bond_k=bond_k,
                                bond_len=bond_len,
                                angle_idx=angle_idx,
                                angle_k=angle_k,
                                angle_equil=angle_equil,
                                torsion_idx=torsion_idx,
                                torsion_k=torsion_k,
                                torsion_phase=torsion_phase,
                                torsion_period=torsion_period,
                                pairs=pairs,
                                sigma=sigma,
                                epsilon=epsilon,
                                charges=charges,
                                pairs_14=pairs_14,
                                charges_14=charges_14,
                                sigma_14=sigma_14,
                                epsilon_14=epsilon_14,
                                scee_14=scee_14,
                                scnb_14=scnb_14,
                                disp_coef=disp_coef,
                                gamma=gamma,
                                electronegativity=electronegativity,
                                hardness=hardness,
                                species=species,
                                name_to_index=0.0,
                                solute_cut=solute_cut)

    return amber_ff

def get_nonbond_pairs(prmtop):
    '''
    Get nonbonded pair information

    Args:

    Returns:

    Raises:
    '''
    num_excl_atoms = prmtop._raw_data['NUMBER_EXCLUDED_ATOMS']
    excl_atoms_list = prmtop._raw_data['EXCLUDED_ATOMS_LIST']
    total = 0
    numAtoms = int(prmtop._raw_data['POINTERS'][0])
    nonbond_pairs = []

    for iatom in range(numAtoms):
        index0 = total
        n = int (num_excl_atoms[iatom])
        total += n
        index1 = total
        excl_list = []
        for jatom in excl_atoms_list[index0:index1]:
            j = int(jatom) - 1
            excl_list.append(j)

        for jatom in range (iatom+1, numAtoms):
            if jatom in excl_list:
                continue
            nonbond_pairs.append ( [iatom, jatom] )

    return nonbond_pairs

# TODO would @partial(jax.vmap) or @partial(jnp.vectorize) on both of these be useful?
def angle(dr_12, dr_32):
    '''
    Calculate the angle between 3 points

    Args:

    Returns:

    Raises:
    '''

    d_12 = jnp.linalg.norm(dr_12 + 1e-7)
    d_32 = jnp.linalg.norm(dr_32 + 1e-7)
    cos_angle = jnp.dot(dr_12, dr_32) / (d_12 * d_32)
    return util.safe_mask((cos_angle < 1) & (cos_angle > -1), jnp.arccos, cos_angle)

def torsion(p1, p2, p3, p4):
    '''
    Calculate the dihedral angle between 4 points
    Praxeolitic formula
    Taken from: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    
    Args:

    Returns:

    Raises:
    '''

    # TODO a more rigorous way of doing this for periodic settings should be added
    # it's not very user friendly, but passing a set of displacements to the function may
    # be a more robust approach than calculating the non periodic displacements internally
    b0 = -1.0*(p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= jnp.linalg.norm(b1 + 1e-10)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - jnp.dot(b0, b1)*b1
    w = b2 - jnp.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)
    r = jnp.arctan2(y, x+1e-10)

    return r

def write_ff():
    '''
    Writes out JAX force field object as AMBER compliant prmtop

    Args:

    Returns:

    Raises:
    '''
    return

def omm_system_to_ff():
    '''
    Converts an OpenMM Topology object to a JAX dataclass

    Args:

    Returns:

    Raises:
    '''
    return

def parmed_system_to_ff():
    '''
    Converts a ParmEd Topology object to a JAX dataclass

    Args:

    Returns:

    Raises:
    '''
    return

def openff_system_to_ff():
    '''
    Converts an OpenFF Topology object to a JAX dataclass

    Args:

    Returns:

    Raises:
    '''
    return

def omm_xml_to_ff():
    '''
    Converts an OpenMM / OpenFF XML Force Field to a JAX template

    Args:

    Returns:

    Raises:
    '''
    return

def calculate_distances():
    '''
    Calculate all relevant distances, angles, and torsions
    Provides flexibility in recomputing observables multiple times
    and abstracts this away from the core energy function

    Args:

    Returns:

    Raises:
    '''
    return

def load_ffq_ff(ffq_file, dtype):
    '''
    Load FFQ parameters to construct charge matrix

    Args:

    Returns:

    Raises:
    '''
    gamma = onp.zeros(NUM_GAFF_TYPES, dtype=dtype)
    electronegativity = onp.zeros(NUM_GAFF_TYPES, dtype=dtype)
    hardness = onp.zeros(NUM_GAFF_TYPES, dtype=dtype)

    with open(ffq_file, 'r') as f:
        lines = f.readlines()

    for i in range(3,len(lines)):
        line = lines[i].strip()
        split_line = line.split()
        atom_name = split_line[0]
        if atom_name in GAFFTYPES:
            idx = GAFFTYPES[atom_name]
            gamma[idx] = float(split_line[1])
            electronegativity[idx] = float(split_line[2])
            hardness[idx] = float(split_line[3])

    gamma = jnp.array(gamma, dtype=dtype)
    electronegativity = jnp.array(electronegativity, dtype=dtype)
    hardness = jnp.array(hardness, dtype=dtype)
    
    ffq_ff = FFQForceField(gamma=gamma,
                            electronegativity=electronegativity,
                            hardness=hardness)
    
    return ffq_ff

def calculate_pme_params():
    '''
    Calculate PME parameters according to OpenMM convention

    Args:

    Returns:
        PME parameter object

    Raises:
    '''
    return

def modify_prmtop():
    '''
    Interface to ParmEd to modify .prmtop files

    Args:
        param_name
        value_dict
        log_file

    Returns:

    Raises:
    '''
    # addDihedral
    # addLJType
    # changeLJ14Pair
    # changeLJPair
    # changeLJSingleType -> printLJTypes
    # deleteBond
    # deleteDihedral
    # setAngle
    # setBond
    return

def input_file_parser():
    '''
    Load coordinates file based on format

    Args:

    Returns:

    Raises:
    '''
    # TODO include support for more file formats
    # e.g. bgf, xyz, gro/top, charmm coord/rst, pdb, etc possibly by
    # extracting file suffix of as in other packages with an option (e.g. grotop_file)
    # also ForceField (openff?) xml, psf, and serialized system/state/integrator xml
    # maybe parmed, rdkit, ase, ask online for other common interoperability tools?
    # also need to add handler for ncrst and getting velocities
    # check omm code to see if there's anything else to be stored from inpcrd object
    return