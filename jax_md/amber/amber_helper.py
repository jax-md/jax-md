# include constants such as unit conversion factors as float32/64
# a -> nm
# kcal/mol -> kj/mol
# coulomb charge multiplier

# angle, and torsion calculations
# physical properties of interest
# utility function to generate box vectors with buffer from system
# prmtop export system
# parmed modification for custom torsions

# input parser for inpcrd and prmtop files
# possibly also add compatibility layer for parmed structures to enable OpenMM support

from amber_forcefield import AmberForceField

KCAL_TO_KJ = 4.184
ONE_4PI_EPS = jnp.float64(138.935456)
CHARGE_SCALE_AMBER = 18.2223

def load_amber_ff(prmtop, isEwald):
    prmtop = omm.app.AmberPrmtopFile(prmtopFile)

    # bonded parameters

    b_k = jnp.array([2.0 * 418.4 * jnp.float64(kval) for kval in prmtop._raw_data['BOND_FORCE_CONSTANT']])
    #A -> NM
    b_l = jnp.array([0.1 * jnp.float64(lval) for lval in prmtop._raw_data['BOND_EQUIL_VALUE']])

    bondidx = prmtop._raw_data["BONDS_INC_HYDROGEN"] + prmtop._raw_data["BONDS_WITHOUT_HYDROGEN"]
    bondidx = jnp.array([int(index) for index in bondidx]).reshape((-1,3))
    b_1_idx = bondidx[:, 0]//3
    b_2_idx = bondidx[:, 1]//3
    #b_idx = bondidx[:, 0:2]//3
    b_prm_idx = bondidx[:, 2]-1

    # angle parameters

    a_k = jnp.array([2.0 * 4.184 * jnp.float64(kval) for kval in prmtop._raw_data['ANGLE_FORCE_CONSTANT']])
    a_eq_ang = jnp.array([jnp.float64(aval) for aval in prmtop._raw_data['ANGLE_EQUIL_VALUE']])

    angleidx = prmtop._raw_data["ANGLES_INC_HYDROGEN"] + prmtop._raw_data["ANGLES_WITHOUT_HYDROGEN"]
    angleidx = jnp.array([int(index) for index in angleidx]).reshape((-1,4))

    a_1_idx = angleidx[:, 0]//3
    a_2_idx = angleidx[:, 1]//3
    a_3_idx = angleidx[:, 2]//3
    #a_idx = angleidx[:, 0:3]//3
    a_prm_idx = angleidx[:, 3]-1

    # torsion parameters

    t_k = jnp.array([4.184 * jnp.float64(kval) for kval in prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT']])

    #TODO: change these to be more similar to either OMM or AMBER
    # where form is 1 + cos(period*theta-phase)
    cos_phase0 = []
    for ph0 in prmtop._raw_data['DIHEDRAL_PHASE']:
        val = jnp.cos (jnp.float64(ph0))
        if val < 0:
            cos_phase0.append (-1.0)
        else:
            cos_phase0.append (1.0)
    t_phase = jnp.array(cos_phase0)

    t_period = []
    for n0 in prmtop._raw_data['DIHEDRAL_PERIODICITY']:
        t_period.append (int(0.5 + jnp.float64(n0)))

    t_period = jnp.array(periodicity)

    torsionidx = prmtop._raw_data["DIHEDRALS_INC_HYDROGEN"] + prmtop._raw_data["DIHEDRALS_WITHOUT_HYDROGEN"]
    torsionidx = jnp.array([int(index) for index in torsionidx]).reshape((-1,5))
    t_1_idx = torsionidx[:, 0]//3
    t_2_idx = torsionidx[:, 1]//3
    t_3_idx = jnp.absolute(torsionidx[:, 2])//3
    t_4_idx = jnp.absolute(torsionidx[:, 3])//3
    #t_idx = angleidx[:, 0:4]//3
    t_prm_idx = torsionidx[:, 4]-1

    # lennard jones parameters
    lj_type = jnp.array([ int(x) - 1 for x in prmtop._raw_data['ATOM_TYPE_INDEX']])
    sigma, epsilon = prm_get_nonbond_terms(prmtop)
    scnb = jnp.array([jnp.float64(x) for x in prmtop._raw_data['SCNB_SCALE_FACTOR']])

    # coulomb parameters
    charges = jnp.array([jnp.float64(x)/18.2223 for x in prmtop._raw_data['CHARGE']])
    scee = jnp.array([jnp.float64(x) for x in prmtop._raw_data['SCEE_SCALE_FACTOR']])

    # nonbonded exclusions and pairs
    pairs14 = prm_get_nonbond14_pairs(prmtop)
    # generating the full n^2 pairs list for large systems will cause memory issues
    if isEwald:
        pairs = []
    else:
        pairs = prm_get_nonbond_pairs(prmtop)

    amber_ff = AmberForceField(b_k=b_k,
                                b_l=b_l,
                                b_1_idx=b_1_idx,
                                b_2_idx=b_2_idx,
                                b_prm_idx=b_prm_idx,
                                a_k=a_k,
                                a_eq_ang=a_eq_ang,
                                a_1_idx=a_1_idx,
                                a_2_idx=a_2_idx,
                                a_3_idx=a_3_idx,
                                a_prm_idx=a_prm_idx,
                                t_k=t_k,
                                t_phase=t_phase,
                                t_period=t_period,
                                t_1_idx=t_1_idx,
                                t_2_idx=t_2_idx,
                                t_3_idx=t_3_idx,
                                t_4_idx=t_4_idx,
                                t_prm_idx=t_prm_idx,
                                pairs=pairs,
                                pairs14=pairs14,
                                lj_type=lj_type,
                                sigma=sigma,
                                epsilon=epsilon,
                                scnb=scnb,
                                charges=charges,
                                scee=scee)

    return amber_ff