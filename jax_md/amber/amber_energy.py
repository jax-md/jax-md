# Author: William Betancourt
# Based on OpenMM, DMFF, JAX MD Pull # 198, JAX Reax-FF

# To-Do List:
# move distance metrics to jax md displacements for internal consistency
# improve docstrings, style, and comments https://numpydoc.readthedocs.io/en/latest/format.html
# modify angle and torsion function to more rationally handle single cases and vmap
#     over vectorized input of displacements rather than sets of points
# create better prmtop handling with dataclass and a native prmtop loader
# lean down energy functions if still possible to offload work to prmtop parser
# implement bond and angle restraints, consider alternative approach to reax style
# make units more internally consistent with either OpenMM or Amber standards
# incorporate native function to assign custom torsion parameters to torsions
#     with common central bond
# fully incorporate jax md neighbor lists for nonbonded calculation along with PME
# create elegant combined energy functions for restrained and unrestrained cases
# integrate jax_md unit testing
# standardize single/double precision switching if jax is configured
# adopt jax_md style of creating potential generators that create neighbor lists and return nb/nrg function

import numpy as np
import jax.numpy as jnp
import jax
import jax_md

from jax_md.reaxff.reaxff_helper import safe_sqrt

def prm_get_nonbond_pairs (prm_raw_data):
    num_excl_atoms = prm_raw_data._raw_data['NUMBER_EXCLUDED_ATOMS']
    excl_atoms_list = prm_raw_data._raw_data['EXCLUDED_ATOMS_LIST']
    total = 0
    numAtoms = int(prm_raw_data._raw_data['POINTERS'][0])
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
        
    return jnp.array(nonbond_pairs)

def prm_get_nonbond_terms (prm_raw_data):
    numTypes = int(prm_raw_data._raw_data['POINTERS'][1])

    LJ_ACOEF = prm_raw_data._raw_data['LENNARD_JONES_ACOEF']
    LJ_BCOEF = prm_raw_data._raw_data['LENNARD_JONES_BCOEF']
    # kcal/mol --> kJ/mol
    energyConversionFactor = 4.184
    # A -> nm
    lengthConversionFactor = 0.1

    sigma = np.zeros (numTypes)
    epsilon = np.zeros (numTypes)

    for i in range(numTypes):

        index = int (prm_raw_data._raw_data['NONBONDED_PARM_INDEX'][numTypes*i+i]) - 1    
        acoef = jnp.float32(LJ_ACOEF[index])
        bcoef = jnp.float32(LJ_BCOEF[index])

        if jnp.isclose(acoef, 0.) or jnp.isclose(bcoef, 0.):
            sig = 1.0
            eps = 0.0
        else:
            sig = (acoef/bcoef)**(1.0/6.0)
            eps = 0.25*bcoef*bcoef/acoef

        # print("get terms")    
        # print("acoef", acoef)
        # print("bcoef", bcoef)
        # print("index", index)
        # print("sig", sig)

        # try:
        #     print("get terms")
        #     print("acoef", acoef)
        #     print("bcoef", bcoef)
        #     print("index", index)
        #     sig = (acoef/bcoef)**(1.0/6.0)
        #     print("sig", sig)
        #     eps = 0.25*bcoef*bcoef/acoef
        # except ZeroDivisionError:
        #     sig = 1.0
        #     eps = 0.0

        sigma[i] = sig*lengthConversionFactor
        epsilon[i] = eps*energyConversionFactor

    return jnp.array(sigma), jnp.array(epsilon)

def prm_get_nonbond14_pairs (prm_raw_data):
    dihedralPointers = prm_raw_data._raw_data["DIHEDRALS_WITHOUT_HYDROGEN"] + \
                            prm_raw_data._raw_data["DIHEDRALS_INC_HYDROGEN"] 

    nonbond14_pairs = []
    
    for ii in range (0, len(dihedralPointers), 5):
        if int(dihedralPointers[ii+2])>0 and int(dihedralPointers[ii+3])>0:
            iAtom = int(dihedralPointers[ii])//3
            lAtom = int(dihedralPointers[ii+3])//3
            parm_idx = int(dihedralPointers[ii+4]) - 1
            
            nonbond14_pairs.append ((iAtom, lAtom, parm_idx))

    return jnp.array(nonbond14_pairs)

def distance(p1v, p2v=None, box=None):
    dR = p1v-p2v
    dv = jnp.mod(dR + box * jnp.float32(0.5), box) - jnp.float32(0.5) * box
    return safe_sqrt(jnp.sum(jnp.power(dv, 2), axis=1))

#a dot b = len(a) * len(b) * cos theta
#theta = arccos((a dot b)/(len(a) * len(b)))
def angle(p1v, p2v, p3v, box):
    d12 = p2v-p1v
    d12 = jnp.where(d12 > 0.5 * box, d12-box, d12)
    d12 = jnp.where(d12 < -0.5 * box, d12+box, d12)
    d23 = p2v-p3v
    d23 = jnp.where(d23 > 0.5 * box, d23-box, d23)
    d23 = jnp.where(d23 < -0.5 * box, d23+box, d23)
    #print(jnp.reshape(distance(p1v, p2v, box), (-1, 1)))
    #jnp.where(x == 0, 1.0, jnp.sin(x) / x)
    d1v = jnp.reshape(distance(p1v, p2v, box), (-1, 1))
    d2v = jnp.reshape(distance(p2v, p3v, box), (-1, 1))

    #v1 = (d12) / safe_mask(d1v > 0, lambda x: x, d1v, 1)
    v1 = (d12) / jnp.where(d1v > 0, d1v, 1)
    v2 = (d23) / jnp.where(d2v > 0, d2v, 1)
    vxx = v1[:, 0] * v2[:, 0]
    vyy = v1[:, 1] * v2[:, 1]
    vzz = v1[:, 2] * v2[:, 2]
    #print("vxx", vxx)
    #print("vyy", vyy)
    #print("vzz", vzz)
    #print("f32 xx", jnp.float32(vxx))

    #TODO: figure out a better way of doing this
    #for some examples, components can work out to just under -1
    #this is likely due to position in f16 to math in f32
    component_sum = vxx + vyy + vzz
    component_sum = jnp.where(component_sum > 1., 1., component_sum)
    component_sum = jnp.where(component_sum < -1., -1., component_sum)

    return jnp.arccos(component_sum)

def torsion(p1v, p2v, p3v, p4v, box):
    b1, b2, b3 = p2v - p1v, p3v - p2v, p4v - p3v
    b1 = jnp.mod(b1 + box * jnp.float32(0.5), box) - jnp.float32(0.5) * box
    b2 = jnp.mod(b2 + box * jnp.float32(0.5), box) - jnp.float32(0.5) * box
    b3 = jnp.mod(b3 + box * jnp.float32(0.5), box) - jnp.float32(0.5) * box

    c1 = jax.vmap(jnp.cross, (0, 0))(b2, b3)
    c2 = jax.vmap(jnp.cross, (0, 0))(b1, b2)

    p1 = (b1 * c1).sum(-1)
    p1 = p1 * safe_sqrt((b2 * b2).sum(-1))
    p2 = (c1 * c2).sum(-1)
    #print("p1", p1)
    #print("p2", p2)
    #p1 = jnp.where(p1 <= 0, 1, p1)
    #p2 = jnp.where(p2 <= 0, 1, p2)
    #TODO: This may not be a robust approach to fixing this issue
    p2 = jnp.where(jnp.isclose(p2, 0.), 1, p2)
    #print("p2 post filter", p2)

    r = jax.vmap(jnp.arctan2, (0, 0))(p1, p2)
    return r

def torsion_single(p1v, p2v, p3v, p4v, box):
    b1, b2, b3 = p2v - p1v, p3v - p2v, p4v - p3v
    b1 = jnp.mod(b1 + box * jnp.float32(0.5), box) - jnp.float32(0.5) * box
    b2 = jnp.mod(b2 + box * jnp.float32(0.5), box) - jnp.float32(0.5) * box
    b3 = jnp.mod(b3 + box * jnp.float32(0.5), box) - jnp.float32(0.5) * box

    c1 = jnp.cross(b2, b3)
    c2 = jnp.cross(b1, b2)

    p1 = (b1 * c1).sum(-1)
    p1 = p1 * safe_sqrt((b2 * b2).sum(-1))
    p2 = (c1 * c2).sum(-1)
    #print("p1", p1)
    #print("p2", p2)
    #p1 = jnp.where(p1 <= 0, 1, p1)
    #p2 = jnp.where(p2 <= 0, 1, p2)
    p2 = jnp.where(jnp.isclose(p2, 0.), 1, p2)
    #print("p2 post filter", p2)

    r = jnp.arctan2(p1, p2)
    return r

def bond_init(prmtop):
    #assuming kcal/mol/A or mulitple by 418.4 for kj/mol/nm?
    #multiply by 2 for openmm standard 0.5 * k(r-r0)^2
    #see http://docs.openmm.org/6.2.0/userguide/theory.html
    #kcal/mol/A -> kJ/mol/nm
    k = jnp.array([2.0 * 418.4 * jnp.float32(kval) for kval in prmtop._raw_data['BOND_FORCE_CONSTANT']])
    #A -> NM
    l = jnp.array([0.1 * jnp.float32(lval) for lval in prmtop._raw_data['BOND_EQUIL_VALUE']])

    bondidx = prmtop._raw_data["BONDS_INC_HYDROGEN"] + prmtop._raw_data["BONDS_WITHOUT_HYDROGEN"]
    bondidx = jnp.array([int(index) for index in bondidx]).reshape((-1,3))
    b1_idx = bondidx[:, 0]//3
    b2_idx = bondidx[:, 1]//3
    param_index = bondidx[:, 2]-1

    return (k, l, b1_idx, b2_idx, param_index)

def bond_get_energy(positions, box, prms):
    k, l, b1_idx, b2_idx, param_index = prms

    mask = param_index >= 0
    
    p1 = positions[b1_idx]
    p2 = positions[b2_idx]
    kprm = k[param_index]
    lprm = l[param_index]
    dist = distance(p1, p2, box)

    # energies = jnp.nan_to_num(0.5 * kprm * jnp.power((dist - lprm), 2))

    # filtered_energies = jnp.where(param_index < 0, 0, energies)
    #print("Bond Energies Unfiltered", 0.5 * kprm * jnp.power((dist - lprm), 2))
    # print("Bond Energies", energies)
    # print("Bond Filtered Energies", filtered_energies)
    #print("Bond prm Idx", param_index)
    #print("Bond 1 Idx", b1_idx)
    #print("Bond 2 Idx", b2_idx)
    # print("Bond Distances", dist)
    #print("kprm", kprm)
    #print("lprm", lprm)
    # print("dist-lprm", dist - lprm)
    # print("pwr dist-lprm", jnp.power((dist - lprm), 2))

    #print("B Mask", mask)
    #energies = 0.5 * kprm * jnp.power((dist - lprm), 2)
    #msk_nrg = jax.lax.select(mask, energies, jnp.zeros(mask.shape))
    #return jnp.sum(msk_nrg)
    #return jnp.sum(filtered_energies)
    return jnp.sum(0.5 * kprm * jnp.power((dist - lprm), 2))
    #return jnp.sum(jnp.nan_to_num(0.5 * kprm * jnp.power((dist - lprm), 2)))


def angle_init(prmtop):
    #kcal/mol/rad -> kJ/mol/rad
    k = jnp.array([2.0 * 4.184 * jnp.float32(kval) for kval in prmtop._raw_data['ANGLE_FORCE_CONSTANT']])
    eqangle = jnp.array([jnp.float32(aval) for aval in prmtop._raw_data['ANGLE_EQUIL_VALUE']])

    angleidx = prmtop._raw_data["ANGLES_INC_HYDROGEN"] + prmtop._raw_data["ANGLES_WITHOUT_HYDROGEN"]
    angleidx = jnp.array([int(index) for index in angleidx]).reshape((-1,4))

    a1_idx = angleidx[:, 0]//3
    a2_idx = angleidx[:, 1]//3
    a3_idx = angleidx[:, 2]//3
    param_index = angleidx[:, 3]-1

    return (k, eqangle, a1_idx, a2_idx, a3_idx, param_index)

def angle_get_energy(positions, box, prms):
    k, eqangle, a1_idx, a2_idx, a3_idx, param_index = prms

    p1 = positions[a1_idx]
    p2 = positions[a2_idx]
    p3 = positions[a3_idx]
    kprm = k[param_index]
    thetaprm = eqangle[param_index]
    theta = jnp.where(param_index < 0, 0, angle(p1, p2, p3, box))
    #print('p1', p1)
    #print('p2', p2)
    #print('p3', p3)
    #print('theta', theta)
    #print('kprm')
    #print("Angle Components", 0.5 * kprm * jnp.power((theta - thetaprm), 2))
    return jnp.sum(0.5 * kprm * jnp.power((theta - thetaprm), 2))

def torsion_init(prmtop):
    k = jnp.array([4.184 * jnp.float32(kval) for kval in prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT']])

    cos_phase0 = []
    for ph0 in prmtop._raw_data['DIHEDRAL_PHASE']:
        val = jnp.cos (jnp.float32(ph0))
        if val < 0:
            cos_phase0.append (-1.0)
        else:
            cos_phase0.append (1.0)
    phase = jnp.array(cos_phase0)

    periodicity = []
    for n0 in prmtop._raw_data['DIHEDRAL_PERIODICITY']:
        periodicity.append (int(0.5 + jnp.float32(n0)))

    periodicity = jnp.array(periodicity)

    torsionidx = prmtop._raw_data["DIHEDRALS_INC_HYDROGEN"] + prmtop._raw_data["DIHEDRALS_WITHOUT_HYDROGEN"]
    torsionidx = jnp.array([int(index) for index in torsionidx]).reshape((-1,5))
    t1_idx = torsionidx[:, 0]//3
    t2_idx = torsionidx[:, 1]//3
    t3_idx = jnp.absolute(torsionidx[:, 2])//3
    t4_idx = jnp.absolute(torsionidx[:, 3])//3
    param_index = torsionidx[:, 4]-1

    return (k, phase, periodicity, t1_idx, t2_idx, t3_idx, t4_idx, param_index)

def torsion_get_energy(positions, box, prms):
    k, phase, periodicity, t1_idx, t2_idx, t3_idx, t4_idx, param_index = prms

    mask = param_index >= 0

    p1 = positions[t1_idx]
    p2 = positions[t2_idx]
    p3 = positions[t3_idx]
    p4 = positions[t4_idx]
    kprm = k[param_index]
    phaseprm = phase[param_index]
    period = periodicity[param_index]
    #theta = jnp.where(param_index < 0, 0, torsion(p1, p2, p3, p4, box))
    theta = torsion(p1, p2, p3, p4, box)
    # print("Torsion Components", kprm * (1.0 + jnp.cos(period * theta) * phaseprm))
    # print("Torsion Components Mask", mask * kprm * (1.0 + jnp.cos(period * theta) * phaseprm))
    # print("Torsion Sum", jnp.sum(mask * kprm * (1.0 + jnp.cos(period * theta) * phaseprm)))
    # print("Torsion Sum mask", jnp.sum(kprm * (1.0 + jnp.cos(period * theta) * phaseprm)))

    return jnp.sum(kprm * (1.0 + jnp.cos(period * theta) * phaseprm))


def lj_init(prmtop):
    atom_type = jnp.array([ int(x) - 1 for x in prmtop._raw_data['ATOM_TYPE_INDEX']])
    sigma, epsilon = prm_get_nonbond_terms(prmtop)
    pairs = prm_get_nonbond_pairs(prmtop)
    pairs14 = prm_get_nonbond14_pairs(prmtop)
    scnb = jnp.array([jnp.float32(x) for x in prmtop._raw_data['SCNB_SCALE_FACTOR']])

    return (pairs, pairs14, atom_type, sigma, epsilon, scnb)

def lj_get_energy_nb(positions, box, prms):
    pairs, pairs14, atom_type, sigma, epsilon, scnb = prms

    p1 = positions[pairs[:,0]]
    p2 = positions[pairs[:,1]]
    dist = distance(p1, p2, box)

    at_type_a = atom_type[pairs[:,0]]
    at_type_b = atom_type[pairs[:,1]]
    sig_ab = 0.5*(sigma[at_type_a]+sigma[at_type_b])
    eps_ab = safe_sqrt (epsilon[at_type_a]*epsilon[at_type_b])

    #idrd = jnp.where(dist <= 0, 1, dist)
    idrd = jnp.where(jnp.isclose(dist, 0.), 1, dist)
    idr = (sig_ab/idrd)
    idr2 = idr*idr
    idr6 = idr2*idr2*idr2
    idr12 = idr6*idr6

    # print("Pairs", pairs)
    # print("eps", epsilon)
    # print("epsab comp", eps_ab)
    # print("sigab comp", sig_ab)

    # print("typeasum", jnp.sum(at_type_a))
    # print("epssuma", jnp.sum(epsilon[at_type_a]))
    # print("sigab", jnp.sum(sig_ab))
    # print("epsab", jnp.sum(eps_ab))
    # print("LJ Components", jnp.float32(4)*eps_ab*(idr12-idr6))
    # energies = jnp.float32(4)*eps_ab*(idr12-idr6)
    # print("LJ Energies", energies)

    # filtered_energies = jnp.where(pairs[:,0] < 0, 0, energies)

    # #return jnp.sum(filtered_energies)
    # print("LJ Total E internal", jnp.sum(jnp.float32(4)*eps_ab*(idr12-idr6)))
    #print("epsab", jnp.sum(eps_ab))
    #print("idr12", jnp.sum(idr12))
    return jnp.sum(jnp.float32(4)*eps_ab*(idr12-idr6))

def lj_get_energy_14(positions, box, prms):
    pairs, pairs14, atom_type, sigma, epsilon, scnb = prms

    p1 = positions[pairs14[:,0]]
    p2 = positions[pairs14[:,1]]
    parm_idx = pairs14[:,2]
    dist = distance(p1,p2, box)

    at_type_a = atom_type[pairs14[:,0]]
    at_type_b = atom_type[pairs14[:,1]]
    sig_ab = 0.5*(sigma[at_type_a]+sigma[at_type_b])
    eps_ab = safe_sqrt (epsilon[at_type_a]*epsilon[at_type_b])

    idrd = jnp.where(jnp.isclose(dist, 0.), 1, dist)
    idr = (sig_ab/idrd)
    idr2 = idr*idr
    idr6 = idr2*idr2*idr2
    idr12 = idr6*idr6

    scnb_filtered = jnp.where(jnp.isclose(scnb, 0.), 0., 1.0/scnb)

    #print("LJ 14 Components", 1.0/scnb[parm_idx] * jnp.float32(4)*eps_ab*(idr12-idr6))
    # print("LJ 14 Components", 1.0/scnb[parm_idx] * jnp.float32(4)*eps_ab*(idr12-idr6))
    # print("LJ 14 Components Filtered", scnb_filtered[parm_idx] * jnp.float32(4)*eps_ab*(idr12-idr6))

    # energies = jnp.nan_to_num(1.0/scnb[parm_idx] * jnp.float32(4)*eps_ab*(idr12-idr6))
    # print("LJ 14 Energies", energies)

    # filtered_energies = jnp.where(parm_idx < 0, 0, energies)

    # #return jnp.sum(filtered_energies)
    # print("LJ 14 Total E", jnp.sum(1.0/scnb[parm_idx] * jnp.float32(4)*eps_ab*(idr12-idr6)))

    return jnp.sum(scnb_filtered[parm_idx] * jnp.float32(4)*eps_ab*(idr12-idr6))

def lj_get_energy(positions, box, prms):
    return lj_get_energy_nb(positions, box, prms) + lj_get_energy_14(positions, box, prms)

def coul_init(prmtop):
    # E (kcal/mol) =  332 * q1*q2/r
    # hence 18.2 division
    charges = jnp.array([jnp.float32(x)/18.2223 for x in prmtop._raw_data['CHARGE']])
    pairs = prm_get_nonbond_pairs(prmtop)
    pairs14 = prm_get_nonbond14_pairs(prmtop)
    scee = jnp.array([jnp.float32(x) for x in prmtop._raw_data['SCEE_SCALE_FACTOR']])

    return (charges, pairs, pairs14, scee)

def coul_init_pme(prmtop, boxVectors, cutoff=0.8, grid_points=96, ewald_error=1e-5):
    # E (kcal/mol) =  332 * q1*q2/r
    # hence 18.2 division
    charges = jnp.array([jnp.float32(x)/18.2223 for x in prmtop._raw_data['CHARGE']])
    pairs = prm_get_nonbond_pairs(prmtop)
    pairs14 = prm_get_nonbond14_pairs(prmtop)
    scee = jnp.array([jnp.float32(x) for x in prmtop._raw_data['SCEE_SCALE_FACTOR']])

    displacement, shift = jax_md.space.periodic(boxVectors)

    # alpha = jnp.sqrt(-jnp.log(2 * ewald_error))/cutoff
    alpha = .34

    # assuming in this case box vectors needs to be shape (3,1)
    # n_mesh = (2 * alpha * boxVectors)/((3 * boxVectors)**.2)

    neighbor_fn, nrg_fn = jax_md.energy.coulomb_neighbor_list(
        displacement,
        boxVectors,
        charges*18.2223,
        grid_points,
        None,
        alpha,
        cutoff,
        False)

    return (charges, pairs, pairs14, scee), neighbor_fn, nrg_fn

def coul_get_energy_nb(positions, box, prms):
    charges, pairs, pairs14, scee = prms

    p1 = positions[pairs[:,0]]
    p2 = positions[pairs[:,1]]
    chg1 = charges[pairs[:, 0]]
    chg2 = charges[pairs[:, 1]]
    dist = distance(p1, p2, box)
    dist = jnp.where(jnp.isclose(dist, 0.), 1, dist)

    # constant for 1 / 4 * pi * epsilon
    # print("Charges", charges)
    # print("pairs", pairs)
    # print("Coul Components", jnp.float32(138.935456) * chg1 * chg2 / dist)
    # print("Coul Total E", jnp.sum(jnp.float32(138.935456) * chg1 * chg2 / dist))
    return jnp.sum(jnp.float32(138.935456) * chg1 * chg2 / dist)

def coul_get_energy_nb_pme(positions, box, prms):
    # currently returning erroneous values, not ready for general use
    charges, pairs, pairs14, scee, neighbor_fn, nrg_fn = prms
    nbList = neighbor_fn.allocate(positions)
    direct_and_recip_nrg = nrg_fn(positions, nbList, charge=charges)
    self_nrg = self_nrg = (-.34/jnp.sqrt(jnp.pi)) * jnp.sum(charges**2)
    return direct_and_recip_nrg + self_nrg

def coul_get_energy_14(positions, box, prms):
    charges, pairs, pairs14, scee = prms

    p1 = positions[pairs14[:,0]]
    p2 = positions[pairs14[:,1]]
    parm_idx = pairs14[:,2]
    chg1 = charges[pairs14[:, 0]]
    chg2 = charges[pairs14[:, 1]]
    dist = distance(p1, p2, box)
    dist = jnp.where(jnp.isclose(dist, 0.), 1, dist)

    scee_filtered = jnp.where(jnp.isclose(scee, 0.), 0., 1.0/scee)

    # constant for 1 / 4 * pi * epsilon
    # print("pairs14", pairs14)
    # print("Coul14 Components", 1.0/scee[parm_idx] * jnp.float32(138.935456) * chg1 * chg2 / dist)
    # print("Coul14 Total E", jnp.sum(1.0/scee[parm_idx] * jnp.float32(138.935456) * chg1 * chg2 / dist))
    #TODO: Remove nan to num references, these may cover up errors that cause gradient instability
    return jnp.sum(scee_filtered[parm_idx] * jnp.float32(138.935456) * chg1 * chg2 / dist)

def coul_get_energy(positions, box, prms):
    return coul_get_energy_nb(positions, box, prms) + coul_get_energy_14(positions, box, prms)

def coul_get_energy_pme(positions, box, prms):
    #TODO: add matching LJ with neighbor list/cutoff, wrap into overall nb function and add lj dispersion correction
    #TODO: also add exclusion policy for 1-2/1-3/1-4, might be able to take exclusion list and mask neighbor list
    #return coul_get_energy_nb_pme(positions, box, prms) + coul_get_energy_14(positions, box, prms)
    #let exclusions be captured in pme sum (d+r+self) and then subtract off at end with
    #one4pieps * chgi * chgj * 1/dist * erf(alpha * dist)
    return coul_get_energy_nb_pme(positions, box, prms)

def bond_rest_get_energy(positions, box, restraint=None):
    return

def angle_rest_get_energy(positions, box, restraint=None):
    return

def torsion_rest_get_energy(positions, box, restraint=None):
    return

def rest_get_energy(positions, box, restraint=None):
    radian_to_degree = 180.0/jnp.pi
    degree_to_radian = 1.0/radian_to_degree
    p1 = positions[restraint[0]]
    p2 = positions[restraint[1]]
    p3 = positions[restraint[2]]
    p4 = positions[restraint[3]]
    current_angle = torsion_single(p1,p2,p3,p4, box)
    target_angle = restraint[4]

    # safe but unoptimized values can be found in the ReaxFF manual
    # effectively the ceiling of the restraing energy
    frc1 = restraint[5]
    # tapering factor from 0 to frc1, lower value equals steeper taper
    frc2 = restraint[6]
    
    current_angle = current_angle * radian_to_degree
    target_angle = target_angle * radian_to_degree

    # theta ranges from -180 to 180, must ensure periodicity
    # e.g. 170 to -170 is a 20 degree difference
    current_angle = jnp.where(current_angle < 0.0, current_angle + 360.0, current_angle)
    target_angle = jnp.where(target_angle < 0.0, target_angle + 360.0, target_angle)

    angular_diff = (current_angle - target_angle) * degree_to_radian

    return frc1 * (1.0 - jnp.exp(-frc2 * angular_diff**2))