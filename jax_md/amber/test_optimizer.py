import sys, json
from scipy.optimize import minimize
import numpy as np
import jax.numpy as jnp
import jax
import jax_md
import matplotlib.pyplot as plt
import amber_energy as amber
import parmed as pmd
import openmm as omm
jax.config.update("jax_enable_x64", True)

# make global array for loss
losses = []
iteration = 0

# Reads json data 
def ReadJsonData(json_path='./data/dh_6-7-9-11/params.json'):

    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    return json_data

# dumps json data into an existing file
def SaveJsonData(field_dict, json_path='./data/dh_6-7-9-11/params.json'):

    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    for key in field_dict:
        json_data[key]=field_dict[key]

    with open(json_path, 'w') as outfile:
        json.dump(json_data, outfile)

def extractCoordinates(flist):
    coordinates = []

    for file in flist:
        with open(file, 'r') as f:
            lines=f.readlines()
        
        crds = []
        for line in lines[2:]:
            crds.append([jnp.float32(i) for i in line.split()[1:]])
        #print(crds)
        crds = jnp.array(crds)
        #A -> NM
        coordinates.append(crds/10)

    return coordinates

def constrained_minimization_vec(crds, prmtop, boxVectors):
    radian_to_degree = 180.0/jnp.pi
    degree_to_radian = 1.0/radian_to_degree
    bondprm = amber.bond_init(prmtop._prmtop)
    angleprm = amber.angle_init(prmtop._prmtop)
    torsionprm = amber.torsion_init(prmtop._prmtop)
    ljprm = amber.lj_init(prmtop._prmtop)
    coulprm = amber.coul_init(prmtop._prmtop)
    prms = (bondprm, angleprm, torsionprm, ljprm, coulprm)
    # restraint format:
    # p1,p2,p3,p4,resangle(radians),frc1,frc2
    # default frc1/frc2 values - 1000.0/0.25
    frc1 = 1000
    frc2 = 0.1
    t = [6,7,9,11]

    def energy_fn(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm) \
                + amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)
                #+ 0)/4.184

    def energy_fn_no_restraint(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm) \
                #+ amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)
                + 0)/4.184)

    masses = jnp.array([jnp.float32(val) for val in prmtop._prmtop._raw_data['MASS']])
    displacement_fn, shift_fn = jax_md.space.periodic_general(boxVectors, fractional_coordinates=False)
    key = jax.random.PRNGKey(0)
    energy_fn = jax.jit(energy_fn)
    init_fn, apply_fn = jax_md.minimize.fire_descent(energy_fn, shift_fn, 1e-3, 1e-3)
    #state = init_fn(positions, mass=masses)

    def body_fn(i, stateList):
        state, restraint, prms = stateList
        state = apply_fn(state, prms=prms, restraint=restraint)
        #return (state, nbpairs)
        return state, restraint, prms

    iter = 2000
    inner = 100
    outer = int(iter/inner)

    initial_torsions = []
    actual_torsions = []
    pre_energies = []
    pre_rest_energies = []
    energies = []
    rest_energies = []
    pairs = []
    mdtimes = []
    post_positions = []

    global iteration
    iteration = iteration + 1

    if iteration % 5 == 0:
        print("Minimization Start")
        #vmap instead of naive loop
        target_angle = jnp.array([i for i in range(36)])
        crds = jnp.array(crds)
        batch_inner = jax.vmap(min_inner, in_axes=(0, 0, None, None, None, None, None, None, None), out_axes=(0,0,0))

        energies, post_positions, actual_torsions = batch_inner(crds, target_angle, energy_fn_no_restraint, energy_fn, init_fn, body_fn, masses, boxVectors, prms)

        deviation = []
        for i, j in enumerate(actual_torsions):
            current_angle = j * radian_to_degree
            current_angle = jnp.where(current_angle < 0.0, current_angle + 360.0, current_angle)
            deviation.append(jnp.absolute(i*10 - current_angle))

        print("Average angular deviation from restrained angle:", jnp.mean(jnp.array(deviation)))
    else:
        post_positions = crds
        energies = jnp.array([energy_fn_no_restraint(p, prms=prms) for p in crds])

    # global iteration
    # iteration = iteration + 1

    return energies, post_positions

def min_inner(crds, target, energy_fn_no_restraint, energy_fn, init_fn, body_fn, masses, boxVectors, prms):
    radian_to_degree = 180.0/jnp.pi
    degree_to_radian = 1.0/radian_to_degree
    frc1 = 1000
    frc2 = 0.1
    t = [7,9,10,15]

    target_angle = target * 10 * degree_to_radian
    curr_rest = [t[0],t[1],t[2],t[3], target_angle, frc1, frc2]

    current_crds = crds

    pre_energies = energy_fn_no_restraint(current_crds, prms=prms)

    state = init_fn(current_crds, mass=masses, restraint=curr_rest, prms=prms)

    state = jax_md.minimize.FireDescentState(jnp.float64(state.position),jnp.float64(state.momentum),\
                                                jnp.float64(state.force), state.mass, state.dt, state.alpha,\
                                                state.n_pos)

    p1 = state.position[t[0]]
    p2 = state.position[t[1]]
    p3 = state.position[t[2]]
    p4 = state.position[t[3]]
    initial_torsions = amber.torsion_single(p1,p2,p3,p4, boxVectors)

    iter = 2000
    inner = 100
    outer = int(iter/inner)

    for i in range(outer):
        state, curr_rest, prms = jax.lax.fori_loop(0, inner, body_fn, (state, curr_rest, prms))

    p1 = state.position[t[0]]
    p2 = state.position[t[1]]
    p3 = state.position[t[2]]
    p4 = state.position[t[3]]
    actual_torsions = amber.torsion_single(p1,p2,p3,p4, boxVectors)

    post_positions = state.position
    energies = energy_fn_no_restraint(state.position, prms=prms)

    return energies, post_positions, actual_torsions

def gradObj(scipy_params, *args):
    crds, boxVectors, ref_ene, post_positions, prms_pre = args
    
    prms = prms_pre

    ##              [ 6,  7,  9, 13, 13],
    ##              [ 8,  7,  9, 13, 16],
    ##
    ##              [ 6,  7,  9, 11, 11],
    ##              [ 6,  7,  9, 10, 12],
    ##              [ 8,  7,  9, 10, 14],
    ##              [ 8,  7,  9, 11, 15],

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][13] = scipy_params[0]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][13] = scipy_params[1]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][13] = scipy_params[2]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][16] = scipy_params[3]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][16] = scipy_params[4]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][16] = scipy_params[5]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][11] = scipy_params[6]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][11] = scipy_params[7]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][11] = scipy_params[8]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][12] = scipy_params[9]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][12] = scipy_params[10]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][12] = scipy_params[11]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][14] = scipy_params[12]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][14] = scipy_params[13]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][14] = scipy_params[14]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][15] = scipy_params[15]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][15] = scipy_params[16]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][15] = scipy_params[17]

    bondprm = amber.bond_init(prms._prmtop)
    angleprm = amber.angle_init(prms._prmtop)
    torsionprm = amber.torsion_init(prms._prmtop)
    ljprm = amber.lj_init(prms._prmtop)
    coulprm = amber.coul_init(prms._prmtop)
    prms = (bondprm, angleprm, torsionprm, ljprm, coulprm)
    def energy_fn(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm))/4.184)
                #+ amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)
    
    ene_list = [energy_fn(p, prms=prms) for p in post_positions]

    min_ene = min(ene_list)

    relative_ene_list = [(x - min_ene) for x in ene_list]

    np_relative_ene_list=jnp.array(relative_ene_list)
    np_ref_ene=jnp.array(ref_ene)

    difference=np_ref_ene-np_relative_ene_list
    nrg_diff_sqrd = jnp.sum(difference ** 2)

    return nrg_diff_sqrd

# updates amber prmtop file, runs constrained optimizations, computes difference between ref and computed energy profiles and RMSD.  
def ObjectiveFunction(scipy_params, *args):
    print("Iteration:", iteration+1)

    crds, boxVectors, ref_ene, params_dict, optvars_dict, prms=args

    print("Parameter Guess:", scipy_params)

    #set new parameters
    # i=0
    # for key, value in params_dict.items():
    #     if(optvars_dict['height']):
    #         value['height']=scipy_params[i]
    #         i+=1

    #     if(optvars_dict['phase']):
    #         value['phase']=scipy_params[i]
    #         i+=1

    #     if(optvars_dict['periodicity']):
    #         value['periodicity']=scipy_params[i]
    #         i+=1

    #     if(optvars_dict['scee']):
    #         value['scee']=scipy_params[i]
    #         i+=1

    #     if(optvars_dict['scnb']):
    #         value['scnb']=scipy_params[i]
    #         i+=1
            
    ##              [ 6,  7,  9, 13, 13],
    ##              [ 8,  7,  9, 13, 16],
    ##
    ##              [ 6,  7,  9, 11, 11],
    ##              [ 6,  7,  9, 10, 12],
    ##              [ 8,  7,  9, 10, 14],
    ##              [ 8,  7,  9, 11, 15],

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][13] = scipy_params[0]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][13] = scipy_params[1]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][13] = scipy_params[2]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][16] = scipy_params[3]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][16] = scipy_params[4]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][16] = scipy_params[5]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][11] = scipy_params[6]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][11] = scipy_params[7]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][11] = scipy_params[8]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][12] = scipy_params[9]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][12] = scipy_params[10]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][12] = scipy_params[11]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][14] = scipy_params[12]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][14] = scipy_params[13]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][14] = scipy_params[14]

    prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][15] = scipy_params[15]
    prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][15] = scipy_params[16]
    prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][15] = scipy_params[17]

    ene_list, post_positions = constrained_minimization_vec(crds, prms, boxVectors)

    loss_and_grad_fn = jax.value_and_grad(gradObj)
    loss_and_grad = loss_and_grad_fn(scipy_params, crds, boxVectors, ref_ene, post_positions, prms)

    loss, grad = loss_and_grad
    grad = grad.astype('float64')
    print("Loss", loss)
    print("Loss Gradient", grad)
    losses.append(loss)

    min_ene = min(ene_list)
    relative_ene_list = [(x - min_ene) for x in ene_list]

    plt.plot(range(0,360,10), ref_ene, marker='o', label="Pre Optimization")
    plt.plot(range(0,360,10), relative_ene_list, marker='o', label="Post Optimization")
    plt.title("dh_6-7-9-11 JAX-AMBER Fitting")
    plt.xlabel("Dihedral (Degree)")
    #should this be kj/mol?
    plt.ylabel("Potential Energy (kcal/mol)")
    plt.legend()
    plt.savefig("./test_optimizer_output/iteration_%s.png" % iteration)
    plt.close()

    # scipy requires jac gradient as list
    return loss, list(grad)

def main(arg_values):
    initial_guess='initial_guess'
    ref_ene=[0.0,0.24041407495730027,0.7180356999677429,1.523482149935944,2.3976774999388795,3.2420603249764213,3.9534696249666013,4.270306924982776,4.275803824991726,4.150128124955188,3.7827582499446066,3.668954849980821,3.443494099972213,3.341964599971732,3.356309249936942,3.4091008249399124,3.5792474499419313,3.9089673249762313,4.470510800001648,5.026563649951754,5.4152183249399855,5.627595699985193,5.775955524968026,5.81154104994738,5.6101888499449615,5.361052524988281,4.971136574998241,4.523440424971454,3.9399093499815763,3.2612116249737255,2.5247650749540185,1.8916175749831154,1.1293870499835634,0.5700021749584039,0.16918654995379256,0.029498775001854938]
    algorithm='L-BFGS-B'
    maxiter=1000
    step_size=0.100000
    crd_dir='./data/dh_6-7-9-11/confs_999-999/dh_6-7-9-11'

    crd_flist=[crd_dir + '/dh_6-7-9-11_%03d' % (i) + '.xyz' for i in range(36)]

    #list of 36 (35,3) numpy arrays from 0-350 deg
    coordinates = extractCoordinates(crd_flist)

    prmtop = "./data/dh_6-7-9-11/prmtop"
    prmtopomm = omm.app.AmberPrmtopFile(prmtop)

    params_dict=ReadJsonData()[initial_guess]

    print("Params Dict", params_dict)

    optvars_dict=ReadJsonData()['optvars']

    print("Opt Vars", optvars_dict)

    bounds_dict=ReadJsonData()['bounds']

    print("Bounds", bounds_dict)
    
    guess=list()
    bounds=list()

    for key, value in params_dict.items():
        if(optvars_dict['height']):
            guess.append(value['height'])
            bounds.append(bounds_dict['height'])

        if(optvars_dict['phase']):
            guess.append(value['phase'])
            bounds.append(bounds_dict['phase'])

        if(optvars_dict['periodicity']):
            guess.append(value['periodicity'])
            bounds.append(bounds_dict['periodicity'])

        if(optvars_dict['scee']):
            guess.append(value['scee'])
            bounds.append(bounds_dict['scee'])

        if(optvars_dict['scnb']):
            guess.append(value['scnb'])
            bounds.append(bounds_dict['scnb'])

    system = prmtopomm.createSystem(nonbondedMethod=omm.app.NoCutoff, removeCMMotion=False, constraints=None)
    boxVectors = jnp.array([v._value for v in system.getDefaultPeriodicBoxVectors()])
    boxVectors = boxVectors.sum(axis=0)
    
    minimization_result=minimize(ObjectiveFunction, guess, jac=True, \
           args=(coordinates, boxVectors, ref_ene, params_dict, optvars_dict, prmtopomm), \
           bounds=bounds, method=algorithm, options={'maxiter':maxiter, 'eps': step_size})

    print("Losses:", losses)

    return
    
if __name__ == "__main__":
    main(sys.argv)