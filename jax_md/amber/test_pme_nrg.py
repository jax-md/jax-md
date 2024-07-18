#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import numpy as np
import jax.numpy as jnp
import jax
import jax_md
import parmed as pmd
import openmm as omm

jax.config.update("jax_enable_x64", True)

import amber_energy as amber
import time

#inpcrdFile = "../data/diala.inpcrd"
#prmtopFile = "../data/diala.prmtop"
#inpcrdFile = "../data/RAMP1_gas.inpcrd"
#prmtopFile = "../data/RAMP1_gas.prmtop"
#inpcrdFile = "../data/benzene.inpcrd"
#prmtopFile = "../data/benzene.prmtop"
#inpcrdFile = "../data/benzene2.inpcrd"
#prmtopFile = "../data/benzene2.prmtop"
inpcrdFile = "../data/benzene_solv.inpcrd"
prmtopFile = "../data/benzene_solv.prmtop"
#inpcrdFile = "../data/benzene_square.inpcrd"
#prmtopFile = "../data/benzene_square.prmtop"

print("inpcrd/prmtop used:")
print(inpcrdFile, prmtopFile)

inpcrd = omm.app.AmberInpcrdFile(inpcrdFile)
prmtop = omm.app.AmberPrmtopFile(prmtopFile)
positions = jnp.array(inpcrd.getPositions(asNumpy=True))

print("Positions Shape", positions.shape)
print("Min Positions", jnp.min(positions, axis=0))
print("Max Positions", jnp.max(positions, axis=0))

# OpenMM reference values
#parm = pmd.load_file(prmtopFile, inpcrdFile)
system = prmtop.createSystem(nonbondedMethod=omm.app.PME, nonbondedCutoff=0.8*omm.unit.nanometer, removeCMMotion=False, rigidWater=False)
#system = prmtop.createSystem(nonbondedMethod=omm.app.NoCutoff, nonbondedCutoff=.8*omm.unit.nanometer, removeCMMotion=False, rigidWater=True)
#boxVectors = jnp.array([v._value for v in system.getDefaultPeriodicBoxVectors()])
#boxVectors = boxVectors.sum(axis=0)
print("Box Vectors OMM PRMTOP Topology", prmtop.topology.getPeriodicBoxVectors())
print("Box Vectors OMM System", system.getDefaultPeriodicBoxVectors())
#bPad = jnp.max(positions, axis=0) - jnp.min(positions, axis=0)
#print("bPad", bPad)
#boxVectors = jnp.array([5.0,5.0,5.0])
#print("Box Vectors", boxVectors)

for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)

system.getForces()[3].setReactionFieldDielectric(1.0)
system.getForces()[3].setUseDispersionCorrection(False)
print("OMM Uses Dispersion", system.getForces()[3].getUseDispersionCorrection())
#system.getForces()[3].setUseSwitchingFunction(True)
#system.getForces()[3].setSwitchingDistance(.1*omm.unit.nanometer)
print("OMM Uses Switching", system.getForces()[3].getUseSwitchingFunction())
print("OMM Switching Distance", system.getForces()[3].getSwitchingDistance())

#prmtopVecs = jnp.array([5.0,5.0,5.0])
prmtopVecs = jnp.array([v._value for v in prmtop._prmtop.getBoxBetaAndDimensions()][1:4])/10
print("Box Vectors PRMTOP", prmtopVecs)

# frcsum = 0
# for force in pmd.openmm.energy_decomposition_system(parm, system):
#     print("OpenMM ", force[0], ": ", force[1], sep="")
#     frcsum = frcsum + force[1]
# print("OpenMM ", "OverallForce", ": ", frcsum, sep="")

#To remove the units, divide return value by unit.angstrom or unit.nanometer.
#platform = omm.Platform.getPlatformByName('CUDA')
properties = {'Precision': 'double'}
integrator = omm.VerletIntegrator(0.0005*omm.unit.picoseconds)
#simulation = omm.app.Simulation(prmtop.topology, system, integrator, platform)
simulation = omm.app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)
#simulation.context.setVelocitiesToTemperature(0, randomSeed)

print("Pre Minimized")
simulation.minimizeEnergy()

#print("Not Pre Minimized")

simulation.reporters.append(omm.app.StateDataReporter(sys.stdout, 100000, step=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True))

#simulation.reporters.append(omm.app.DCDReporter("out_5_omm.dcd", 10000, enforcePeriodicBox=True))
simulation.reporters.append(omm.app.DCDReporter("out_52_omm.dcd", 10000))
frcsum = 0
for i, f in enumerate(system.getForces()):
    state = simulation.context.getState(getEnergy=True, groups={i})
    print(f.getName(), (state.getPotentialEnergy()._value)/4.184)
    print(f.getName(), "Uses PBCs", f.usesPeriodicBoundaryConditions())
    frcsum = frcsum + state.getPotentialEnergy()._value
print("OpenMM ", "OverallForce", ": ", frcsum/4.184, sep="")

print("Set PME params to 3.2853261060980823,96,96,96")
system.getForces()[3].setPMEParameters(3.2853261060980823,96,96,96)

#sys.exit()

ommState = simulation.context.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=True, getVelocities=True, getForces=True)
print("OMM Starting Statistics - PE , KE , TotalE - in kJ/mol")
omm_pe = ommState.getPotentialEnergy()
omm_ke = ommState.getKineticEnergy()
print(omm_pe, omm_ke, omm_pe+omm_ke)
#get positions before md but after min
omm_pos = ommState.getPositions(asNumpy=True)._value
omm_vel = ommState.getVelocities(asNumpy=True)

print("OMM Velocities Pre MD/Post Min", omm_vel)

for i, f in enumerate(system.getForces()):
    state = simulation.context.getState(getForces=True, groups={i})
    print(f.getName(), "Forces", (state.getForces(asNumpy=True)._value[:10]))

omm_frcs = ommState.getForces(asNumpy=True)._value
print("OMM Forces Pre MD/Post Min", omm_frcs)
print("OMM Forces Magnitude", (jnp.sqrt(jnp.sum(jnp.power(omm_frcs, 2), axis=1)))[:10])

print("OMM State Min Positions", jnp.min(omm_pos, axis=0))
print("OMM State Max Positions", jnp.max(omm_pos, axis=0))


ommStart = time.time()
#simulation.step(6000000)
#simulation.step(20000)
#simulation.step(1000000)
ommEnd = time.time()
print("OMM Time", ommEnd-ommStart)

ommState = simulation.context.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=True, getVelocities=True)
print("OMM Final Statistics - PE , KE , TotalE - in kJ/mol")
omm_pe = ommState.getPotentialEnergy()
omm_ke = ommState.getKineticEnergy()
print(omm_pe, omm_ke, omm_pe+omm_ke)
#omm_pos = ommState.getPositions(asNumpy=True)

#sys.exit()

boxVectors = prmtopVecs
print("Final JAX Box Vectors", boxVectors)
# d_fn, s_fn = jax_md.space.periodic(boxVectors)
# positions = s_fn(positions, 0)

positions = jnp.array(omm_pos)
print("JAX Min Positions From OMM", jnp.min(positions, axis=0))
print("JAX Max Positions From OMM", jnp.max(positions, axis=0))

#ewald_error = 5e-4
cutoff = .8
#alphaEwald = jnp.sqrt(-jnp.log(2 * ewald_error))/(cutoff*10)
print("PME Init")
g_space = 96
print("PME Grid spacing (in each direction): ", g_space)
coulprm = amber.coul_init_pme(prmtop._prmtop, boxVectors, grid_points = g_space, cutoff = cutoff, dr_threshold = 0.0)

#print("Ewald Init")
#coulprm = amber.coul_init_ewald(prmtop._prmtop, boxVectors, grid_points = 96, cutoff = cutoff, dr_threshold = 0.0)


charges, pairs, pairs14, scee, neighbor_fn, nrg_fn, exceptions, exceptions14, alphaEwald = coulprm
nbList = neighbor_fn.allocate(positions)
print("Alpha Ewald", alphaEwald)

def energy_fn(pos, prms=None, nbList=None, nbListDense=None, restraint=None):
    bprm, aprm, tprm, lprm, cprm, = prms
    charges, pairs, pairs14, scee, exceptions, exceptions14, alphaEwald = cprm

    #cprm = ffq_charges, pairs, pairs14, scee, neighbor_fn, nrg_fn, exceptions, exceptions14, alphaEwald
    cprm = charges, pairs, pairs14, scee, neighbor_fn, nrg_fn, exceptions, exceptions14, alphaEwald
    #cprm = charges, pairs, pairs14, scee

    return jnp.float64((amber.bond_get_energy(pos, boxVectors, bprm) \
            + amber.angle_get_energy(pos, boxVectors, aprm) \
            + amber.torsion_get_energy(pos, boxVectors, tprm) \
            #+ amber.lj_get_energy(pos, boxVectors, lprm) \
            #+ amber.coul_get_energy(pos, boxVectors, cprm) \
            + amber.lj_get_energy(pos, boxVectors, lprm, nbList) \
            + amber.coul_get_energy_pme(pos, boxVectors, cprm, nbList) \
            #+ amber.coul_get_energy_ewald(pos, boxVectors, cprm, nbList) \
            + 0)/4.184)


bondprm = amber.bond_init(prmtop._prmtop)
angleprm = amber.angle_init(prmtop._prmtop)
torsionprm = amber.torsion_init(prmtop._prmtop)
ljprm = amber.lj_init(prmtop._prmtop)
#coulprm = amber.coul_init(prmtop._prmtop)
cprm = charges, pairs, pairs14, scee, exceptions, exceptions14, alphaEwald
prms = (bondprm, angleprm, torsionprm, ljprm, cprm)

bondStart = time.time()
print("JAX Bond Energy:", amber.bond_get_energy(positions, boxVectors, bondprm)/4.184)
bondEnd = time.time()
angleStart = time.time()
print("JAX Angle Energy:", amber.angle_get_energy(positions, boxVectors, angleprm)/4.184)
angleEnd = time.time()
torsStart = time.time()
print("JAX Torsion Energy:", amber.torsion_get_energy(positions, boxVectors, torsionprm)/4.184)
torsEnd = time.time()
coulStart = time.time()
coul_nrg = amber.coul_get_energy_pme(positions, boxVectors, coulprm, nbList)/4.184
coulEnd = time.time()
ljStart = time.time()
lj_nrg = amber.lj_get_energy(positions, boxVectors, ljprm, nbList)/4.184
ljEnd = time.time()
print("JAX Coulomb Energy:", coul_nrg)
print("JAX LJ Energy:", lj_nrg)
print("JAX NB Energy:", coul_nrg + lj_nrg)

coul_nrg_all = amber.coul_get_energy(positions, boxVectors, (charges, pairs, pairs14, scee))/4.184
lj_nrg_all = amber.lj_get_energy(positions, boxVectors, ljprm, nbList)/4.184
print("JAX Coul Energy All Pairs (not in total):", coul_nrg_all)
print("JAX LJ Energy All Pairs (not in total):", lj_nrg_all)
print("JAX NB Energy All Pairs (not in total):", coul_nrg_all + lj_nrg_all)


print("JAX Overall Energy:", energy_fn(positions, prms, nbList))

print("Bond Calculation Time:", bondEnd-bondStart)
print("Angle Calculation Time:", angleEnd-angleStart)
print("Torsion Calculation Time:", torsEnd-torsStart)
print("Coulomb Calculation Time:", coulEnd-coulStart)
print("Lennard Jones Calculation Time:", ljEnd-ljStart)

masses = jnp.array([jnp.float64(val) for val in prmtop._prmtop._raw_data['MASS']])
#displacement_fn, shift_fn = jax_md.space.periodic_general(boxVectors, fractional_coordinates=False)
displacement_fn, shift_fn = jax_md.space.periodic(boxVectors, wrapped=True)

kB = 0.00831446267 / 4.184
init_temp = 0
print("Temperature to initialize momentum in K", init_temp)
key = jax.random.PRNGKey(0)
#energy_fn = jax.jit(energy_fn)
init_fn, apply_fn = jax_md.simulate.nve(energy_fn, shift_fn, 5e-4)
state = init_fn(key, positions, mass=masses, restraint=None, prms=prms, nbList=nbList, kT=init_temp*kB)

#print("Momentum set to OMM values (may be 0)")
#state = state.set(momentum = jnp.array(omm_vel._value) * state.mass)

print("JAX LJ Force w/ Map Neighbor", amber.lj_get_energy_nbr(positions, boxVectors, ljprm, nbList, displacement_fn)/4.184)

print("JAX Min Positions", jnp.min(state.position, axis=0))
print("JAX Max Positions", jnp.max(state.position, axis=0))

#nbTime = 0
#nrgTime = 0

print("JAX Starting Statistics - PE , KE , TotalE - in kJ/mol, temp")
pE = energy_fn(state.position, prms, nbList)*4.184
kE = jax_md.quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)*4.184
temp = jax_md.quantity.temperature(momentum=state.momentum, mass=state.mass)/kB
print(pE, kE, pE+kE, temp)
print("pE dtype", pE.dtype)

grad_fn = jax.grad(energy_fn)


print("JAX Bond Forces:", jax.grad(amber.bond_get_energy)(positions, boxVectors, bondprm)[:10])
print("JAX Angle Forces:", jax.grad(amber.angle_get_energy)(positions, boxVectors, angleprm)[:10])
print("JAX Torsion Forces:", jax.grad(amber.torsion_get_energy)(positions, boxVectors, torsionprm)[:10])
coul_frcs = jax.grad(amber.coul_get_energy_pme)(positions, boxVectors, coulprm, nbList)
lj_frcs = jax.grad(amber.lj_get_energy)(positions, boxVectors, ljprm, nbList)
#print("JAX Coulomb Energy:", coul_nrg)
#print("JAX LJ Energy:", lj_nrg)
nb_frcs = coul_frcs + lj_frcs
print("JAX NB Forces:", nb_frcs[:10])

jax_frcs = grad_fn(state.position, prms, nbList)*4.184
print("JAX Forces", jax_frcs[:10])

jnp.set_printoptions(threshold=sys.maxsize, suppress=True)
print("Overall Force Differences (Magnitude)", (jnp.sqrt(jnp.sum(jnp.power(omm_frcs, 2), axis=1)) - jnp.sqrt(jnp.sum(jnp.power(jax_frcs, 2), axis=1)))[:10])

def cosine_simularity(v1, v2):
    return jnp.dot(v1,v2)/(jnp.linalg.norm(v1)*jnp.linalg.norm(v2))

omm_frc_state = simulation.context.getState(getForces=True, groups={3})
omm_nb_frc = omm_frc_state.getForces(asNumpy=True)._value
omm_nb_frc_mag = jnp.sqrt(jnp.sum(jnp.power(omm_nb_frc, 2), axis=1))
print("OMM NB Force Magnitude", omm_nb_frc_mag[:10])
jax_nb_frc_mag = jnp.sqrt(jnp.sum(jnp.power(nb_frcs, 2), axis=1))
print("JAX NB Force Magnitude", jax_nb_frc_mag[:10])
print("NB Force Magnitude Difference", (jax_nb_frc_mag-omm_nb_frc_mag)[:10])

vec_cos_sim = jax.vmap(cosine_simularity)

print("NB Cosine Simularity", vec_cos_sim(omm_nb_frc, nb_frcs)[:10])

sys.exit()

def body_fn(i, stateList):
    state, restraint, prms, nbList = stateList
    #should this be the length of the nb list?
    #check after each iter and check lj to make sure it's using nblist
    #also check to make sure size constraint on argwhere in mask works
    #compare with no size versus size in loop
    #nbStart = time.time()
    #TODO: test if this can be jitted, also consider checking recompilations
    nbList = nbList.update(state.position)
    #nbList = nbList.update(jnp.mod(state.position, boxVectors))
    #nbEnd = time.time()
    #nbTime += nbEnd-nbStart

    #nrgStart = time.time()
    state = apply_fn(state, prms=prms, restraint=restraint, nbList=nbList)
    #nrgEnd = time.time()
    #nrgTime += nrgEnd-nrgStart

    return state, restraint, prms, nbList

f = open("out_52.dcd", "wb")
dcd_file = omm.app.DCDFile(f, prmtop.topology, 0.0005)

#iter = 100000
#inner = 10000
#iter = 200000
#inner = 10000
iter = 6000000
inner = 10000
outer = int(iter/inner)

n_times_reallocated = 10 # start with 2^ some power extra capacity as a heurisitc if there's an overflow
print("NBList Initial Shape", nbList.idx.shape)
runStart = time.time()
#pairs = []
curr_rest = None
# 0.00831446267 boltzmann's constant in kj/mol
# energy function returns kcal so this constant needs to be scaled
#kB = 0.00831446267 / 4.184
print("step , PE , KE , TotalE - in kJ/mol, kT")
for i in range(outer):
    #new_state, curr_rest, prms, nbList = jax.lax.fori_loop(0, inner, body_fn, (state, curr_rest, prms, nbList))
    for j in range(inner):
        new_state, curr_rest, prms, nbList = body_fn(j, (state, curr_rest, prms, nbList))
    if nbList.did_buffer_overflow:
        print('Neighbor list overflowed, reallocating.')
        #n_times_reallocated
        #nbList = neighbor_fn.allocate(state.position, extra_capacity=2**n_times_reallocated)
        nbList = neighbor_fn.allocate(state.position)
    else:
        state = new_state

    #print("momentum shape", state.momentum.shape)
    pE = energy_fn(state.position, prms, nbList)*4.184
    kE = jax_md.quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)*4.184
    temp = jax_md.quantity.temperature(momentum=state.momentum, mass=state.mass)/kB
    print((i+1)*inner, ', ', end='')
    #print(pE, kE, pE+kE, jnp.min(state.position, axis=0), jnp.max(state.position, axis=0), end='')
    #print(pE, kE, pE+kE, temp/kB, end='')
    pEkE = pE + kE
    jax.debug.print("{pE}, {kE}, {pEkE}, {temp}", pE=pE, kE=kE, pEkE=pEkE, temp=temp)

    simulation.context.setPositions(state.position)
    frcsum = 0
    for i, f in enumerate(system.getForces()):
        ommState = simulation.context.getState(getEnergy=True, groups={i})
        print(f.getName(), (ommState.getPotentialEnergy()._value)/4.184)
        #print(f.getName(), "Uses PBCs", f.usesPeriodicBoundaryConditions())
        frcsum = frcsum + ommState.getPotentialEnergy()._value
    print("OpenMM ", "OverallForce", ": ", frcsum/4.184, sep="")

    #sys.stdout.flush()

    #with open("out.dcd", "ab") as f:
    dcd_file.writeModel(state.position)


    bond_nrg = amber.bond_get_energy(state.position, boxVectors, bondprm)/4.184
    angle_nrg = amber.angle_get_energy(state.position, boxVectors, angleprm)/4.184
    torsion_nrg = amber.torsion_get_energy(state.position, boxVectors, torsionprm)/4.184
    print("JAX Bond Energy:", bond_nrg)
    print("JAX Angle Energy:", angle_nrg)
    print("JAX Torsion Energy:", torsion_nrg)

    coul_nrg = amber.coul_get_energy(state.position, boxVectors, (charges, pairs, pairs14, scee))/4.184
    lj_nrg = amber.lj_get_energy(state.position, boxVectors, ljprm)/4.184
    print("JAX Coulomb (All Pairs) Energy", coul_nrg)
    print("JAX LJ (All Pairs) Energy:", lj_nrg)
    print("JAX Total NB (All Pairs)", coul_nrg + lj_nrg)

    coul_nrg = amber.coul_get_energy_pme(state.position, boxVectors, coulprm, nbList)/4.184
    lj_nrg = amber.lj_get_energy(state.position, boxVectors, ljprm, nbList)/4.184
    print("JAX Coulomb (PME) Force:", coul_nrg)
    print("JAX LJ (NbList) Force:", lj_nrg)
    print("JAX Total NB (PME/Cut)", coul_nrg + lj_nrg)

    sys.stdout.flush()

    print()
runFinish = time.time()

print("MD Time", runFinish-runStart)

f.close()

#print("NB Time", nbTime)
#print("NRG Time", nrgTime)

# Example Output
#
# OpenMM HarmonicBondForce: 0.020598034671798035
# OpenMM HarmonicAngleForce: 0.36199587480971734
# OpenMM PeriodicTorsionForce: 9.644000525684246
# OpenMM NonbondedForce: -23.361734860713803
# OpenMM OverallForce: -13.33514042554804
# "Step","Potential Energy (kJ/mole)"
# 100,-74.21282958984375
# 200,-72.41244506835938
# 300,-68.67630004882812
# 400,-67.6129150390625
# 500,-68.16668701171875
# 600,-73.24057006835938
# 700,-67.79931640625
# 800,-70.92703247070312
# 900,-70.86770629882812
# 1000,-70.487548828125
#
# JAX Bond Force: 0.020599548
# JAX Angle Force: 0.3619943
# JAX Torsion Force: 9.644
# JAX Coulomb Force: -31.189417
# JAX LJ Force: 7.827678
# JAX NB Force: -23.361738
# JAX Overall Force: -13.335144
# 100 , -69.37387
# 200 , -75.07478
# 300 , -76.04286
# 400 , -71.68538
# 500 , -71.279755
# 600 , -69.46986
# 700 , -69.73034
# 800 , -66.77943
# 900 , -66.254425
# 1000 , -67.8342
