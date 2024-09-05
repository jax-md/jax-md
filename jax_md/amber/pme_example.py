import sys
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jax_md
import openmm as omm
import openmm.app as app
import amber_energy as amber

inpcrdFile = "./data/benzene_solv.inpcrd"
prmtopFile = "./data/benzene_solv.prmtop"

inpcrd = app.AmberInpcrdFile(inpcrdFile)
prmtop = app.AmberPrmtopFile(prmtopFile)
positions = jnp.array(inpcrd.getPositions(asNumpy=True))

system = prmtop.createSystem(nonbondedMethod=omm.app.PME, nonbondedCutoff=0.8*omm.unit.nanometer, removeCMMotion=False, rigidWater=False)

for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)

system.getForces()[3].setUseDispersionCorrection(False)

platform = omm.Platform.getPlatformByName('CUDA')
properties = {'Precision': 'double'}
integrator = omm.VerletIntegrator(0.001*omm.unit.picoseconds)
simulation = omm.app.Simulation(prmtop.topology, system, integrator, platform=platform, platformProperties=properties)
simulation.context.setPositions(inpcrd.positions)
simulation.context.setVelocitiesToTemperature(0)

simulation.reporters.append(omm.app.StateDataReporter(sys.stdout, 100, step=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True))

frcsum = 0
for i, f in enumerate(system.getForces()):
    state = simulation.context.getState(getEnergy=True, groups={i})
    print(f.getName(), (state.getPotentialEnergy()._value)/4.184)
    frcsum = frcsum + state.getPotentialEnergy()._value
print("OpenMM ", "OverallForce", ": ", frcsum/4.184, sep="")

ommState = simulation.context.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=True, getVelocities=True, getForces=True)
print("OMM Starting Statistics - PE , KE , TotalE - in kJ/mol")
omm_pe = ommState.getPotentialEnergy()
omm_ke = ommState.getKineticEnergy()
print(omm_pe, omm_ke, omm_pe+omm_ke)

simulation.step(1000)

boxVectors = jnp.array([v._value for v in prmtop._prmtop.getBoxBetaAndDimensions()][1:4])/10
cutoff = 0.8
g_space = 96
coulprm = amber.coul_init_pme(prmtop._prmtop, boxVectors, grid_points = g_space, cutoff = cutoff, dr_threshold = 0.2)

charges, pairs, pairs14, scee, neighbor_fn, nrg_fn, exceptions, exceptions14, alphaEwald = coulprm

displacement_fn, shift_fn = jax_md.space.periodic(boxVectors, wrapped=True)

def energy_fn(pos, prms=None, nbList=None):
    bprm, aprm, tprm, lprm, cprm, = prms
    # This is to specifically get around an issue with JAX MD and functions as explicit arguments
    # in their simulation setup functions, this will be fixed in the first full release of the code
    cprm = charges, pairs, pairs14, scee, neighbor_fn, nrg_fn, exceptions, exceptions14, alphaEwald

    return jnp.float64((amber.bond_get_energy(pos, boxVectors, bprm) \
            + amber.angle_get_energy(pos, boxVectors, aprm) \
            + amber.torsion_get_energy(pos, boxVectors, tprm) \
            + amber.lj_get_energy(pos, boxVectors, lprm, nbList) \
            + amber.coul_get_energy_pme(pos, boxVectors, cprm, nbList) \
            + 0)/4.184)

charges, pairs, pairs14, scee, neighbor_fn, nrg_fn, exceptions, exceptions14, alphaEwald = coulprm
nbList = neighbor_fn.allocate(positions)
bondprm = amber.bond_init(prmtop._prmtop)
angleprm = amber.angle_init(prmtop._prmtop)
torsionprm = amber.torsion_init(prmtop._prmtop)
ljprm = amber.lj_init(prmtop._prmtop)
cprm = charges, pairs, pairs14, scee, exceptions, exceptions14, alphaEwald
prms = (bondprm, angleprm, torsionprm, ljprm, cprm)

print("JAX Bond Energy:", amber.bond_get_energy(positions, boxVectors, bondprm)/4.184)
print("JAX Angle Energy:", amber.angle_get_energy(positions, boxVectors, angleprm)/4.184)
print("JAX Torsion Energy:", amber.torsion_get_energy(positions, boxVectors, torsionprm)/4.184)
coul_nrg = amber.coul_get_energy_pme(positions, boxVectors, coulprm, nbList)/4.184
lj_nrg = amber.lj_get_energy(positions, boxVectors, ljprm, nbList)/4.184
print("JAX Coulomb Energy:", coul_nrg)
print("JAX LJ Energy:", lj_nrg)
print("JAX NB Energy:", coul_nrg + lj_nrg)

print("JAX Overall Energy:", energy_fn(positions, prms, nbList))

masses = jnp.array([jnp.float64(val) for val in prmtop._prmtop._raw_data['MASS']])

kB = 0.00831446267 / 4.184
init_temp = 0
key = jax.random.PRNGKey(0)
init_fn, apply_fn = jax_md.simulate.nve(energy_fn, shift_fn, 1e-3)
state = init_fn(key, positions, mass=masses, prms=prms, nbList=nbList, kT=init_temp*kB)

print("JAX Starting Statistics - PE , KE , TotalE - in kJ/mol, temp")
pE = energy_fn(state.position, prms, nbList)*4.184
kE = jax_md.quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)*4.184
temp = jax_md.quantity.temperature(momentum=state.momentum, mass=state.mass)/kB
print(pE, kE, pE+kE, temp)

def body_fn(i, stateList):
    state, prms, nbList = stateList
    nbList = nbList.update(state.position)
    state = apply_fn(state, prms=prms, nbList=nbList)

    return state, prms, nbList

iter = 1000
inner = 100
outer = int(iter/inner)

print("step , PE , KE , TotalE - in kJ/mol, kT")
for i in range(outer):
    new_state, prms, nbList = jax.lax.fori_loop(0, inner, body_fn, (state, prms, nbList))

    if nbList.did_buffer_overflow:
        print('Neighbor list overflowed, reallocating.')
        nbList = neighbor_fn.allocate(state.position)
    else:
        state = new_state

    pE = energy_fn(state.position, prms, nbList)*4.184
    kE = jax_md.quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)*4.184
    pEkE = pE + kE
    temp = jax_md.quantity.temperature(momentum=state.momentum, mass=state.mass)/kB
    
    print((i+1)*inner, ', ', end='')
    jax.debug.print("{pE}, {kE}, {pEkE}, {temp}", pE=pE, kE=kE, pEkE=pEkE, temp=temp)
    sys.stdout.flush()