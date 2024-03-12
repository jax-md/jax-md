import sys
import numpy as np
import jax.numpy as jnp
import jax
import jax_md
import parmed as pmd
import openmm as omm

import amber_energy as amber

inpcrd = omm.app.AmberInpcrdFile("data/diala.inpcrd")
prmtop = omm.app.AmberPrmtopFile("data/diala.prmtop")
positions = jnp.array(inpcrd.getPositions(asNumpy=True))

# OpenMM reference values
parm = pmd.load_file('data/diala.prmtop', 'data/diala.inpcrd')
system = prmtop.createSystem(nonbondedMethod=omm.app.NoCutoff, removeCMMotion=False)
boxVectors = jnp.array([v._value for v in system.getDefaultPeriodicBoxVectors()])
boxVectors = boxVectors.sum(axis=0)

frcsum = 0
for force in pmd.openmm.energy_decomposition_system(parm, system):
    print("OpenMM ", force[0], ": ", force[1], sep="")
    frcsum = frcsum + force[1]
print("OpenMM ", "OverallForce", ": ", frcsum, sep="")

integrator = omm.VerletIntegrator(0.001*omm.unit.picoseconds)
simulation = omm.app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)
simulation.reporters.append(omm.app.StateDataReporter(sys.stdout, 100, step=True,
        potentialEnergy=True, temperature=False))
simulation.step(1000)

def energy_fn(pos, prms=None, restraint=None):
    bprm, aprm, tprm, lprm, cprm, = prms
    return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
            + amber.angle_get_energy(pos, boxVectors, aprm) \
            + amber.torsion_get_energy(pos, boxVectors, tprm) \
            + amber.lj_get_energy(pos, boxVectors, lprm) \
            + amber.coul_get_energy(pos, boxVectors, cprm) \
            + 0)/4.184)

bondprm = amber.bond_init(prmtop._prmtop)
angleprm = amber.angle_init(prmtop._prmtop)
torsionprm = amber.torsion_init(prmtop._prmtop)
ljprm = amber.lj_init(prmtop._prmtop)
coulprm = amber.coul_init(prmtop._prmtop)
prms = (bondprm, angleprm, torsionprm, ljprm, coulprm)

print("JAX Bond Force:", amber.bond_get_energy(positions, boxVectors, bondprm)/4.184)
print("JAX Angle Force:", amber.angle_get_energy(positions, boxVectors, angleprm)/4.184)
print("JAX Torsion Force:", amber.torsion_get_energy(positions, boxVectors, torsionprm)/4.184)
coul_nrg = amber.coul_get_energy(positions, boxVectors, coulprm)/4.184
lj_nrg = amber.lj_get_energy(positions, boxVectors, ljprm)/4.184
print("JAX Coulomb Force:", coul_nrg)
print("JAX LJ Force:", lj_nrg)
print("JAX NB Force:", coul_nrg + lj_nrg)

print("JAX Overall Force:", energy_fn(positions, prms))

masses = jnp.array([jnp.float32(val) for val in prmtop._prmtop._raw_data['MASS']])
displacement_fn, shift_fn = jax_md.space.periodic_general(boxVectors, fractional_coordinates=False)
key = jax.random.PRNGKey(0)
energy_fn = jax.jit(energy_fn)
init_fn, apply_fn = jax_md.simulate.nve(energy_fn, shift_fn, 1e-3)
state = init_fn(key, positions, mass=masses, restraint=None, prms=prms, kT=0)

def body_fn(i, stateList):
    state, restraint, prms = stateList
    state = apply_fn(state, prms=prms, restraint=restraint)
    return state, restraint, prms

iter = 1000
inner = 100
outer = int(iter/inner)

pairs = []
curr_rest = None
for i in range(outer):
    state, curr_rest, prms = jax.lax.fori_loop(0, inner, body_fn, (state, curr_rest, prms))

    print((i+1)*inner, ', ', end='')
    print(energy_fn(state.position, prms)*4.184, end='')
    print()

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