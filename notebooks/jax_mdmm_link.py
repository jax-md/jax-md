# -*- coding: utf-8 -*-from openmm import app, unit
from jax.config import config; config.update("jax_enable_x64", True)
import openmm
import jax
from openmm.app import Modeller
import numpy as onp
from functools import partial
import functools
from jax import numpy as jnp
from jax import vmap, jit
from jax_md import smap, util, mm_utils, energy, space, partition, mm, quantity, simulate
import functools
from jax_md.util import Array

"""# Build/query an `openmm.System`
create an explicitly-solvated `openmm.System` of alanine (peptide) and query it's potential energy
"""

pdb = app.PDBFile("/content/jax-md/tests/data/alanine-dipeptide-explicit.pdb")
ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
model = app.Modeller(pdb.topology, pdb.positions)
#model.deleteWater()
mmSystem = ff.createSystem(model.topology,
    nonbondedMethod=app.CutoffPeriodic,
    constraints=None,
    rigidWater=False,
    removeCMMotion=False)

"""pull out the `jax-md` readable parameters and exceptions, then create a reaction-field system"""

params = mm_utils.parameters_from_openmm_system(mmSystem, use_rf_exceptions=True)
exceptions = params.nonbonded_exception_parameters.particles

rf_system = mm_utils.ReactionFieldConverter(
system=mmSystem).rf_system

"""build an `openmm.Context` with the `openmm.System` and query the potential energy/forces given initial positions"""

integrator = openmm.VerletIntegrator(1.*unit.femtoseconds)
context = openmm.Context(rf_system,
                         integrator)
context.setPositions(model.getPositions())
context.setPeriodicBoxVectors(*mmSystem.getDefaultPeriodicBoxVectors())

state = context.getState(getEnergy=True,
                         getForces=True)

state.getPotentialEnergy()

omm_forces = state.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)

omm_forces[:5]

"""# Build `jax-md` potential energy function.

extract periodic box vectors/positions
"""

mm_bvs = tuple(mmSystem.getDefaultPeriodicBoxVectors())
bvs = mm_utils.get_box_vectors_from_vec3s(mm_bvs)

positions = jnp.array(model.getPositions().value_in_unit_system(unit.md_unit_system))

"""build a neighbor list, create a reaction field pairwise interaction, and build the potential energy fn"""

energy_fn, neighbor_list_fns = mm.rf_mm_energy_fn(bvs, params, space.periodic,
                                                  aux_neighbor_list_kwargs={'dr_threshold': 1e-1})

nbrs = neighbor_list_fns.allocate(positions)

# get kT
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
kT = kB * 300*unit.kelvin
in_kT = kT.value_in_unit_system(unit.md_unit_system)

_, shift_fn = space.periodic(bvs)
dt=1e-3 # picoseconds
init, apply = simulate.nvt_langevin(energy_fn,
                                       shift_fn,
                                       dt,
                                       in_kT,
                                       gamma=1.)
# get masses in daltons
list_masses = jnp.array([mmSystem.getParticleMass(i).value_in_unit_system(unit.md_unit_system)
               for i in range(mmSystem.getNumParticles())])

state = init(jax.random.PRNGKey(0), positions, mass=list_masses, neighbor=nbrs)

def run_sim(num_steps, neighbor_list_fns):
  """
  run an md simulation; partial out second kwarg
  """
  def body_fn(_in, state_and_nbrs):
    state, nbrs = state_and_nbrs
    nbrs = neighbor_list_fns.update(state.position, nbrs)
    new_state = apply(state, neighbor=nbrs)
    return (new_state, nbrs)

  call_fn = lambda x: jax.lax.fori_loop(0, num_steps, body_fn, x) # x is tuple of state and neighbor
  return call_fn

run_1k_sim = jax.jit(partial(run_sim, neighbor_list_fns = neighbor_list_fns)(1000))

import tensorflow as tf
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
#jax.profiler.start_trace(log_dir='/tmp/simple_demo2')

_dr_threshold = 7.5e-2
_aux_neighbor_list_kwargs = {'dr_threshold': _dr_threshold,
                               'capacity_multiplier': 1.25}
energy_fn, neighbor_list_fns = mm.rf_mm_energy_fn(bvs,
                                                    params,
                                                    space.periodic,
                                                    aux_neighbor_list_kwargs = \
                                                    _aux_neighbor_list_kwargs)
nbrs = neighbor_list_fns.allocate(positions)
state = init(jax.random.PRNGKey(0), positions, mass=list_masses, neighbor=nbrs)
run_1k_sim = jax.jit(partial(run_sim, neighbor_list_fns = neighbor_list_fns)(10))
out_state, out_nbrs = run_1k_sim((state, nbrs))
_ = out_state.position.block_until_ready()

jax.profiler.start_trace(log_dir='/tmp/simple_demo',create_perfetto_link=True, create_perfetto_trace=True)
with jax.profiler.TraceAnnotation("train"):
  out_state, out_nbrs = run_1k_sim((state, nbrs))
  _ = out_state.position.block_until_ready()
