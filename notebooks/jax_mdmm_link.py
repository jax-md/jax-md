from jax.config import config; config.update("jax_enable_x64", True)
#import openmm
import jax
#from openmm.app import Modeller
import numpy as onp
from functools import partial
import functools
from jax import numpy as jnp
from jax import vmap, jit
from jax_md import smap, util, mm_utils, energy, space, partition, mm, quantity, simulate, test_util
import functools
from jax_md.util import Array

"""build a neighbor list, create a reaction field pairwise interaction, and build the potential energy fn"""
out_expl_rf = test_util.decompress_pickle(
    #'/content/jax-md/tests/data/alanine_dipeptide_explicit_rf.pbz2') # what is this location
    '/mnt/c/Users/domin/github/jax-md/tests/data/alanine_dipeptide_explicit_rf.pbz2')

bvs = out_expl_rf['box_vectors']
params = out_expl_rf['mm_parameters']

energy_fn, neighbor_list_fns = mm.rf_mm_energy_fn(
  box_vectors=bvs,
  default_mm_parameters=params,
  aux_neighbor_list_kwargs={'dr_threshold': 1e-1}
  )

positions = out_expl_rf['positions']
nbrs = neighbor_list_fns.allocate(positions)

# get kT
in_kT = out_expl_rf['kT']

_, shift_fn = space.periodic(bvs)
dt=1e-3 # picoseconds
init, apply = simulate.nvt_langevin(energy_fn,
                                       shift_fn,
                                       dt,
                                       in_kT,
                                       gamma=1.)
# get masses in daltons
list_masses = out_expl_rf['list_masses']

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

#import tensorflow as tf
import jax
import jax.numpy as jnp
#import tensorflow_datasets as tfds
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
