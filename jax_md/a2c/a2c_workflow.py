# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Demonstration of a2c method for the prediction of the crystallization products 
# of amorphous phases as described in https://arxiv.org/abs/2310.01117 

import jax
from jax import lax
from jax import numpy as jnp
import numpy as np

from absl import app
from collections import defaultdict
from tqdm import tqdm

from jax_md import space, energy, quantity, simulate
from jax_md.minimize import fire_descent

import pymatgen as mg
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

from jax_md.a2c import make_amorphous_utils
from jax_md.a2c import crystallizer_utils

jax.config.update("jax_enable_x64", True)

# CONSTANTS
kb = 8.61733362e-5  # boltzmann constant in units of eV/K
fs = 0.09822693530999717  # femtoseconds in units of mass, energy, length == 1


def get_kt(step, equi_steps, cool_steps, T_high, T_low) -> float:
  # Gives a typical melt-quench-equilibrate profile at each step..
  if jnp.greater(equi_steps, step):
    return kb * T_high
  elif jnp.greater(cool_steps + equi_steps, step):
    return kb * (T_low + (T_high - T_low) * (1.0 - (step - equi_steps) / (cool_steps)))
  else:
    return kb * T_low

def main(unused_argv):

  # We will make a 64-atom amorphous Si phase by melt-quench NVT-MD
  composition = mg.core.Composition('Si64')
  box = jnp.array([[11.1, 0.0, 0.0], [0.0, 11.1, 0.0], [0.0, 0.0, 11.1]])
  # Get initial random packed structure with reduced overlap
  random_packed_structure = make_amorphous_utils.random_packed_structure(
      composition, lattice=box, auto_diameter=True)
  displacement, shift = space.periodic_general(box, fractional_coordinates=True,
                                              wrapped=False)
  # In this demo, we will use the Stillinger Weber potential for Si.
  # Set three_body_strength=2.0 (see jax-md documentation)
  energy_fn = energy.stillinger_weber(displacement, three_body_strength=2.0)
  energy_fn = jax.jit(energy_fn)

  equi_steps = 2500 # MD steps for melt equilibration
  cool_steps = 2500 # MD steps for quenching equilibration
  fina_steps = 2500 # MD steps for amorphous phase equilibration
  T_high =2000 # Melt temperature
  T_low = 300 # Quench to this temperature
  dt = 2 * fs # tim step = 2fs
  tau = 40 # oscillation period in Nose-Hoover thermostat

  simulation_steps = equi_steps + cool_steps + fina_steps
  position = random_packed_structure.frac_coords.copy() % 1.0
  mass = jnp.array([site.specie.atomic_mass for site in random_packed_structure])
  temperatures = jnp.array([
      get_kt(step, equi_steps, cool_steps, T_high, T_low)
      for step in range(0, simulation_steps, 1)])

  init, apply = simulate.nvt_nose_hoover(
      energy_or_force_fn=energy_fn,
      shift_fn=shift,
      dt=dt,
      kT=temperatures[0],
      tau=tau * dt)

  state = jax.jit(init)(jax.random.PRNGKey(42), position, mass)

  def step_fn(i, state_and_log):
    state, log = state_and_log
    T = quantity.temperature(velocity=state.velocity, mass=state.mass)
    log['kT'] = log['kT'].at[i].set(T)
    H = simulate.nvt_nose_hoover_invariant(energy_fn, state, T)
    log['H'] = log['H'].at[i].set(H)
    log['stress'] = log['stress'].at[i].set(quantity.stress(energy_fn, state.position, box))
    state = apply(state, **{'kT': temperatures[i]})
    return state, log

  steps = simulation_steps
  log = {
    'kT': jnp.zeros((steps,)),
    'H': jnp.zeros((steps,)),
    'stress': jnp.zeros((steps,3,3)),}
  
  # Run NVT-MD with the melt-quench-equilibrate temperature profile
  state, log = lax.fori_loop(0, steps, step_fn, (state, log))
  position = state.position
  amorphous_structure = mg.core.Structure(lattice=box.T, species=random_packed_structure.species, coords=state.position)
  print("Amorphous structure is ready:", amorphous_structure)

  subcells = crystallizer_utils.get_subcells_to_crystallize(amorphous_structure, 0.1, 2, 8)
  print("Created %d subcells from a-Si" % len(subcells))
  
  # To save time in this example, we (i) keep only the "cubic" subcells where a==b==c, and 
  # (ii) keep if number of atoms in the subcell is 2, 4 or 8. This rreduces the number of 
  # subcells to relax from approx. 80k to around 160.
  subcells = [subcell for subcell in subcells if np.all((subcell[2]-subcell[1]) == (subcell[2]-subcell[1])[0]) and subcell[0].shape[0] in (2,4,8)]
  print("Subcells kept for this example: %d" % len(subcells))

  structures = crystallizer_utils.subcells_to_structures(subcells, box=box, position=amorphous_structure.frac_coords, species=amorphous_structure.species)

  def get_energy_fn(box):
    # Get Stillinger Weber potential 
    displacement, shift = space.periodic_general(box, fractional_coordinates=True, wrapped=False)
    energy_fn = energy.stillinger_weber(displacement, three_body_strength=2.0)
    return jax.jit(energy_fn), shift

  # To relax the subcels, we apply fire_descent on atomic coordinates, and a basic gradient descent on the cell vectors.
  def relax_structure(s, n_steps=500, alpha=0.05):
    R = s.frac_coords
    box = s.lattice.matrix.T
    energy_fn, shift = get_energy_fn(box)
    fire_init, _ = fire_descent(energy_fn, shift)
    log = {'energy': jnp.zeros((n_steps,)),
         'stress': jnp.zeros((n_steps,3,3))}
    state = fire_init(R)

    def step_fn(i, state_box_log):
      state, box, log = state_box_log
      energy_fn, shift = get_energy_fn(box)
      log['energy'] = log['energy'].at[i].set(energy_fn(state.position))

      _,fire_apply = fire_descent(energy_fn, shift)
      state = fire_apply(state)

      stress = quantity.stress(energy_fn, state.position, box)
      log['stress'] =log['stress'].at[i].set(stress)

      box += alpha*stress
      return state, box, log

    state, box, log = lax.fori_loop(0, n_steps, step_fn, (state, box, log))
    energy_fn, _ = get_energy_fn(box)
    final_e, final_p, = energy_fn(state.position), jnp.sum(jnp.diagonal(quantity.stress(energy_fn, state.position, box)))/3
    return mg.core.Structure(lattice=box.T, species=s.species, coords=state.position), log, final_e, final_p

  relaxed_structures = []
  for s in tqdm(structures):
    relaxed_structures.append(relax_structure(s))

  lowest_e_struct = sorted(relaxed_structures, key=lambda x: x[-2]/x[0].num_sites)[0]
  spg = SpacegroupAnalyzer(lowest_e_struct[0])
  print("Space group of predicted crystallization product:", spg.get_space_group_symbol())

  spg_counter = defaultdict(lambda: 0)
  for s in relaxed_structures:
    try:
      sp = SpacegroupAnalyzer(s[0]).get_space_group_symbol()
    except TypeError:
      continue
    spg_counter[sp] += 1
  
  print("All space groups encountered:", dict(spg_counter))
  si_diamond = mg.core.Structure.from_str("""Si
  1.0
  0.000000000000   2.732954000000   2.732954000000
  2.732954000000   0.000000000000   2.732954000000
  2.732954000000   2.732954000000   0.000000000000
  Si
  2
  Direct
  0.500000000000   0.500000000000   0.500000000000
  0.750000000000   0.750000000000   0.750000000000""", fmt='poscar')
  print("Prediction matches diamond-cubic Si?", StructureMatcher().fit(lowest_e_struct[0], si_diamond))

if __name__ == '__main__':
  app.run(main)
