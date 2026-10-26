# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# [![Download Notebook](https://img.shields.io/badge/Download-Notebook-blue?style=for-the-badge&logo=jupyter)](https://jax-md.readthedocs.io/en/main/notebooks/getting_started.ipynb)
# [![Download Python Script](https://img.shields.io/badge/Download-Python_Script-green?style=for-the-badge&logo=python)](https://raw.githubusercontent.com/google/jax-md/main/examples/getting_started.py)

# %% [markdown]
# # Getting Started
#
# JAX MD stores particle data in arrays and describes a simulation as functions that map one state to the next. Everything is written in JAX, so the same code compiles with `jit`, runs on CPU, GPU or TPU, and can be differentiated.
#
# This page builds a small simulation of soft disks step by step: a periodic space, an energy function, an energy minimization, a simulation at constant temperature, and the same simulation with a neighbor list.

# %% [markdown]
# ## Imports

# %%
import os

IN_COLAB = 'COLAB_RELEASE_TAG' in os.environ
if IN_COLAB:
  import subprocess
  import sys

  subprocess.run(
    [
      sys.executable,
      '-m',
      'pip',
      'install',
      '-q',
      'git+https://github.com/jax-md/jax-md.git',
    ]
  )

import jax
import jax.numpy as jnp
from jax import jit, lax, random

import matplotlib.pyplot as plt

from jax_md import energy, minimize, quantity, simulate, space

SMOKE_TEST = os.environ.get('READTHEDOCS', False)

# %% [markdown]
# ## Spaces
#
# A space is a pair of functions. `displacement_fn(Ra, Rb)` returns the displacement from `Rb` to `Ra`, and `shift_fn(R, dR)` moves positions `R` by `dR`. Here we make a periodic square box that holds 1000 disks at a number density of 0.84. Displacements follow the minimum image convention, so two points near opposite edges of the box are close together.

# %%
N = 1000
dimension = 2
box_size = quantity.box_size_at_number_density(N, 0.84, dimension)
displacement_fn, shift_fn = space.periodic(box_size)

Ra = jnp.array([1.0, 1.0])
Rb = jnp.array([box_size - 1.0, 1.0])
print(f'Box size: {box_size:.3f}')
print(f'Displacement: {displacement_fn(Ra, Rb)}')

# %% [markdown]
# ## Energy and Forces
#
# `energy.soft_sphere_pair` returns a function from an `(N, dimension)` array of positions to the total energy of the system. Forces are the negative gradient of the energy, which `quantity.force` computes with automatic differentiation.

# %%
key = random.PRNGKey(0)
key, split = random.split(key)
R = random.uniform(split, (N, dimension), maxval=box_size)

energy_fn = energy.soft_sphere_pair(displacement_fn)
force_fn = quantity.force(energy_fn)

print(f'Energy: {energy_fn(R):.3f}')
print(f'Force on the first disk: {force_fn(R)[0]}')

# %% [markdown]
# ## Minimization
#
# Random positions overlap. FIRE descent moves the disks to a nearby energy minimum. Like the simulation routines, `minimize.fire_descent` returns a pair of functions: `init_fn` builds a state from positions and `apply_fn` advances it by one step. `jit` compiles the step.

# %%
fire_init, fire_apply = minimize.fire_descent(energy_fn, shift_fn)
fire_apply = jit(fire_apply)

fire_state = fire_init(R)
for _ in range(200):
  fire_state = fire_apply(fire_state)
R = fire_state.position
print(f'Energy after minimization: {energy_fn(R):.3f}')

# %% [markdown]
# ## Simulation
#
# `simulate.nvt_nose_hoover` integrates the equations of motion at constant temperature with a Nose-Hoover chain thermostat. The simulation state holds the positions, momenta, forces and thermostat variables. To keep the loop compiled, `lax.fori_loop` runs `write_every` steps at a time and `lax.scan` records the temperature after each block.

# %%
kT = 0.1
dt = 5e-3
steps = 1000 if SMOKE_TEST else 10000
write_every = 100

init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
key, split = random.split(key)
state = init_fn(split, R)


@jit
def run(state):
  def block(state, _):
    state = lax.fori_loop(0, write_every, lambda _, s: apply_fn(s), state)
    return state, simulate.temperature(state)

  return lax.scan(block, state, None, length=steps // write_every)


state, temperature = run(state)

time = dt * write_every * jnp.arange(1, len(temperature) + 1)
plt.plot(time, temperature)
plt.axhline(kT, color='black', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()

# %% [markdown]
# ## Neighbor Lists
#
# The pair energy above loops over all N² pairs. A neighbor list keeps, for each disk, the disks within the interaction range plus a buffer, which makes the cost linear in N. `energy.soft_sphere_neighbor_list` returns a function that builds neighbor lists and an energy function that takes one. `allocate` sizes the list from the current positions. `update` rebuilds it with the same size, so it can run inside compiled code.

# %%
neighbor_fn, neighbor_energy_fn = energy.soft_sphere_neighbor_list(
  displacement_fn, box_size
)
nbrs = neighbor_fn.allocate(state.position)
print(f'All pairs energy: {energy_fn(state.position):.6f}')
print(f'Neighbor list energy: {neighbor_energy_fn(state.position, nbrs):.6f}')

# %% [markdown]
# The simulation passes the neighbor list to the energy function as a keyword argument and updates it every step.

# %%
init_fn, apply_fn = simulate.nvt_nose_hoover(
  neighbor_energy_fn, shift_fn, dt, kT
)
key, split = random.split(key)
state = init_fn(split, state.position, neighbor=nbrs)


@jit
def run_neighbor_list(state, nbrs):
  def step(i, carry):
    state, nbrs = carry
    nbrs = nbrs.update(state.position)
    return apply_fn(state, neighbor=nbrs), nbrs

  return lax.fori_loop(0, steps, step, (state, nbrs))


state, nbrs = run_neighbor_list(state, nbrs)
print(f'Temperature: {simulate.temperature(state):.3f}')
print(f'Neighbor list overflowed: {bool(nbrs.did_buffer_overflow)}')

# %% [markdown]
# If `did_buffer_overflow` is true, the list ran out of space during the run. Allocate a new one from the current positions and run again.

# %% [markdown]
# ## Derivatives with Respect to Parameters
#
# Energy functions are ordinary JAX functions, so they can be differentiated with respect to their parameters as well as the positions. For example, the derivative of the energy of the final configuration with respect to the disk diameter:


# %%
def energy_at_diameter(sigma):
  return energy.soft_sphere_pair(displacement_fn, sigma=sigma)(state.position)


print(f'dE/dsigma: {jax.grad(energy_at_diameter)(1.0):.3f}')

# %% [markdown]
# ## Next Steps
#
# - The [examples](https://jax-md.readthedocs.io/en/main/examples.html) cover NVE, NVT and NPT simulations, neural network potentials, and more.
# - The [tutorial](https://jax-md.readthedocs.io/en/main/examples_tutorial.html) introduces JAX MD in five chapters.
# - The [API reference](https://jax-md.readthedocs.io/en/main/) documents each module.
# - [FEATURES.md](https://github.com/jax-md/jax-md/blob/main/FEATURES.md) is a tour of the main components of the library.
