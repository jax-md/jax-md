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
# [![Download Notebook](https://img.shields.io/badge/Download-Notebook-blue?style=for-the-badge&logo=jupyter)](https://jax-md.readthedocs.io/en/main/notebooks/minimize.ipynb)
# [![Download Python Script](https://img.shields.io/badge/Download-Python_Script-green?style=for-the-badge&logo=python)](https://raw.githubusercontent.com/google/jax-md/main/examples/minimize.py)

# %% [markdown]
# Copyright 2019 Google LLC
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

# %% [markdown]
# # Energy Minimization
#
# This example demonstrates energy minimization of a two-dimensional system using the FIRE algorithm.

# %%
from jax import random, config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax_md import space, energy, minimize, quantity
from jax_md.util import f32, i32

# %% [markdown]
# ## Setup System Parameters
#
# We create a 2D system with 500 particles in a periodic box.

# %%
key = random.PRNGKey(0)

N = 500
dimension = 2
box_size = f32(25.0)

# Create periodic boundary conditions
displacement, shift = space.periodic(box_size)

# %% [markdown]
# ## Initialize Particle Positions
#
# Generate random initial positions and create a 50:50 mixture of two particle species.

# %%
key, split = random.split(key)
R = random.uniform(
    split, (N, dimension), minval=0.0, maxval=box_size, dtype=f32)

# Two particle types with different sizes
sigma = np.array([[1.0, 1.2], [1.2, 1.4]], dtype=f32)
N_2 = int(N / 2)
species = np.array([0] * N_2 + [1] * N_2, dtype=i32)

# %% [markdown]
# ## Create Energy Function and Minimizer
#
# We use a soft sphere pair potential and the FIRE descent minimization algorithm.

# %%
energy_fn = energy.soft_sphere_pair(displacement, species, sigma)
force_fn = quantity.force(energy_fn)

init_fn, apply_fn = minimize.fire_descent(energy_fn, shift)
opt_state = init_fn(R)

# %% [markdown]
# ## Run Minimization
#
# Minimize the system energy and track the progress.

# %%
minimize_steps = 50
print_every = 10

print('Minimizing.')
print('Step\tEnergy\tMax Force')
print('-----------------------------------')

for step in range(minimize_steps):
    opt_state = apply_fn(opt_state)
    
    if step % print_every == 0:
        R = opt_state.position
        print('{:.2f}\t{:.2f}\t{:.2f}'.format(
            step, energy_fn(R), np.max(force_fn(R))))

# %% [markdown]
# ## Optional: Save Trajectory
#
# For saving the minimization trajectory, you can use the I/O functions:

# %%
# Uncomment to save trajectory
# from jax_md.io import write_xyz
# 
# for step in range(minimize_steps):
#     opt_state = apply_fn(opt_state)
#     
#     if step % print_every == 0:
#         R = opt_state.position
#         print('{:.2f}\t{:.2f}\t{:.2f}'.format(
#             step, energy_fn(R), np.max(force_fn(R))))
#         # Save configuration
#         write_xyz(f"min_{step:06d}.xyz", species, R)
