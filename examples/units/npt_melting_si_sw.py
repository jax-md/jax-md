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
# [![Download Notebook](https://img.shields.io/badge/Download-Notebook-blue?style=for-the-badge&logo=jupyter)](https://jax-md.readthedocs.io/en/main/notebooks/units/npt_melting_si_sw.ipynb)
# [![Download Python Script](https://img.shields.io/badge/Download-Python_Script-green?style=for-the-badge&logo=python)](https://raw.githubusercontent.com/google/jax-md/main/examples/units/npt_melting_si_sw.py)

# %% [markdown]
# # Metal Units (NPT Simulation - Melting)
#
# This notebook demonstrates the use of a unit system (metal units) for the simulation of the Silicon crystal containing 512 atoms in the NPT ensemble with the Stillinger-Weber potential. The Si crystal is heated from 300 K to 3300 K at a rate of 10 K/ps. The crystal melts to form a liquid past its melting temperature. This notebook use lammps velocities and positions as a starting point for the simulation and for comparison.
#
# More about the unit system https://docs.lammps.org/units.html

# %% [markdown]
# ## Imports & Utils

# %%
import os

IN_COLAB = 'COLAB_RELEASE_TAG' in os.environ
if IN_COLAB:
  import subprocess, sys

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

import jax.numpy as jnp
import numpy as onp
from functools import partial
from jax import debug
from jax import jit
from jax import grad
from jax import random
from jax import lax
from jax import config

config.update('jax_enable_x64', True)
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import elasticity
from jax_md import quantity
from jax_md import dataclasses
from jax_md.util import f64

# Other libraries
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

# %% [markdown]
# ## Download LAMMPS Data

# %%
# LAMMPS data
import urllib.request

SMOKE_TEST = os.environ.get('READTHEDOCS', False)


def download_file(url, filename):
  if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)


base_url = 'https://raw.githubusercontent.com/abhijeetgangan/Silicon-data/main/Si-SW-MD/NPT-Melting/'
download_file(base_url + 'lammps.dat', 'npt_melting.dat')

# Download initial positions
base_url_nve = 'https://raw.githubusercontent.com/abhijeetgangan/Silicon-data/main/Si-SW-MD/NVE-300K/'
download_file(base_url_nve + 'step_1.traj', 'step_1.traj')

data_lammps = pd.read_csv('npt_melting.dat', delim_whitespace=True, header=None)
data_lammps = data_lammps.dropna(axis=1)
data_lammps.columns = ['Time', 'T', 'P', 'V', 'E']
t_l, T, P, V, E = (
  data_lammps['Time'],
  data_lammps['T'],
  data_lammps['P'],
  data_lammps['V'],
  data_lammps['E'],
)

# %% [markdown]
# ## Load LAMMPS Positions and Velocities

# %%
lammps_step_0 = onp.loadtxt('step_1.traj', dtype=f64)

# %%
# Load positions from lammps
positions = jnp.array(lammps_step_0[:, 2:5], dtype=f64)
# Load velocities from lammps
velocity = jnp.array(lammps_step_0[:, 5:8], dtype=f64)
latvec = jnp.array(
  [
    [21.724, 0.000000, 0.000000],
    [0.00000, 21.724, 0.00000],
    [0.00000, 0.0000, 21.724],
  ]
)

# %% [markdown]
# ## Units and Simulation Parameters

# %%
# Import unit system
from jax_md import units

# Metal units
unit = units.metal_unit_system()

# %%
# Simulation parameters
timestep = 1e-3
fs = timestep * unit['time']
ps = unit['time']
dt = fs
write_every = 100
box = latvec
T_init = 300 * unit['temperature']
T_final = (
  330 * unit['temperature'] if SMOKE_TEST else 3300 * unit['temperature']
)
P_init = 0.0 * unit['pressure']
Mass = 28.0855 * unit['mass']
key = random.PRNGKey(121)
NSTEPS_SIM = 3000 if SMOKE_TEST else 300000

# %%
# Logger to save data
log = {
  'E': jnp.zeros((NSTEPS_SIM // write_every,)),
  'P': jnp.zeros((NSTEPS_SIM // write_every,)),
  'V': jnp.zeros((NSTEPS_SIM // write_every,)),
  'kT': jnp.zeros((NSTEPS_SIM // write_every,)),
}

# %% [markdown]
# ## Simulation Setup

# %%
# Setup the periodic boundary conditions.
displacement, shift = space.periodic_general(latvec)
dist_fun = space.metric(displacement)
neighbor_fn, energy_fn = energy.stillinger_weber_neighbor_list(
  displacement, latvec, disable_cell_list=True
)
energy_fn = jit(energy_fn)


# %%
# Thermostat and barostat parameters same as LAMMPS
def default_nhc_kwargs(tau: f64, overrides: Dict) -> Dict:
  default_kwargs = {
    'chain_length': 3,
    'chain_steps': 1,
    'sy_steps': 1,
    'tau': tau,
  }
  if overrides is None:
    return default_kwargs
  return {
    key: overrides.get(key, default_kwargs[key]) for key in default_kwargs
  }


new_kwargs = {
  'chain_length': 3,
  'chain_steps': 1,
  'sy_steps': 1,
}

# %%
# Extra capacity to prevent overflow
nbrs = neighbor_fn.allocate(positions, box=box, extra_capacity=4)

# NPT simulation
init_fn, apply_fn = simulate.npt_nose_hoover(
  energy_fn,
  shift,
  dt=dt,
  pressure=P_init,
  kT=T_init,
  barostat_kwargs=default_nhc_kwargs(1000 * dt, new_kwargs),
  thermostat_kwargs=default_nhc_kwargs(100 * dt, new_kwargs),
)
apply_fn = jit(apply_fn)
state = init_fn(key, positions, box=box, neighbor=nbrs, kT=T_init, mass=Mass)

# Restart from LAMMPS velocities
state = dataclasses.replace(state, momentum=Mass * velocity * unit['velocity'])

# %% [markdown]
# ## NPT Simulation

# %% [markdown]
# ## Heating Schedule


# %%
# 10K/ps heating
@jit
def T_schedule(T_init, T_final, Nsteps, i):
  TI, TF = T_init, T_final
  kT = ((TF - TI) / (Nsteps)) * (i) + TI
  return kT


T_schedule = jit(partial(T_schedule, T_init, T_final, NSTEPS_SIM))


# %%
@jit
def step_fn(i, state_nbrs_box_j):
  state, nbrs, box, j = state_nbrs_box_j
  # Take a simulation step.
  t = i * dt
  state = apply_fn(
    state, neighbor=nbrs, kT=T_schedule(j * write_every + i), pressure=P_init
  )
  box = simulate.npt_box(state)
  nbrs = nbrs.update(state.position, neighbor=nbrs, box=box)
  return state, nbrs, box, j


@jit
def outer_sim_fn(j, state_nbrs_log_box):
  state, nbrs, log, box = state_nbrs_log_box

  # Quantities to calculate
  K = quantity.kinetic_energy(momentum=state.momentum, mass=Mass)
  E = energy_fn(state.position, box=box, neighbor=nbrs)
  kT = quantity.temperature(momentum=state.momentum, mass=Mass)
  P = quantity.pressure(energy_fn, state.position, box, K, neighbor=nbrs)

  # Save the quantities
  log['V'] = log['V'].at[j].set(quantity.volume(3, box=box))
  log['E'] = log['E'].at[j].set(E)
  log['kT'] = log['kT'].at[j].set(kT)
  log['P'] = log['P'].at[j].set(P)

  # Print the quantities
  # debug.print('Step = {j} | Temp = {T}', j=j*write_every, T= kT / unit['temperature'])

  @jit
  def inner_sim_fn(i, state_nbrs_box):
    return step_fn(i, state_nbrs_box)

  state, nbrs, box, j = lax.fori_loop(
    0, write_every, inner_sim_fn, (state, nbrs, box, j)
  )

  return state, nbrs, log, box


# %%
state_r, nbrs_r, log_r, box_r = lax.fori_loop(
  0, int(NSTEPS_SIM / write_every), outer_sim_fn, (state, nbrs, log, box)
)

# %%
# Check if neighbors overflowed
print(nbrs_r.did_buffer_overflow)

# %% [markdown]
# ## Comparison Plot

# %% [markdown]
# Note that you have to reconvert the units again.

# %%
NSTEPS = int(NSTEPS_SIM / write_every)
t = jnp.arange(0, NSTEPS, dtype=f64) * timestep * write_every

# %%
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(t_l[:NSTEPS], T[:NSTEPS], lw=1, label='LAMMPS')
ax1.plot(t, log_r['kT'] / unit['temperature'], lw=1, label='JAX MD')
ax1.set_title('Temperature', fontsize=16)
ax1.set_ylabel('$T\\ (K)$', fontsize=16)
ax1.set_xlabel('$t\\ (ps)$', fontsize=16)
ax1.legend()

ax2 = plt.subplot(2, 2, 2)
ax2.plot(t_l[:NSTEPS], P[:NSTEPS] / 10000, lw=1, label='LAMMPS')
ax2.plot(t, (log_r['P'] / unit['pressure']) / 10000, lw=1, label='JAX MD')
ax2.set_title('Pressure', fontsize=16)
ax2.set_ylabel('$P\\ (GPa)$', fontsize=16)
ax2.set_xlabel('$t\\ (ps)$', fontsize=16)
ax2.legend()

ax3 = plt.subplot(2, 2, 3)
ax3.plot(t_l[:NSTEPS], E[:NSTEPS], lw=1, label='LAMMPS')
ax3.plot(t, log_r['E'], lw=1, label='JAX MD')
ax3.set_title('Potential Energy', fontsize=16)
ax3.set_ylabel('$E_{PE}\\ (eV)$', fontsize=16)
ax3.set_xlabel('$t\\ (ps)$', fontsize=16)
ax3.legend()

ax4 = plt.subplot(2, 2, 4)
ax4.plot(t_l[:NSTEPS], V[:NSTEPS], lw=1, label='LAMMPS')
ax4.plot(t, log_r['V'], lw=1, label='JAX MD')
ax4.set_title('Volume', fontsize=16)
ax4.set_ylabel('$V\\  (\\AA^3)$', fontsize=16)
ax4.set_xlabel('$t\\ (ps)$', fontsize=16)
ax4.legend()

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Temperature vs Other Properties

# %%
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(t_l[:NSTEPS], T[:NSTEPS], lw=1, label='LAMMPS')
ax1.plot(t, log_r['kT'] / unit['temperature'], lw=1, label='JAX MD')
ax1.set_title('Temperature', fontsize=16)
ax1.set_ylabel('$T\\ (K)$', fontsize=16)
ax1.set_xlabel('$t\\ (ps)$', fontsize=16)
ax1.legend()

ax2 = plt.subplot(2, 2, 2)
ax2.plot(T[:NSTEPS], P[:NSTEPS] / 10000, lw=1, label='LAMMPS')
ax2.plot(
  log_r['kT'] / unit['temperature'],
  (log_r['P'] / unit['pressure']) / 10000,
  lw=1,
  label='JAX MD',
)
ax2.set_title('Pressure', fontsize=16)
ax2.set_ylabel('$P\\ (GPa)$', fontsize=16)
ax2.set_xlabel('$T\\ (K)$', fontsize=16)
ax2.legend()

ax3 = plt.subplot(2, 2, 3)
ax3.plot(T[:NSTEPS], E[:NSTEPS], lw=1, label='LAMMPS')
ax3.plot(log_r['kT'] / unit['temperature'], log_r['E'], lw=1, label='JAX MD')
ax3.set_title('Potential Energy', fontsize=16)
ax3.set_ylabel('$E_{PE}\\ (eV)$', fontsize=16)
ax3.set_xlabel('$T\\ (K)$', fontsize=16)
ax3.legend()

ax4 = plt.subplot(2, 2, 4)
ax4.plot(T[:NSTEPS], V[:NSTEPS], lw=1, label='LAMMPS')
ax4.plot(log_r['kT'] / unit['temperature'], log_r['V'], lw=1, label='JAX MD')
ax4.set_title('Volume', fontsize=16)
ax4.set_ylabel('$V\\  (\\AA^3)$', fontsize=16)
ax4.set_xlabel('$T\\ (K)$', fontsize=16)
ax4.legend()

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Radial Distribution Function (RDF)
#
# Calculate RDF at initial and final states

# %%
# RDF parameters
dr = 0.1  # Bin width in Angstroms
r_max = 10.0  # Maximum distance
radii = jnp.arange(0, r_max, dr)

# Create RDF function
neighbor_fn_rdf, g_fn = quantity.pair_correlation_neighbor_list(
  displacement, latvec, radii, dr, species=None
)

# %%
# Calculate initial RDF (crystalline state)
nbrs_rdf_init = neighbor_fn_rdf.allocate(
  positions, box=latvec, extra_capacity=2
)
g_r_init = g_fn(positions, neighbor=nbrs_rdf_init)

# Calculate final RDF (molten state)
nbrs_rdf_final = neighbor_fn_rdf.allocate(
  state_r.position, box=box_r, extra_capacity=2
)
g_r_final = g_fn(state_r.position, neighbor=nbrs_rdf_final)

# %% [markdown]
# ## Plot RDF

# %%
plt.figure(figsize=(10, 6))
plt.plot(
  radii,
  jnp.mean(g_r_init, axis=0),
  lw=2,
  label=f'Initial (T = {T_init / unit["temperature"]:.0f} K)',
)
plt.plot(
  radii,
  jnp.mean(g_r_final, axis=0),
  lw=2,
  label=f'Final (T = {log_r["kT"][-1] / unit["temperature"]:.0f} K)',
)
plt.xlabel(r'$r\ (\AA)$', fontsize=14)
plt.ylabel('g(r)', fontsize=14)
plt.title('Radial Distribution Function', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, None)
plt.tight_layout()
plt.show()
