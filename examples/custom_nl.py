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
# [![Download Notebook](https://img.shields.io/badge/Download-Notebook-blue?style=for-the-badge&logo=jupyter)](https://jax-md.readthedocs.io/en/main/notebooks/custom_nl.ipynb)
# [![Download Python Script](https://img.shields.io/badge/Download-Python_Script-green?style=for-the-badge&logo=python)](https://raw.githubusercontent.com/google/jax-md/main/examples/custom_nl.py)

# %% [markdown]
# # Multi-Image Neighbor Lists for Small Periodic Boxes
#
# This tutorial demonstrates how to use **multi-image neighbor lists** in JAX-MD
# for systems where the cutoff radius is larger than half the box length
# ($r_\text{cut} > L/2$).
#
# ## The Problem with Standard Neighbor Lists
#
# Standard neighbor lists in JAX-MD use the **Minimum Image Convention (MIC)**,
# which assumes each atom interacts with at most one periodic image of every
# other atom. This works well when:
#
# $$r_\text{cut} < \frac{L}{2}$$
#
# However, for small periodic boxes (common in ab initio MD or machine learning
# potentials with longer cutoffs), an atom may interact with **multiple periodic
# images** of the same neighbor. The multi-image neighbor list explicitly
# enumerates all images within the cutoff.

# %% [markdown]
# ## Imports & Setup

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

import numpy as onp

from jax import config

config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax import random, jit, lax

import time
import matplotlib.pyplot as plt
import seaborn as sns

from jax_md import space, energy, partition, quantity, simulate
from jax_md.custom_partition import (
  neighbor_list_multi_image,
  estimate_max_neighbors_from_box,
  estimate_max_neighbors,
)
from jax_md.custom_smap import pair_neighbor_list_multi_image

SMOKE_TEST = os.environ.get('READTHEDOCS', False)

sns.set_style(style='white')


def format_plot(x, y):
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1],
  )
  plt.tight_layout()


# %% [markdown]
# ## Helper: Create Crystal Structures


# %%
def make_fcc(n_cells, a=1.0):
  """Create FCC crystal positions in fractional coordinates.

  Args:
    n_cells: Number of unit cells in each direction.
    a: Lattice constant.

  Returns:
    R: Fractional positions of shape [N, 3] in [0, 1).
    box: Box matrix of shape [3, 3] with columns as lattice vectors.
  """
  # FCC basis: 4 atoms per unit cell
  basis = onp.array(
    [
      [0.0, 0.0, 0.0],
      [0.5, 0.5, 0.0],
      [0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5],
    ]
  )

  positions = []
  for i in range(n_cells):
    for j in range(n_cells):
      for k in range(n_cells):
        for b in basis:
          pos = (onp.array([i, j, k]) + b) / n_cells
          positions.append(pos)

  R = onp.array(positions)
  L = n_cells * a
  box = onp.eye(3) * L
  return jnp.array(R), jnp.array(box)


def make_diamond_cubic(n_cells, a=5.43):
  """Create diamond cubic crystal using the 2-atom primitive cell.

  Uses the FCC primitive cell with a 2-atom basis:
  - Lattice vectors: a1=(0,1,1)a/2, a2=(1,0,1)a/2, a3=(1,1,0)a/2
  - Basis: (0,0,0) and (1/4,1/4,1/4) in fractional coordinates

  This is more efficient than the 8-atom conventional cell.
  Used for silicon (a=5.43 Å) and germanium (a=5.66 Å).

  Args:
    n_cells: Number of primitive cells in each direction.
    a: Conventional cubic lattice constant (default 5.43 Å for silicon).

  Returns:
    R: Fractional positions of shape [N, 3] in [0, 1).
    box: Box matrix of shape [3, 3] with columns as FCC primitive vectors.
  """
  # 2-atom basis in fractional coordinates of primitive cell
  basis = onp.array(
    [
      [0.0, 0.0, 0.0],
      [0.25, 0.25, 0.25],
    ]
  )

  positions = []
  for i in range(n_cells):
    for j in range(n_cells):
      for k in range(n_cells):
        for b in basis:
          pos = (onp.array([i, j, k]) + b) / n_cells
          positions.append(pos)

  R = onp.array(positions)

  # FCC primitive lattice vectors (columns of box matrix)
  # a1 = (0, a/2, a/2), a2 = (a/2, 0, a/2), a3 = (a/2, a/2, 0)
  box = (a / 2.0) * onp.array(
    [
      [0.0, 1.0, 1.0],
      [1.0, 0.0, 1.0],
      [1.0, 1.0, 0.0],
    ]
  )
  # Scale by n_cells
  box = box * n_cells

  return jnp.array(R), jnp.array(box)


# %% [markdown]
# ## Example 1: Lennard-Jones with All Three neighbor list formats
#
# We compute LJ energy for a small FCC argon crystal using all three neighbor
# list formats to verify they produce identical results:
#
# - **Dense**: Per-atom neighbor arrays `[N, max_neighbors]`
# - **Sparse**: Edge list `[2, capacity]` with both directions
# - **OrderedSparse**: Edge list with one direction per pair (most efficient)

# %%
# Argon LJ parameters (reduced units: sigma=1, epsilon=1)
sigma = 1.0  # Length unit
epsilon = 1.0  # Energy unit
r_cutoff = 2.5 * sigma
r_onset = 2.0 * sigma

# Create small FCC argon crystal where r_cut > L/2
# In reduced units, equilibrium nearest-neighbor distance ≈ 2^(1/6) * sigma ≈ 1.12
# FCC lattice constant a = sqrt(2) * nearest_neighbor ≈ 1.58 in reduced units
n_cells = 2
a_reduced = 1.55  # Small box to test multi-image (r_cut/L > 0.5)
R, box = make_fcc(n_cells, a=a_reduced)
N = len(R)
L = float(box[0, 0])

print(f'System: {N} Ar atoms in {n_cells}x{n_cells}x{n_cells} FCC')
print(f'Box size L = {L:.2f}sigma, r_cutoff = {r_cutoff:.2f}sigma')
print(f'r_cutoff / L = {r_cutoff / L:.2f} (> 0.5: multi-image needed)')

# Estimate max neighbors for capacity allocation
max_nbrs = estimate_max_neighbors_from_box(box, r_cutoff, n_atoms=N)
print(f'Estimated max neighbors: {max_nbrs}')

# Setup displacement function
displacement_fn, shift_fn = space.periodic_general(
  box, fractional_coordinates=True
)

# Test all three formats
formats = [
  ('Dense', partition.Dense),
  ('Sparse', partition.Sparse),
  ('OrderedSparse', partition.OrderedSparse),
]

energies = {}
for name, fmt in formats:
  neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
    displacement_fn,
    box,
    sigma=sigma,
    epsilon=epsilon,
    r_onset=r_onset / sigma,
    r_cutoff=r_cutoff / sigma,
    fractional_coordinates=True,
    neighbor_list_fn=neighbor_list_multi_image,
    pair_neighbor_list_fn=pair_neighbor_list_multi_image,
    max_neighbors=max_nbrs,
    format=fmt,
  )

  nbrs = neighbor_fn.allocate(R)
  E = float(energy_fn(R, nbrs))
  energies[name] = E

  # Get neighbor count
  if partition.is_sparse(fmt):
    n_edges = int(jnp.sum(nbrs.idx[0] < N))
  else:
    n_edges = int(jnp.sum(nbrs.idx < N))

  print(f'{name:15s}: E = {E:12.6f}, edges = {n_edges}')

# Verify all formats give the same energy
E_ref = energies['Dense']
for name, E in energies.items():
  assert abs(E - E_ref) < 1e-5, f'{name} energy mismatch: {E} vs {E_ref}'
# %% [markdown]
# ## Example 2: Stillinger-Weber (Three-Body Potential)
#
# Stillinger-Weber is a three-body potential for silicon that requires
# **Dense** format for the angular terms. We use a 2x2x2 supercell of the
# 2-atom primitive cell.
#
# **Note**: Stillinger-Weber internally uses `space.map_neighbor` for
# displacement computation, which applies the minimum image convention (MIC).
# For small boxes where `r_cut > L/2`, the multi-image neighbor list finds
# the correct neighbors, but the energy computation would still use MIC
# displacements. Therefore, we use a larger box where MIC is valid.

# %%
# Stillinger-Weber parameters for silicon
sw_sigma = 2.0951  # Angstrom
sw_cutoff = 1.8 * sw_sigma  # ~3.77 Angstrom
r_cutoff_sw = 3.77118

# Create 3x3x3 supercell so that MIC is valid
# For SW, the box must be large enough that r_cut < L/2
n_cells_sw = 3
a_sw = 5.43  # Si lattice constant
R_sw, box_sw = make_diamond_cubic(n_cells_sw, a=a_sw)
N_sw = len(R_sw)

# For non-cubic boxes, compute minimum perpendicular height
inv_box_T = jnp.linalg.inv(box_sw).T
heights_sw = 1.0 / jnp.linalg.norm(inv_box_T, axis=0)
L_min_sw = float(jnp.min(heights_sw))

print(f'System: {N_sw} Si atoms in {n_cells_sw}x{n_cells_sw}x{n_cells_sw} diamond cubic supercell')
print(f'Min box height = {L_min_sw:.2f} Angstrom, SW cutoff = {sw_cutoff:.2f} Angstrom')
print(f'cutoff / L_min = {sw_cutoff / L_min_sw:.2f} (< 0.5: MIC is valid)')

displacement_sw, shift_sw = space.periodic_general(
  box_sw, fractional_coordinates=True
)

# Stillinger-Weber only supports Dense format (three-body terms)
# Note: SW uses MIC internally, so multi-image NL only helps with neighbor finding
neighbor_fn_sw, energy_fn_sw = energy.stillinger_weber_neighbor_list(
  displacement_sw,
  box_sw,
  neighbor_list_fn=neighbor_list_multi_image,
  max_neighbors=estimate_max_neighbors(r_cutoff_sw, atomic_density=0.15, dim=3),
  format=partition.Dense,
  fractional_coordinates=True,
)

nbrs_sw = neighbor_fn_sw.allocate(R_sw)
E_sw = float(energy_fn_sw(R_sw, nbrs_sw))
n_edges_sw = int(jnp.sum(nbrs_sw.idx < N_sw))

print(f'Stillinger-Weber energy: {E_sw:.6f} eV')
print(f'Number of edges: {n_edges_sw}')
print('Stillinger-Weber computes correctly (MIC valid for this box size).')

# %% [markdown]
# ## Example 3: NVE Molecular Dynamics
#
# We run NVE (constant energy) molecular dynamics with the multi-image neighbor
# list. This demonstrates rebuild tracking and overflow handling following the
# pattern recommended in `partition.neighbor_list` documentation.

# %%

# Simulation parameters
N_md = 500
dimension = 2
box_size = 40.0 if SMOKE_TEST else 60.0

# Create box matrix for 2D
box_md = jnp.eye(dimension) * box_size

# Random initial positions (fractional coordinates in [0, 1))
key = random.PRNGKey(0)
R_md = random.uniform(key, (N_md, dimension), minval=0.0, maxval=1.0)

# 50:50 mixture of two species
sigma_md = jnp.array([[1.0, 1.2], [1.2, 1.4]])
N_half = N_md // 2
species = jnp.where(jnp.arange(N_md) < N_half, 0, 1)

# Cutoff
r_cutoff_md = 2.5

print(f'System: {N_md} atoms in {dimension}D box of size {box_size}')
print(f'Cutoff: {r_cutoff_md}, cutoff/L = {r_cutoff_md / box_size:.3f}')

# Setup displacement function for fractional coordinates
displacement_md, shift_md = space.periodic_general(
  box_md, fractional_coordinates=True
)

# For random positions, use generous capacity to avoid overflow
# Random positions can cluster, requiring more capacity than uniform estimates
max_nbrs_md = 50  # Conservative value for random systems
print(f'Max neighbors: {max_nbrs_md}')

# Use soft sphere potential with multi-image neighbor list
neighbor_fn_md, energy_fn_md = energy.soft_sphere_neighbor_list(
  displacement_md,
  box_md,
  species=species,
  sigma=sigma_md,
  fractional_coordinates=True,
  neighbor_list_fn=neighbor_list_multi_image,
  pair_neighbor_list_fn=pair_neighbor_list_multi_image,
  max_neighbors=max_nbrs_md,
  format=partition.Sparse,
)

# Initialize neighbor list
nbrs_md = neighbor_fn_md.allocate(R_md)
if nbrs_md.did_buffer_overflow:
  raise RuntimeError('Neighbor list overflowed - increase max_neighbors')

# Setup NVE integrator
dt = 1e-2
init_fn, apply_fn = simulate.nve(energy_fn_md, shift_md, dt)

# Initialize state with zero temperature
state = init_fn(key, R_md, neighbor=nbrs_md, kT=0.0)


# JIT-compiled step function with neighbor list update
@jit
def step_fn(i, state_and_nbrs):
  state, nbrs = state_and_nbrs
  state = apply_fn(state, neighbor=nbrs)
  nbrs = nbrs.update(state.position)
  return state, nbrs


# %%
# Run simulation following JAX-MD's recommended pattern for overflow handling.
# See partition.neighbor_list docstring for the canonical example.
N_steps = 200 if SMOKE_TEST else 1000
print_every = 20
inner_steps = 10

PE = []
KE = []
rebuild_count = 0
realloc_count = 0

print(f'{"Step":>4} {"KE":>5} {"PE":>6} {"Total":>6} {"dt":>6} {"rebuild":>7} {"realloc":>7}')
old_time = time.time()

for i in range(N_steps):
  # Track reference position before inner loop
  old_ref_pos = nbrs_md.reference_position

  # Run inner_steps using fori_loop for efficiency
  new_state, new_nbrs = lax.fori_loop(0, inner_steps, step_fn, (state, nbrs_md))

  # Check for buffer overflow after the loop
  # If overflow: discard new state, reallocate with extra capacity
  # If no overflow: accept new state
  if new_nbrs.did_buffer_overflow:
    # Reallocate with extra capacity (10 more neighbors per atom)
    nbrs_md = neighbor_fn_md.allocate(state.position, extra_capacity=10)
    realloc_count += 1
    print(f'  [Overflow at step {i * inner_steps}! Reallocating with extra capacity...]')
    # Don't advance state - retry from last good state
  else:
    # Accept the new state
    state = new_state
    nbrs_md = new_nbrs

    # Check if rebuild happened (reference position changed)
    new_ref_pos = nbrs_md.reference_position
    if not jnp.allclose(old_ref_pos, new_ref_pos):
      rebuild_count += 1

  pe = float(energy_fn_md(state.position, nbrs_md))
  ke = float(quantity.kinetic_energy(momentum=state.momentum))
  PE.append(pe)
  KE.append(ke)

  if i % print_every == 0 and i > 0:
    new_time = time.time()
    step_time = (new_time - old_time) / print_every / inner_steps
    print(
      f'{i * inner_steps:4d} {ke:5.1f} {pe:6.1f} {ke + pe:6.2f} '
      f'{step_time:6.4f} {rebuild_count:7d} {realloc_count:7d}'
    )
    old_time = new_time

PE = jnp.array(PE)
KE = jnp.array(KE)

print(f'Total energy drift: {abs(float(PE[-1] + KE[-1] - PE[0] - KE[0])):.2e}')
print(f'Total rebuilds: {rebuild_count}, reallocs: {realloc_count}')

# %% [markdown]
# ## Plot Energy Evolution
#
# We verify energy conservation by plotting PE, KE, and total energy over time.

# %%
t = onp.arange(N_steps) * dt * inner_steps

plt.figure(figsize=(10, 6))
plt.plot(t, PE, label='PE', linewidth=2)
plt.plot(t, KE, label='KE', linewidth=2)
plt.plot(t, PE + KE, label='Total Energy', linewidth=2, linestyle='--')
plt.legend(fontsize=12)
format_plot('Time', 'Energy')
plt.title('NVE Energy Conservation with Multi-Image Neighbor List', fontsize=14)
finalize_plot()
plt.savefig('nve_multi_image.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Visualize Final Configuration

# %%
ms = 40 if SMOKE_TEST else 15
R_final = onp.array(state.position)

# Convert from fractional to Cartesian for plotting
R_cart = R_final * box_size

plt.figure(figsize=(8, 8))
plt.plot(
  R_cart[:N_half, 0], R_cart[:N_half, 1], 'o', markersize=ms * 0.5, alpha=0.7
)
plt.plot(
  R_cart[N_half:, 0], R_cart[N_half:, 1], 'o', markersize=ms * 0.7, alpha=0.7
)
plt.xlim([0, box_size])
plt.ylim([0, box_size])
plt.axis('off')
plt.title('Final Configuration', fontsize=14)
finalize_plot((2, 2))
plt.savefig('nve_final_config.png', dpi=150, bbox_inches='tight')
plt.show()
