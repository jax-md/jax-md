# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Molecular Dynamics with UMA and JAX-MD
#
# This example demonstrates running an NVE molecular dynamics simulation
# using the UMA neural network potential with JAX-MD.
#
# The simulation is fully differentiable and JIT-compiled for performance.

# %%
import jax.numpy as jnp
from jax import jit, random
import numpy as np

from jax_md import space, energy, simulate, quantity
from jax_md._nn.uma.model import UMAConfig

# %% [markdown]
# ## System Setup

# %%
# Diamond cubic silicon
a = 5.43
box = jnp.eye(3) * a

# Basis positions (fractional -> Cartesian)
basis_frac = jnp.array([
  [0.00, 0.00, 0.00],
  [0.50, 0.50, 0.00],
  [0.50, 0.00, 0.50],
  [0.00, 0.50, 0.50],
  [0.25, 0.25, 0.25],
  [0.75, 0.75, 0.25],
  [0.75, 0.25, 0.75],
  [0.25, 0.75, 0.75],
])
positions = basis_frac @ box
atoms = jnp.array([14] * 8, dtype=jnp.int32)  # Silicon

print(f"System: {len(atoms)} Si atoms")
print(f"Box: {a:.2f} x {a:.2f} x {a:.2f} A")

# %% [markdown]
# ## Build UMA Potential

# %%
displacement_fn, shift_fn = space.periodic(a)

# Small config for demo
cfg = UMAConfig(
  sphere_channels=32,
  lmax=2,
  mmax=2,
  num_layers=1,
  hidden_channels=32,
  cutoff=5.0,
  edge_channels=32,
  num_distance_basis=64,
  use_dataset_embedding=False,
)

neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
  displacement_fn, a, cfg=cfg, atoms=atoms,
)

# Initialize
key = random.PRNGKey(0)
nbrs = neighbor_fn.allocate(positions)
params = init_fn(key, positions, nbrs)

# Test energy and forces
E0 = energy_fn(params, positions, nbrs)
F0 = quantity.force(lambda R: energy_fn(params, R, nbrs))(positions)
print(f"Initial energy: {E0:.6f}")
print(f"Max force: {jnp.max(jnp.abs(F0)):.6f}")

# %% [markdown]
# ## NVE Molecular Dynamics
#
# Run microcanonical (constant energy) dynamics. The forces are computed
# as `F = -grad(E)` via JAX autodiff, ensuring energy conservation.

# %%
dt = 0.001  # Timestep (reduced units)
kT = 0.1    # Initial temperature (reduced units)

# Energy function that includes neighbor list update
def nve_energy_fn(R, neighbor, **kwargs):
  return energy_fn(params, R, neighbor)

# Initialize NVE simulation
init_fn_nve, apply_fn_nve = simulate.nve(nve_energy_fn, shift_fn, dt=dt)
apply_fn_nve = jit(apply_fn_nve)

# Initialize with thermal velocities
key, subkey = random.split(key)
state = init_fn_nve(subkey, positions, kT=kT, neighbor=nbrs)

print("\nInitial state:")
print(f"  KE: {quantity.kinetic_energy(state.momentum, state.mass):.6f}")
print(f"  PE: {nve_energy_fn(state.position, nbrs):.6f}")

# %% [markdown]
# ## Run Simulation

# %%
num_steps = 200
save_every = 10

trajectory = []
energies = {'kinetic': [], 'potential': [], 'total': []}

print(f"\n{'Step':>5} {'KE':>12} {'PE':>12} {'Total':>12}")
print("-" * 45)

for step in range(num_steps):
  # Update neighbor list periodically
  nbrs = nbrs.update(state.position)

  state = apply_fn_nve(state, neighbor=nbrs)

  if step % save_every == 0:
    KE = float(quantity.kinetic_energy(state.momentum, state.mass))
    PE = float(nve_energy_fn(state.position, nbrs))
    total = KE + PE

    energies['kinetic'].append(KE)
    energies['potential'].append(PE)
    energies['total'].append(total)
    trajectory.append(np.array(state.position))

    if step % (save_every * 5) == 0:
      print(f"{step:5d} {KE:12.6f} {PE:12.6f} {total:12.6f}")

print(f"\nSimulation complete: {num_steps} steps")

# %% [markdown]
# ## Analysis

# %%
try:
  import matplotlib.pyplot as plt

  steps = np.arange(0, num_steps, save_every)

  fig, axes = plt.subplots(1, 3, figsize=(15, 4))

  # Energy conservation
  total = np.array(energies['total'])
  drift = (total - total[0]) / np.abs(total[0]) * 100
  axes[0].plot(steps, drift, linewidth=2)
  axes[0].set_xlabel('Step')
  axes[0].set_ylabel('Energy Drift (%)')
  axes[0].set_title('Energy Conservation')
  axes[0].grid(True)

  # KE and PE
  axes[1].plot(steps, energies['kinetic'], label='KE', linewidth=2)
  axes[1].plot(steps, energies['potential'], label='PE', linewidth=2)
  axes[1].plot(steps, energies['total'], label='Total', linewidth=2, linestyle='--')
  axes[1].set_xlabel('Step')
  axes[1].set_ylabel('Energy')
  axes[1].set_title('Energy Components')
  axes[1].legend()
  axes[1].grid(True)

  # Temperature
  N = len(atoms)
  temps = [2 * KE / (3 * N) for KE in energies['kinetic']]  # kB = 1 in reduced units
  axes[2].plot(steps, temps, linewidth=2)
  axes[2].set_xlabel('Step')
  axes[2].set_ylabel('Temperature')
  axes[2].set_title('Temperature')
  axes[2].grid(True)

  plt.tight_layout()
  plt.savefig('uma_md_simulation.png', dpi=150)
  plt.show()
  print("Plot saved to uma_md_simulation.png")
except ImportError:
  print("matplotlib not available, skipping plots")

# %% [markdown]
# ## NVT Dynamics (Nose-Hoover Thermostat)
#
# For constant-temperature simulations, use JAX-MD's Nose-Hoover thermostat:

# %%
kT_target = 0.1
tau = dt * 100  # Thermostat coupling time

init_fn_nvt, apply_fn_nvt = simulate.nvt_nose_hoover(
  nve_energy_fn, shift_fn, dt=dt, kT=kT_target, tau=tau,
)
apply_fn_nvt = jit(apply_fn_nvt)

state_nvt = init_fn_nvt(key, positions, kT=kT_target, neighbor=nbrs)

print("Running NVT (Nose-Hoover)...")
for step in range(100):
  nbrs = nbrs.update(state_nvt.position)
  state_nvt = apply_fn_nvt(state_nvt, kT=kT_target, neighbor=nbrs)

KE = float(quantity.kinetic_energy(state_nvt.momentum, state_nvt.mass))
T = 2 * KE / (3 * len(atoms))
print(f"Final temperature: {T:.4f} (target: {kT_target})")

# %% [markdown]
# ## Using with Pretrained Weights
#
# ```python
# # Load pretrained UMA checkpoint
# neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
#     displacement_fn, box,
#     checkpoint_path='path/to/uma_sm_conserve.pt',
#     atoms=atomic_numbers,
# )
#
# nbrs = neighbor_fn.allocate(positions)
# params = init_fn(key, positions, nbrs)
#
# # NVE with pretrained model
# init_nve, apply_nve = simulate.nve(
#     lambda R, neighbor: energy_fn(params, R, neighbor),
#     shift_fn, dt=0.5,  # fs (with real units)
# )
# ```
