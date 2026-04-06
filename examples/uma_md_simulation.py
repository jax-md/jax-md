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
# # Molecular Dynamics with Pretrained UMA and JAX-MD
#
# This example demonstrates running NVE and NVT molecular dynamics
# using a pretrained UMA model with JAX-MD.
#
# Two APIs are shown:
# - **High-level**: `energy.uma_neighbor_list` for periodic bulk systems
# - **Low-level**: `load_pretrained` + `UMAMoEBackbone` for molecular systems
#
# The simulation is fully differentiable and JIT-compiled for performance.

# %%
import jax
import jax.numpy as jnp
from jax import jit, random
import numpy as np

from jax_md import space, energy, simulate, quantity
from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices
from jax_md._nn.uma.heads import MLPEnergyHead

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

neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
  displacement_fn, a, checkpoint_path='uma-s-1p1', atoms=atoms,
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
# ## Low-level API: molecular systems
#
# For non-periodic molecular systems (or when you need direct control over
# graph construction and dataset routing), use `load_pretrained` with
# `UMAMoEBackbone` and `MLPEnergyHead` directly.

# %%
config, moe_params, head_params = load_pretrained('uma-s-1p1')
model = UMAMoEBackbone(config=config)
head = MLPEnergyHead(
  sphere_channels=config.sphere_channels,
  hidden_channels=config.hidden_channels,
)

def build_edges(pos, cutoff):
  n = len(pos)
  s, d = [], []
  pos_np = np.asarray(pos)
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos_np[i] - pos_np[j]) < cutoff:
        s.append(j)
        d.append(i)
  return jnp.array([s, d], dtype=jnp.int32)

print("\n=== LiF neutral pair (omol) ===")
lif_pos = jnp.array([
  [0.0, 0.0, 0.0],   # Li
  [2.0, 0.0, 0.0],   # F
], dtype=jnp.float32)
lif_Z = jnp.array([3, 9], dtype=jnp.int32)
lif_batch = jnp.zeros(2, dtype=jnp.int32)
lif_ds = dataset_names_to_indices(['omol'], config.dataset_list)
lif_charge = jnp.array([0], dtype=jnp.int32)
lif_spin = jnp.array([1], dtype=jnp.int32)

lif_ei = build_edges(lif_pos, config.cutoff)

emb_lif = model.apply(moe_params, lif_pos, lif_Z, lif_batch, lif_ei,
                       lif_pos[lif_ei[0]] - lif_pos[lif_ei[1]],
                       lif_charge, lif_spin, lif_ds)
print(f"LiF embedding: {emb_lif['node_embedding'].shape}")
print(f"Li l=0: {emb_lif['node_embedding'][0, 0, :4]}")
print(f"F  l=0: {emb_lif['node_embedding'][1, 0, :4]}")

result = head.apply(head_params, emb_lif['node_embedding'], lif_batch, 1)
print(f"LiF energy: {float(result['energy'][0]):.6f} eV")

def lif_energy(pos):
  ev = pos[lif_ei[0]] - pos[lif_ei[1]]
  emb = model.apply(moe_params, pos, lif_Z, lif_batch, lif_ei, ev,
                     lif_charge, lif_spin, lif_ds)
  return head.apply(head_params, emb['node_embedding'], lif_batch, 1)['energy'][0]

forces_lif = -jax.grad(lif_energy)(lif_pos)
print(f"LiF forces:\n{forces_lif}")

