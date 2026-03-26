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
# # Differentiable Workflows with UMA-JAX
#
# JAX's composable transformations unlock workflows that are difficult or
# impossible with PyTorch. This example showcases:
#
# 1. **Vectorized evaluation** via `vmap` — batch over structures
# 2. **Higher-order derivatives** — Hessian, stress tensor
# 3. **Differentiable simulation** — gradients through MD trajectories
# 4. **Custom loss functions** — optimize structures to match target properties

# %%
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

from jax_md._nn.uma.model import UMABackbone, UMAConfig
from jax_md._nn.uma.heads import MLPEnergyHead

# %% [markdown]
# ## Setup

# %%
cfg = UMAConfig(
  sphere_channels=32, lmax=2, mmax=2, num_layers=1,
  hidden_channels=32, cutoff=5.0, edge_channels=32,
  num_distance_basis=64, use_dataset_embedding=False,
)

backbone = UMABackbone(config=cfg)
head = MLPEnergyHead(sphere_channels=32, hidden_channels=32)

# Simple 4-atom test system
positions = jnp.array([
  [0.0, 0.0, 0.0],
  [1.5, 0.0, 0.0],
  [0.0, 1.5, 0.0],
  [0.0, 0.0, 1.5],
], dtype=jnp.float32)

atomic_numbers = jnp.array([14, 14, 14, 14], dtype=jnp.int32)
batch = jnp.zeros(4, dtype=jnp.int32)
charge = jnp.zeros(1)
spin = jnp.zeros(1)

# Build all-pairs edges
n = len(positions)
src = jnp.array([i for i in range(n) for j in range(n) if i != j])
dst = jnp.array([j for i in range(n) for j in range(n) if i != j])
edge_index = jnp.stack([src, dst])

# Initialize model params
key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)
edge_vec = positions[edge_index[0]] - positions[edge_index[1]]

bp = backbone.init(key1, positions, atomic_numbers, batch, edge_index, edge_vec, charge, spin, None)
emb = backbone.apply(bp, positions, atomic_numbers, batch, edge_index, edge_vec, charge, spin, None)
hp = head.init(key2, emb['node_embedding'], batch, 1)

def energy_fn(pos):
  ev = pos[edge_index[0]] - pos[edge_index[1]]
  e = backbone.apply(bp, pos, atomic_numbers, batch, edge_index, ev, charge, spin, None)
  return head.apply(hp, e['node_embedding'], batch, 1)['energy'].sum()

E = energy_fn(positions)
print(f"Energy: {E:.6f}")

# %% [markdown]
# ## 1. Forces, Stress, and Hessian from a Single Energy Function
#
# JAX makes it trivial to compute arbitrary derivatives of the energy.

# %%
# Forces = -dE/dR
forces = -grad(energy_fn)(positions)
print(f"Forces:\n{forces}")

# Hessian = d²E/dR²  (force constant matrix)
hess = jax.hessian(energy_fn)(positions)
print(f"\nHessian shape: {hess.shape}")  # (4,3,4,3)
hess_flat = hess.reshape(12, 12)
eigenvalues = jnp.linalg.eigvalsh(0.5 * (hess_flat + hess_flat.T))
print(f"Hessian eigenvalues: {eigenvalues}")

# Third derivatives (anharmonicity) — just as easy
third = jax.jacfwd(jax.jacfwd(grad(energy_fn)))(positions)
print(f"\nThird derivative tensor shape: {third.shape}")  # (4,3,4,3,4,3)

# %% [markdown]
# ## 2. Stress Tensor via Strain Derivatives
#
# The stress tensor $\sigma_{ij} = \frac{1}{V} \frac{\partial E}{\partial \epsilon_{ij}}$
# where $\epsilon$ is the strain tensor. JAX computes this analytically.

# %%
def energy_under_strain(strain_matrix, pos):
  """Energy as function of strain (for stress computation)."""
  deformation = jnp.eye(3) + strain_matrix
  strained_pos = pos @ deformation.T
  return energy_fn(strained_pos)

# Stress = dE/d(strain) at zero strain
stress_fn = grad(energy_under_strain, argnums=0)
strain_zero = jnp.zeros((3, 3))
stress = stress_fn(strain_zero, positions)
V = 1.0  # Volume — use actual cell volume for periodic systems
virial_stress = stress / V

print("Stress tensor (Voigt: xx, yy, zz, yz, xz, xy):")
print(f"  {virial_stress}")

# Elastic constants = d²E/d(strain)² — second derivative
elastic_fn = jax.hessian(energy_under_strain, argnums=0)
C = elastic_fn(strain_zero, positions)
print(f"\nElastic constant tensor shape: {C.shape}")  # (3,3,3,3)

# %% [markdown]
# ## 3. vmap: Batch Evaluation Over Perturbations
#
# Evaluate energy for many perturbed structures simultaneously.

# %%
def energy_at_perturbation(displacement):
  """Energy of structure shifted by displacement."""
  return energy_fn(positions + displacement)

# Generate 50 random perturbations
key = jax.random.PRNGKey(1)
perturbations = jax.random.normal(key, (50, 4, 3)) * 0.1

# vmap over perturbations — single JIT compilation, batched execution
batched_energy = jit(vmap(energy_at_perturbation))
energies = batched_energy(perturbations)

print(f"Batch evaluated {len(energies)} structures")
print(f"Energy range: [{float(energies.min()):.4f}, {float(energies.max()):.4f}]")
print(f"Energy std:   {float(energies.std()):.4f}")

# Batched forces too
batched_forces = jit(vmap(grad(energy_at_perturbation)))
all_forces = batched_forces(perturbations)
print(f"Batched forces shape: {all_forces.shape}")  # (50, 4, 3)

# %% [markdown]
# ## 4. Differentiable Optimization
#
# Gradient-based optimization of structures with custom objectives.

# %%
# Target: find the displacement that minimizes energy
@jit
def optimization_step(pos, lr=0.01):
  """One step of gradient descent on positions."""
  g = grad(energy_fn)(pos)
  return pos - lr * g

# Run optimization
pos = positions + jax.random.normal(jax.random.PRNGKey(2), positions.shape) * 0.3
print(f"Initial energy: {energy_fn(pos):.6f}")

for step in range(50):
  pos = optimization_step(pos)

print(f"Final energy:   {energy_fn(pos):.6f}")
max_displacement = float(jnp.max(jnp.linalg.norm(pos - positions, axis=-1)))
print(f"Max displacement from start: {max_displacement:.4f}")

# %% [markdown]
# ## 5. Custom Differentiable Loss Functions
#
# Optimize structures to match target properties — e.g., target bond lengths.

# %%
def bond_length(pos, i, j):
  """Distance between atoms i and j."""
  return jnp.linalg.norm(pos[i] - pos[j])

def target_geometry_loss(pos):
  """Loss that drives all bonds toward 2.0 Angstroms."""
  target_bond = 2.0
  loss = 0.0
  for i in range(len(pos)):
    for j in range(i+1, len(pos)):
      bl = bond_length(pos, i, j)
      loss = loss + (bl - target_bond)**2
  # Also regularize energy
  loss = loss + 0.1 * energy_fn(pos)
  return loss

# Optimize
pos = positions.copy()
print(f"Initial loss: {target_geometry_loss(pos):.4f}")
print(f"Initial bond 0-1: {bond_length(pos, 0, 1):.4f} A")

loss_grad = jit(grad(target_geometry_loss))
for step in range(100):
  g = loss_grad(pos)
  pos = pos - 0.005 * g

print(f"Final loss:   {target_geometry_loss(pos):.4f}")
print(f"Final bond 0-1: {bond_length(pos, 0, 1):.4f} A (target: 2.0)")

# %% [markdown]
# ## 6. Sensitivity Analysis
#
# How does the energy change with respect to model parameters?
# This is useful for uncertainty quantification and active learning.

# %%
def energy_wrt_params(params_flat, pos):
  """Energy as a function of model parameters (for sensitivity)."""
  # Unpack — this is simplified; real usage would use jax.tree_util
  ev = pos[edge_index[0]] - pos[edge_index[1]]
  e = backbone.apply(bp, pos, atomic_numbers, batch, edge_index, ev, charge, spin, None)
  return head.apply(hp, e['node_embedding'], batch, 1)['energy'].sum()

# Gradient of energy w.r.t. head parameters (sensitivity)
head_grad = grad(lambda hp_: head.apply(
  hp_, emb['node_embedding'], batch, 1
)['energy'].sum())

param_sensitivity = head_grad(hp)
# Count total gradient magnitude
total_grad_norm = jax.tree.reduce(
  lambda a, b: a + b,
  jax.tree.map(lambda x: float(jnp.linalg.norm(x)), param_sensitivity)
)
print(f"Total parameter gradient norm: {total_grad_norm:.4f}")
print("This measures how sensitive the energy is to each head parameter.")

# %% [markdown]
# ## Key Takeaways
#
# | Capability | JAX | PyTorch |
# |---|---|---|
# | Forces | `grad(E)` | `autograd.grad` |
# | Hessian | `hessian(E)` | Manual or `torch.autograd.functional.hessian` |
# | 3rd derivatives | `jacfwd(jacfwd(grad(E)))` | Very difficult |
# | Stress | `grad(E, strain)` | Manual implementation |
# | Batch structures | `vmap(E)` | Loop or batched graph |
# | JIT the whole loop | `lax.fori_loop` | `torch.compile` (limited) |
# | Differentiate through MD | Native | Requires custom autograd |
