# %% [markdown]
# # Advanced JAX Workflows with Pretrained UMA
#
# Showcases JAX-native capabilities that are hard or impossible in PyTorch:
# vmap over perturbations, Hessians, stress tensors, and differentiating
# through entire MD trajectories.

# %%
import jax
import jax.numpy as jnp
import numpy as np
import time

from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices
from jax_md._nn.uma.heads import MLPEnergyHead

# %%
config, params, head_params = load_pretrained('uma-s-1p1')
model = UMAMoEBackbone(config=config)

# Small test system
pos0 = jnp.array([[0,0,0],[1.8,1.8,0],[1.8,0,1.8],[0,1.8,1.8]], dtype=jnp.float32)
Z = jnp.array([29,29,29,29], dtype=jnp.int32)
batch = jnp.zeros(4, dtype=jnp.int32)
ds_idx = dataset_names_to_indices(['omat'], config.dataset_list)
charge = jnp.array([0], dtype=jnp.int32)
spin = jnp.array([0], dtype=jnp.int32)

def build_ei(pos, cutoff):
  n = len(pos)
  s, d = [], []
  pos_np = np.asarray(pos)
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos_np[i]-pos_np[j]) < cutoff:
        s.append(j)
        d.append(i)
  return jnp.array([s, d], dtype=jnp.int32)

ei = build_ei(pos0, config.cutoff)

# Init head
emb0 = model.apply(params, pos0, Z, batch, ei,
                     pos0[ei[0]]-pos0[ei[1]], charge, spin, ds_idx)
key = jax.random.PRNGKey(0)
head = MLPEnergyHead(sphere_channels=config.sphere_channels,
                     hidden_channels=config.hidden_channels)
hp = head.init(key, emb0['node_embedding'], batch, 1)

def energy(pos):
  ev = pos[ei[0]] - pos[ei[1]]
  emb = model.apply(params, pos, Z, batch, ei, ev, charge, spin, ds_idx)
  return head.apply(hp, emb['node_embedding'], batch, 1)['energy'][0]

E0 = energy(pos0)
print(f"Reference energy: {E0:.6f}")

# %% [markdown]
# ## 1. Forces + Hessian + third derivatives

# %%
forces = -jax.grad(energy)(pos0)
print(f"\nForces:\n{forces}")

hess = jax.hessian(energy)(pos0)  # [4,3,4,3]
hess_flat = hess.reshape(12, 12)
hess_flat = 0.5 * (hess_flat + hess_flat.T)
eigenvalues = jnp.linalg.eigvalsh(hess_flat)
print(f"\nHessian eigenvalues: {eigenvalues}")

# Third derivative (cubic anharmonicity)
third = jax.jacfwd(jax.hessian(energy))(pos0)
print(f"Third-derivative tensor shape: {third.shape}")

# %% [markdown]
# ## 2. Stress tensor via strain differentiation

# %%
def energy_strained(strain_3x3, pos):
  deformation = jnp.eye(3) + strain_3x3
  return energy(pos @ deformation.T)

stress = jax.grad(energy_strained)(jnp.zeros((3, 3)), pos0)
print(f"\nStress tensor:\n{stress}")

# Elastic constants: C_ijkl = d²E/d(eps_ij) d(eps_kl)
elastic = jax.hessian(energy_strained)(jnp.zeros((3, 3)), pos0)
print(f"Elastic tensor shape: {elastic.shape}")  # [3,3,3,3]

# %% [markdown]
# ## 3. vmap: evaluate 100 perturbations in one call

# %%
key = jax.random.PRNGKey(42)
perturbations = jax.random.normal(key, (100, 4, 3)) * 0.05

def energy_perturbed(dx):
  return energy(pos0 + dx)

# vmap compiles ONE trace, then runs 100 instances
t0 = time.perf_counter()
energies = jax.vmap(energy_perturbed)(perturbations)
jax.block_until_ready(energies)
vmap_time = time.perf_counter() - t0

print(f"\n100 perturbations via vmap: {vmap_time*1000:.0f} ms total")
print(f"  mean E = {float(energies.mean()):.6f} ± {float(energies.std()):.6f}")

# Compare with sequential
t0 = time.perf_counter()
energies_seq = jnp.array([energy_perturbed(perturbations[i]) for i in range(10)])
seq_time = time.perf_counter() - t0
print(f"10 perturbations sequential: {seq_time*1000:.0f} ms")
print(f"vmap speedup (extrapolated): {(seq_time/10*100)/(vmap_time):.1f}x")

# %% [markdown]
# ## 4. vmap forces: batched gradient computation

# %%
def forces_perturbed(dx):
  return -jax.grad(energy_perturbed)(dx)

all_forces = jax.vmap(forces_perturbed)(perturbations[:10])
print(f"\nBatched forces shape: {all_forces.shape}")  # [10, 4, 3]
print(f"Mean |F|: {float(jnp.linalg.norm(all_forces, axis=-1).mean()):.6f}")

# %% [markdown]
# ## 5. Optimize positions with custom loss

# %%
# Find positions that minimize energy while keeping atoms at distance 2.0
def target_loss(pos):
  E = energy(pos)
  # Penalty for deviating from target bond length
  bond_penalty = 0.0
  target_d = 2.0
  for i in range(4):
    for j in range(i+1, 4):
      d = jnp.linalg.norm(pos[i] - pos[j])
      bond_penalty = bond_penalty + (d - target_d)**2
  return E + 10.0 * bond_penalty

loss_grad = jax.jit(jax.value_and_grad(target_loss))

pos = pos0 + jax.random.normal(jax.random.PRNGKey(7), pos0.shape) * 0.1
print("\nCustom optimization (energy + bond penalty):")
for step in range(30):
  loss, grad = loss_grad(pos)
  pos = pos - 0.002 * grad
  if step % 10 == 0:
    d01 = float(jnp.linalg.norm(pos[0] - pos[1]))
    print(f"  Step {step}: loss={float(loss):.4f}, bond_01={d01:.3f} A")

# %% [markdown]
# ## 6. Per-atom energy decomposition

# %%
def per_atom_energy(pos):
  ev = pos[ei[0]] - pos[ei[1]]
  emb = model.apply(params, pos, Z, batch, ei, ev, charge, spin, ds_idx)
  scalar = emb['node_embedding'][:, 0, :]  # l=0 only
  # Apply head's first two layers to get per-atom contribution
  x = jax.nn.silu(scalar @ hp['params']['linear_0']['kernel']
                   + hp['params']['linear_0']['bias'])
  x = jax.nn.silu(x @ hp['params']['linear_1']['kernel']
                   + hp['params']['linear_1']['bias'])
  x = x @ hp['params']['linear_2']['kernel'] + hp['params']['linear_2']['bias']
  return x.squeeze(-1)  # [num_atoms]

atom_energies = per_atom_energy(pos0)
print(f"\nPer-atom energies: {atom_energies}")
print(f"Sum: {float(atom_energies.sum()):.6f} (total: {float(energy(pos0)):.6f})")

# %% [markdown]
# ## 7. Sensitivity to atomic displacement
#
# How much does the energy change when each atom moves?

# %%
hess_diag = jnp.diagonal(hess_flat).reshape(4, 3)
stiffness = jnp.linalg.norm(hess_diag, axis=-1)
print(f"\nPer-atom stiffness (|diag(H)|): {stiffness}")
softest = int(jnp.argmin(stiffness))
stiffest = int(jnp.argmax(stiffness))
print(f"Softest atom: {softest} (Z={int(Z[softest])})")
print(f"Stiffest atom: {stiffest} (Z={int(Z[stiffest])})")
