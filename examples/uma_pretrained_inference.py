# %% [markdown]
# # Pretrained UMA Inference
#
# Load a pretrained UMA MoE model from HuggingFace and run inference
# on real atomic systems. No expert merging — the full 32-expert
# MoE architecture runs natively in JAX with JIT compilation.
#
# ```bash
# pip install jax jaxlib flax torch huggingface_hub
# huggingface-cli login  # accept license at https://huggingface.co/facebook/UMA
# ```

# %%
import jax
import jax.numpy as jnp
import numpy as np
import time

from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices
from jax_md._nn.uma.heads import MLPEnergyHead

# %% [markdown]
# ## Load pretrained model

# %%
# Downloads uma-s-1p1 (~1.2 GB) from HuggingFace on first call
# Subsequent calls use cache at ~/.cache/fairchem/
config, params, head_params = load_pretrained('uma-s-1p1')

print(f"Model: {config.num_layers} layers, {config.sphere_channels} channels, "
      f"{config.num_experts} experts")
print(f"Cutoff: {config.cutoff} A, lmax={config.lmax}, mmax={config.mmax}")
print(f"Datasets: {config.dataset_list}")

model = UMAMoEBackbone(config=config)
predict = jax.jit(model.apply)

# %% [markdown]
# ## Helper: build edges from positions

# %%
def build_system(positions, atomic_numbers, cutoff, dataset='omat',
                 charge=0, spin=0, dataset_list=None):
  """Build all inputs needed for UMA from positions and atomic numbers."""
  n = len(positions)
  pos = np.asarray(positions, dtype=np.float32)
  Z = np.asarray(atomic_numbers, dtype=np.int32)

  src, dst = [], []
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos[i] - pos[j]) < cutoff:
        src.append(j)
        dst.append(i)
  ei = np.array([src, dst], dtype=np.int32)
  ev = (pos[ei[0]] - pos[ei[1]]).astype(np.float32)

  return {
    'positions': jnp.array(pos),
    'atomic_numbers': jnp.array(Z),
    'batch': jnp.zeros(n, dtype=jnp.int32),
    'edge_index': jnp.array(ei),
    'edge_vec': jnp.array(ev),
    'charge': jnp.array([charge], dtype=jnp.int32),
    'spin': jnp.array([spin], dtype=jnp.int32),
    'dataset_idx': dataset_names_to_indices([dataset], dataset_list),
  }


def run(model_fn, params, sys):
  """Run model on a system dict."""
  return model_fn(params, sys['positions'], sys['atomic_numbers'],
                  sys['batch'], sys['edge_index'], sys['edge_vec'],
                  sys['charge'], sys['spin'], sys['dataset_idx'])

# %% [markdown]
# ## Example 1: FCC Copper (materials science — omat)

# %%
a = 3.615  # Cu FCC lattice constant
cu_pos = np.array([
  [0, 0, 0], [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2],
], dtype=np.float32)
cu_Z = [29, 29, 29, 29]

sys_cu = build_system(cu_pos, cu_Z, config.cutoff, dataset='omat',
                      dataset_list=config.dataset_list)
out = run(predict, params, sys_cu)
print(f"Cu FCC: {out['node_embedding'].shape}, "
      f"l=0 mean = {float(out['node_embedding'][:, 0, :].mean()):.6f}")

# %% [markdown]
# ## Example 2: Water molecule (molecular chemistry — omol)

# %%
h2o_pos = np.array([
  [0.0000, 0.0000, 0.1173],   # O
  [0.0000, 0.7572, -0.4692],  # H
  [0.0000, -0.7572, -0.4692], # H
], dtype=np.float32)
h2o_Z = [8, 1, 1]

sys_h2o = build_system(h2o_pos, h2o_Z, config.cutoff, dataset='omol',
                       dataset_list=config.dataset_list)
out = run(predict, params, sys_h2o)
print(f"H2O: {out['node_embedding'].shape}")

# %% [markdown]
# ## Example 3: Batch inference — multiple systems in one call

# %%
# System A: Cu4 (omat)
# System B: H2O (omol)
# System C: CO2 (omol, charge=0, spin=1)

co2_pos = np.array([
  [0.0, 0.0, 0.0],     # C
  [0.0, 0.0, 1.16],    # O
  [0.0, 0.0, -1.16],   # O
], dtype=np.float32)
co2_Z = [6, 8, 8]

# Concatenate atoms
all_pos = jnp.array(np.concatenate([cu_pos, h2o_pos, co2_pos]))
all_Z = jnp.array(np.concatenate([cu_Z, h2o_Z, co2_Z]).astype(np.int32))
batch = jnp.array([0,0,0,0, 1,1,1, 2,2,2], dtype=jnp.int32)

# Build edges per system with offset
def edges_with_offset(pos, cutoff, offset):
  n = len(pos)
  s, d = [], []
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos[i] - pos[j]) < cutoff:
        s.append(j + offset)
        d.append(i + offset)
  return s, d

s1, d1 = edges_with_offset(cu_pos, config.cutoff, 0)
s2, d2 = edges_with_offset(h2o_pos, config.cutoff, 4)
s3, d3 = edges_with_offset(co2_pos, config.cutoff, 7)
ei = jnp.array([s1+s2+s3, d1+d2+d3], dtype=jnp.int32)
ev = all_pos[ei[0]] - all_pos[ei[1]]

charge = jnp.array([0, 0, 0], dtype=jnp.int32)
spin = jnp.array([0, 0, 1], dtype=jnp.int32)
ds_idx = dataset_names_to_indices(['omat', 'omol', 'omol'], config.dataset_list)

print(f"Batch: {len(all_Z)} atoms, {ei.shape[1]} edges, 3 systems")

out = predict(params, all_pos, all_Z, batch, ei, ev, charge, spin, ds_idx)
emb = out['node_embedding']

print(f"Output: {emb.shape}")
print(f"  Cu4 (omat):  l=0 mean = {float(emb[:4, 0, :].mean()):.6f}")
print(f"  H2O (omol):  l=0 mean = {float(emb[4:7, 0, :].mean()):.6f}")
print(f"  CO2 (omol):  l=0 mean = {float(emb[7:, 0, :].mean()):.6f}")

# %% [markdown]
# ## Example 4: Energy prediction with MLP head

# %%
head = MLPEnergyHead(
  sphere_channels=config.sphere_channels,
  hidden_channels=config.hidden_channels,
)

# head_params was loaded from the checkpoint by load_pretrained()
result = head.apply(head_params, emb, batch, 3)
print(f"\nPer-system energies: {result['energy']}")
print(f"  Cu4:  {float(result['energy'][0]):.6f}")
print(f"  H2O:  {float(result['energy'][1]):.6f}")
print(f"  CO2:  {float(result['energy'][2]):.6f}")

# %% [markdown]
# ## Example 5: Forces via autodiff

# %%
def total_energy(positions, Z, batch, ei, charge, spin, ds_idx):
  ev = positions[ei[0]] - positions[ei[1]]
  emb = model.apply(params, positions, Z, batch, ei, ev,
                     charge, spin, ds_idx)['node_embedding']
  num_systems = int(charge.shape[0])
  return head.apply(head_params, emb, batch, num_systems)['energy'].sum()

grad_fn = jax.jit(jax.grad(total_energy))
forces = -grad_fn(all_pos, all_Z, batch, ei, charge, spin, ds_idx)
print(f"\nForces: {forces.shape}")
print(f"  Cu atom 0: [{forces[0,0]:.6f}, {forces[0,1]:.6f}, {forces[0,2]:.6f}]")
print(f"  O in H2O:  [{forces[4,0]:.6f}, {forces[4,1]:.6f}, {forces[4,2]:.6f}]")
print(f"  C in CO2:  [{forces[7,0]:.6f}, {forces[7,1]:.6f}, {forces[7,2]:.6f}]")

# %% [markdown]
# ## Example 6: Speed benchmark

# %%
# Warmup
for _ in range(3):
  predict(params, all_pos, all_Z, batch, ei, ev, charge, spin, ds_idx
          )['node_embedding'].block_until_ready()

times = []
for _ in range(30):
  t0 = time.perf_counter()
  predict(params, all_pos, all_Z, batch, ei, ev, charge, spin, ds_idx
          )['node_embedding'].block_until_ready()
  times.append(time.perf_counter() - t0)

print(f"\nMoE JIT speed (10 atoms, 3 systems): "
      f"{np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms")

# %% [markdown]
# ## Example 7: Different datasets produce different embeddings
#
# The MoE routing produces different expert mixtures for each dataset,
# so the same atoms get different embeddings depending on the DFT level.

# %%
print("\nSame Cu4 system, different datasets:")
for ds in ['omat', 'oc20', 'omol']:
  sys = build_system(cu_pos, cu_Z, config.cutoff, dataset=ds,
                     dataset_list=config.dataset_list)
  out = run(predict, params, sys)
  mean = float(out['node_embedding'][:, 0, :].mean())
  std = float(out['node_embedding'][:, 0, :].std())
  print(f"  {ds:5s}: l=0 mean={mean:+.6f}, std={std:.6f}")
