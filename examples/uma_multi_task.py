# %% [markdown]
# # Multi-Task and Multi-Dataset UMA
#
# UMA is a universal model trained on 5 different DFT datasets.
# Each dataset uses a different DFT functional and basis set.
# The MoE routing automatically specializes expert mixtures
# for each dataset.
#
# This example shows how to:
# - Switch between datasets for the same structure
# - Compare embeddings across DFT levels
# - Handle charged/spin systems (omol)
# - Use dataset-specific inference

# %%
import jax
import jax.numpy as jnp
import numpy as np

from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices

# %%
config, params, head_params = load_pretrained('uma-s-1p1')
model = UMAMoEBackbone(config=config)
predict = jax.jit(model.apply)

print(f"Supported datasets: {config.dataset_list}")
print("""
  oc20:  RPBE (VASP) — catalysis surfaces
  omol:  wB97M-V/def2-TZVPD (ORCA6) — organic molecules
  omat:  PBE/PBE+U (VASP) — inorganic materials
  odac:  PBE+D3 (VASP) — direct air capture
  omc:   PBE+D3 (VASP) — organic crystals
""")

# %% [markdown]
# ## Same system, different DFT levels

# %%
def build_and_run(pos_np, Z_np, dataset, charge=0, spin=0):
  pos = jnp.array(pos_np, dtype=jnp.float32)
  Z = jnp.array(Z_np, dtype=jnp.int32)
  n = len(Z)
  batch = jnp.zeros(n, dtype=jnp.int32)
  s, d = [], []
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos_np[i] - pos_np[j]) < config.cutoff:
        s.append(j)
        d.append(i)
  ei = jnp.array([s, d], dtype=jnp.int32)
  ev = pos[ei[0]] - pos[ei[1]]
  ch = jnp.array([charge], dtype=jnp.int32)
  sp = jnp.array([spin], dtype=jnp.int32)
  ds = dataset_names_to_indices([dataset], config.dataset_list)
  return predict(params, pos, Z, batch, ei, ev, ch, sp, ds)

# Cu FCC unit cell
a = 3.615
cu_pos = np.array([[0,0,0],[a/2,a/2,0],[a/2,0,a/2],[0,a/2,a/2]], dtype=np.float32)
cu_Z = [29, 29, 29, 29]

print("=== Cu FCC across datasets ===")
print(f"{'Dataset':>6} {'l=0 mean':>12} {'l=0 std':>12} {'l=1 norm':>12}")
for ds in config.dataset_list:
  out = build_and_run(cu_pos, cu_Z, ds)
  emb = out['node_embedding']
  l0_mean = float(emb[:, 0, :].mean())
  l0_std = float(emb[:, 0, :].std())
  l1_norm = float(jnp.linalg.norm(emb[:, 1:4, :]))
  print(f"{ds:>6} {l0_mean:>12.6f} {l0_std:>12.6f} {l1_norm:>12.6f}")

# %% [markdown]
# ## Molecule: different charge states

# %%
# Water
h2o_pos = np.array([[0,0,.12],[0,.76,-.47],[0,-.76,-.47]], dtype=np.float32)
h2o_Z = [8, 1, 1]

print("\n=== H2O charge/spin variants (omol) ===")
variants = [
  ('neutral singlet', 0, 1),
  ('neutral triplet', 0, 3),
  ('cation doublet',  1, 2),
  ('anion doublet',  -1, 2),
]

for label, charge, spin in variants:
  out = build_and_run(h2o_pos, h2o_Z, 'omol', charge=charge, spin=spin)
  emb = out['node_embedding']
  l0 = float(emb[:, 0, :].mean())
  print(f"  {label:20s} (q={charge:+d}, s={spin}): l=0 mean = {l0:+.6f}")

# %% [markdown]
# ## Embedding similarity across datasets

# %%

print("\n=== Embedding similarity: Cu across datasets ===")
embeddings = {}
for ds in config.dataset_list:
  out = build_and_run(cu_pos, cu_Z, ds)
  embeddings[ds] = np.array(out['node_embedding'].reshape(-1))

print(f"{'':>8}", end='')
for ds in config.dataset_list:
  print(f"{ds:>8}", end='')
print()

for ds1 in config.dataset_list:
  print(f"{ds1:>8}", end='')
  for ds2 in config.dataset_list:
    e1 = embeddings[ds1]
    e2 = embeddings[ds2]
    cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    print(f"{cos_sim:>8.3f}", end='')
  print()

# %% [markdown]
# ## Batch: mixed datasets in one call

# %%
print("\n=== Mixed-dataset batch ===")

# Three systems, three different DFT levels
systems = [
  ('Cu_omat', cu_pos, cu_Z, 'omat', 0, 0),
  ('H2O_omol', h2o_pos, h2o_Z, 'omol', 0, 1),
  ('Cu_oc20', cu_pos, cu_Z, 'oc20', 0, 0),
]

all_pos, all_Z, all_batch = [], [], []
all_src, all_dst = [], []
ds_names = []
offset = 0

for name, pos, Z_list, ds, _, _ in systems:
  n = len(Z_list)
  all_pos.append(np.asarray(pos))
  all_Z.extend(Z_list)
  all_batch.extend([len(ds_names)] * n)
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos[i] - pos[j]) < config.cutoff:
        all_src.append(j + offset)
        all_dst.append(i + offset)
  offset += n
  ds_names.append(ds)

charges = jnp.array([c for _, _, _, _, c, _ in systems], dtype=jnp.int32)
spins = jnp.array([s for _, _, _, _, _, s in systems], dtype=jnp.int32)
positions = jnp.array(np.concatenate(all_pos))
Z = jnp.array(all_Z, dtype=jnp.int32)
batch = jnp.array(all_batch, dtype=jnp.int32)
ei = jnp.array([all_src, all_dst], dtype=jnp.int32)
ev = positions[ei[0]] - positions[ei[1]]
ds_idx = dataset_names_to_indices(ds_names, config.dataset_list)

out = predict(params, positions, Z, batch, ei, ev, charges, spins, ds_idx)
emb = out['node_embedding']

offset = 0
for name, _, Z_list, ds, _, _ in systems:
  n = len(Z_list)
  l0 = float(emb[offset:offset+n, 0, :].mean())
  print(f"  {name:12s} ({ds:5s}): l=0 mean = {l0:+.6f}")
  offset += n

print("\nCu_omat vs Cu_oc20: same atoms, different expert routing → different embeddings.")
