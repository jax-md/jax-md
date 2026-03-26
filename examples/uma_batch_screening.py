# %% [markdown]
# # Batch Screening with Pretrained UMA
#
# Screen a library of structures for their energies using batched
# inference. Demonstrates efficient throughput by packing multiple
# systems into a single JIT-compiled forward pass.

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
predict = jax.jit(model.apply)

# %% [markdown]
# ## Build a library of FCC metals

# %%
metals = {
  'Al': (13, 4.050),
  'Cu': (29, 3.615),
  'Ag': (47, 4.085),
  'Au': (79, 4.078),
  'Ni': (28, 3.524),
  'Pt': (78, 3.924),
  'Pd': (46, 3.890),
  'Rh': (45, 3.803),
}

def fcc_unit_cell(a):
  return np.array([
    [0, 0, 0], [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2],
  ], dtype=np.float32)


def build_edges(pos, cutoff, offset=0):
  n = len(pos)
  s, d = [], []
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos[i] - pos[j]) < cutoff:
        s.append(j + offset)
        d.append(i + offset)
  return s, d

# %% [markdown]
# ## Strategy 1: One-at-a-time (baseline)

# %%
print("=== Sequential (one system at a time) ===")
t0 = time.perf_counter()
results_seq = {}

for name, (Z_num, a) in metals.items():
  pos = fcc_unit_cell(a)
  Z = np.full(4, Z_num, dtype=np.int32)
  s, d = build_edges(pos, config.cutoff)
  ei = jnp.array([s, d], dtype=jnp.int32)
  ev = jnp.array(pos)[ei[0]] - jnp.array(pos)[ei[1]]

  out = predict(params, jnp.array(pos), jnp.array(Z),
                jnp.zeros(4, dtype=jnp.int32), ei, ev,
                jnp.array([0], dtype=jnp.int32),
                jnp.array([0], dtype=jnp.int32),
                dataset_names_to_indices(['omat'], config.dataset_list))
  jax.block_until_ready(out['node_embedding'])
  results_seq[name] = float(out['node_embedding'][:, 0, :].mean())

seq_time = time.perf_counter() - t0
for name, val in results_seq.items():
  print(f"  {name:2s}: l=0 mean = {val:+.6f}")
print(f"  Time: {seq_time*1000:.0f} ms total")

# %% [markdown]
# ## Strategy 2: Batched (all systems in one call)

# %%
print("\n=== Batched (all systems in one call) ===")

# Pack all systems
all_pos, all_Z, all_batch = [], [], []
all_src, all_dst = [], []
offset = 0
sys_names = []

for name, (Z_num, a) in metals.items():
  pos = fcc_unit_cell(a)
  n = len(pos)
  all_pos.append(pos)
  all_Z.extend([Z_num] * n)
  all_batch.extend([len(sys_names)] * n)
  s, d = build_edges(pos, config.cutoff, offset)
  all_src.extend(s)
  all_dst.extend(d)
  offset += n
  sys_names.append(name)

positions = jnp.array(np.concatenate(all_pos))
Z = jnp.array(all_Z, dtype=jnp.int32)
batch = jnp.array(all_batch, dtype=jnp.int32)
ei = jnp.array([all_src, all_dst], dtype=jnp.int32)
ev = positions[ei[0]] - positions[ei[1]]
charge = jnp.zeros(len(sys_names), dtype=jnp.int32)
spin = jnp.zeros(len(sys_names), dtype=jnp.int32)
ds_idx = dataset_names_to_indices(['omat'] * len(sys_names), config.dataset_list)

print(f"Packed: {len(Z)} atoms, {ei.shape[1]} edges, {len(sys_names)} systems")

# Warmup
predict(params, positions, Z, batch, ei, ev, charge, spin, ds_idx
        )['node_embedding'].block_until_ready()

t0 = time.perf_counter()
out = predict(params, positions, Z, batch, ei, ev, charge, spin, ds_idx)
jax.block_until_ready(out['node_embedding'])
batch_time = time.perf_counter() - t0

emb = out['node_embedding']
for i, name in enumerate(sys_names):
  start = i * 4
  val = float(emb[start:start+4, 0, :].mean())
  # Verify matches sequential
  diff = abs(val - results_seq[name])
  print(f"  {name:2s}: l=0 mean = {val:+.6f}  (diff from seq: {diff:.2e})")

print(f"  Time: {batch_time*1000:.1f} ms")
print(f"  Speedup: {seq_time/batch_time:.1f}x vs sequential")

# %% [markdown]
# ## Strategy 3: Batched with energy head

# %%
print("\n=== Batched energy prediction ===")

head = MLPEnergyHead(
  sphere_channels=config.sphere_channels,
  hidden_channels=config.hidden_channels,
)
key = jax.random.PRNGKey(0)
head_params = head.init(key, emb, batch, len(sys_names))

result = head.apply(head_params, emb, batch, len(sys_names))
for i, name in enumerate(sys_names):
  e = float(result['energy'][i])
  print(f"  {name:2s}: E = {e:+.6f} (per unit cell, random head weights)")

# %% [markdown]
# ## Strategy 4: Mixed-dataset batch
#
# Different systems can use different DFT levels in the same batch.

# %%
print("\n=== Mixed-dataset batch ===")

# Cu with omat (PBE), H2O with omol (wB97M), bulk Si with oc20 (RPBE)
systems = [
  ('Cu_omat', [29]*4, fcc_unit_cell(3.615), 'omat'),
  ('H2O_omol', [8,1,1], np.array([[0,0,.12],[0,.76,-.47],[0,-.76,-.47]], dtype=np.float32), 'omol'),
  ('Si_oc20', [14]*4, fcc_unit_cell(5.43) * 0.5, 'oc20'),
]

all_pos, all_Z, all_batch = [], [], []
all_src, all_dst = [], []
ds_names = []
offset = 0
for name, Z_list, pos, ds in systems:
  n = len(Z_list)
  all_pos.append(pos)
  all_Z.extend(Z_list)
  all_batch.extend([len(ds_names)] * n)
  s, d = build_edges(pos, config.cutoff, offset)
  all_src.extend(s)
  all_dst.extend(d)
  offset += n
  ds_names.append(ds)

positions = jnp.array(np.concatenate(all_pos))
Z = jnp.array(all_Z, dtype=jnp.int32)
batch = jnp.array(all_batch, dtype=jnp.int32)
ei = jnp.array([all_src, all_dst], dtype=jnp.int32)
ev = positions[ei[0]] - positions[ei[1]]
ds_idx = dataset_names_to_indices(ds_names, config.dataset_list)

out = predict(params, positions, Z, batch, ei, ev,
              jnp.zeros(3, dtype=jnp.int32),
              jnp.zeros(3, dtype=jnp.int32), ds_idx)
emb = out['node_embedding']

offset = 0
for i, (name, Z_list, _, ds) in enumerate(systems):
  n = len(Z_list)
  mean = float(emb[offset:offset+n, 0, :].mean())
  print(f"  {name:12s} ({ds:5s}): l=0 mean = {mean:+.6f}")
  offset += n

print("\nEach system uses different expert routing based on its dataset.")
