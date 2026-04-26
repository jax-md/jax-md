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
# # UMA (Universal Models for Atoms) — JAX-MD
#
# End-to-end walkthrough of the UMA model: pretrained inference, dataset
# routing, batched evaluation, molecular dynamics, and structure relaxation.
#
# ```bash
# pip install jax jaxlib flax torch huggingface_hub
# huggingface-cli login  # accept license at https://huggingface.co/facebook/UMA
# ```

# %%
import jax
import jax.numpy as jnp
from jax import jit, random
import numpy as np

from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices
from jax_md._nn.uma.heads import MLPEnergyHead

# %% [markdown]
# ## 1. Load pretrained model

# %%
config, params, head_params = load_pretrained('uma-s-1p2')
model = UMAMoEBackbone(config=config)
head = MLPEnergyHead(
  sphere_channels=config.sphere_channels,
  hidden_channels=config.hidden_channels,
)
predict = jax.jit(model.apply)

print(f"Model: {config.num_layers} layers, {config.sphere_channels} channels, "
      f"{config.num_experts} experts")
print(f"Datasets: {config.dataset_list}")


def build_edges(pos_np, cutoff):
  """All-pairs edges within cutoff (for small non-periodic systems)."""
  n = len(pos_np)
  s, d = [], []
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos_np[i] - pos_np[j]) < cutoff:
        s.append(j)
        d.append(i)
  return np.array([s, d], dtype=np.int32)

# %% [markdown]
# ## 2. Single-system inference (energy + forces)

# %%
a = 3.615
cu_pos = np.array([
  [0, 0, 0], [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2],
], dtype=np.float32)
cu_Z = jnp.array([29, 29, 29, 29], dtype=jnp.int32)
cu_batch = jnp.zeros(4, dtype=jnp.int32)
cu_ds = dataset_names_to_indices(['omat'], config.dataset_list)
cu_ei = jnp.array(build_edges(cu_pos, config.cutoff))
cu_pos = jnp.array(cu_pos)
cu_ev = cu_pos[cu_ei[0]] - cu_pos[cu_ei[1]]
charge0 = jnp.array([0], dtype=jnp.int32)
spin0 = jnp.array([0], dtype=jnp.int32)

emb = predict(params, cu_pos, cu_Z, cu_batch, cu_ei, cu_ev,
              charge0, spin0, cu_ds)
result = head.apply(head_params, emb['node_embedding'], cu_batch, 1)
print(f"Cu FCC energy: {float(result['energy'][0]):.6f} eV")

def cu_energy(pos):
  ev = pos[cu_ei[0]] - pos[cu_ei[1]]
  e = model.apply(params, pos, cu_Z, cu_batch, cu_ei, ev,
                   charge0, spin0, cu_ds)
  return head.apply(head_params, e['node_embedding'], cu_batch, 1)['energy'][0]

forces = -jax.grad(cu_energy)(cu_pos)
print(f"Max |force|: {float(jnp.max(jnp.abs(forces))):.6f} eV/A")

# %% [markdown]
# ## 3. Dataset routing — same atoms, different DFT levels

# %%
print("Cu FCC across datasets:")
for ds_name in config.dataset_list:
  ds = dataset_names_to_indices([ds_name], config.dataset_list)
  out = predict(params, cu_pos, cu_Z, cu_batch, cu_ei, cu_ev,
                charge0, spin0, ds)
  l0 = float(out['node_embedding'][:, 0, :].mean())
  print(f"{ds_name:5s}: l=0 mean: {l0:+.6f}")

# %% [markdown]
# ## 4. Batched inference — multiple systems in one call

# %%
h2o_pos = np.array([[0, 0, .12], [0, .76, -.47], [0, -.76, -.47]],
                    dtype=np.float32)
h2o_Z = [8, 1, 1]

all_pos_np = np.concatenate([np.asarray(cu_pos), h2o_pos])
all_Z = jnp.array([29, 29, 29, 29, 8, 1, 1], dtype=jnp.int32)
batch = jnp.array([0, 0, 0, 0, 1, 1, 1], dtype=jnp.int32)

ei_cu = build_edges(np.asarray(cu_pos), config.cutoff)
ei_h2o = build_edges(h2o_pos, config.cutoff) + 4  # offset for H2O
ei = jnp.array(np.concatenate([ei_cu, ei_h2o], axis=1))
all_pos = jnp.array(all_pos_np)
ev = all_pos[ei[0]] - all_pos[ei[1]]
ds_idx = dataset_names_to_indices(['omat', 'omol'], config.dataset_list)

out = predict(params, all_pos, all_Z, batch, ei, ev,
              jnp.array([0, 0], dtype=jnp.int32),
              jnp.array([0, 0], dtype=jnp.int32), ds_idx)
result = head.apply(head_params, out['node_embedding'], batch, 2)
print(f"Batched energies: Cu4={float(result['energy'][0]):.4f}, "
      f"H2O={float(result['energy'][1]):.4f} eV")

# %% [markdown]
# ## 5. Charge and spin (omol task)

# %%
print("H2O charge/spin variants:")
for label, q, s in [('neutral', 0, 1), ('cation', 1, 2), ('anion', -1, 2)]:
  h2o_ei = jnp.array(build_edges(h2o_pos, config.cutoff))
  h2o_jnp = jnp.array(h2o_pos)
  h2o_ev = h2o_jnp[h2o_ei[0]] - h2o_jnp[h2o_ei[1]]
  omol = dataset_names_to_indices(['omol'], config.dataset_list)
  out = predict(params, h2o_jnp, jnp.array([8, 1, 1], dtype=jnp.int32),
                jnp.zeros(3, dtype=jnp.int32), h2o_ei, h2o_ev,
                jnp.array([q], dtype=jnp.int32),
                jnp.array([s], dtype=jnp.int32), omol)
  l0 = float(out['node_embedding'][:, 0, :].mean())
  print(f"{label:8s} (q={q:+d}, s={s}): l=0 mean: {l0:+.6f}")

# %% [markdown]
# ## 6. Molecular dynamics (NVE, periodic Si)

# %%
from jax_md import space, energy, simulate, quantity

a_si = 5.43
si_basis = jnp.array([
  [0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
  [.25, .25, .25], [.75, .75, .25], [.75, .25, .75], [.25, .75, .75],
]) * a_si
si_Z = jnp.array([14] * 8, dtype=jnp.int32)

displacement_fn, shift_fn = space.periodic(a_si)
neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
  displacement_fn, a_si, checkpoint_path='uma-s-1p2', atoms=si_Z,
)

key = random.PRNGKey(0)
nbrs = neighbor_fn.allocate(si_basis)
md_params = init_fn(key, si_basis, nbrs)

def nve_energy(R, neighbor, **kw):
  return energy_fn(md_params, R, neighbor)

init_nve, apply_nve = simulate.nve(nve_energy, shift_fn, dt=0.001)
apply_nve = jit(apply_nve)

key, subkey = random.split(key)
state = init_nve(subkey, si_basis, kT=0.1, neighbor=nbrs)

print(f"NVE on {len(si_Z)} Si atoms:")
for step in range(50):
  nbrs = nbrs.update(state.position)
  state = apply_nve(state, neighbor=nbrs)
  if step % 25 == 0:
    KE = float(quantity.kinetic_energy(state.momentum, state.mass))
    PE = float(nve_energy(state.position, nbrs))
    print(f"step {step:3d}: KE: {KE:.6f}  PE: {PE:.6f}  total: {KE+PE:.6f}")

# For NVT, use simulate.nvt_nose_hoover(nve_energy, shift_fn, dt, kT, tau)

# %% [markdown]
# ## 7. Structure relaxation (FIRE)

# %%
from jax_md import minimize
from jax_md._nn.uma.model import UMAConfig

cfg = UMAConfig(
  sphere_channels=32, lmax=2, mmax=2, num_layers=1,
  hidden_channels=32, cutoff=5.0, edge_channels=32,
  num_distance_basis=64, use_dataset_embedding=False,
)

disp_fn, shift_fn_relax = space.periodic(a_si)
nbr_fn, init_fn_relax, e_fn = energy.uma_neighbor_list(
  disp_fn, a_si, cfg=cfg, atoms=si_Z,
)

key = random.PRNGKey(42)
perturbed = si_basis + random.normal(key, si_basis.shape) * 0.1
nbrs_relax = nbr_fn.allocate(perturbed)
relax_params = init_fn_relax(key, perturbed, nbrs_relax)

def relax_energy(R, **kw):
  return e_fn(relax_params, R, nbrs_relax.update(R))

fire_init, fire_apply = minimize.fire_descent(
  relax_energy, shift_fn_relax, dt_start=0.1, dt_max=0.4,
)
fire_apply = jit(fire_apply)
fire_state = fire_init(perturbed)

print(f"FIRE relaxation ({len(si_Z)} Si atoms):")
for step in range(100):
  fire_state = fire_apply(fire_state)
  if step % 20 == 0:
    F = -jax.grad(relax_energy)(fire_state.position)
    fmax = float(jnp.max(jnp.abs(F)))
    print(f"step {step:3d}: fmax: {fmax:.6f}")
    if fmax < 0.01:
      print(f"Converged at step {step}")
      break

# %% [markdown]
# ## 8. Checkpoint conversion (requires torch)
#
# `load_pretrained` (section 1) handles this automatically, but here we
# show the individual steps: download, inspect, convert, save as numpy
# for torch-free loading.

# %%
try:
  import os
  from jax_md._nn.uma.pretrained import (
    download_pretrained, convert_checkpoint, print_conversion_report,
    load_checkpoint_raw, extract_config, PRETRAINED_MODELS,
  )

  print("Available pretrained models:")
  for name, info in PRETRAINED_MODELS.items():
    print(f"  {name}: {info['description']}")

  # Download from HuggingFace
  ckpt_path = download_pretrained('uma-s-1p2')

  # Inspect raw checkpoint
  ckpt = load_checkpoint_raw(ckpt_path)
  raw_cfg = extract_config(ckpt)
  print(f"Checkpoint config: {raw_cfg.num_layers} layers, "
        f"{raw_cfg.sphere_channels} ch, lmax={raw_cfg.lmax}")
  print(f"State dict: {len(ckpt.model_state_dict)} parameters")

  # Convert to JAX params (preserves all MoE experts)
  cfg_conv, jax_params, metadata = convert_checkpoint(ckpt_path, use_ema=True)
  print_conversion_report(metadata)

  # Save as numpy for torch-free loading
  save_dir = os.path.expanduser('~/.cache/fairchem/uma_jax')
  os.makedirs(save_dir, exist_ok=True)
  flat = jax.tree.leaves_with_path(jax_params)
  param_dict = {
    '/'.join(str(p.key) if hasattr(p, 'key') else str(p.idx) for p in path): np.array(v)
    for path, v in flat
  }
  npz_path = os.path.join(save_dir, 'uma-s-1p2_jax.npz')
  np.savez_compressed(npz_path, **param_dict)
  print(f"Saved {len(param_dict)} arrays to {npz_path}")

  # Reload without torch
  loaded = np.load(npz_path)
  print(f"Reloaded {len(loaded.files)} arrays (no torch needed)")

except ImportError:
  print("torch not installed — skipping checkpoint conversion demo")
except Exception as e:
  print(f"Checkpoint conversion skipped: {e}")
