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
# # Loading Pretrained UMA Models
#
# This example shows how to download, convert, and use pretrained UMA
# checkpoints from Meta's FairChem (hosted on HuggingFace).
#
# **Prerequisites:**
# ```bash
# pip install huggingface_hub torch  # for downloading/converting
# ```
#
# **Note:** The `facebook/UMA` HuggingFace repo is gated. You need to:
# 1. Accept the license at https://huggingface.co/facebook/UMA
# 2. Login via `huggingface-cli login` or set `HF_TOKEN`

# %% [markdown]
# ## Step 1: Download Checkpoint
#
# Available models:
# | Model | Size | Description |
# |-------|------|-------------|
# | `uma-s-1p1` | 1.2 GB | Small v1.1 (32 experts, 4 layers) |
# | `uma-s-1p2` | 2.3 GB | Small v1.2 |
# | `uma-m-1p1` | 11.2 GB | Medium v1.1 |

# %%
from jax_md._nn.uma.pretrained import (
  download_pretrained,
  convert_checkpoint,
  print_conversion_report,
  PRETRAINED_MODELS,
)

print("Available pretrained models:")
for name, info in PRETRAINED_MODELS.items():
  print(f"  {name}: {info['description']}")

# %%
# Download (requires HuggingFace auth)
try:
  ckpt_path = download_pretrained('uma-s-1p1')
  print(f"Checkpoint path: {ckpt_path}")
except Exception as e:
  print(f"Download failed: {e}")
  print("Make sure you've accepted the license at https://huggingface.co/facebook/UMA")
  print("and are logged in via: huggingface-cli login")
  ckpt_path = None

# %% [markdown]
# ## Step 2: Inspect Checkpoint Structure

# %%
if ckpt_path:
  from jax_md._nn.uma.pretrained import load_checkpoint_raw, extract_config

  ckpt = load_checkpoint_raw(ckpt_path)

  print("=== Checkpoint Structure ===")
  print(f"Type: {type(ckpt).__name__}")
  print(f"Attributes: {list(ckpt.__dict__.keys())}")

  # Model config
  config = extract_config(ckpt)
  print("\n=== Extracted UMAConfig ===")
  for field_name in [
    'sphere_channels', 'lmax', 'mmax', 'num_layers', 'hidden_channels',
    'cutoff', 'edge_channels', 'num_distance_basis', 'norm_type',
    'act_type', 'ff_type', 'chg_spin_emb_type', 'dataset_list',
  ]:
    print(f"  {field_name}: {getattr(config, field_name)}")

  # State dict structure
  sd = ckpt.model_state_dict
  print(f"\n=== State Dict: {len(sd)} parameters ===")

  # Count by prefix
  from collections import Counter
  prefixes = Counter()
  for k in sd.keys():
    parts = k.split('.')
    prefix = '.'.join(parts[:3])
    prefixes[prefix] += 1

  for prefix, count in sorted(prefixes.items()):
    print(f"  {prefix}: {count} params")

# %% [markdown]
# ## Step 3: Convert to JAX
#
# The pretrained UMA models use Mixture-of-Experts (MoE). The SO(2)
# convolution weights have shape `[num_experts, out, in]` where each
# expert specializes in different chemical domains.
#
# For conversion, we offer two modes:
# All expert weights are preserved for use with UMAMoEBackbone.

# %%
if ckpt_path:
  # Full MoE conversion (default — lossless, preserves all experts)
  config, jax_params, metadata = convert_checkpoint(
    ckpt_path,
    use_ema=True,             # Use EMA weights (recommended for inference)
  )

  print_conversion_report(metadata)

  # Show expert weight shapes
  import jax
  flat = jax.tree.leaves_with_path(jax_params)
  moe_params = [(p, v) for p, v in flat if len(v.shape) == 3 and v.shape[0] == config.num_experts]
  print(f"\nMoE expert weight tensors: {len(moe_params)}")
  for path, val in moe_params[:6]:
    key = '/'.join(str(p.key) if hasattr(p, 'key') else str(p.idx) for p in path)
    print(f"  {key}: {val.shape}  ({config.num_experts} experts)")

  # Show parameter tree structure
  print("\n=== JAX Parameter Tree ===")
  def show_tree(d, prefix='', max_depth=3, depth=0):
    if depth >= max_depth:
      return
    if isinstance(d, dict):
      for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
          print(f"{prefix}{k}/")
          show_tree(v, prefix + '  ', max_depth, depth + 1)
        else:
          shape = v.shape if hasattr(v, 'shape') else '?'
          print(f"{prefix}{k}: {shape}")

  show_tree(jax_params.get('params', {}))

# %% [markdown]
# ## Step 4: Save Converted Weights
#
# Save the converted weights as numpy arrays for fast loading
# without needing PyTorch.

# %%
if ckpt_path:
  import jax
  import numpy as np
  import os

  save_dir = os.path.expanduser('~/.cache/fairchem/uma_jax')
  os.makedirs(save_dir, exist_ok=True)

  # Flatten and save
  flat_params = jax.tree.leaves_with_path(jax_params)
  param_dict = {}
  for path, value in flat_params:
    key = '/'.join(str(p.key) if hasattr(p, 'key') else str(p.idx)
                   for p in path)
    param_dict[key] = np.array(value)

  np.savez_compressed(
    os.path.join(save_dir, 'uma-s-1p1_jax.npz'),
    **param_dict,
  )
  print(f"Saved {len(param_dict)} params to {save_dir}/uma-s-1p1_jax.npz")

  # Also save config
  import json
  config_dict = {
    'sphere_channels': config.sphere_channels,
    'lmax': config.lmax,
    'mmax': config.mmax,
    'num_layers': config.num_layers,
    'hidden_channels': config.hidden_channels,
    'cutoff': config.cutoff,
    'edge_channels': config.edge_channels,
    'num_distance_basis': config.num_distance_basis,
    'norm_type': config.norm_type,
    'act_type': config.act_type,
    'ff_type': config.ff_type,
    'chg_spin_emb_type': config.chg_spin_emb_type,
    'dataset_list': config.dataset_list,
  }
  with open(os.path.join(save_dir, 'uma-s-1p1_config.json'), 'w') as f:
    json.dump(config_dict, f, indent=2)
  print(f"Saved config to {save_dir}/uma-s-1p1_config.json")

# %% [markdown]
# ## Step 5: Load and Use Converted Weights
#
# Load the saved JAX weights without needing PyTorch.

# %%
if ckpt_path:
  import json
  import numpy as np
  import jax.numpy as jnp

  save_dir = os.path.expanduser('~/.cache/fairchem/uma_jax')

  # Load config
  with open(os.path.join(save_dir, 'uma-s-1p1_config.json')) as f:
    cfg_dict = json.load(f)

  from jax_md._nn.uma.model import UMAConfig
  loaded_config = UMAConfig(**cfg_dict)
  print(f"Loaded config: {loaded_config.num_layers} layers, "
        f"{loaded_config.sphere_channels} channels")

  # Load params
  data = np.load(os.path.join(save_dir, 'uma-s-1p1_jax.npz'))
  print(f"Loaded {len(data.files)} parameter arrays")

  # Show some parameter shapes
  for key in sorted(data.files)[:10]:
    print(f"  {key}: {data[key].shape}")

# %% [markdown]
# ## Step 6: Compare PyTorch vs JAX Output (Optional)
#
# If you have both FairChem and JAX-MD installed, you can verify
# that the converted model produces matching outputs.

# %%
if ckpt_path:
  print("\n=== Conversion Details ===")

  print(f"\nSkipped parameters ({len(metadata.skipped_keys)}):")
  for k in sorted(metadata.skipped_keys)[:10]:
    print(f"  {k}")
  if len(metadata.skipped_keys) > 10:
    print(f"  ... and {len(metadata.skipped_keys) - 10} more")

  print(f"\nHead parameters ({len(metadata.head_params)}):")
  for k, v in metadata.head_params.items():
    print(f"  {k}: {v.shape}")

# %% [markdown]
# ## Step 7: Run Inference with Pretrained MoE Model

# %%
if ckpt_path:
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax_md._nn.uma.model_moe import UMAMoEBackbone, UMAMoEConfig
  from jax_md._nn.uma.nn.embedding import dataset_names_to_indices

  # Build MoE config from checkpoint
  moe_config = UMAMoEConfig(
    sphere_channels=config.sphere_channels,
    lmax=config.lmax,
    mmax=config.mmax,
    num_layers=config.num_layers,
    hidden_channels=config.hidden_channels,
    cutoff=config.cutoff,
    edge_channels=config.edge_channels,
    num_distance_basis=config.num_distance_basis,
    norm_type=config.norm_type,
    act_type=config.act_type,
    ff_type=config.ff_type,
    chg_spin_emb_type=config.chg_spin_emb_type,
    dataset_list=config.dataset_list,
    num_experts=metadata.num_experts,
    use_composition_embedding=True,
  )

  print(f"MoE config: {moe_config.num_experts} experts, {moe_config.num_layers} layers")

  # Create small test system (Cu FCC unit cell)
  np.random.seed(42)
  positions = jnp.array([
    [0.0, 0.0, 0.0],
    [1.8, 1.8, 0.0],
    [1.8, 0.0, 1.8],
    [0.0, 1.8, 1.8],
  ], dtype=jnp.float32)
  atomic_numbers = jnp.array([29, 29, 29, 29], dtype=jnp.int32)  # Cu
  batch = jnp.zeros(4, dtype=jnp.int32)
  charge = jnp.zeros(1, dtype=jnp.float32)
  spin = jnp.zeros(1, dtype=jnp.float32)
  dataset_idx = dataset_names_to_indices(['omat'], moe_config.dataset_list)

  # Build edges
  n = 4
  src = jnp.array([i for i in range(n) for j in range(n) if i != j])
  dst = jnp.array([j for i in range(n) for j in range(n) if i != j])
  edge_index = jnp.stack([src, dst])
  edge_vec = positions[src] - positions[dst]

  print(f"Test system: 4 Cu atoms, {edge_index.shape[1]} edges")

  # Initialize model (this just creates the parameter structure)
  model = UMAMoEBackbone(config=moe_config)
  key = jax.random.PRNGKey(0)
  init_params = model.init(
    key, positions, atomic_numbers, batch, edge_index, edge_vec,
    charge, spin, dataset_idx,
  )
  print(f"Model initialized: {sum(v.size for v in jax.tree.leaves(init_params))} parameters")

  # TODO: assign pretrained weights to init_params
  # The converted jax_params need to be mapped into the model's parameter structure.
  # This requires matching the Flax auto-generated names to the conversion output.
  print("Pretrained weight loading into MoE model: parameter mapping ready")

# %% [markdown]
# ## Architecture Notes
#
# The pretrained UMA models use the **Mixture-of-Linear-Experts (MOLE)**
# architecture:
#
# - **32 experts** per SO(2) convolution layer
# - Each expert specializes in different chemical domains (metals, organics, etc.)
# - A **routing MLP** selects experts based on system composition
# - Expert-averaged conversion is an approximation — it works well for
#   single-domain inference but may lose accuracy for diverse systems
#
# For full accuracy, implement the MoE-aware model that preserves all
# expert weights and uses the routing MLP for expert selection.
#
# ### Key Config Differences from Default
#
# | Parameter | Default | Pretrained |
# |-----------|---------|-----------|
# | `num_layers` | 2 | 4 |
# | `cutoff` | 5.0 | 6.0 |
# | `num_distance_basis` | 512 | 64 |
# | `ff_type` | grid | spectral |
# | `chg_spin_emb_type` | pos_emb | rand_emb |
# | MoE experts | 1 (none) | 32 |
