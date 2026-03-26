# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrained UMA model loading and conversion utilities.

Downloads and converts pretrained UMA checkpoints from HuggingFace
(facebook/UMA) for use with JAX-MD.

The pretrained UMA models use a Mixture-of-Experts (MoE) architecture
where SO(2) convolution weights have shape [num_experts, out, in].
This module provides two conversion strategies:

1. **Full MoE conversion** (default): Preserves all expert weights for
   use with UMAMoEBackbone. This is lossless.

2. **Expert averaging**: Averages expert weights for a simpler non-MoE model.
   This is lossy but works with the standard UMABackbone.

Example:
    >>> from jax_md._nn.uma.pretrained import download_pretrained, convert_checkpoint
    >>>
    >>> # Download from HuggingFace
    >>> ckpt_path = download_pretrained('uma-s-1p1')
    >>>
    >>> # Convert to JAX format (with expert averaging)
    >>> config, params, metadata = convert_checkpoint(ckpt_path)
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from jax_md._nn.uma.model import UMAConfig


# Available pretrained models on HuggingFace
PRETRAINED_MODELS = {
  'uma-s-1p1': {
    'filename': 'uma-s-1p1.pt',
    'repo_id': 'facebook/UMA',
    'subfolder': 'checkpoints',
    'description': 'UMA Small v1.1 (1.2 GB, 32 experts, 4 layers)',
  },
  'uma-s-1p2': {
    'filename': 'uma-s-1p2.pt',
    'repo_id': 'facebook/UMA',
    'subfolder': 'checkpoints',
    'description': 'UMA Small v1.2 (2.3 GB)',
  },
  'uma-m-1p1': {
    'filename': 'uma-m-1p1.pt',
    'repo_id': 'facebook/UMA',
    'subfolder': 'checkpoints',
    'description': 'UMA Medium v1.1 (11.2 GB)',
  },
}


@dataclass
class ConversionMetadata:
  """Metadata from checkpoint conversion."""
  model_name: str
  is_moe: bool
  num_experts: int
  num_layers: int
  dataset_list: List[str]
  tasks_config: Any
  head_params: Dict[str, Any]
  skipped_keys: List[str]
  converted_keys: List[str]


def download_pretrained(
  model_name: str,
  cache_dir: Optional[str] = None,
) -> str:
  """Download a pretrained UMA checkpoint from HuggingFace.

  Args:
      model_name: One of 'uma-s-1p1', 'uma-s-1p2', 'uma-m-1p1'.
      cache_dir: Cache directory (default: ~/.cache/fairchem).

  Returns:
      Local path to the downloaded checkpoint file.
  """
  try:
    from huggingface_hub import hf_hub_download
  except ImportError:
    raise ImportError(
      'huggingface_hub is required to download pretrained models. '
      'Install with: pip install huggingface_hub'
    )

  if model_name not in PRETRAINED_MODELS:
    raise ValueError(
      f'Unknown model: {model_name}. '
      f'Available: {list(PRETRAINED_MODELS.keys())}'
    )

  info = PRETRAINED_MODELS[model_name]
  if cache_dir is None:
    cache_dir = os.path.expanduser('~/.cache/fairchem')

  return hf_hub_download(
    filename=info['filename'],
    repo_id=info['repo_id'],
    subfolder=info['subfolder'],
    cache_dir=cache_dir,
  )


def load_checkpoint_raw(checkpoint_path: str) -> Any:
  """Load a FairChem UMA checkpoint without requiring fairchem installed.

  Uses a stub unpickler to handle fairchem-specific dataclasses.

  Args:
      checkpoint_path: Path to .pt checkpoint file.

  Returns:
      Checkpoint object with attributes: model_config, model_state_dict,
      ema_state_dict, tasks_config.
  """
  import torch

  class _Stub:
    def __init__(self, *args, **kwargs):
      self.__dict__.update(kwargs)

  class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
      if module.startswith('fairchem'):
        return type(name, (_Stub,), {'__module__': module})
      return super().find_class(module, name)

  class _PickleModule:
    Unpickler = _StubUnpickler
    load = pickle.load

  with open(checkpoint_path, 'rb') as f:
    return torch.load(f, map_location='cpu', weights_only=False,
                      pickle_module=_PickleModule)


def extract_config(checkpoint) -> UMAConfig:
  """Extract UMAConfig from a loaded checkpoint.

  Args:
      checkpoint: Loaded checkpoint object.

  Returns:
      UMAConfig matching the checkpoint architecture.
  """
  mc = checkpoint.model_config
  bb = mc.get('backbone', mc) if isinstance(mc, dict) else getattr(mc, 'backbone', mc)

  if isinstance(bb, dict):
    get = bb.get
  else:
    get = lambda k, d=None: getattr(bb, k, d)

  # Parse dataset_list (may be string, list, omegaconf, or missing)
  # Some models use dataset_mapping instead of dataset_list
  ds = get('dataset_list', None)
  if ds is None or (hasattr(ds, '__len__') and len(ds) == 0):
    # Try dataset_mapping from backbone or heads config
    dm = get('dataset_mapping', None)
    if dm and isinstance(dm, dict):
      ds = sorted(dm.keys())
    else:
      # Try heads config
      heads_cfg = mc.get('heads', {}) if isinstance(mc, dict) else {}
      if isinstance(heads_cfg, dict):
        for _, hcfg in heads_cfg.items():
          if isinstance(hcfg, dict):
            hdm = hcfg.get('dataset_mapping', None)
            if hdm and isinstance(hdm, dict):
              ds = sorted(hdm.keys())
              break
    if ds is None:
      ds = ['oc20', 'omol', 'omat', 'odac', 'omc']
  if isinstance(ds, str):
    import ast
    try:
      ds = ast.literal_eval(ds)
    except (ValueError, SyntaxError):
      ds = ['oc20', 'omol', 'omat', 'odac', 'omc']
  ds = list(ds)

  return UMAConfig(
    max_num_elements=get('max_num_elements', 100),
    sphere_channels=get('sphere_channels', 128),
    lmax=get('lmax', 2),
    mmax=get('mmax', 2),
    num_layers=get('num_layers', 4),
    hidden_channels=get('hidden_channels', 128),
    cutoff=get('cutoff', 6.0),
    edge_channels=get('edge_channels', 128),
    num_distance_basis=get('num_distance_basis', 64),
    norm_type=get('norm_type', 'rms_norm_sh'),
    act_type=get('act_type', 'gate'),
    ff_type=get('ff_type', 'spectral'),
    chg_spin_emb_type=get('chg_spin_emb_type', 'rand_emb'),
    dataset_list=ds,
    use_dataset_embedding=True,
  )


def convert_checkpoint(
  checkpoint_path: str,
  use_ema: bool = True,
) -> Tuple[UMAConfig, Dict[str, Any], ConversionMetadata]:
  """Convert a FairChem UMA checkpoint to JAX format.

  Preserves all MoE expert weights for use with UMAMoEBackbone.

  Args:
      checkpoint_path: Path to .pt checkpoint file.
      use_ema: Use EMA weights (recommended for inference).

  Returns:
      Tuple of (config, jax_params, metadata).
  """
  ckpt = load_checkpoint_raw(checkpoint_path)
  config = extract_config(ckpt)

  # Select state dict
  if use_ema and hasattr(ckpt, 'ema_state_dict') and ckpt.ema_state_dict:
    sd = ckpt.ema_state_dict
    # Strip 'module.' prefix from EMA keys
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
  else:
    sd = ckpt.model_state_dict

  # Detect MoE
  num_experts = 1
  for k, v in sd.items():
    if 'fc_m0.weights' in k and len(v.shape) == 3:
      num_experts = v.shape[0]
      break

  is_moe = num_experts > 1

  # Convert
  backbone_params = {}
  head_params = {}
  dataset_embs = {}  # Collect per-dataset embeddings for stacking
  skipped = []
  converted = []

  for pt_key, pt_value in sd.items():
    pt_np = pt_value.cpu().numpy() if hasattr(pt_value, 'cpu') else np.array(pt_value)

    # Skip buffers
    if _is_buffer(pt_key):
      skipped.append(pt_key)
      continue

    # Route to head params
    if pt_key.startswith('output_heads.'):
      flax_key = _convert_head_key(pt_key)
      if flax_key is not None:
        head_params[flax_key] = _convert_value(pt_key, pt_np)
        converted.append(pt_key)
      else:
        skipped.append(pt_key)
      continue

    # Strip backbone prefix
    key = pt_key
    if key.startswith('backbone.'):
      key = key[len('backbone.'):]

    # Handle MoE weights: store as 'weights' to match MOLELinear param name
    if is_moe and len(pt_np.shape) == 3 and _is_moe_weight(key):
      flax_key = _convert_backbone_key(key)
      if flax_key is not None:
        flax_key = tuple(('weights' if p == 'kernel' else p) for p in flax_key)
        _set_nested(backbone_params, flax_key, jnp.array(pt_np))
        converted.append(pt_key)
      continue

    # Handle dataset embedding dict — collect for stacking
    if 'dataset_emb_dict.' in key:
      parts = key.split('.')
      for j, p in enumerate(parts):
        if p == 'dataset_emb_dict' and j + 1 < len(parts):
          dataset_embs[parts[j + 1]] = pt_np
          break
      converted.append(pt_key)
      continue

    # Handle charge/spin embedding (rand_emb wrapping)
    if 'charge_embedding.rand_emb.' in key or 'spin_embedding.rand_emb.' in key:
      # rand_emb.weight -> charge_embedding/embedding/embedding
      clean_key = key.replace('.rand_emb.weight', '')
      flax_key = (clean_key, 'embedding', 'embedding')
      _set_nested(backbone_params, flax_key, jnp.array(pt_np))
      converted.append(pt_key)
      continue

    # Handle routing_mlp (MoE-specific)
    if 'routing_mlp.' in key:
      flax_key = _convert_routing_key(key)
      if flax_key:
        _set_nested(backbone_params, flax_key, _convert_value(key, pt_np))
        converted.append(pt_key)
      continue

    # Handle composition_embedding
    if key == 'composition_embedding.weight':
      _set_nested(backbone_params,
                  ('composition_embedding', 'embedding'),
                  jnp.array(pt_np))
      converted.append(pt_key)
      continue

    # Standard conversion
    flax_key = _convert_backbone_key(key)
    if flax_key is not None:
      val = _convert_value(key, pt_np)
      _set_nested(backbone_params, flax_key, val)
      converted.append(pt_key)
    else:
      skipped.append(pt_key)

  # Stack per-dataset embeddings into single matrix
  if dataset_embs:
    ds_order = config.dataset_list or sorted(dataset_embs.keys())
    emb_matrix = np.concatenate(
      [dataset_embs[ds] for ds in ds_order if ds in dataset_embs],
      axis=0,
    )
    _set_nested(backbone_params,
                ('dataset_embedding', 'embedding', 'embedding'),
                jnp.array(emb_matrix))

  jax_params = {'params': backbone_params}

  metadata = ConversionMetadata(
    model_name=os.path.basename(checkpoint_path),
    is_moe=is_moe,
    num_experts=num_experts,
    num_layers=config.num_layers,
    dataset_list=config.dataset_list,
    tasks_config=getattr(ckpt, 'tasks_config', None),
    head_params=head_params,
    skipped_keys=skipped,
    converted_keys=converted,
  )

  return config, jax_params, metadata


def _is_buffer(key: str) -> bool:
  patterns = [
    'Jd_', 'expand_index', 'coefficient_idx', 'l_harmonic',
    'm_harmonic', 'm_complex', 'to_m', 'balance_degree_weight',
    'num_batches_tracked', '_float_tensor', 'mole_sizes',
    'n_averaged',
  ]
  return any(p in key for p in patterns)


def _is_moe_weight(key: str) -> bool:
  """Check if a key corresponds to an MoE weight (has expert dim)."""
  return ('fc_m0.weights' in key or
          'so2_m_conv' in key and '.fc.weights' in key)


def _convert_backbone_key(key: str) -> Optional[Tuple[str, ...]]:
  """Convert a backbone state_dict key to Flax key tuple."""
  parts = key.split('.')
  flax_parts = []
  i = 0

  while i < len(parts):
    part = parts[i]

    if part == 'net':
      i += 1
      continue

    if part.isdigit():
      idx = int(part)
      if not flax_parts:
        i += 1
        continue

      prev = flax_parts[-1]

      # RadialMLP net indices
      if any(p == 'rad_func' for p in flax_parts):
        pair_idx = idx // 3
        is_norm = (idx % 3 == 1)
        flax_parts.append(f'norm_{pair_idx + 1}' if is_norm else f'linear_{pair_idx + 1}')
        i += 1
        continue

      # blocks.{i}
      if prev == 'blocks':
        flax_parts[-1] = f'blocks_{idx}'
        i += 1
        continue

      # so2_m_conv.{i}
      if prev == 'so2_m_conv':
        flax_parts[-1] = f'so2_m_conv_{idx + 1}'
        i += 1
        continue

      # scalar_mlp.{i} in SpectralAtomwise -> Dense_{i//2}
      # PT: atom_wise.scalar_mlp.0.weight -> Flax: atom_wise/Dense_0/kernel
      if prev == 'scalar_mlp':
        flax_parts.pop()  # remove 'scalar_mlp'
        dense_idx = idx // 2
        flax_parts.append(f'Dense_{dense_idx}')
        i += 1
        continue

      # Generic
      flax_parts[-1] = f'{prev}_{idx}'
      i += 1
      continue

    # Terminal parameter names
    if part == 'weight' or part == 'weights':
      if _is_embedding_key(flax_parts):
        flax_parts.append('embedding')
      elif _is_so3_key(flax_parts):
        flax_parts.append('weight')
      elif _is_norm_key(flax_parts, parts, i):
        flax_parts.append('scale')
      else:
        flax_parts.append('kernel')
      i += 1
      continue

    if part == 'bias':
      flax_parts.append('bias')
      i += 1
      continue

    flax_parts.append(part)
    i += 1

  return tuple(flax_parts) if flax_parts else None


def _convert_head_key(key: str) -> Optional[str]:
  """Convert head state_dict key to a flat string key."""
  # output_heads.energyandforcehead.head.energy_block.{i}.weight/bias
  if 'energy_block' in key:
    parts = key.split('.')
    # Find the index after energy_block
    for j, p in enumerate(parts):
      if p == 'energy_block' and j + 1 < len(parts):
        idx = parts[j + 1]
        param = parts[j + 2] if j + 2 < len(parts) else 'weight'
        flax_param = 'kernel' if param in ('weight', 'weights') else param
        return f'energy_block_{idx}_{flax_param}'
  return None


def _convert_dataset_emb_key(
  key: str, dataset_list: List[str]
) -> Optional[Tuple[str, ...]]:
  """Convert dataset_embedding.dataset_emb_dict.{name}.weight."""
  for ds_name in dataset_list:
    if f'dataset_emb_dict.{ds_name}.weight' in key:
      idx = dataset_list.index(ds_name)
      return ('dataset_embedding', 'embedding', 'embedding', f'_{idx}')
  return None


def _convert_routing_key(key: str) -> Optional[Tuple[str, ...]]:
  """Convert routing_mlp keys."""
  parts = key.split('.')
  flax_parts = ['routing_mlp']
  for i, p in enumerate(parts):
    if p == 'routing_mlp':
      continue
    if p.isdigit():
      flax_parts.append(f'layers_{p}')
    elif p == 'weight':
      flax_parts.append('kernel')
    else:
      flax_parts.append(p)
  return tuple(flax_parts)


def _is_embedding_key(parts: List[str]) -> bool:
  for p in parts:
    if p in ('sphere_embedding', 'source_embedding', 'target_embedding',
             'composition_embedding'):
      return True
  return False


def _is_so3_key(parts: List[str]) -> bool:
  return any('so3_linear' in p for p in parts)


def _is_norm_key(parts: List[str], all_parts: List[str], idx: int) -> bool:
  for p in parts:
    if p.startswith('norm_') or p == 'norm_l0':
      return True
  # Check if this is a LayerNorm in RadialMLP (odd sequential index)
  if any(p == 'rad_func' for p in parts):
    for j, p in enumerate(all_parts):
      if p == 'net' and j + 1 < len(all_parts) and all_parts[j + 1].isdigit():
        net_idx = int(all_parts[j + 1])
        if net_idx % 3 == 1:
          return True
  return False


def _convert_value(key: str, value: np.ndarray) -> jnp.ndarray:
  """Convert weight tensor from PyTorch to JAX format."""
  if (key.endswith('.weight') or key.endswith('.weights')) and len(value.shape) == 2:
    if not _key_is_embedding(key) and 'so3_linear' not in key:
      if not _key_is_norm(key):
        value = value.T
  return jnp.array(value)


def _key_is_embedding(key: str) -> bool:
  for p in ('sphere_embedding', 'source_embedding', 'target_embedding',
            'composition_embedding', 'dataset_emb_dict', 'rand_emb'):
    if p in key:
      return True
  return False


def _key_is_norm(key: str) -> bool:
  parts = key.split('.')
  for i, p in enumerate(parts):
    if p == 'net' and i + 1 < len(parts) and parts[i + 1].isdigit():
      if int(parts[i + 1]) % 3 == 1:
        return True
  return False


def _set_nested(d: Dict, keys: Tuple[str, ...], value: Any) -> None:
  for key in keys[:-1]:
    if key not in d:
      d[key] = {}
    d = d[key]
  d[keys[-1]] = value


def print_conversion_report(metadata: ConversionMetadata) -> None:
  """Print a summary of the checkpoint conversion."""
  print(f'Model: {metadata.model_name}')
  print(f'MoE: {metadata.is_moe} ({metadata.num_experts} experts)')
  print(f'Layers: {metadata.num_layers}')
  print(f'Datasets: {metadata.dataset_list}')
  print(f'Converted: {len(metadata.converted_keys)} params')
  print(f'Skipped: {len(metadata.skipped_keys)} params (buffers/MoE-specific)')
  if metadata.head_params:
    print(f'Head params: {len(metadata.head_params)} params')
