"""Weight conversion utilities for loading PyTorch checkpoints.

This module provides utilities to convert PyTorch UMA weights to JAX format
so that pretrained models can be used in JAX-MD.

Key conventions:
- PyTorch Linear: weight shape [out, in], JAX Dense: kernel shape [in, out]
- PyTorch nn.Embedding: weight shape [num, dim], JAX nn.Embed: embedding shape [num, dim]
- SO3Linear weight: shape [lmax+1, out, in] — NOT transposed (einsum handles it)
- RadialMLP: PyTorch net.{3*i} -> JAX linear_{i+1}, net.{3*i+1} -> norm_{i+1}
- SO2MConv: PyTorch so2_m_conv.{i} -> JAX so2_m_conv_{i+1}
- Blocks: PyTorch blocks.{i} -> JAX blocks_{i}
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np


def load_pytorch_checkpoint(
  checkpoint_path: str,
  backbone_prefix: str = 'backbone.',
  map_location: str = 'cpu',
) -> Dict[str, Any]:
  """Load a PyTorch UMA checkpoint and convert to JAX format.

  Args:
      checkpoint_path: Path to the PyTorch checkpoint file (.pt).
      backbone_prefix: Prefix to strip from PyTorch keys (e.g. 'backbone.').
          Set to '' if the state_dict already has backbone-level keys.
      map_location: Device to load the checkpoint to.

  Returns:
      Dictionary of JAX-compatible parameters (nested dict under 'params').
  """
  try:
    import torch
  except ImportError:
    raise ImportError(
      'PyTorch is required to load PyTorch checkpoints. '
      'Install with: pip install torch'
    )

  checkpoint = torch.load(
    checkpoint_path, map_location=map_location, weights_only=False
  )

  # Handle different checkpoint formats
  if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
  elif 'model' in checkpoint:
    state_dict = checkpoint['model']
  else:
    state_dict = checkpoint

  # Convert to numpy
  np_state_dict = {}
  for k, v in state_dict.items():
    if hasattr(v, 'cpu'):
      np_state_dict[k] = v.cpu().numpy()
    else:
      np_state_dict[k] = np.array(v)

  return convert_pytorch_state_dict(
    np_state_dict, backbone_prefix=backbone_prefix
  )


def convert_pytorch_state_dict(
  state_dict: Dict[str, np.ndarray],
  backbone_prefix: str = 'backbone.',
  dataset_list: List[str] | None = None,
) -> Dict[str, Any]:
  """Convert PyTorch state dict to JAX/Flax params format.

  Args:
      state_dict: PyTorch state dict with numpy arrays.
      backbone_prefix: Prefix to strip from keys.
      dataset_list: Ordered dataset names for stacking per-dataset embeddings.

  Returns:
      Nested dictionary of parameters in Flax format: {'params': {...}}.
  """
  jax_params = {}
  skipped = []
  dataset_embs = {}  # Collect per-dataset embeddings

  for pt_key, pt_value in state_dict.items():
    if _is_buffer(pt_key):
      skipped.append(pt_key)
      continue

    key = pt_key
    if backbone_prefix and key.startswith(backbone_prefix):
      key = key[len(backbone_prefix) :]

    # Collect dataset_emb_dict entries for stacking
    if 'dataset_emb_dict.' in key:
      parts = key.split('.')
      for j, p in enumerate(parts):
        if p == 'dataset_emb_dict' and j + 1 < len(parts):
          dataset_embs[parts[j + 1]] = pt_value
          break
      continue

    flax_key = _convert_key(key)
    if flax_key is None:
      skipped.append(pt_key)
      continue

    flax_value = _convert_weight(key, pt_value)
    _set_nested(jax_params, flax_key, flax_value)

  # Stack dataset embeddings into single nn.Embed matrix
  if dataset_embs:
    if dataset_list is None:
      dataset_list = sorted(dataset_embs.keys())
    emb_matrix = np.concatenate(
      [dataset_embs[ds] for ds in dataset_list if ds in dataset_embs],
      axis=0,
    )
    _set_nested(
      jax_params,
      ('dataset_embedding', 'embedding', 'embedding'),
      jnp.array(emb_matrix),
    )

  return {'params': jax_params}


def _is_buffer(key: str) -> bool:
  """Check if a key corresponds to a non-trainable buffer."""
  buffer_patterns = [
    'Jd_',
    'expand_index',
    'coefficient_idx',
    'l_harmonic',
    'm_harmonic',
    'm_complex',
    'to_m',
    'balance_degree_weight',
    '.num_batches_tracked',
    '_float_tensor',
  ]
  return any(p in key for p in buffer_patterns)


def _convert_key(pt_key: str) -> Tuple[str, ...] | None:
  """Convert a PyTorch parameter key to a Flax key tuple.

  Returns None if the key should be skipped.
  """
  parts = pt_key.split('.')
  flax_parts = []
  i = 0

  while i < len(parts):
    part = parts[i]

    # Skip 'net' wrapper in RadialMLP (handled by index mapping below)
    if part == 'net':
      i += 1
      continue

    # Handle numbered indices (e.g., blocks.0, so2_m_conv.0, net.0)
    if part.isdigit():
      idx = int(part)
      if flax_parts:
        prev = flax_parts[-1]

        # RadialMLP net indices: map sequential index to our naming
        # net: [Linear, LayerNorm, SiLU, Linear, LayerNorm, SiLU, Linear, ...]
        # Param indices (skipping SiLU): 0,1, 3,4, 6,7, 9,10, ...
        if _is_in_radial_mlp_context(flax_parts):
          layer_pair_idx = idx // 3  # which (Linear, LayerNorm) pair
          is_norm = idx % 3 == 1
          if is_norm:
            flax_parts.append(f'norm_{layer_pair_idx + 1}')
          else:
            flax_parts.append(f'linear_{layer_pair_idx + 1}')
          i += 1
          continue

        # blocks.{i} -> blocks_{i}
        if prev == 'blocks':
          flax_parts[-1] = f'blocks_{idx}'
          i += 1
          continue

        # so2_m_conv.{i} -> so2_m_conv_{i+1}
        # PyTorch: 0-indexed for m=1,2,..., JAX: named by m value
        if prev == 'so2_m_conv':
          flax_parts[-1] = f'so2_m_conv_{idx + 1}'
          i += 1
          continue

        # so2_m_fc.{i} -> same pattern
        if prev == 'so2_m_fc':
          flax_parts[-1] = f'so2_m_fc_{idx + 1}'
          i += 1
          continue

        # grid_mlp.{i} (Sequential) -> Dense_{i//2} or layers_{i}
        if prev == 'grid_mlp':
          # grid_mlp: [Dense(0), SiLU(1), Dense(2), SiLU(3), Dense(4)]
          # Only even indices have params
          flax_parts.append(f'layers_{idx}')
          i += 1
          continue

        # scalar_mlp.{i} (Sequential in SpectralAtomwise) -> Dense_{i//2}
        # PT: scalar_mlp.0.weight -> Flax: Dense_0/kernel
        if prev == 'scalar_mlp':
          flax_parts.pop()  # remove 'scalar_mlp'
          dense_idx = idx // 2  # 0->Dense_0, 2->Dense_1, etc.
          flax_parts.append(f'Dense_{dense_idx}')
          i += 1
          continue

        # Generic numbered child: parent_{idx}
        flax_parts[-1] = f'{prev}_{idx}'
        i += 1
        continue

      i += 1
      continue

    # Handle terminal parameter names
    if part == 'weight':
      # Determine if this is a Linear weight or Embedding weight
      if _is_embedding_context(flax_parts):
        flax_parts.append('embedding')
      elif _is_so3_linear_context(flax_parts):
        flax_parts.append('weight')  # SO3Linear keeps 'weight' name
      elif _is_layer_norm_context(flax_parts, parts, i):
        flax_parts.append('scale')  # Flax LayerNorm uses 'scale'
      else:
        flax_parts.append('kernel')  # Flax Dense uses 'kernel'
      i += 1
      continue

    if part == 'bias':
      flax_parts.append('bias')
      i += 1
      continue

    # Dataset embedding: dataset_emb_dict.{name}.weight -> skip (handled separately)
    if part == 'dataset_emb_dict':
      return None

    # rand_emb wrapper: charge_embedding.rand_emb.weight -> charge_embedding/embedding/embedding
    if part == 'rand_emb':
      # Replace with 'embedding' (the nn.Embed module name), then 'weight' below
      # adds the final 'embedding' via _is_embedding_context
      flax_parts.append('embedding')
      i += 1
      continue

    # Default: keep the part as-is
    flax_parts.append(part)
    i += 1

  if not flax_parts:
    return None

  return tuple(flax_parts)


def _is_embedding_context(parts: List[str]) -> bool:
  """Check if we're inside an embedding layer."""
  for p in parts:
    if p in (
      'sphere_embedding',
      'source_embedding',
      'target_embedding',
      'embedding',
    ):
      return True
    # charge_embedding/spin_embedding with rand_emb -> nn.Embed
    if p in ('charge_embedding', 'spin_embedding'):
      return True
  return False


def _is_so3_linear_context(parts: List[str]) -> bool:
  """Check if we're inside an SO3Linear layer."""
  for p in parts:
    if (
      'so3_linear' in p.lower()
      or p == 'linear'
      and any('force' in x.lower() for x in parts)
    ):
      return True
  return False


def _is_layer_norm_context(
  parts: List[str], all_parts: List[str], current_idx: int
) -> bool:
  """Check if the 'weight' parameter belongs to a LayerNorm."""
  # In RadialMLP, norm_* layers are LayerNorm
  for p in parts:
    if p.startswith('norm_') or p == 'norm_l0':
      return True
  return False


def _is_in_radial_mlp_context(parts: List[str]) -> bool:
  """Check if we're inside a RadialMLP net Sequential."""
  return any(p == 'rad_func' for p in parts)


def _convert_weight(key: str, value: np.ndarray) -> jnp.ndarray:
  """Convert weight tensor from PyTorch to JAX format."""
  # Linear layer weights: PyTorch [out, in] -> JAX [in, out]
  if key.endswith('.weight') and len(value.shape) == 2:
    # Don't transpose embeddings
    if not _key_is_embedding(key):
      # Don't transpose SO3Linear (handled by einsum convention)
      if 'so3_linear' not in key:
        # Don't transpose LayerNorm weights
        parts = key.split('.')
        idx = len(parts) - 2
        if idx >= 0 and not _is_sequential_norm_index(parts):
          value = value.T

  return jnp.array(value)


def _key_is_embedding(key: str) -> bool:
  """Check if key is for an embedding layer (should NOT be transposed)."""
  embedding_parents = [
    'sphere_embedding',
    'source_embedding',
    'target_embedding',
    'dataset_emb_dict',
    'rand_emb',
    'charge_embedding',
    'spin_embedding',
  ]
  for parent in embedding_parents:
    if parent in key:
      return True
  return False


def _is_sequential_norm_index(parts: List[str]) -> bool:
  """Check if the weight is from a Sequential LayerNorm (odd index in net)."""
  for i, p in enumerate(parts):
    if p == 'net' and i + 1 < len(parts) and parts[i + 1].isdigit():
      idx = int(parts[i + 1])
      if idx % 3 == 1:  # LayerNorm index in RadialMLP
        return True
  return False


def _set_nested(d: Dict, keys: Tuple[str, ...], value: Any) -> None:
  """Set a value in a nested dictionary."""
  for key in keys[:-1]:
    if key not in d:
      d[key] = {}
    d = d[key]
  d[keys[-1]] = value


def _get_nested(d: Dict, keys: Tuple[str, ...]) -> Any:
  """Get a value from a nested dictionary. Returns None if not found."""
  for key in keys:
    if not isinstance(d, dict) or key not in d:
      return None
    d = d[key]
  return d


def extract_config_from_checkpoint(
  checkpoint_path: str,
) -> Dict[str, Any] | None:
  """Extract model configuration from a PyTorch checkpoint.

  Args:
      checkpoint_path: Path to the PyTorch checkpoint file.

  Returns:
      Configuration dictionary if available, None otherwise.
  """
  try:
    import torch
  except ImportError:
    raise ImportError(
      'PyTorch is required to load PyTorch checkpoints. '
      'Install with: pip install torch'
    )

  checkpoint = torch.load(
    checkpoint_path, map_location='cpu', weights_only=False
  )

  for config_key in ('config', 'hyper_parameters', 'model_config'):
    if config_key in checkpoint:
      return checkpoint[config_key]

  return None


def config_from_pytorch_checkpoint(checkpoint_path: str):
  """Create UMAConfig from a PyTorch checkpoint.

  Args:
      checkpoint_path: Path to the PyTorch checkpoint file.

  Returns:
      UMAConfig instance matching the checkpoint.
  """
  from jax_md._nn.uma.model import UMAConfig

  pt_config = extract_config_from_checkpoint(checkpoint_path)

  if pt_config is None:
    return UMAConfig()

  return UMAConfig(
    max_num_elements=pt_config.get('max_num_elements', 100),
    sphere_channels=pt_config.get('sphere_channels', 128),
    lmax=pt_config.get('lmax', 2),
    mmax=pt_config.get('mmax', 2),
    num_layers=pt_config.get('num_layers', 2),
    hidden_channels=pt_config.get('hidden_channels', 128),
    cutoff=pt_config.get('cutoff', 5.0),
    edge_channels=pt_config.get('edge_channels', 128),
    num_distance_basis=pt_config.get('num_distance_basis', 512),
    norm_type=pt_config.get('norm_type', 'rms_norm_sh'),
    act_type=pt_config.get('act_type', 'gate'),
    ff_type=pt_config.get('ff_type', 'grid'),
    grid_resolution=pt_config.get('grid_resolution'),
    chg_spin_emb_type=pt_config.get('chg_spin_emb_type', 'pos_emb'),
    dataset_list=pt_config.get('dataset_list'),
    use_dataset_embedding=pt_config.get('use_dataset_embedding', True),
  )


def verify_weight_loading(
  jax_params: Dict[str, Any],
  pytorch_state_dict: Dict[str, np.ndarray],
  backbone_prefix: str = 'backbone.',
  rtol: float = 1e-5,
  atol: float = 1e-5,
) -> Dict[str, str]:
  """Verify that weights were loaded correctly.

  Args:
      jax_params: Converted JAX parameters.
      pytorch_state_dict: Original PyTorch state dict.
      backbone_prefix: Prefix to strip from PyTorch keys.
      rtol: Relative tolerance for comparison.
      atol: Absolute tolerance for comparison.

  Returns:
      Dictionary mapping parameter names to status:
      'ok', 'mismatch', 'missing', or 'skipped'.
  """
  results = {}

  for pt_key, pt_value in pytorch_state_dict.items():
    if _is_buffer(pt_key):
      results[pt_key] = 'skipped'
      continue

    key = pt_key
    if backbone_prefix and key.startswith(backbone_prefix):
      key = key[len(backbone_prefix) :]

    flax_key = _convert_key(key)
    if flax_key is None:
      results[pt_key] = 'skipped'
      continue

    jax_value = _get_nested(jax_params.get('params', {}), flax_key)
    if jax_value is None:
      results[pt_key] = 'missing'
      continue

    expected = _convert_weight(key, pt_value)
    if np.allclose(
      np.array(jax_value), np.array(expected), rtol=rtol, atol=atol
    ):
      results[pt_key] = 'ok'
    else:
      results[pt_key] = 'mismatch'

  return results
