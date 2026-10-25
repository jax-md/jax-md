"""Pretrained UMA model loading and conversion utilities.

Downloads and converts pretrained UMA checkpoints from HuggingFace
repo facebook/UMA for use with JAX-MD.

The pretrained UMA models use a Mixture-of-Experts architecture
where SO(2) convolution weights have shape [num_experts, out, in].
This module provides two conversion strategies:

1. **Full MoE conversion**, the default: Preserves all expert weights for
   use with UMAMoEBackbone. This is lossless.

2. **Expert averaging**: Averages expert weights for a simpler non-MoE model.
   This is lossy but works with the standard UMABackbone.

Example:
    >>> from jax_md._nn.uma.pretrained import download_pretrained, convert_checkpoint
    >>>
    >>> # Download from HuggingFace
    >>> ckpt_path = download_pretrained('uma-s-1p1')
    >>>
    >>> # Convert to JAX format with expert averaging
    >>> config, params, metadata = convert_checkpoint(ckpt_path)
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from jax_md._nn.uma.model import UMAConfig

T = TypeVar('T')


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
  dataset_list: List[str] | None
  tasks_config: Any
  head_params: Dict[str, Any]
  skipped_keys: List[str]
  converted_keys: List[str]


def download_pretrained(
  model_name: str,
  cache_dir: str | None = None,
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


def load_atom_references(
  model_name: str,
  dataset: str = 'omat',
  cache_dir: str | None = None,
) -> jnp.ndarray:
  """Load UMA isolated atom references from FairChem's external ref file."""
  try:
    from huggingface_hub import hf_hub_download
  except ImportError:
    raise ImportError(
      'huggingface_hub is required to download UMA atom references. '
      'Install with: pip install huggingface_hub'
    )
  try:
    import yaml
  except ImportError:
    raise ImportError(
      'pyyaml is required to parse UMA atom references. '
      'Install with: pip install pyyaml'
    )

  if model_name not in PRETRAINED_MODELS:
    raise ValueError(
      'UMA atom references can be loaded automatically only for known '
      f'pretrained model names, got {model_name!r}. Pass atom_refs explicitly '
      'when using a local checkpoint path.'
    )

  if cache_dir is None:
    cache_dir = os.path.expanduser('~/.cache/fairchem')

  path = hf_hub_download(
    filename='iso_atom_elem_refs.yaml',
    repo_id=PRETRAINED_MODELS[model_name]['repo_id'],
    subfolder='references',
    cache_dir=cache_dir,
  )

  with open(path) as f:
    refs = yaml.safe_load(f)

  key = f'{dataset}_elem_refs'
  if key not in refs:
    raise ValueError(
      f'Missing {key!r} in UMA atom references. Available keys: '
      f'{sorted(k for k in refs if k.endswith("_elem_refs"))}'
    )
  values = refs[key]
  if not isinstance(values, (list, tuple)):
    raise ValueError(
      f'UMA atom references for {dataset!r} are not a flat element array. '
      'Pass atom_refs explicitly for this dataset.'
    )
  return jnp.asarray(values, dtype=jnp.float32)


def config_get(config: Any, key: str, default: Any = None) -> Any:
  if config is None:
    return default
  if isinstance(config, dict):
    return config.get(key, default)
  get = getattr(config, 'get', None)
  if callable(get):
    try:
      return get(key, default)
    except TypeError:
      pass
  if hasattr(config, key):
    return getattr(config, key)
  try:
    return config[key]
  except (KeyError, IndexError, TypeError):
    return default


def config_list(value: Any) -> list[Any]:
  if value is None:
    return []
  if isinstance(value, str):
    return [value]
  try:
    return list(value)
  except TypeError:
    return [value]


def config_float(value: Any, default: float) -> float:
  if value is None:
    return default
  if hasattr(value, 'detach'):
    value = value.detach().cpu().numpy()
  array = np.asarray(value)
  if array.size == 0:
    return default
  return float(array.reshape(-1)[0])


def extract_element_references(element_references: Any) -> jnp.ndarray | None:
  if element_references is None:
    return None

  values = config_get(element_references, 'element_references')
  args = config_get(values, '_args_')
  if args is not None and len(config_list(args)) > 0:
    values = config_list(args)[0]

  if hasattr(values, 'detach'):
    values = values.detach().cpu().numpy()
  if values is None:
    return None

  return jnp.asarray(values, dtype=jnp.float32)


def extract_energy_correction(
  tasks_config: Any,
  dataset: str = 'omat',
) -> Dict[str, Any]:
  """Extract FairChem energy normalizer and element references from tasks."""
  energy_task = None
  expected_name = f'{dataset}_energy'
  for task in tasks_config:
    name = config_get(task, 'name')
    prop = config_get(task, 'property')
    datasets = config_list(config_get(task, 'datasets'))
    if name == expected_name or (prop == 'energy' and dataset in datasets):
      energy_task = task
      break

  if energy_task is None:
    raise ValueError(f'No energy task found for dataset {dataset!r}.')

  normalizer = config_get(energy_task, 'normalizer')
  mean = config_float(config_get(normalizer, 'mean'), 0.0)
  rmsd = config_float(config_get(normalizer, 'rmsd'), 1.0)
  element_refs = extract_element_references(
    config_get(energy_task, 'element_references')
  )

  return {
    'mean': jnp.asarray(mean, dtype=jnp.float32),
    'rmsd': jnp.asarray(rmsd, dtype=jnp.float32),
    'element_refs': element_refs,
  }


def load_energy_correction(
  model_name_or_path: str,
  dataset: str = 'omat',
  cache_dir: str | None = None,
) -> Dict[str, Any]:
  """Load FairChem task-level energy correction from a UMA checkpoint."""
  checkpoint_path = model_name_or_path
  if model_name_or_path in PRETRAINED_MODELS:
    checkpoint_path = download_pretrained(model_name_or_path, cache_dir)

  ckpt = load_checkpoint_raw(checkpoint_path)
  return extract_energy_correction(ckpt.tasks_config, dataset)


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

  class Stub:
    def __init__(self, *args, **kwargs):
      self.__dict__.update(kwargs)

  class StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
      if module.startswith('fairchem'):
        return type(name, (Stub,), {'__module__': module})
      return super().find_class(module, name)

  class PickleModule:
    Unpickler = StubUnpickler
    load = pickle.load

  with open(checkpoint_path, 'rb') as f:
    return torch.load(
      f, map_location='cpu', weights_only=False, pickle_module=PickleModule
    )


def extract_config(checkpoint: Any) -> UMAConfig:
  """Extract UMAConfig from a loaded checkpoint.

  Args:
      checkpoint: Loaded checkpoint object.

  Returns:
      UMAConfig matching the checkpoint architecture.
  """
  mc = checkpoint.model_config
  bb = (
    mc.get('backbone', mc)
    if isinstance(mc, dict)
    else getattr(mc, 'backbone', mc)
  )

  def get(k: str, d: Any = None) -> Any:
    if isinstance(bb, dict):
      return bb.get(k, d)
    return getattr(bb, k, d)

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
      use_ema: Use EMA weights, recommended for inference.

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
    pt_np = (
      pt_value.cpu().numpy() if hasattr(pt_value, 'cpu') else np.array(pt_value)
    )

    # Skip buffers
    if is_buffer(pt_key):
      skipped.append(pt_key)
      continue

    # Route to head params
    if pt_key.startswith('output_heads.'):
      param_key = convert_head_key(pt_key)
      if param_key is not None:
        head_params[param_key] = convert_value(pt_key, pt_np)
        converted.append(pt_key)
      else:
        skipped.append(pt_key)
      continue

    # Strip backbone prefix
    key = pt_key
    if key.startswith('backbone.'):
      key = key[len('backbone.') :]

    # Handle MoE weights: store as 'weights' to match MOLELinear param name
    if is_moe and len(pt_np.shape) == 3 and is_moe_weight(key):
      param_key = convert_backbone_key(key)
      if param_key is not None:
        param_key = tuple(
          ('weights' if p == 'kernel' else p) for p in param_key
        )
        set_nested(backbone_params, param_key, jnp.array(pt_np))
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
      param_key = (clean_key, 'embedding', 'embedding')
      set_nested(backbone_params, param_key, jnp.array(pt_np))
      converted.append(pt_key)
      continue

    # Handle routing_mlp (MoE-specific)
    if 'routing_mlp.' in key:
      param_key = convert_routing_key(key)
      if param_key:
        set_nested(backbone_params, param_key, convert_value(key, pt_np))
        converted.append(pt_key)
      continue

    # Handle composition_embedding
    if key == 'composition_embedding.weight':
      set_nested(
        backbone_params,
        ('composition_embedding', 'embedding'),
        jnp.array(pt_np),
      )
      converted.append(pt_key)
      continue

    # Standard conversion
    param_key = convert_backbone_key(key)
    if param_key is not None:
      val = convert_value(key, pt_np)
      set_nested(backbone_params, param_key, val)
      converted.append(pt_key)
    else:
      skipped.append(pt_key)

  # Stack per-dataset embeddings into single matrix
  if dataset_embs:
    ds_order = config.dataset_list or sorted(dataset_embs.keys())
    missing = [ds for ds in ds_order if ds not in dataset_embs]
    if missing:
      raise ValueError(f'Checkpoint has no dataset embedding for {missing}.')
    emb_matrix = np.concatenate([dataset_embs[ds] for ds in ds_order], axis=0)
    set_nested(
      backbone_params,
      ('dataset_embedding', 'embedding', 'embedding'),
      jnp.array(emb_matrix),
    )

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


def is_buffer(key: str) -> bool:
  patterns = [
    'Jd_',
    'expand_index',
    'coefficient_idx',
    'l_harmonic',
    'm_harmonic',
    'm_complex',
    'to_m',
    'balance_degree_weight',
    'num_batches_tracked',
    '_float_tensor',
    'mole_sizes',
    'n_averaged',
  ]
  return any(p in key for p in patterns)


def is_moe_weight(key: str) -> bool:
  """Check if a key corresponds to an MoE weight (has expert dim)."""
  return 'fc_m0.weights' in key or 'so2_m_conv' in key and '.fc.weights' in key


def convert_backbone_key(key: str) -> Tuple[str, ...] | None:
  """Convert a backbone state_dict key to a nested key tuple."""
  parts = key.split('.')
  segments = []
  i = 0

  while i < len(parts):
    part = parts[i]

    if part == 'net':
      i += 1
      continue

    if part.isdigit():
      idx = int(part)
      if not segments:
        i += 1
        continue

      prev = segments[-1]

      # RadialMLP net indices
      if any(p == 'rad_func' for p in segments):
        pair_idx = idx // 3
        is_norm = idx % 3 == 1
        segments.append(
          f'norm_{pair_idx + 1}' if is_norm else f'linear_{pair_idx + 1}'
        )
        i += 1
        continue

      # blocks.{i}
      if prev == 'blocks':
        segments[-1] = f'blocks_{idx}'
        i += 1
        continue

      # so2_m_conv.{i}
      if prev == 'so2_m_conv':
        segments[-1] = f'so2_m_conv_{idx + 1}'
        i += 1
        continue

      # scalar_mlp.{i} and grid_mlp.{i} map to Dense_{i//2}
      if prev in ('scalar_mlp', 'grid_mlp'):
        segments.pop()
        dense_idx = idx // 2
        segments.append(f'Dense_{dense_idx}')
        i += 1
        continue

      # Generic
      segments[-1] = f'{prev}_{idx}'
      i += 1
      continue

    # Terminal parameter names
    if part == 'weight' or part == 'weights':
      if is_embedding_key(segments):
        segments.append('embedding')
      elif is_so3_key(segments):
        segments.append('weight')
      elif is_norm_key(segments, parts, i):
        segments.append('scale')
      else:
        segments.append('kernel')
      i += 1
      continue

    if part == 'bias':
      segments.append('bias')
      i += 1
      continue

    segments.append(part)
    i += 1

  return tuple(segments) if segments else None


def convert_head_key(key: str) -> str | None:
  """Convert head state_dict key to a flat string key."""
  # output_heads.energyandforcehead.head.energy_block.{i}.weight/bias
  if 'energy_block' in key:
    parts = key.split('.')
    # Find the index after energy_block
    for j, p in enumerate(parts):
      if p == 'energy_block' and j + 1 < len(parts):
        idx = parts[j + 1]
        param = parts[j + 2] if j + 2 < len(parts) else 'weight'
        param_name = 'kernel' if param in ('weight', 'weights') else param
        return f'energy_block_{idx}_{param_name}'
  return None


def convert_routing_key(key: str) -> Tuple[str, ...] | None:
  """Convert routing_mlp keys."""
  parts = key.split('.')
  segments = ['routing_mlp']
  for i, p in enumerate(parts):
    if p == 'routing_mlp':
      continue
    if p.isdigit():
      segments.append(f'layers_{p}')
    elif p == 'weight':
      segments.append('kernel')
    else:
      segments.append(p)
  return tuple(segments)


def is_embedding_key(parts: List[str]) -> bool:
  for p in parts:
    if p in (
      'sphere_embedding',
      'source_embedding',
      'target_embedding',
      'composition_embedding',
    ):
      return True
  return False


def is_so3_key(parts: List[str]) -> bool:
  return any('so3_linear' in p for p in parts)


def is_norm_key(parts: List[str], all_parts: List[str], idx: int) -> bool:
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


def convert_value(key: str, value: np.ndarray) -> jnp.ndarray:
  """Convert weight tensor from PyTorch to JAX format."""
  if (key.endswith('.weight') or key.endswith('.weights')) and len(
    value.shape
  ) == 2:
    if not key_is_embedding(key) and 'so3_linear' not in key:
      if not key_is_norm(key):
        value = value.T
  return jnp.array(value)


def key_is_embedding(key: str) -> bool:
  for p in (
    'sphere_embedding',
    'source_embedding',
    'target_embedding',
    'composition_embedding',
    'dataset_emb_dict',
    'rand_emb',
  ):
    if p in key:
      return True
  return False


def key_is_norm(key: str) -> bool:
  parts = key.split('.')
  for i, p in enumerate(parts):
    if p == 'net' and i + 1 < len(parts) and parts[i + 1].isdigit():
      if int(parts[i + 1]) % 3 == 1:
        return True
  return False


def set_nested(d: Dict, keys: Tuple[str, ...], value: Any) -> None:
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


def load_params_into_module(params: Any, module: T, strict: bool = False) -> T:
  """Return *module* with its arrays taken from a linen-shaped params dict.

  Leaves are matched by path: a tuple field contributes ``name_i``, a dict
  field contributes its key, and everything else its attribute name.
  Module leaves the checkpoint does not name keep their computed values,
  unless ``strict`` is set, which rejects any entry on either side that has
  no counterpart. A full checkpoint loaded into the module it was converted
  from matches exactly, so strict mode catches a renamed parameter path.

  A matched entry must have the shape of the module leaf and keeps the
  checkpoint dtype. The module may come from ``eqx.filter_eval_shape``, in
  which case strict mode guarantees that every leaf is filled.
  """
  import jax

  if 'params' in params and len(params) == 1:
    params = params['params']

  flat = {}
  for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
    flat['/'.join(str(getattr(e, 'key', e)) for e in path)] = value

  leaves, treedef = jax.tree_util.tree_flatten_with_path(module)
  out = []
  keys = []
  for path, current in leaves:
    segments = []
    for entry in path:
      if isinstance(entry, jax.tree_util.SequenceKey):
        segments[-1] = f'{segments[-1]}_{entry.idx}'
      elif isinstance(entry, jax.tree_util.DictKey):
        segments[-1] = str(entry.key)
      else:
        segments.append(entry.name)
    keys.append('/'.join(segments))
    if keys[-1] not in flat:
      out.append(current)
      continue
    value = flat[keys[-1]]
    if not hasattr(current, 'shape'):
      out.append(value)
      continue
    if tuple(value.shape) != tuple(current.shape):
      raise ValueError(
        f'Shape mismatch at {keys[-1]}: checkpoint {tuple(value.shape)}, '
        f'module {tuple(current.shape)}.'
      )
    out.append(jnp.asarray(value))

  if strict:
    unused = sorted(set(flat) - set(keys))
    unmatched = [key for key in keys if key not in flat]
    if unused or unmatched:
      raise ValueError(
        'Checkpoint and module parameters have different paths.\n'
        f'Unused checkpoint entries: {unused[:5]}\n'
        f'Module leaves not in the checkpoint: {unmatched[:5]}'
      )
  return jax.tree_util.tree_unflatten(treedef, out)


def module_to_params(module: eqx.Module) -> Dict[str, Any]:
  """Return the linen-shaped nested params dict for an equinox *module*.

  The inverse of :func:`load_params_into_module`.
  """
  import jax

  tree = {}
  for path, value in jax.tree_util.tree_flatten_with_path(module)[0]:
    segments = []
    for entry in path:
      if isinstance(entry, jax.tree_util.SequenceKey):
        segments[-1] = f'{segments[-1]}_{entry.idx}'
      elif isinstance(entry, jax.tree_util.DictKey):
        segments[-1] = str(entry.key)
      else:
        segments.append(entry.name)
    cursor = tree
    for seg in segments[:-1]:
      cursor = cursor.setdefault(seg, {})
    cursor[segments[-1]] = value
  return tree
