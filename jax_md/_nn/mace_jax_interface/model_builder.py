"""Shared helpers to build MACE-JAX models from serialized configs."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps

from flax import nnx

from mace_jax.adapters.nnx import resolve_gate_callable
from mace_jax.modules import interaction_classes, readout_classes
from mace_jax.modules.models import MACE, ScaleShiftMACE
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig


def normalize_cueq_config(
  config: dict[str, Any],
  cueq_config: CuEquivarianceConfig | None,
) -> CuEquivarianceConfig | None:
  if cueq_config is not None:
    return cueq_config

  raw_config = config.get('cueq_config')
  if isinstance(raw_config, CuEquivarianceConfig):
    return raw_config

  if isinstance(raw_config, dict):
    raw_config = dict(raw_config)
    if not raw_config.get('enabled', False):
      return None
    return CuEquivarianceConfig(**raw_config)

  if config.get('cue_conv_fusion'):
    return CuEquivarianceConfig(
      enabled=True,
      optimize_all=True,
      conv_fusion=True,
      layout='mul_ir',
    )

  return None


def normalize_atomic_config(
  config: dict[str, Any],
  *,
  dtype: np.dtype = np.float32,
) -> tuple[dict[str, Any], tuple[int, ...], np.ndarray]:
  atomic_numbers = tuple(int(z) for z in config.get('atomic_numbers', []))
  if not atomic_numbers:
    raise ValueError('Config is missing atomic_numbers.')
  if 'atomic_energies' not in config:
    raise ValueError('Config is missing atomic_energies.')
  atomic_energies = np.asarray(config.get('atomic_energies'), dtype=dtype)
  if atomic_energies.size == 0:
    raise ValueError('Config has empty atomic_energies.')
  num_elements = len(atomic_numbers)
  if atomic_energies.ndim == 1:
    if atomic_energies.shape[0] != num_elements:
      raise ValueError(
        'atomic_energies length does not match atomic_numbers '
        f'({atomic_energies.shape[0]} vs {num_elements}).'
      )
  else:
    last_dim = atomic_energies.shape[-1]
    if last_dim != num_elements:
      raise ValueError(
        'atomic_energies last dimension does not match atomic_numbers '
        f'({last_dim} vs {num_elements}).'
      )
  normalized = dict(config)
  normalized['atomic_numbers'] = list(atomic_numbers)
  normalized['atomic_energies'] = atomic_energies.tolist()
  return normalized, atomic_numbers, atomic_energies


def parse_parity(parity: Any) -> int:
  if parity is None:
    return 1
  if isinstance(parity, str):
    p = parity.strip().lower()
    if p in {'e', 'even'}:
      return 1
    if p in {'o', 'odd'}:
      return -1
  try:
    parity_int = int(parity)
  except (TypeError, ValueError):
    return 1
  return 1 if parity_int >= 0 else -1


def as_l_parity(rep: Any):
  """
  Best-effort extraction of (l, parity) from torch/e3nn Irrep-like objects.
  """

  try:
    module = getattr(rep, '__module__', '')
  except Exception:
    module = ''

  # Unwrap _MulIr (torch) to its contained irrep.
  if hasattr(rep, 'ir') and not hasattr(rep, 'l'):
    try:
      rep = rep.ir  # type: ignore[attr-defined]
    except Exception:
      pass

  if hasattr(rep, 'l') and hasattr(rep, 'p'):
    try:
      return int(rep.l), parse_parity(rep.p)
    except Exception:
      return None

  if module.startswith('e3nn.o3') and hasattr(rep, '__len__'):
    try:
      if len(rep) == 2 and all(isinstance(x, (int, np.integer)) for x in rep):
        return int(rep[0]), parse_parity(rep[1])
    except Exception:
      return None

  return None


def as_irrep_entry(entry: Any):
  if isinstance(entry, dict):
    mul = entry.get('mul') or entry.get('multiplicity') or entry.get('n')
    rep = entry.get('irrep') or entry.get('rep') or entry.get('l')
    parity = entry.get('p') or entry.get('parity')
    if isinstance(rep, dict):
      l_val = rep.get('l')
      parity = parity or rep.get('p') or rep.get('parity')
    else:
      l_val = rep
    if mul is None or l_val is None:
      return None
    return int(mul), (int(l_val), parse_parity(parity))

  if isinstance(entry, (list, tuple)):
    if len(entry) == 2 and isinstance(entry[0], (int, np.integer)):
      mul = int(entry[0])
      rep = entry[1]
      l_parity = as_l_parity(rep)
      if l_parity is not None:
        return mul, l_parity
      if isinstance(rep, (list, tuple)):
        if rep is None:
          return None
        l_val = int(rep[0])
        parity = parse_parity(rep[1] if len(rep) > 1 else None)
        return mul, (l_val, parity)
      if isinstance(rep, dict):
        l_val = rep.get('l')
        parity = rep.get('p') or rep.get('parity')
        if l_val is None:
          return None
        return mul, (int(l_val), parse_parity(parity))
      if isinstance(rep, (int, np.integer)):
        return mul, (int(rep), 1)
  l_parity = as_l_parity(entry)
  if l_parity is not None:
    return 1, l_parity
  return None


def normalize_irreps(value: Any):
  if isinstance(value, dict):
    value = [value]

  if isinstance(value, (list, tuple)):
    if value and as_irrep_entry(value) is not None:
      entries = [value]
    else:
      entries = value

    parsed = []
    for item in entries:
      entry = as_irrep_entry(item)
      if entry is None:
        return None
      parsed.append(entry)

    return parsed

  return None


def as_irreps(value: Any) -> Irreps:
  if isinstance(value, Irreps):
    return value
  module = getattr(value, '__module__', '')
  if isinstance(module, str) and module.startswith('e3nn.o3'):
    try:
      return Irreps(str(value))
    except Exception:
      pass
  if isinstance(value, str):
    return Irreps(value)
  if isinstance(value, int):
    return Irreps(f'{value}x0e')
  normalized = normalize_irreps(value)
  if normalized is not None:
    return Irreps(normalized)
  return Irreps(str(value))


def interaction(name_or_cls: Any):
  name = name_or_cls if isinstance(name_or_cls, str) else name_or_cls.__name__
  if name not in interaction_classes:
    raise ValueError(f'Unsupported interaction class {name!r} in config')
  return interaction_classes[name]


def readout(name_or_cls: Any):
  if name_or_cls is None:
    return readout_classes['NonLinearReadoutBlock']
  name = name_or_cls if isinstance(name_or_cls, str) else name_or_cls.__name__
  if name not in readout_classes:
    raise ValueError(f'Unsupported readout class {name!r} in config')
  return readout_classes[name]


def build_configuration(atomic_numbers: tuple[int, ...], r_max: float):
  num_atoms = len(atomic_numbers)
  spacing = max(r_max / max(num_atoms, 1), 0.5)
  positions = np.zeros((num_atoms, 3), dtype=float)
  for i in range(num_atoms):
    positions[i, 0] = spacing * i
    positions[i, 1] = spacing * (i % 2)
    positions[i, 2] = 0.0
  cell = np.eye(3) * (spacing * max(num_atoms, 1) * 2)
  return positions, cell


def pad_template_dict(
  data: dict[str, jnp.ndarray], *, n_node: int, n_edge: int, r_max: float
) -> dict[str, jnp.ndarray]:
  out = dict(data)

  def pad0(x, n):
    cur = int(x.shape[0])
    if cur > n:
      raise ValueError(f'Template has {cur} > cap {n}')
    if cur == n:
      return x
    pad = jnp.zeros((n - cur, *x.shape[1:]), dtype=x.dtype)
    return jnp.concatenate([x, pad], axis=0)

  # Nodes
  out['positions'] = pad0(out['positions'], n_node)
  out['node_attrs'] = pad0(out['node_attrs'], n_node)
  out['node_attrs_index'] = pad0(out['node_attrs_index'], n_node)
  out['batch'] = pad0(out['batch'], n_node)

  out['ptr'] = (
    out['ptr'].at[-1].set(jnp.asarray(n_node, dtype=out['ptr'].dtype))
  )

  # Edges
  E0 = int(out['edge_index'].shape[1])
  if E0 > n_edge:
    raise ValueError(f'Template has {E0} edges > cap {n_edge}')

  if E0 < n_edge:
    pad_n = n_edge - E0

    # Set padded edge_index to self-edges (0->0) to keep gathers safe
    pad_ei = jnp.zeros((2, pad_n), dtype=out['edge_index'].dtype)
    out['edge_index'] = jnp.concatenate([out['edge_index'], pad_ei], axis=1)

    # Make padded shifts far so they don't interact
    far = jnp.asarray(r_max + 1.0, dtype=out['shifts'].dtype)
    far_shift = jnp.array([far, 0.0, 0.0], dtype=out['shifts'].dtype)

    sh_pad = jnp.tile(far_shift[None, :], (pad_n, 1))
    out['shifts'] = jnp.concatenate(
      [out['shifts'], sh_pad.astype(out['shifts'].dtype)], axis=0
    )

    us_pad = jnp.zeros((pad_n, 3), dtype=out['unit_shifts'].dtype)
    out['unit_shifts'] = jnp.concatenate([out['unit_shifts'], us_pad], axis=0)

  return out


def prepare_template_data(
  config: dict[str, Any],
  *,
  n_node: int | None = None,
  n_edge: int | None = None,
) -> dict[str, jnp.ndarray]:
  atomic_numbers = tuple(int(z) for z in config['atomic_numbers'])
  positions_np, cell_np = build_configuration(atomic_numbers, config['r_max'])
  num_atoms = len(atomic_numbers)
  senders = []
  receivers = []
  for sender in range(num_atoms):
    for receiver in range(num_atoms):
      if sender != receiver:
        senders.append(sender)
        receivers.append(receiver)

  edge_index = jnp.array([senders, receivers], dtype=jnp.int32)
  edge_count = int(edge_index.shape[1])
  data = {
    'positions': jnp.asarray(positions_np, dtype=jnp.float32),
    'node_attrs': jnp.eye(num_atoms, dtype=jnp.float32),
    'node_attrs_index': jnp.arange(num_atoms, dtype=jnp.int32),
    'edge_index': edge_index,
    'shifts': jnp.zeros((edge_count, 3), dtype=jnp.float32),
    'unit_shifts': jnp.zeros((edge_count, 3), dtype=jnp.float32),
    'batch': jnp.zeros((num_atoms,), dtype=jnp.int32),
    'ptr': jnp.array([0, num_atoms], dtype=jnp.int32),
    'cell': jnp.asarray(cell_np[None, :, :], dtype=jnp.float32),
  }

  if n_node is not None or n_edge is not None:
    n_node = int(n_node or data['positions'].shape[0])
    n_edge = int(n_edge or data['edge_index'].shape[1])
    data = pad_template_dict(
      data, n_node=n_node, n_edge=n_edge, r_max=float(config['r_max'])
    )

  return data


def build_jax_model(
  config: dict[str, Any],
  *,
  cueq_config: CuEquivarianceConfig | None = None,
  rngs: nnx.Rngs | None = None,
):
  if rngs is None:
    rngs = nnx.Rngs(0)
  collapse_hidden_irreps = config.get('collapse_hidden_irreps', None)
  if collapse_hidden_irreps is None:
    try:
      num_interactions = int(config.get('num_interactions', 0))
    except Exception:
      num_interactions = 0
    if num_interactions == 1 and config.get('hidden_irreps') is not None:
      try:
        hidden_irreps = as_irreps(config['hidden_irreps'])
        collapse_hidden_irreps = len(hidden_irreps) <= 1
      except Exception:
        collapse_hidden_irreps = None

  cue_config_obj = normalize_cueq_config(config, cueq_config)

  config, atomic_numbers, atomic_energies = normalize_atomic_config(
    config,
    dtype=np.float32,
  )
  num_elements = len(atomic_numbers)

  common_kwargs = dict(
    r_max=config['r_max'],
    num_bessel=config['num_bessel'],
    num_polynomial_cutoff=config['num_polynomial_cutoff'],
    max_ell=config['max_ell'],
    interaction_cls=interaction(config['interaction_cls']),
    interaction_cls_first=interaction(config['interaction_cls_first']),
    num_interactions=config['num_interactions'],
    num_elements=num_elements,
    hidden_irreps=as_irreps(config['hidden_irreps']),
    MLP_irreps=as_irreps(config['MLP_irreps']),
    atomic_numbers=atomic_numbers,
    atomic_energies=atomic_energies,
    avg_num_neighbors=float(config['avg_num_neighbors']),
    correlation=config['correlation'],
    radial_type=config.get('radial_type', 'bessel'),
    pair_repulsion=config.get('pair_repulsion', False),
    distance_transform=config.get('distance_transform', None),
    embedding_specs=config.get('embedding_specs'),
    use_so3=config.get('use_so3', False),
    use_reduced_cg=config.get('use_reduced_cg', True),
    use_agnostic_product=config.get('use_agnostic_product', False),
    use_last_readout_only=config.get('use_last_readout_only', False),
    use_embedding_readout=config.get('use_embedding_readout', False),
    collapse_hidden_irreps=(
      True if collapse_hidden_irreps is None else bool(collapse_hidden_irreps)
    ),
    readout_cls=readout(config.get('readout_cls', None)),
    gate=resolve_gate_callable(config.get('gate', None)),
    cueq_config=cue_config_obj,
  )

  if config.get('normalize2mom_consts') is not None:
    common_kwargs['normalize2mom_consts'] = {
      str(k): float(v) for k, v in config['normalize2mom_consts'].items()
    }

  if config.get('radial_MLP') is not None:
    common_kwargs['radial_MLP'] = tuple(int(x) for x in config['radial_MLP'])

  if config.get('edge_irreps') is not None:
    common_kwargs['edge_irreps'] = as_irreps(config['edge_irreps'])

  if config.get('apply_cutoff') is not None:
    common_kwargs['apply_cutoff'] = bool(config['apply_cutoff'])

  torch_class = config.get('torch_model_class', 'MACE')
  if torch_class == 'ScaleShiftMACE' or 'atomic_inter_scale' in config:
    return ScaleShiftMACE(
      atomic_inter_scale=np.asarray(config.get('atomic_inter_scale', 1.0)),
      atomic_inter_shift=np.asarray(config.get('atomic_inter_shift', 0.0)),
      rngs=rngs,
      **common_kwargs,
    )
  return MACE(rngs=rngs, **common_kwargs)
