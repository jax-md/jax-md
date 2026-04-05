#!/usr/bin/env python
"""
Generate PT reference data for multiple pretrained model configs.
Reuses the same 61 structures from generate_pt_reference.py.

Usage:
    uv run python tests/generate_pt_reference_multi.py
"""

import os
import sys
import time
import numpy as np
import torch

# Import structure builder from the existing script
sys.path.insert(0, os.path.dirname(__file__))
from test_structures import make_structures
from generate_pt_reference import atoms_to_data


def generate_for_model(model_name, ckpt_path):
  from jax_md._nn.uma.pretrained import load_checkpoint_raw

  ckpt = load_checkpoint_raw(ckpt_path)
  mc = ckpt.model_config['backbone']
  ema_sd = {
    k.replace('module.', '', 1): v for k, v in ckpt.ema_state_dict.items()
  }
  get = mc.get if isinstance(mc, dict) else lambda k, d=None: getattr(mc, k, d)
  ds_list = list(get('dataset_list', ['oc20', 'omol', 'omat', 'odac', 'omc']))
  cutoff = float(get('cutoff', 6.0))

  for k in list(sys.modules.keys()):
    if k.startswith('fairchem') and not hasattr(sys.modules[k], '__file__'):
      del sys.modules[k]
  from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

  pt_model = eSCNMDMoeBackbone(
    num_experts=get('num_experts', 32),
    sphere_channels=get('sphere_channels', 128),
    max_num_elements=get('max_num_elements', 100),
    lmax=get('lmax', 2),
    mmax=get('mmax', 2),
    num_layers=get('num_layers', 4),
    hidden_channels=get('hidden_channels', 128),
    cutoff=cutoff,
    edge_channels=get('edge_channels', 128),
    num_distance_basis=get('num_distance_basis', 64),
    norm_type=get('norm_type', 'rms_norm_sh'),
    act_type=get('act_type', 'gate'),
    ff_type=get('ff_type', 'spectral'),
    chg_spin_emb_type=get('chg_spin_emb_type', 'rand_emb'),
    dataset_list=ds_list,
    use_composition_embedding=get('use_composition_embedding', True),
    model_version=get('model_version', 1.1),
    moe_dropout=0.0,
    otf_graph=False,
  )
  pt_model.eval()
  bb_sd = {
    k.replace('backbone.', ''): v
    for k, v in ema_sd.items()
    if k.startswith('backbone.')
  }
  pt_model.load_state_dict(bb_sd, strict=False)

  # Head weights — handle both standard and MoE heads
  prefix = 'output_heads.energyandforcehead.head.energy_block.'

  def silu(x):
    return x / (1 + np.exp(-np.clip(x, -88, 88)))

  # Check if head uses MoE weights (3D) or standard weights (2D)
  key0 = (
    prefix + '0.weight'
    if prefix + '0.weight' in ema_sd
    else prefix + '0.weights'
  )
  head_is_moe = ema_sd[key0].dim() == 3
  if head_is_moe:
    # MoE head: weights shape [num_dataset_experts, out, in]
    # We'll select the correct expert per-dataset during evaluation
    head_w0 = ema_sd[prefix + '0.weights'].numpy()
    head_b0 = ema_sd[prefix + '0.bias'].numpy()
    head_w2 = ema_sd[prefix + '2.weights'].numpy()
    head_b2 = ema_sd[prefix + '2.bias'].numpy()
    head_w4 = ema_sd[prefix + '4.weights'].numpy()
    head_b4 = ema_sd[prefix + '4.bias'].numpy()
    # Dataset names for head expert selection
    # DatasetSpecificMoEWrapper uses sorted(dataset_names) from head config
    # which may differ from the backbone dataset_list
    heads_cfg = ckpt.model_config.get('heads', {})
    head_ds_names = None
    if isinstance(heads_cfg, dict):
      for _, hcfg in heads_cfg.items():
        if isinstance(hcfg, dict):
          dm = hcfg.get('dataset_mapping', None)
          if dm and isinstance(dm, dict):
            head_ds_names = sorted(dm.keys())
          dn = hcfg.get('dataset_names', None)
          if dn and not head_ds_names:
            head_ds_names = sorted(
              list(dn) if not isinstance(dn, str) else eval(dn)
            )
    if head_ds_names is None:
      head_ds_names = sorted(ds_list)
    head_ds_to_idx = {n: i for i, n in enumerate(head_ds_names)}
    print(f'  Head experts: {head_ds_names}')
  else:
    w0 = ema_sd[prefix + '0.weight'].numpy()
    b0 = ema_sd[prefix + '0.bias'].numpy()
    w2 = ema_sd[prefix + '2.weight'].numpy()
    b2 = ema_sd[prefix + '2.bias'].numpy()
    w4 = ema_sd[prefix + '4.weight'].numpy()
    b4 = ema_sd[prefix + '4.bias'].numpy()

  out_dir = os.path.join(
    os.path.dirname(__file__), 'data', f'pt_ref_{model_name}'
  )
  os.makedirs(out_dir, exist_ok=True)

  structures = make_structures()
  print(f'\n=== {model_name}: {len(structures)} structures ===')
  t0 = time.perf_counter()
  count = 0

  for name, (atoms, task) in sorted(structures.items()):
    data = atoms_to_data(atoms, task, cutoff, ds_list)
    try:
      with torch.no_grad():
        output = pt_model(data)
      node_emb = output['node_embedding'].numpy()
      scalar = node_emb[:, 0, :]
      if head_is_moe:
        ds_idx = head_ds_to_idx.get(task, 0)
        x = silu(scalar @ head_w0[ds_idx].T + head_b0)
        x = silu(x @ head_w2[ds_idx].T + head_b2)
        x = x @ head_w4[ds_idx].T + head_b4
      else:
        x = silu(scalar @ w0.T + b0)
        x = silu(x @ w2.T + b2)
        x = x @ w4.T + b4
      total_energy = x.sum()

      np.savez_compressed(
        os.path.join(out_dir, f'{name}.npz'),
        positions=atoms.get_positions().astype(np.float32),
        atomic_numbers=atoms.get_atomic_numbers().astype(np.int32),
        cell=np.array(atoms.get_cell()).astype(np.float32),
        pbc=np.array(atoms.pbc),
        task=task,
        charge=int(atoms.info.get('charge', 0)),
        spin=int(atoms.info.get('spin', 0)),
        node_embedding=node_emb,
        total_energy=total_energy,
      )
      count += 1
    except Exception as e:
      print(f'  SKIP {name}: {e}')

  elapsed = time.perf_counter() - t0
  print(f'  {count} structures in {elapsed:.1f}s -> {out_dir}/')
  return count


def main():
  cache = os.path.expanduser('~/.cache/fairchem/models--facebook--UMA')
  models = {}
  for root, dirs, files in os.walk(cache):
    for f in files:
      if f.endswith('.pt') and f.startswith('uma-'):
        name = f.replace('.pt', '')
        models[name] = os.path.join(root, f)

  # Filter to known good checkpoints
  from jax_md._nn.uma.pretrained import PRETRAINED_MODELS

  valid = {n: p for n, p in models.items() if n in PRETRAINED_MODELS}
  print(f'Found {len(valid)} valid checkpoints: {list(valid.keys())}')
  for name, path in sorted(valid.items()):
    generate_for_model(name, path)


if __name__ == '__main__':
  main()
