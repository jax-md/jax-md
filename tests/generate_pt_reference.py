#!/usr/bin/env python
"""
Generate PyTorch reference data for relaxation comparison tests.

Run this ONCE to produce reference energies, forces, and relaxed structures
using FairChem's PyTorch UMA model. The output is saved as .npz files
that the JAX comparison tests load without needing PyTorch.

Usage:
    uv run python tests/generate_pt_reference.py

Output:
    tests/data/pt_reference_{name}.npz for each test structure
"""

import os
import sys
import numpy as np
import torch

# Structures
from ase.build import bulk, molecule
from ase.neighborlist import neighbor_list as ase_nl


def make_structures():
  """Return dict of {name: (atoms, task)}."""
  structures = {}

  # 1. Cu FCC (4 atoms, periodic, omat)
  cu = bulk('Cu', 'fcc', a=3.615, cubic=True)
  rng = np.random.default_rng(42)
  cu.positions += rng.normal(scale=0.03, size=cu.positions.shape)
  structures['cu_fcc'] = (cu, 'omat')

  # 2. Fe BCC 2x2x2 (16 atoms, periodic, omat)
  fe = bulk('Fe', 'bcc', a=2.87).repeat((2, 2, 2))
  rng = np.random.default_rng(42)
  fe.positions += rng.normal(scale=0.02, size=fe.positions.shape)
  structures['fe_bcc'] = (fe, 'omat')

  # 3. H2O molecule (3 atoms, non-periodic, omol)
  h2o = molecule('H2O')
  h2o.center(vacuum=10.0)
  h2o.pbc = False
  h2o.info['charge'] = 0
  h2o.info['spin'] = 1
  rng = np.random.default_rng(42)
  h2o.positions += rng.normal(scale=0.05, size=h2o.positions.shape)
  structures['h2o'] = (h2o, 'omol')

  # 4. Si diamond (8 atoms, periodic, omat)
  si = bulk('Si', 'diamond', a=5.43)
  rng = np.random.default_rng(42)
  si.positions += rng.normal(scale=0.03, size=si.positions.shape)
  structures['si_diamond'] = (si, 'omat')

  # 5. NaCl rocksalt (8 atoms, periodic, omat)
  from ase import Atoms
  a_nacl = 5.64
  nacl = Atoms(
    'NaClNaClNaClNaCl',
    scaled_positions=[
      (0, 0, 0), (0.5, 0.5, 0.5),
      (0.5, 0, 0), (0, 0.5, 0.5),
      (0, 0.5, 0), (0.5, 0, 0.5),
      (0, 0, 0.5), (0.5, 0.5, 0),
    ],
    cell=[a_nacl, a_nacl, a_nacl],
    pbc=True,
  )
  rng = np.random.default_rng(42)
  nacl.positions += rng.normal(scale=0.02, size=nacl.positions.shape)
  structures['nacl'] = (nacl, 'omat')

  # 6. CO2 molecule (3 atoms, non-periodic, omol)
  co2 = Atoms('CO2', positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
  co2.center(vacuum=10.0)
  co2.pbc = False
  co2.info['charge'] = 0
  co2.info['spin'] = 1
  rng = np.random.default_rng(42)
  co2.positions += rng.normal(scale=0.03, size=co2.positions.shape)
  structures['co2'] = (co2, 'omol')

  return structures


def atoms_to_data(atoms, task, cutoff, ds_list):
  """Convert ASE atoms to PT model input dict."""
  positions = atoms.get_positions().astype(np.float32)
  Z = atoms.get_atomic_numbers().astype(np.int64)
  n = len(atoms)

  cell_offsets_np = None
  if any(atoms.pbc):
    # ASE convention: center_idx, neighbor_idx, offsets (cell offset vectors)
    center_idx, neighbor_idx, offsets = ase_nl('ijS', atoms, cutoff=cutoff, self_interaction=False)
    cell = np.array(atoms.get_cell())
    # FairChem edge_index: [source=neighbor, target=center]
    idx_i = neighbor_idx  # source
    idx_j = center_idx    # target
    # Store the cell_offsets for the PT model (it computes edge_vec internally)
    # PT model convention: cell_offsets are for the SOURCE atom relative to TARGET
    # shifts = cell_offsets @ cell gives displacement from target to source image
    cell_offsets_np = offsets.astype(np.float32)
  else:
    idx_i, idx_j = [], []
    for i in range(n):
      for j in range(n):
        if i != j and np.linalg.norm(positions[i] - positions[j]) < cutoff:
          idx_i.append(j)  # source = neighbor
          idx_j.append(i)  # target = center
    idx_i = np.array(idx_i, dtype=np.int64)
    idx_j = np.array(idx_j, dtype=np.int64)

  ei = np.stack([idx_i, idx_j]).astype(np.int64)
  n_e = ei.shape[1]
  if cell_offsets_np is None:
    cell_offsets_np = np.zeros((n_e, 3), dtype=np.float32)
  charge = int(atoms.info.get('charge', 0))
  spin_val = int(atoms.info.get('spin', 0))

  class DB(dict):
    def __getattr__(self, k):
      try:
        return self[k]
      except KeyError:
        raise AttributeError(k)
    def __setattr__(self, k, v):
      self[k] = v
    def get(self, k, *a, **kw):
      return super().get(k, kw.get('default', a[0] if a else None))

  data = DB(
    pos=torch.tensor(positions),
    atomic_numbers=torch.tensor(Z),
    atomic_numbers_full=torch.tensor(Z),
    batch=torch.zeros(n, dtype=torch.long),
    batch_full=torch.zeros(n, dtype=torch.long),
    edge_index=torch.tensor(ei),
    charge=torch.tensor([charge], dtype=torch.long),
    spin=torch.tensor([spin_val], dtype=torch.long),
    dataset=[task], natoms=torch.tensor([n]),
    cell=torch.tensor(np.array(atoms.get_cell()), dtype=torch.float32).unsqueeze(0) if any(atoms.pbc) else torch.eye(3).unsqueeze(0),
    cell_offsets=torch.tensor(cell_offsets_np),
    nedges=torch.tensor([n_e]),
  )
  return data


def main():
  # Load model
  from jax_md._nn.uma.pretrained import load_checkpoint_raw

  cache = os.path.expanduser('~/.cache/fairchem/models--facebook--UMA')
  ckpt_path = None
  for root, dirs, files in os.walk(cache):
    if 'uma-s-1p1.pt' in files:
      ckpt_path = os.path.join(root, 'uma-s-1p1.pt')
  if ckpt_path is None:
    print('ERROR: uma-s-1p1.pt not found. Run download first.')
    sys.exit(1)

  ckpt = load_checkpoint_raw(ckpt_path)
  mc = ckpt.model_config['backbone']
  ema_sd = {k.replace('module.', '', 1): v for k, v in ckpt.ema_state_dict.items()}
  get = mc.get if isinstance(mc, dict) else lambda k, d=None: getattr(mc, k, d)
  ds_list = list(get('dataset_list', ['oc20', 'omol', 'omat', 'odac', 'omc']))
  cutoff = float(get('cutoff', 6.0))

  for k in list(sys.modules.keys()):
    if k.startswith('fairchem') and not hasattr(sys.modules[k], '__file__'):
      del sys.modules[k]
  from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

  pt_model = eSCNMDMoeBackbone(
    num_experts=get('num_experts', 32), sphere_channels=get('sphere_channels', 128),
    max_num_elements=get('max_num_elements', 100), lmax=get('lmax', 2), mmax=get('mmax', 2),
    num_layers=get('num_layers', 4), hidden_channels=get('hidden_channels', 128),
    cutoff=cutoff, edge_channels=get('edge_channels', 128),
    num_distance_basis=get('num_distance_basis', 64), norm_type=get('norm_type', 'rms_norm_sh'),
    act_type=get('act_type', 'gate'), ff_type=get('ff_type', 'spectral'),
    chg_spin_emb_type=get('chg_spin_emb_type', 'rand_emb'), dataset_list=ds_list,
    use_composition_embedding=get('use_composition_embedding', True),
    model_version=get('model_version', 1.1), moe_dropout=0.0, otf_graph=False,
  )
  pt_model.eval()
  bb_sd = {k.replace('backbone.', ''): v for k, v in ema_sd.items() if k.startswith('backbone.')}
  pt_model.load_state_dict(bb_sd, strict=False)

  # Head weights
  head_sd = {k: v for k, v in ema_sd.items() if 'output_heads' in k}

  # Output dir
  out_dir = os.path.join(os.path.dirname(__file__), 'data', 'pt_reference')
  os.makedirs(out_dir, exist_ok=True)

  structures = make_structures()
  print(f'Generating PT reference for {len(structures)} structures...')

  for name, (atoms, task) in structures.items():
    print(f'\n--- {name} ({task}, {len(atoms)} atoms) ---')
    data = atoms_to_data(atoms, task, cutoff, ds_list)

    with torch.no_grad():
      output = pt_model(data)

    node_emb = output['node_embedding'].numpy()

    # Also compute energy using the head
    scalar = node_emb[:, 0, :]  # l=0
    # Manual MLP: energy_block.0 -> silu -> .2 -> silu -> .4
    prefix = 'output_heads.energyandforcehead.head.energy_block.'
    w0 = ema_sd[prefix + '0.weight'].numpy()
    b0 = ema_sd[prefix + '0.bias'].numpy()
    w2 = ema_sd[prefix + '2.weight'].numpy()
    b2 = ema_sd[prefix + '2.bias'].numpy()
    w4 = ema_sd[prefix + '4.weight'].numpy()
    b4 = ema_sd[prefix + '4.bias'].numpy()

    def silu(x): return x / (1 + np.exp(-x))
    x = silu(scalar @ w0.T + b0)
    x = silu(x @ w2.T + b2)
    x = x @ w4.T + b4
    atom_energies = x.squeeze(-1)
    total_energy = atom_energies.sum()

    print(f'  Embedding: {node_emb.shape}, mean={node_emb.mean():.6f}')
    print(f'  Energy: {total_energy:.6f} eV')
    print(f'  Per-atom: {atom_energies[:3]}...')

    # Save
    np.savez(
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
      atom_energies=atom_energies,
    )
    print(f'  Saved to {out_dir}/{name}.npz')

  print(f'\nDone! Reference data in {out_dir}/')


if __name__ == '__main__':
  main()
