"""
Compare JAX UMA MoE against PyTorch on 61 diverse structures,
for ALL available pretrained model configs.

Reference data dirs: tests/data/pt_ref_{model_name}/
Generate with: uv run python tests/generate_pt_reference_multi.py
"""

import os
import time
import numpy as np
from absl.testing import absltest

import jax.numpy as jnp

from jax_md import test_util

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _find_ref_dirs():
  """Find all pt_ref_* directories."""
  if not os.path.isdir(DATA_DIR):
    return {}
  return {
    d.replace('pt_ref_', ''): os.path.join(DATA_DIR, d)
    for d in os.listdir(DATA_DIR)
    if d.startswith('pt_ref_') and os.path.isdir(os.path.join(DATA_DIR, d))
  }


def _find_checkpoint(model_name):
  cache = os.path.expanduser('~/.cache/fairchem/models--facebook--UMA')
  if not os.path.isdir(cache):
    return None
  from jax_md._nn.uma.pretrained import PRETRAINED_MODELS

  if model_name not in PRETRAINED_MODELS:
    return None
  fname = PRETRAINED_MODELS[model_name]['filename']
  for root, dirs, files in os.walk(cache):
    if fname in files:
      return os.path.join(root, fname)
  return None


def _run_comparison(model_name, ref_dir):
  """Run JAX model on all structures and compare with PT reference."""
  from jax_md._nn.uma.model_moe import load_pretrained, UMAMoEBackbone
  from jax_md._nn.uma.heads import MLPEnergyHead
  from jax_md._nn.uma.nn.embedding import dataset_names_to_indices
  from ase import Atoms
  from ase.neighborlist import neighbor_list as ase_nl

  ckpt_path = _find_checkpoint(model_name)
  # Load model (head_dataset doesn't matter for backbone — we load per-structure below)
  config, params, _ = load_pretrained(ckpt_path, head_dataset='omat')
  model = UMAMoEBackbone(config=config)

  # Cache head params per dataset to handle MoE heads
  head = MLPEnergyHead(
    sphere_channels=config.sphere_channels,
    hidden_channels=config.hidden_channels,
  )
  head_params_cache = {}

  def get_head_params(task):
    if task not in head_params_cache:
      _, _, hp = load_pretrained(ckpt_path, head_dataset=task)
      head_params_cache[task] = hp
    return head_params_cache[task]

  ref_files = sorted(f for f in os.listdir(ref_dir) if f.endswith('.npz'))
  emb_diffs = []
  energy_diffs = []
  failures = []
  names_ok = []

  for fname in ref_files:
    name = fname.replace('.npz', '')
    ref = dict(np.load(os.path.join(ref_dir, fname), allow_pickle=True))

    positions = ref['positions']
    Z = ref['atomic_numbers']
    cell = ref['cell']
    pbc = ref['pbc']
    task = str(ref['task'])
    charge = int(ref['charge'])
    spin_val = int(ref['spin'])
    n = len(Z)
    pt_emb = ref['node_embedding']
    pt_energy = float(ref['total_energy'])

    try:
      atoms = Atoms(numbers=Z, positions=positions, cell=cell, pbc=pbc)
      if any(pbc):
        ci, ni, off = ase_nl(
          'ijS', atoms, cutoff=config.cutoff, self_interaction=False
        )
        ev = (positions[ni] + (off @ np.array(cell)) - positions[ci]).astype(
          np.float32
        )
        src, tgt = ni, ci
      else:
        src, tgt = [], []
        for i in range(n):
          for j in range(n):
            if (
              i != j
              and np.linalg.norm(positions[i] - positions[j]) < config.cutoff
            ):
              src.append(j)
              tgt.append(i)
        src = np.array(src)
        tgt = np.array(tgt)
        ev = (positions[src] - positions[tgt]).astype(np.float32)

      if len(src) == 0:
        failures.append((name, 'no edges'))
        continue

      ei = jnp.array(np.stack([src, tgt]).astype(np.int32))
      batch = jnp.zeros(n, dtype=jnp.int32)
      ds = dataset_names_to_indices([task], config.dataset_list)

      out = model.apply(
        params,
        jnp.array(positions),
        jnp.array(Z),
        batch,
        ei,
        jnp.array(ev),
        jnp.array([charge], dtype=jnp.int32),
        jnp.array([spin_val], dtype=jnp.int32),
        ds,
      )
      jax_emb = np.array(out['node_embedding'])
      hp = get_head_params(task)
      e_jax = float(
        head.apply(hp, out['node_embedding'], batch, 1)['energy'][0]
      )

      if not np.all(np.isfinite(jax_emb)):
        failures.append((name, 'NaN/Inf'))
        continue

      emb_diffs.append(np.max(np.abs(pt_emb - jax_emb)))
      energy_diffs.append(abs(pt_energy - e_jax))
      names_ok.append(name)

    except Exception as e:
      failures.append((name, str(e)[:80]))

  return {
    'emb_diffs': np.array(emb_diffs),
    'energy_diffs': np.array(energy_diffs),
    'names': names_ok,
    'failures': failures,
    'total': len(ref_files),
  }


class MultiModelStructureTest(test_util.JAXMDTestCase):
  """Compare JAX vs PyTorch across all pretrained models and 61 structures."""

  def setUp(self):
    super().setUp()
    self.ref_dirs = _find_ref_dirs()
    if not self.ref_dirs:
      self.skipTest(
        'No reference data found. Run generate_pt_reference_multi.py'
      )
    try:
      import torch
    except ImportError:
      self.skipTest('torch not available')

  def _run_model(self, model_name):
    if model_name not in self.ref_dirs:
      self.skipTest(f'No reference for {model_name}')
    if _find_checkpoint(model_name) is None:
      self.skipTest(f'{model_name} checkpoint not downloaded')

    ref_dir = self.ref_dirs[model_name]
    t0 = time.perf_counter()
    results = _run_comparison(model_name, ref_dir)
    elapsed = time.perf_counter() - t0

    ed = results['emb_diffs']
    ee = results['energy_diffs']

    # Assertions
    self.assertEqual(len(results['failures']), 0)
    self.assertLess(ed.max(), 0.05)
    self.assertLess(np.median(ed), 0.005)

    return results

  def test_uma_s_1p1(self):
    self._run_model('uma-s-1p1')

  def test_uma_s_1p2(self):
    self._run_model('uma-s-1p2')

  def test_uma_m_1p1(self):
    self._run_model('uma-m-1p1')


if __name__ == '__main__':
  absltest.main()
