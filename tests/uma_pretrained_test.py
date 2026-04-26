"""
End-to-end tests for pretrained UMA MoE model.

Tests that pretrained checkpoints load and run correctly in JAX with
full MoE, and benchmarks inference speed.

Checkpoints are downloaded automatically via huggingface_hub when
HF_TOKEN is set.  The CI workflow caches ~/.cache/fairchem across runs.

Requires:
  - torch (for checkpoint loading)
  - huggingface_hub (for downloading gated weights)
  - jax, flax
"""

import time

import numpy as np
from absl.testing import absltest

import jax
import jax.numpy as jnp

from jax_md import test_util
from jax_md._nn.uma.model_moe import UMAMoEBackbone, load_pretrained
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices

MODEL_NAME = 'uma-s-1p2'


def _make_test_system(num_atoms=6, cutoff=6.0):
  positions = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.8, 1.8, 0.0],
      [1.8, 0.0, 1.8],
      [0.0, 1.8, 1.8],
      [0.9, 0.9, 0.9],
      [2.7, 0.9, 0.9],
    ],
    dtype=np.float32,
  )[:num_atoms]
  atomic_numbers = np.array([29, 29, 29, 29, 8, 8], dtype=np.int32)[:num_atoms]
  batch = np.zeros(num_atoms, dtype=np.int32)
  src, dst = [], []
  for i in range(num_atoms):
    for j in range(num_atoms):
      if i != j and np.linalg.norm(positions[i] - positions[j]) < cutoff:
        src.append(j)
        dst.append(i)
  edge_index = np.array([src, dst], dtype=np.int32)
  return positions, atomic_numbers, batch, edge_index


class LoadPretrainedTest(test_util.JAXMDTestCase):
  """Test loading pretrained UMA models as full MoE."""

  def setUp(self):
    super().setUp()
    try:
      import torch  # noqa: F401
    except ImportError:
      self.skipTest('torch not available')
    try:
      self._loaded = load_pretrained(MODEL_NAME)
    except Exception as e:
      self.skipTest(f'Could not load {MODEL_NAME}: {e}')

  def _run_moe(self, cutoff=6.0, dataset='omat'):
    config, params, _hp = self._loaded
    model = UMAMoEBackbone(config=config)

    pos, Z, batch, ei = _make_test_system(cutoff=cutoff)
    ev = (pos[ei[0]] - pos[ei[1]]).astype(np.float32)
    ds_idx = dataset_names_to_indices([dataset], config.dataset_list)

    output = model.apply(
      params,
      jnp.array(pos),
      jnp.array(Z),
      jnp.array(batch),
      jnp.array(ei),
      jnp.array(ev),
      jnp.array([0], dtype=jnp.int32),
      jnp.array([0], dtype=jnp.int32),
      ds_idx,
    )
    return output, model, params, config

  def test_loads_and_runs(self):
    output, _, _, config = self._run_moe()
    emb = output['node_embedding']
    self.assertEqual(emb.shape, (6, 9, 128))
    self.assertTrue(jnp.all(jnp.isfinite(emb)))
    self.assertEqual(config.num_experts, 64)
    self.assertEqual(config.num_layers, 4)

  def test_omol_task(self):
    output, _, _, _ = self._run_moe(dataset='omol')
    self.assertTrue(jnp.all(jnp.isfinite(output['node_embedding'])))

  def test_jit_compiles(self):
    output, model, params, config = self._run_moe()
    eager_emb = np.array(output['node_embedding'])

    pos, Z, batch, ei = _make_test_system()
    ev = (pos[ei[0]] - pos[ei[1]]).astype(np.float32)
    ds_idx = dataset_names_to_indices(['omat'], config.dataset_list)

    jit_fn = jax.jit(model.apply)
    jit_out = jit_fn(
      params,
      jnp.array(pos),
      jnp.array(Z),
      jnp.array(batch),
      jnp.array(ei),
      jnp.array(ev),
      jnp.array([0], dtype=jnp.int32),
      jnp.array([0], dtype=jnp.int32),
      ds_idx,
    )
    jax.block_until_ready(jit_out['node_embedding'])
    jit_emb = np.array(jit_out['node_embedding'])

    # JIT should produce same result as eager
    np.testing.assert_allclose(jit_emb, eager_emb, atol=1e-4)

  def test_deterministic(self):
    out1, model, params, config = self._run_moe()
    pos, Z, batch, ei = _make_test_system()
    ev = (pos[ei[0]] - pos[ei[1]]).astype(np.float32)
    ds_idx = dataset_names_to_indices(['omat'], config.dataset_list)

    out2 = model.apply(
      params,
      jnp.array(pos),
      jnp.array(Z),
      jnp.array(batch),
      jnp.array(ei),
      jnp.array(ev),
      jnp.array([0], dtype=jnp.int32),
      jnp.array([0], dtype=jnp.int32),
      ds_idx,
    )
    np.testing.assert_array_equal(
      np.array(out1['node_embedding']),
      np.array(out2['node_embedding']),
    )


class MoEBenchmark(test_util.JAXMDTestCase):
  """Benchmark pretrained MoE inference speed."""

  def test_benchmark(self):
    try:
      import torch  # noqa: F401
    except ImportError:
      self.skipTest('torch not available')
    try:
      config, params, _hp = load_pretrained(MODEL_NAME)
    except Exception as e:
      self.skipTest(f'Could not load {MODEL_NAME}: {e}')

    model = UMAMoEBackbone(config=config)

    pos, Z, batch, ei = _make_test_system()
    ev = (pos[ei[0]] - pos[ei[1]]).astype(np.float32)
    ds_idx = dataset_names_to_indices(['omat'], config.dataset_list)
    args = (
      params,
      jnp.array(pos),
      jnp.array(Z),
      jnp.array(batch),
      jnp.array(ei),
      jnp.array(ev),
      jnp.array([0], dtype=jnp.int32),
      jnp.array([0], dtype=jnp.int32),
      ds_idx,
    )

    # JIT compile
    jit_fn = jax.jit(model.apply)
    jit_fn(*args)['node_embedding'].block_until_ready()

    # Warmup
    for _ in range(5):
      jit_fn(*args)['node_embedding'].block_until_ready()

    # Benchmark
    times = []
    for _ in range(20):
      t0 = time.perf_counter()
      jit_fn(*args)['node_embedding'].block_until_ready()
      times.append(time.perf_counter() - t0)

    ms = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f'{MODEL_NAME} MoE JIT (6 atoms, CPU): {ms:.1f} +/- {std:.1f} ms')


if __name__ == '__main__':
  absltest.main()
