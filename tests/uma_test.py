"""Tests for jax_md._nn.uma.

These tests verify the UMA model components and full forward pass work correctly.
"""

import os
from dataclasses import replace

from absl.testing import absltest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jax_md import energy
from jax_md import space
from jax_md import test_util
from jax_md._nn.uma.model import UMABackbone, UMAConfig
from jax_md._nn.uma.model_moe import (
  SO2MConvMoE,
  UMAMoEBackbone,
  UMAMoEConfig,
  merge_mole_params,
)
from jax_md._nn.uma.heads import MLPEnergyHead, LinearForceHead
from jax_md._nn.uma.nn.mole import MOLELinear
from jax_md._nn.uma.nn.so2_layers import SO2MConv
from jax_md._nn.uma.nn.so3_layers import SO3Linear
from jax_md._nn.uma.nn.radial import PolynomialEnvelope, RadialMLP
from jax_md._nn.uma.nn.layer_norm import EquivariantRMSNorm
from jax_md._nn.uma.pretrained import (
  convert_backbone_key,
  extract_energy_correction,
  load_params_into_module,
  module_to_params,
)
from jax_md._nn.uma.weight_conversion import convert_key
from jax_md._nn.uma.weight_conversion import convert_pytorch_state_dict
from jax_md._nn.uma.kernels import segment_mm
from jax_md._nn.uma.kernels import segment_mm_reference
from jax_md._nn.uma.nn.embedding import (
  DatasetEmbedding,
  dataset_names_to_indices,
)
from jax_md._nn.uma.common.rotation import (
  safe_acos,
  init_edge_rot_euler_angles,
  eulers_to_wigner,
  load_jacobi_matrices_from_file,
)
from jax_md._nn.uma.common.so3 import (
  create_coefficient_mapping,
  create_so3_grid,
)


TEST_TOL = 1e-12


def create_test_data(num_atoms=10, num_systems=2, cutoff=5.0, seed=42):
  """Create synthetic test data for UMA model testing."""
  np.random.seed(seed)

  positions = np.random.randn(num_atoms, 3).astype(np.float32) * 2.0
  atomic_numbers = np.random.choice([1, 6, 7, 8], size=num_atoms).astype(
    np.int32
  )

  atoms_per_system = num_atoms // num_systems
  batch = np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32)
  if len(batch) < num_atoms:
    batch = np.concatenate(
      [batch, np.full(num_atoms - len(batch), num_systems - 1)]
    ).astype(np.int32)

  # Build edges (within cutoff)
  edge_src, edge_dst = [], []
  for i in range(num_atoms):
    for j in range(num_atoms):
      if i != j:
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < cutoff:
          edge_src.append(i)
          edge_dst.append(j)

  edge_index = np.array([edge_src, edge_dst], dtype=np.int32)
  edge_distance_vec = (
    positions[edge_index[0]] - positions[edge_index[1]]
  ).astype(np.float32)

  charge = np.zeros(num_systems, dtype=np.float32)
  spin = np.zeros(num_systems, dtype=np.float32)
  dataset_idx = np.zeros(num_systems, dtype=np.int32)

  return {
    'positions': jnp.array(positions),
    'atomic_numbers': jnp.array(atomic_numbers),
    'batch': jnp.array(batch),
    'edge_index': jnp.array(edge_index),
    'edge_distance_vec': jnp.array(edge_distance_vec),
    'charge': jnp.array(charge),
    'spin': jnp.array(spin),
    'dataset_idx': jnp.array(dataset_idx),
  }


class SafeAcosTest(test_util.JAXMDTestCase):
  """Tests for safe_acos."""

  def test_safe_acos_forward_exact(self):
    """Forward pass should be exact arccos, not clamped."""
    x = jnp.array([0.0, 0.5, -0.5, 0.99, -0.99])
    result = safe_acos(x)
    expected = jnp.arccos(x)
    self.assertTrue(jnp.allclose(result, expected, atol=1e-6))

  def test_safe_acos_gradient_stable(self):
    """Gradient should not produce NaN near ±1."""
    x = jnp.array([0.999999, -0.999999, 1.0 - 1e-8, -1.0 + 1e-8])
    grad_fn = jax.grad(lambda x: safe_acos(x).sum())
    grads = grad_fn(x)
    self.assertTrue(jnp.all(jnp.isfinite(grads)))


class CoefficientMappingTest(test_util.JAXMDTestCase):
  """Tests for CoefficientMapping."""

  def test_mapping_sizes(self):
    mapping = create_coefficient_mapping(2, 2)
    # lmax=2, mmax=2: l=0 has 1 coeff, l=1 has 3, l=2 has 5 -> total 9
    self.assertEqual(mapping.res_size, 9)
    # m_size: m=0 has 3 (l=0,1,2), m=1 has 4 (2 real + 2 imag from l=1,2),
    # m=2 has 2 (1 real + 1 imag from l=2)
    self.assertEqual(mapping.m_size[0], 3)
    self.assertEqual(mapping.m_size[1], 4)
    self.assertEqual(mapping.m_size[2], 2)

  def test_to_m_is_permutation(self):
    """to_m should be a permutation matrix."""
    mapping = create_coefficient_mapping(2, 2)
    # Each row and column should have exactly one 1
    row_sums = jnp.sum(mapping.to_m, axis=1)
    col_sums = jnp.sum(mapping.to_m, axis=0)
    self.assertTrue(jnp.allclose(row_sums, 1.0))
    self.assertTrue(jnp.allclose(col_sums, 1.0))


class WignerDTest(test_util.JAXMDTestCase):
  """Tests for Wigner D-matrix computation."""

  def test_wigner_identity(self):
    """D(0,0,0) should be identity when using the bundled Jd matrices."""
    Jd_list = load_jacobi_matrices_from_file(2)
    alpha = jnp.zeros(1)
    beta = jnp.zeros(1)
    gamma = jnp.zeros(1)
    wigner = eulers_to_wigner((alpha, beta, gamma), 0, 2, Jd_list)
    expected = jnp.eye(9).reshape(1, 9, 9)
    self.assertAllClose(wigner, expected, atol=TEST_TOL, rtol=TEST_TOL)

  def test_wigner_orthogonal(self):
    """Wigner D-matrices should be orthogonal (D @ D^T = I)."""
    Jd_list = load_jacobi_matrices_from_file(2)
    alpha = jnp.array([0.5])
    beta = jnp.array([1.2])
    gamma = jnp.array([0.3])
    wigner = eulers_to_wigner((alpha, beta, gamma), 0, 2, Jd_list)
    product = jnp.einsum('bij,bkj->bik', wigner, wigner)
    expected = jnp.eye(9).reshape(1, 9, 9)
    self.assertAllClose(product, expected, atol=TEST_TOL, rtol=TEST_TOL)


class SO2MConvTest(test_util.JAXMDTestCase):
  """Tests for SO2MConv."""

  def test_so2mconv_output_shape(self):
    """Test SO2MConv produces correct output shape."""
    key = jax.random.PRNGKey(0)
    conv = SO2MConv(
      m=1, sphere_channels=32, m_output_channels=16, lmax=2, key=key
    )
    # m=1: num_coefficients = lmax - m + 1 = 2
    # Input: (E, 2, num_coeffs * sphere_channels) = (5, 2, 64)
    x = jax.random.normal(key, (5, 2, 2 * 32))
    x_r, x_i = conv(x)
    # Output: (E, num_coeffs, m_output_channels) = (5, 2, 16)
    self.assertEqual(x_r.shape, (5, 2, 16))
    self.assertEqual(x_i.shape, (5, 2, 16))

  def test_so2mconv_jit(self):
    """Test SO2MConv is JIT-compatible."""
    key = jax.random.PRNGKey(0)
    conv = SO2MConv(
      m=1, sphere_channels=16, m_output_channels=8, lmax=2, key=key
    )
    x = jax.random.normal(key, (3, 2, 2 * 16))
    x_r, x_i = eqx.filter_jit(conv)(x)
    self.assertTrue(jnp.all(jnp.isfinite(x_r)))


def manual_mole_linear(layer, x, expert_coefficients, batch_indices, use_bias):
  mixed_weights = jnp.einsum('eoi,be->boi', layer.weights, expert_coefficients)
  per_item_weights = mixed_weights[batch_indices]
  if x.ndim == 2:
    out = jnp.einsum('ni,noi->no', x, per_item_weights)
  else:
    out = jnp.einsum('nci,noi->nco', x, per_item_weights)
  if use_bias:
    out = out + layer.bias
  return out


class SegmentMMTest(test_util.JAXMDTestCase):
  """Tests for segmented MOLE matmul."""

  def skip_without_gpu(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas segment_mm path requires a GPU backend.')

  def test_segment_mm_matches_reference(self):
    key = jax.random.PRNGKey(101)
    x = jax.random.normal(key, (7, 5))
    weights = jax.random.normal(key, (3, 4, 5))
    sizes = jnp.array([2, 3, 2], dtype=jnp.int32)
    expected = jnp.concatenate(
      [x[0:2] @ weights[0].T, x[2:5] @ weights[1].T, x[5:7] @ weights[2].T]
    )

    self.assertAllClose(
      segment_mm(x, weights, sizes),
      expected,
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

  def test_segment_mm_pallas_matches_reference(self):
    self.skip_without_gpu()
    key = jax.random.PRNGKey(104)
    x = jax.random.normal(key, (9, 5))
    weights = jax.random.normal(key, (3, 4, 5))
    sizes = jnp.array([2, 3, 4], dtype=jnp.int32)

    actual = segment_mm(x, weights, sizes, use_pallas=True, max_size=4)
    expected = segment_mm_reference(x, weights, sizes)

    self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)

  def test_segment_mm_pallas_follows_default_matmul_precision(self):
    self.skip_without_gpu()
    key = jax.random.PRNGKey(105)
    x = jax.random.normal(key, (300, 64), dtype=jnp.float32)
    weights = jax.random.normal(key, (2, 48, 64), dtype=jnp.float32)
    sizes = jnp.array([120, 180], dtype=jnp.int32)
    exact = np.concatenate(
      [
        np.asarray(x[:120], np.float64) @ np.asarray(weights[0], np.float64).T,
        np.asarray(x[120:], np.float64) @ np.asarray(weights[1], np.float64).T,
      ]
    )
    scale = np.abs(exact).max()

    with jax.default_matmul_precision('highest'):
      out = segment_mm(x, weights, sizes, use_pallas=True, max_size=180)
    self.assertLess(np.abs(np.asarray(out) - exact).max() / scale, 1e-5)

    # the default lets the kernel use tensor cores like every other f32 matmul
    out = segment_mm(x, weights, sizes, use_pallas=True, max_size=180)
    self.assertLess(np.abs(np.asarray(out) - exact).max() / scale, 5e-3)

  def test_segment_mm_pallas_gradients_match_reference(self):
    self.skip_without_gpu()
    key = jax.random.PRNGKey(105)
    x = jax.random.normal(key, (9, 4))
    weights = jax.random.normal(key, (3, 5, 4))
    sizes = jnp.array([2, 3, 4], dtype=jnp.int32)

    def loss(fn, x, weights):
      out = fn(x, weights, sizes)
      return jnp.sum(out**2)

    dx, dw = jax.grad(
      lambda x, w: loss(
        lambda x, w, sizes: segment_mm(
          x, w, sizes, use_pallas=True, max_size=4
        ),
        x,
        w,
      ),
      (0, 1),
    )(x, weights)
    dx_ref, dw_ref = jax.grad(
      lambda x, w: loss(segment_mm_reference, x, w), (0, 1)
    )(x, weights)

    self.assertAllClose(dx, dx_ref, atol=TEST_TOL, rtol=TEST_TOL)
    self.assertAllClose(dw, dw_ref, atol=TEST_TOL, rtol=TEST_TOL)

  def test_segment_mm_pallas_hessian_matches_reference(self):
    self.skip_without_gpu()
    key = jax.random.PRNGKey(107)
    kx, kw, ktx, ktw = jax.random.split(key, 4)
    x = jax.random.normal(kx, (5, 3))
    weights = jax.random.normal(kw, (2, 4, 3))
    sizes = jnp.array([2, 3], dtype=jnp.int32)
    tx = 0.01 * jax.random.normal(ktx, x.shape)
    tw = 0.01 * jax.random.normal(ktw, weights.shape)

    def pallas_loss(x, weights):
      out = segment_mm(x, weights, sizes, use_pallas=True, max_size=3)
      return jnp.sum(out**2)

    def reference_loss(x, weights):
      out = segment_mm_reference(x, weights, sizes)
      return jnp.sum(out**2)

    argnums = (0, 1)
    for transform in (
      lambda fn: jax.hessian(fn, argnums=argnums),
      lambda fn: jax.jacfwd(jax.jacrev(fn, argnums=argnums), argnums=argnums),
      lambda fn: jax.jacrev(jax.jacfwd(fn, argnums=argnums), argnums=argnums),
    ):
      self.assertAllClose(
        transform(pallas_loss)(x, weights),
        transform(reference_loss)(x, weights),
        atol=TEST_TOL,
        rtol=TEST_TOL,
      )

    def hvp(loss_fn, tangents):
      grad_fn = jax.grad(loss_fn, argnums=argnums)
      return jax.jvp(lambda x, w: grad_fn(x, w), (x, weights), tangents)[1]

    self.assertAllClose(
      hvp(pallas_loss, (tx, tw)),
      hvp(reference_loss, (tx, tw)),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

    batched_hvp = jax.vmap(lambda tx, tw: hvp(pallas_loss, (tx, tw)))
    batched_hvp_ref = jax.vmap(lambda tx, tw: hvp(reference_loss, (tx, tw)))
    self.assertAllClose(
      batched_hvp(jnp.stack([tx, 2.0 * tx]), jnp.stack([tw, -tw])),
      batched_hvp_ref(jnp.stack([tx, 2.0 * tx]), jnp.stack([tw, -tw])),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

  def test_segment_mm_pallas_ignores_padded_rows(self):
    self.skip_without_gpu()
    key = jax.random.PRNGKey(108)
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (7, 3))
    weights = jax.random.normal(kw, (2, 4, 3))
    sizes = jnp.array([2, 3], dtype=jnp.int32)

    def pallas_loss(x, weights):
      out = segment_mm(x, weights, sizes, use_pallas=True, max_size=3)
      return jnp.sum(out[: jnp.sum(sizes)] ** 2)

    def reference_loss(x, weights):
      out = segment_mm_reference(x, weights, sizes)
      return jnp.sum(out[: jnp.sum(sizes)] ** 2)

    actual = segment_mm(x, weights, sizes, use_pallas=True, max_size=3)
    expected = segment_mm_reference(x, weights, sizes)
    self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)
    self.assertAllClose(actual[5:], jnp.zeros_like(actual[5:]))

    dx, dw = jax.grad(pallas_loss, argnums=(0, 1))(x, weights)
    dx_ref, dw_ref = jax.grad(reference_loss, argnums=(0, 1))(x, weights)
    self.assertAllClose(dx, dx_ref, atol=TEST_TOL, rtol=TEST_TOL)
    self.assertAllClose(dw, dw_ref, atol=TEST_TOL, rtol=TEST_TOL)
    self.assertAllClose(dx[5:], jnp.zeros_like(dx[5:]))

    self.assertAllClose(
      jax.hessian(pallas_loss, argnums=(0, 1))(x, weights),
      jax.hessian(reference_loss, argnums=(0, 1))(x, weights),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

  def test_segment_mm_3d_matches_flat_reference(self):
    key = jax.random.PRNGKey(103)
    x = jax.random.normal(key, (5, 2, 4))
    weights = jax.random.normal(key, (2, 3, 4))
    sizes = jnp.array([2, 3], dtype=jnp.int32)

    actual = segment_mm(x, weights, sizes)
    expected = segment_mm_reference(
      x.reshape(10, 4), weights, sizes * x.shape[1]
    ).reshape(5, 2, 3)

    self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)

  def test_segment_mm_pallas_3d_matches_flat_reference(self):
    self.skip_without_gpu()
    key = jax.random.PRNGKey(106)
    x = jax.random.normal(key, (5, 2, 4))
    weights = jax.random.normal(key, (2, 3, 4))
    sizes = jnp.array([2, 3], dtype=jnp.int32)

    actual = segment_mm(x, weights, sizes, use_pallas=True, max_size=3)
    expected = segment_mm_reference(
      x.reshape(10, 4), weights, sizes * x.shape[1]
    ).reshape(5, 2, 3)

    self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)


class MOLEInferencePathTest(test_util.JAXMDTestCase):
  """Tests for UMA MoE inference helpers."""

  def test_mole_linear_inference_paths_match_reference(self):
    key = jax.random.PRNGKey(12)
    keys = jax.random.split(key, 8)
    cases = [
      (
        jax.random.normal(keys[0], (7, 5)),
        jax.nn.softmax(jax.random.normal(keys[1], (1, 4)), axis=-1),
        jnp.zeros((7,), dtype=jnp.int32),
        True,
      ),
      (
        jax.random.normal(keys[2], (8, 2, 5)),
        jax.nn.softmax(jax.random.normal(keys[3], (2, 4)), axis=-1),
        jnp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32),
        False,
      ),
      (
        jax.random.normal(keys[4], (6, 5)),
        jax.nn.softmax(jax.random.normal(keys[5], (3, 4)), axis=-1),
        jnp.array([0, 1, 0, 2, 1, 2], dtype=jnp.int32),
        True,
      ),
    ]
    for x, coefficients, batch_indices, use_bias in cases:
      layer = MOLELinear(4, 5, 6, use_bias=use_bias, key=keys[6])
      actual = layer(x, coefficients, batch_indices)
      expected = manual_mole_linear(
        layer, x, coefficients, batch_indices, use_bias
      )
      self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)

  def test_mole_linear_segment_mm_requires_contiguous_opt_in(self):
    key = jax.random.PRNGKey(18)
    keys = jax.random.split(key, 4)
    x = jax.random.normal(keys[0], (6, 5))
    coefficients = jax.nn.softmax(jax.random.normal(keys[1], (3, 4)), axis=-1)
    batch_indices = jnp.array([0, 1, 0, 2, 1, 2], dtype=jnp.int32)
    layer = MOLELinear(
      4,
      5,
      6,
      use_bias=True,
      use_segment_mm_pallas=True,
      max_segment_size=2,
      key=keys[2],
    )

    actual = layer(x, coefficients, batch_indices)
    expected = manual_mole_linear(
      layer, x, coefficients, batch_indices, use_bias=True
    )

    self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)

  def test_mole_linear_segment_mm_contiguous_opt_in_matches_reference(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas segment_mm path requires a GPU backend.')

    key = jax.random.PRNGKey(19)
    keys = jax.random.split(key, 4)
    x = jax.random.normal(keys[0], (5, 5))
    coefficients = jax.nn.softmax(jax.random.normal(keys[1], (2, 4)), axis=-1)
    batch_indices = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
    layer = MOLELinear(
      4,
      5,
      6,
      use_bias=True,
      use_segment_mm_pallas=True,
      max_segment_size=3,
      assume_segment_contiguous_batches=True,
      key=keys[2],
    )

    actual = layer(x, coefficients, batch_indices)
    expected = manual_mole_linear(
      layer, x, coefficients, batch_indices, use_bias=True
    )

    self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)

  def test_mole_linear_accepts_merged_weights(self):
    key = jax.random.PRNGKey(14)
    keys = jax.random.split(key, 4)
    x = jax.random.normal(keys[0], (6, 5))
    coefficients = jax.nn.softmax(jax.random.normal(keys[1], (1, 4)), axis=-1)
    batch_indices = jnp.zeros((6,), dtype=jnp.int32)
    layer = MOLELinear(4, 5, 6, use_bias=True, key=keys[2])
    expected = layer(x, coefficients, batch_indices)
    merged_params = {
      'params': {
        'weights': jnp.einsum('e,eoi->oi', coefficients[0], layer.weights),
        'bias': layer.bias,
      }
    }
    merged_layer = load_params_into_module(
      merged_params,
      MOLELinear(4, 5, 6, use_bias=True, merged=True, key=keys[3]),
    )
    actual = merged_layer(x, coefficients, batch_indices)
    self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)

  def test_so2_m_conv_block_gemm_matches_reference(self):
    key = jax.random.PRNGKey(16)
    keys = jax.random.split(key, 4)
    x_m = jax.random.normal(keys[0], (5, 2, 6))
    coefficients = jax.nn.softmax(jax.random.normal(keys[1], (1, 2)), axis=-1)
    edge_batch = jnp.zeros((5,), dtype=jnp.int32)
    ref = SO2MConvMoE(
      m=1,
      sphere_channels=3,
      m_output_channels=4,
      lmax=2,
      num_experts=2,
      so2_block_gemm=False,
      key=keys[2],
    )
    params = {'params': {'fc': {'weights': ref.fc.weights}}}
    block = load_params_into_module(
      params,
      SO2MConvMoE(
        m=1,
        sphere_channels=3,
        m_output_channels=4,
        lmax=2,
        num_experts=2,
        so2_block_gemm=True,
        key=keys[3],
      ),
    )
    ref_out = ref(x_m, coefficients, edge_batch)
    block_out = block(x_m, coefficients, edge_batch)
    self.assertAllClose(block_out[0], ref_out[0], atol=TEST_TOL, rtol=TEST_TOL)
    self.assertAllClose(block_out[1], ref_out[1], atol=TEST_TOL, rtol=TEST_TOL)

    merged_params = merge_mole_params(params, coefficients)
    merged_block = load_params_into_module(
      merged_params,
      SO2MConvMoE(
        m=1,
        sphere_channels=3,
        m_output_channels=4,
        lmax=2,
        num_experts=2,
        merged_mole=True,
        so2_block_gemm=True,
        key=keys[3],
      ),
    )
    merged_out = merged_block(x_m, coefficients, edge_batch)
    self.assertAllClose(merged_out[0], ref_out[0], atol=TEST_TOL, rtol=TEST_TOL)
    self.assertAllClose(merged_out[1], ref_out[1], atol=TEST_TOL, rtol=TEST_TOL)

  def test_extract_energy_correction_from_task_config(self):
    tasks = [
      {
        'name': 'omat_energy',
        'property': 'energy',
        'datasets': ['omat'],
        'normalizer': {'mean': 1.25, 'rmsd': 3.5},
        'element_references': {
          'element_references': {'_args_': [[0.0, -1.0, 2.0]]}
        },
      }
    ]
    correction = extract_energy_correction(tasks, 'omat')
    self.assertAllClose(
      correction['mean'], jnp.asarray(1.25, dtype=jnp.float32)
    )
    self.assertAllClose(correction['rmsd'], jnp.asarray(3.5, dtype=jnp.float32))
    self.assertAllClose(
      correction['element_refs'],
      jnp.array([0.0, -1.0, 2.0], dtype=jnp.float32),
    )

  def test_uma_neighbor_list_merge_mole_matches_unmerged(self):
    cfg = UMAMoEConfig(
      sphere_channels=4,
      lmax=1,
      mmax=1,
      num_layers=1,
      hidden_channels=4,
      cutoff=2.0,
      edge_channels=4,
      num_distance_basis=4,
      ff_type='spectral',
      use_dataset_embedding=False,
      use_composition_embedding=False,
      num_experts=2,
      routing_hidden_channels=4,
      use_kernels=False,
    )
    box = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32)
    displacement_fn, _ = space.periodic(box)
    atoms = jnp.array([1, 6, 8], dtype=jnp.int32)
    position = jnp.array(
      [[0.1, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.9, 0.1]],
      dtype=jnp.float32,
    )
    neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
      displacement_fn,
      box,
      cfg=cfg,
      atoms=atoms,
      disable_cell_list=True,
    )
    _, merge_init_fn, merge_energy_fn = energy.uma_neighbor_list(
      displacement_fn,
      box,
      cfg=cfg,
      atoms=atoms,
      merge_mole=True,
      disable_cell_list=True,
    )
    neighbors = neighbor_fn.allocate(position)
    key = jax.random.PRNGKey(15)
    params = init_fn(key, position, neighbors)
    merged_params = merge_init_fn(key, position, neighbors)
    self.assertAllClose(
      merge_energy_fn(merged_params, position, neighbors),
      energy_fn(params, position, neighbors),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

  def test_uma_neighbor_list_rejects_linear_head_for_checkpoint(self):
    box = jnp.eye(3) * 5.0
    displacement_fn, _ = space.periodic_general(box)
    with self.assertRaisesRegex(ValueError, "head_type='mlp'"):
      energy.uma_neighbor_list(
        displacement_fn,
        box,
        checkpoint_path='uma-s-1p1',
        atoms=jnp.array([1], dtype=jnp.int32),
        head_type='linear',
      )


class SO3LinearTest(test_util.JAXMDTestCase):
  """Tests for SO3Linear."""

  def test_so3linear_output_shape(self):
    key = jax.random.PRNGKey(0)
    linear = SO3Linear(32, 16, 2, key=key)
    x = jax.random.normal(key, (5, 9, 32))  # (batch, (lmax+1)^2, in_features)
    out = linear(x)
    self.assertEqual(out.shape, (5, 9, 16))

  def test_so3linear_no_weight_shift(self):
    """Verify the weight is used as-is (no subtraction bug)."""
    key = jax.random.PRNGKey(0)
    linear = SO3Linear(16, 8, 1, key=key, use_bias=False)
    x = jax.random.normal(key, (2, 4, 16))

    # Set weights to known values
    linear = eqx.tree_at(
      lambda t: t.weight, linear, jnp.ones_like(linear.weight)
    )
    out1 = linear(x)
    # Run again — should be identical (no cumulative shift)
    out2 = linear(x)
    self.assertTrue(jnp.allclose(out1, out2))
    self.assertTrue(jnp.all(jnp.isfinite(out1)))


class RadialMLPTest(test_util.JAXMDTestCase):
  """Tests for RadialMLP."""

  def test_radial_mlp_output_shape(self):
    key = jax.random.PRNGKey(0)
    mlp = RadialMLP(channels_list=[64, 32, 16], key=key)
    x = jax.random.normal(key, (10, 64))
    out = mlp(x)
    self.assertEqual(out.shape, (10, 16))

  def test_polynomial_envelope(self):
    env = PolynomialEnvelope(exponent=5)
    d = jnp.array([0.0, 0.5, 0.99, 1.0, 1.5])
    out = env(d)
    # At d=0, envelope should be 1
    self.assertAlmostEqual(float(out[0]), 1.0, places=5)
    # At d>=1, envelope should be 0
    self.assertAlmostEqual(float(out[3]), 0.0, places=5)
    self.assertAlmostEqual(float(out[4]), 0.0, places=5)


class DatasetEmbeddingTest(test_util.JAXMDTestCase):
  """Tests for DatasetEmbedding with integer indices."""

  def test_dataset_embedding_jit(self):
    """DatasetEmbedding should be JIT-compatible with integer indices."""
    key = jax.random.PRNGKey(0)
    emb = DatasetEmbedding(embedding_size=32, num_datasets=5, key=key)
    idx = jnp.array([0, 2, 4])

    out = eqx.filter_jit(emb)(idx)
    self.assertEqual(out.shape, (3, 32))
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_dataset_names_to_indices(self):
    """Test string to index conversion."""
    dataset_list = ['oc20', 'omol', 'omat', 'odac', 'omc']
    indices = dataset_names_to_indices(['omat', 'mptrj', 'oc20'], dataset_list)
    # omat -> 2, mptrj -> omat -> 2, oc20 -> 0
    np.testing.assert_array_equal(np.array(indices), [2, 2, 0])


class EquivariantRMSNormTest(test_util.JAXMDTestCase):
  """Tests for EquivariantRMSNorm."""

  def test_rms_norm_output_shape(self):
    norm = EquivariantRMSNorm(lmax=2, num_channels=32)
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 9, 32))
    out = norm(x)
    self.assertEqual(out.shape, (5, 9, 32))
    self.assertTrue(jnp.all(jnp.isfinite(out)))


class CheckpointKeyTest(test_util.JAXMDTestCase):
  def test_grid_mlp_layers_land_on_grid_atomwise_dense(self):
    for index, dense in ((0, 'Dense_0'), (2, 'Dense_1'), (4, 'Dense_2')):
      key = f'blocks.3.atom_wise.grid_mlp.{index}.weight'
      expected = ('blocks_3', 'atom_wise', dense, 'kernel')
      self.assertEqual(convert_backbone_key(key), expected)
      self.assertEqual(convert_key(key), expected)

  def test_strict_loading_rejects_a_stray_checkpoint_entry(self):
    head = MLPEnergyHead(
      sphere_channels=4, hidden_channels=4, key=jax.random.PRNGKey(0)
    )
    params = module_to_params(head)
    params['stray'] = np.zeros(2, np.float32)
    load_params_into_module(params, head)
    with self.assertRaisesRegex(ValueError, 'stray'):
      load_params_into_module(params, head, strict=True)

  def test_loading_rejects_a_shape_mismatch(self):
    head = MLPEnergyHead(
      sphere_channels=4, hidden_channels=4, key=jax.random.PRNGKey(0)
    )
    params = module_to_params(head)
    params['linear_0']['kernel'] = np.zeros((4, 5), np.float32)
    with self.assertRaisesRegex(ValueError, 'linear_0/kernel'):
      load_params_into_module(params, head)

  def test_embedding_tables_load_into_uma_backbone(self):
    config = UMAConfig(
      sphere_channels=4,
      num_layers=1,
      hidden_channels=4,
      edge_channels=4,
      num_distance_basis=4,
    )
    names = ('sphere_embedding', 'source_embedding', 'target_embedding')
    state_dict = {
      f'backbone.{name}.weight': np.full((100, 4), i + 1.0, np.float32)
      for i, name in enumerate(names)
    }
    model = load_params_into_module(
      convert_pytorch_state_dict(state_dict),
      UMABackbone(config=config, key=jax.random.PRNGKey(0)),
    )
    for i, name in enumerate(names):
      self.assertAllClose(
        getattr(model, name).embedding,
        jnp.full((100, 4), i + 1.0, jnp.float32),
      )

  def test_grid_model_has_no_constant_leaves(self):
    config = UMAConfig(
      sphere_channels=4,
      num_layers=1,
      hidden_channels=4,
      edge_channels=4,
      num_distance_basis=4,
      ff_type='grid',
      act_type='s2',
    )
    model = UMABackbone(config=config, key=jax.random.PRNGKey(0))
    template = eqx.filter_eval_shape(
      UMABackbone, config=config, key=jax.random.PRNGKey(1)
    )
    loaded = load_params_into_module(
      module_to_params(model), template, strict=True
    )
    self.assertTrue(eqx.tree_equal(loaded, model))


class UMAConfigTest(test_util.JAXMDTestCase):
  def test_uma_config_defaults(self):
    config = UMAConfig()
    self.assertEqual(config.sphere_channels, 128)
    self.assertEqual(config.lmax, 2)
    self.assertEqual(config.mmax, 2)
    self.assertEqual(config.num_layers, 2)


class UMAHeadsTest(test_util.JAXMDTestCase):
  def test_mlp_energy_head(self):
    key = jax.random.PRNGKey(0)
    head = MLPEnergyHead(sphere_channels=32, hidden_channels=32, key=key)
    node_emb = jax.random.normal(key, (8, 9, 32))
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    output = head(node_emb, batch, 2)
    self.assertEqual(output['energy'].shape, (2,))
    self.assertTrue(jnp.all(jnp.isfinite(output['energy'])))

  def test_mlp_energy_head_jit(self):
    key = jax.random.PRNGKey(0)
    head = MLPEnergyHead(sphere_channels=32, hidden_channels=32, key=key)
    node_emb = jax.random.normal(key, (8, 9, 32))
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    output = eqx.filter_jit(head)(node_emb, batch, 2)
    self.assertEqual(output['energy'].shape, (2,))

  def test_linear_force_head(self):
    key = jax.random.PRNGKey(0)
    head = LinearForceHead(sphere_channels=32, key=key)
    node_emb = jax.random.normal(key, (8, 9, 32))
    output = head(node_emb)
    self.assertEqual(output['forces'].shape, (8, 3))


class UMABackboneForwardTest(test_util.JAXMDTestCase):
  def test_uma_backbone_forward_pass(self):
    """Test full UMABackbone forward pass with synthetic data."""
    config = UMAConfig(
      max_num_elements=100,
      sphere_channels=32,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=32,
      cutoff=5.0,
      edge_channels=32,
      num_distance_basis=64,
      norm_type='rms_norm_sh',
      act_type='gate',
      ff_type='grid',
      chg_spin_emb_type='pos_emb',
      dataset_list=['oc20', 'omol', 'omat', 'odac', 'omc'],
      use_dataset_embedding=True,
    )

    key = jax.random.PRNGKey(0)
    model = UMABackbone(config=config, key=key)
    data = create_test_data(num_atoms=8, num_systems=2, cutoff=config.cutoff)

    output = model(
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      data['dataset_idx'],
    )

    self.assertIn('node_embedding', output)
    num_atoms = data['positions'].shape[0]
    sph_size = (config.lmax + 1) ** 2
    self.assertEqual(
      output['node_embedding'].shape,
      (num_atoms, sph_size, config.sphere_channels),
    )
    self.assertTrue(jnp.all(jnp.isfinite(output['node_embedding'])))

  def test_uma_backbone_requires_dataset_idx(self):
    config = UMAConfig(
      sphere_channels=8,
      num_layers=1,
      hidden_channels=8,
      edge_channels=8,
      num_distance_basis=8,
      dataset_list=['omat', 'omol'],
    )
    model = UMABackbone(config=config, key=jax.random.PRNGKey(0))
    data = create_test_data(num_atoms=4, num_systems=1, cutoff=config.cutoff)
    with self.assertRaisesRegex(ValueError, 'dataset_idx'):
      model(
        data['positions'],
        data['atomic_numbers'],
        data['batch'],
        data['edge_index'],
        data['edge_distance_vec'],
        data['charge'],
        data['spin'],
      )

  def test_uma_backbone_no_dataset_embedding(self):
    config = UMAConfig(
      sphere_channels=32,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=32,
      cutoff=5.0,
      edge_channels=32,
      num_distance_basis=64,
      use_dataset_embedding=False,
    )

    key = jax.random.PRNGKey(42)
    model = UMABackbone(config=config, key=key)
    data = create_test_data(num_atoms=6, num_systems=2, cutoff=config.cutoff)

    output = model(
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    self.assertTrue(jnp.all(jnp.isfinite(output['node_embedding'])))

  def test_uma_backbone_deterministic(self):
    config = UMAConfig(
      sphere_channels=32,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=32,
      cutoff=5.0,
      edge_channels=32,
      num_distance_basis=64,
      use_dataset_embedding=False,
    )

    key = jax.random.PRNGKey(0)
    model = UMABackbone(config=config, key=key)
    data = create_test_data(num_atoms=6, num_systems=2, cutoff=config.cutoff)

    out1 = model(
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )
    out2 = model(
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    self.assertTrue(
      jnp.allclose(out1['node_embedding'], out2['node_embedding'])
    )

  def test_uma_backbone_multiple_layers(self):
    config = UMAConfig(
      sphere_channels=32,
      lmax=2,
      mmax=2,
      num_layers=2,
      hidden_channels=32,
      cutoff=5.0,
      edge_channels=32,
      num_distance_basis=64,
      use_dataset_embedding=False,
    )

    key = jax.random.PRNGKey(0)
    model = UMABackbone(config=config, key=key)
    data = create_test_data(num_atoms=6, num_systems=2, cutoff=config.cutoff)

    output = model(
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    self.assertTrue(jnp.all(jnp.isfinite(output['node_embedding'])))

  def test_uma_backbone_spectral_ffn(self):
    """Test with spectral SO3Linear feed-forward instead of grid."""
    config = UMAConfig(
      sphere_channels=32,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=32,
      cutoff=5.0,
      edge_channels=32,
      num_distance_basis=64,
      ff_type='spectral',
      use_dataset_embedding=False,
    )

    key = jax.random.PRNGKey(0)
    model = UMABackbone(config=config, key=key)
    data = create_test_data(num_atoms=6, num_systems=2, cutoff=config.cutoff)

    output = model(
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    self.assertTrue(jnp.all(jnp.isfinite(output['node_embedding'])))


class UMAGradientTest(test_util.JAXMDTestCase):
  """Test that gradients flow through the model, for force computation."""

  def test_energy_gradient_wrt_positions(self):
    """Test that we can compute forces via energy gradient."""
    config = UMAConfig(
      sphere_channels=32,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=32,
      cutoff=5.0,
      edge_channels=32,
      num_distance_basis=64,
      use_dataset_embedding=False,
    )

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    model = UMABackbone(config=config, key=key1)
    head = MLPEnergyHead(sphere_channels=32, hidden_channels=32, key=key2)

    data = create_test_data(num_atoms=6, num_systems=1, cutoff=config.cutoff)

    def energy_fn(positions):
      edge_vec = (
        positions[data['edge_index'][0]] - positions[data['edge_index'][1]]
      )
      emb = model(
        positions,
        data['atomic_numbers'],
        data['batch'],
        data['edge_index'],
        edge_vec,
        data['charge'],
        data['spin'],
        None,
      )
      result = head(emb['node_embedding'], data['batch'], 1)
      return result['energy'].sum()

    # Compute gradient (forces = -grad)
    grad_fn = jax.grad(energy_fn)
    forces = -grad_fn(data['positions'])

    self.assertEqual(forces.shape, data['positions'].shape)
    self.assertTrue(jnp.all(jnp.isfinite(forces)))

  def test_moe_activation_checkpointing_matches_plain(self):
    cfg = UMAMoEConfig(
      sphere_channels=4,
      lmax=2,
      mmax=2,
      num_layers=2,
      hidden_channels=4,
      cutoff=3.0,
      edge_channels=4,
      num_distance_basis=4,
      ff_type='spectral',
      use_dataset_embedding=False,
      use_composition_embedding=False,
      num_experts=2,
      routing_hidden_channels=4,
      use_kernels=False,
    )
    key = jax.random.PRNGKey(17)
    plain = UMAMoEBackbone(config=cfg, key=key)
    remat = UMAMoEBackbone(
      config=replace(cfg, activation_checkpointing=True), key=key
    )
    position = jnp.array(
      [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.1, 0.8, 0.1]], dtype=jnp.float32
    )
    atomic_numbers = jnp.array([1, 6, 8], dtype=jnp.int32)
    batch = jnp.zeros(3, dtype=jnp.int32)
    edge_index = jnp.array([[0, 1, 2, 0], [1, 2, 0, 2]], jnp.int32)
    zero = jnp.zeros(1, dtype=jnp.int32)

    def energy(model, position):
      edge_vec = position[edge_index[0]] - position[edge_index[1]]
      out = model(
        position, atomic_numbers, batch, edge_index, edge_vec, zero, zero, zero
      )
      return jnp.sum(out['node_embedding'])

    self.assertAllClose(
      energy(remat, position),
      energy(plain, position),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )
    self.assertAllClose(
      jax.grad(lambda p: energy(remat, p))(position),
      jax.grad(lambda p: energy(plain, p))(position),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

  def test_moe_so2_block_gemm_matches_plain_for_batches(self):
    cfg = UMAMoEConfig(
      sphere_channels=4,
      lmax=2,
      mmax=2,
      num_layers=2,
      hidden_channels=4,
      cutoff=3.0,
      edge_channels=4,
      num_distance_basis=4,
      ff_type='spectral',
      use_dataset_embedding=False,
      use_composition_embedding=False,
      num_experts=2,
      routing_hidden_channels=4,
      use_kernels=False,
    )
    key = jax.random.PRNGKey(23)
    plain = UMAMoEBackbone(config=cfg, key=key)
    block = UMAMoEBackbone(config=replace(cfg, so2_block_gemm=True), key=key)
    position = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [0.7, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [5.0, 5.0, 5.0],
        [5.6, 5.2, 5.1],
      ],
      dtype=jnp.float32,
    )
    atomic_numbers = jnp.array([1, 6, 8, 6, 1], dtype=jnp.int32)
    # two systems, edges grouped by system
    batch = jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32)
    edge_index = jnp.array([[0, 1, 2, 0, 3, 4], [1, 2, 0, 2, 4, 3]], jnp.int32)
    zeros = jnp.zeros(2, dtype=jnp.int32)

    def energy(model, position):
      edge_vec = position[edge_index[0]] - position[edge_index[1]]
      out = model(
        position,
        atomic_numbers,
        batch,
        edge_index,
        edge_vec,
        zeros,
        zeros,
        zeros,
      )
      return jnp.sum(out['node_embedding'])

    self.assertAllClose(
      energy(block, position),
      energy(plain, position),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )
    self.assertAllClose(
      jax.grad(lambda p: energy(block, p))(position),
      jax.grad(lambda p: energy(plain, p))(position),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )


def load_pt_module(module_name, file_path):
  """Load a fairchem module directly, bypassing __init__ chains."""
  import importlib.util

  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f'Could not load module spec for {file_path}')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def stub_fairchem():
  """Stub fairchem namespace so submodules can be loaded in isolation."""
  import sys
  import types

  for name in [
    'fairchem',
    'fairchem.core',
    'fairchem.core.models',
    'fairchem.core.models.uma',
    'fairchem.core.models.uma.common',
    'fairchem.core.models.uma.nn',
  ]:
    if name not in sys.modules:
      sys.modules[name] = types.ModuleType(name)


def find_fairchem_src():
  """Locate FairChem UMA source"""
  if 'FAIRCHEM_SRC' in os.environ:
    return os.environ['FAIRCHEM_SRC']
  try:
    import fairchem.core.models.uma as fairchem_uma

    return os.path.dirname(fairchem_uma.__file__)
  except ImportError:
    return None


FAIRCHEM_SRC = find_fairchem_src()


def pt_available():
  """Check if PyTorch and fairchem source are accessible."""
  try:
    import torch  # noqa: F401

    return FAIRCHEM_SRC is not None and os.path.isdir(FAIRCHEM_SRC)
  except ImportError:
    return False


class PyTorchComparisonTest(test_util.JAXMDTestCase):
  """Compare JAX UMA layers against PyTorch originals with copied weights."""

  def setUp(self):
    super().setUp()
    if not pt_available():
      self.skipTest('PyTorch or fairchem source not available')
    stub_fairchem()

  def test_z_rot_mat_matches_pytorch(self):
    """_z_rot_mat should match PyTorch for all l."""
    import torch

    pt_rot = load_pt_module('pt_rot', f'{FAIRCHEM_SRC}/common/rotation.py')
    from jax_md._nn.uma.common.rotation import _z_rot_mat

    angles = np.array([0.0, 0.7, -1.3, 3.14], dtype=np.float64)
    for l in range(4):
      M_pt = pt_rot._z_rot_mat(torch.tensor(angles), l).numpy()
      M_jax = np.array(_z_rot_mat(jnp.array(angles), l))
      np.testing.assert_allclose(
        M_jax,
        M_pt,
        atol=TEST_TOL,
        rtol=TEST_TOL,
        err_msg=f'_z_rot_mat mismatch at l={l}',
      )

  def test_wigner_D_matches_pytorch(self):
    """wigner_D should match PyTorch for all l."""
    import torch

    pt_rot = load_pt_module('pt_rot', f'{FAIRCHEM_SRC}/common/rotation.py')
    from jax_md._nn.uma.common.rotation import wigner_D as jax_wigner_D

    Jd_pt = torch.load(
      'jax_md/_nn/uma/Jd.pt', map_location='cpu', weights_only=False
    )
    Jd_pt = [J.double() for J in Jd_pt]
    Jd_jax = load_jacobi_matrices_from_file(3)

    alpha = np.array([0.5, -0.3], dtype=np.float64)
    beta = np.array([0.8, 1.5], dtype=np.float64)
    gamma = np.zeros(2, dtype=np.float64)

    for l in range(4):
      D_pt = pt_rot.wigner_D(
        l, torch.tensor(alpha), torch.tensor(beta), torch.tensor(gamma), Jd_pt
      ).numpy()
      D_jax = np.array(
        jax_wigner_D(
          l, jnp.array(alpha), jnp.array(beta), jnp.array(gamma), Jd_jax[l]
        )
      )
      np.testing.assert_allclose(
        D_jax,
        D_pt,
        atol=TEST_TOL,
        rtol=TEST_TOL,
        err_msg=f'wigner_D mismatch at l={l}',
      )

  def test_coefficient_mapping_to_m_matches_pytorch(self):
    """to_m permutation matrix should be identical to PyTorch."""
    pt_so3 = load_pt_module('pt_so3', f'{FAIRCHEM_SRC}/common/so3.py')
    pt_cm = pt_so3.CoefficientMapping(2, 2)
    jax_cm = create_coefficient_mapping(2, 2)
    np.testing.assert_array_equal(
      np.array(jax_cm.to_m),
      pt_cm.to_m.numpy(),
    )

  def test_so3_grid_matrices_match_pytorch(self):
    """Grid transform matrices should match PyTorch (via e3nn)."""
    pt_so3 = load_pt_module('pt_so3', f'{FAIRCHEM_SRC}/common/so3.py')
    for lmax, mmax in [(2, 2), (2, 1), (3, 2)]:
      pt_grid = pt_so3.SO3_Grid(lmax, mmax)
      jax_grid = create_so3_grid(lmax, mmax)
      np.testing.assert_allclose(
        np.array(jax_grid.to_grid_mat),
        pt_grid.to_grid_mat.numpy(),
        atol=1e-6,
        err_msg=f'to_grid mismatch lmax={lmax},mmax={mmax}',
      )
      np.testing.assert_allclose(
        np.array(jax_grid.from_grid_mat),
        pt_grid.from_grid_mat.numpy(),
        atol=1e-6,
        err_msg=f'from_grid mismatch lmax={lmax},mmax={mmax}',
      )

  def test_so2mconv_matches_pytorch(self):
    """SO2MConv with copied weights should match PyTorch output."""
    import torch
    import torch.nn as tnn

    # Build a minimal PT SO2_m_Conv inline (avoids import issues)
    m, lmax, mmax, sc, moc = 1, 2, 2, 16, 8
    num_coefficients = lmax - m + 1
    num_channels = num_coefficients * sc
    out_channels_half = moc * num_coefficients

    pt_fc = tnn.Linear(num_channels, 2 * out_channels_half, bias=False).double()

    def pt_so2m_forward(x_m):
      x_m = pt_fc(x_m)
      x_m = x_m.reshape(x_m.shape[0], -1, out_channels_half).split(1, dim=1)
      x_r_0, x_i_0, x_r_1, x_i_1 = x_m
      x_m_r = (x_r_0 - x_i_1).view(-1, num_coefficients, moc)
      x_m_i = (x_r_1 + x_i_0).view(-1, num_coefficients, moc)
      return x_m_r, x_m_i

    key = jax.random.PRNGKey(0)
    jax_conv = SO2MConv(
      m=m, sphere_channels=sc, m_output_channels=moc, lmax=lmax, key=key
    )

    np.random.seed(123)
    x_np = np.random.randn(7, 2, num_channels).astype(np.float64)

    # Copy PT weights to JAX (transpose: PT [out,in] -> JAX [in,out])
    jax_conv = eqx.tree_at(
      lambda t: t.fc.kernel,
      jax_conv,
      jnp.array(pt_fc.weight.data.numpy().T),
    )

    with torch.no_grad():
      pt_r, pt_i = pt_so2m_forward(torch.tensor(x_np))
    jax_r, jax_i = jax_conv(jnp.array(x_np))

    np.testing.assert_allclose(
      np.array(jax_r),
      pt_r.numpy(),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )
    np.testing.assert_allclose(
      np.array(jax_i),
      pt_i.numpy(),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

  def test_so3linear_matches_pytorch(self):
    """SO3Linear with copied weights should match PyTorch output."""
    import torch

    pt_so3l = load_pt_module('pt_so3l', f'{FAIRCHEM_SRC}/nn/so3_layers.py')

    lmax, in_f, out_f = 2, 16, 8
    sph = (lmax + 1) ** 2

    pt_lin = pt_so3l.SO3_Linear(
      in_features=in_f, out_features=out_f, lmax=lmax
    ).double()
    key = jax.random.PRNGKey(0)
    jax_lin = SO3Linear(in_f, out_f, lmax, key=key, use_bias=True)

    np.random.seed(77)
    x_np = np.random.randn(4, sph, in_f).astype(np.float64)

    jax_lin = load_params_into_module(
      {
        'params': {
          'weight': jnp.array(pt_lin.weight.data.numpy()),
          'bias': jnp.array(pt_lin.bias.data.numpy()),
        }
      },
      jax_lin,
    )

    with torch.no_grad():
      pt_out = pt_lin(torch.tensor(x_np)).numpy()
    jax_out = np.array(jax_lin(jnp.array(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, atol=TEST_TOL, rtol=TEST_TOL)

  def test_rms_norm_matches_pytorch(self):
    """EquivariantRMSNorm with copied weights should match PyTorch."""
    import torch

    pt_ln = load_pt_module('pt_ln', f'{FAIRCHEM_SRC}/nn/layer_norm.py')

    lmax, ch = 2, 16
    sph = (lmax + 1) ** 2

    torch_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
      pt_norm = pt_ln.EquivariantRMSNormArraySphericalHarmonicsV2(lmax, ch)
    finally:
      torch.set_default_dtype(torch_default_dtype)
    jax_norm = EquivariantRMSNorm(lmax=lmax, num_channels=ch)

    np.random.seed(55)
    x_np = np.random.randn(5, sph, ch).astype(np.float64)

    jax_norm = load_params_into_module(
      {
        'params': {
          'affine_weight': jnp.array(pt_norm.affine_weight.data.numpy()),
          'affine_bias': jnp.array(pt_norm.affine_bias.data.numpy()),
        }
      },
      jax_norm,
    )

    with torch.no_grad():
      pt_out = pt_norm(torch.tensor(x_np)).numpy()
    jax_out = np.array(jax_norm(jnp.array(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, atol=TEST_TOL, rtol=TEST_TOL)

  def test_euler_angles_match_pytorch(self):
    """Euler angle computation should match PyTorch (alpha, beta)."""
    import torch

    pt_rot = load_pt_module('pt_rot', f'{FAIRCHEM_SRC}/common/rotation.py')

    np.random.seed(99)
    vecs = np.random.randn(20, 3).astype(np.float64)

    g_pt, b_pt, a_pt = pt_rot.init_edge_rot_euler_angles(torch.tensor(vecs))
    g_jax, b_jax, a_jax = init_edge_rot_euler_angles(jnp.array(vecs))

    # alpha and beta should match; gamma differs (random vs zero)
    np.testing.assert_allclose(
      np.array(b_jax), b_pt.numpy(), atol=TEST_TOL, rtol=TEST_TOL
    )
    np.testing.assert_allclose(
      np.array(a_jax), a_pt.numpy(), atol=TEST_TOL, rtol=TEST_TOL
    )


class EndToEndComparisonTest(test_util.JAXMDTestCase):
  """End-to-end JAX and PyTorch UMA forward pass on synthetic data."""

  def create_numpy_test_data(self, num_atoms=10, num_systems=2, seed=42):
    np.random.seed(seed)
    positions = np.random.randn(num_atoms, 3).astype(np.float32) * 2.0
    atomic_numbers = np.random.choice([1, 6, 7, 8], size=num_atoms).astype(
      np.int32
    )
    atoms_per_system = num_atoms // num_systems
    batch = np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32)
    if len(batch) < num_atoms:
      batch = np.concatenate(
        [batch, np.full(num_atoms - len(batch), num_systems - 1)]
      ).astype(np.int32)
    cutoff = 5.0
    edge_src, edge_dst = [], []
    for i in range(num_atoms):
      for j in range(num_atoms):
        if i != j and np.linalg.norm(positions[i] - positions[j]) < cutoff:
          edge_src.append(i)
          edge_dst.append(j)
    edge_index = np.array([edge_src, edge_dst], dtype=np.int32)
    edge_distance_vec = (
      positions[edge_index[0]] - positions[edge_index[1]]
    ).astype(np.float32)
    return {
      'positions': positions,
      'atomic_numbers': atomic_numbers,
      'batch': batch,
      'edge_index': edge_index,
      'edge_distance_vec': edge_distance_vec,
      'charge': np.zeros(num_systems, dtype=np.float32),
      'spin': np.zeros(num_systems, dtype=np.float32),
      'dataset': ['omat'] * num_systems,
      'natoms': np.array([atoms_per_system] * num_systems, dtype=np.int32),
    }

  def test_jax_forward_pass(self):
    """Production JAX UMABackbone produces finite embeddings."""
    config = UMAConfig(
      max_num_elements=100,
      sphere_channels=64,
      lmax=2,
      mmax=2,
      num_layers=2,
      hidden_channels=64,
      cutoff=5.0,
      edge_channels=64,
      num_distance_basis=128,
      norm_type='rms_norm_sh',
      act_type='gate',
      ff_type='grid',
      chg_spin_emb_type='pos_emb',
      dataset_list=['oc20', 'omol', 'omat', 'odac', 'omc'],
      use_dataset_embedding=True,
    )
    key = jax.random.PRNGKey(0)
    model = UMABackbone(config=config, key=key)
    data = self.create_numpy_test_data(num_atoms=10, num_systems=2)

    positions = jnp.array(data['positions'])
    atomic_numbers = jnp.array(data['atomic_numbers'])
    batch = jnp.array(data['batch'])
    edge_index = jnp.array(data['edge_index'])
    edge_distance_vec = jnp.array(data['edge_distance_vec'])
    charge = jnp.array(data['charge'])
    spin = jnp.array(data['spin'])
    dataset_idx = dataset_names_to_indices(data['dataset'], config.dataset_list)

    output = model(
      positions,
      atomic_numbers,
      batch,
      edge_index,
      edge_distance_vec,
      charge,
      spin,
      dataset_idx,
    )
    emb = output['node_embedding']
    self.assertEqual(emb.shape, (10, 9, 64))
    self.assertTrue(jnp.all(jnp.isfinite(emb)))

  def test_pytorch_forward_pass(self):
    """FairChem PyTorch eSCNMDBackbone produces finite embeddings."""
    if not pt_available():
      self.skipTest('torch not available')
    try:
      import sys

      fairchem_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        '..',
        '..',
        '..',
        'fairchem',
        'src',
      )
      if os.path.exists(fairchem_path):
        sys.path.insert(0, fairchem_path)
      from fairchem.core.models.uma.escn_md import eSCNMDBackbone
    except ImportError:
      self.skipTest('fairchem not available')

    import torch

    model = eSCNMDBackbone(
      max_num_elements=100,
      sphere_channels=64,
      lmax=2,
      mmax=2,
      num_layers=2,
      hidden_channels=64,
      cutoff=5.0,
      edge_channels=64,
      num_distance_basis=128,
      norm_type='rms_norm_sh',
      act_type='gate',
      ff_type='grid',
      chg_spin_emb_type='pos_emb',
      dataset_list=['oc20', 'omol', 'omat', 'odac', 'omc'],
      use_dataset_embedding=True,
      otf_graph=False,
    )
    model.eval()

    data = self.create_numpy_test_data(num_atoms=10, num_systems=2)

    class AtomicDataDict(dict):
      def __getattr__(self, k):
        try:
          return self[k]
        except KeyError:
          raise AttributeError(k)

      def __setattr__(self, k, v):
        self[k] = v

      def get(self, k, *a, **kw):
        return super().get(k, kw.get('default', a[0] if a else None))

    data_dict = AtomicDataDict(
      pos=torch.tensor(data['positions']),
      atomic_numbers=torch.tensor(data['atomic_numbers']).long(),
      atomic_numbers_full=torch.tensor(data['atomic_numbers']).long(),
      batch=torch.tensor(data['batch']).long(),
      batch_full=torch.tensor(data['batch']).long(),
      edge_index=torch.tensor(data['edge_index']).long(),
      cell_offsets=torch.zeros(data['edge_index'].shape[1], 3),
      cell=torch.eye(3).unsqueeze(0).expand(2, 3, 3),
      charge=torch.tensor(data['charge']),
      spin=torch.tensor(data['spin']),
      dataset=data['dataset'],
      natoms=torch.tensor(data['natoms']),
      nedges=torch.tensor(
        [data['edge_index'].shape[1] // 2, data['edge_index'].shape[1] // 2]
      ),
    )

    with torch.no_grad():
      output = model(data_dict)

    emb = output['node_embedding']
    self.assertEqual(tuple(emb.shape), (10, 9, 64))
    self.assertTrue(torch.all(torch.isfinite(emb)))


UMA_FULL_TO_M = (0, 5, 1, 3, 8, 6, 2, 4, 7)
UMA_BLOCKS = ((0, 1), (1, 3), (4, 5))


def uma_node_to_edge_reference(x, edge_index, wigner):
  sender = edge_index[0]
  receiver = edge_index[1]
  sender_valid = (sender >= 0) & (sender < x.shape[0])
  receiver_valid = (receiver >= 0) & (receiver < x.shape[0])
  sender = jnp.clip(sender, 0, x.shape[0] - 1)
  receiver = jnp.clip(receiver, 0, x.shape[0] - 1)
  xs = jnp.where(sender_valid[:, None, None], x[sender], 0.0)
  xt = jnp.where(receiver_valid[:, None, None], x[receiver], 0.0)
  ys = jnp.zeros((edge_index.shape[1], 9, x.shape[2]), dtype=x.dtype)
  yt = jnp.zeros_like(ys)

  for start, width in UMA_BLOCKS:
    for row in range(start, start + width):
      y_s = jnp.zeros((edge_index.shape[1], x.shape[2]), dtype=x.dtype)
      y_t = jnp.zeros_like(y_s)
      for col in range(start, start + width):
        w = wigner[:, row, col, None]
        y_s = y_s + w * xs[:, col, :]
        y_t = y_t + w * xt[:, col, :]
      ys = ys.at[:, UMA_FULL_TO_M[row], :].set(y_s)
      yt = yt.at[:, UMA_FULL_TO_M[row], :].set(y_t)
  return jnp.concatenate([ys, yt], axis=2)


def uma_edge_to_node_reference(messages, edge_index, wigner_inv, num_nodes):
  edge_messages = jnp.zeros(
    (edge_index.shape[1], 9, messages.shape[2]), dtype=messages.dtype
  )
  for start, width in UMA_BLOCKS:
    for row in range(start, start + width):
      y = jnp.zeros(
        (edge_index.shape[1], messages.shape[2]), dtype=messages.dtype
      )
      for col in range(start, start + width):
        y = (
          y + wigner_inv[:, row, col, None] * messages[:, UMA_FULL_TO_M[col], :]
        )
      edge_messages = edge_messages.at[:, row, :].set(y)

  receiver = edge_index[1]
  receiver_valid = (receiver >= 0) & (receiver < num_nodes)
  receiver = jnp.clip(receiver, 0, num_nodes - 1)
  edge_messages = jnp.where(receiver_valid[:, None, None], edge_messages, 0.0)
  return (
    jnp.zeros((num_nodes, 9, messages.shape[2]), dtype=messages.dtype)
    .at[receiver]
    .add(edge_messages)
  )


class UMAKernelBackendTest(test_util.JAXMDTestCase):
  """GPU-only UMA kernel backend parity checks."""

  def test_pallas_wigner_kernels_match_jax(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas GPU backend is not available.')
    from jax_md._nn.uma import kernels

    key = jax.random.PRNGKey(0)
    kx, kw, km = jax.random.split(key, 3)
    x = jax.random.normal(kx, (4, 9, 5))
    wigner0 = jax.random.normal(kw, (6, 9, 9))
    raw_wigner = (
      jnp.zeros_like(wigner0)
      .at[:, 0:1, 0:1]
      .set(wigner0[:, 0:1, 0:1])
      .at[:, 1:4, 1:4]
      .set(wigner0[:, 1:4, 1:4])
      .at[:, 4:9, 4:9]
      .set(wigner0[:, 4:9, 4:9])
    )
    messages = jax.random.normal(km, (6, 9, 5))
    edge_index = jnp.array(
      [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 1]], dtype=jnp.int32
    )
    to_m = create_coefficient_mapping(2, 2).to_m
    node_wigner = jnp.einsum('mk,ekj->emj', to_m, raw_wigner)
    edge_wigner = jnp.einsum('ejk,mk->ejm', raw_wigner, to_m)
    x_edge = jnp.concatenate([x[edge_index[0]], x[edge_index[1]]], axis=2)
    expected_n2e = jnp.einsum('emj,ejc->emc', node_wigner, x_edge)
    actual_n2e = kernels.node_to_edge_wigner_permute(x, edge_index, raw_wigner)
    self.assertAllClose(actual_n2e, expected_n2e, atol=TEST_TOL, rtol=TEST_TOL)

    edge_messages = jnp.einsum('ejm,emc->ejc', edge_wigner, messages)
    expected_e2n = (
      jnp.zeros((x.shape[0], 9, messages.shape[-1]))
      .at[edge_index[1]]
      .add(edge_messages)
    )
    actual_e2n = kernels.edge_to_node_wigner_inverse(
      messages, edge_index, raw_wigner, x.shape[0]
    )
    self.assertAllClose(actual_e2n, expected_e2n, atol=TEST_TOL, rtol=TEST_TOL)

  def test_pallas_wigner_kernel_transform_orders_match_jax(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas GPU backend is not available.')
    from jax_md._nn.uma import kernels

    key = jax.random.PRNGKey(23)
    kx, kw, km, ktx, ktw, ktm = jax.random.split(key, 6)
    num_nodes, num_edges, channels = 3, 2, 1
    x = 0.1 * jax.random.normal(kx, (num_nodes, 9, channels))
    wigner0 = 0.1 * jax.random.normal(kw, (num_edges, 9, 9))
    raw_wigner = (
      jnp.zeros_like(wigner0)
      .at[:, 0:1, 0:1]
      .set(wigner0[:, 0:1, 0:1])
      .at[:, 1:4, 1:4]
      .set(wigner0[:, 1:4, 1:4])
      .at[:, 4:9, 4:9]
      .set(wigner0[:, 4:9, 4:9])
    )
    messages = 0.1 * jax.random.normal(km, (num_edges, 9, channels))
    edge_index = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)

    tx = 0.01 * jax.random.normal(ktx, x.shape)
    tw = 0.01 * jax.random.normal(ktw, raw_wigner.shape)
    tm = 0.01 * jax.random.normal(ktm, messages.shape)

    def squared_loss(fn):
      return lambda *args: jnp.sum(fn(*args) ** 2)

    def forward_over_reverse_hvp(loss_fn, primals, tangents):
      grad_fn = jax.grad(loss_fn, argnums=tuple(range(len(primals))))
      return jax.jvp(lambda *args: grad_fn(*args), primals, tangents)[1]

    def reverse_over_reverse_hvp(loss_fn, primals, tangents):
      argnums = tuple(range(len(primals)))
      grad_fn = jax.grad(loss_fn, argnums=argnums)

      def grad_dot(*args):
        return sum(
          jnp.vdot(grad, tangent)
          for grad, tangent in zip(grad_fn(*args), tangents)
        )

      return jax.grad(grad_dot, argnums=argnums)(*primals)

    cases = (
      (
        squared_loss(
          lambda x_, w_: uma_node_to_edge_reference(x_, edge_index, w_)
        ),
        squared_loss(
          lambda x_, w_: kernels.node_to_edge_wigner_permute(x_, edge_index, w_)
        ),
        (x, raw_wigner),
        (tx, tw),
        (jnp.stack([tx, 2.0 * tx]), jnp.stack([tw, -tw])),
      ),
      (
        squared_loss(
          lambda messages_, w_: uma_edge_to_node_reference(
            messages_, edge_index, w_, num_nodes
          )
        ),
        squared_loss(
          lambda messages_, w_: kernels.edge_to_node_wigner_inverse(
            messages_, edge_index, w_, num_nodes
          )
        ),
        (messages, raw_wigner),
        (tm, tw),
        (jnp.stack([tm, 2.0 * tm]), jnp.stack([tw, -tw])),
      ),
    )

    for ref_loss, kernel_loss, primals, tangents, batched_tangents in cases:
      argnums = tuple(range(len(primals)))
      for transform in (
        lambda fn: jax.grad(fn, argnums=argnums),
        lambda fn: jax.hessian(fn, argnums=argnums),
        lambda fn: jax.jacfwd(jax.jacrev(fn, argnums=argnums), argnums=argnums),
        lambda fn: jax.jacrev(jax.jacfwd(fn, argnums=argnums), argnums=argnums),
      ):
        self.assertAllClose(
          transform(kernel_loss)(*primals),
          transform(ref_loss)(*primals),
          atol=TEST_TOL,
          rtol=TEST_TOL,
        )

      self.assertAllClose(
        forward_over_reverse_hvp(kernel_loss, primals, tangents),
        forward_over_reverse_hvp(ref_loss, primals, tangents),
        atol=TEST_TOL,
        rtol=TEST_TOL,
      )

      # Batched reverse-over-reverse HVPs exercise primitive batching rules.
      actual_batched_hvp = jax.vmap(
        lambda *batched_tangent: reverse_over_reverse_hvp(
          kernel_loss, primals, batched_tangent
        )
      )(*batched_tangents)
      expected_batched_hvp = jax.vmap(
        lambda *batched_tangent: reverse_over_reverse_hvp(
          ref_loss, primals, batched_tangent
        )
      )(*batched_tangents)
      self.assertAllClose(
        actual_batched_hvp,
        expected_batched_hvp,
        atol=TEST_TOL,
        rtol=TEST_TOL,
      )

  def test_pallas_wigner_hessian_ignores_padded_edges(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas GPU backend is not available.')
    from jax_md._nn.uma import kernels

    key = jax.random.PRNGKey(31)
    kx, kw = jax.random.split(key)
    x = 0.1 * jax.random.normal(kx, (3, 9, 1))
    wigner0 = 0.1 * jax.random.normal(kw, (4, 9, 9))
    raw_wigner = (
      jnp.zeros_like(wigner0)
      .at[:, 0:1, 0:1]
      .set(wigner0[:, 0:1, 0:1])
      .at[:, 1:4, 1:4]
      .set(wigner0[:, 1:4, 1:4])
      .at[:, 4:9, 4:9]
      .set(wigner0[:, 4:9, 4:9])
    )
    edge_index = jnp.array([[0, -1, 2, 3], [1, 2, 3, -1]], dtype=jnp.int32)

    def ref_loss(x_, w_):
      out = uma_node_to_edge_reference(x_, edge_index, w_)
      return jnp.sum(out * out)

    def kernel_loss(x_, w_):
      out = kernels.node_to_edge_wigner_permute(x_, edge_index, w_)
      return jnp.sum(out * out)

    self.assertAllClose(
      jax.hessian(kernel_loss, argnums=(0, 1))(x, raw_wigner),
      jax.hessian(ref_loss, argnums=(0, 1))(x, raw_wigner),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

  def test_pallas_edge_to_node_ignores_padded_edges(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas GPU backend is not available.')
    from jax_md._nn.uma import kernels

    key = jax.random.PRNGKey(37)
    km = jax.random.split(key, 1)[0]
    num_nodes, num_edges, channels = 4, 4, 2
    messages = jax.random.normal(km, (num_edges, 9, channels))
    raw_wigner = jnp.broadcast_to(jnp.eye(9), (num_edges, 9, 9))
    edge_index = jnp.array([[0, 1, 2, 3], [1, -1, 3, 10]], dtype=jnp.int32)

    expected = uma_edge_to_node_reference(
      messages, edge_index, raw_wigner, num_nodes
    )
    actual = kernels.edge_to_node_wigner_inverse(
      messages, edge_index, raw_wigner, num_nodes
    )
    self.assertAllClose(actual, expected, atol=TEST_TOL, rtol=TEST_TOL)

    def kernel_loss(messages_, w_):
      out = kernels.edge_to_node_wigner_inverse(
        messages_, edge_index, w_, num_nodes
      )
      return jnp.sum(out * out)

    def ref_loss(messages_, w_):
      out = uma_edge_to_node_reference(messages_, edge_index, w_, num_nodes)
      return jnp.sum(out * out)

    self.assertAllClose(
      jax.grad(kernel_loss, argnums=(0, 1))(messages, raw_wigner),
      jax.grad(ref_loss, argnums=(0, 1))(messages, raw_wigner),
      atol=TEST_TOL,
      rtol=TEST_TOL,
    )

  def test_pallas_wigner_zero_edge_gradients(self):
    from jax_md._nn.uma import kernels

    num_nodes, channels = 3, 2
    x = jnp.ones((num_nodes, 9, channels), dtype=jnp.float32)
    messages = jnp.ones((0, 9, channels), dtype=jnp.float32)
    raw_wigner = jnp.zeros((0, 9, 9), dtype=jnp.float32)
    edge_index = jnp.zeros((2, 0), dtype=jnp.int32)

    def n2e_loss(x_, w_):
      out = kernels.node_to_edge_wigner_permute(x_, edge_index, w_)
      self.assertEqual(out.shape, (0, 9, 2 * channels))
      return jnp.sum(out * out)

    def e2n_loss(messages_, w_):
      out = kernels.edge_to_node_wigner_inverse(
        messages_, edge_index, w_, num_nodes
      )
      self.assertEqual(out.shape, (num_nodes, 9, channels))
      return jnp.sum(out * out)

    gx, gw_n2e = jax.grad(n2e_loss, argnums=(0, 1))(x, raw_wigner)
    gm, gw_e2n = jax.grad(e2n_loss, argnums=(0, 1))(messages, raw_wigner)
    self.assertAllClose(gx, jnp.zeros_like(x), atol=TEST_TOL, rtol=TEST_TOL)
    self.assertEqual(gw_n2e.shape, raw_wigner.shape)
    self.assertEqual(gm.shape, messages.shape)
    self.assertEqual(gw_e2n.shape, raw_wigner.shape)

  def test_edgewise_fallback_ignores_invalid_targets(self):
    from jax_md._nn.uma.blocks import Edgewise

    key = jax.random.PRNGKey(41)
    num_nodes, channels = 4, 2
    x = jax.random.normal(key, (num_nodes, 9, channels))
    x_shifted = x.at[-1].add(10.0)
    x_edge = jnp.ones((3, 4), dtype=x.dtype)
    edge_index = jnp.array([[0, 1, -1], [1, -1, 2]], dtype=jnp.int32)
    wigner = jnp.broadcast_to(jnp.eye(9, dtype=x.dtype), (3, 9, 9))
    edge_envelope = jnp.ones((3, 1, 1), dtype=x.dtype)
    mapping = create_coefficient_mapping(2, 2)
    edgewise = Edgewise(
      sphere_channels=channels,
      hidden_channels=2,
      lmax=2,
      mmax=2,
      edge_channels_list=[4],
      m_size=list(mapping.m_size),
      use_kernels=False,
      key=key,
    )

    actual = edgewise(
      x,
      x_edge,
      edge_index,
      wigner,
      wigner,
      None,
      None,
      edge_envelope,
    )
    actual_shifted = edgewise(
      x_shifted,
      x_edge,
      edge_index,
      wigner,
      wigner,
      None,
      None,
      edge_envelope,
    )
    self.assertAllClose(
      actual[3], jnp.zeros_like(actual[3]), atol=TEST_TOL, rtol=TEST_TOL
    )
    self.assertAllClose(
      actual[2], actual_shifted[2], atol=TEST_TOL, rtol=TEST_TOL
    )

  def test_edgewise_moe_fallback_ignores_invalid_targets(self):
    from jax_md._nn.uma.model_moe import EdgewiseMoE

    key = jax.random.PRNGKey(43)
    num_nodes, channels = 4, 2
    x = jax.random.normal(key, (num_nodes, 9, channels))
    x_shifted = x.at[-1].add(10.0)
    x_edge = jnp.ones((3, 4), dtype=x.dtype)
    edge_index = jnp.array([[0, 1, -1], [1, -1, 2]], dtype=jnp.int32)
    wigner = jnp.broadcast_to(jnp.eye(9, dtype=x.dtype), (3, 9, 9))
    edge_envelope = jnp.ones((3, 1, 1), dtype=x.dtype)
    expert_coefficients = jnp.array([[0.25, 0.75]], dtype=x.dtype)
    edge_batch = jnp.zeros((3,), dtype=jnp.int32)
    mapping = create_coefficient_mapping(2, 2)
    edgewise = EdgewiseMoE(
      sphere_channels=channels,
      hidden_channels=2,
      lmax=2,
      mmax=2,
      edge_channels_list=[4],
      m_size=list(mapping.m_size),
      num_experts=2,
      use_kernels=False,
      key=key,
    )

    actual = edgewise(
      x,
      x_edge,
      edge_index,
      wigner,
      wigner,
      None,
      None,
      edge_envelope,
      expert_coefficients,
      edge_batch,
    )
    actual_shifted = edgewise(
      x_shifted,
      x_edge,
      edge_index,
      wigner,
      wigner,
      None,
      None,
      edge_envelope,
      expert_coefficients,
      edge_batch,
    )
    self.assertAllClose(
      actual[3], jnp.zeros_like(actual[3]), atol=TEST_TOL, rtol=TEST_TOL
    )
    self.assertAllClose(
      actual[2], actual_shifted[2], atol=TEST_TOL, rtol=TEST_TOL
    )

  def test_pallas_backbone_matches_jax(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas GPU backend is not available.')
    inputs = (
      jnp.array(
        [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.1, 0.8, 0.1]],
        dtype=jnp.float32,
      ),
      jnp.array([1, 6, 8], dtype=jnp.int32),
      jnp.zeros(3, dtype=jnp.int32),
      jnp.array([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=jnp.int32),
      jnp.array(
        [
          [-0.7, -0.1, 0.0],
          [0.6, -0.7, 0.0],
          [0.1, 0.8, 0.1],
          [-0.1, -0.8, -0.1],
        ],
        dtype=jnp.float32,
      ),
      jnp.zeros(1, dtype=jnp.float32),
      jnp.zeros(1, dtype=jnp.float32),
      jnp.zeros(1, dtype=jnp.int32),
    )
    cfg = UMAConfig(
      sphere_channels=4,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=4,
      cutoff=3.0,
      edge_channels=4,
      num_distance_basis=4,
      ff_type='spectral',
      use_dataset_embedding=False,
      use_kernels=False,
    )
    key = jax.random.PRNGKey(17)
    jax_model = UMABackbone(config=cfg, key=key)
    pallas_model = UMABackbone(config=replace(cfg, use_kernels=True), key=key)
    # use_kernels is static, so the leaves transplant one to one
    pallas_model = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(pallas_model),
      jax.tree_util.tree_leaves(jax_model),
    )
    jax_full = jax_model(*inputs)
    pallas_full = pallas_model(*inputs)
    jax_out = jax_full['node_embedding']
    pallas_out = pallas_full['node_embedding']
    self.assertAllClose(pallas_out, jax_out, atol=TEST_TOL, rtol=TEST_TOL)

  def test_pallas_moe_forces_match_jax(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas GPU backend is not available.')
    cfg = UMAMoEConfig(
      sphere_channels=4,
      lmax=2,
      mmax=2,
      num_layers=2,
      hidden_channels=4,
      cutoff=3.0,
      edge_channels=4,
      num_distance_basis=4,
      ff_type='spectral',
      use_dataset_embedding=False,
      use_composition_embedding=False,
      num_experts=2,
      routing_hidden_channels=4,
      use_kernels=False,
    )
    key = jax.random.PRNGKey(17)
    jax_model = UMAMoEBackbone(config=cfg, key=key)
    pallas_model = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(
        UMAMoEBackbone(config=replace(cfg, use_kernels=True), key=key)
      ),
      jax.tree_util.tree_leaves(jax_model),
    )
    position = jnp.array(
      [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.1, 0.8, 0.1]], dtype=jnp.float32
    )
    atomic_numbers = jnp.array([1, 6, 8], dtype=jnp.int32)
    batch = jnp.zeros(3, dtype=jnp.int32)
    # the last two edges have an invalid sender, then an invalid receiver
    edge_index = jnp.array(
      [[0, 1, 2, 0, -1, 0], [1, 2, 0, 2, 1, -1]], jnp.int32
    )
    zero = jnp.zeros(1, dtype=jnp.int32)

    def energy(model, position):
      edge_vec = position[edge_index[0]] - position[edge_index[1]]
      out = model(
        position, atomic_numbers, batch, edge_index, edge_vec, zero, zero, zero
      )
      return jnp.sum(out['node_embedding'])

    jax_forces = jax.grad(lambda p: energy(jax_model, p))(position)
    pallas_forces = jax.grad(lambda p: energy(pallas_model, p))(position)
    self.assertAllClose(pallas_forces, jax_forces, atol=TEST_TOL, rtol=TEST_TOL)

  def test_pallas_wigner_vjp_transpose_matches_forward(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('Pallas GPU backend is not available.')
    from jax_md._nn.uma import kernels

    num_nodes, num_edges, channels = 5, 7, 8
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (num_nodes, 9, channels), dtype=jnp.float32)
    wigner = jax.random.normal(key, (num_edges, 9, 9), dtype=jnp.float32)
    messages = jax.random.normal(
      key, (num_edges, 9, channels), dtype=jnp.float32
    )
    edge_index = jnp.array(
      [[0, 1, 2, 3, 4, 0, -1], [1, 2, 3, 4, 0, 2, -1]], jnp.int32
    )

    def n2e(x_):
      return kernels.node_to_edge_wigner_permute(x_, edge_index, wigner)

    def e2n(messages_):
      return kernels.edge_to_node_wigner_inverse(
        messages_, edge_index, wigner, num_nodes
      )

    # the transpose of the vjp is the forward map, so it must reproduce it
    for fn, v in ((n2e, x), (e2n, messages)):
      out, vjp_fn = jax.vjp(fn, v)
      back = jax.linear_transpose(lambda g: vjp_fn(g)[0], out)(v)[0]
      self.assertAllClose(back, fn(v), atol=TEST_TOL, rtol=TEST_TOL)


class UMAKernelBatchingTest(test_util.JAXMDTestCase):
  """Traces the batching rules of the UMA Pallas primitives on any backend."""

  def test_vmap_of_primitives_and_their_derivatives(self):
    from jax_md._nn.uma import kernels

    weights = jnp.ones((2, 5, 4))
    sizes = jnp.array([3, 3], dtype=jnp.int32)
    starts = jnp.array([0, 3], dtype=jnp.int32)
    edge_index = jnp.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=jnp.int32)
    wigner = jnp.ones((5, 9, 9))
    cases = [
      (
        lambda a: kernels.segment_mm_p.bind(
          a, weights, sizes, starts, max_size=4
        ),
        jnp.ones((3, 6, 4)),
        (3, 6, 5),
      ),
      (
        lambda a: kernels.node_to_edge_p.bind(a, edge_index, wigner),
        jnp.ones((3, 6, 9, 4)),
        (3, 5, 9, 8),
      ),
      (
        lambda a: kernels.edge_to_node_p.bind(
          a, edge_index, wigner, num_nodes=6
        ),
        jnp.ones((3, 5, 9, 4)),
        (3, 6, 9, 4),
      ),
    ]
    for fn, arg, shape in cases:
      out = jax.eval_shape(jax.vmap(fn), arg)
      tangent = jax.eval_shape(
        jax.vmap(lambda a: jax.jvp(fn, (a,), (a,))[1]), arg
      )
      grad = jax.eval_shape(jax.vmap(jax.grad(lambda a: fn(a).sum())), arg)
      self.assertEqual(out.shape, shape)
      self.assertEqual(tangent.shape, shape)
      self.assertEqual(grad.shape, arg.shape)


if __name__ == '__main__':
  absltest.main()
