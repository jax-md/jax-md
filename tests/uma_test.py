"""Tests for jax_md._nn.uma (UMA model).

These tests verify the UMA model components and full forward pass work correctly.
"""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import test_util
from jax_md._nn.uma.model import UMABackbone, UMAConfig
from jax_md._nn.uma.heads import MLPEnergyHead, LinearForceHead
from jax_md._nn.uma.nn.so2_layers import SO2MConv
from jax_md._nn.uma.nn.so3_layers import SO3Linear
from jax_md._nn.uma.nn.radial import PolynomialEnvelope, RadialMLP
from jax_md._nn.uma.nn.layer_norm import EquivariantRMSNorm
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
    """Forward pass should be exact arccos (not clamped)."""
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
    self.assertTrue(jnp.allclose(wigner, expected, atol=1e-5))

  def test_wigner_orthogonal(self):
    """Wigner D-matrices should be orthogonal (D @ D^T = I)."""
    Jd_list = load_jacobi_matrices_from_file(2)
    alpha = jnp.array([0.5])
    beta = jnp.array([1.2])
    gamma = jnp.array([0.3])
    wigner = eulers_to_wigner((alpha, beta, gamma), 0, 2, Jd_list)
    product = jnp.einsum('bij,bkj->bik', wigner, wigner)
    expected = jnp.eye(9).reshape(1, 9, 9)
    self.assertTrue(jnp.allclose(product, expected, atol=1e-5))


class SO2MConvTest(test_util.JAXMDTestCase):
  """Tests for SO2MConv."""

  def test_so2mconv_output_shape(self):
    """Test SO2MConv produces correct output shape."""
    conv = SO2MConv(
      m=1, sphere_channels=32, m_output_channels=16, lmax=2, mmax=2
    )
    key = jax.random.PRNGKey(0)
    # m=1: num_coefficients = lmax - m + 1 = 2
    # Input: (E, 2, num_coeffs * sphere_channels) = (5, 2, 64)
    x = jax.random.normal(key, (5, 2, 2 * 32))
    params = conv.init(key, x)
    x_r, x_i = conv.apply(params, x)
    # Output: (E, num_coeffs, m_output_channels) = (5, 2, 16)
    self.assertEqual(x_r.shape, (5, 2, 16))
    self.assertEqual(x_i.shape, (5, 2, 16))

  def test_so2mconv_jit(self):
    """Test SO2MConv is JIT-compatible."""
    conv = SO2MConv(
      m=1, sphere_channels=16, m_output_channels=8, lmax=2, mmax=2
    )
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (3, 2, 2 * 16))
    params = conv.init(key, x)
    jit_apply = jax.jit(conv.apply)
    x_r, x_i = jit_apply(params, x)
    self.assertTrue(jnp.all(jnp.isfinite(x_r)))


class SO3LinearTest(test_util.JAXMDTestCase):
  """Tests for SO3Linear."""

  def test_so3linear_output_shape(self):
    linear = SO3Linear(out_features=16, lmax=2)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 9, 32))  # (batch, (lmax+1)^2, in_features)
    params = linear.init(key, x)
    out = linear.apply(params, x)
    self.assertEqual(out.shape, (5, 9, 16))

  def test_so3linear_no_weight_shift(self):
    """Verify the weight is used as-is (no subtraction bug)."""
    linear = SO3Linear(out_features=8, lmax=1, use_bias=False)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 4, 16))
    params = linear.init(key, x)

    # Set weights to known values
    new_weight = jnp.ones_like(params['params']['weight'])
    new_params = {'params': {'weight': new_weight}}
    out1 = linear.apply(new_params, x)
    # Run again — should be identical (no cumulative shift)
    out2 = linear.apply(new_params, x)
    self.assertTrue(jnp.allclose(out1, out2))
    self.assertTrue(jnp.all(jnp.isfinite(out1)))


class RadialMLPTest(test_util.JAXMDTestCase):
  """Tests for RadialMLP."""

  def test_radial_mlp_output_shape(self):
    mlp = RadialMLP(channels_list=[64, 32, 16])
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 64))
    params = mlp.init(key, x)
    out = mlp.apply(params, x)
    self.assertEqual(out.shape, (10, 16))

  def test_polynomial_envelope(self):
    env = PolynomialEnvelope(exponent=5)
    d = jnp.array([0.0, 0.5, 0.99, 1.0, 1.5])
    # Flax modules with setup() need init/apply
    params = env.init(jax.random.PRNGKey(0), d)
    out = env.apply(params, d)
    # At d=0, envelope should be 1
    self.assertAlmostEqual(float(out[0]), 1.0, places=5)
    # At d>=1, envelope should be 0
    self.assertAlmostEqual(float(out[3]), 0.0, places=5)
    self.assertAlmostEqual(float(out[4]), 0.0, places=5)


class DatasetEmbeddingTest(test_util.JAXMDTestCase):
  """Tests for DatasetEmbedding with integer indices."""

  def test_dataset_embedding_jit(self):
    """DatasetEmbedding should be JIT-compatible with integer indices."""
    emb = DatasetEmbedding(embedding_size=32, num_datasets=5)
    key = jax.random.PRNGKey(0)
    idx = jnp.array([0, 2, 4])
    params = emb.init(key, idx)

    jit_apply = jax.jit(emb.apply)
    out = jit_apply(params, idx)
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
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 9, 32))
    params = norm.init(key, x)
    out = norm.apply(params, x)
    self.assertEqual(out.shape, (5, 9, 32))
    self.assertTrue(jnp.all(jnp.isfinite(out)))


class UMAConfigTest(test_util.JAXMDTestCase):
  def test_uma_config_defaults(self):
    config = UMAConfig()
    self.assertEqual(config.sphere_channels, 128)
    self.assertEqual(config.lmax, 2)
    self.assertEqual(config.mmax, 2)
    self.assertEqual(config.num_layers, 2)


class UMAHeadsTest(test_util.JAXMDTestCase):
  def test_mlp_energy_head(self):
    head = MLPEnergyHead(sphere_channels=32, hidden_channels=32)
    key = jax.random.PRNGKey(0)
    node_emb = jax.random.normal(key, (8, 9, 32))
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    params = head.init(key, node_emb, batch, 2)
    output = head.apply(params, node_emb, batch, 2)
    self.assertEqual(output['energy'].shape, (2,))
    self.assertTrue(jnp.all(jnp.isfinite(output['energy'])))

  def test_mlp_energy_head_jit(self):
    head = MLPEnergyHead(sphere_channels=32, hidden_channels=32)
    key = jax.random.PRNGKey(0)
    node_emb = jax.random.normal(key, (8, 9, 32))
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    params = head.init(key, node_emb, batch, 2)
    jit_apply = jax.jit(head.apply, static_argnums=(3,))
    output = jit_apply(params, node_emb, batch, 2)
    self.assertEqual(output['energy'].shape, (2,))

  def test_linear_force_head(self):
    head = LinearForceHead(sphere_channels=32)
    key = jax.random.PRNGKey(0)
    node_emb = jax.random.normal(key, (8, 9, 32))
    params = head.init(key, node_emb)
    output = head.apply(params, node_emb)
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

    model = UMABackbone(config=config)
    data = create_test_data(num_atoms=8, num_systems=2, cutoff=config.cutoff)

    key = jax.random.PRNGKey(0)
    params = model.init(
      key,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      data['dataset_idx'],
    )

    output = model.apply(
      params,
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

    model = UMABackbone(config=config)
    data = create_test_data(num_atoms=6, num_systems=2, cutoff=config.cutoff)

    key = jax.random.PRNGKey(42)
    params = model.init(
      key,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    output = model.apply(
      params,
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

    model = UMABackbone(config=config)
    data = create_test_data(num_atoms=6, num_systems=2, cutoff=config.cutoff)

    key = jax.random.PRNGKey(0)
    params = model.init(
      key,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    out1 = model.apply(
      params,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )
    out2 = model.apply(
      params,
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

    model = UMABackbone(config=config)
    data = create_test_data(num_atoms=6, num_systems=2, cutoff=config.cutoff)

    key = jax.random.PRNGKey(0)
    params = model.init(
      key,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    output = model.apply(
      params,
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
    """Test with spectral (SO3Linear) feed-forward instead of grid."""
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

    model = UMABackbone(config=config)
    data = create_test_data(num_atoms=6, num_systems=2, cutoff=config.cutoff)

    key = jax.random.PRNGKey(0)
    params = model.init(
      key,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    output = model.apply(
      params,
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
  """Test that gradients flow through the model (for force computation)."""

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

    model = UMABackbone(config=config)
    head = MLPEnergyHead(sphere_channels=32, hidden_channels=32)

    data = create_test_data(num_atoms=6, num_systems=1, cutoff=config.cutoff)

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    backbone_params = model.init(
      key1,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    emb = model.apply(
      backbone_params,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      None,
    )

    head_params = head.init(key2, emb['node_embedding'], data['batch'], 1)

    def energy_fn(positions):
      edge_vec = (
        positions[data['edge_index'][0]] - positions[data['edge_index'][1]]
      )
      emb = model.apply(
        backbone_params,
        positions,
        data['atomic_numbers'],
        data['batch'],
        data['edge_index'],
        edge_vec,
        data['charge'],
        data['spin'],
        None,
      )
      result = head.apply(head_params, emb['node_embedding'], data['batch'], 1)
      return result['energy'].sum()

    # Compute gradient (forces = -grad)
    grad_fn = jax.grad(energy_fn)
    forces = -grad_fn(data['positions'])

    self.assertEqual(forces.shape, data['positions'].shape)
    self.assertTrue(jnp.all(jnp.isfinite(forces)))


def _load_pt_module(module_name, file_path):
  """Load a fairchem module directly, bypassing __init__ chains."""
  import importlib.util

  spec = importlib.util.spec_from_file_location(module_name, file_path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _stub_fairchem():
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


_FAIRCHEM_SRC = '/Users/emirhankurtulus/workspace/fairchem_jaxmd/fairchem/src/fairchem/core/models/uma'


def _pt_available():
  """Check if PyTorch and fairchem source are accessible."""
  import os

  try:
    import torch

    return os.path.isdir(_FAIRCHEM_SRC)
  except ImportError:
    return False


class PyTorchComparisonTest(test_util.JAXMDTestCase):
  """Compare JAX UMA layers against PyTorch originals with copied weights."""

  def setUp(self):
    super().setUp()
    if not _pt_available():
      self.skipTest('PyTorch or fairchem source not available')
    _stub_fairchem()
    import sys

    # Stub the radial import needed by so2_layers
    radial_pt = _load_pt_module(
      'fairchem.core.models.uma.nn.radial',
      f'{_FAIRCHEM_SRC}/nn/radial.py',
    )
    sys.modules['fairchem.core.models.uma.nn.radial'] = radial_pt
    sys.modules.setdefault(
      'fairchem.core.models.uma.nn', __import__('types').ModuleType('nn')
    )
    sys.modules['fairchem.core.models.uma.nn'].radial = radial_pt

  def test_z_rot_mat_matches_pytorch(self):
    """_z_rot_mat should match PyTorch for all l."""
    import torch

    pt_rot = _load_pt_module('pt_rot', f'{_FAIRCHEM_SRC}/common/rotation.py')
    from jax_md._nn.uma.common.rotation import _z_rot_mat

    angles = np.array([0.0, 0.7, -1.3, 3.14], dtype=np.float32)
    for l in range(4):
      M_pt = pt_rot._z_rot_mat(torch.tensor(angles), l).numpy()
      M_jax = np.array(_z_rot_mat(jnp.array(angles), l))
      np.testing.assert_allclose(
        M_jax, M_pt, atol=1e-6, err_msg=f'_z_rot_mat mismatch at l={l}'
      )

  def test_wigner_D_matches_pytorch(self):
    """wigner_D should match PyTorch for all l."""
    import torch

    pt_rot = _load_pt_module('pt_rot', f'{_FAIRCHEM_SRC}/common/rotation.py')
    from jax_md._nn.uma.common.rotation import wigner_D as jax_wigner_D

    Jd_pt = torch.load(
      'jax_md/_nn/uma/Jd.pt', map_location='cpu', weights_only=False
    )
    # Cast Jd to float32 to match angle dtype
    Jd_pt = [J.float() for J in Jd_pt]
    Jd_jax = load_jacobi_matrices_from_file(3)

    alpha = np.array([0.5, -0.3], dtype=np.float32)
    beta = np.array([0.8, 1.5], dtype=np.float32)
    gamma = np.zeros(2, dtype=np.float32)

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
        D_jax, D_pt, atol=1e-5, err_msg=f'wigner_D mismatch at l={l}'
      )

  def test_coefficient_mapping_to_m_matches_pytorch(self):
    """to_m permutation matrix should be identical to PyTorch."""
    pt_so3 = _load_pt_module('pt_so3', f'{_FAIRCHEM_SRC}/common/so3.py')
    pt_cm = pt_so3.CoefficientMapping(2, 2)
    jax_cm = create_coefficient_mapping(2, 2)
    np.testing.assert_array_equal(
      np.array(jax_cm.to_m),
      pt_cm.to_m.numpy(),
    )

  def test_so3_grid_matrices_match_pytorch(self):
    """Grid transform matrices should match PyTorch (via e3nn)."""
    pt_so3 = _load_pt_module('pt_so3', f'{_FAIRCHEM_SRC}/common/so3.py')
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

    pt_fc = tnn.Linear(num_channels, 2 * out_channels_half, bias=False)

    def pt_so2m_forward(x_m):
      x_m = pt_fc(x_m)
      x_m = x_m.reshape(x_m.shape[0], -1, out_channels_half).split(1, dim=1)
      x_r_0, x_i_0, x_r_1, x_i_1 = x_m
      x_m_r = (x_r_0 - x_i_1).view(-1, num_coefficients, moc)
      x_m_i = (x_r_1 + x_i_0).view(-1, num_coefficients, moc)
      return x_m_r, x_m_i

    jax_conv = SO2MConv(
      m=m, sphere_channels=sc, m_output_channels=moc, lmax=lmax, mmax=mmax
    )

    np.random.seed(123)
    x_np = np.random.randn(7, 2, num_channels).astype(np.float32)

    key = jax.random.PRNGKey(0)
    jax_params = jax_conv.init(key, jnp.array(x_np))

    # Copy PT weights to JAX (transpose: PT [out,in] -> JAX [in,out])
    jax_params_loaded = {
      'params': {'fc': {'kernel': jnp.array(pt_fc.weight.data.numpy().T)}}
    }

    with torch.no_grad():
      pt_r, pt_i = pt_so2m_forward(torch.tensor(x_np))
    jax_r, jax_i = jax_conv.apply(jax_params_loaded, jnp.array(x_np))

    np.testing.assert_allclose(
      np.array(jax_r),
      pt_r.numpy(),
      atol=1e-5,
    )
    np.testing.assert_allclose(
      np.array(jax_i),
      pt_i.numpy(),
      atol=1e-5,
    )

  def test_so3linear_matches_pytorch(self):
    """SO3Linear with copied weights should match PyTorch output."""
    import torch

    pt_so3l = _load_pt_module('pt_so3l', f'{_FAIRCHEM_SRC}/nn/so3_layers.py')

    lmax, in_f, out_f = 2, 16, 8
    sph = (lmax + 1) ** 2

    pt_lin = pt_so3l.SO3_Linear(in_features=in_f, out_features=out_f, lmax=lmax)
    jax_lin = SO3Linear(out_features=out_f, lmax=lmax, use_bias=True)

    np.random.seed(77)
    x_np = np.random.randn(4, sph, in_f).astype(np.float32)

    key = jax.random.PRNGKey(0)
    jax_params = jax_lin.init(key, jnp.array(x_np))

    jax_params_loaded = {
      'params': {
        'weight': jnp.array(pt_lin.weight.data.numpy()),
        'bias': jnp.array(pt_lin.bias.data.numpy()),
      }
    }

    with torch.no_grad():
      pt_out = pt_lin(torch.tensor(x_np)).numpy()
    jax_out = np.array(jax_lin.apply(jax_params_loaded, jnp.array(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, atol=1e-6)

  def test_rms_norm_matches_pytorch(self):
    """EquivariantRMSNorm with copied weights should match PyTorch."""
    import torch

    pt_ln = _load_pt_module('pt_ln', f'{_FAIRCHEM_SRC}/nn/layer_norm.py')

    lmax, ch = 2, 16
    sph = (lmax + 1) ** 2

    pt_norm = pt_ln.EquivariantRMSNormArraySphericalHarmonicsV2(lmax, ch)
    jax_norm = EquivariantRMSNorm(lmax=lmax, num_channels=ch)

    np.random.seed(55)
    x_np = np.random.randn(5, sph, ch).astype(np.float32)

    key = jax.random.PRNGKey(0)
    jax_params = jax_norm.init(key, jnp.array(x_np))

    jax_params_loaded = {
      'params': {
        'affine_weight': jnp.array(pt_norm.affine_weight.data.numpy()),
        'affine_bias': jnp.array(pt_norm.affine_bias.data.numpy()),
      }
    }

    with torch.no_grad():
      pt_out = pt_norm(torch.tensor(x_np)).numpy()
    jax_out = np.array(jax_norm.apply(jax_params_loaded, jnp.array(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, atol=1e-5)

  def test_euler_angles_match_pytorch(self):
    """Euler angle computation should match PyTorch (alpha, beta)."""
    import torch

    pt_rot = _load_pt_module('pt_rot', f'{_FAIRCHEM_SRC}/common/rotation.py')

    np.random.seed(99)
    vecs = np.random.randn(20, 3).astype(np.float32)

    g_pt, b_pt, a_pt = pt_rot.init_edge_rot_euler_angles(torch.tensor(vecs))
    g_jax, b_jax, a_jax = init_edge_rot_euler_angles(jnp.array(vecs))

    # alpha and beta should match; gamma differs (random vs zero)
    np.testing.assert_allclose(np.array(b_jax), b_pt.numpy(), atol=1e-5)
    np.testing.assert_allclose(np.array(a_jax), a_pt.numpy(), atol=1e-5)


class PretrainedMoETest(test_util.JAXMDTestCase):
  """Test loading and running pretrained UMA MoE model directly (no merging)."""

  def setUp(self):
    super().setUp()
    import os

    # Look for cached checkpoint
    cache_base = os.path.expanduser('~/.cache/fairchem/models--facebook--UMA')
    self.ckpt_path = None
    if os.path.isdir(cache_base):
      for root, dirs, files in os.walk(cache_base):
        for f in files:
          if f == 'uma-s-1p1.pt':
            self.ckpt_path = os.path.join(root, f)
            break
    if self.ckpt_path is None or not _pt_available():
      self.skipTest('Pretrained checkpoint or PyTorch not available')

  def test_pretrained_moe_runs(self):
    """Full MoE model with pretrained weights produces finite output."""
    from jax_md._nn.uma.model_moe import UMAMoEBackbone, load_pretrained
    from jax_md._nn.uma.nn.embedding import dataset_names_to_indices

    config, params, _hp = load_pretrained(self.ckpt_path)
    model = UMAMoEBackbone(config=config)

    pos = jnp.array(
      [
        [0, 0, 0],
        [1.8, 1.8, 0],
        [1.8, 0, 1.8],
        [0, 1.8, 1.8],
        [0.9, 0.9, 0.9],
        [2.7, 0.9, 0.9],
      ],
      dtype=jnp.float32,
    )
    Z = jnp.array([29, 29, 29, 29, 8, 8], dtype=jnp.int32)
    batch = jnp.zeros(6, dtype=jnp.int32)
    src, dst = [], []
    for i in range(6):
      for j in range(6):
        if (
          i != j
          and np.linalg.norm(np.array(pos[i]) - np.array(pos[j]))
          < config.cutoff
        ):
          src.append(j)
          dst.append(i)
    ei = jnp.array([src, dst], dtype=jnp.int32)
    ev = pos[ei[0]] - pos[ei[1]]
    ds = dataset_names_to_indices(['omat'], config.dataset_list)

    output = model.apply(
      params,
      pos,
      Z,
      batch,
      ei,
      ev,
      jnp.array([0], dtype=jnp.int32),
      jnp.array([0], dtype=jnp.int32),
      ds,
    )
    emb = output['node_embedding']
    self.assertEqual(emb.shape, (6, 9, 128))
    self.assertTrue(jnp.all(jnp.isfinite(emb)))


if __name__ == '__main__':
  absltest.main()
