"""Tests for jax_md._nn.uma (UMA model).

These tests verify the UMA model components and full forward pass work correctly.
"""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import test_util
from jax_md._nn import uma


def create_test_data(num_atoms=10, num_systems=2, cutoff=5.0, seed=42):
  """Create synthetic test data for UMA model testing."""
  np.random.seed(seed)

  # Positions
  positions = np.random.randn(num_atoms, 3).astype(np.float32) * 2.0

  # Atomic numbers (H, C, N, O)
  atomic_numbers = np.random.choice([1, 6, 7, 8], size=num_atoms).astype(
    np.int32
  )

  # Batch indices
  atoms_per_system = num_atoms // num_systems
  batch = np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32)
  if len(batch) < num_atoms:
    batch = np.concatenate(
      [batch, np.full(num_atoms - len(batch), num_systems - 1)]
    ).astype(np.int32)

  # Build edges (within cutoff)
  edge_src = []
  edge_dst = []
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

  # System-level properties
  charge = np.zeros(num_systems, dtype=np.float32)
  spin = np.zeros(num_systems, dtype=np.float32)
  dataset = ['omat'] * num_systems

  return {
    'positions': jnp.array(positions),
    'atomic_numbers': jnp.array(atomic_numbers),
    'batch': jnp.array(batch),
    'edge_index': jnp.array(edge_index),
    'edge_distance_vec': jnp.array(edge_distance_vec),
    'charge': jnp.array(charge),
    'spin': jnp.array(spin),
    'dataset': dataset,
  }


class UMAConfigTest(test_util.JAXMDTestCase):
  """Tests for UMA configuration."""

  def test_uma_config_defaults(self):
    """Test that UMAConfig can be instantiated with defaults."""
    config = uma.UMAConfig()
    self.assertEqual(config.sphere_channels, 128)
    self.assertEqual(config.lmax, 2)
    self.assertEqual(config.mmax, 2)
    self.assertEqual(config.num_layers, 2)
    self.assertEqual(config.hidden_channels, 128)
    self.assertEqual(config.cutoff, 5.0)

  def test_uma_config_custom(self):
    """Test UMAConfig with custom values."""
    config = uma.UMAConfig(
      sphere_channels=64,
      lmax=3,
      mmax=2,
      num_layers=4,
      hidden_channels=256,
      cutoff=6.0,
    )
    self.assertEqual(config.sphere_channels, 64)
    self.assertEqual(config.lmax, 3)
    self.assertEqual(config.mmax, 2)
    self.assertEqual(config.num_layers, 4)


class UMAHeadsTest(test_util.JAXMDTestCase):
  """Tests for UMA prediction heads."""

  def test_mlp_energy_head(self):
    """Test MLPEnergyHead module."""
    sphere_channels = 32
    hidden_channels = 32
    num_atoms = 8
    num_systems = 2
    lmax = 2
    sph_size = (lmax + 1) ** 2

    head = uma.MLPEnergyHead(
      sphere_channels=sphere_channels,
      hidden_channels=hidden_channels,
    )

    key = jax.random.PRNGKey(0)
    node_embedding = jax.random.normal(
      key, (num_atoms, sph_size, sphere_channels)
    )
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])

    params = head.init(key, node_embedding, batch, num_systems)
    output = head.apply(params, node_embedding, batch, num_systems)

    self.assertEqual(output['energy'].shape, (num_systems,))
    self.assertTrue(jnp.all(jnp.isfinite(output['energy'])))

  def test_mlp_energy_head_jit(self):
    """Test MLPEnergyHead with JIT compilation."""
    sphere_channels = 32
    hidden_channels = 32
    num_atoms = 8
    num_systems = 2
    lmax = 2
    sph_size = (lmax + 1) ** 2

    head = uma.MLPEnergyHead(
      sphere_channels=sphere_channels,
      hidden_channels=hidden_channels,
    )

    key = jax.random.PRNGKey(0)
    node_embedding = jax.random.normal(
      key, (num_atoms, sph_size, sphere_channels)
    )
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])

    params = head.init(key, node_embedding, batch, num_systems)

    # JIT compile with static num_systems
    jit_apply = jax.jit(head.apply, static_argnums=(3,))
    output = jit_apply(params, node_embedding, batch, num_systems)

    self.assertEqual(output['energy'].shape, (num_systems,))
    self.assertTrue(jnp.all(jnp.isfinite(output['energy'])))

  def test_linear_energy_head(self):
    """Test LinearEnergyHead module."""
    sphere_channels = 32
    num_atoms = 8
    num_systems = 2
    lmax = 2
    sph_size = (lmax + 1) ** 2

    head = uma.LinearEnergyHead(sphere_channels=sphere_channels)

    key = jax.random.PRNGKey(0)
    node_embedding = jax.random.normal(
      key, (num_atoms, sph_size, sphere_channels)
    )
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])

    params = head.init(key, node_embedding, batch, num_systems)
    output = head.apply(params, node_embedding, batch, num_systems)

    self.assertEqual(output['energy'].shape, (num_systems,))
    self.assertTrue(jnp.all(jnp.isfinite(output['energy'])))

  def test_linear_force_head(self):
    """Test LinearForceHead module."""
    sphere_channels = 32
    num_atoms = 8
    lmax = 2
    sph_size = (lmax + 1) ** 2

    head = uma.LinearForceHead(sphere_channels=sphere_channels)

    key = jax.random.PRNGKey(0)
    node_embedding = jax.random.normal(
      key, (num_atoms, sph_size, sphere_channels)
    )

    params = head.init(key, node_embedding)
    output = head.apply(params, node_embedding)

    self.assertEqual(output['forces'].shape, (num_atoms, 3))
    self.assertTrue(jnp.all(jnp.isfinite(output['forces'])))

  def test_energy_head_mean_reduction(self):
    """Test energy head with mean reduction."""
    sphere_channels = 32
    hidden_channels = 32
    num_atoms = 8
    num_systems = 2
    lmax = 2
    sph_size = (lmax + 1) ** 2

    head = uma.MLPEnergyHead(
      sphere_channels=sphere_channels,
      hidden_channels=hidden_channels,
      reduce='mean',
    )

    key = jax.random.PRNGKey(0)
    node_embedding = jax.random.normal(
      key, (num_atoms, sph_size, sphere_channels)
    )
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    natoms = jnp.array([4, 4])

    params = head.init(key, node_embedding, batch, num_systems, natoms)
    output = head.apply(params, node_embedding, batch, num_systems, natoms)

    self.assertEqual(output['energy'].shape, (num_systems,))
    self.assertTrue(jnp.all(jnp.isfinite(output['energy'])))


class UMAModulesTest(test_util.JAXMDTestCase):
  """Tests for UMA sub-modules."""

  def test_uma_backbone_instantiation(self):
    """Test that UMABackbone can be instantiated."""
    config = uma.UMAConfig(
      sphere_channels=32,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=32,
    )
    model = uma.UMABackbone(config=config)
    self.assertIsNotNone(model)

  def test_uma_block_instantiation(self):
    """Test that UMABlock can be instantiated."""
    block = uma.UMABlock(
      sphere_channels=32,
      hidden_channels=32,
      lmax=2,
      mmax=2,
      m_size=(3, 4, 2),  # Example m_size for lmax=2, mmax=2
      edge_channels_list=[64, 32, 32],
      cutoff=5.0,
    )
    self.assertIsNotNone(block)


class UMABackboneForwardTest(test_util.JAXMDTestCase):
  """Tests for UMA backbone forward pass."""

  def test_uma_backbone_forward_pass(self):
    """Test full UMABackbone forward pass with synthetic data."""
    # Use config values that match the working test_uma_comparison.py
    config = uma.UMAConfig(
      max_num_elements=100,
      sphere_channels=64,
      lmax=2,
      mmax=2,
      num_layers=1,
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

    model = uma.UMABackbone(config=config)
    data = create_test_data(num_atoms=10, num_systems=2, cutoff=config.cutoff)

    # Initialize parameters
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
      data['dataset'],
    )

    # Forward pass
    output = model.apply(
      params,
      data['positions'],
      data['atomic_numbers'],
      data['batch'],
      data['edge_index'],
      data['edge_distance_vec'],
      data['charge'],
      data['spin'],
      data['dataset'],
    )

    # Verify output structure
    self.assertIn('node_embedding', output)
    self.assertIn('batch', output)

    # Verify shapes
    num_atoms = data['positions'].shape[0]
    sph_size = (config.lmax + 1) ** 2
    self.assertEqual(
      output['node_embedding'].shape,
      (num_atoms, sph_size, config.sphere_channels),
    )

    # Verify outputs are finite
    self.assertTrue(jnp.all(jnp.isfinite(output['node_embedding'])))

  def test_uma_backbone_forward_pass_no_dataset_embedding(self):
    """Test UMABackbone forward pass without dataset embedding."""
    config = uma.UMAConfig(
      max_num_elements=100,
      sphere_channels=64,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=64,
      cutoff=5.0,
      edge_channels=64,
      num_distance_basis=128,
      use_dataset_embedding=False,
    )

    model = uma.UMABackbone(config=config)
    data = create_test_data(num_atoms=8, num_systems=2, cutoff=config.cutoff)

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
      None,  # No dataset
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

    self.assertIn('node_embedding', output)
    self.assertTrue(jnp.all(jnp.isfinite(output['node_embedding'])))

  def test_uma_backbone_deterministic(self):
    """Test UMABackbone produces deterministic outputs."""
    config = uma.UMAConfig(
      max_num_elements=100,
      sphere_channels=64,
      lmax=2,
      mmax=2,
      num_layers=1,
      hidden_channels=64,
      cutoff=5.0,
      edge_channels=64,
      num_distance_basis=128,
      use_dataset_embedding=False,
    )

    model = uma.UMABackbone(config=config)
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
      None,
    )

    # Run forward pass twice
    output1 = model.apply(
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

    output2 = model.apply(
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

    # Outputs should be identical
    self.assertTrue(
      jnp.allclose(output1['node_embedding'], output2['node_embedding'])
    )

  def test_uma_backbone_multiple_layers(self):
    """Test UMABackbone with multiple layers."""
    config = uma.UMAConfig(
      max_num_elements=100,
      sphere_channels=64,
      lmax=2,
      mmax=2,
      num_layers=2,  # Multiple layers
      hidden_channels=64,
      cutoff=5.0,
      edge_channels=64,
      num_distance_basis=128,
      use_dataset_embedding=False,
    )

    model = uma.UMABackbone(config=config)
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

    self.assertIn('node_embedding', output)
    self.assertTrue(jnp.all(jnp.isfinite(output['node_embedding'])))


if __name__ == '__main__':
  absltest.main()
