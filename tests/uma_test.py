"""Tests for jax_md._nn.uma (UMA model).

Note: The full UMA backbone model requires specific configurations to work.
These tests verify the basic components and heads work correctly.
"""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from jax_md import test_util
from jax_md._nn import uma


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


if __name__ == '__main__':
  absltest.main()
