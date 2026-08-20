"""Tests for the pretrained Vivace energy wrapper."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import energy, partition, space
from jax_md._nn import vivace

jax.config.parse_flags_with_absl()


class VivaceTest(absltest.TestCase):
  def test_pretrained_water_energy_and_forces(self):
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    positions = jnp.asarray(
      [
        [1.0, 1.0, 1.0],
        [1.9572, 1.0, 1.0],
        [0.760013, 1.926627, 1.0],
      ],
      dtype=dtype,
    )
    species = jnp.asarray([8, 1, 1], dtype=jnp.int32)
    box = jnp.eye(3, dtype=dtype) * 100.0
    displacement_fn, _ = space.free()

    model = vivace.load_model()
    self.assertIsInstance(model, vivace.Vivace)
    self.assertEqual(model.cutoff, 6.5)

    neighbor_fn, energy_fn = energy.vivace_neighbor_list(
      displacement_fn,
      box,
      species=species,
      disable_cell_list=True,
    )
    neighbors = neighbor_fn.allocate(positions)
    predicted_energy = energy_fn(positions, neighbors)
    forces = -jax.grad(lambda p: energy_fn(p, neighbors))(positions)

    self.assertAlmostEqual(float(predicted_energy), -2079.8477132790313, 3)
    np.testing.assert_allclose(
      np.asarray(forces),
      np.asarray(
        [
          [-0.1143299064, -0.1477287956, 0.0],
          [0.2000847350, -0.0367609653, 0.0],
          [-0.0857548286, 0.1844897609, 0.0],
        ]
      ),
      rtol=2e-5,
      atol=5e-6,
    )

    sparse_neighbor_fn, sparse_energy_fn = energy.vivace_neighbor_list(
      displacement_fn,
      box,
      species=species,
      disable_cell_list=True,
      format=partition.Sparse,
    )
    sparse_neighbors = sparse_neighbor_fn.allocate(positions)
    np.testing.assert_allclose(
      sparse_energy_fn(positions, sparse_neighbors),
      predicted_energy,
      rtol=1e-6,
    )


if __name__ == '__main__':
  absltest.main()
