"""Tests for the MACE-JAX JAX-MD bridge."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import space
from jax_md._nn.mace_jax_interface import mace_jaxmd_bridge

jax.config.parse_flags_with_absl()


def template_batch(n_particles=2, k_neighbors=1):
  edge_count = n_particles * k_neighbors
  return {
    'positions': jnp.zeros((n_particles, 3), dtype=jnp.float32),
    'edge_index': jnp.zeros((2, edge_count), dtype=jnp.int32),
    'shifts': jnp.zeros((edge_count, 3), dtype=jnp.float32),
    'unit_shifts': jnp.zeros((edge_count, 3), dtype=jnp.float32),
    'node_attrs': jnp.zeros((n_particles, 1), dtype=jnp.float32),
    'node_attrs_index': jnp.zeros((n_particles,), dtype=jnp.int32),
    'batch': jnp.zeros((n_particles,), dtype=jnp.int32),
    'ptr': jnp.array([0, n_particles], dtype=jnp.int32),
    'cell': jnp.zeros((1, 3, 3), dtype=jnp.float32),
  }


def shift_projection_model(batch):
  weights = jnp.array([1.0, 10.0, 100.0], dtype=batch['shifts'].dtype)
  return jnp.dot(batch['shifts'][0], weights)


def make_bridge(box, *, fractional_coordinates):
  return mace_jaxmd_bridge.make_mace_jaxmd_energy(
    jax_model=shift_projection_model,
    template_batch=template_batch(),
    config={'atomic_numbers': (1,)},
    box=box,
    z_atomic=jnp.array([1, 1], dtype=jnp.int32),
    r_cutoff=1.0,
    dr_threshold=0.0,
    k_neighbors=1,
    include_head=False,
    fractional_coordinates=fractional_coordinates,
  )


class MaceJaxmdBridgeTest(absltest.TestCase):
  def test_fractional_matrix_box_uses_jax_md_transform_convention(self):
    cell = jnp.array(
      [
        [2.0, 0.3, 0.1],
        [0.0, 1.7, 0.2],
        [0.0, 0.0, 1.5],
      ],
      dtype=jnp.float32,
    )
    R_frac = jnp.array(
      [
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.1],
      ],
      dtype=jnp.float32,
    )
    neighbor_idx = jnp.array([[1], [0]], dtype=jnp.int32)

    R_cart = mace_jaxmd_bridge._to_cartesian(
      R_frac, cell, fractional_coordinates=True
    )
    np.testing.assert_allclose(R_cart, space.transform(cell, R_frac))

    _, _, _, freeze_graph_fn, make_fixed_graph_energy_fn = make_bridge(
      cell, fractional_coordinates=True
    )
    fixed_graph = freeze_graph_fn(R_frac, neighbor_idx=neighbor_idx)

    expected_unit_shifts = jnp.array([[0, 1, 0], [0, -1, 0]], dtype=jnp.int32)
    np.testing.assert_allclose(
      fixed_graph['unit_shifts_int'], expected_unit_shifts
    )

    fixed_energy_fn = make_fixed_graph_energy_fn(fixed_graph)
    energy = fixed_energy_fn(R_frac)
    expected_shift = space.transform(
      cell, expected_unit_shifts[:1].astype(jnp.float32)
    )[0]
    weights = jnp.array([1.0, 10.0, 100.0], dtype=jnp.float32)
    np.testing.assert_allclose(energy, jnp.dot(expected_shift, weights))

  def test_cartesian_matrix_box_uses_periodic_general(self):
    cell = jnp.array(
      [
        [2.0, 0.3, 0.1],
        [0.0, 1.7, 0.2],
        [0.0, 0.0, 1.5],
      ],
      dtype=jnp.float32,
    )
    R_frac = jnp.array(
      [
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.1],
      ],
      dtype=jnp.float32,
    )
    R_cart = space.transform(cell, R_frac)

    neighbor_fn, _, _, freeze_graph_fn, _ = make_bridge(
      cell, fractional_coordinates=False
    )
    neighbors = neighbor_fn.allocate(R_cart)

    self.assertEqual(int(jnp.sum(neighbors.idx[0] == 1)), 1)
    self.assertEqual(int(jnp.sum(neighbors.idx[1] == 0)), 1)

    fixed_graph = freeze_graph_fn(R_cart, box=cell, neighbors=neighbors)
    expected_unit_shifts = jnp.array([[0, 1, 0], [0, -1, 0]], dtype=jnp.int32)
    np.testing.assert_allclose(
      fixed_graph['unit_shifts_int'], expected_unit_shifts
    )

  def test_cartesian_vector_box_path_still_allocates_neighbors(self):
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    R = jnp.array(
      [
        [0.1, 1.8, 0.1],
        [0.1, 0.2, 0.1],
      ],
      dtype=jnp.float32,
    )

    neighbor_fn, _, _, freeze_graph_fn, _ = make_bridge(
      box, fractional_coordinates=False
    )
    neighbors = neighbor_fn.allocate(R)
    fixed_graph = freeze_graph_fn(R, neighbors=neighbors)

    expected_unit_shifts = jnp.array([[0, 1, 0], [0, -1, 0]], dtype=jnp.int32)
    np.testing.assert_allclose(
      fixed_graph['unit_shifts_int'], expected_unit_shifts
    )

  def test_real_mace_jax_model_smoke(self):
    from flax import nnx
    from jax_md._nn.mace_jax_interface import model_builder

    config = {
      'atomic_numbers': [1],
      'atomic_energies': [0.0],
      'r_max': 1.5,
      'num_bessel': 2,
      'num_polynomial_cutoff': 2,
      'max_ell': 0,
      'interaction_cls': 'RealAgnosticResidualInteractionBlock',
      'interaction_cls_first': 'RealAgnosticResidualInteractionBlock',
      'num_interactions': 1,
      'hidden_irreps': '4x0e',
      'MLP_irreps': '4x0e',
      'avg_num_neighbors': 1.0,
      'correlation': 1,
      'readout_cls': 'LinearReadoutBlock',
    }

    jax_model = model_builder._build_jax_model(config, rngs=nnx.Rngs(0))
    real_template_batch = dict(
      model_builder._prepare_template_data(config, n_node=2, n_edge=2)
    )

    if 'unit_shifts' not in real_template_batch:
      real_template_batch['unit_shifts'] = jnp.zeros_like(
        real_template_batch['shifts']
      )

    cell = jnp.array(
      [
        [2.0, 0.3, 0.1],
        [0.0, 1.7, 0.2],
        [0.0, 0.0, 1.5],
      ],
      dtype=jnp.float32,
    )
    R_frac = jnp.array(
      [
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.1],
      ],
      dtype=jnp.float32,
    )
    R_cart = space.transform(cell, R_frac)

    neighbor_fn, _, energy_fn, _, _ = mace_jaxmd_bridge.make_mace_jaxmd_energy(
      jax_model=jax_model,
      template_batch=real_template_batch,
      config=config,
      box=cell,
      z_atomic=jnp.array([1, 1], dtype=jnp.int32),
      r_cutoff=1.0,
      dr_threshold=0.0,
      k_neighbors=1,
      include_head=False,
      fractional_coordinates=False,
    )
    neighbors = neighbor_fn.allocate(R_cart)
    energy = energy_fn(R_cart, neighbors=neighbors)

    self.assertEqual(energy.shape, ())
    self.assertTrue(jnp.isfinite(energy))


if __name__ == '__main__':
  absltest.main()
