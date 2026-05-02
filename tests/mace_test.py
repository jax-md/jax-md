"""Tests for the MACE-JAX JAX-MD energy wrapper."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import partition, quantity, space
from jax_md._nn import mace
from jax_md.custom_partition import neighbor_list_multi_image

jax.config.parse_flags_with_absl()


def template_batch(n_particles=2, k_neighbors=1, n_species=1):
  edge_count = n_particles * k_neighbors
  return {
    'positions': jnp.zeros((n_particles, 3), dtype=jnp.float32),
    'edge_index': jnp.zeros((2, edge_count), dtype=jnp.int32),
    'shifts': jnp.zeros((edge_count, 3), dtype=jnp.float32),
    'unit_shifts': jnp.zeros((edge_count, 3), dtype=jnp.float32),
    'node_attrs': jnp.zeros((n_particles, n_species), dtype=jnp.float32),
    'node_attrs_index': jnp.zeros((n_particles,), dtype=jnp.int32),
    'batch': jnp.zeros((n_particles,), dtype=jnp.int32),
    'ptr': jnp.array([0, n_particles], dtype=jnp.int32),
    'cell': jnp.zeros((1, 3, 3), dtype=jnp.float32),
  }


def shift_projection_model(batch):
  weights = jnp.array([1.0, 10.0, 100.0], dtype=batch['shifts'].dtype)
  return jnp.dot(batch['shifts'][0], weights)


def quadratic_edge_model(batch):
  senders = batch['edge_index'][0]
  receivers = batch['edge_index'][1]
  edge_vectors = (
    batch['positions'][receivers]
    + batch['shifts']
    - batch['positions'][senders]
  )
  return 0.5 * jnp.sum(edge_vectors**2)


def nearest_edge_distance_model(batch):
  senders = batch['edge_index'][0]
  receivers = batch['edge_index'][1]
  edge_vectors = (
    batch['positions'][receivers]
    + batch['shifts']
    - batch['positions'][senders]
  )
  return jnp.min(jnp.sum(edge_vectors**2, axis=-1))


def species_projection_model(batch):
  return (
    batch['node_attrs'][0, 0]
    + 10.0 * batch['node_attrs'][1, 1]
    + batch['node_attrs_index'][1].astype(batch['node_attrs'].dtype)
  )


def make_mace_energy(
  box,
  *,
  fractional_coordinates,
  jax_model=shift_projection_model,
  format=partition.Dense,
  neighbor_list_fn=partition.neighbor_list,
  k_neighbors=1,
):
  return mace.mace_neighbor_list(
    jax_model=jax_model,
    template_batch=template_batch(k_neighbors=k_neighbors),
    config={'atomic_numbers': (1,)},
    box=box,
    z_atomic=jnp.array([1, 1], dtype=jnp.int32),
    r_cutoff=1.0,
    dr_threshold=0.0,
    k_neighbors=k_neighbors,
    include_head=False,
    fractional_coordinates=fractional_coordinates,
    format=format,
    neighbor_list_fn=neighbor_list_fn,
  )


class MaceTest(absltest.TestCase):
  def test_cueq_config_normalization_enables_runnable_default(self):
    try:
      from jax_md._nn.mace_jax_interface import model_builder
    except ImportError as err:
      self.skipTest(f'MACE-JAX dependencies are not installed: {err}')

    cueq_config = model_builder.normalize_cueq_config(
      {'cueq_config': {'enabled': True}},
      None,
    )

    self.assertIsNotNone(cueq_config)
    self.assertTrue(cueq_config.enabled)
    self.assertTrue(cueq_config.optimize_all)

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
    R_cart = mace.to_cartesian(R_frac, cell, fractional_coordinates=True)
    np.testing.assert_allclose(R_cart, space.transform(cell, R_frac))

    neighbor_fn, energy_fn = make_mace_energy(cell, fractional_coordinates=True)
    neighbors = neighbor_fn.allocate(R_frac, box=cell)

    energy = energy_fn(R_frac, neighbor=neighbors)
    expected_unit_shifts = jnp.array([[0, 1, 0], [0, -1, 0]], dtype=jnp.int32)
    expected_shift = space.transform(
      cell, expected_unit_shifts[:1].astype(jnp.float32)
    )[0]
    weights = jnp.array([1.0, 10.0, 100.0], dtype=jnp.float32)
    np.testing.assert_allclose(
      energy,
      jnp.dot(expected_shift, weights),
      rtol=1e-6,
    )

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

    neighbor_fn, energy_fn = make_mace_energy(
      cell, fractional_coordinates=False
    )
    neighbors = neighbor_fn.allocate(R_cart, box=cell)

    self.assertEqual(int(jnp.sum(neighbors.idx[0] == 1)), 1)
    self.assertEqual(int(jnp.sum(neighbors.idx[1] == 0)), 1)

    expected_unit_shifts = jnp.array([[0, 1, 0], [0, -1, 0]], dtype=jnp.int32)
    expected_shift = space.transform(
      cell, expected_unit_shifts[:1].astype(jnp.float32)
    )[0]
    weights = jnp.array([1.0, 10.0, 100.0], dtype=jnp.float32)
    np.testing.assert_allclose(
      energy_fn(R_cart, box=cell, neighbor=neighbors),
      jnp.dot(expected_shift, weights),
      rtol=1e-6,
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

    neighbor_fn, energy_fn = make_mace_energy(box, fractional_coordinates=False)
    neighbors = neighbor_fn.allocate(R)

    expected_unit_shifts = jnp.array([[0, 1, 0], [0, -1, 0]], dtype=jnp.int32)
    expected_shift = space.transform(
      jnp.diag(box), expected_unit_shifts[:1].astype(jnp.float32)
    )[0]
    weights = jnp.array([1.0, 10.0, 100.0], dtype=jnp.float32)
    np.testing.assert_allclose(
      energy_fn(R, neighbor=neighbors),
      jnp.dot(expected_shift, weights),
    )

  def test_energy_accepts_standard_neighbor_keyword(self):
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    R = jnp.array(
      [
        [0.1, 1.8, 0.1],
        [0.1, 0.2, 0.1],
      ],
      dtype=jnp.float32,
    )

    neighbor_fn, energy_fn = make_mace_energy(box, fractional_coordinates=False)
    neighbor = neighbor_fn.allocate(R)

    energy_with_neighbor = energy_fn(R, neighbor=neighbor)
    energy_with_neighbors = energy_fn(R, neighbors=neighbor)
    np.testing.assert_allclose(energy_with_neighbor, energy_with_neighbors)

  def test_standard_neighbor_list_formats_produce_same_energy(self):
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    R = jnp.array(
      [
        [0.1, 1.8, 0.1],
        [0.1, 0.2, 0.1],
      ],
      dtype=jnp.float32,
    )

    energies = []
    for neighbor_format in (
      partition.Dense,
      partition.Sparse,
      partition.OrderedSparse,
    ):
      neighbor_fn, energy_fn = make_mace_energy(
        box,
        fractional_coordinates=False,
        format=neighbor_format,
        jax_model=nearest_edge_distance_model,
      )
      neighbors = neighbor_fn.allocate(R)
      energy = energy_fn(R, neighbor=neighbors)
      self.assertEqual(energy.shape, ())
      self.assertTrue(jnp.isfinite(energy))
      energies.append(energy)

    np.testing.assert_allclose(energies, jnp.repeat(energies[0], 3), rtol=1e-6)

  def test_multi_image_neighbor_list_formats_produce_same_energy(self):
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

    energies = []
    for neighbor_format in (
      partition.Dense,
      partition.Sparse,
      partition.OrderedSparse,
    ):
      neighbor_fn, energy_fn = make_mace_energy(
        cell,
        fractional_coordinates=True,
        format=neighbor_format,
        neighbor_list_fn=neighbor_list_multi_image,
        k_neighbors=2,
        jax_model=nearest_edge_distance_model,
      )
      neighbors = neighbor_fn.allocate(R_frac, box=cell)
      energy = energy_fn(R_frac, neighbor=neighbors)
      self.assertEqual(energy.shape, ())
      self.assertTrue(jnp.isfinite(energy))
      energies.append(energy)

    np.testing.assert_allclose(energies, jnp.repeat(energies[0], 3), rtol=1e-6)

  def test_neighbor_capacity_overflow_raises_instead_of_truncating(self):
    box = jnp.array([3.0, 3.0, 3.0], dtype=jnp.float32)
    _, energy_fn = mace.mace_neighbor_list(
      jax_model=shift_projection_model,
      template_batch=template_batch(n_particles=3, k_neighbors=1),
      config={'atomic_numbers': (1,)},
      box=box,
      z_atomic=jnp.array([1, 1, 1], dtype=jnp.int32),
      r_cutoff=2.0,
      dr_threshold=0.0,
      k_neighbors=1,
      include_head=False,
      fractional_coordinates=False,
    )
    R = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
      ],
      dtype=jnp.float32,
    )
    neighbor_idx = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)

    with self.assertRaisesRegex(ValueError, 'template only supports'):
      energy_fn(R, neighbor_idx=neighbor_idx)

  def test_unsupported_atomic_number_raises(self):
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, 'not present in config'):
      mace.mace_neighbor_list(
        jax_model=shift_projection_model,
        template_batch=template_batch(),
        config={'atomic_numbers': (1,)},
        box=box,
        z_atomic=jnp.array([1, 6], dtype=jnp.int32),
        r_cutoff=1.0,
        dr_threshold=0.0,
        k_neighbors=1,
        include_head=False,
        fractional_coordinates=False,
      )

  def test_energy_supports_quantity_stress_perturbation(self):
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
    neighbor_fn, energy_fn = make_mace_energy(
      cell,
      fractional_coordinates=False,
      jax_model=quadratic_edge_model,
    )
    neighbors = neighbor_fn.allocate(R_cart, box=cell)

    stress = quantity.stress(energy_fn, R_cart, cell, neighbor=neighbors)

    senders = jnp.array([0, 1], dtype=jnp.int32)
    receivers = jnp.array([1, 0], dtype=jnp.int32)
    unit_shifts = jnp.array([[0, 1, 0], [0, -1, 0]], dtype=jnp.float32)
    shifts = space.transform(cell, unit_shifts)
    edge_vectors = R_cart[receivers] + shifts - R_cart[senders]
    expected_stress = -jnp.einsum(
      'ei,ej->ij', edge_vectors, edge_vectors
    ) / quantity.volume(3, cell)

    np.testing.assert_allclose(stress, expected_stress, rtol=1e-5, atol=1e-5)

  def test_energy_force_gradient_is_finite_and_correct(self):
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    R = jnp.array(
      [
        [0.1, 0.1, 0.1],
        [0.4, 0.1, 0.1],
      ],
      dtype=jnp.float32,
    )
    neighbor_fn, energy_fn = make_mace_energy(
      box,
      fractional_coordinates=False,
      jax_model=quadratic_edge_model,
    )
    neighbors = neighbor_fn.allocate(R)

    grad_R = jax.grad(lambda R_: energy_fn(R_, neighbor=neighbors))(R)

    expected_grad = jnp.array(
      [
        [-0.6, 0.0, 0.0],
        [0.6, 0.0, 0.0],
      ],
      dtype=jnp.float32,
    )
    self.assertTrue(jnp.all(jnp.isfinite(grad_R)))
    np.testing.assert_allclose(grad_R, expected_grad, rtol=1e-6, atol=1e-6)

  def test_fractional_coordinates_support_quantity_stress_perturbation(self):
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
    neighbor_fn, energy_fn = make_mace_energy(
      cell,
      fractional_coordinates=True,
      jax_model=quadratic_edge_model,
    )
    neighbors = neighbor_fn.allocate(R_frac, box=cell)

    stress = quantity.stress(energy_fn, R_frac, cell, neighbor=neighbors)

    senders = jnp.array([0, 1], dtype=jnp.int32)
    receivers = jnp.array([1, 0], dtype=jnp.int32)
    unit_shifts = jnp.array([[0, 1, 0], [0, -1, 0]], dtype=jnp.float32)
    shifts = space.transform(cell, unit_shifts)
    edge_vectors = R_cart[receivers] + shifts - R_cart[senders]
    expected_stress = -jnp.einsum(
      'ei,ej->ij', edge_vectors, edge_vectors
    ) / quantity.volume(3, cell)

    np.testing.assert_allclose(stress, expected_stress, rtol=1e-5, atol=1e-5)

  def test_multi_species_node_attrs_are_encoded(self):
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    R = jnp.array(
      [
        [0.1, 0.1, 0.1],
        [0.4, 0.1, 0.1],
      ],
      dtype=jnp.float32,
    )
    neighbor_fn, energy_fn = mace.mace_neighbor_list(
      jax_model=species_projection_model,
      template_batch=template_batch(n_species=2),
      config={'atomic_numbers': (1, 8)},
      box=box,
      z_atomic=jnp.array([1, 8], dtype=jnp.int32),
      r_cutoff=1.0,
      dr_threshold=0.0,
      k_neighbors=1,
      include_head=False,
      fractional_coordinates=False,
    )
    neighbors = neighbor_fn.allocate(R)

    np.testing.assert_allclose(energy_fn(R, neighbor=neighbors), 12.0)

  def test_real_mace_jax_model_smoke(self):
    try:
      from flax import nnx
      from jax_md._nn.mace_jax_interface import model_builder
    except ImportError as err:
      self.skipTest(f'MACE-JAX dependencies are not installed: {err}')

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

    jax_model = model_builder.build_jax_model(config, rngs=nnx.Rngs(0))
    real_template_batch = dict(
      model_builder.prepare_template_data(config, n_node=2, n_edge=2)
    )

    self.assertIn('unit_shifts', real_template_batch)

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

    neighbor_fn, energy_fn = mace.mace_neighbor_list(
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
    neighbors = neighbor_fn.allocate(R_cart, box=cell)
    energy = energy_fn(R_cart, neighbors=neighbors)

    self.assertEqual(energy.shape, ())
    self.assertTrue(jnp.isfinite(energy))


if __name__ == '__main__':
  absltest.main()
