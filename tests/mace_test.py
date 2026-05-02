"""Tests for the MACE-JAX JAX-MD energy wrapper."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import energy, partition, quantity, space
from jax_md.custom_partition import neighbor_list_multi_image

jax.config.parse_flags_with_absl()


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
  model=shift_projection_model,
  neighbor_list_fn=partition.neighbor_list,
  featurizer_fn=None,
  **kwargs,
):
  box_arr = jnp.asarray(box)
  if fractional_coordinates or box_arr.shape == (3, 3):
    displacement_fn, _ = space.periodic_general(
      box_arr, fractional_coordinates=fractional_coordinates
    )
  else:
    displacement_fn, _ = space.periodic(box_arr)
  return energy.mace_neighbor_list(
    displacement_fn,
    box,
    model=model,
    config={'atomic_numbers': (1,)},
    z_atomic=jnp.array([1, 1], dtype=jnp.int32),
    r_cutoff=1.0,
    dr_threshold=0.0,
    fractional_coordinates=fractional_coordinates,
    neighbor_list_fn=neighbor_list_fn,
    featurizer_fn=featurizer_fn,
    **kwargs,
  )


class MaceTest(absltest.TestCase):
  def test_cueq_config_normalization_passes_through_enabled(self):
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
    np.testing.assert_allclose(space.transform(cell, R_frac), R_frac @ cell.T)

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
    energy_with_neighbor2 = energy_fn(R, neighbor=neighbor)
    np.testing.assert_allclose(energy_with_neighbor, energy_with_neighbor2)

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
        model=nearest_edge_distance_model,
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
      from jax_md._nn.mace_jax_interface.featurizer import (
        mace_multi_image_featurizer,
      )

      neighbor_fn, energy_fn = make_mace_energy(
        cell,
        fractional_coordinates=True,
        format=neighbor_format,
        neighbor_list_fn=neighbor_list_multi_image,
        model=nearest_edge_distance_model,
        featurizer_fn=mace_multi_image_featurizer,
      )
      neighbors = neighbor_fn.allocate(R_frac, box=cell)
      energy = energy_fn(R_frac, neighbor=neighbors)
      self.assertEqual(energy.shape, ())
      self.assertTrue(jnp.isfinite(energy))
      energies.append(energy)

    np.testing.assert_allclose(energies, jnp.repeat(energies[0], 3), rtol=1e-6)

  def test_unsupported_atomic_number_raises(self):
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    displacement_fn, _ = space.periodic(box)
    with self.assertRaisesRegex(ValueError, 'not present in config'):
      energy.mace_neighbor_list(
        displacement_fn,
        box,
        model=shift_projection_model,
        config={'atomic_numbers': (1,)},
        z_atomic=jnp.array([1, 6], dtype=jnp.int32),
        r_cutoff=1.0,
        dr_threshold=0.0,
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
      model=quadratic_edge_model,
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
      model=quadratic_edge_model,
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
      model=quadratic_edge_model,
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
    displacement_fn, _ = space.periodic(box)
    neighbor_fn, energy_fn = energy.mace_neighbor_list(
      displacement_fn,
      box,
      model=species_projection_model,
      config={'atomic_numbers': (1, 8)},
      z_atomic=jnp.array([1, 8], dtype=jnp.int32),
      r_cutoff=1.0,
      dr_threshold=0.0,
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

    displacement_fn, _ = space.periodic_general(
      cell, fractional_coordinates=False
    )
    neighbor_fn, energy_fn = energy.mace_neighbor_list(
      displacement_fn,
      cell,
      model=jax_model,
      config=config,
      z_atomic=jnp.array([1, 1], dtype=jnp.int32),
      r_cutoff=1.0,
      dr_threshold=0.0,
      fractional_coordinates=False,
    )
    neighbors = neighbor_fn.allocate(R_cart, box=cell)
    e = energy_fn(R_cart, neighbor=neighbors)

    self.assertEqual(e.shape, ())
    self.assertTrue(jnp.isfinite(e))


if __name__ == '__main__':
  absltest.main()
