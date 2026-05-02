"""Tests for the MACE-JAX JAX-MD energy wrapper."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from jax_md import energy, partition, quantity, space
from jax_md.custom_partition import neighbor_list_multi_image

jax.config.parse_flags_with_absl()


def tiny_mace_config(atomic_numbers=(1,)):
  return {
    'atomic_numbers': list(atomic_numbers),
    'atomic_energies': [0.0] * len(atomic_numbers),
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


def make_mace_energy(
  box,
  *,
  fractional_coordinates,
  model,
  config,
  z_atomic,
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
    config=config,
    z_atomic=z_atomic,
    r_cutoff=float(config['r_max']),
    dr_threshold=0.0,
    fractional_coordinates=fractional_coordinates,
    neighbor_list_fn=neighbor_list_fn,
    featurizer_fn=featurizer_fn,
    **kwargs,
  )


class MaceTest(absltest.TestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    try:
      from flax import nnx
      from jax_md._nn.mace_jax_interface import model_builder
    except ImportError as err:
      cls.import_error = err
      return

    cls.import_error = None
    cls.config = tiny_mace_config()
    cls.model = model_builder.build_jax_model(cls.config, rngs=nnx.Rngs(0))
    cls.mixed_config = tiny_mace_config((1, 8))
    cls.mixed_model = model_builder.build_jax_model(
      cls.mixed_config, rngs=nnx.Rngs(1)
    )

  def skip_if_missing_mace_jax(self):
    if self.import_error is not None:
      self.skipTest(
        f'MACE-JAX dependencies are not installed: {self.import_error}'
      )

  def test_fractional_matrix_box_runs(self):
    self.skip_if_missing_mace_jax()
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

    neighbor_fn, energy_fn = make_mace_energy(
      cell,
      fractional_coordinates=True,
      model=self.model,
      config=self.config,
      z_atomic=jnp.array([1, 1], dtype=jnp.int32),
    )
    neighbors = neighbor_fn.allocate(R_frac, box=cell)

    e = energy_fn(R_frac, neighbor=neighbors)
    self.assertEqual(e.shape, ())
    self.assertTrue(jnp.isfinite(e))

  def test_cartesian_matrix_box_runs(self):
    self.skip_if_missing_mace_jax()
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
      model=self.model,
      config=self.config,
      z_atomic=jnp.array([1, 1], dtype=jnp.int32),
    )
    neighbors = neighbor_fn.allocate(R_cart, box=cell)

    e = energy_fn(R_cart, box=cell, neighbor=neighbors)
    self.assertEqual(e.shape, ())
    self.assertTrue(jnp.isfinite(e))

  def test_standard_neighbor_list_formats_run(self):
    self.skip_if_missing_mace_jax()
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    R = jnp.array(
      [
        [0.1, 1.8, 0.1],
        [0.1, 0.2, 0.1],
      ],
      dtype=jnp.float32,
    )

    for neighbor_format in (
      partition.Dense,
      partition.Sparse,
      partition.OrderedSparse,
    ):
      neighbor_fn, energy_fn = make_mace_energy(
        box,
        fractional_coordinates=False,
        format=neighbor_format,
        model=self.model,
        config=self.config,
        z_atomic=jnp.array([1, 1], dtype=jnp.int32),
      )
      neighbors = neighbor_fn.allocate(R)
      energy = energy_fn(R, neighbor=neighbors)
      self.assertEqual(energy.shape, ())
      self.assertTrue(jnp.isfinite(energy))

  def test_multi_image_neighbor_list_formats_run(self):
    self.skip_if_missing_mace_jax()
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
        model=self.model,
        config=self.config,
        z_atomic=jnp.array([1, 1], dtype=jnp.int32),
        featurizer_fn=mace_multi_image_featurizer,
      )
      neighbors = neighbor_fn.allocate(R_frac, box=cell)
      energy = energy_fn(R_frac, neighbor=neighbors)
      self.assertEqual(energy.shape, ())
      self.assertTrue(jnp.isfinite(energy))

  def test_unsupported_atomic_number_raises(self):
    self.skip_if_missing_mace_jax()
    box = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    displacement_fn, _ = space.periodic(box)
    with self.assertRaisesRegex(ValueError, 'not present in config'):
      energy.mace_neighbor_list(
        displacement_fn,
        box,
        model=self.model,
        config=self.config,
        z_atomic=jnp.array([1, 6], dtype=jnp.int32),
        r_cutoff=float(self.config['r_max']),
        dr_threshold=0.0,
        fractional_coordinates=False,
      )

  def test_energy_supports_quantity_stress_perturbation(self):
    self.skip_if_missing_mace_jax()
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
      model=self.model,
      config=self.config,
      z_atomic=jnp.array([1, 1], dtype=jnp.int32),
    )
    neighbors = neighbor_fn.allocate(R_cart, box=cell)

    stress = quantity.stress(energy_fn, R_cart, cell, neighbor=neighbors)

    self.assertEqual(stress.shape, (3, 3))
    self.assertTrue(jnp.all(jnp.isfinite(stress)))

  def test_energy_force_gradient_is_finite(self):
    self.skip_if_missing_mace_jax()
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
      model=self.model,
      config=self.config,
      z_atomic=jnp.array([1, 1], dtype=jnp.int32),
    )
    neighbors = neighbor_fn.allocate(R)

    grad_R = jax.grad(lambda R_: energy_fn(R_, neighbor=neighbors))(R)

    self.assertEqual(grad_R.shape, R.shape)
    self.assertTrue(jnp.all(jnp.isfinite(grad_R)))

  def test_fractional_coordinates_support_quantity_stress_perturbation(self):
    self.skip_if_missing_mace_jax()
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
    neighbor_fn, energy_fn = make_mace_energy(
      cell,
      fractional_coordinates=True,
      model=self.model,
      config=self.config,
      z_atomic=jnp.array([1, 1], dtype=jnp.int32),
    )
    neighbors = neighbor_fn.allocate(R_frac, box=cell)

    stress = quantity.stress(energy_fn, R_frac, cell, neighbor=neighbors)

    self.assertEqual(stress.shape, (3, 3))
    self.assertTrue(jnp.all(jnp.isfinite(stress)))

  def test_multi_species_model_runs(self):
    self.skip_if_missing_mace_jax()
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
      model=self.mixed_model,
      config=self.mixed_config,
      z_atomic=jnp.array([1, 8], dtype=jnp.int32),
      r_cutoff=float(self.mixed_config['r_max']),
      dr_threshold=0.0,
      fractional_coordinates=False,
    )
    neighbors = neighbor_fn.allocate(R)
    e = energy_fn(R, neighbor=neighbors)

    self.assertEqual(e.shape, ())
    self.assertTrue(jnp.isfinite(e))


if __name__ == '__main__':
  absltest.main()
