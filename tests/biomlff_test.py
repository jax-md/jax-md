"""Tests for the bio-mlff pretrained model energy wrappers."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from jax_md import energy, partition, space
from jax_md._nn.orb import load_model as load_orb

jax.config.parse_flags_with_absl()

# Reference energy of the isolated water molecule below.
ORB_ISOLATED_ENERGY = -2078.8386


def water_molecule():
  positions = jnp.asarray(
    [
      [1.0, 1.0, 1.0],
      [1.9572, 1.0, 1.0],
      [0.760013, 1.926627, 1.0],
    ],
    dtype=jnp.float32,
  )
  species = jnp.asarray([8, 1, 1], dtype=jnp.int32)
  return positions, species


class OrbEnergyTest(absltest.TestCase):
  def test_free_energy_matches_reference(self):
    positions, species = water_molecule()
    box = jnp.eye(3, dtype=jnp.float32) * 100.0
    displacement_fn, _ = space.free()
    neighbor_fn, energy_fn = energy.orb_neighbor_list(
      displacement_fn, box, species=species
    )
    e = energy_fn(positions, neighbor_fn.allocate(positions))
    self.assertAlmostEqual(float(e), ORB_ISOLATED_ENERGY, places=2)

  def test_periodic_large_box_matches_free(self):
    # A box far larger than the cutoff must match the free-space energy.
    positions, species = water_molecule()
    box = jnp.eye(3, dtype=jnp.float32) * 100.0

    free_disp, _ = space.free()
    _, free_fn = energy.orb_neighbor_list(free_disp, box, species=species)
    per_disp, _ = space.periodic_general(box, fractional_coordinates=False)
    nbr_fn, per_fn = energy.orb_neighbor_list(per_disp, box, species=species)

    free = free_fn(positions, nbr_fn.allocate(positions))
    per = per_fn(positions, nbr_fn.allocate(positions))
    self.assertAlmostEqual(float(free), float(per), places=3)

  def test_displacement_fn_drives_edges(self):
    # 12 A apart in a 14 A box is 2 A under the minimum image convention,
    # so the free and periodic energies must differ.
    positions = jnp.asarray([[1.0, 1.0, 1.0], [13.0, 1.0, 1.0]], jnp.float32)
    species = jnp.asarray([8, 8], dtype=jnp.int32)
    box = jnp.eye(3, dtype=jnp.float32) * 14.0

    free_disp, _ = space.free()
    nbr_free, free_fn = energy.orb_neighbor_list(
      free_disp, box, species=species
    )
    per_disp, _ = space.periodic_general(box, fractional_coordinates=False)
    nbr_per, per_fn = energy.orb_neighbor_list(per_disp, box, species=species)

    free = free_fn(positions, nbr_free.allocate(positions))
    per = per_fn(positions, nbr_per.allocate(positions))
    self.assertGreater(abs(float(free) - float(per)), 1.0)

  def test_triclinic_isolated_matches_free(self):
    # Row lattice vectors transpose to a jax-md column box.
    positions, species = water_molecule()
    box_vectors = jnp.asarray(
      [[90.0, 0.0, 0.0], [15.0, 95.0, 0.0], [7.0, 4.0, 100.0]],
      dtype=jnp.float32,
    )
    box = jnp.swapaxes(box_vectors, -1, -2)
    displacement_fn, _ = space.periodic_general(
      box, fractional_coordinates=False
    )
    neighbor_fn, energy_fn = energy.orb_neighbor_list(
      displacement_fn, box, species=species
    )
    e = energy_fn(positions, neighbor_fn.allocate(positions))
    self.assertAlmostEqual(float(e), ORB_ISOLATED_ENERGY, places=2)

  def test_forces_match_manual_neighbor_list(self):
    # The wrapper must match a direct model call in value and gradient.
    positions, species = water_molecule()
    net = load_orb()
    box = jnp.eye(3, dtype=jnp.float32) * 100.0
    displacement_fn, _ = space.periodic_general(
      box, fractional_coordinates=False
    )
    neighbor_fn, energy_fn = energy.orb_neighbor_list(
      displacement_fn, box, species=species
    )
    charge = jnp.zeros((1,))
    spin = jnp.zeros((1,))

    def reference(p):
      neighbor = neighbor_fn.allocate(p)
      return net(
        p,
        species,
        charge,
        spin,
        displacement_fn=displacement_fn,
        neighbors=neighbor,
      )

    wrapped_force = -jax.grad(lambda p: energy_fn(p, neighbor_fn.allocate(p)))(
      positions
    )
    reference_force = -jax.grad(reference)(positions)
    self.assertLess(
      float(jnp.max(jnp.abs(wrapped_force - reference_force))), 1e-4
    )

  def test_default_displacement_fn_is_periodic(self):
    # Omitting displacement_fn defaults to space.periodic_general(box).
    positions, species = water_molecule()
    box = jnp.eye(3, dtype=jnp.float32) * 100.0

    nbr_default, default_fn = energy.orb_neighbor_list(box=box, species=species)
    per_disp, _ = space.periodic_general(box, fractional_coordinates=False)
    nbr_explicit, explicit_fn = energy.orb_neighbor_list(
      per_disp, box, species=species
    )

    default = default_fn(positions, nbr_default.allocate(positions))
    explicit = explicit_fn(positions, nbr_explicit.allocate(positions))
    self.assertAlmostEqual(float(default), float(explicit), places=4)

  def test_box_required(self):
    with self.assertRaises(ValueError):
      energy.orb_neighbor_list()

  def test_sparse_format_matches_dense(self):
    # Energies and forces must agree between the two formats.
    positions, species = water_molecule()
    box = jnp.eye(3, dtype=jnp.float32) * 100.0
    displacement_fn, _ = space.periodic_general(
      box, fractional_coordinates=False
    )
    nbr_dense, dense_fn = energy.orb_neighbor_list(
      displacement_fn, box, species=species
    )
    nbr_sparse, sparse_fn = energy.orb_neighbor_list(
      displacement_fn, box, species=species, format=partition.Sparse
    )

    e_dense = dense_fn(positions, nbr_dense.allocate(positions))
    e_sparse = sparse_fn(positions, nbr_sparse.allocate(positions))
    self.assertAlmostEqual(float(e_dense), float(e_sparse), places=4)

    f_dense = -jax.grad(lambda p: dense_fn(p, nbr_dense.allocate(p)))(positions)
    f_sparse = -jax.grad(lambda p: sparse_fn(p, nbr_sparse.allocate(p)))(
      positions
    )
    self.assertLess(float(jnp.max(jnp.abs(f_dense - f_sparse))), 1e-4)

  def test_model_requires_displacement_and_neighbors(self):
    positions, species = water_molecule()
    net = load_orb()
    with self.assertRaises(ValueError):
      net(positions, species, jnp.zeros((1,)), jnp.zeros((1,)))

  def test_species_required(self):
    positions, _ = water_molecule()
    box = jnp.eye(3, dtype=jnp.float32) * 100.0
    displacement_fn, _ = space.periodic_general(
      box, fractional_coordinates=False
    )
    neighbor_fn, energy_fn = energy.orb_neighbor_list(displacement_fn, box)
    neighbor = neighbor_fn.allocate(positions)
    with self.assertRaises(ValueError):
      energy_fn(positions, neighbor)


if __name__ == '__main__':
  absltest.main()
