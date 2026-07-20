"""Tests for the bio-mlff pretrained model energy wrappers."""

from collections import namedtuple

from absl.testing import absltest, parameterized

import jax
import jax.numpy as jnp

from jax_md import energy, partition, space
from jax_md._nn.aceff import load_model as load_aceff
from jax_md._nn.ani import load_model as load_ani
from jax_md._nn.orb import load_model as load_orb
from jax_md._nn.so3lr import load_model as load_so3lr

jax.config.parse_flags_with_absl()

# x64 comes from JAX_ENABLE_X64 in CI; tolerances tighten with it.
X64 = jax.config.jax_enable_x64
DTYPE = jnp.float64 if X64 else jnp.float32
# Relative tolerance, so it holds across models of differing energy scale.
ENERGY_RTOL = 1e-12 if X64 else 5e-6
FORCE_TOL = 1e-8 if X64 else 1e-6


def water_molecule():
  positions = jnp.asarray(
    [
      [1.0, 1.0, 1.0],
      [1.9572, 1.0, 1.0],
      [0.760013, 1.926627, 1.0],
    ],
    dtype=DTYPE,
  )
  species = jnp.asarray([8, 1, 1], dtype=jnp.int32)
  return positions, species


Model = namedtuple(
  'Model', 'wrapper load call_bare reference places supports_sparse'
)

# Orb and SO3LR run fp32 internally, so their references are x64-invariant.
# ANI and SO3LR use one neighbor format only, so they skip the sparse check.
ORB = Model(
  wrapper=energy.orb_neighbor_list,
  load=load_orb,
  call_bare=lambda net, pos, sp: net(pos, sp, jnp.zeros((1,)), jnp.zeros((1,))),
  reference=-2078.838623,
  places=6,
  supports_sparse=True,
)
ACEFF = Model(
  wrapper=energy.aceff_neighbor_list,
  load=load_aceff,
  call_bare=lambda net, pos, sp: net(pos, sp),
  reference=-0.99136613 if X64 else -0.9913653,
  places=8 if X64 else 6,
  supports_sparse=True,
)
ANI = Model(
  wrapper=energy.ani_neighbor_list,
  load=load_ani,
  call_bare=lambda net, pos, sp: net(pos, sp),
  reference=-535.6742933 if X64 else -535.674255,
  places=6 if X64 else 4,
  supports_sparse=False,
)
SO3LR = Model(
  wrapper=energy.so3lr_neighbor_list,
  load=load_so3lr,
  call_bare=lambda net, pos, sp: net(pos, sp),
  reference=-5.014963,
  places=5,
  supports_sparse=False,
)
MODELS = (('orb', ORB), ('aceff', ACEFF), ('ani', ANI), ('so3lr', SO3LR))


class EnergyWrapperTest(parameterized.TestCase):
  @parameterized.named_parameters(*MODELS)
  def test_energy_matches_reference(self, model):
    # Boxes larger than the cutoff isolate the molecule, so all cells agree.
    positions, species = water_molecule()
    ortho = jnp.eye(3, dtype=DTYPE) * 100.0
    triclinic = jnp.swapaxes(
      jnp.asarray(
        [[90.0, 0.0, 0.0], [15.0, 95.0, 0.0], [7.0, 4.0, 100.0]],
        dtype=DTYPE,
      ),
      -1,
      -2,
    )
    free_disp, _ = space.free()
    per_disp, _ = space.periodic_general(ortho, fractional_coordinates=False)
    energies = []
    for disp, box in [(free_disp, ortho), (per_disp, ortho), (None, triclinic)]:
      neighbor_fn, energy_fn = model.wrapper(disp, box, species=species)
      energies.append(
        float(energy_fn(positions, neighbor_fn.allocate(positions)))
      )
    for e in energies[1:]:
      self.assertLessEqual(
        abs(e - energies[0]), ENERGY_RTOL * max(abs(e), abs(energies[0]))
      )
    self.assertAlmostEqual(energies[0], model.reference, places=model.places)

  @parameterized.named_parameters(*MODELS)
  def test_displacement_fn_drives_edges(self, model):
    # A bonded pair keeps the list non-empty; the third atom is only a
    # neighbor of the pair through the periodic image, so free and periodic
    # energies differ.
    positions = jnp.asarray(
      [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [13.0, 1.0, 1.0]], DTYPE
    )
    species = jnp.asarray([8, 1, 1], dtype=jnp.int32)
    box = jnp.eye(3, dtype=DTYPE) * 14.0

    free_disp, _ = space.free()
    nbr_free, free_fn = model.wrapper(free_disp, box, species=species)
    per_disp, _ = space.periodic_general(box, fractional_coordinates=False)
    nbr_per, per_fn = model.wrapper(per_disp, box, species=species)

    free = free_fn(positions, nbr_free.allocate(positions))
    per = per_fn(positions, nbr_per.allocate(positions))
    self.assertGreater(abs(float(free) - float(per)), 0.01)

  @parameterized.named_parameters(*MODELS)
  def test_sparse_format_matches_dense(self, model):
    if not model.supports_sparse:
      self.skipTest('model uses a Dense neighbor list only')
    positions, species = water_molecule()
    box = jnp.eye(3, dtype=DTYPE) * 100.0
    disp, _ = space.periodic_general(box, fractional_coordinates=False)
    nbr_dense, dense_fn = model.wrapper(disp, box, species=species)
    nbr_sparse, sparse_fn = model.wrapper(
      disp, box, species=species, format=partition.Sparse
    )

    e_dense = float(dense_fn(positions, nbr_dense.allocate(positions)))
    e_sparse = float(sparse_fn(positions, nbr_sparse.allocate(positions)))
    self.assertLessEqual(
      abs(e_dense - e_sparse), ENERGY_RTOL * max(abs(e_dense), abs(e_sparse))
    )

    f_dense = -jax.grad(lambda p: dense_fn(p, nbr_dense.allocate(p)))(positions)
    f_sparse = -jax.grad(lambda p: sparse_fn(p, nbr_sparse.allocate(p)))(
      positions
    )
    self.assertLess(float(jnp.max(jnp.abs(f_dense - f_sparse))), FORCE_TOL)

  @parameterized.named_parameters(*MODELS)
  def test_default_displacement_fn_is_periodic(self, model):
    positions, species = water_molecule()
    box = jnp.eye(3, dtype=DTYPE) * 100.0

    nbr_default, default_fn = model.wrapper(box=box, species=species)
    disp, _ = space.periodic_general(box, fractional_coordinates=False)
    nbr_explicit, explicit_fn = model.wrapper(disp, box, species=species)

    default = float(default_fn(positions, nbr_default.allocate(positions)))
    explicit = float(explicit_fn(positions, nbr_explicit.allocate(positions)))
    self.assertLessEqual(
      abs(default - explicit), ENERGY_RTOL * max(abs(default), abs(explicit))
    )

  @parameterized.named_parameters(*MODELS)
  def test_invalid_inputs_raise(self, model):
    positions, species = water_molecule()
    box = jnp.eye(3, dtype=DTYPE) * 100.0
    disp, _ = space.periodic_general(box, fractional_coordinates=False)

    with self.assertRaises(ValueError):
      model.wrapper()

    neighbor_fn, energy_fn = model.wrapper(disp, box)
    with self.assertRaises(ValueError):
      energy_fn(positions, neighbor_fn.allocate(positions))

    with self.assertRaises(ValueError):
      model.call_bare(model.load(), positions, species)


class Aimnet2Test(parameterized.TestCase):
  # AIMNet2 has two cutoffs and a frame-dependent Coulomb term, so it does not
  # join the shared parameterization. It forces fp64 internally, so its
  # references are x64-invariant.
  def test_energies(self):
    positions, species = water_molecule()
    ortho = jnp.eye(3, dtype=DTYPE) * 100.0
    triclinic = jnp.swapaxes(
      jnp.asarray(
        [[90.0, 0.0, 0.0], [15.0, 95.0, 0.0], [7.0, 4.0, 100.0]],
        dtype=DTYPE,
      ),
      -1,
      -2,
    )
    nbr_free, free_fn = energy.aimnet2_neighbor_list(
      box=ortho, species=species, periodic=False
    )
    per_disp, _ = space.periodic_general(ortho, fractional_coordinates=False)
    nbr_per, per_fn = energy.aimnet2_neighbor_list(
      per_disp, ortho, species=species
    )
    nbr_tri, tri_fn = energy.aimnet2_neighbor_list(
      None, triclinic, species=species
    )

    free = float(free_fn(positions, nbr_free.allocate(positions)))
    per = float(per_fn(positions, nbr_per.allocate(positions)))
    tri = float(tri_fn(positions, nbr_tri.allocate(positions)))

    self.assertAlmostEqual(free, -2081.046247, places=6)
    self.assertAlmostEqual(per, -2081.0533717, places=6)
    # Periodic frames agree; the damped Coulomb makes free differ from periodic.
    self.assertAlmostEqual(tri, per, places=3)
    self.assertGreater(abs(free - per), 0.001)

    # AIMNet2 forces fp64 internally, so gradients need x64 inputs.
    if X64:
      forces = -jax.grad(lambda p: per_fn(p, nbr_per.allocate(p)))(positions)
      self.assertTrue(bool(jnp.all(jnp.isfinite(forces))))

  def test_invalid_inputs_raise(self):
    positions, species = water_molecule()
    box = jnp.eye(3, dtype=DTYPE) * 100.0
    disp, _ = space.periodic_general(box, fractional_coordinates=False)

    with self.assertRaises(ValueError):
      energy.aimnet2_neighbor_list()

    neighbor_fn, energy_fn = energy.aimnet2_neighbor_list(disp, box)
    with self.assertRaises(ValueError):
      energy_fn(positions, neighbor_fn.allocate(positions))

    # Free space uses all-pairs Coulomb and ignores a displacement_fn, so
    # passing one with periodic=False is rejected rather than silently dropped.
    with self.assertRaises(ValueError):
      energy.aimnet2_neighbor_list(disp, box, species=species, periodic=False)


if __name__ == '__main__':
  absltest.main()
