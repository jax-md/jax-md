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

# CI runs every suite with JAX_ENABLE_X64=1. These models run in float64, so
# the tests require x64 and skip otherwise (see the class decorators below).
X64 = jax.config.jax_enable_x64
DTYPE = jnp.float64
# Relative tolerances, magnitude-independent; machine precision in f64.
ENERGY_RTOL = 1e-12
REFERENCE_RTOL = 1e-10
FORCE_TOL = 1e-10


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


def toluene():
  # bio-mlff's gas-phase toluene test system (C7H8): a larger, carbon-bearing
  # molecule that exercises more of each model than water does.
  positions = jnp.asarray(
    [
      [2.199, -0.143, 0.062],
      [0.713, -0.073, 0.011],
      [0.076, 1.155, 0.101],
      [-1.298, 1.288, 0.060],
      [-2.077, 0.159, -0.076],
      [-1.445, -1.066, -0.167],
      [-0.070, -1.200, -0.126],
      [2.463, -0.598, 1.048],
      [2.621, 0.858, 0.021],
      [2.604, -0.841, -0.701],
      [0.727, 2.035, 0.209],
      [-1.731, 2.295, 0.138],
      [-3.158, 0.259, -0.109],
      [-2.051, -1.972, -0.275],
      [0.429, -2.158, -0.196],
    ],
    dtype=DTYPE,
  )
  species = jnp.asarray(
    [6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1], jnp.int32
  )
  return positions, species


Model = namedtuple(
  'Model', 'wrapper load call_bare reference toluene_reference supports_sparse'
)

# ANI and SO3LR use one neighbor format only, so they skip the sparse check.
ORB = Model(
  wrapper=energy.orb_neighbor_list,
  load=load_orb,
  call_bare=lambda net, pos, sp: net(pos, sp, jnp.zeros((1,)), jnp.zeros((1,))),
  reference=-2078.838596045984,
  toluene_reference=-7387.069418860392,
  supports_sparse=True,
)
ACEFF = Model(
  wrapper=energy.aceff_neighbor_list,
  load=load_aceff,
  call_bare=lambda net, pos, sp: net(pos, sp),
  reference=-0.9913662521142793,
  toluene_reference=-3.756645443780321,
  supports_sparse=True,
)
ANI = Model(
  wrapper=energy.ani_neighbor_list,
  load=load_ani,
  call_bare=lambda net, pos, sp: net(pos, sp),
  reference=-535.6743021328892,
  toluene_reference=-3516.8905524992483,
  supports_sparse=False,
)
SO3LR = Model(
  wrapper=energy.so3lr_neighbor_list,
  load=load_so3lr,
  call_bare=lambda net, pos, sp: net(pos, sp),
  reference=-5.014962248099789,
  toluene_reference=-26.209399658104502,
  supports_sparse=False,
)
MODELS = (('orb', ORB), ('aceff', ACEFF), ('ani', ANI), ('so3lr', SO3LR))


@absltest.skipUnless(
  X64, 'bio-mlff models run in float64; set JAX_ENABLE_X64=1'
)
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
    self.assertLessEqual(
      abs(energies[0] - model.reference),
      REFERENCE_RTOL * abs(model.reference),
    )

  @parameterized.named_parameters(*MODELS)
  def test_toluene_matches_reference(self, model):
    # bio-mlff's toluene system exercises carbon and a larger graph; the
    # reference was verified equal to bio-mlff's own JAX energy.
    positions, species = toluene()
    box = jnp.eye(3, dtype=DTYPE) * 100.0
    free_disp, _ = space.free()
    neighbor_fn, energy_fn = model.wrapper(free_disp, box, species=species)
    e = float(energy_fn(positions, neighbor_fn.allocate(positions)))
    self.assertLessEqual(
      abs(e - model.toluene_reference),
      REFERENCE_RTOL * abs(model.toluene_reference),
    )

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


@absltest.skipUnless(
  X64, 'bio-mlff models run in float64; set JAX_ENABLE_X64=1'
)
class Aimnet2Test(parameterized.TestCase):
  # AIMNet2 has two cutoffs and a frame-dependent Coulomb term, so it does not
  # join the shared parameterization.
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

    free_ref = -2081.0462439761864
    per_ref = -2079.8795450872276
    self.assertLessEqual(abs(free - free_ref), REFERENCE_RTOL * abs(free_ref))
    self.assertLessEqual(abs(per - per_ref), REFERENCE_RTOL * abs(per_ref))
    # Periodic frames agree; the damped Coulomb makes free differ from periodic.
    self.assertLessEqual(abs(tri - per), ENERGY_RTOL * abs(per))
    self.assertGreater(abs(free - per), 0.001)

    forces = -jax.grad(lambda p: per_fn(p, nbr_per.allocate(p)))(positions)
    self.assertTrue(bool(jnp.all(jnp.isfinite(forces))))

  def test_toluene_matches_reference(self):
    positions, species = toluene()
    box = jnp.eye(3, dtype=DTYPE) * 100.0
    nbr, energy_fn = energy.aimnet2_neighbor_list(
      box=box, species=species, periodic=False
    )
    e = float(energy_fn(positions, nbr.allocate(positions)))
    ref = -7394.574774904318
    self.assertLessEqual(abs(e - ref), REFERENCE_RTOL * abs(ref))

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
