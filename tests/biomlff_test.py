"""Tests for the bio-mlff pretrained model energy wrappers."""

from collections import namedtuple

from absl.testing import absltest, parameterized

import jax
import jax.numpy as jnp
import numpy as onp

from jax_md import energy, partition, space
from jax_md.units import EV_TO_KJMOL
from jax_md._nn.aceff import load_model as load_aceff
from jax_md._nn.aimnet2 import load_model as load_aimnet2
from jax_md._nn.ani import load_model as load_ani
from jax_md._nn.orb import load_model as load_orb
from jax_md._nn.so3lr import load_model as load_so3lr

jax.config.parse_flags_with_absl()

# These models run in float64, so the tests require x64 and skip otherwise.
X64 = jax.config.jax_enable_x64
DTYPE = jnp.float64
# Internal-consistency tolerances, magnitude-independent; machine precision.
ENERGY_RTOL = 1e-12
FORCE_TOL = 1e-10
# Reference tolerances mirror bio-mlff's Test*Potential.py: energy and force
# compared at rtol 1e-10 with a per-model force atol. so3lr's JAX
# reimplementation differs from the upstream so3lr package by ~6e-10 energy and
# ~3e-6 force -- a floor bio-mlff shares -- so it carries looser per-model bounds.
FORCE_RTOL = 1e-10


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
  # bio-mlff's gas-phase toluene test system C7H8.
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
  'Model',
  'wrapper load call_bare make_free toluene_reference_kjmol '
  'toluene_forces_kjmol energy_rtol force_atol supports_sparse',
)

# supports_sparse is False for Dense-only models, which skip the sparse check.
ORB = Model(
  wrapper=energy.orb_neighbor_list,
  load=load_orb,
  call_bare=lambda net, pos, sp: net(pos, sp, jnp.zeros((1,)), jnp.zeros((1,))),
  make_free=lambda w, box, sp: w(space.free()[0], box, species=sp),
  toluene_reference_kjmol=-712903.5201361555,
  toluene_forces_kjmol=jnp.asarray(
    [
      [31.678148650446897, -108.76705081851658, 79.24784592869504],
      [-17.92525780732101, 43.31598460640205, -25.13621578418634],
      [77.02721959249759, 72.77552722820194, 10.537647369809816],
      [3.386397932114493, 79.86516657068701, 11.312697384681043],
      [-26.067509212818656, 22.338510713456184, 0.7019760213890679],
      [-64.17431281667433, -120.135334063762, -11.0830915150865],
      [1.70032631732334, 3.7989736025772203, 1.2219939911255717],
      [10.361223024400422, 38.47509888458903, -59.678345921213726],
      [27.60875242570873, -14.992980173221062, -20.533945044733912],
      [-10.749742943237441, 67.93321059829024, 27.624383525855748],
      [-46.253459758516854, -24.00776164774232, -5.914103497214447],
      [7.1375104864533325, -54.4910181498578, -5.300857442208508],
      [13.31345476986596, -20.755145810441118, -2.0137720312422034],
      [16.180758719422066, 31.75305923469404, 3.14189754676992],
      [-23.223509379664506, -17.106240775356806, -4.128110532440591],
    ],
    dtype=DTYPE,
  ),
  energy_rtol=1e-10,
  force_atol=0.0,
  supports_sparse=True,
)
ACEFF = Model(
  wrapper=energy.aceff_neighbor_list,
  load=load_aceff,
  call_bare=lambda net, pos, sp: net(pos, sp),
  make_free=lambda w, box, sp: w(space.free()[0], box, species=sp),
  toluene_reference_kjmol=-362.46118331266337,
  toluene_forces_kjmol=jnp.asarray(
    [
      [38.74614589854023, -109.03280463335601, 79.99649527971694],
      [-21.099402500070838, 43.28152263971333, -24.955982305391405],
      [78.80186200466132, 74.14524471250908, 11.108330460974434],
      [-0.4746447106279367, 81.58622979724302, 10.763326016945426],
      [-28.908497754458857, 23.364533614722312, 0.8725064090450448],
      [-68.34819699053114, -122.41713874755104, -11.594921957064637],
      [2.897827145745934, 2.996736448223807, 1.4061060484990067],
      [9.517010138957543, 39.66766049689898, -62.386010520394336],
      [25.14265925853048, -18.44019167450396, -20.52802276688851],
      [-12.326400711304684, 69.76355229528373, 29.57936255825579],
      [-51.72521604459046, -30.694180316959567, -6.870325508611947],
      [11.370470946253501, -62.37528869561318, -5.932727811084642],
      [21.92489053146936, -21.756861592090257, -1.6224987650984855],
      [21.686621354595292, 39.244131305651244, 3.8882291076766213],
      [-27.205128567169766, -9.33314565017151, -3.7238662465792807],
    ],
    dtype=DTYPE,
  ),
  energy_rtol=1e-10,
  force_atol=5e-11,
  supports_sparse=True,
)
ANI = Model(
  wrapper=energy.ani_neighbor_list,
  load=load_ani,
  call_bare=lambda net, pos, sp: net(pos, sp),
  make_free=lambda w, box, sp: w(space.free()[0], box, species=sp),
  toluene_reference_kjmol=-712776.6445064595,
  toluene_forces_kjmol=jnp.asarray(
    [
      [28.668967613700588, -114.19465652328087, 86.12166468597633],
      [-14.649727478721127, 46.91878977768792, -25.21626641059562],
      [89.52693762457135, 78.15270069609218, 12.592046589048367],
      [-0.8174457951499045, 89.34171652314355, 9.08754464042536],
      [-33.18988682754674, 19.887881325674588, -0.051341353808798944],
      [-75.86326926188656, -125.39726212485584, -13.954032174934065],
      [5.51703341856447, -5.650177509830918, 0.9408517054149147],
      [17.824238445432076, 35.56609948813367, -51.629525484441935],
      [37.94214508271477, -3.745892914315296, -23.028536995781064],
      [-3.5936551883049543, 64.42833854986405, 16.130701944774504],
      [-38.85570380077297, -11.337192522479096, -3.442580133594403],
      [-0.7563920131007629, -40.09114183422332, -3.9922307041311216],
      [-3.1168844349216083, -20.056436203677436, -1.857487469221962],
      [7.502651913270668, 18.18067455603506, 1.806031143151201],
      [-16.1390092978493, -32.0034412839682, -3.5068399822816803],
    ],
    dtype=DTYPE,
  ),
  energy_rtol=1e-10,
  force_atol=3e-6,
  supports_sparse=False,
)
SO3LR = Model(
  wrapper=energy.so3lr_neighbor_list,
  load=load_so3lr,
  call_bare=lambda net, pos, sp: net(pos, sp),
  make_free=lambda w, box, sp: w(space.free()[0], box, species=sp),
  toluene_reference_kjmol=-2528.822629129894,
  toluene_forces_kjmol=jnp.asarray(
    [
      [10.77457079448285, -105.89226613431423, 77.83098075274768],
      [-5.289749937929357, 43.02859953566828, -25.34350016470746],
      [69.18782800849404, 68.15623581211693, 9.51551054146353],
      [9.987989347286057, 70.94650916026923, 9.719015497353277],
      [-17.300597165852754, 20.30208945518387, 0.8020660876427157],
      [-57.49559513647594, -111.03029623178688, -9.753654449846284],
      [-5.172349097585983, 11.134052349154427, 0.9235882689124691],
      [13.017474829500404, 34.52190693028682, -55.01350697336072],
      [27.00309092425673, -10.20297626489736, -20.94768766007825],
      [-8.252906006904334, 64.4213611462904, 25.383090518104563],
      [-43.46069344263362, -21.18018053215143, -5.44805781073874],
      [5.727782447252731, -50.12402807339619, -4.598174823428787],
      [8.27361509565024, -19.9593035641137, -1.5504097927011973],
      [13.331241339074761, 27.515687265225385, 2.66371616175272],
      [-20.331701998615802, -21.637390853535504, -4.182976153115509],
    ],
    dtype=DTYPE,
  ),
  energy_rtol=5e-9,
  force_atol=2e-5,
  supports_sparse=False,
)
AIMNET2 = Model(
  wrapper=energy.aimnet2_neighbor_list,
  load=load_aimnet2,
  call_bare=lambda net, pos, sp: net(pos, sp),
  make_free=lambda w, box, sp: w(box=box, species=sp, periodic=False),
  toluene_reference_kjmol=-713468.0030672933,
  toluene_forces_kjmol=jnp.asarray(
    [
      [42.11829378790869, -107.61867986535026, 78.5741554335958],
      [-20.658371112147606, 42.83037314293471, -27.479174389445095],
      [79.35069187344195, 76.97149898336036, 13.527849580806938],
      [5.048907750299418, 84.44353725054836, 9.088435468943503],
      [-30.451554700795356, 21.738130280384453, 0.5429969184803164],
      [-69.87582305696036, -124.01374480431299, -13.364971556804617],
      [1.1839383566043582, 0.09611847965489832, 2.4219602801120694],
      [9.999424905326784, 40.266031796405436, -65.97822640750829],
      [25.274389965382362, -22.722898853671364, -20.081087806076713],
      [-11.953487306612844, 72.6822933895183, 33.45699066238386],
      [-52.62549255047514, -28.923157598454065, -5.815059892286944],
      [9.726458113937042, -60.1822839203639, -5.36207189857633],
      [20.584680715504284, -22.2829431265299, -1.1383103605242761],
      [19.132438209739934, 36.13966671867287, 3.9251228919101604],
      [-26.854494951153633, -9.42394187279693, -2.3186089250104005],
    ],
    dtype=DTYPE,
  ),
  energy_rtol=1e-10,
  force_atol=0.0,
  supports_sparse=False,
)
MODELS = (
  ('orb', ORB),
  ('aceff', ACEFF),
  ('ani', ANI),
  ('so3lr', SO3LR),
  ('aimnet2', AIMNET2),
)


@absltest.skipUnless(
  X64, 'bio-mlff models run in float64; set JAX_ENABLE_X64=1'
)
class EnergyWrapperTest(parameterized.TestCase):
  @parameterized.named_parameters(*MODELS)
  def test_periodic_frames_agree(self, model):
    # An isolated molecule has the same energy in any box past the cutoff.
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
    per_disp, _ = space.periodic_general(ortho, fractional_coordinates=False)
    nbr_o, fn_o = model.wrapper(per_disp, ortho, species=species)
    nbr_t, fn_t = model.wrapper(None, triclinic, species=species)
    e_o = float(fn_o(positions, nbr_o.allocate(positions)))
    e_t = float(fn_t(positions, nbr_t.allocate(positions)))
    self.assertLessEqual(abs(e_o - e_t), ENERGY_RTOL * max(abs(e_o), abs(e_t)))

  @parameterized.named_parameters(*MODELS)
  def test_toluene_matches_reference(self, model):
    # jax-md energy in eV -> kJ/mol vs the reference implementation.
    positions, species = toluene()
    box = jnp.eye(3, dtype=DTYPE) * 100.0
    neighbor_fn, energy_fn = model.make_free(model.wrapper, box, species)
    e_kjmol = (
      float(energy_fn(positions, neighbor_fn.allocate(positions))) * EV_TO_KJMOL
    )
    self.assertLessEqual(
      abs(e_kjmol - model.toluene_reference_kjmol),
      model.energy_rtol * abs(model.toluene_reference_kjmol),
    )

  @parameterized.named_parameters(*MODELS)
  def test_toluene_forces_match_reference(self, model):
    # jax-md forces in eV/A -> kJ/mol/A vs the reference implementation.
    positions, species = toluene()
    box = jnp.eye(3, dtype=DTYPE) * 100.0
    neighbor_fn, energy_fn = model.make_free(model.wrapper, box, species)
    forces = -jax.grad(lambda p: energy_fn(p, neighbor_fn.allocate(p)))(
      positions
    )
    forces_kjmol = forces * EV_TO_KJMOL
    onp.testing.assert_allclose(
      onp.asarray(model.toluene_forces_kjmol),
      onp.asarray(forces_kjmol),
      rtol=FORCE_RTOL,
      atol=model.force_atol,
    )

  @parameterized.named_parameters(*MODELS)
  def test_displacement_fn_drives_edges(self, model):
    # The third atom is a neighbor only via the periodic image.
    positions = jnp.asarray(
      [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [13.0, 1.0, 1.0]], DTYPE
    )
    species = jnp.asarray([8, 1, 1], dtype=jnp.int32)
    box = jnp.eye(3, dtype=DTYPE) * 14.0

    nbr_free, free_fn = model.make_free(model.wrapper, box, species)
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

  def test_aimnet2_rejects_free_with_displacement(self):
    # Free space ignores a displacement_fn, so periodic=False rejects one.
    _, species = water_molecule()
    box = jnp.eye(3, dtype=DTYPE) * 100.0
    disp, _ = space.periodic_general(box, fractional_coordinates=False)
    with self.assertRaises(ValueError):
      energy.aimnet2_neighbor_list(disp, box, species=species, periodic=False)


if __name__ == '__main__':
  absltest.main()
