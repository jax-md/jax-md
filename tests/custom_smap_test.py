"""Tests for jax_md.custom_smap (multi-image neighbor list smap functions).

Note: custom_smap currently only supports pair potentials via
`pair_neighbor_list_multi_image`. Three-body/triplet potentials are not yet
implemented for multi-image neighbor lists.
"""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
import jax.numpy as np

from jax_md import smap, space, partition, quantity
from jax_md.custom_partition import (
  neighbor_list_multi_image,
  estimate_max_neighbors_from_box,
  NeighborListFormat,
)
from jax_md.custom_smap import pair_neighbor_list_multi_image
from jax_md.util import f32, f64, i32
from jax_md import test_util

jax.config.parse_flags_with_absl()

test_util.update_test_tolerance(f32_tol=5e-5, f64_tol=1e-13)

STOCHASTIC_SAMPLES = 3
SPATIAL_DIMENSION = [2, 3]
NEIGHBOR_LIST_FORMAT = [
  NeighborListFormat.Dense,
  NeighborListFormat.Sparse,
  NeighborListFormat.OrderedSparse,
]

# Smaller particle count for multi-image tests (small boxes)
PARTICLE_COUNT = 50

if jax.config.jax_enable_x64:
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]


class CustomSmapTest(test_util.JAXMDTestCase):
  """Tests for pair_neighbor_list_multi_image."""

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_format={format}',
        'spatial_dimension': dim,
        'dtype': dtype,
        'format': format,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
      # Only test Sparse formats for comparison; Dense has different handling
      for format in [
        NeighborListFormat.Sparse,
        NeighborListFormat.OrderedSparse,
      ]
    )
  )
  def test_pair_multi_image_matches_standard_large_box(
    self, spatial_dimension, dtype, format
  ):
    """Test that multi-image matches standard smap for large boxes (r_cut < L/2)."""
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr**2, f32(0.0))

    N = PARTICLE_COUNT
    # Large box so MIC is valid (r_cut < L/2)
    box_size = 4.0 * N ** (1.0 / spatial_dimension)
    sigma = 1.5

    key, split = random.split(key)
    disp, _ = space.periodic(box_size)
    metric = space.metric(disp)

    # Standard smap
    standard_energy = jax.jit(
      smap.pair_neighbor_list(truncated_square, metric, sigma=sigma)
    )

    # Multi-image smap
    multi_image_energy = jax.jit(
      pair_neighbor_list_multi_image(
        truncated_square, sigma=sigma, fractional_coordinates=True
      )
    )

    for _ in range(STOCHASTIC_SAMPLES):
      key, split = random.split(key)
      R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
      # Convert to fractional coordinates for multi-image
      R_frac = R / box_size

      # Standard neighbor list
      neighbor_fn = partition.neighbor_list(
        disp, box_size, sigma, 0.0, format=format
      )
      nbrs = neighbor_fn.allocate(R)

      # Multi-image neighbor list
      box = np.eye(spatial_dimension, dtype=dtype) * box_size
      max_neighbors = estimate_max_neighbors_from_box(box, sigma, N)
      mi_neighbor_fn = neighbor_list_multi_image(
        None, box, sigma, max_neighbors=max_neighbors, format=format
      )
      mi_nbrs = mi_neighbor_fn.allocate(R_frac)

      E_standard = standard_energy(R, nbrs)
      E_multi_image = multi_image_energy(R_frac, mi_nbrs)

      # Check values match with appropriate tolerance for float32
      self.assertAllClose(
        E_standard, E_multi_image, check_dtypes=False, rtol=1e-5, atol=1e-5
      )

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_format={format}',
        'spatial_dimension': dim,
        'dtype': dtype,
        'format': format,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
      for format in NEIGHBOR_LIST_FORMAT
    )
  )
  def test_pair_multi_image_small_box(self, spatial_dimension, dtype, format):
    """Test multi-image for small boxes where r_cut > L/2."""
    key = random.PRNGKey(42)

    def soft_pair(dr, sigma=1.0):
      """Soft potential that doesn't diverge at short distances."""
      return np.where(dr < sigma, (1.0 - dr / sigma) ** 2, f32(0.0))

    # Small box: 4 atoms in a 2x2x2 (or 2x2 for 2D) box
    N = 4
    box_size = 2.0
    sigma = 1.5
    r_cutoff = 2.5  # > L/2 = 1.0, so MIC will miss images

    key, split = random.split(key)
    # Place atoms at corners of a small cube/square (well-separated)
    if spatial_dimension == 3:
      R_frac = np.array(
        [
          [0.25, 0.25, 0.25],
          [0.75, 0.25, 0.25],
          [0.25, 0.75, 0.25],
          [0.25, 0.25, 0.75],
        ],
        dtype=dtype,
      )
    else:
      R_frac = np.array(
        [
          [0.25, 0.25],
          [0.75, 0.25],
          [0.25, 0.75],
          [0.75, 0.75],
        ],
        dtype=dtype,
      )

    box = np.eye(spatial_dimension, dtype=dtype) * box_size

    # Multi-image should find more neighbors than MIC for this setup
    max_neighbors = estimate_max_neighbors_from_box(box, r_cutoff, N)
    mi_neighbor_fn = neighbor_list_multi_image(
      None, box, r_cutoff, max_neighbors=max_neighbors, format=format
    )
    mi_nbrs = mi_neighbor_fn.allocate(R_frac)

    # Multi-image energy
    multi_image_energy = pair_neighbor_list_multi_image(
      soft_pair, sigma=sigma, fractional_coordinates=True
    )
    E_mi = multi_image_energy(R_frac, mi_nbrs)

    # Verify we get a finite, positive energy
    self.assertTrue(np.isfinite(E_mi))
    self.assertGreater(float(E_mi), 0.0)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_format={format}',
        'spatial_dimension': dim,
        'dtype': dtype,
        'format': format,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
      for format in [NeighborListFormat.Sparse, NeighborListFormat.Dense]
    )
  )
  def test_pair_multi_image_species(self, spatial_dimension, dtype, format):
    """Test species-dependent parameters."""
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr**2, f32(0.0))

    N = PARTICLE_COUNT
    box_size = 4.0 * N ** (1.0 / spatial_dimension)

    # Create species array (half type 0, half type 1)
    species = np.zeros((N,), i32)
    species = np.where(np.arange(N) >= N // 2, 1, species)

    # Species-dependent sigma
    sigma_matrix = np.array([[1.0, 1.5], [1.5, 2.0]], dtype=dtype)

    key, split = random.split(key)
    R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
    R_frac = R / box_size

    box = np.eye(spatial_dimension, dtype=dtype) * box_size
    max_neighbors = estimate_max_neighbors_from_box(
      box, float(np.max(sigma_matrix)), N
    )

    mi_neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      float(np.max(sigma_matrix)),
      max_neighbors=max_neighbors,
      format=format,
    )
    mi_nbrs = mi_neighbor_fn.allocate(R_frac)

    # Energy with species
    energy_fn = pair_neighbor_list_multi_image(
      truncated_square,
      species=species,
      sigma=sigma_matrix,
      fractional_coordinates=True,
    )
    E = energy_fn(R_frac, mi_nbrs)

    self.assertTrue(np.isfinite(E))

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_format={format}',
        'spatial_dimension': dim,
        'dtype': dtype,
        'format': format,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
      # OrderedSparse doesn't support per-particle energies
      for format in [NeighborListFormat.Sparse, NeighborListFormat.Dense]
    )
  )
  def test_pair_multi_image_per_particle(
    self, spatial_dimension, dtype, format
  ):
    """Test per-particle energy reduction."""
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr**2, f32(0.0))

    N = PARTICLE_COUNT
    box_size = 4.0 * N ** (1.0 / spatial_dimension)
    sigma = 1.5

    key, split = random.split(key)
    R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
    R_frac = R / box_size

    box = np.eye(spatial_dimension, dtype=dtype) * box_size
    max_neighbors = estimate_max_neighbors_from_box(box, sigma, N)

    mi_neighbor_fn = neighbor_list_multi_image(
      None, box, sigma, max_neighbors=max_neighbors, format=format
    )
    mi_nbrs = mi_neighbor_fn.allocate(R_frac)

    # Total energy
    total_energy_fn = pair_neighbor_list_multi_image(
      truncated_square, sigma=sigma, fractional_coordinates=True
    )
    E_total = total_energy_fn(R_frac, mi_nbrs)

    # Per-particle energy
    per_particle_energy_fn = pair_neighbor_list_multi_image(
      truncated_square,
      sigma=sigma,
      reduce_axis=(1,),
      fractional_coordinates=True,
    )
    E_per_particle = per_particle_energy_fn(R_frac, mi_nbrs)

    # Sum of per-particle should equal total
    self.assertEqual(E_per_particle.shape, (N,))
    self.assertAllClose(np.sum(E_per_particle), E_total)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}',
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
    )
  )
  def test_pair_multi_image_ordered_sparse_per_particle_raises(
    self, spatial_dimension, dtype
  ):
    """Test that OrderedSparse raises ValueError for per-particle energies."""
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr**2, f32(0.0))

    N = 10
    box_size = 5.0
    sigma = 1.5

    R_frac = random.uniform(key, (N, spatial_dimension), dtype=dtype)
    box = np.eye(spatial_dimension, dtype=dtype) * box_size

    mi_neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      sigma,
      max_neighbors=50,
      format=NeighborListFormat.OrderedSparse,
    )
    mi_nbrs = mi_neighbor_fn.allocate(R_frac)

    per_particle_energy_fn = pair_neighbor_list_multi_image(
      truncated_square,
      sigma=sigma,
      reduce_axis=(1,),
      fractional_coordinates=True,
    )

    with self.assertRaises(ValueError):
      per_particle_energy_fn(R_frac, mi_nbrs)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_format={format}',
        'spatial_dimension': dim,
        'dtype': dtype,
        'format': format,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
      for format in NEIGHBOR_LIST_FORMAT
    )
  )
  def test_pair_multi_image_forces(self, spatial_dimension, dtype, format):
    """Test gradient computation (forces)."""
    key = random.PRNGKey(0)

    def lj_pair(dr, sigma=1.0, epsilon=1.0):
      """Simple LJ potential."""
      idr = sigma / dr
      idr6 = idr**6
      idr12 = idr6**2
      return f32(4.0) * epsilon * (idr12 - idr6)

    N = 20
    box_size = 5.0
    sigma = 1.0
    r_cutoff = 2.5

    key, split = random.split(key)
    R_frac = random.uniform(split, (N, spatial_dimension), dtype=dtype)
    box = np.eye(spatial_dimension, dtype=dtype) * box_size

    max_neighbors = estimate_max_neighbors_from_box(box, r_cutoff, N)
    mi_neighbor_fn = neighbor_list_multi_image(
      None, box, r_cutoff, max_neighbors=max_neighbors, format=format
    )
    mi_nbrs = mi_neighbor_fn.allocate(R_frac)

    energy_fn = pair_neighbor_list_multi_image(
      lj_pair, sigma=sigma, epsilon=1.0, fractional_coordinates=True
    )

    # Compute forces via autodiff
    force_fn = quantity.force(energy_fn)
    F = force_fn(R_frac, mi_nbrs)

    # Forces should have correct shape
    self.assertEqual(F.shape, (N, spatial_dimension))
    # Forces should be finite
    self.assertTrue(np.all(np.isfinite(F)))

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_format={format}',
        'spatial_dimension': dim,
        'dtype': dtype,
        'format': format,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
      for format in NEIGHBOR_LIST_FORMAT
    )
  )
  def test_pair_multi_image_cartesian_coordinates(
    self, spatial_dimension, dtype, format
  ):
    """Test with fractional_coordinates=False."""
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr**2, f32(0.0))

    N = PARTICLE_COUNT
    box_size = 4.0 * N ** (1.0 / spatial_dimension)
    sigma = 1.5

    key, split = random.split(key)
    R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)

    box = np.eye(spatial_dimension, dtype=dtype) * box_size
    max_neighbors = estimate_max_neighbors_from_box(box, sigma, N)

    mi_neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      sigma,
      max_neighbors=max_neighbors,
      format=format,
      fractional_coordinates=False,  # Use Cartesian
    )
    mi_nbrs = mi_neighbor_fn.allocate(R)

    energy_fn = pair_neighbor_list_multi_image(
      truncated_square, sigma=sigma, fractional_coordinates=False
    )
    E = energy_fn(R, mi_nbrs)

    self.assertTrue(np.isfinite(E))

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_format={format}',
        'spatial_dimension': dim,
        'dtype': dtype,
        'format': format,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
      for format in NEIGHBOR_LIST_FORMAT
    )
  )
  def test_pair_multi_image_dynamic_kwargs(
    self, spatial_dimension, dtype, format
  ):
    """Test that kwargs can be overridden at call time."""
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr**2, f32(0.0))

    N = PARTICLE_COUNT
    box_size = 4.0 * N ** (1.0 / spatial_dimension)
    sigma_static = 1.5
    sigma_dynamic = 2.0

    key, split = random.split(key)
    R_frac = random.uniform(split, (N, spatial_dimension), dtype=dtype)
    box = np.eye(spatial_dimension, dtype=dtype) * box_size

    max_neighbors = estimate_max_neighbors_from_box(box, sigma_dynamic, N)
    mi_neighbor_fn = neighbor_list_multi_image(
      None, box, sigma_dynamic, max_neighbors=max_neighbors, format=format
    )
    mi_nbrs = mi_neighbor_fn.allocate(R_frac)

    # Create with static sigma
    energy_fn = pair_neighbor_list_multi_image(
      truncated_square, sigma=sigma_static, fractional_coordinates=True
    )

    # Call with static sigma
    E_static = energy_fn(R_frac, mi_nbrs)

    # Call with dynamic sigma (should override)
    E_dynamic = energy_fn(R_frac, mi_nbrs, sigma=sigma_dynamic)

    # Energies should differ
    self.assertFalse(np.allclose(E_static, E_dynamic))

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}',
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
    )
  )
  def test_pair_multi_image_dense_large_box(self, spatial_dimension, dtype):
    """Test Dense format produces finite energies for large boxes."""
    key = random.PRNGKey(0)

    def truncated_square(dr, sigma):
      return np.where(dr < sigma, dr**2, f32(0.0))

    N = PARTICLE_COUNT
    box_size = 4.0 * N ** (1.0 / spatial_dimension)
    sigma = 1.5

    key, split = random.split(key)
    R_frac = random.uniform(split, (N, spatial_dimension), dtype=dtype)
    box = np.eye(spatial_dimension, dtype=dtype) * box_size

    max_neighbors = estimate_max_neighbors_from_box(box, sigma, N)
    mi_neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      sigma,
      max_neighbors=max_neighbors,
      format=NeighborListFormat.Dense,
    )
    mi_nbrs = mi_neighbor_fn.allocate(R_frac)

    energy_fn = pair_neighbor_list_multi_image(
      truncated_square, sigma=sigma, fractional_coordinates=True
    )
    E = energy_fn(R_frac, mi_nbrs)

    # Energy should be finite and positive (truncated_square is positive)
    self.assertTrue(np.isfinite(E))
    self.assertGreater(float(E), 0.0)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}',
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in POSITION_DTYPE
    )
  )
  def test_multi_image_finds_more_neighbors_small_box(
    self, spatial_dimension, dtype
  ):
    """Verify multi-image finds more neighbors than MIC for small boxes."""
    key = random.PRNGKey(123)

    # Small box where r_cutoff > L/2
    N = 8
    box_size = 2.0
    r_cutoff = 2.5  # > L/2 = 1.0

    key, split = random.split(key)
    R = box_size * random.uniform(split, (N, spatial_dimension), dtype=dtype)
    R_frac = R / box_size

    # Standard MIC neighbor list
    disp, _ = space.periodic(box_size)
    mic_neighbor_fn = partition.neighbor_list(
      disp, box_size, r_cutoff, 0.0, format=partition.Sparse
    )
    mic_nbrs = mic_neighbor_fn.allocate(R)
    mic_n_edges = int(np.sum(mic_nbrs.idx[0] < N))

    # Multi-image neighbor list
    box = np.eye(spatial_dimension, dtype=dtype) * box_size
    max_neighbors = estimate_max_neighbors_from_box(box, r_cutoff, N)
    mi_neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      r_cutoff,
      max_neighbors=max_neighbors,
      format=NeighborListFormat.Sparse,
    )
    mi_nbrs = mi_neighbor_fn.allocate(R_frac)
    mi_n_edges = int(np.sum(mi_nbrs.idx[0] < N))

    # Multi-image should find at least as many neighbors
    self.assertGreaterEqual(mi_n_edges, mic_n_edges)


if __name__ == '__main__':
  absltest.main()
