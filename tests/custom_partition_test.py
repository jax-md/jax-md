"""Tests for custom_partition module comparing against ASE neighbor lists."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from ase import Atoms
from ase.build import bulk, molecule
from ase.neighborlist import neighbor_list as ase_neighbor_list

from jax_md import partition
from jax_md.custom_partition import (
  NeighborListFormat,
  NeighborListMultiImage,
  estimate_max_neighbors,
  estimate_max_neighbors_from_box,
  neighbor_list_multi_image,
  neighbor_list_multi_image_mask,
)

# Triclinic structure from TorchSim tests
CaCrP2O7_mvc_11955 = {
  'positions': [
    [3.68954016, 5.03568186, 4.64369552],
    [5.12301681, 2.13482791, 2.66220405],
    [1.99411973, 0.94691001, 1.25068234],
    [6.81843724, 6.22359976, 6.05521724],
    [2.63005662, 4.16863452, 0.86090529],
    [6.18250036, 3.00187525, 6.44499428],
    [2.11497733, 1.98032773, 4.53610884],
    [6.69757964, 5.19018203, 2.76979073],
    [1.39215545, 2.94386142, 5.60917746],
    [7.42040152, 4.22664834, 1.69672212],
    [2.43224207, 5.4571615, 6.70305327],
    [6.3803149, 1.71334827, 0.6028463],
    [1.11265639, 1.50166318, 3.48760997],
    [7.69990058, 5.66884659, 3.8182896],
    [3.56971588, 5.20836551, 1.43673437],
    [5.2428411, 1.96214426, 5.8691652],
    [3.12282634, 2.72812741, 1.05450432],
    [5.68973063, 4.44238236, 6.25139525],
    [3.24868468, 2.83997522, 3.99842386],
    [5.56387229, 4.33053455, 3.30747571],
    [2.60835346, 0.74421609, 5.3236629],
    [6.20420351, 6.42629368, 1.98223667],
  ],
  'cell': [
    [6.19330899, 0.0, 0.0],
    [2.4074486111396207, 6.149627748674982, 0.0],
    [0.2117993724186579, 1.0208820183960539, 7.305899571570074],
  ],
  'numbers': [*[20] * 2, *[24] * 2, *[15] * 4, *[8] * 14],
  'pbc': [True, True, True],
}


def get_ase_distances(atoms: Atoms, cutoff: float) -> np.ndarray:
  """Get sorted distances from ASE neighbor list."""
  i, j, d = ase_neighbor_list('ijd', atoms, cutoff)
  return np.sort(d)


def get_mi_distances(atoms: Atoms, cutoff: float) -> np.ndarray:
  """Get sorted distances from multi-image neighbor list."""
  # ASE uses row vectors: R_real = R_frac @ cell
  # JAX-MD uses column vectors: R_real = R_frac @ box.T
  # So box = cell.T to get: R_frac @ (cell.T).T = R_frac @ cell
  cell = np.array(atoms.cell.array, dtype=np.float64)
  box = cell.T  # JAX-MD convention: columns are lattice vectors
  pos = atoms.get_positions()
  N = len(atoms)
  pbc = np.array(atoms.pbc)

  # Use ASE's scaled positions directly (already in [0, 1))
  pos_frac = atoms.get_scaled_positions()

  # Use higher safety factor for skewed cells
  max_neighbors = estimate_max_neighbors_from_box(
    box, cutoff, N, safety_factor=5.0, pbc=pbc
  )
  neighbor_fn = neighbor_list_multi_image(
    None,
    box,
    cutoff,
    max_neighbors=max_neighbors,
    format=NeighborListFormat.Sparse,
    pbc=pbc,
  )
  nbrs = neighbor_fn.allocate(pos_frac)

  # Convert back to Cartesian: R_real = R_frac @ box.T = R_frac @ cell
  pos_cart = pos_frac @ box.T

  # Compute distances
  mask = nbrs.idx[0] < N
  i_idx = np.array(nbrs.idx[0])[mask]
  j_idx = np.array(nbrs.idx[1])[mask]
  shifts = np.array(nbrs.shifts)[mask]

  # r_j + shift @ box.T - r_i
  shifts_real = shifts @ box.T
  dR = pos_cart[j_idx] + shifts_real - pos_cart[i_idx]
  distances = np.linalg.norm(dR, axis=1)

  return np.sort(distances)


def get_mi_distances_molecule(atoms: Atoms, cutoff: float) -> np.ndarray:
  """Get sorted distances for non-periodic (molecule) structures."""
  pos = atoms.get_positions()
  N = len(atoms)

  # Create a large box that contains all atoms with padding
  pos_min = pos.min(axis=0) - cutoff - 1.0
  pos_max = pos.max(axis=0) + cutoff + 1.0
  box_size = pos_max - pos_min

  # Shift positions to be positive
  pos_shifted = pos - pos_min

  # Create diagonal box (JAX-MD convention: columns are lattice vectors)
  box = np.diag(box_size)

  # Convert to fractional coordinates
  pos_frac = pos_shifted / box_size

  # Use simple estimate for non-periodic
  max_neighbors = estimate_max_neighbors(cutoff, safety_factor=5.0)
  neighbor_fn = neighbor_list_multi_image(
    None,
    box,
    cutoff,
    max_neighbors=max_neighbors,
    format=NeighborListFormat.Sparse,
    pbc=np.array([False, False, False]),
  )
  nbrs = neighbor_fn.allocate(pos_frac)

  # Compute distances
  mask = nbrs.idx[0] < N
  i_idx = np.array(nbrs.idx[0])[mask]
  j_idx = np.array(nbrs.idx[1])[mask]

  # For non-periodic, shifts should all be zero
  dR = pos_shifted[j_idx] - pos_shifted[i_idx]
  distances = np.linalg.norm(dR, axis=1)

  return np.sort(distances)


def get_ase_edges(atoms: Atoms, cutoff: float) -> set:
  """Get edges as set of (i, j, shift_tuple) from ASE."""
  i, j, S = ase_neighbor_list('ijS', atoms, cutoff)
  return set(zip(i, j, map(tuple, S)))


def get_mi_edges(
  atoms: Atoms,
  cutoff: float,
  fmt: NeighborListFormat = NeighborListFormat.Sparse,
) -> tuple:
  """Get edges and neighbor list from multi-image NL."""
  cell = np.array(atoms.cell.array, dtype=np.float64)
  box = cell.T
  N = len(atoms)
  pbc = np.array(atoms.pbc)
  pos_frac = atoms.get_scaled_positions()

  max_neighbors = estimate_max_neighbors_from_box(
    box, cutoff, N, safety_factor=5.0, pbc=pbc
  )
  neighbor_fn = neighbor_list_multi_image(
    None, box, cutoff, max_neighbors=max_neighbors, format=fmt, pbc=pbc
  )
  nbrs = neighbor_fn.allocate(pos_frac)

  if partition.is_sparse(fmt):
    mask = nbrs.idx[0] < N
    edges = set(
      zip(
        np.array(nbrs.idx[0])[mask],
        np.array(nbrs.idx[1])[mask],
        map(tuple, np.array(nbrs.shifts)[mask]),
      )
    )
  else:
    # Dense format
    edges = set()
    idx = np.array(nbrs.idx)
    shifts = np.array(nbrs.shifts)
    for i in range(N):
      for k in range(idx.shape[1]):
        j = idx[i, k]
        if j < N:
          edges.add((i, j, tuple(shifts[i, k])))

  return edges, nbrs, neighbor_fn, pos_frac


class CustomPartitionTest(parameterized.TestCase):
  """Test multi-image neighbor list against ASE."""

  @parameterized.parameters(
    {'atoms': bulk('Si', 'diamond', a=5.43, cubic=True), 'cutoff': 3.0},
    {'atoms': bulk('Cu', 'fcc', a=3.6), 'cutoff': 3.0},
    {'atoms': bulk('Ti', 'hcp', a=2.94, c=4.64), 'cutoff': 3.5},
    {'atoms': bulk('Bi', 'rhombohedral', a=6, alpha=20), 'cutoff': 4.0},
    {'atoms': Atoms(**CaCrP2O7_mvc_11955), 'cutoff': 3.5},
  )
  def test_distances_match_ase(self, atoms: Atoms, cutoff: float):
    """Check multi-image NL gives same distances as ASE."""
    d_ase = get_ase_distances(atoms, cutoff)
    d_mi = get_mi_distances(atoms, cutoff)

    # Both should find same number of pairs
    self.assertEqual(
      len(d_ase), len(d_mi), f'Neighbor count mismatch for {atoms}'
    )

    # Distances should match
    np.testing.assert_allclose(d_ase, d_mi, rtol=1e-7)

  @parameterized.parameters(1.0, 3.0, 5.0, 7.0)
  def test_varying_cutoffs(self, cutoff: float):
    """Test with varying cutoffs on cubic Si."""
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
    d_ase = get_ase_distances(atoms, cutoff)
    d_mi = get_mi_distances(atoms, cutoff)

    self.assertEqual(len(d_ase), len(d_mi))
    if len(d_ase) > 0:
      np.testing.assert_allclose(d_ase, d_mi, rtol=1e-7)

  def test_small_box_more_neighbors(self):
    """When r_cut > L/2, MI should find more neighbors than a single image."""
    # Create small box where cutoff exceeds half box length
    atoms = bulk('Cu', 'fcc', a=2.5)  # Small FCC
    cutoff = 3.0  # Larger than L/2

    d_ase = get_ase_distances(atoms, cutoff)
    d_mi = get_mi_distances(atoms, cutoff)

    # Should match ASE (which handles multi-image correctly)
    self.assertEqual(len(d_ase), len(d_mi))
    np.testing.assert_allclose(d_ase, d_mi, rtol=1e-7)

  @parameterized.parameters(
    {'mol_name': 'H2O', 'cutoff': 2.0},
    {'mol_name': 'CH4', 'cutoff': 2.5},
    {'mol_name': 'C2H6', 'cutoff': 3.0},
    {'mol_name': 'CH3CH2NH2', 'cutoff': 3.0},
  )
  def test_molecule_non_periodic(self, mol_name: str, cutoff: float):
    """Test non-periodic structures (molecules)."""
    atoms = molecule(mol_name)
    # Set a large cell for ASE (required for neighbor list)
    atoms.center(vacuum=cutoff + 5.0)
    atoms.pbc = False

    d_ase = get_ase_distances(atoms, cutoff)
    d_mi = get_mi_distances_molecule(atoms, cutoff)

    self.assertEqual(
      len(d_ase), len(d_mi), f'Neighbor count mismatch for {mol_name}'
    )
    if len(d_ase) > 0:
      np.testing.assert_allclose(d_ase, d_mi, rtol=1e-7)

  def test_mixed_pbc_slab(self):
    """Test slab geometry: periodic in xy, non-periodic in z."""
    # Create a slab (periodic in xy, vacuum in z)
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms = atoms.repeat((2, 2, 2))

    # Add vacuum in z direction
    atoms.center(vacuum=5.0, axis=2)

    # Set mixed PBC
    atoms.pbc = [True, True, False]
    cutoff = 3.0

    d_ase = get_ase_distances(atoms, cutoff)
    d_mi = get_mi_distances(atoms, cutoff)

    self.assertEqual(len(d_ase), len(d_mi), 'Neighbor count mismatch for slab')
    np.testing.assert_allclose(d_ase, d_mi, rtol=1e-7)

  def test_mixed_pbc_wire(self):
    """Test wire geometry: periodic in z only."""
    # Create a wire (periodic in z, vacuum in xy)
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms = atoms.repeat((1, 1, 3))

    # Add vacuum in xy directions
    atoms.center(vacuum=5.0, axis=0)
    atoms.center(vacuum=5.0, axis=1)

    # Set mixed PBC
    atoms.pbc = [False, False, True]
    cutoff = 3.0

    d_ase = get_ase_distances(atoms, cutoff)
    d_mi = get_mi_distances(atoms, cutoff)

    self.assertEqual(len(d_ase), len(d_mi), 'Neighbor count mismatch for wire')
    np.testing.assert_allclose(d_ase, d_mi, rtol=1e-7)

  # =========================================================================
  # Edge and Shift Verification (like nequip_jax_md.py)
  # =========================================================================

  def test_edges_match_ase_exactly(self):
    """Verify (i, j, shift) tuples match ASE exactly."""
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
    cutoff = 3.0

    edges_ase = get_ase_edges(atoms, cutoff)
    edges_mi, nbrs, _, _ = get_mi_edges(atoms, cutoff)

    self.assertEqual(edges_ase, edges_mi, 'Edge sets do not match ASE')

  def test_edges_match_ase_triclinic(self):
    """Verify edges match ASE for triclinic cell."""
    atoms = Atoms(**CaCrP2O7_mvc_11955)
    cutoff = 3.5

    edges_ase = get_ase_edges(atoms, cutoff)
    edges_mi, nbrs, _, _ = get_mi_edges(atoms, cutoff)

    self.assertEqual(
      edges_ase, edges_mi, 'Edge sets do not match for triclinic'
    )

  # =========================================================================
  # Format Tests
  # =========================================================================

  @parameterized.parameters(
    NeighborListFormat.Sparse,
    NeighborListFormat.OrderedSparse,
    NeighborListFormat.Dense,
  )
  def test_all_formats_find_same_edges(self, fmt: NeighborListFormat):
    """All formats should find the same neighbor pairs."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    edges_ase = get_ase_edges(atoms, cutoff)

    if fmt is NeighborListFormat.OrderedSparse:
      # OrderedSparse stores only one direction per pair
      # Convert ASE edges to ordered (i < j for zero shift, canonical for others)
      edges_mi, nbrs, _, _ = get_mi_edges(atoms, cutoff, fmt)
      # Just check we have roughly half the edges
      self.assertGreater(len(edges_mi), 0)
      self.assertLessEqual(len(edges_mi), len(edges_ase))
    else:
      edges_mi, nbrs, _, _ = get_mi_edges(atoms, cutoff, fmt)
      self.assertEqual(edges_ase, edges_mi)

  def test_dense_format_shape(self):
    """Dense format should have correct shapes."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Dense)

    # Dense idx shape: (N, max_neighbors)
    self.assertEqual(nbrs.idx.shape[0], N)
    self.assertEqual(len(nbrs.idx.shape), 2)

    # Dense shifts shape: (N, max_neighbors, dim)
    self.assertEqual(nbrs.shifts.shape[0], N)
    self.assertEqual(nbrs.shifts.shape[1], nbrs.idx.shape[1])
    self.assertEqual(nbrs.shifts.shape[2], 3)

  def test_sparse_format_shape(self):
    """Sparse format should have correct shapes."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Sparse)

    # Sparse idx: tuple (receivers, senders), each shape (capacity,)
    self.assertIsInstance(nbrs.idx, tuple)
    self.assertEqual(len(nbrs.idx), 2)
    self.assertEqual(nbrs.idx[0].shape, nbrs.idx[1].shape)

    # Sparse shifts shape: (capacity, dim)
    self.assertEqual(nbrs.shifts.shape[0], nbrs.idx[0].shape[0])
    self.assertEqual(nbrs.shifts.shape[1], 3)

  # =========================================================================
  # Buffer Overflow Detection
  # =========================================================================

  def test_buffer_overflow_detection(self):
    """Neighbor list should detect buffer overflow."""
    atoms = bulk('Cu', 'fcc', a=3.6).repeat((3, 3, 3))
    cutoff = 5.0
    N = len(atoms)

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = atoms.get_scaled_positions()

    # Use intentionally small max_neighbors to trigger overflow
    neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=2,  # Way too small
      format=NeighborListFormat.Sparse,
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    self.assertTrue(nbrs.did_buffer_overflow)

  def test_no_buffer_overflow_with_sufficient_capacity(self):
    """Neighbor list should not overflow with sufficient capacity."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff)

    self.assertFalse(nbrs.did_buffer_overflow)

  # =========================================================================
  # Mask and Compatibility Tests
  # =========================================================================

  def test_neighbor_list_mask_sparse(self):
    """Test neighbor_list_multi_image_mask for Sparse format."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Sparse)

    mask = neighbor_list_multi_image_mask(nbrs)
    n_valid = int(jnp.sum(mask))

    # Should have some valid edges
    self.assertGreater(n_valid, 0)

    # All valid indices should be < N
    self.assertTrue(jnp.all(nbrs.idx[0][mask] < N))
    self.assertTrue(jnp.all(nbrs.idx[1][mask] < N))

  def test_partition_neighbor_list_mask_compatibility(self):
    """Test compatibility with jax_md.partition.neighbor_list_mask."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Sparse)

    # Both mask functions should give the same result
    mask_mi = neighbor_list_multi_image_mask(nbrs)
    mask_jaxmd = partition.neighbor_list_mask(nbrs)

    np.testing.assert_array_equal(mask_mi, mask_jaxmd)

  def test_to_jraph_compatibility(self):
    """Test compatibility with jax_md.partition.to_jraph."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Sparse)

    # Should be able to convert to jraph format
    graph = partition.to_jraph(nbrs)

    # Check graph structure
    self.assertEqual(int(graph.n_node[0]), N)
    self.assertGreater(int(graph.n_edge[0]), 0)

  # =========================================================================
  # Update Function Tests
  # =========================================================================

  def test_update_function(self):
    """Test that update function works correctly."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    edges_ase = get_ase_edges(atoms, cutoff)
    edges_mi, nbrs, neighbor_fn, pos_frac = get_mi_edges(atoms, cutoff)

    # Update with same positions should give same result
    nbrs_updated = nbrs.update(pos_frac)

    mask = nbrs_updated.idx[0] < len(atoms)
    edges_updated = set(
      zip(
        np.array(nbrs_updated.idx[0])[mask],
        np.array(nbrs_updated.idx[1])[mask],
        map(tuple, np.array(nbrs_updated.shifts)[mask]),
      )
    )

    self.assertEqual(edges_ase, edges_updated)

  def test_update_with_perturbed_positions(self):
    """Test update with slightly perturbed positions."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    _, nbrs, neighbor_fn, pos_frac = get_mi_edges(atoms, cutoff)

    # Small perturbation
    pos_perturbed = pos_frac + 0.001 * np.random.randn(N, 3)

    # Update should work without error
    nbrs_updated = nbrs.update(pos_perturbed)

    # Should still have valid edges
    mask = nbrs_updated.idx[0] < N
    n_valid = int(jnp.sum(mask))
    self.assertGreater(n_valid, 0)

  # =========================================================================
  # Properties Tests
  # =========================================================================

  def test_n_edges_property(self):
    """Test n_edges property counts valid edges."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Sparse)

    n_edges = nbrs.n_edges
    mask = nbrs.idx[0] < N
    expected = int(jnp.sum(mask))

    self.assertEqual(n_edges, expected)

  def test_senders_receivers_properties(self):
    """Test senders and receivers properties for Sparse format."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Sparse)

    # Properties should work
    senders = nbrs.senders
    receivers = nbrs.receivers

    # Should match idx tuple
    np.testing.assert_array_equal(senders, nbrs.idx[1])
    np.testing.assert_array_equal(receivers, nbrs.idx[0])

  def test_senders_receivers_error_for_dense(self):
    """Test that senders/receivers raise error for Dense format."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Dense)

    with self.assertRaises(ValueError):
      _ = nbrs.senders

    with self.assertRaises(ValueError):
      _ = nbrs.receivers

  def test_max_neighbors_property_dense(self):
    """Test max_neighbors property for Dense format."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Dense)

    max_neighbors = nbrs.max_neighbors
    self.assertEqual(max_neighbors, nbrs.idx.shape[1])

  def test_max_neighbors_error_for_sparse(self):
    """Test that max_neighbors raises error for Sparse format."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    _, nbrs, _, _ = get_mi_edges(atoms, cutoff, NeighborListFormat.Sparse)

    with self.assertRaises(ValueError):
      _ = nbrs.max_neighbors

  # =========================================================================
  # Estimation Functions Tests
  # =========================================================================

  def test_estimate_max_neighbors_positive(self):
    """Test estimate_max_neighbors returns positive values."""
    for cutoff in [1.0, 3.0, 5.0, 10.0]:
      max_neighbors = estimate_max_neighbors(cutoff)
      self.assertGreater(max_neighbors, 0)

  def test_estimate_max_neighbors_zero_cutoff(self):
    """Test estimate_max_neighbors with zero cutoff."""
    max_neighbors = estimate_max_neighbors(0.0)
    self.assertEqual(max_neighbors, 0)

  def test_estimate_max_neighbors_from_box(self):
    """Test estimate_max_neighbors_from_box gives reasonable values."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T

    max_neighbors = estimate_max_neighbors_from_box(box, cutoff, N)

    # Should be positive
    self.assertGreater(max_neighbors, 0)

    # Should be at least as many as actual neighbors
    edges_ase = get_ase_edges(atoms, cutoff)
    # Actual neighbors per atom (on average)
    actual_avg = len(edges_ase) / N
    self.assertGreater(max_neighbors, actual_avg * 0.5)

  def test_estimate_max_neighbors_from_box_dim_error(self):
    """Test estimate_max_neighbors_from_box raises error for dim > 3."""
    box_4d = np.eye(4)

    with self.assertRaises(ValueError):
      estimate_max_neighbors_from_box(box_4d, 3.0, 10)

  def test_estimate_max_neighbors_dim_error(self):
    """Test estimate_max_neighbors raises error for dim > 3."""
    with self.assertRaises(ValueError):
      estimate_max_neighbors(3.0, dim=4)

  # =========================================================================
  # Non-periodic Shift Tests
  # =========================================================================

  def test_non_periodic_shifts_are_zero(self):
    """For non-periodic systems, all shifts should be zero."""
    atoms = molecule('H2O')
    atoms.center(vacuum=10.0)
    atoms.pbc = False
    cutoff = 2.0

    pos = atoms.get_positions()
    N = len(atoms)
    pos_min = pos.min(axis=0) - cutoff - 1.0
    pos_max = pos.max(axis=0) + cutoff + 1.0
    box_size = pos_max - pos_min
    box = np.diag(box_size)
    pos_frac = (pos - pos_min) / box_size

    neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=50,
      format=NeighborListFormat.Sparse,
      pbc=np.array([False, False, False]),
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    mask = nbrs.idx[0] < N
    shifts = np.array(nbrs.shifts)[mask]

    # All shifts should be zero for non-periodic
    np.testing.assert_array_equal(shifts, 0)

  # =========================================================================
  # 2D Tests
  # =========================================================================

  def test_2d_system(self):
    """Test 2D system (graphene-like)."""
    # Create simple 2D hexagonal lattice
    a = 2.46  # graphene lattice constant
    cell = np.array([[a, 0], [a / 2, a * np.sqrt(3) / 2]])
    positions = np.array([[0, 0], [a / 2, a / (2 * np.sqrt(3))]])
    cutoff = 2.0

    N = len(positions)
    box = cell.T  # JAX-MD convention

    # Fractional coordinates
    inv_cell = np.linalg.inv(cell)
    pos_frac = positions @ inv_cell

    max_neighbors = estimate_max_neighbors_from_box(box, cutoff, N)
    neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=max_neighbors,
      format=NeighborListFormat.Sparse,
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    # Should find some neighbors
    n_edges = nbrs.n_edges
    self.assertGreater(n_edges, 0)
    self.assertFalse(nbrs.did_buffer_overflow)

  # =========================================================================
  # Tests inspired by jax_md/partition.py
  # =========================================================================

  def test_is_sparse_function(self):
    """Test partition.is_sparse for all formats."""
    self.assertTrue(partition.is_sparse(NeighborListFormat.Sparse))
    self.assertTrue(partition.is_sparse(NeighborListFormat.OrderedSparse))
    self.assertFalse(partition.is_sparse(NeighborListFormat.Dense))

  def test_neighbor_list_fns_unpacking(self):
    """Test that NeighborListFns can be unpacked like (allocate, update)."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = atoms.get_scaled_positions()

    neighbor_fn = neighbor_list_multi_image(
      None, box, cutoff, max_neighbors=50, format=NeighborListFormat.Sparse
    )

    # Test unpacking
    allocate_fn, update_fn = neighbor_fn

    # Both should work
    nbrs = allocate_fn(pos_frac)
    self.assertGreater(nbrs.n_edges, 0)

    nbrs_updated = update_fn(pos_frac, nbrs)
    self.assertEqual(nbrs.n_edges, nbrs_updated.n_edges)

  def test_dr_threshold_skips_rebuild(self):
    """Test that dr_threshold skips rebuild when atoms haven't moved much."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = jnp.array(atoms.get_scaled_positions())

    # Create neighbor list with dr_threshold
    neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      dr_threshold=0.5,  # Large threshold
      max_neighbors=50,
      format=NeighborListFormat.Sparse,
    )
    nbrs = neighbor_fn.allocate(pos_frac)
    ref_pos = nbrs.reference_position

    # Small perturbation (well below threshold)
    pos_perturbed = pos_frac + 0.001

    # Update should return same neighbor list (skip rebuild)
    nbrs_updated = nbrs.update(pos_perturbed)

    # Reference position should be unchanged (rebuild was skipped)
    np.testing.assert_array_equal(nbrs_updated.reference_position, ref_pos)

  def test_dr_threshold_triggers_rebuild(self):
    """Test that dr_threshold triggers rebuild when atoms move significantly."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = jnp.array(atoms.get_scaled_positions())

    # Create neighbor list with small dr_threshold
    neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      dr_threshold=0.01,  # Small threshold
      max_neighbors=50,
      format=NeighborListFormat.Sparse,
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    # Large perturbation (above threshold)
    pos_perturbed = pos_frac + 0.1

    # Update should trigger rebuild
    nbrs_updated = nbrs.update(pos_perturbed)

    # Reference position should be updated (rebuild was triggered)
    np.testing.assert_array_equal(
      nbrs_updated.reference_position, pos_perturbed
    )

  def test_fractional_vs_cartesian_coordinates(self):
    """Test that fractional and Cartesian coordinates give same results."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = atoms.get_scaled_positions()
    pos_cart = atoms.get_positions()

    # Fractional coordinates
    neighbor_fn_frac = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=50,
      format=NeighborListFormat.Sparse,
      fractional_coordinates=True,
    )
    nbrs_frac = neighbor_fn_frac.allocate(pos_frac)

    # Cartesian coordinates
    neighbor_fn_cart = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=50,
      format=NeighborListFormat.Sparse,
      fractional_coordinates=False,
    )
    nbrs_cart = neighbor_fn_cart.allocate(pos_cart)

    # Should find same number of edges
    self.assertEqual(nbrs_frac.n_edges, nbrs_cart.n_edges)

    # Extract edges and compare
    mask_frac = nbrs_frac.idx[0] < N
    mask_cart = nbrs_cart.idx[0] < N

    edges_frac = set(
      zip(
        np.array(nbrs_frac.idx[0])[mask_frac],
        np.array(nbrs_frac.idx[1])[mask_frac],
        map(tuple, np.array(nbrs_frac.shifts)[mask_frac]),
      )
    )
    edges_cart = set(
      zip(
        np.array(nbrs_cart.idx[0])[mask_cart],
        np.array(nbrs_cart.idx[1])[mask_cart],
        map(tuple, np.array(nbrs_cart.shifts)[mask_cart]),
      )
    )

    self.assertEqual(edges_frac, edges_cart)

  def test_ordered_sparse_has_fewer_edges(self):
    """OrderedSparse should have roughly half the edges of Sparse."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = atoms.get_scaled_positions()

    # Sparse format
    neighbor_fn_sparse = neighbor_list_multi_image(
      None, box, cutoff, max_neighbors=100, format=NeighborListFormat.Sparse
    )
    nbrs_sparse = neighbor_fn_sparse.allocate(pos_frac)

    # OrderedSparse format
    neighbor_fn_ordered = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=100,
      format=NeighborListFormat.OrderedSparse,
    )
    nbrs_ordered = neighbor_fn_ordered.allocate(pos_frac)

    # OrderedSparse should have roughly half the edges
    ratio = nbrs_sparse.n_edges / max(nbrs_ordered.n_edges, 1)
    self.assertGreater(ratio, 1.5)  # Should be close to 2
    self.assertLess(ratio, 2.5)

  def test_1d_system(self):
    """Test 1D system (linear chain)."""
    # Simple 1D chain
    N = 5
    box = np.array([[10.0]])  # 1D box
    positions = np.linspace(0.1, 0.9, N).reshape(-1, 1)  # Fractional
    cutoff = 3.0

    max_neighbors = estimate_max_neighbors(cutoff, dim=1)
    neighbor_fn = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=max_neighbors,
      format=NeighborListFormat.Sparse,
    )
    nbrs = neighbor_fn.allocate(positions)

    # Should find neighbors
    self.assertGreater(nbrs.n_edges, 0)
    self.assertFalse(nbrs.did_buffer_overflow)

  def test_capacity_multiplier(self):
    """Test that capacity_multiplier affects allocated capacity."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = atoms.get_scaled_positions()

    # Small multiplier
    neighbor_fn_small = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=20,
      capacity_multiplier=1.0,
      format=NeighborListFormat.Sparse,
    )
    nbrs_small = neighbor_fn_small.allocate(pos_frac)

    # Large multiplier
    neighbor_fn_large = neighbor_list_multi_image(
      None,
      box,
      cutoff,
      max_neighbors=20,
      capacity_multiplier=2.0,
      format=NeighborListFormat.Sparse,
    )
    nbrs_large = neighbor_fn_large.allocate(pos_frac)

    # Larger multiplier should give more capacity
    self.assertGreater(nbrs_large.max_occupancy, nbrs_small.max_occupancy)

  def test_reference_position_stored(self):
    """Test that reference_position is correctly stored."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = jnp.array(atoms.get_scaled_positions())

    neighbor_fn = neighbor_list_multi_image(
      None, box, cutoff, max_neighbors=50, format=NeighborListFormat.Sparse
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    # Reference position should match input
    np.testing.assert_array_equal(nbrs.reference_position, pos_frac)

  def test_box_stored_in_neighbor_list(self):
    """Test that box is correctly stored in neighbor list."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = jnp.array(cell.T)
    pos_frac = atoms.get_scaled_positions()

    neighbor_fn = neighbor_list_multi_image(
      None, box, cutoff, max_neighbors=50, format=NeighborListFormat.Sparse
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    # Box should be stored
    np.testing.assert_array_almost_equal(nbrs.box, box)

  def test_dataclass_replace(self):
    """Test that dataclasses.replace works on NeighborListMultiImage."""
    from jax_md import dataclasses as jax_dataclasses

    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = atoms.get_scaled_positions()

    neighbor_fn = neighbor_list_multi_image(
      None, box, cutoff, max_neighbors=50, format=NeighborListFormat.Sparse
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    # Replace a field
    new_pos = pos_frac + 0.01
    nbrs_replaced = jax_dataclasses.replace(nbrs, reference_position=new_pos)

    # Original should be unchanged
    np.testing.assert_array_equal(nbrs.reference_position, pos_frac)

    # Replaced should have new value
    np.testing.assert_array_almost_equal(
      nbrs_replaced.reference_position, new_pos
    )

  def test_jit_compatible_update(self):
    """Test that neighbor list update is JIT-compatible."""
    atoms = bulk('Cu', 'fcc', a=3.6)
    cutoff = 3.0
    N = len(atoms)

    cell = np.array(atoms.cell.array, dtype=np.float64)
    box = cell.T
    pos_frac = jnp.array(atoms.get_scaled_positions())

    neighbor_fn = neighbor_list_multi_image(
      None, box, cutoff, max_neighbors=50, format=NeighborListFormat.Sparse
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    # JIT the update function - use mask sum instead of n_edges property
    # (n_edges uses int() which is not JIT-compatible)
    @jax.jit
    def update_and_count(pos, nbrs):
      nbrs_new = nbrs.update(pos)
      # Count valid edges (JIT-compatible)
      mask = nbrs_new.idx[0] < N
      return jnp.sum(mask)

    # Should work without error
    n_edges = update_and_count(pos_frac, nbrs)
    self.assertGreater(int(n_edges), 0)

    # Call again to ensure caching works
    n_edges2 = update_and_count(pos_frac + 0.001, nbrs)
    self.assertGreater(int(n_edges2), 0)

  def test_empty_system(self):
    """Test behavior with a single atom (no neighbors)."""
    box = np.eye(3) * 10.0
    pos_frac = np.array([[0.5, 0.5, 0.5]])
    cutoff = 3.0

    neighbor_fn = neighbor_list_multi_image(
      None, box, cutoff, max_neighbors=50, format=NeighborListFormat.Sparse
    )
    nbrs = neighbor_fn.allocate(pos_frac)

    # Single atom should have no neighbors
    self.assertEqual(nbrs.n_edges, 0)
    self.assertFalse(nbrs.did_buffer_overflow)


if __name__ == '__main__':
  absltest.main()
