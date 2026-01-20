import jax
import jax.numpy as jnp
from jax_md import dataclasses
from jax_md.partition import NeighborListFormat
from typing import Callable, Tuple, Union, Optional

# Type aliases
Array = jnp.ndarray
f32 = jnp.float32
i32 = jnp.int32


@dataclasses.dataclass
class NeighborListMultiImage:
  """A struct containing the state of a multi-image neighbor list.

  This data structure is compatible with `jax_md.partition.to_jraph` and
  `jax_md.partition.neighbor_list_mask`. It stores edges between atoms,
  including all periodic images within the cutoff (not just the nearest).

  Supports two storage formats:

  **Sparse/OrderedSparse**:
    - idx: Tuple (receivers, senders), each shape (capacity,)
    - shifts: Shape (capacity, dim)

  **Dense**:
    - idx: Shape (N, max_neighbors), neighbor indices for each atom
    - shifts: Shape (N, max_neighbors, dim), shift for each neighbor

  Attributes:
    idx: For Sparse: tuple (receivers, senders). For Dense: array (N, max_neighbors).
      Invalid entries are padded with index N (number of atoms).
    shifts: Integer shift vectors. Sparse: (capacity, dim). Dense: (N, max_neighbors, dim).
      The real-space shift is ``shifts @ box.T``.
    reference_position: Positions when the list was built, shape (N, dim).
    box: An affine transformation; see ``jax_md.space.periodic_general``.
    format: NeighborListFormat.Sparse, OrderedSparse, or Dense.
    max_occupancy: For Sparse: total capacity. For Dense: max_neighbors.
    update_fn: Function to update the neighbor list.
    did_buffer_overflow: True if more edges/neighbors were found than capacity allows.
  """

  # Sparse/OrderedSparse: Tuple[Array[capacity], Array[capacity]] = (receivers, senders)
  # Dense: Array[N, max_neighbors]
  idx: Union[Tuple[Array, Array], Array]
  # Sparse/OrderedSparse: Array[capacity, dim]
  # Dense: Array[N, max_neighbors, dim]
  shifts: Array  # real-space shift = shifts @ box.T
  reference_position: Array  # [N, dim]
  box: Array  # [dim, dim]
  format: NeighborListFormat = dataclasses.static_field()
  max_occupancy: int = dataclasses.static_field()
  update_fn: Callable[..., 'NeighborListMultiImage'] = (
    dataclasses.static_field()
  )
  did_buffer_overflow: bool = False

  def update(self, position: Array, **kwargs) -> 'NeighborListMultiImage':
    """Update neighbor list with new positions."""
    return self.update_fn(position, self, **kwargs)

  @property
  def senders(self) -> Array:
    """Sender atom indices (Sparse format only)."""
    if self.format is NeighborListFormat.Dense:
      raise ValueError(
        'senders property not available for Dense format. Use idx directly.'
      )
    return self.idx[1]

  @property
  def receivers(self) -> Array:
    """Receiver atom indices (Sparse format only)."""
    if self.format is NeighborListFormat.Dense:
      raise ValueError(
        'receivers property not available for Dense format. Use idx directly.'
      )
    return self.idx[0]

  @property
  def n_edges(self) -> int:
    """Number of valid edges (excluding padding)."""
    N = len(self.reference_position)
    if self.format is NeighborListFormat.Dense:
      # Count valid entries in Dense format
      return int(jnp.sum(self.idx < N))
    return int(jnp.sum(self.idx[0] < N))

  @property
  def max_neighbors(self) -> int:
    """Maximum neighbors per atom (Dense format only)."""
    if self.format is not NeighborListFormat.Dense:
      raise ValueError(
        'max_neighbors property only available for Dense format.'
      )
    return self.idx.shape[1]

  @property
  def n_node(self) -> int:
    """Number of atoms (for to_jraph compatibility)."""
    return len(self.reference_position)


# Type alias for neighbor list functions
# AllocateFn: (position: Array[N, dim], **kwargs) -> NeighborListMultiImage
# UpdateFn: (position: Array[N, dim], neighbors: NeighborListMultiImage, **kwargs) -> NeighborListMultiImage
AllocateFn = Callable[..., NeighborListMultiImage]
UpdateFn = Callable[[Array, NeighborListMultiImage], NeighborListMultiImage]


@dataclasses.dataclass
class NeighborListMultiImageFns:
  """A struct containing functions to allocate and update neighbor lists.

  This mirrors the `jax_md.partition.NeighborListFns` interface.

  Attributes:
    allocate: A function to allocate a new neighbor list. This function cannot
      be compiled, since it uses the values of positions to infer the shapes.
      Signature: `(position: Array[N, dim], **kwargs) -> NeighborListMultiImage`
    update: A function to update a neighbor list given a new set of positions
      and a previously allocated neighbor list.
      Signature: `(position: Array[N, dim], neighbors: NeighborListMultiImage, **kwargs) -> NeighborListMultiImage`
  """

  allocate: AllocateFn = dataclasses.static_field()
  update: UpdateFn = dataclasses.static_field()

  def __iter__(self):
    """Allow unpacking: allocate_fn, update_fn = neighbor_fn."""
    return iter((self.allocate, self.update))


def _compute_shift_ranges(
  box: Array,  # [dim, dim]
  r_cutoff: float,
  pbc: Array,  # [dim]
) -> Array:  # [num_shifts, dim]
  r"""Compute integer shift vectors for multi-image neighbor search.

  For each lattice direction, determines how many periodic images are needed
  to capture all neighbors within :math:`r_\text{cut}`. Uses the reciprocal
  lattice to compute perpendicular box heights:

  .. math::

    h_i = \frac{1}{\|\mathbf{b}_i\|}

  where :math:`\mathbf{b}_i` is the :math:`i`-th column of the inverse box
  transpose (i.e., the :math:`i`-th reciprocal lattice vector). The number of
  shifts along direction :math:`i` is :math:`n_i = \lceil r_\text{cut} / h_i \rceil`.

  The total number of shift vectors is :math:`\prod_i (2 n_i + 1)` for periodic
  directions. The real-space shift for a given integer shift vector
  :math:`\mathbf{s}` is :math:`\mathbf{s} \cdot \mathbf{T}` where
  :math:`\mathbf{T}` is the box matrix.

  This is the same algorithm used by ASE and matscipy for neighbor list
  construction with periodic boundary conditions.

  Args:
    box: Affine transformation (see ``periodic_general``). Shape ``[dim, dim]``.
    r_cutoff: Interaction cutoff distance (scalar).
    pbc: Boolean array indicating which directions are periodic.
      Shape ``[dim]``. Non-periodic directions get zero shifts.

  Returns:
    Integer shift vectors spanning the required range. Shape
    ``[num_shifts, dim]``. Each row is a shift vector :math:`(n_1, n_2, \ldots, n_d)`.
  """
  # Reciprocal lattice vectors (columns of inv_box.T)
  inv_box_T = jnp.linalg.inv(box).T  # [dim, dim]
  # Perpendicular heights of the box
  heights = 1.0 / jnp.linalg.norm(inv_box_T, axis=0)  # [dim]
  # Number of shifts needed per direction.
  n_max = jnp.ceil(r_cutoff / heights).astype(i32)  # [dim]
  n_max = jnp.where(pbc, n_max, 0)  # [dim], zero for non-periodic

  # Build Cartesian product of shift ranges
  dim = box.shape[0]
  n_max_int = [int(n_max[i]) for i in range(dim)]
  ranges = [jnp.arange(-n, n + 1) for n in n_max_int]  # List of [2*n+1]
  grids = jnp.meshgrid(*ranges, indexing='ij')
  return jnp.stack([g.ravel() for g in grids], axis=-1)  # [num_shifts, dim]


def _compute_distances_sq(
  position: Array,  # [N, dim]
  box: Array,  # [dim, dim]
  shifts_real: Array,  # [num_shifts, dim]
  fractional_coordinates: bool,
) -> Array:  # [num_shifts, N, N]
  r"""Compute squared distances for all (shift, i, j) combinations.

  For each shift :math:`\mathbf{s}` and atom pair :math:`(i, j)`:

  .. math::

    d_{ij}^{\mathbf{s}2} = \|\mathbf{r}_j + \mathbf{s} \cdot \mathbf{T} - \mathbf{r}_i\|^2

  Args:
    position: Atom positions. Shape ``[N, dim]``.
    box: Box matrix. Shape ``[dim, dim]``.
    shifts_real: Real-space shifts. Shape ``[num_shifts, dim]``.
    fractional_coordinates: If True, positions are fractional.

  Returns:
    Squared distances. Shape ``[num_shifts, N, N]``.
  """
  if fractional_coordinates:
    position_real = position @ box.T  # [N, dim]
  else:
    position_real = position

  D_all = (
    position_real[None, None, :, :]  # [1, 1, N, dim]
    + shifts_real[:, None, None, :]  # [num_shifts, 1, 1, dim]
    - position_real[None, :, None, :]  # [1, N, 1, dim]
  )  # [num_shifts, N, N, dim]

  return jnp.sum(D_all**2, axis=-1)  # [num_shifts, N, N]


def _compute_pairwise_mask(
  position: Array,  # [N, dim]
  box: Array,  # [dim, dim]
  shifts_real: Array,  # [num_shifts, dim]
  zero_shift_idx: int,
  r_cutoff: float,
  fractional_coordinates: bool,
) -> Array:  # [num_shifts, N, N]
  r"""Compute boolean mask for pairs within cutoff across all shifts.

  For each shift vector :math:`\mathbf{s}` and atom pair :math:`(i, j)`,
  computes whether the distance is within the cutoff:

  .. math::

    \|\mathbf{r}_j + \mathbf{s} \cdot \mathbf{T} - \mathbf{r}_i\| < r_\text{cut}

  Self-interactions (:math:`i = j` with zero shift) are excluded.

  Uses ``vmap`` over shifts for better memory locality and fused computation.

  Args:
    position: Atom positions. Shape ``[N, dim]``.
    box: Affine transformation (see ``periodic_general``). Shape ``[dim, dim]``.
    shifts_real: Real-space shift vectors. Shape ``[num_shifts, dim]``.
    zero_shift_idx: Index of the zero shift in ``shifts_real``.
    r_cutoff: Cutoff distance (scalar).
    fractional_coordinates: If True, positions are fractional.

  Returns:
    Boolean mask. Shape ``[num_shifts, N, N]``. Entry ``[s, i, j]`` is True
    if atom ``j`` with shift ``s`` is a neighbor of atom ``i``.
  """
  N = position.shape[0]
  num_shifts = shifts_real.shape[0]
  cutoff_sq = r_cutoff**2

  # Convert to real coordinates if needed
  if fractional_coordinates:
    position_real = position @ box.T  # [N, dim]
  else:
    position_real = position

  # Fused distance and mask computation per shift using vmap
  def mask_for_shift(shift):  # shift: [dim]
    # D[i,j] = r_j + shift - r_i
    D = (
      position_real[None, :, :] + shift - position_real[:, None, :]
    )  # [N, N, dim]
    dist_sq = jnp.sum(D**2, axis=-1)  # [N, N]
    return dist_sq < cutoff_sq

  within_cutoff = jax.vmap(mask_for_shift)(shifts_real)  # [num_shifts, N, N]

  # Exclude self-interactions (i == j with zero shift)
  self_mask = jnp.eye(N, dtype=bool)  # [N, N]
  zero_shift_mask = jnp.arange(num_shifts) == zero_shift_idx  # [num_shifts]
  self_interaction = zero_shift_mask[:, None, None] & self_mask[None, :, :]
  within_cutoff = within_cutoff & ~self_interaction  # [num_shifts, N, N]

  return within_cutoff


def _scatter_to_sparse(
  valid_mask: Array,  # [num_shifts, N, N]
  shifts: Array,  # [num_shifts, dim]
  capacity: int,
  N: int,
) -> Tuple[Array, Array, Array, Array]:
  r"""Scatter valid pairs into fixed-size sparse arrays.

  Uses cumulative sum for compaction of valid entries.

  Args:
    valid_mask: Boolean mask indicating valid pairs. Shape ``[num_shifts, N, N]``.
    shifts: Integer shift vectors. Shape ``[num_shifts, dim]``.
    capacity: Maximum number of edges to store.
    N: Number of atoms.

  Returns:
    Tuple ``(senders, receivers, edge_shifts, n_valid)``:

    - ``senders``: Shape ``[capacity]``. Padded with ``N``.
    - ``receivers``: Shape ``[capacity]``. Padded with ``N``.
    - ``edge_shifts``: Shape ``[capacity, dim]``.
    - ``n_valid``: Total number of valid pairs (scalar).
  """
  num_shifts = shifts.shape[0]
  dim = shifts.shape[1]

  valid_flat = valid_mask.ravel()  # [num_shifts * N * N]

  # Create index grids
  s_grid, i_grid, j_grid = jnp.meshgrid(
    jnp.arange(num_shifts), jnp.arange(N), jnp.arange(N), indexing='ij'
  )  # each [num_shifts, N, N]
  s_flat, i_flat, j_flat = s_grid.ravel(), i_grid.ravel(), j_grid.ravel()

  # Compact using cumsum
  n_valid = jnp.sum(valid_flat)
  cumsum = jnp.cumsum(valid_flat) - 1  # [num_shifts * N * N]

  # Pre-allocate with padding index = N
  senders = N * jnp.ones(capacity, dtype=i32)  # [capacity]
  receivers = N * jnp.ones(capacity, dtype=i32)  # [capacity]
  edge_shifts = jnp.zeros(
    (capacity, dim), dtype=shifts.dtype
  )  # [capacity, dim]

  # Scatter valid entries
  write_idx = jnp.where(valid_flat, cumsum, capacity)
  write_mask = valid_flat & (cumsum < capacity)

  senders = senders.at[write_idx].set(
    jnp.where(write_mask, j_flat, N), mode='drop'
  )
  receivers = receivers.at[write_idx].set(
    jnp.where(write_mask, i_flat, N), mode='drop'
  )

  shift_vals = shifts[s_flat]  # [num_shifts * N * N, dim]
  shift_vals_masked = jnp.where(write_mask[:, None], shift_vals, 0)
  edge_shifts = edge_shifts.at[write_idx].set(shift_vals_masked, mode='drop')

  return senders, receivers, edge_shifts, n_valid


def _build_neighbor_list_sparse(
  position: Array,  # [N, dim]
  box: Array,  # [dim, dim]
  shifts: Array,  # [num_shifts, dim]
  shifts_real: Array,  # [num_shifts, dim]
  zero_shift_idx: int,
  r_cutoff: float,
  capacity: int,
  fractional_coordinates: bool,
  update_fn: Callable,
) -> NeighborListMultiImage:
  r"""Build neighbor list in Sparse format.

  Stores **both directions** for each pair: if :math:`(i, j)` is a neighbor,
  both :math:`i \to j` and :math:`j \to i` are stored. Required for GNNs
  and asymmetric potentials.

  Args:
    position: Atom positions. Shape ``[N, dim]``.
    box: Box matrix. Shape ``[dim, dim]``.
    shifts: Integer shift vectors. Shape ``[num_shifts, dim]``.
    shifts_real: Real-space shifts (``shifts @ box.T``). Shape ``[num_shifts, dim]``.
    zero_shift_idx: Index of the zero shift vector.
    r_cutoff: Cutoff distance.
    capacity: Maximum edges to store.
    fractional_coordinates: If True, positions are fractional.
    update_fn: Function to update the neighbor list.

  Returns:
    NeighborListMultiImage with ``format=Sparse``:

    - ``idx``: ``(receivers, senders)`` each shape ``[capacity]``.
    - ``shifts``: Shape ``[capacity, dim]``.
  """
  within_cutoff = _compute_pairwise_mask(
    position, box, shifts_real, zero_shift_idx, r_cutoff, fractional_coordinates
  )  # [num_shifts, N, N]

  N = position.shape[0]
  senders, receivers, edge_shifts, n_valid = _scatter_to_sparse(
    within_cutoff, shifts, capacity, N
  )

  return NeighborListMultiImage(
    idx=(receivers, senders),
    shifts=edge_shifts,
    reference_position=position,
    box=box,
    format=NeighborListFormat.Sparse,
    did_buffer_overflow=(n_valid > capacity),
    max_occupancy=capacity,
    update_fn=update_fn,
  )


def _build_neighbor_list_orderedsparse(
  position: Array,  # [N, dim]
  box: Array,  # [dim, dim]
  shifts: Array,  # [num_shifts, dim]
  shifts_real: Array,  # [num_shifts, dim]
  zero_shift_idx: int,
  r_cutoff: float,
  capacity: int,
  fractional_coordinates: bool,
  update_fn: Callable,
) -> NeighborListMultiImage:
  r"""Build neighbor list in OrderedSparse format.

  Stores **one direction** per pair to avoid double-counting. Uses 2x less
  memory than Sparse format.

  **Ordering rules:**

  - **Zero shift** (:math:`\mathbf{s} = \mathbf{0}`): Store only :math:`i < j`.
  - **Non-zero shift**: A shift is "canonical" if its first non-zero component
    is positive. Only canonical shifts are stored.

  Args:
    position: Atom positions. Shape ``[N, dim]``.
    box: Box matrix. Shape ``[dim, dim]``.
    shifts: Integer shift vectors. Shape ``[num_shifts, dim]``.
    shifts_real: Real-space shifts (``shifts @ box.T``). Shape ``[num_shifts, dim]``.
    zero_shift_idx: Index of the zero shift vector.
    r_cutoff: Cutoff distance.
    capacity: Maximum edges to store.
    fractional_coordinates: If True, positions are fractional.
    update_fn: Function to update the neighbor list.

  Returns:
    NeighborListMultiImage with ``format=OrderedSparse``:

    - ``idx``: ``(receivers, senders)`` each shape ``[capacity]``.
    - ``shifts``: Shape ``[capacity, dim]``.
  """
  N = position.shape[0]
  num_shifts = shifts.shape[0]

  within_cutoff = _compute_pairwise_mask(
    position, box, shifts_real, zero_shift_idx, r_cutoff, fractional_coordinates
  )  # [num_shifts, N, N]

  # Apply ordering to eliminate double-counting
  i_idx = jnp.arange(N)[None, :, None]  # [1, N, 1]
  j_idx = jnp.arange(N)[None, None, :]  # [1, 1, N]
  zero_shift_mask = jnp.arange(num_shifts) == zero_shift_idx  # [num_shifts]

  def is_shift_canonical(s):
    """Check if shift is canonical (first non-zero component positive)."""
    nonzero_mask = s != 0
    first_nonzero_idx = jnp.argmax(nonzero_mask)
    first_val = s[first_nonzero_idx]
    is_zero = jnp.all(s == 0)
    return jnp.where(is_zero, True, first_val > 0)

  shift_is_canonical = jax.vmap(is_shift_canonical)(shifts)  # [num_shifts]

  # Keep mask: zero shift -> i < j, non-zero -> canonical shifts only
  keep_mask = jnp.where(
    zero_shift_mask[:, None, None],
    i_idx < j_idx,
    shift_is_canonical[:, None, None],
  )  # [num_shifts, N, N]

  within_cutoff = within_cutoff & keep_mask

  senders, receivers, edge_shifts, n_valid = _scatter_to_sparse(
    within_cutoff, shifts, capacity, N
  )

  return NeighborListMultiImage(
    idx=(receivers, senders),
    shifts=edge_shifts,
    reference_position=position,
    box=box,
    format=NeighborListFormat.OrderedSparse,
    did_buffer_overflow=(n_valid > capacity),
    max_occupancy=capacity,
    update_fn=update_fn,
  )


def _build_neighbor_list_dense(
  position: Array,  # [N, dim]
  box: Array,  # [dim, dim]
  shifts: Array,  # [num_shifts, dim]
  shifts_real: Array,  # [num_shifts, dim]
  zero_shift_idx: int,
  r_cutoff: float,
  max_neighbors: int,
  fractional_coordinates: bool,
  update_fn: Callable,
) -> NeighborListMultiImage:
  r"""Build neighbor list in Dense format.

  Dense format stores neighbors per atom as a ``[N, max_neighbors]`` array,
  enabling efficient three-body potential computation via vectorized operations.

  For each atom :math:`i`, finds the closest ``max_neighbors`` atoms :math:`j`
  (with shifts :math:`\mathbf{s}`) satisfying:

  .. math::

    \|\mathbf{r}_j + \mathbf{s} \cdot \mathbf{T} - \mathbf{r}_i\| < r_\text{cut}

  Uses ``argsort`` to select top-k neighbors per atom.

  Args:
    position: Atom positions. Shape ``[N, dim]``.
    box: Affine transformation (see ``periodic_general``). Shape ``[dim, dim]``.
    shifts: Integer shift vectors. Shape ``[num_shifts, dim]``.
    shifts_real: Real-space shifts (``shifts @ box.T``). Shape ``[num_shifts, dim]``.
    zero_shift_idx: Index of the zero shift vector.
    r_cutoff: Cutoff distance (scalar).
    max_neighbors: Maximum neighbors per atom.
    fractional_coordinates: If True, positions are fractional.
    update_fn: Function to update the neighbor list.

  Returns:
    NeighborListMultiImage with ``format=Dense``:

    - ``idx``: Neighbor indices. Shape ``[N, max_neighbors]``. Padded with ``N``.
    - ``shifts``: Shift vectors. Shape ``[N, max_neighbors, dim]``.
  """
  N = position.shape[0]
  num_shifts = shifts.shape[0]

  # Compute squared distances for all (shift, i, j)
  dist_sq = _compute_distances_sq(
    position, box, shifts_real, fractional_coordinates
  )  # [num_shifts, N, N]

  # Valid neighbors: within cutoff, excluding self (i==j with zero shift)
  within_cutoff = dist_sq < r_cutoff**2  # [num_shifts, N, N]
  self_mask = jnp.eye(N, dtype=bool)  # [N, N]
  zero_shift_mask = jnp.arange(num_shifts) == zero_shift_idx  # [num_shifts]
  self_interaction = zero_shift_mask[:, None, None] & self_mask[None, :, :]
  valid = within_cutoff & ~self_interaction  # [num_shifts, N, N]

  # Reshape to per-atom view: [N, num_shifts * N]
  # Transpose [num_shifts, N, N] -> [N, num_shifts, N] -> [N, num_shifts * N]
  valid_per_atom = valid.transpose(1, 0, 2).reshape(N, num_shifts * N)
  dist_sq_per_atom = dist_sq.transpose(1, 0, 2).reshape(N, num_shifts * N)

  # Set invalid distances to inf for sorting
  dist_for_sort = jnp.where(valid_per_atom, dist_sq_per_atom, jnp.inf)

  # Select top-k closest neighbors per atom via argsort
  top_k_flat_idx = jnp.argsort(dist_for_sort, axis=-1)[
    :, :max_neighbors
  ]  # [N, max_neighbors]

  # Decode flat index -> (shift_idx, j)
  neighbor_shift_idx = top_k_flat_idx // N  # [N, max_neighbors]
  neighbor_j = top_k_flat_idx % N  # [N, max_neighbors]

  # Gather shift vectors
  neighbor_shifts = shifts[neighbor_shift_idx]  # [N, max_neighbors, dim]

  # Check which entries are actually valid (not inf padding)
  gathered_valid = jnp.take_along_axis(
    valid_per_atom, top_k_flat_idx, axis=-1
  )  # [N, max_neighbors]

  # Replace invalid entries with padding sentinel N
  neighbor_idx = jnp.where(gathered_valid, neighbor_j, N)  # [N, max_neighbors]
  neighbor_shifts = jnp.where(
    gathered_valid[:, :, None], neighbor_shifts, 0
  )  # [N, max_neighbors, dim]

  # Check for overflow
  total_valid_per_atom = jnp.sum(valid_per_atom, axis=-1)  # [N]
  did_overflow = jnp.any(total_valid_per_atom > max_neighbors)

  return NeighborListMultiImage(
    idx=neighbor_idx,
    shifts=neighbor_shifts,
    reference_position=position,
    box=box,
    format=NeighborListFormat.Dense,
    did_buffer_overflow=did_overflow,
    max_occupancy=max_neighbors,
    update_fn=update_fn,
  )


def estimate_max_neighbors(
  r_cutoff: float,
  atomic_density: float = 0.1,
  safety_factor: float = 2.0,
  dim: int = 3,
) -> int:
  r"""Estimate maximum neighbors per atom from atomic density.

  Quick estimation when you don't have box/n_atoms information. Uses:

  .. math::

    N_\text{neighbors} = \text{safety\_factor} \cdot \rho \cdot V_\text{sphere}

  where :math:`\rho` is the atomic density and :math:`V_\text{sphere}` is the
  volume of a sphere with radius ``r_cutoff``.

  Args:
    r_cutoff: Interaction cutoff distance (scalar).
    atomic_density: Atomic density in atoms per unit volume. Default: 0.1.
      Common values: ~0.03 for gases, ~0.1 for liquids, ~0.15 for solids.
    safety_factor: Multiplier for safety margin. Default: 2.0.
    dim: Spatial dimension. Default: 3.

  Returns:
    Estimated maximum neighbors per atom.

  Example:

    .. code-block:: python

       from jax_md.custom_partition import (
           neighbor_list_multi_image,
           estimate_max_neighbors,
       )

       max_nbrs = estimate_max_neighbors(r_cutoff=2.5, atomic_density=0.1)
       neighbor_fn = neighbor_list_multi_image(None, box, r_cutoff, max_neighbors=max_nbrs)

  See Also:
    ``estimate_max_neighbors_from_box``: More accurate estimation when box and
    n_atoms are known, accounts for multiple periodic images.
  """
  if r_cutoff <= 0:
    return 0

  if dim > 3:
    raise ValueError(f'dim must be 1, 2, or 3, got {dim}')

  # Compute sphere volume based on dimension
  if dim == 3:
    sphere_volume = 4.0 / 3.0 * jnp.pi * r_cutoff**3
  elif dim == 2:
    sphere_volume = jnp.pi * r_cutoff**2
  else:  # dim == 1
    sphere_volume = 2.0 * r_cutoff

  expected_neighbors = safety_factor * atomic_density * sphere_volume
  return max(int(jnp.ceil(expected_neighbors)), 1)


def estimate_max_neighbors_from_box(
  box: Array,  # [dim, dim]
  r_cutoff: float,
  n_atoms: int,
  safety_factor: float = 2.0,
  pbc: Optional[Array] = None,  # [dim]
) -> int:
  r"""Estimate maximum neighbors per atom from box and atom count.

  Accurate estimation for multi-image neighbor lists, supporting both fully
  periodic and mixed boundary conditions.

  .. math::

    N_\text{neighbors} = \text{safety\_factor} \cdot \rho \cdot V_\text{eff}

  where :math:`\rho = N / V_\text{box}` is the number density and
  :math:`V_\text{eff}` is the effective cutoff volume (sphere for fully
  periodic, capped for non-periodic directions).

  For non-periodic directions, the cutoff is capped at the box length since
  atoms beyond the box boundary don't exist. This reduces the effective
  cutoff volume and thus the expected neighbor count.

  Args:
    box: Affine transformation (see ``jax_md.space.periodic_general``).
      Shape ``[dim, dim]``. Columns are lattice vectors.
    r_cutoff: Interaction cutoff distance (scalar).
    n_atoms: Number of atoms in the system.
    safety_factor: Multiplier for safety margin. Default: 2.0.
    pbc: Boolean array indicating periodic directions. Shape ``[dim]``.
      Default: all True. For non-periodic directions, the cutoff is capped
      at the box length in that direction.

  Returns:
    Estimated maximum neighbors per atom.

  Raises:
    ValueError: If spatial dimension is not 1, 2, or 3.

  Example:

    .. code-block:: python

       from jax_md.custom_partition import (
           neighbor_list_multi_image,
           estimate_max_neighbors_from_box,
       )

       # Fully periodic
       max_nbrs = estimate_max_neighbors_from_box(box, r_cutoff, n_atoms=N)

       # Mixed: periodic in x,y but not z
       max_nbrs = estimate_max_neighbors_from_box(
           box, r_cutoff, n_atoms=N, pbc=[True, True, False]
       )
       neighbor_fn = neighbor_list_multi_image(
           None, box, r_cutoff, max_neighbors=max_nbrs, pbc=[True, True, False]
       )

  See Also:
    ``estimate_max_neighbors``: Quick estimation when box is not available.
  """
  if r_cutoff <= 0:
    return 0

  box = jnp.asarray(box)
  dim = box.shape[0]

  if dim > 3:
    raise ValueError(f'dim must be 1, 2, or 3, got {dim}')

  if pbc is None:
    pbc = jnp.ones(dim, dtype=bool)
  pbc = jnp.asarray(pbc)

  # Compute density from box
  box_volume = float(jnp.abs(jnp.linalg.det(box)))
  density = n_atoms / box_volume

  # For non-periodic directions, cap the effective cutoff at the box length.
  # This is because atoms can only exist within the box in non-periodic directions.
  # Compute box lengths along each axis (diagonal for orthorhombic, more complex otherwise)
  box_lengths = jnp.linalg.norm(
    box, axis=0
  )  # [dim] - length of each lattice vector

  # Effective cutoff per direction: r_cutoff if periodic, min(r_cutoff, L) if not
  r_eff = jnp.where(pbc, r_cutoff, jnp.minimum(r_cutoff, box_lengths))

  # For computing effective volume, use the minimum effective cutoff
  # This is conservative (may overestimate slightly for anisotropic boxes)
  r_eff_min = float(jnp.min(r_eff))

  # Compute sphere/circle/line volume based on dimension using effective cutoff
  if dim == 3:
    sphere_volume = 4.0 / 3.0 * jnp.pi * r_eff_min**3
  elif dim == 2:
    sphere_volume = jnp.pi * r_eff_min**2
  else:  # dim == 1
    sphere_volume = 2.0 * r_eff_min

  # For multi-image neighbor lists, when r_cutoff > L/2, the cutoff sphere
  # extends into multiple periodic images. However, the atoms in those images
  # are the same physical atoms, just translated. The expected neighbor count
  # is still density * sphere_volume (the number of atoms within the sphere),
  # regardless of how many images are involved.
  expected_neighbors = safety_factor * density * sphere_volume
  return max(int(jnp.ceil(expected_neighbors)), 1)


def neighbor_list_multi_image(
  displacement_or_metric,  # Ignored, for API compatibility
  box: Array,  # [dim, dim]
  r_cutoff: float,
  dr_threshold: float = 0.0,
  capacity_multiplier: float = 1.25,
  pbc: Optional[Array] = None,  # [dim]
  fractional_coordinates: bool = True,
  ordered: bool = False,
  format: NeighborListFormat = NeighborListFormat.Sparse,
  max_neighbors: Optional[int] = None,
  **kwargs,
) -> NeighborListMultiImageFns:
  r"""Returns functions to build neighbor lists for small periodic boxes.

  This function mirrors the API of ``jax_md.partition.neighbor_list`` but
  correctly handles small boxes where :math:`r_\text{cut} > L/2` by
  explicitly enumerating periodic images. Works for any dimension.

  **Algorithm:**

  For each lattice direction :math:`i`, computes the number of shifts needed:

  .. math::

    n_i = \lceil r_\text{cut} / h_i \rceil

  where :math:`h_i` is the perpendicular height of the box along direction
  :math:`i`. Then enumerates all integer shift vectors
  :math:`\mathbf{s} \in [-n_1, n_1] \times \ldots \times [-n_d, n_d]` and finds
  pairs :math:`(i, j)` with:

  .. math::

    \|\mathbf{r}_j + \mathbf{s} \cdot \mathbf{T} - \mathbf{r}_i\| < r_\text{cut}

  **Usage:**

  .. code-block:: python

     from jax_md.custom_partition import (
         neighbor_list_multi_image,
         estimate_max_neighbors,
     )

     # Basic usage: uses default max_neighbors=100
     neighbor_fn = neighbor_list_multi_image(None, box, r_cutoff)
     nbrs = neighbor_fn.allocate(R)

     # With explicit max_neighbors
     neighbor_fn = neighbor_list_multi_image(None, box, r_cutoff, max_neighbors=50)

     # Use estimate_max_neighbors for density-based estimation
     max_nbrs = estimate_max_neighbors(box, r_cutoff, n_atoms=N)
     neighbor_fn = neighbor_list_multi_image(None, box, r_cutoff, max_neighbors=max_nbrs)

     nbrs = neighbor_fn.allocate(R)

     for _ in range(steps):
       nbrs = nbrs.update(state.position)
       if nbrs.did_buffer_overflow:
         nbrs = neighbor_fn.allocate(state.position)
       state = apply_fn(state, nbrs)

  Args:
    displacement_or_metric: Ignored. Accepted for API compatibility with
      ``partition.neighbor_list``. Multi-image computes displacements using
      explicit lattice shifts.
    box: Affine transformation (see ``jax_md.space.periodic_general``).
      Shape ``[dim, dim]``. Columns are lattice vectors.
    r_cutoff: Interaction cutoff distance (scalar).
    dr_threshold: Maximum distance atoms can move before rebuilding.
      Set to 0 to always rebuild. Uses :math:`d_\text{max} < d_\text{thresh}/2`
      as the skip condition.
    capacity_multiplier: Safety factor for neighbor list capacity.
    pbc: Boolean array indicating periodic directions. Shape ``[dim]``.
      Default: all True.
    fractional_coordinates: If True, positions are in fractional coordinates.
    ordered: If True, use OrderedSparse format (one direction per pair).
      Uses 2x less memory. Ignored for Dense format.
    format: Neighbor list format:

      - ``Sparse``: Edge list ``(receivers, senders)``. Shape ``[capacity]``.
      - ``OrderedSparse``: Like Sparse but only :math:`i < j` pairs.
      - ``Dense``: Per-atom neighbors. Shape ``[N, max_neighbors]``.

    max_neighbors: Maximum neighbors per atom. Default: 100. Use
      ``estimate_max_neighbors()`` for density-based estimation.
    **kwargs: Additional arguments (ignored, for API compatibility).

  Returns:
    ``NeighborListMultiImageFns`` with:

    - ``allocate(position)``: Create new neighbor list from positions ``[N, dim]``.
    - ``update(position, neighbors)``: Update existing neighbor list.
  """
  del displacement_or_metric  # Unused - multi-image uses explicit shifts

  # Configuration constants
  DEFAULT_MAX_NEIGHBORS = 256  # Default max neighbors per atom if not specified

  box = jnp.asarray(box)  # [dim, dim]
  dim = box.shape[0]
  use_dense = format is NeighborListFormat.Dense
  use_ordered = ordered or (format is NeighborListFormat.OrderedSparse)

  if pbc is None:
    pbc = jnp.ones(dim, dtype=bool)
  pbc = jnp.asarray(pbc)  # [dim]

  # Pre-compute shift vectors using reciprocal lattice heights
  shifts = _compute_shift_ranges(box, r_cutoff, pbc)  # [num_shifts, dim]
  shifts_real = shifts @ box.T  # [num_shifts, dim]
  zero_shift_idx = int(jnp.argmin(jnp.sum(shifts**2, axis=1)))

  # Compute neighbors_per_atom from max_neighbors or default
  if max_neighbors is not None:
    neighbors_per_atom = int(max_neighbors * capacity_multiplier)
  else:
    neighbors_per_atom = int(DEFAULT_MAX_NEIGHBORS * capacity_multiplier)

  # Displacement threshold for skipping rebuild
  threshold_sq = (dr_threshold / 2.0) ** 2

  # Placeholder for circular reference in NeighborListMultiImage.update
  def update_fn_placeholder(position, neighbors, **kwargs):
    raise NotImplementedError()

  update_fn_ref = [update_fn_placeholder]

  # Cache for JIT-compiled build functions per capacity
  # This avoids recompilation when N stays constant across calls
  build_fn_cache = {}

  def compute_capacity(N: int) -> int:
    """Compute capacity from number of atoms."""
    if use_dense:
      return neighbors_per_atom
    else:
      cap = N * neighbors_per_atom
      if use_ordered:
        cap = cap // 2 + N  # Ordered stores ~half the edges
      return cap

  def get_build_fn(capacity: int):
    """Get or create JIT-compiled build function for given capacity."""
    if capacity in build_fn_cache:
      return build_fn_cache[capacity]

    # Create build function for this capacity
    if use_dense:

      @jax.jit
      def build_fn(pos):  # pos: [N, dim]
        return _build_neighbor_list_dense(
          pos,
          box,
          shifts,
          shifts_real,
          zero_shift_idx,
          r_cutoff,
          capacity,
          fractional_coordinates,
          update_fn_ref[0],
        )

    elif use_ordered:

      @jax.jit
      def build_fn(pos):  # pos: [N, dim]
        return _build_neighbor_list_orderedsparse(
          pos,
          box,
          shifts,
          shifts_real,
          zero_shift_idx,
          r_cutoff,
          capacity,
          fractional_coordinates,
          update_fn_ref[0],
        )

    else:

      @jax.jit
      def build_fn(pos):  # pos: [N, dim]
        return _build_neighbor_list_sparse(
          pos,
          box,
          shifts,
          shifts_real,
          zero_shift_idx,
          r_cutoff,
          capacity,
          fractional_coordinates,
          update_fn_ref[0],
        )

    build_fn_cache[capacity] = build_fn
    return build_fn

  @jax.jit
  def check_needs_rebuild(
    position: Array,  # [N, dim]
    reference_position: Array,  # [N, dim]
  ) -> Array:  # scalar bool
    """Check if maximum displacement exceeds threshold."""
    if fractional_coordinates:
      pos_new = position @ box.T  # [N, dim]
      pos_old = reference_position @ box.T
    else:
      pos_new = position
      pos_old = reference_position
    max_disp_sq = jnp.max(jnp.sum((pos_new - pos_old) ** 2, axis=-1))
    return max_disp_sq >= threshold_sq

  # Choose update strategy at function creation time (not trace time)
  use_threshold = dr_threshold > 0

  def allocate_fn(position: Array, **kwargs) -> NeighborListMultiImage:
    """Allocate a new neighbor list from positions [N, dim].

    Computes capacity based on position.shape[0] and caches the JIT-compiled
    build function for efficient reuse with the same N.
    """
    position = jnp.asarray(position)
    N = position.shape[0]
    capacity = compute_capacity(N)
    build_fn = get_build_fn(capacity)
    return build_fn(position)

  def neighbor_list_fn(
    position: Array,  # [N, dim]
    neighbors: Optional[NeighborListMultiImage] = None,
    **kwargs,
  ) -> NeighborListMultiImage:
    """Build or update neighbor list.

    Args:
      position: Atom positions. Shape ``[N, dim]``.
      neighbors: Existing neighbor list, or None to allocate new.
      **kwargs: Additional arguments (unused, for API compatibility).

    Returns:
      Updated neighbor list.
    """
    position = jnp.asarray(position)

    if neighbors is None:
      # First call: allocate with capacity computed from position
      return allocate_fn(position, **kwargs)

    # Update: reuse existing capacity from neighbor list
    capacity = neighbors.max_occupancy
    build_fn = get_build_fn(capacity)

    # Check if rebuild needed based on displacement threshold
    if use_threshold:
      return jax.lax.cond(
        check_needs_rebuild(position, neighbors.reference_position),
        build_fn,  # True branch: rebuild
        lambda pos: neighbors,  # False branch: return existing
        position,
      )

    return build_fn(position)

  # Close the circular reference
  update_fn_ref[0] = neighbor_list_fn

  return NeighborListMultiImageFns(allocate_fn, neighbor_list_fn)


def neighbor_list_multi_image_mask(
  neighbors: NeighborListMultiImage,
) -> Array:  # [capacity]
  r"""Compute a boolean mask for valid edges in a neighbor list.

  This is equivalent to ``jax_md.partition.neighbor_list_mask``. An edge is
  valid if its receiver index is less than ``N`` (invalid edges are padded
  with index ``N``).

  Args:
    neighbors: A ``NeighborListMultiImage`` (Sparse or OrderedSparse format).

  Returns:
    Boolean mask. Shape ``[capacity]``. True indicates a valid edge.
  """
  N = len(neighbors.reference_position)
  return neighbors.idx[0] < N  # [capacity]


def _compute_displacements(
  position: Array,  # [N, dim]
  neighbors: NeighborListMultiImage,
  fractional_coordinates: bool = True,
) -> Array:  # [capacity, dim]
  r"""Compute displacement vectors for all edges in a neighbor list.

  For each edge from receiver :math:`i` to sender :math:`j` with shift
  :math:`\mathbf{s}`, computes:

  .. math::

    \mathbf{d}_{ij}^{\mathbf{s}} = \mathbf{r}_j + \mathbf{s} \cdot \mathbf{T} - \mathbf{r}_i

  where :math:`\mathbf{T}` is the box matrix.

  Args:
    position: Atom positions. Shape ``[N, dim]``.
    neighbors: A ``NeighborListMultiImage`` (Sparse or OrderedSparse format).
    fractional_coordinates: If True, positions are in fractional coordinates.

  Returns:
    Displacement vectors in Cartesian coordinates. Shape ``[capacity, dim]``.
    Invalid edges are set to zero. Use ``neighbor_list_mask(neighbors)`` to
    filter valid edges.
  """
  box = neighbors.box  # [dim, dim]
  N = position.shape[0]
  mask = neighbor_list_multi_image_mask(neighbors)  # [capacity]

  if fractional_coordinates:
    position_real = position @ box.T  # [N, dim]
  else:
    position_real = position

  # Safe indexing for padding (clip to valid range)
  i_safe = jnp.clip(neighbors.receivers, 0, N - 1)  # [capacity]
  j_safe = jnp.clip(neighbors.senders, 0, N - 1)  # [capacity]
  shifts_real = neighbors.shifts @ box.T  # [capacity, dim]

  # Displacement: r_j + shift - r_i
  dR = (
    position_real[j_safe] + shifts_real - position_real[i_safe]
  )  # [capacity, dim]
  return jnp.where(mask[:, None], dR, 0.0)


def neighbor_list_featurizer(
  fractional_coordinates: bool = True,
):
  r"""Returns a featurizer function for multi-image neighbor lists.

  This mirrors ``jax_md.nn.util.neighbor_list_featurizer`` but works with
  ``NeighborListMultiImage`` which stores explicit shift vectors.

  The returned function computes displacement vectors:

  .. math::

    \mathbf{d}_{ij}^{\mathbf{s}} = \mathbf{r}_j + \mathbf{s} \cdot \mathbf{T} - \mathbf{r}_i

  The box matrix :math:`\mathbf{T}` is taken from the neighbor list itself
  (``neighbor.box``), ensuring consistency between the integer shifts and
  the coordinate transformation.

  Args:
    fractional_coordinates: If True, positions are in fractional coordinates.

  Returns:
    A featurizer function with signature:
    ``featurize(position, neighbor, **kwargs) -> displacements``

    - Input ``position``: Shape ``[N, dim]``.
    - Input ``neighbor``: A ``NeighborListMultiImage``.
    - Output: Displacement vectors. Shape ``[capacity, dim]``.
      Invalid edges are set to unit vectors (to avoid NaN in normalization).

  Example:

    .. code-block:: python

       featurizer = neighbor_list_featurizer(fractional_coordinates=True)
       dR = featurizer(positions, nbrs)  # Shape: [capacity, dim]
       dr = jnp.linalg.norm(dR, axis=-1)  # Shape: [capacity]
  """

  def featurize(
    position: Array,  # [N, dim]
    neighbor: NeighborListMultiImage,
    **kwargs,
  ) -> Array:  # [capacity, dim]
    """Compute displacement vectors from positions + neighbor list."""
    box = neighbor.box  # [dim, dim] - use neighbor list's box for consistency
    N = position.shape[0]
    mask = neighbor_list_multi_image_mask(neighbor)  # [capacity]

    if fractional_coordinates:
      pos_real = position @ box.T  # [N, dim]
    else:
      pos_real = position

    i_safe = jnp.clip(neighbor.receivers, 0, N - 1)  # [capacity]
    j_safe = jnp.clip(neighbor.senders, 0, N - 1)  # [capacity]
    shifts_real = neighbor.shifts @ box.T  # [capacity, dim]

    dR = pos_real[j_safe] + shifts_real - pos_real[i_safe]  # [capacity, dim]
    # Set invalid edges to unit vector to avoid NaN in normalization
    dR = jnp.where(mask[:, None], dR, 1.0)

    return dR

  return featurize
