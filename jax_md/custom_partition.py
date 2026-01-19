import jax.numpy as jnp
from jax_md import dataclasses
from jax_md.partition import NeighborListFormat
from typing import Callable, Tuple, Union

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
      The real-space shift is `shifts @ box`.
    reference_position: Positions when the list was built, shape (N, dim).
    box: Box matrix with rows as lattice vectors, shape (dim, dim).
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
  shifts: Array  # real-space shift = shifts @ box
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
    box: Box matrix with rows as lattice vectors. Shape ``[dim, dim]``.
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
  # Number of shifts needed per direction
  n_max = jnp.ceil(r_cutoff / heights).astype(i32)  # [dim]
  n_max = jnp.where(pbc, n_max, 0)  # [dim], zero for non-periodic

  # Build Cartesian product of shift ranges
  dim = box.shape[0]
  n_max_int = [int(n_max[i]) for i in range(dim)]
  ranges = [jnp.arange(-n, n + 1) for n in n_max_int]  # List of [2*n+1]
  grids = jnp.meshgrid(*ranges, indexing='ij')
  return jnp.stack([g.ravel() for g in grids], axis=-1)  # [num_shifts, dim]
