"""Structure-mapped functions for multi-image neighbor lists.

This module provides analogues to ``jax_md.smap`` functions that work with
``NeighborListMultiImage`` to correctly handle small periodic boxes where
:math:`r_\\text{cut} > L/2`.

The key difference from standard ``smap`` functions is that these use explicit
lattice shifts stored in the neighbor list rather than relying on the minimum
image convention.
"""

from typing import Callable, Tuple, Optional
import functools
from functools import partial

import jax
import jax.numpy as jnp
from jax import ops
from jax_md import partition, space, util
from jax_md.custom_partition import (
  NeighborListMultiImage,
  NeighborListFormat,
  neighbor_list_multi_image_mask,
)

# Type aliases
Array = jnp.ndarray
f32 = jnp.float32
i32 = jnp.int32


def pair_neighbor_list_multi_image(
  pair_fn: Callable[..., Array],
  displacement_fn=None,  # Ignored; always uses space.free() internally
  species: Optional[Array] = None,  # [N] or None
  reduce_axis: Optional[Tuple[int, ...]] = None,
  ignore_unused_parameters: bool = False,  # For API compatibility
  fractional_coordinates: bool = True,
  **static_kwargs,
) -> Callable[[Array, NeighborListMultiImage], Array]:
  r"""Creates a function for pair potentials using multi-image neighbors.

  This function is analogous to ``jax_md.smap.pair_neighbor_list`` but works
  with ``NeighborListMultiImage`` to correctly handle small periodic boxes
  where :math:`r_\text{cut} > L/2`.

  **Displacement Function:**

  Always uses ``space.free()`` for displacement computation since periodicity
  is handled via explicit lattice shifts. The ``displacement_fn`` parameter is
  ignored (for API compatibility with ``smap.pair_neighbor_list``). Supports
  ``perturbation`` kwarg for stress calculation via ``quantity.stress``.

  For each edge :math:`(i, j)` with shift :math:`\mathbf{s}`, computes:

  .. math::

    E_{ij}^{\mathbf{s}} = f\left(\|\mathbf{r}_j + \mathbf{s} \cdot \mathbf{T} - \mathbf{r}_i\|\right)

  where :math:`f` is the pair function and :math:`\mathbf{T}` is the box matrix.

  The pair function should have the signature::

    pair_fn(dr, **kwargs) -> energy

  where ``dr`` is an array of scalar pairwise distances of shape ``[capacity]``.
  This matches the signature expected by ``smap.pair_neighbor_list``, so the
  same pair function (e.g., ``energy.lennard_jones``) works with both.

  **Gradient handling:**

  Uses ``space.transform`` for coordinate transformations, which has a custom
  JVP that keeps gradients in the same coordinate system as inputs. When using
  fractional coordinates, ``jax.grad(energy_fn)`` returns forces in fractional
  coordinates (compatible with fractional-coordinate dynamics).

  **Format handling:**

  - ``Sparse``: Both :math:`i \\to j` and :math:`j \\to i` are stored, so
    energies are divided by 2. Supports per-particle energies.
  - ``OrderedSparse``: Only one direction per pair, no division needed.
    Does **not** support per-particle energies (raises ``ValueError``).
  - ``Dense``: Stores neighbors per atom as ``[N, max_neighbors]``. Both
    directions are stored, so energies are divided by 2. Supports per-particle
    energies via sum over axis 1.

  Args:
    pair_fn: A function that computes pairwise energies from distances.
      Examples: ``energy.lennard_jones``, ``energy.morse``, ``energy.soft_sphere``.
      Signature: ``(dr: Array[capacity], **kwargs) -> Array[capacity]``.
    displacement_fn: Ignored. Always uses ``space.free()`` internally.
    species: Optional species array. Shape ``[N]``. If provided, kwargs like
      ``sigma`` and ``epsilon`` should have shape ``[max_species, max_species]``
      and will be indexed per-pair.
    fractional_coordinates: If True, positions are in fractional coordinates.
    reduce_axis: Axis over which to reduce the energy. If ``None`` (default),
      sums all pair energies to a scalar. If specified, returns per-atom
      energies of shape ``[N]``. **Note:** Per-atom energies are not supported
      with ``OrderedSparse`` format (raises ``ValueError``).
    **static_kwargs: Static parameters passed to the pair function (e.g.,
      ``sigma``, ``epsilon``). Can be overridden at call time.

  Returns:
    An energy function with signature:
    ``energy_fn(R, neighbor, **kwargs) -> Array``

    - Input ``R``: Positions. Shape ``[N, dim]``.
    - Input ``neighbor``: A ``NeighborListMultiImage``.
    - Output: Total energy (scalar) or per-atom energies (shape ``[N]``).

  Example:

    .. code-block:: python

       from jax_md import energy
       from jax_md.custom_partition import neighbor_list_multi_image
       from jax_md.custom_smap import pair_neighbor_list_multi_image

       # Create Lennard-Jones energy function for multi-image neighbors
       lj_energy = pair_neighbor_list_multi_image(
           energy.lennard_jones,
           sigma=1.0,
           epsilon=1.0,
       )

       # Use with multi-image neighbor list
       neighbor_fn = neighbor_list_multi_image(None, box, r_cutoff, n_atoms=N)
       nbrs = neighbor_fn.allocate(positions)
       E = lj_energy(positions, nbrs)

       # Compute forces via autodiff
       force_fn = jax.grad(lambda R, nbrs: -lj_energy(R, nbrs))
       F = force_fn(positions, nbrs)  # Shape: [N, dim]
  """
  # Use space.free() displacement which handles perturbation via kwargs
  # Delete the passed displacement_fn since it is ignored.
  del displacement_fn
  displacement_fn, _ = space.free()

  def energy_fn(
    R: Array,  # [N, dim]
    neighbor: NeighborListMultiImage,
    **kwargs,
  ) -> Array:  # scalar or [N]
    """Compute total pair energy."""
    merged_kwargs = {**static_kwargs, **kwargs}
    _species = merged_kwargs.pop('species', species)

    box = neighbor.box  # [dim, dim]
    N = R.shape[0]

    # Compute Cartesian positions using space.transform for correct gradients.
    # Note: space.transform has a custom JVP that keeps gradients in the same
    # coordinate system as inputs (fractional -> fractional forces).
    if fractional_coordinates:
      R_real = space.transform(box, R)  # [N, dim]
    else:
      R_real = R  # [N, dim]

    # Handle Dense vs Sparse formats differently
    if partition.is_sparse(neighbor.format):
      # Sparse/OrderedSparse: edge-list format
      # idx shape: [2, capacity], shifts shape: [capacity, dim]
      mask = neighbor_list_multi_image_mask(neighbor)  # [capacity]

      # Compute displacement vectors: r_j + shift - r_i
      i_safe = jnp.clip(neighbor.receivers, 0, N - 1)  # [capacity]
      j_safe = jnp.clip(neighbor.senders, 0, N - 1)  # [capacity]
      shifts_real = space.transform(box, neighbor.shifts)  # [capacity, dim]

      # Use displacement function which handles perturbation naturally
      Ra = R_real[i_safe]  # [capacity, dim]
      Rb_shifted = R_real[j_safe] + shifts_real  # [capacity, dim]
      d = jax.vmap(partial(displacement_fn, **merged_kwargs))
      dR = d(Ra, Rb_shifted)  # [capacity, dim]

      # Compute scalar distances
      dr = space.distance(dR)  # [capacity]

      # Handle species-dependent parameters
      if _species is not None:
        species_i = _species[i_safe]  # [capacity]
        species_j = _species[j_safe]  # [capacity]
        processed_kwargs = {}
        for key, val in merged_kwargs.items():
          if jnp.ndim(val) == 2:
            processed_kwargs[key] = val[species_i, species_j]  # [capacity]
          else:
            processed_kwargs[key] = val
        merged_kwargs = processed_kwargs

      # Compute pair energies
      pair_energies = pair_fn(dr, **merged_kwargs)  # [capacity]
      zero = jnp.zeros((), dtype=pair_energies.dtype)
      pair_energies = jnp.where(mask, pair_energies, zero)  # [capacity]

      # Normalization: OrderedSparse stores one direction, Sparse stores both
      normalization = (
        1.0 if neighbor.format is NeighborListFormat.OrderedSparse else 2.0
      )

      if reduce_axis is None:
        return util.high_precision_sum(pair_energies) / normalization
      else:
        # Per-particle energy via segment_sum
        if neighbor.format is NeighborListFormat.OrderedSparse:
          raise ValueError(
            'Cannot compute per-particle energies with OrderedSparse format. '
            'OrderedSparse stores only one direction per pair, so segment_sum '
            'would assign the full pair energy to the receiver atom only. '
            'Use Sparse or Dense format for per-particle energies.'
          )
        particle_energies = ops.segment_sum(
          pair_energies * mask, neighbor.receivers, N
        )  # [N]
        return particle_energies / normalization

    else:
      # Dense format: per-atom neighbor arrays
      # idx shape: [N, max_neighbors], shifts shape: [N, max_neighbors, dim]
      idx = neighbor.idx  # [N, max_neighbors]
      shifts = neighbor.shifts  # [N, max_neighbors, dim]
      mask = idx < N  # [N, max_neighbors]

      # Get neighbor positions with safe indexing
      j_safe = jnp.clip(idx, 0, N - 1)  # [N, max_neighbors]
      R_neigh = R_real[j_safe]  # [N, max_neighbors, dim]

      # Transform shifts to real space
      # shifts has shape [N, max_neighbors, dim], need to reshape for transform
      shifts_flat = shifts.reshape(
        -1, shifts.shape[-1]
      )  # [N*max_neighbors, dim]
      shifts_real_flat = space.transform(
        box, shifts_flat
      )  # [N*max_neighbors, dim]
      shifts_real = shifts_real_flat.reshape(
        shifts.shape
      )  # [N, max_neighbors, dim]

      # Compute displacements using displacement function (handles perturbation)
      # R_real[:, None, :] broadcasts to [N, 1, dim]
      Ra = jnp.broadcast_to(
        R_real[:, None, :], R_neigh.shape
      )  # [N, max_neighbors, dim]
      Rb_shifted = R_neigh + shifts_real  # [N, max_neighbors, dim]

      # Flatten for vmap, compute displacements, reshape back
      Ra_flat = Ra.reshape(-1, Ra.shape[-1])  # [N*max_neighbors, dim]
      Rb_flat = Rb_shifted.reshape(
        -1, Rb_shifted.shape[-1]
      )  # [N*max_neighbors, dim]
      d = jax.vmap(partial(displacement_fn, **merged_kwargs))
      dR_flat = d(Ra_flat, Rb_flat)  # [N*max_neighbors, dim]
      dR = dR_flat.reshape(
        idx.shape[0], idx.shape[1], -1
      )  # [N, max_neighbors, dim]

      # Compute scalar distances
      dr = space.distance(dR)  # [N, max_neighbors]

      # Handle species-dependent parameters
      if _species is not None:
        species_i = jnp.broadcast_to(
          _species[:, None], idx.shape
        )  # [N, max_neighbors]
        species_j = _species[j_safe]  # [N, max_neighbors]
        processed_kwargs = {}
        for key, val in merged_kwargs.items():
          if jnp.ndim(val) == 2:
            processed_kwargs[key] = val[
              species_i, species_j
            ]  # [N, max_neighbors]
          else:
            processed_kwargs[key] = val
        merged_kwargs = processed_kwargs

      # Compute pair energies
      pair_energies = pair_fn(dr, **merged_kwargs)  # [N, max_neighbors]
      zero = jnp.zeros((), dtype=pair_energies.dtype)
      pair_energies = jnp.where(mask, pair_energies, zero)  # [N, max_neighbors]

      # Dense stores both directions, so divide by 2
      normalization = 2.0

      if reduce_axis is None:
        return util.high_precision_sum(pair_energies) / normalization
      else:
        # Per-particle energy: sum over neighbors (axis 1)
        particle_energies = util.high_precision_sum(
          pair_energies, axis=1
        )  # [N]
        return particle_energies / normalization

  return energy_fn
