"""Featurizer for converting JAX-MD data structures to UMA input format.

This module converts JAX-MD neighbor lists and atomic data into the
input format expected by the UMA backbone model.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict

import jax.numpy as jnp
from jax import vmap

from jax_md import partition


def uma_featurizer(
  displacement_fn: Callable,
  cutoff: float = 5.0,
) -> Callable:
  """Create a featurizer that converts JAX-MD data to UMA input format.

  Args:
      displacement_fn: Displacement function from jax_md.space.
      cutoff: Distance cutoff in Angstroms.

  Returns:
      Function with signature:
        featurize(atomic_numbers, position, neighbor, **kwargs) -> dict
  """

  def featurize(
    atomic_numbers: jnp.ndarray,
    position: jnp.ndarray,
    neighbor,
    charge: jnp.ndarray | None = None,
    spin: jnp.ndarray | None = None,
    dataset_idx: jnp.ndarray | None = None,
    **kwargs,
  ) -> Dict[str, jnp.ndarray]:
    """Convert JAX-MD data to UMA input format.

    Args:
        atomic_numbers: Integer atomic numbers, shape [num_atoms].
        position: Atomic positions, shape [num_atoms, 3].
        neighbor: JAX-MD neighbor list (Sparse format).
        charge: System charge, shape [num_systems] (default: [0]).
        spin: System spin, shape [num_systems] (default: [0]).
        dataset_idx: Integer dataset index, shape [num_systems] (default: [0]).
        **kwargs: Additional kwargs passed to displacement_fn.

    Returns:
        Dictionary with UMA input tensors.
    """
    num_atoms = position.shape[0]

    # Build edge index from neighbor list
    if hasattr(neighbor, 'idx') and neighbor.format == partition.Sparse:
      # Sparse format: idx is [2, num_neighbors] = [receivers, senders]
      receivers = neighbor.idx[0]
      senders = neighbor.idx[1]
      mask = partition.neighbor_list_mask(neighbor, True)
    elif hasattr(neighbor, 'idx'):
      # Dense format: convert to sparse
      graph = partition.to_jraph(neighbor)
      senders = graph.senders
      receivers = graph.receivers
      if senders is None or receivers is None:
        raise ValueError(
          'Converted neighbor list did not produce sparse edges.'
        )
      mask = (senders < num_atoms) & (receivers < num_atoms)
    else:
      raise ValueError(f'Unsupported neighbor list format: {type(neighbor)}')

    edge_index = jnp.stack([senders, receivers], axis=0)

    # Compute displacement vectors in FairChem's convention:
    # source image - target = pos[sender] - pos[receiver].
    d_fn = vmap(partial(displacement_fn, **kwargs))
    R_senders = position[senders]
    R_receivers = position[receivers]
    edge_distance_vec = d_fn(R_senders, R_receivers)

    # Replace invalid edge vectors with a displacement whose norm exceeds
    # the cutoff so the backbone's PolynomialEnvelope returns exactly 0.
    # Using zero would give norm=0 < cutoff, causing envelope=1 and
    # degenerate Wigner matrices for padding entries.
    _beyond_cutoff = jnp.array([cutoff + 1.0, 0.0, 0.0])
    edge_distance_vec = jnp.where(
      mask[:, None], edge_distance_vec, _beyond_cutoff
    )

    # Filter edges beyond cutoff
    edge_distance = jnp.linalg.norm(edge_distance_vec, axis=-1)
    valid = mask & (edge_distance < cutoff) & (edge_distance > 1e-8)

    # For single-system MD, batch is all zeros
    batch = jnp.zeros(num_atoms, dtype=jnp.int32)

    # Default system-level properties
    if charge is None:
      charge = jnp.zeros(1)
    if spin is None:
      spin = jnp.zeros(1)
    if dataset_idx is None:
      dataset_idx = jnp.zeros(1, dtype=jnp.int32)

    return {
      'positions': position,
      'atomic_numbers': atomic_numbers,
      'batch': batch,
      'edge_index': edge_index,
      'edge_distance_vec': edge_distance_vec,
      'charge': charge,
      'spin': spin,
      'dataset_idx': dataset_idx,
      'edge_mask': valid,
    }

  return featurize


def uma_multi_image_featurizer(
  displacement_fn: Callable | None = None,
  cutoff: float = 5.0,
) -> Callable:
  """Create a UMA featurizer for multi-image neighbor lists.

  ``custom_partition.neighbor_list_multi_image`` stores explicit lattice
  shifts, so this featurizer computes edge vectors directly as
  ``pos[sender] + shift @ box.T - pos[receiver]`` instead of using a minimum
  image displacement function. It keeps same-atom periodic image edges and
  only removes true zero-shift self loops.

  Args:
      displacement_fn: Unused. Present for API compatibility with
          ``uma_featurizer`` and other JAX-MD featurizer factories.
      cutoff: Distance cutoff in Angstroms.

  Returns:
      Function with signature:
        featurize(atomic_numbers, position, neighbor, **kwargs) -> dict
  """
  del displacement_fn

  def featurize(
    atomic_numbers: jnp.ndarray,
    position: jnp.ndarray,
    neighbor,
    charge: jnp.ndarray | None = None,
    spin: jnp.ndarray | None = None,
    dataset_idx: jnp.ndarray | None = None,
    **kwargs,
  ) -> Dict[str, jnp.ndarray]:
    """Convert a multi-image neighbor list to UMA input format."""
    from jax_md.custom_partition import neighbor_list_multi_image_mask

    num_atoms = position.shape[0]
    receivers = neighbor.idx[0]
    senders = neighbor.idx[1]
    edge_index = jnp.stack([senders, receivers], axis=0)

    mask = neighbor_list_multi_image_mask(neighbor)
    is_true_self = (senders == receivers) & jnp.all(
      neighbor.shifts == 0, axis=-1
    )
    mask = mask & ~is_true_self

    sender_pos = position[senders]
    receiver_pos = position[receivers]
    box = kwargs.get('box', neighbor.box)
    shifts = neighbor.shifts.astype(position.dtype)
    shifts_cart = shifts @ box.T
    edge_distance_vec = sender_pos + shifts_cart - receiver_pos

    _beyond_cutoff = jnp.array([cutoff + 1.0, 0.0, 0.0], dtype=position.dtype)
    edge_distance_vec = jnp.where(
      mask[:, None], edge_distance_vec, _beyond_cutoff
    )

    edge_distance = jnp.linalg.norm(edge_distance_vec, axis=-1)
    valid = mask & (edge_distance < cutoff) & (edge_distance > 1e-8)

    batch = jnp.zeros(num_atoms, dtype=jnp.int32)

    if charge is None:
      charge = jnp.zeros(1)
    if spin is None:
      spin = jnp.zeros(1)
    if dataset_idx is None:
      dataset_idx = jnp.zeros(1, dtype=jnp.int32)

    return {
      'positions': position,
      'atomic_numbers': atomic_numbers,
      'batch': batch,
      'edge_index': edge_index,
      'edge_distance_vec': edge_distance_vec,
      'charge': charge,
      'spin': spin,
      'dataset_idx': dataset_idx,
      'edge_mask': valid,
    }

  return featurize
