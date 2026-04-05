# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Featurizer for converting JAX-MD data structures to UMA input format.

This module converts JAX-MD neighbor lists and atomic data into the
input format expected by the UMA backbone model.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional

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
      # Sparse format: idx is [2, num_neighbors]
      senders = neighbor.idx[0]
      receivers = neighbor.idx[1]
      mask = partition.neighbor_list_mask(neighbor, True)
    elif hasattr(neighbor, 'idx'):
      # Dense format: convert to sparse
      graph = partition.to_jraph(neighbor)
      senders = graph.senders
      receivers = graph.receivers
      mask = (senders < num_atoms) & (receivers < num_atoms)
    else:
      raise ValueError(f'Unsupported neighbor list format: {type(neighbor)}')

    edge_index = jnp.stack([senders, receivers], axis=0)

    # Compute displacement vectors
    d_fn = vmap(partial(displacement_fn, **kwargs))
    R_senders = position[senders]
    R_receivers = position[receivers]
    edge_distance_vec = d_fn(R_receivers, R_senders)

    # Mask out invalid edges
    edge_distance_vec = jnp.where(mask[:, None], edge_distance_vec, 0.0)

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
