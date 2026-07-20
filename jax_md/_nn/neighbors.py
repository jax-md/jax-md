"""Shared neighbor-list construction for the pretrained models."""

from __future__ import annotations

import jax.numpy as jnp
from jax_md import partition, space


def neighbor_displacement(positions, box=None, *, periodic: bool):
  """Return a displacement function and neighbor kwargs for the space."""
  if not periodic:
    displacement, _ = space.free()
    return displacement, {}
  if box is None:
    raise ValueError('periodic neighbor lists require a box.')
  jax_box = jnp.swapaxes(jnp.asarray(box, dtype=positions.dtype), -1, -2)
  displacement, _ = space.periodic_general(
    jax_box,
    fractional_coordinates=False,
  )
  return displacement, {'box': jax_box}


def get_neighbors(
  positions,
  box=None,
  *,
  cutoff: float,
  format: partition.NeighborListFormat = partition.NeighborListFormat.Dense,
  cell_atom_threshold: int = 64,
  cell_capacity_multiplier: float = 1.5,
  neighbors=None,
  periodic: bool = False,
  dr_threshold: float = 0.0,
):
  """Allocate a neighbor list, or update ``neighbors`` when given."""
  if neighbors is not None:
    return neighbors.update(positions)

  num_atoms = int(positions.shape[0])
  use_cell_list = periodic and num_atoms >= int(cell_atom_threshold)
  displacement, neighbor_kwargs = neighbor_displacement(
    positions,
    box,
    periodic=periodic,
  )
  neighbor_fn = partition.neighbor_list(
    displacement,
    jnp.asarray(1.0, dtype=positions.dtype),
    float(cutoff),
    dr_threshold=float(dr_threshold),
    capacity_multiplier=float(cell_capacity_multiplier),
    disable_cell_list=not use_cell_list,
    mask_self=True,
    fractional_coordinates=False,
    format=format,
  )
  return neighbor_fn.allocate(positions, **neighbor_kwargs)
