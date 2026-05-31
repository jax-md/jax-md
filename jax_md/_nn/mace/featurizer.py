"""MACE featurizers: convert JAX-MD neighbor lists into MACE batch dicts."""

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from jax_md import space, partition


def mace_featurizer(
  displacement_fn, config, z_atomic, *, fractional_coordinates=False, head=None
):
  """Create a featurizer for standard MIC neighbor lists.

  Works with Dense, Sparse, and OrderedSparse formats from
  ``partition.neighbor_list``.

  Args:
    displacement_fn: Displacement function from ``jax_md.space``.
    config: Model config containing ``atomic_numbers``.
    z_atomic: Atomic numbers array, shape ``(N,)``.
    fractional_coordinates: Whether R is in fractional coords.
    head: Optional head array for multi-head models.

  Returns:
    ``featurize(R, neighbor, *, box, perturbation=None) -> dict``
  """
  atomic_numbers = tuple(int(x) for x in config['atomic_numbers'])
  C_node = len(atomic_numbers)
  max_Z = int(max(atomic_numbers))
  Z_to_index = -jnp.ones((max_Z + 1,), dtype=jnp.int32)
  Z_to_index = Z_to_index.at[jnp.array(atomic_numbers, dtype=jnp.int32)].set(
    jnp.arange(C_node, dtype=jnp.int32)
  )

  z_values = np.asarray(z_atomic, dtype=np.int32)
  if z_values.ndim != 1:
    raise ValueError('z_atomic must be a one-dimensional array.')
  unsupported = sorted(set(int(z) for z in z_values) - set(atomic_numbers))
  if unsupported:
    raise ValueError(
      f'z_atomic contains atomic numbers not present in config: {unsupported}.'
    )
  z_jax = jnp.asarray(z_values, dtype=jnp.int32)

  r_cutoff = float(config.get('r_max', 5.0))
  far = jnp.asarray(r_cutoff + 1.0, dtype=jnp.float32)
  FAR_SHIFT = jnp.array([far, 0.0, 0.0], dtype=jnp.float32)

  head_value = None
  if head is not None:
    head_value = jnp.asarray(head, dtype=jnp.int32).reshape(-1)

  def featurize(R, neighbor, *, box, perturbation=None):
    R_in = jnp.asarray(R, dtype=jnp.float32)
    cell = jnp.asarray(box)
    if cell.shape == (3,):
      cell = jnp.diag(cell)
    N = R_in.shape[0]

    if partition.is_sparse(neighbor.format):
      idx = jnp.asarray(neighbor.idx, dtype=jnp.int32)
      recv, send = idx[0], idx[1]
      valid = (
        (send >= 0) & (send < N) & (recv >= 0) & (recv < N) & (send != recv)
      )
      send = jnp.where(valid, send, jnp.zeros_like(send))
      recv = jnp.where(valid, recv, send)
      if neighbor.format is partition.OrderedSparse:
        send, recv = (
          jnp.concatenate([send, recv], axis=0),
          jnp.concatenate([recv, send], axis=0),
        )
        valid = jnp.concatenate([valid, valid], axis=0)
    else:
      idx = jnp.asarray(neighbor.idx, dtype=jnp.int32)
      M = idx.shape[1]
      slot_valid = (idx >= 0) & (idx < N)
      recv = jnp.where(slot_valid, idx, jnp.zeros_like(idx))
      self_mask = recv == jnp.arange(N, dtype=jnp.int32)[:, None]
      slot_valid = slot_valid & (~self_mask)
      recv = jnp.where(slot_valid, idx, jnp.zeros_like(idx))
      send = jnp.repeat(jnp.arange(N, dtype=jnp.int32), M)
      recv = recv.reshape(-1).astype(jnp.int32)
      valid = slot_valid.reshape(-1)
      recv = jnp.where(valid, recv, send)

    if fractional_coordinates:
      disp = jax.vmap(lambda Ra, Rb: displacement_fn(Ra, Rb, box=cell))
    else:
      disp = jax.vmap(lambda Ra, Rb: displacement_fn(Ra, Rb, new_box=cell))
    dR = -disp(R_in[send], R_in[recv])

    R_cart = space.transform(cell, R_in) if fractional_coordinates else R_in
    R_cart = R_cart.astype(jnp.float32)
    delta = R_cart[recv] - R_cart[send]
    exact_shifts = (dR - delta).astype(jnp.float32)

    inv_cell = space.inverse(cell)
    unit_shifts = jnp.rint(space.transform(inv_cell, exact_shifts)).astype(
      jnp.int32
    )
    unit_shifts = lax.stop_gradient(unit_shifts).astype(jnp.float32)
    shifts = lax.stop_gradient(exact_shifts)

    shifts = jnp.where(valid[:, None], shifts, FAR_SHIFT[None, :])
    unit_shifts = jnp.where(
      valid[:, None], unit_shifts, jnp.zeros_like(unit_shifts)
    )

    if perturbation is not None:
      pert = jnp.asarray(perturbation)
      if pert.ndim == 0:
        cell = cell * pert
      else:
        cell = space.raw_transform(pert, cell)
      R_cart = space.raw_transform(pert, R_cart).astype(jnp.float32)
      shifts = space.raw_transform(pert, shifts).astype(jnp.float32)

    species = Z_to_index[z_jax[:N]]

    out = {
      'positions': R_cart,
      'node_attrs': jax.nn.one_hot(species, num_classes=C_node),
      'node_attrs_index': species,
      'edge_index': jnp.stack([send, recv], axis=0),
      'shifts': shifts,
      'unit_shifts': unit_shifts,
      'batch': jnp.zeros((N,), dtype=jnp.int32),
      'ptr': jnp.array([0, N], dtype=jnp.int32),
      'cell': cell[None, :, :],
    }
    if head_value is not None:
      out['head'] = head_value
    return out

  return featurize


def mace_multi_image_featurizer(
  config, z_atomic, *, fractional_coordinates=False, head=None
):
  """Create a featurizer for multi-image neighbor lists with explicit shifts.

  Works with ``custom_partition.neighbor_list_multi_image`` which stores
  integer lattice shifts on the neighbor list. Keeps periodic-image
  self-edges and only removes true zero-shift self-loops.

  Args:
    config: Model config containing ``atomic_numbers``.
    z_atomic: Atomic numbers array, shape ``(N,)``.
    head: Optional head array for multi-head models.

  Returns:
    ``featurize(R, neighbor, *, box, perturbation=None) -> dict``
  """
  atomic_numbers = tuple(int(x) for x in config['atomic_numbers'])
  C_node = len(atomic_numbers)
  max_Z = int(max(atomic_numbers))
  Z_to_index = -jnp.ones((max_Z + 1,), dtype=jnp.int32)
  Z_to_index = Z_to_index.at[jnp.array(atomic_numbers, dtype=jnp.int32)].set(
    jnp.arange(C_node, dtype=jnp.int32)
  )

  z_values = np.asarray(z_atomic, dtype=np.int32)
  if z_values.ndim != 1:
    raise ValueError('z_atomic must be a one-dimensional array.')
  unsupported = sorted(set(int(z) for z in z_values) - set(atomic_numbers))
  if unsupported:
    raise ValueError(
      f'z_atomic contains atomic numbers not present in config: {unsupported}.'
    )
  z_jax = jnp.asarray(z_values, dtype=jnp.int32)

  r_cutoff = float(config.get('r_max', 5.0))
  far = jnp.asarray(r_cutoff + 1.0, dtype=jnp.float32)
  FAR_SHIFT = jnp.array([far, 0.0, 0.0], dtype=jnp.float32)

  head_value = None
  if head is not None:
    head_value = jnp.asarray(head, dtype=jnp.int32).reshape(-1)

  def featurize(R, neighbor, *, box, perturbation=None):
    R_in = jnp.asarray(R, dtype=jnp.float32)
    cell = jnp.asarray(box)
    if cell.shape == (3,):
      cell = jnp.diag(cell)
    N = R_in.shape[0]

    if partition.is_sparse(neighbor.format):
      receivers, senders = neighbor.idx
      send = jnp.asarray(senders, dtype=jnp.int32)
      recv = jnp.asarray(receivers, dtype=jnp.int32)
      valid = (send >= 0) & (send < N) & (recv >= 0) & (recv < N)
      send = jnp.where(valid, send, jnp.zeros_like(send))
      recv = jnp.where(valid, recv, send)
    else:
      idx = jnp.asarray(neighbor.idx, dtype=jnp.int32)
      M = idx.shape[1]
      slot_valid = (idx >= 0) & (idx < N)
      send = jnp.where(slot_valid, idx, jnp.zeros_like(idx))
      recv = jnp.repeat(jnp.arange(N, dtype=jnp.int32), M)
      send = send.reshape(-1).astype(jnp.int32)
      valid = slot_valid.reshape(-1)
      send = jnp.where(valid, send, recv)

    unit_shifts = lax.stop_gradient(
      -jnp.asarray(neighbor.shifts, dtype=jnp.float32)
    )
    if unit_shifts.ndim == 3:
      unit_shifts = unit_shifts.reshape((-1, 3))

    if neighbor.format is partition.OrderedSparse:
      send, recv = (
        jnp.concatenate([send, recv], axis=0),
        jnp.concatenate([recv, send], axis=0),
      )
      valid = jnp.concatenate([valid, valid], axis=0)
      unit_shifts = jnp.concatenate([unit_shifts, -unit_shifts], axis=0)

    zero_shift_self = jnp.all(unit_shifts == 0, axis=-1) & (send == recv)
    valid = valid & ~zero_shift_self

    shifts = space.transform(cell, unit_shifts).astype(jnp.float32)
    shifts = jnp.where(valid[:, None], shifts, FAR_SHIFT[None, :])
    unit_shifts = jnp.where(
      valid[:, None], unit_shifts, jnp.zeros_like(unit_shifts)
    )

    R_cart = space.transform(cell, R_in) if fractional_coordinates else R_in
    R_cart = R_cart.astype(jnp.float32)
    if perturbation is not None:
      pert = jnp.asarray(perturbation)
      if pert.ndim == 0:
        cell = cell * pert
      else:
        cell = space.raw_transform(pert, cell)
      R_cart = space.raw_transform(pert, R_cart).astype(jnp.float32)
      shifts = space.raw_transform(pert, shifts).astype(jnp.float32)

    species = Z_to_index[z_jax[:N]]

    out = {
      'positions': R_cart,
      'node_attrs': jax.nn.one_hot(species, num_classes=C_node),
      'node_attrs_index': species,
      'edge_index': jnp.stack([send, recv], axis=0),
      'shifts': shifts,
      'unit_shifts': unit_shifts,
      'batch': jnp.zeros((N,), dtype=jnp.int32),
      'ptr': jnp.array([0, N], dtype=jnp.int32),
      'cell': cell[None, :, :],
    }
    if head_value is not None:
      out['head'] = head_value
    return out

  return featurize
