from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from jax_md import space, partition


def as_scalar(x):
  x = jnp.asarray(x)
  return jnp.reshape(x, ()) if x.shape == (1,) else x


def make_cell_1x3x3(box_in, dtype):
  b = jnp.asarray(box_in)
  if b.shape == (3,):
    cell = jnp.diag(b)
  elif b.shape == (3, 3):
    cell = b
  else:
    raise ValueError(f'Unexpected box shape {b.shape} (expected (3,) or (3,3))')
  return cell[None, :, :].astype(dtype)


def to_cartesian(R, cell, fractional_coordinates: bool):
  R = jnp.asarray(R)
  if fractional_coordinates:
    return space.transform(cell, R)
  return R


def perturb_cell(cell, perturbation):
  perturbation = jnp.asarray(perturbation)
  if perturbation.ndim == 0:
    return cell * perturbation
  if perturbation.shape == (3,):
    return jnp.diag(perturbation) @ cell
  if perturbation.shape == (3, 3):
    return perturbation @ cell
  raise ValueError(
    f'Unexpected perturbation shape {perturbation.shape} '
    '(expected scalar, (3,), or (3,3))'
  )


def validate_atomic_numbers(z_atomic, atomic_numbers, n_template):
  z_values = np.asarray(z_atomic, dtype=np.int32)
  if z_values.ndim != 1:
    raise ValueError('z_atomic must be a one-dimensional array.')
  if z_values.shape[0] > n_template:
    raise ValueError(
      f'z_atomic has {z_values.shape[0]} entries, but template only supports '
      f'{n_template} atoms.'
    )
  unsupported = sorted(set(int(z) for z in z_values) - set(atomic_numbers))
  if unsupported:
    raise ValueError(
      f'z_atomic contains atomic numbers not present in config: {unsupported}.'
    )
  return jnp.asarray(z_values, dtype=jnp.int32)


def dense_neighbor_topology(neighbor_idx, n_particles):
  neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
  neighbor_count = neighbor_idx.shape[1]

  valid_slot = (neighbor_idx >= 0) & (neighbor_idx < n_particles)
  receivers0 = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))

  self_edge = receivers0 == jnp.arange(n_particles, dtype=jnp.int32)[:, None]
  valid_slot = valid_slot & (~self_edge)
  receivers0 = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))

  senders = jnp.repeat(jnp.arange(n_particles, dtype=jnp.int32), neighbor_count)
  receivers = receivers0.reshape(-1).astype(jnp.int32)
  valid_e = valid_slot.reshape(-1)
  receivers = jnp.where(valid_e, receivers, senders)
  return senders, receivers, valid_e


def dense_neighbor_topology_with_shifts(neighbor_idx, n_particles):
  neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
  neighbor_count = neighbor_idx.shape[1]

  valid_slot = (neighbor_idx >= 0) & (neighbor_idx < n_particles)
  receivers = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))
  senders = jnp.repeat(jnp.arange(n_particles, dtype=jnp.int32), neighbor_count)
  receivers = receivers.reshape(-1).astype(jnp.int32)
  valid_e = valid_slot.reshape(-1)
  receivers = jnp.where(valid_e, receivers, senders)
  return senders, receivers, valid_e


def sparse_neighbor_topology(neighbor_idx, n_particles):
  neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
  receivers = neighbor_idx[0]
  senders = neighbor_idx[1]
  valid_e = (
    (senders >= 0)
    & (senders < n_particles)
    & (receivers >= 0)
    & (receivers < n_particles)
    & (senders != receivers)
  )
  senders = jnp.where(valid_e, senders, jnp.zeros_like(senders))
  receivers = jnp.where(valid_e, receivers, senders)
  return senders, receivers, valid_e


def sparse_multi_image_topology(neighbor_idx, n_particles):
  # Multi-image neighbor lists store (receivers, senders); MACE expects the
  # reverse ordering.
  receivers0, senders0 = neighbor_idx
  senders = jnp.asarray(receivers0, dtype=jnp.int32)
  receivers = jnp.asarray(senders0, dtype=jnp.int32)
  valid_e = (
    (senders >= 0)
    & (senders < n_particles)
    & (receivers >= 0)
    & (receivers < n_particles)
  )
  senders = jnp.where(valid_e, senders, jnp.zeros_like(senders))
  receivers = jnp.where(valid_e, receivers, senders)
  return senders, receivers, valid_e


def filter_zero_shift_self_edges(senders, receivers, unit_shifts, valid_e):
  zero_shift = jnp.all(unit_shifts == 0, axis=-1)
  return valid_e & ~((senders == receivers) & zero_shift)


def node_fields(
  R_cart,
  z_atomic,
  z_to_index,
  n_template,
  c_node,
  pos_dtype,
  node_attrs_dtype,
  node_idx_dtype,
):
  n_particles = R_cart.shape[0]
  positions = jnp.zeros((n_template, 3), dtype=pos_dtype)
  positions = positions.at[:n_particles].set(R_cart)

  species_idx = z_to_index[z_atomic[:n_particles]]
  node_attrs = jnp.zeros((n_template, c_node), dtype=node_attrs_dtype)
  node_attrs = node_attrs.at[:n_particles].set(
    jax.nn.one_hot(
      species_idx,
      num_classes=c_node,
      dtype=node_attrs_dtype,
    )
  )

  node_attrs_index = jnp.zeros((n_template,), dtype=node_idx_dtype)
  node_attrs_index = node_attrs_index.at[:n_particles].set(
    species_idx.astype(node_idx_dtype)
  )
  return positions, node_attrs, node_attrs_index


def edge_image_data(
  R_input,
  R_cart,
  senders,
  receivers,
  valid_e,
  cell,
  displacement_fn,
  shift_dtype,
  unit_shift_dtype,
  far_shift,
):
  Ri_in = R_input[senders]
  Rj_in = R_input[receivers]
  dR_min_in = jax.vmap(displacement_fn)(Ri_in, Rj_in)

  dR_model = -dR_min_in
  delta_cart = R_cart[receivers] - R_cart[senders]
  exact_shifts = (dR_model - delta_cart).astype(shift_dtype)

  inv_cell = space.inverse(cell)
  unit_shifts_int = jnp.rint(space.transform(inv_cell, exact_shifts)).astype(
    jnp.int32
  )
  unit_shifts_int = lax.stop_gradient(unit_shifts_int)

  unit_shifts = lax.stop_gradient(unit_shifts_int.astype(unit_shift_dtype))
  shifts = lax.stop_gradient(exact_shifts)

  edge_count = senders.shape[0]
  shifts = jnp.where(valid_e[:, None], shifts, far_shift[None, :])
  unit_shifts = jnp.where(
    valid_e[:, None],
    unit_shifts,
    jnp.zeros((edge_count, 3), dtype=unit_shift_dtype),
  )
  return shifts, unit_shifts, unit_shifts_int


def pad_edges(
  senders,
  receivers,
  shifts,
  unit_shifts,
  e_template,
  edge_dtype,
  shift_dtype,
  unit_shift_dtype,
  far_shift,
):
  edge_count = int(senders.shape[0])
  if edge_count > e_template:
    raise ValueError(
      f'Neighbor graph has {edge_count} edges, but template only supports '
      f'{e_template}. Reconvert model with a larger e_template.'
    )
  pad_n = e_template - edge_count

  send_pad = jnp.zeros((pad_n,), dtype=jnp.int32)
  recv_pad = jnp.zeros((pad_n,), dtype=jnp.int32)
  us_pad = jnp.zeros((pad_n, 3), dtype=unit_shift_dtype)
  sh_pad = jnp.tile(far_shift[None, :], (pad_n, 1))

  send2 = jnp.concatenate([senders, send_pad], axis=0)
  recv2 = jnp.concatenate([receivers, recv_pad], axis=0)
  edge_index = jnp.stack([send2, recv2], axis=0).astype(edge_dtype)
  unit_shifts = jnp.concatenate([unit_shifts, us_pad], axis=0)
  shifts = jnp.concatenate([shifts, sh_pad], axis=0)
  return (
    edge_index,
    shifts.astype(shift_dtype),
    unit_shifts.astype(unit_shift_dtype),
  )


def assemble_batch(
  *,
  positions,
  node_attrs,
  node_attrs_index,
  edge_index,
  shifts,
  unit_shifts,
  template_batch,
  cell,
  head_value,
):
  out = {
    'positions': positions,
    'node_attrs': node_attrs,
    'node_attrs_index': node_attrs_index,
    'edge_index': edge_index,
    'shifts': shifts,
    'unit_shifts': unit_shifts,
    'batch': template_batch['batch'],
    'ptr': template_batch['ptr'],
    'cell': cell,
  }
  if head_value is not None:
    out['head'] = head_value
  return out


def mace_neighbor_list(
  *,
  jax_model,
  template_batch: dict,
  config: dict,
  box,
  z_atomic,  # (N_real,) atomic numbers for real atoms only
  r_cutoff: float,
  dr_threshold: float,
  k_neighbors: int = 64,
  capacity_multiplier: float = 2.0,
  include_head: bool = True,
  fractional_coordinates: bool = False,
  format=partition.Dense,
  neighbor_list_fn=partition.neighbor_list,
  **neighbor_kwargs,
):
  """Wraps a MACE-JAX potential as a JAX MD neighbor-list energy.

  Args:
    jax_model: Callable MACE-JAX model accepting a MACE batch dictionary.
    template_batch: Fixed-shape template batch produced during model conversion.
    config: Model configuration containing at least ``atomic_numbers``.
    box: Periodic box with shape ``(3,)`` or ``(3, 3)``.
    z_atomic: Atomic numbers for real atoms.
    r_cutoff: Neighbor cutoff radius.
    dr_threshold: Neighbor-list rebuild threshold.
    k_neighbors: Expected neighbor capacity per template atom.
    capacity_multiplier: Neighbor-list capacity multiplier.
    include_head: Whether to pass ``head`` from the template batch.
    fractional_coordinates: Whether positions are fractional coordinates.
    format: JAX MD neighbor-list format.
    neighbor_list_fn: Neighbor-list factory.
    **neighbor_kwargs: Additional keyword arguments for ``neighbor_list_fn``.

  Returns:
    A pair ``(neighbor_fn, energy_fn)`` where ``energy_fn`` has signature
    ``energy_fn(R, *, box=None, neighbor=None, neighbors=None,
    neighbor_idx=None, perturbation=None)`` and returns a scalar energy.

  Notes:
    If ``fractional_coordinates`` is ``False``, ``R`` is interpreted as
    Cartesian coordinates. If ``fractional_coordinates`` is ``True``, ``R`` is
    interpreted as fractional coordinates in the unit cell. The MACE model
    always receives Cartesian positions and shifts internally.
  """

  template_batch = dict(template_batch)
  if 'unit_shifts' not in template_batch:
    template_batch['unit_shifts'] = jnp.zeros_like(template_batch['shifts'])

  box_default = jnp.asarray(box)

  # Space / shift / neighbor list setup
  if fractional_coordinates:
    displacement_fn_fixed, _ = space.periodic_general(
      box_default,
      fractional_coordinates=True,
    )
  else:
    if box_default.shape == (3,):
      displacement_fn_fixed, _ = space.periodic(box_default)
    elif box_default.shape == (3, 3):
      displacement_fn_fixed, _ = space.periodic_general(
        box_default,
        fractional_coordinates=False,
      )
      if neighbor_list_fn is partition.neighbor_list:
        neighbor_kwargs.setdefault('disable_cell_list', True)
    else:
      raise ValueError(
        f'Unexpected box shape {box_default.shape} (expected (3,) or (3,3))'
      )

  neighbor_kwargs.setdefault('capacity_multiplier', capacity_multiplier)
  neighbor_kwargs.setdefault('fractional_coordinates', fractional_coordinates)
  neighbor_kwargs.setdefault('format', format)
  neighbor_fn = neighbor_list_fn(
    displacement_fn_fixed,
    box_default,
    r_cutoff,
    dr_threshold=dr_threshold,
    **neighbor_kwargs,
  )

  # Template sizes / dtypes expected by converted MACE model
  N_template = int(template_batch['positions'].shape[0])
  E_template = int(template_batch['edge_index'].shape[1])
  C_node = int(template_batch['node_attrs'].shape[1])

  if E_template < N_template * k_neighbors:
    raise ValueError(
      f'E_template ({E_template}) must be >= N_template*k_neighbors '
      f'({N_template * k_neighbors}). Reconvert model with larger e_template '
      f'or reduce k_neighbors.'
    )

  pos_dtype = template_batch['positions'].dtype
  edge_dtype = template_batch['edge_index'].dtype
  shift_dtype = template_batch['shifts'].dtype
  us_dtype = template_batch['unit_shifts'].dtype
  node_attrs_dtype = template_batch['node_attrs'].dtype
  node_idx_dtype = template_batch['node_attrs_index'].dtype
  cell_dtype = template_batch['cell'].dtype

  far = jnp.asarray(r_cutoff + 2.0 * dr_threshold + 1.0, dtype=shift_dtype)
  FAR_SHIFT = jnp.array([far, 0.0, 0.0], dtype=shift_dtype)

  # Z -> species index lookup
  atomic_numbers = tuple(int(x) for x in config['atomic_numbers'])
  max_Z = int(max(atomic_numbers))
  Z_to_index = -jnp.ones((max_Z + 1,), dtype=jnp.int32)
  Z_to_index = Z_to_index.at[jnp.array(atomic_numbers, dtype=jnp.int32)].set(
    jnp.arange(len(atomic_numbers), dtype=jnp.int32)
  )
  z_atomic = validate_atomic_numbers(z_atomic, atomic_numbers, N_template)

  head_value = None
  if include_head and ('head' in template_batch):
    head_value = template_batch['head'].astype(jnp.int32).reshape(-1)

  def runtime_displacement_fn(box_now):
    box_arr = jnp.asarray(box_now)
    if fractional_coordinates:
      return space.periodic_general(
        box_arr,
        fractional_coordinates=True,
      )[0]
    else:
      if box_arr.shape == (3,):
        return space.periodic(box_arr)[0]
      elif box_arr.shape == (3, 3):
        return space.periodic_general(
          box_arr,
          fractional_coordinates=False,
        )[0]
      else:
        raise ValueError(f'Unexpected runtime box shape: {box_arr.shape}')

  def batch_from_edges(
    R_input,
    senders,
    receivers,
    valid_e,
    shifts,
    unit_shifts,
    cell_1x3x3,
    perturbation,
  ):
    cell = cell_1x3x3[0]
    R_cart = to_cartesian(R_input, cell, fractional_coordinates).astype(
      pos_dtype
    )

    if perturbation is not None:
      cell = perturb_cell(cell, perturbation).astype(cell_dtype)
      cell_1x3x3 = cell[None, :, :]
      R_cart = space.raw_transform(perturbation, R_cart).astype(pos_dtype)
      shifts = space.raw_transform(perturbation, shifts).astype(shift_dtype)

    positions, node_attrs, node_attrs_index = node_fields(
      R_cart,
      z_atomic,
      Z_to_index,
      N_template,
      C_node,
      pos_dtype,
      node_attrs_dtype,
      node_idx_dtype,
    )
    shifts = jnp.where(valid_e[:, None], shifts, FAR_SHIFT[None, :])
    unit_shifts = jnp.where(
      valid_e[:, None],
      unit_shifts,
      jnp.zeros_like(unit_shifts, dtype=us_dtype),
    )
    edge_index, shifts, unit_shifts = pad_edges(
      senders,
      receivers,
      shifts,
      unit_shifts,
      E_template,
      edge_dtype,
      shift_dtype,
      us_dtype,
      FAR_SHIFT,
    )
    return assemble_batch(
      positions=positions,
      node_attrs=node_attrs,
      node_attrs_index=node_attrs_index,
      edge_index=edge_index,
      shifts=shifts,
      unit_shifts=unit_shifts,
      template_batch=template_batch,
      cell=cell_1x3x3,
      head_value=head_value,
    )

  def batch_from_standard_edges(
    R_input,
    senders,
    receivers,
    valid_e,
    cell_1x3x3,
    box_now,
    perturbation,
  ):
    cell = cell_1x3x3[0]
    disp_now = runtime_displacement_fn(box_now)
    R_cart = to_cartesian(R_input, cell, fractional_coordinates).astype(
      pos_dtype
    )
    shifts, unit_shifts, _ = edge_image_data(
      R_input,
      R_cart,
      senders,
      receivers,
      valid_e,
      cell,
      disp_now,
      shift_dtype,
      us_dtype,
      FAR_SHIFT,
    )
    return batch_from_edges(
      R_input,
      senders,
      receivers,
      valid_e,
      shifts,
      unit_shifts,
      cell_1x3x3,
      perturbation,
    )

  @jax.jit
  def make_batch_from_neighbor_idx(
    R_real, neighbor_idx, box_now, perturbation=None
  ):
    R_input = jnp.asarray(R_real, dtype=pos_dtype)
    neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)

    cell_1x3x3 = make_cell_1x3x3(box_now, cell_dtype)

    N = R_input.shape[0]
    if N > N_template:
      raise ValueError(
        f'R has {N} atoms, but template only supports {N_template}.'
      )

    senders, receivers, valid_e = dense_neighbor_topology(neighbor_idx, N)
    return batch_from_standard_edges(
      R_input,
      senders,
      receivers,
      valid_e,
      cell_1x3x3,
      box_now,
      perturbation,
    )

  @jax.jit
  def make_batch_from_neighbor(
    R_real, neighbor_data, box_now, perturbation=None
  ):
    R_input = jnp.asarray(R_real, dtype=pos_dtype)
    cell_1x3x3 = make_cell_1x3x3(box_now, cell_dtype)
    cell = cell_1x3x3[0]

    N = R_input.shape[0]
    if N > N_template:
      raise ValueError(
        f'R has {N} atoms, but template only supports {N_template}.'
      )

    if hasattr(neighbor_data, 'shifts'):
      if partition.is_sparse(neighbor_data.format):
        senders, receivers, valid_e = sparse_multi_image_topology(
          neighbor_data.idx, N
        )
        unit_shifts = lax.stop_gradient(
          jnp.asarray(neighbor_data.shifts, dtype=us_dtype)
        )
        valid_e = filter_zero_shift_self_edges(
          senders, receivers, unit_shifts, valid_e
        )
      else:
        senders, receivers, valid_e = dense_neighbor_topology_with_shifts(
          neighbor_data.idx, N
        )
        unit_shifts = lax.stop_gradient(
          jnp.asarray(neighbor_data.shifts, dtype=us_dtype).reshape((-1, 3))
        )
        valid_e = filter_zero_shift_self_edges(
          senders, receivers, unit_shifts, valid_e
        )
      shifts = space.transform(cell, unit_shifts).astype(shift_dtype)
      return batch_from_edges(
        R_input,
        senders,
        receivers,
        valid_e,
        shifts,
        unit_shifts,
        cell_1x3x3,
        perturbation,
      )

    if partition.is_sparse(neighbor_data.format):
      senders, receivers, valid_e = sparse_neighbor_topology(
        neighbor_data.idx, N
      )
    else:
      senders, receivers, valid_e = dense_neighbor_topology(
        neighbor_data.idx, N
      )
    return batch_from_standard_edges(
      R_input,
      senders,
      receivers,
      valid_e,
      cell_1x3x3,
      box_now,
      perturbation,
    )

  @jax.jit
  def energy_fn(
    R,
    *,
    box=None,
    neighbor=None,
    neighbors=None,
    neighbor_idx=None,
    perturbation=None,
  ):
    neighbor_data = neighbor if neighbor is not None else neighbors
    if box is not None:
      box_now = box
    elif neighbor_data is not None and hasattr(neighbor_data, 'box'):
      box_now = neighbor_data.box
    else:
      box_now = box_default

    if neighbor_idx is not None:
      batch = make_batch_from_neighbor_idx(
        R, neighbor_idx, box_now, perturbation
      )
    else:
      if neighbor_data is None:
        raise ValueError(
          'Provide either neighbor=..., neighbors=..., or neighbor_idx=...'
        )
      batch = make_batch_from_neighbor(R, neighbor_data, box_now, perturbation)

    out = jax_model(batch)
    e = out['energy'] if isinstance(out, dict) and 'energy' in out else out
    return as_scalar(e)

  return neighbor_fn, energy_fn
