"""Pallas implementations for UMA Wigner edge kernels."""

from functools import partial

import jax
import jax.numpy as jnp

from jax._src import core
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir

from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


COEFFICIENT_DIM = 9
CHANNEL_BLOCK_SIZE = 128
MAX_GRID_EDGE_BLOCKS = 65535


def ceil_div(x: int, y: int) -> int:
  return (x + y - 1) // y


def next_power_of_two(x: int) -> int:
  return 1 << (x - 1).bit_length()


def compiler_params(
  block_c: int, *, num_stages: int = 1, num_warps: int | None = None
):
  """Compute Triton compiler parameters for a given channel block size.

  Warp count heuristic: each warp handles 32 threads. For the 9-coefficient
  Wigner kernels, the arithmetic intensity is moderate (scalar Wigner loads
  multiplied against CHANNEL_BLOCK_SIZE-wide vectors). We scale warps to match the
  vector width so that memory latency is hidden without over-subscribing
  register files.

  Args:
    block_c: Channel block size processed per program instance.
    num_stages: Software pipeline stages for memory latency hiding. Higher
      values (2-3) can help when memory-bound, at the cost of register
      pressure.
    num_warps: Override warp count. If None, uses the block_c heuristic.
  """
  if num_warps is None:
    if block_c <= 64:
      num_warps = 1
    elif block_c <= 128:
      num_warps = 2
    elif block_c <= 256:
      num_warps = 4
    else:
      num_warps = 8
  return plt.CompilerParams(num_warps=num_warps, num_stages=num_stages)


def check_coefficients(*arrays: jnp.ndarray) -> None:
  for array in arrays:
    if array.shape[-2] != COEFFICIENT_DIM:
      raise ValueError('UMA Pallas kernels currently require 9 coefficients.')


def check_wigner_matrix(*arrays: jnp.ndarray) -> None:
  for array in arrays:
    if array.shape[-2:] != (COEFFICIENT_DIM, COEFFICIENT_DIM):
      raise ValueError(
        'UMA Pallas kernels require raw [E, 9, 9] Wigner matrices.'
      )


def node_to_edge_kernel(
  x_ref,
  edge_index_ref,
  wigner_ref,
  out_ref,
  *,
  block_c: int,
):
  edge = pl.program_id(0)
  c_offsets = pl.program_id(1) * block_c + jnp.arange(0, block_c)
  channels = x_ref.shape[2]
  c_mask = c_offsets < channels

  sender = plt.load(edge_index_ref.at[0, edge])
  receiver = plt.load(edge_index_ref.at[1, edge])
  sender_valid = (sender >= 0) & (sender < x_ref.shape[0])
  receiver_valid = (receiver >= 0) & (receiver < x_ref.shape[0])
  sender = jnp.clip(sender, 0, x_ref.shape[0] - 1)
  receiver = jnp.clip(receiver, 0, x_ref.shape[0] - 1)

  x0s = plt.load(
    x_ref.at[sender, 0, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x0t = plt.load(
    x_ref.at[receiver, 0, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x1s = plt.load(
    x_ref.at[sender, 1, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x1t = plt.load(
    x_ref.at[receiver, 1, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x2s = plt.load(
    x_ref.at[sender, 2, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x2t = plt.load(
    x_ref.at[receiver, 2, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x3s = plt.load(
    x_ref.at[sender, 3, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x3t = plt.load(
    x_ref.at[receiver, 3, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x4s = plt.load(
    x_ref.at[sender, 4, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x4t = plt.load(
    x_ref.at[receiver, 4, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x5s = plt.load(
    x_ref.at[sender, 5, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x5t = plt.load(
    x_ref.at[receiver, 5, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x6s = plt.load(
    x_ref.at[sender, 6, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x6t = plt.load(
    x_ref.at[receiver, 6, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x7s = plt.load(
    x_ref.at[sender, 7, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x7t = plt.load(
    x_ref.at[receiver, 7, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x8s = plt.load(
    x_ref.at[sender, 8, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x8t = plt.load(
    x_ref.at[receiver, 8, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )

  w00 = plt.load(wigner_ref.at[edge, 0, 0])
  y0s = w00 * x0s
  y0t = w00 * x0t

  w11 = plt.load(wigner_ref.at[edge, 1, 1])
  w12 = plt.load(wigner_ref.at[edge, 1, 2])
  w13 = plt.load(wigner_ref.at[edge, 1, 3])
  w21 = plt.load(wigner_ref.at[edge, 2, 1])
  w22 = plt.load(wigner_ref.at[edge, 2, 2])
  w23 = plt.load(wigner_ref.at[edge, 2, 3])
  w31 = plt.load(wigner_ref.at[edge, 3, 1])
  w32 = plt.load(wigner_ref.at[edge, 3, 2])
  w33 = plt.load(wigner_ref.at[edge, 3, 3])
  y1s = w11 * x1s + w12 * x2s + w13 * x3s
  y2s = w21 * x1s + w22 * x2s + w23 * x3s
  y3s = w31 * x1s + w32 * x2s + w33 * x3s
  y1t = w11 * x1t + w12 * x2t + w13 * x3t
  y2t = w21 * x1t + w22 * x2t + w23 * x3t
  y3t = w31 * x1t + w32 * x2t + w33 * x3t

  w44 = plt.load(wigner_ref.at[edge, 4, 4])
  w45 = plt.load(wigner_ref.at[edge, 4, 5])
  w46 = plt.load(wigner_ref.at[edge, 4, 6])
  w47 = plt.load(wigner_ref.at[edge, 4, 7])
  w48 = plt.load(wigner_ref.at[edge, 4, 8])
  w54 = plt.load(wigner_ref.at[edge, 5, 4])
  w55 = plt.load(wigner_ref.at[edge, 5, 5])
  w56 = plt.load(wigner_ref.at[edge, 5, 6])
  w57 = plt.load(wigner_ref.at[edge, 5, 7])
  w58 = plt.load(wigner_ref.at[edge, 5, 8])
  w64 = plt.load(wigner_ref.at[edge, 6, 4])
  w65 = plt.load(wigner_ref.at[edge, 6, 5])
  w66 = plt.load(wigner_ref.at[edge, 6, 6])
  w67 = plt.load(wigner_ref.at[edge, 6, 7])
  w68 = plt.load(wigner_ref.at[edge, 6, 8])
  w74 = plt.load(wigner_ref.at[edge, 7, 4])
  w75 = plt.load(wigner_ref.at[edge, 7, 5])
  w76 = plt.load(wigner_ref.at[edge, 7, 6])
  w77 = plt.load(wigner_ref.at[edge, 7, 7])
  w78 = plt.load(wigner_ref.at[edge, 7, 8])
  w84 = plt.load(wigner_ref.at[edge, 8, 4])
  w85 = plt.load(wigner_ref.at[edge, 8, 5])
  w86 = plt.load(wigner_ref.at[edge, 8, 6])
  w87 = plt.load(wigner_ref.at[edge, 8, 7])
  w88 = plt.load(wigner_ref.at[edge, 8, 8])
  y4s = w44 * x4s + w45 * x5s + w46 * x6s + w47 * x7s + w48 * x8s
  y5s = w54 * x4s + w55 * x5s + w56 * x6s + w57 * x7s + w58 * x8s
  y6s = w64 * x4s + w65 * x5s + w66 * x6s + w67 * x7s + w68 * x8s
  y7s = w74 * x4s + w75 * x5s + w76 * x6s + w77 * x7s + w78 * x8s
  y8s = w84 * x4s + w85 * x5s + w86 * x6s + w87 * x7s + w88 * x8s
  y4t = w44 * x4t + w45 * x5t + w46 * x6t + w47 * x7t + w48 * x8t
  y5t = w54 * x4t + w55 * x5t + w56 * x6t + w57 * x7t + w58 * x8t
  y6t = w64 * x4t + w65 * x5t + w66 * x6t + w67 * x7t + w68 * x8t
  y7t = w74 * x4t + w75 * x5t + w76 * x6t + w77 * x7t + w78 * x8t
  y8t = w84 * x4t + w85 * x5t + w86 * x6t + w87 * x7t + w88 * x8t

  plt.store(out_ref.at[edge, 0, c_offsets], y0s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 0, channels + c_offsets],
    y0t,
    mask=c_mask,
  )
  plt.store(out_ref.at[edge, 5, c_offsets], y1s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 5, channels + c_offsets],
    y1t,
    mask=c_mask,
  )
  plt.store(out_ref.at[edge, 1, c_offsets], y2s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 1, channels + c_offsets],
    y2t,
    mask=c_mask,
  )
  plt.store(out_ref.at[edge, 3, c_offsets], y3s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 3, channels + c_offsets],
    y3t,
    mask=c_mask,
  )
  plt.store(out_ref.at[edge, 8, c_offsets], y4s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 8, channels + c_offsets],
    y4t,
    mask=c_mask,
  )
  plt.store(out_ref.at[edge, 6, c_offsets], y5s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 6, channels + c_offsets],
    y5t,
    mask=c_mask,
  )
  plt.store(out_ref.at[edge, 2, c_offsets], y6s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 2, channels + c_offsets],
    y6t,
    mask=c_mask,
  )
  plt.store(out_ref.at[edge, 4, c_offsets], y7s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 4, channels + c_offsets],
    y7t,
    mask=c_mask,
  )
  plt.store(out_ref.at[edge, 7, c_offsets], y8s, mask=c_mask)
  plt.store(
    out_ref.at[edge, 7, channels + c_offsets],
    y8t,
    mask=c_mask,
  )


def _node_to_edge_impl(
  x: jnp.ndarray,
  edge_index: jnp.ndarray,
  wigner_and_m_mapping: jnp.ndarray,
) -> jnp.ndarray:
  check_coefficients(x)
  check_wigner_matrix(wigner_and_m_mapping)
  num_edges = edge_index.shape[1]
  channels = x.shape[2]
  if num_edges == 0:
    return jnp.zeros((0, COEFFICIENT_DIM, 2 * channels), dtype=x.dtype)
  out_shape = jax.ShapeDtypeStruct(
    (num_edges, COEFFICIENT_DIM, 2 * channels), dtype=x.dtype
  )
  kernel = partial(node_to_edge_kernel, block_c=CHANNEL_BLOCK_SIZE)
  return pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid=(num_edges, ceil_div(channels, CHANNEL_BLOCK_SIZE)),
    compiler_params=compiler_params(CHANNEL_BLOCK_SIZE, num_stages=2),
  )(x, edge_index, wigner_and_m_mapping)


def gather_x_edge(x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
  sender = edge_index[0]
  receiver = edge_index[1]
  sender_valid = (sender >= 0) & (sender < x.shape[0])
  receiver_valid = (receiver >= 0) & (receiver < x.shape[0])
  sender = jnp.clip(sender, 0, x.shape[0] - 1)
  receiver = jnp.clip(receiver, 0, x.shape[0] - 1)
  x_source = jnp.where(sender_valid[:, None, None], x[sender], 0.0)
  x_target = jnp.where(receiver_valid[:, None, None], x[receiver], 0.0)
  return jnp.concatenate([x_source, x_target], axis=2)


def node_to_edge_bwd_dx_scatter_kernel(
  gout_ref,
  edge_index_ref,
  wigner_ref,
  init_ref,
  out_ref,
  *,
  block_c: int,
):
  """Fused: transpose-Wigner on gout + atomic scatter-add to dx."""
  del init_ref
  edge = pl.program_id(0)
  c_offsets = pl.program_id(1) * block_c + jnp.arange(0, block_c)
  channels = gout_ref.shape[2] // 2
  c_mask = c_offsets < channels

  num_nodes = out_ref.shape[0]
  sender = plt.load(edge_index_ref.at[0, edge])
  receiver = plt.load(edge_index_ref.at[1, edge])
  sender_valid = (sender >= 0) & (sender < num_nodes)
  receiver_valid = (receiver >= 0) & (receiver < num_nodes)
  sender = jnp.clip(sender, 0, num_nodes - 1)
  receiver = jnp.clip(receiver, 0, num_nodes - 1)

  dy0s = plt.load(gout_ref.at[edge, 0, c_offsets], mask=c_mask, other=0.0)
  dy1s = plt.load(gout_ref.at[edge, 5, c_offsets], mask=c_mask, other=0.0)
  dy2s = plt.load(gout_ref.at[edge, 1, c_offsets], mask=c_mask, other=0.0)
  dy3s = plt.load(gout_ref.at[edge, 3, c_offsets], mask=c_mask, other=0.0)
  dy4s = plt.load(gout_ref.at[edge, 8, c_offsets], mask=c_mask, other=0.0)
  dy5s = plt.load(gout_ref.at[edge, 6, c_offsets], mask=c_mask, other=0.0)
  dy6s = plt.load(gout_ref.at[edge, 2, c_offsets], mask=c_mask, other=0.0)
  dy7s = plt.load(gout_ref.at[edge, 4, c_offsets], mask=c_mask, other=0.0)
  dy8s = plt.load(gout_ref.at[edge, 7, c_offsets], mask=c_mask, other=0.0)
  dy0t = plt.load(
    gout_ref.at[edge, 0, channels + c_offsets], mask=c_mask, other=0.0
  )
  dy1t = plt.load(
    gout_ref.at[edge, 5, channels + c_offsets], mask=c_mask, other=0.0
  )
  dy2t = plt.load(
    gout_ref.at[edge, 1, channels + c_offsets], mask=c_mask, other=0.0
  )
  dy3t = plt.load(
    gout_ref.at[edge, 3, channels + c_offsets], mask=c_mask, other=0.0
  )
  dy4t = plt.load(
    gout_ref.at[edge, 8, channels + c_offsets], mask=c_mask, other=0.0
  )
  dy5t = plt.load(
    gout_ref.at[edge, 6, channels + c_offsets], mask=c_mask, other=0.0
  )
  dy6t = plt.load(
    gout_ref.at[edge, 2, channels + c_offsets], mask=c_mask, other=0.0
  )
  dy7t = plt.load(
    gout_ref.at[edge, 4, channels + c_offsets], mask=c_mask, other=0.0
  )
  dy8t = plt.load(
    gout_ref.at[edge, 7, channels + c_offsets], mask=c_mask, other=0.0
  )

  w00 = plt.load(wigner_ref.at[edge, 0, 0])
  w11 = plt.load(wigner_ref.at[edge, 1, 1])
  w12 = plt.load(wigner_ref.at[edge, 1, 2])
  w13 = plt.load(wigner_ref.at[edge, 1, 3])
  w21 = plt.load(wigner_ref.at[edge, 2, 1])
  w22 = plt.load(wigner_ref.at[edge, 2, 2])
  w23 = plt.load(wigner_ref.at[edge, 2, 3])
  w31 = plt.load(wigner_ref.at[edge, 3, 1])
  w32 = plt.load(wigner_ref.at[edge, 3, 2])
  w33 = plt.load(wigner_ref.at[edge, 3, 3])
  w44 = plt.load(wigner_ref.at[edge, 4, 4])
  w45 = plt.load(wigner_ref.at[edge, 4, 5])
  w46 = plt.load(wigner_ref.at[edge, 4, 6])
  w47 = plt.load(wigner_ref.at[edge, 4, 7])
  w48 = plt.load(wigner_ref.at[edge, 4, 8])
  w54 = plt.load(wigner_ref.at[edge, 5, 4])
  w55 = plt.load(wigner_ref.at[edge, 5, 5])
  w56 = plt.load(wigner_ref.at[edge, 5, 6])
  w57 = plt.load(wigner_ref.at[edge, 5, 7])
  w58 = plt.load(wigner_ref.at[edge, 5, 8])
  w64 = plt.load(wigner_ref.at[edge, 6, 4])
  w65 = plt.load(wigner_ref.at[edge, 6, 5])
  w66 = plt.load(wigner_ref.at[edge, 6, 6])
  w67 = plt.load(wigner_ref.at[edge, 6, 7])
  w68 = plt.load(wigner_ref.at[edge, 6, 8])
  w74 = plt.load(wigner_ref.at[edge, 7, 4])
  w75 = plt.load(wigner_ref.at[edge, 7, 5])
  w76 = plt.load(wigner_ref.at[edge, 7, 6])
  w77 = plt.load(wigner_ref.at[edge, 7, 7])
  w78 = plt.load(wigner_ref.at[edge, 7, 8])
  w84 = plt.load(wigner_ref.at[edge, 8, 4])
  w85 = plt.load(wigner_ref.at[edge, 8, 5])
  w86 = plt.load(wigner_ref.at[edge, 8, 6])
  w87 = plt.load(wigner_ref.at[edge, 8, 7])
  w88 = plt.load(wigner_ref.at[edge, 8, 8])

  dx_vals = [
    (w00 * dy0s, w00 * dy0t),
    (
      w11 * dy1s + w21 * dy2s + w31 * dy3s,
      w11 * dy1t + w21 * dy2t + w31 * dy3t,
    ),
    (
      w12 * dy1s + w22 * dy2s + w32 * dy3s,
      w12 * dy1t + w22 * dy2t + w32 * dy3t,
    ),
    (
      w13 * dy1s + w23 * dy2s + w33 * dy3s,
      w13 * dy1t + w23 * dy2t + w33 * dy3t,
    ),
    (
      w44 * dy4s + w54 * dy5s + w64 * dy6s + w74 * dy7s + w84 * dy8s,
      w44 * dy4t + w54 * dy5t + w64 * dy6t + w74 * dy7t + w84 * dy8t,
    ),
    (
      w45 * dy4s + w55 * dy5s + w65 * dy6s + w75 * dy7s + w85 * dy8s,
      w45 * dy4t + w55 * dy5t + w65 * dy6t + w75 * dy7t + w85 * dy8t,
    ),
    (
      w46 * dy4s + w56 * dy5s + w66 * dy6s + w76 * dy7s + w86 * dy8s,
      w46 * dy4t + w56 * dy5t + w66 * dy6t + w76 * dy7t + w86 * dy8t,
    ),
    (
      w47 * dy4s + w57 * dy5s + w67 * dy6s + w77 * dy7s + w87 * dy8s,
      w47 * dy4t + w57 * dy5t + w67 * dy6t + w77 * dy7t + w87 * dy8t,
    ),
    (
      w48 * dy4s + w58 * dy5s + w68 * dy6s + w78 * dy7s + w88 * dy8s,
      w48 * dy4t + w58 * dy5t + w68 * dy6t + w78 * dy7t + w88 * dy8t,
    ),
  ]

  for j, (dxs, dxt) in enumerate(dx_vals):
    plt.atomic_add(
      out_ref, (sender, j, c_offsets), dxs, mask=c_mask & sender_valid
    )
    plt.atomic_add(
      out_ref, (receiver, j, c_offsets), dxt, mask=c_mask & receiver_valid
    )


def _node_to_edge_bwd_dx_scatter(
  gout: jnp.ndarray,
  edge_index: jnp.ndarray,
  wigner_and_m_mapping: jnp.ndarray,
  x_shape: tuple[int, int, int],
) -> jnp.ndarray:
  channels = x_shape[2]
  block_c = CHANNEL_BLOCK_SIZE
  init = jnp.zeros(x_shape, dtype=gout.dtype)
  if gout.shape[0] == 0:
    return init
  kernel = partial(node_to_edge_bwd_dx_scatter_kernel, block_c=block_c)
  return pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(init.shape, init.dtype),
    grid=(gout.shape[0], ceil_div(channels, block_c)),
    input_output_aliases={3: 0},
    compiler_params=compiler_params(block_c),
  )(gout, edge_index, wigner_and_m_mapping, init)


def node_to_edge_bwd_dwigner_gather_kernel(
  x_ref,
  edge_index_ref,
  gout_ref,
  init_ref,
  out_ref,
  *,
  block_c: int,
  grid_x: int = MAX_GRID_EDGE_BLOCKS,
  num_edges: int = 0,
):
  """Compute dW by gathering x[sender]/x[receiver] directly, no x_edge."""
  del init_ref
  edge = pl.program_id(0) + pl.program_id(1) * grid_x
  safe_edge = jnp.minimum(edge, num_edges - 1)
  channels = x_ref.shape[2]
  c_offsets = jnp.arange(0, block_c)
  c_mask = c_offsets < channels

  sender = plt.load(edge_index_ref.at[0, safe_edge])
  receiver = plt.load(edge_index_ref.at[1, safe_edge])
  sender_valid = (sender >= 0) & (sender < x_ref.shape[0])
  receiver_valid = (receiver >= 0) & (receiver < x_ref.shape[0])
  sender = jnp.clip(sender, 0, x_ref.shape[0] - 1)
  receiver = jnp.clip(receiver, 0, x_ref.shape[0] - 1)

  dy0s = plt.load(gout_ref.at[safe_edge, 0, c_offsets], mask=c_mask, other=0.0)
  dy0t = plt.load(
    gout_ref.at[safe_edge, 0, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  dy1s = plt.load(gout_ref.at[safe_edge, 5, c_offsets], mask=c_mask, other=0.0)
  dy1t = plt.load(
    gout_ref.at[safe_edge, 5, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  dy2s = plt.load(gout_ref.at[safe_edge, 1, c_offsets], mask=c_mask, other=0.0)
  dy2t = plt.load(
    gout_ref.at[safe_edge, 1, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  dy3s = plt.load(gout_ref.at[safe_edge, 3, c_offsets], mask=c_mask, other=0.0)
  dy3t = plt.load(
    gout_ref.at[safe_edge, 3, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  dy4s = plt.load(gout_ref.at[safe_edge, 8, c_offsets], mask=c_mask, other=0.0)
  dy4t = plt.load(
    gout_ref.at[safe_edge, 8, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  dy5s = plt.load(gout_ref.at[safe_edge, 6, c_offsets], mask=c_mask, other=0.0)
  dy5t = plt.load(
    gout_ref.at[safe_edge, 6, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  dy6s = plt.load(gout_ref.at[safe_edge, 2, c_offsets], mask=c_mask, other=0.0)
  dy6t = plt.load(
    gout_ref.at[safe_edge, 2, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  dy7s = plt.load(gout_ref.at[safe_edge, 4, c_offsets], mask=c_mask, other=0.0)
  dy7t = plt.load(
    gout_ref.at[safe_edge, 4, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  dy8s = plt.load(gout_ref.at[safe_edge, 7, c_offsets], mask=c_mask, other=0.0)
  dy8t = plt.load(
    gout_ref.at[safe_edge, 7, channels + c_offsets],
    mask=c_mask,
    other=0.0,
  )
  x0s = plt.load(
    x_ref.at[sender, 0, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x0t = plt.load(
    x_ref.at[receiver, 0, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x1s = plt.load(
    x_ref.at[sender, 1, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x1t = plt.load(
    x_ref.at[receiver, 1, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x2s = plt.load(
    x_ref.at[sender, 2, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x2t = plt.load(
    x_ref.at[receiver, 2, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x3s = plt.load(
    x_ref.at[sender, 3, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x3t = plt.load(
    x_ref.at[receiver, 3, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x4s = plt.load(
    x_ref.at[sender, 4, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x4t = plt.load(
    x_ref.at[receiver, 4, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x5s = plt.load(
    x_ref.at[sender, 5, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x5t = plt.load(
    x_ref.at[receiver, 5, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x6s = plt.load(
    x_ref.at[sender, 6, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x6t = plt.load(
    x_ref.at[receiver, 6, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x7s = plt.load(
    x_ref.at[sender, 7, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x7t = plt.load(
    x_ref.at[receiver, 7, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x8s = plt.load(
    x_ref.at[sender, 8, c_offsets],
    mask=c_mask & sender_valid,
    other=0.0,
  )
  x8t = plt.load(
    x_ref.at[receiver, 8, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )

  plt.store(out_ref.at[edge, 0, 0], jnp.sum(dy0s * x0s + dy0t * x0t))
  plt.store(out_ref.at[edge, 1, 1], jnp.sum(dy1s * x1s + dy1t * x1t))
  plt.store(out_ref.at[edge, 1, 2], jnp.sum(dy1s * x2s + dy1t * x2t))
  plt.store(out_ref.at[edge, 1, 3], jnp.sum(dy1s * x3s + dy1t * x3t))
  plt.store(out_ref.at[edge, 2, 1], jnp.sum(dy2s * x1s + dy2t * x1t))
  plt.store(out_ref.at[edge, 2, 2], jnp.sum(dy2s * x2s + dy2t * x2t))
  plt.store(out_ref.at[edge, 2, 3], jnp.sum(dy2s * x3s + dy2t * x3t))
  plt.store(out_ref.at[edge, 3, 1], jnp.sum(dy3s * x1s + dy3t * x1t))
  plt.store(out_ref.at[edge, 3, 2], jnp.sum(dy3s * x2s + dy3t * x2t))
  plt.store(out_ref.at[edge, 3, 3], jnp.sum(dy3s * x3s + dy3t * x3t))
  plt.store(out_ref.at[edge, 4, 4], jnp.sum(dy4s * x4s + dy4t * x4t))
  plt.store(out_ref.at[edge, 4, 5], jnp.sum(dy4s * x5s + dy4t * x5t))
  plt.store(out_ref.at[edge, 4, 6], jnp.sum(dy4s * x6s + dy4t * x6t))
  plt.store(out_ref.at[edge, 4, 7], jnp.sum(dy4s * x7s + dy4t * x7t))
  plt.store(out_ref.at[edge, 4, 8], jnp.sum(dy4s * x8s + dy4t * x8t))
  plt.store(out_ref.at[edge, 5, 4], jnp.sum(dy5s * x4s + dy5t * x4t))
  plt.store(out_ref.at[edge, 5, 5], jnp.sum(dy5s * x5s + dy5t * x5t))
  plt.store(out_ref.at[edge, 5, 6], jnp.sum(dy5s * x6s + dy5t * x6t))
  plt.store(out_ref.at[edge, 5, 7], jnp.sum(dy5s * x7s + dy5t * x7t))
  plt.store(out_ref.at[edge, 5, 8], jnp.sum(dy5s * x8s + dy5t * x8t))
  plt.store(out_ref.at[edge, 6, 4], jnp.sum(dy6s * x4s + dy6t * x4t))
  plt.store(out_ref.at[edge, 6, 5], jnp.sum(dy6s * x5s + dy6t * x5t))
  plt.store(out_ref.at[edge, 6, 6], jnp.sum(dy6s * x6s + dy6t * x6t))
  plt.store(out_ref.at[edge, 6, 7], jnp.sum(dy6s * x7s + dy6t * x7t))
  plt.store(out_ref.at[edge, 6, 8], jnp.sum(dy6s * x8s + dy6t * x8t))
  plt.store(out_ref.at[edge, 7, 4], jnp.sum(dy7s * x4s + dy7t * x4t))
  plt.store(out_ref.at[edge, 7, 5], jnp.sum(dy7s * x5s + dy7t * x5t))
  plt.store(out_ref.at[edge, 7, 6], jnp.sum(dy7s * x6s + dy7t * x6t))
  plt.store(out_ref.at[edge, 7, 7], jnp.sum(dy7s * x7s + dy7t * x7t))
  plt.store(out_ref.at[edge, 7, 8], jnp.sum(dy7s * x8s + dy7t * x8t))
  plt.store(out_ref.at[edge, 8, 4], jnp.sum(dy8s * x4s + dy8t * x4t))
  plt.store(out_ref.at[edge, 8, 5], jnp.sum(dy8s * x5s + dy8t * x5t))
  plt.store(out_ref.at[edge, 8, 6], jnp.sum(dy8s * x6s + dy8t * x6t))
  plt.store(out_ref.at[edge, 8, 7], jnp.sum(dy8s * x7s + dy8t * x7t))
  plt.store(out_ref.at[edge, 8, 8], jnp.sum(dy8s * x8s + dy8t * x8t))


def _node_to_edge_bwd_dwigner_gather(
  x: jnp.ndarray,
  edge_index: jnp.ndarray,
  gout: jnp.ndarray,
) -> jnp.ndarray:
  num_edges = gout.shape[0]
  if num_edges == 0:
    return jnp.zeros((0, COEFFICIENT_DIM, COEFFICIENT_DIM), dtype=gout.dtype)
  block_c = next_power_of_two(x.shape[2])
  grid_x = min(num_edges, MAX_GRID_EDGE_BLOCKS)
  grid_y = ceil_div(num_edges, grid_x)
  padded = grid_x * grid_y
  kernel = partial(
    node_to_edge_bwd_dwigner_gather_kernel,
    block_c=block_c,
    grid_x=grid_x,
    num_edges=num_edges,
  )
  params = compiler_params(block_c)
  init = jnp.zeros((padded, COEFFICIENT_DIM, COEFFICIENT_DIM), dtype=gout.dtype)
  result = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(init.shape, init.dtype),
    grid=(grid_x, grid_y),
    input_output_aliases={3: 0},
    compiler_params=params,
  )(x, edge_index, gout, init)
  return result[:num_edges]


def inverse_wigner_kernel(
  messages_ref,
  wigner_inv_ref,
  out_ref,
  *,
  block_c: int,
):
  edge = pl.program_id(0)
  c_offsets = pl.program_id(1) * block_c + jnp.arange(0, block_c)
  c_mask = c_offsets < messages_ref.shape[2]

  x0 = plt.load(messages_ref.at[edge, 0, c_offsets], mask=c_mask, other=0.0)
  x1 = plt.load(messages_ref.at[edge, 5, c_offsets], mask=c_mask, other=0.0)
  x2 = plt.load(messages_ref.at[edge, 1, c_offsets], mask=c_mask, other=0.0)
  x3 = plt.load(messages_ref.at[edge, 3, c_offsets], mask=c_mask, other=0.0)
  x4 = plt.load(messages_ref.at[edge, 8, c_offsets], mask=c_mask, other=0.0)
  x5 = plt.load(messages_ref.at[edge, 6, c_offsets], mask=c_mask, other=0.0)
  x6 = plt.load(messages_ref.at[edge, 2, c_offsets], mask=c_mask, other=0.0)
  x7 = plt.load(messages_ref.at[edge, 4, c_offsets], mask=c_mask, other=0.0)
  x8 = plt.load(messages_ref.at[edge, 7, c_offsets], mask=c_mask, other=0.0)

  w00 = plt.load(wigner_inv_ref.at[edge, 0, 0])
  y0 = w00 * x0

  w11 = plt.load(wigner_inv_ref.at[edge, 1, 1])
  w12 = plt.load(wigner_inv_ref.at[edge, 1, 2])
  w13 = plt.load(wigner_inv_ref.at[edge, 1, 3])
  w21 = plt.load(wigner_inv_ref.at[edge, 2, 1])
  w22 = plt.load(wigner_inv_ref.at[edge, 2, 2])
  w23 = plt.load(wigner_inv_ref.at[edge, 2, 3])
  w31 = plt.load(wigner_inv_ref.at[edge, 3, 1])
  w32 = plt.load(wigner_inv_ref.at[edge, 3, 2])
  w33 = plt.load(wigner_inv_ref.at[edge, 3, 3])
  y1 = w11 * x1 + w12 * x2 + w13 * x3
  y2 = w21 * x1 + w22 * x2 + w23 * x3
  y3 = w31 * x1 + w32 * x2 + w33 * x3

  w44 = plt.load(wigner_inv_ref.at[edge, 4, 4])
  w45 = plt.load(wigner_inv_ref.at[edge, 4, 5])
  w46 = plt.load(wigner_inv_ref.at[edge, 4, 6])
  w47 = plt.load(wigner_inv_ref.at[edge, 4, 7])
  w48 = plt.load(wigner_inv_ref.at[edge, 4, 8])
  w54 = plt.load(wigner_inv_ref.at[edge, 5, 4])
  w55 = plt.load(wigner_inv_ref.at[edge, 5, 5])
  w56 = plt.load(wigner_inv_ref.at[edge, 5, 6])
  w57 = plt.load(wigner_inv_ref.at[edge, 5, 7])
  w58 = plt.load(wigner_inv_ref.at[edge, 5, 8])
  w64 = plt.load(wigner_inv_ref.at[edge, 6, 4])
  w65 = plt.load(wigner_inv_ref.at[edge, 6, 5])
  w66 = plt.load(wigner_inv_ref.at[edge, 6, 6])
  w67 = plt.load(wigner_inv_ref.at[edge, 6, 7])
  w68 = plt.load(wigner_inv_ref.at[edge, 6, 8])
  w74 = plt.load(wigner_inv_ref.at[edge, 7, 4])
  w75 = plt.load(wigner_inv_ref.at[edge, 7, 5])
  w76 = plt.load(wigner_inv_ref.at[edge, 7, 6])
  w77 = plt.load(wigner_inv_ref.at[edge, 7, 7])
  w78 = plt.load(wigner_inv_ref.at[edge, 7, 8])
  w84 = plt.load(wigner_inv_ref.at[edge, 8, 4])
  w85 = plt.load(wigner_inv_ref.at[edge, 8, 5])
  w86 = plt.load(wigner_inv_ref.at[edge, 8, 6])
  w87 = plt.load(wigner_inv_ref.at[edge, 8, 7])
  w88 = plt.load(wigner_inv_ref.at[edge, 8, 8])
  y4 = w44 * x4 + w45 * x5 + w46 * x6 + w47 * x7 + w48 * x8
  y5 = w54 * x4 + w55 * x5 + w56 * x6 + w57 * x7 + w58 * x8
  y6 = w64 * x4 + w65 * x5 + w66 * x6 + w67 * x7 + w68 * x8
  y7 = w74 * x4 + w75 * x5 + w76 * x6 + w77 * x7 + w78 * x8
  y8 = w84 * x4 + w85 * x5 + w86 * x6 + w87 * x7 + w88 * x8

  plt.store(out_ref.at[edge, 0, c_offsets], y0, mask=c_mask)
  plt.store(out_ref.at[edge, 1, c_offsets], y1, mask=c_mask)
  plt.store(out_ref.at[edge, 2, c_offsets], y2, mask=c_mask)
  plt.store(out_ref.at[edge, 3, c_offsets], y3, mask=c_mask)
  plt.store(out_ref.at[edge, 4, c_offsets], y4, mask=c_mask)
  plt.store(out_ref.at[edge, 5, c_offsets], y5, mask=c_mask)
  plt.store(out_ref.at[edge, 6, c_offsets], y6, mask=c_mask)
  plt.store(out_ref.at[edge, 7, c_offsets], y7, mask=c_mask)
  plt.store(out_ref.at[edge, 8, c_offsets], y8, mask=c_mask)


def _inverse_wigner(
  messages: jnp.ndarray,
  wigner_and_m_mapping_inv: jnp.ndarray,
) -> jnp.ndarray:
  check_coefficients(messages)
  check_wigner_matrix(wigner_and_m_mapping_inv)
  num_edges = messages.shape[0]
  channels = messages.shape[2]
  if num_edges == 0:
    return jnp.zeros((0, COEFFICIENT_DIM, channels), dtype=messages.dtype)
  out_shape = jax.ShapeDtypeStruct(
    (num_edges, COEFFICIENT_DIM, channels), messages.dtype
  )
  kernel = partial(inverse_wigner_kernel, block_c=CHANNEL_BLOCK_SIZE)
  return pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid=(num_edges, ceil_div(channels, CHANNEL_BLOCK_SIZE)),
    compiler_params=compiler_params(CHANNEL_BLOCK_SIZE, num_stages=2),
  )(messages, wigner_and_m_mapping_inv)


def edge_to_node_bwd_dmessages_kernel(
  gout_ref,
  edge_index_ref,
  wigner_inv_ref,
  out_ref,
  *,
  block_c: int,
):
  edge = pl.program_id(0)
  c_offsets = pl.program_id(1) * block_c + jnp.arange(0, block_c)
  c_mask = c_offsets < out_ref.shape[2]

  receiver = plt.load(edge_index_ref.at[1, edge])
  receiver_valid = (receiver >= 0) & (receiver < gout_ref.shape[0])
  receiver = jnp.clip(receiver, 0, gout_ref.shape[0] - 1)

  g0 = plt.load(
    gout_ref.at[receiver, 0, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g1 = plt.load(
    gout_ref.at[receiver, 1, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g2 = plt.load(
    gout_ref.at[receiver, 2, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g3 = plt.load(
    gout_ref.at[receiver, 3, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g4 = plt.load(
    gout_ref.at[receiver, 4, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g5 = plt.load(
    gout_ref.at[receiver, 5, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g6 = plt.load(
    gout_ref.at[receiver, 6, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g7 = plt.load(
    gout_ref.at[receiver, 7, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g8 = plt.load(
    gout_ref.at[receiver, 8, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )

  dx0 = plt.load(wigner_inv_ref.at[edge, 0, 0]) * g0

  w11 = plt.load(wigner_inv_ref.at[edge, 1, 1])
  w12 = plt.load(wigner_inv_ref.at[edge, 1, 2])
  w13 = plt.load(wigner_inv_ref.at[edge, 1, 3])
  w21 = plt.load(wigner_inv_ref.at[edge, 2, 1])
  w22 = plt.load(wigner_inv_ref.at[edge, 2, 2])
  w23 = plt.load(wigner_inv_ref.at[edge, 2, 3])
  w31 = plt.load(wigner_inv_ref.at[edge, 3, 1])
  w32 = plt.load(wigner_inv_ref.at[edge, 3, 2])
  w33 = plt.load(wigner_inv_ref.at[edge, 3, 3])
  dx1 = w11 * g1 + w21 * g2 + w31 * g3
  dx2 = w12 * g1 + w22 * g2 + w32 * g3
  dx3 = w13 * g1 + w23 * g2 + w33 * g3

  w44 = plt.load(wigner_inv_ref.at[edge, 4, 4])
  w45 = plt.load(wigner_inv_ref.at[edge, 4, 5])
  w46 = plt.load(wigner_inv_ref.at[edge, 4, 6])
  w47 = plt.load(wigner_inv_ref.at[edge, 4, 7])
  w48 = plt.load(wigner_inv_ref.at[edge, 4, 8])
  w54 = plt.load(wigner_inv_ref.at[edge, 5, 4])
  w55 = plt.load(wigner_inv_ref.at[edge, 5, 5])
  w56 = plt.load(wigner_inv_ref.at[edge, 5, 6])
  w57 = plt.load(wigner_inv_ref.at[edge, 5, 7])
  w58 = plt.load(wigner_inv_ref.at[edge, 5, 8])
  w64 = plt.load(wigner_inv_ref.at[edge, 6, 4])
  w65 = plt.load(wigner_inv_ref.at[edge, 6, 5])
  w66 = plt.load(wigner_inv_ref.at[edge, 6, 6])
  w67 = plt.load(wigner_inv_ref.at[edge, 6, 7])
  w68 = plt.load(wigner_inv_ref.at[edge, 6, 8])
  w74 = plt.load(wigner_inv_ref.at[edge, 7, 4])
  w75 = plt.load(wigner_inv_ref.at[edge, 7, 5])
  w76 = plt.load(wigner_inv_ref.at[edge, 7, 6])
  w77 = plt.load(wigner_inv_ref.at[edge, 7, 7])
  w78 = plt.load(wigner_inv_ref.at[edge, 7, 8])
  w84 = plt.load(wigner_inv_ref.at[edge, 8, 4])
  w85 = plt.load(wigner_inv_ref.at[edge, 8, 5])
  w86 = plt.load(wigner_inv_ref.at[edge, 8, 6])
  w87 = plt.load(wigner_inv_ref.at[edge, 8, 7])
  w88 = plt.load(wigner_inv_ref.at[edge, 8, 8])
  dx4 = w44 * g4 + w54 * g5 + w64 * g6 + w74 * g7 + w84 * g8
  dx5 = w45 * g4 + w55 * g5 + w65 * g6 + w75 * g7 + w85 * g8
  dx6 = w46 * g4 + w56 * g5 + w66 * g6 + w76 * g7 + w86 * g8
  dx7 = w47 * g4 + w57 * g5 + w67 * g6 + w77 * g7 + w87 * g8
  dx8 = w48 * g4 + w58 * g5 + w68 * g6 + w78 * g7 + w88 * g8

  plt.store(out_ref.at[edge, 0, c_offsets], dx0, mask=c_mask)
  plt.store(out_ref.at[edge, 5, c_offsets], dx1, mask=c_mask)
  plt.store(out_ref.at[edge, 1, c_offsets], dx2, mask=c_mask)
  plt.store(out_ref.at[edge, 3, c_offsets], dx3, mask=c_mask)
  plt.store(out_ref.at[edge, 8, c_offsets], dx4, mask=c_mask)
  plt.store(out_ref.at[edge, 6, c_offsets], dx5, mask=c_mask)
  plt.store(out_ref.at[edge, 2, c_offsets], dx6, mask=c_mask)
  plt.store(out_ref.at[edge, 4, c_offsets], dx7, mask=c_mask)
  plt.store(out_ref.at[edge, 7, c_offsets], dx8, mask=c_mask)


def _edge_to_node_bwd_dmessages(
  gout: jnp.ndarray,
  edge_index: jnp.ndarray,
  wigner_and_m_mapping_inv: jnp.ndarray,
  messages_shape: tuple[int, int, int],
) -> jnp.ndarray:
  init_shape = messages_shape
  if messages_shape[0] == 0:
    return jnp.zeros(init_shape, dtype=gout.dtype)
  kernel = partial(
    edge_to_node_bwd_dmessages_kernel, block_c=CHANNEL_BLOCK_SIZE
  )
  return pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(init_shape, gout.dtype),
    grid=(messages_shape[0], ceil_div(messages_shape[2], CHANNEL_BLOCK_SIZE)),
    compiler_params=compiler_params(CHANNEL_BLOCK_SIZE),
  )(gout, edge_index, wigner_and_m_mapping_inv)


def edge_to_node_bwd_dwigner_kernel(
  messages_ref,
  gout_ref,
  edge_index_ref,
  init_ref,
  out_ref,
  *,
  block_c: int,
  grid_x: int = MAX_GRID_EDGE_BLOCKS,
  num_edges: int = 0,
):
  del init_ref
  edge = pl.program_id(0) + pl.program_id(1) * grid_x
  safe_edge = jnp.minimum(edge, num_edges - 1)
  channels = messages_ref.shape[2]
  c_offsets = jnp.arange(0, block_c)
  c_mask = c_offsets < channels

  receiver = plt.load(edge_index_ref.at[1, safe_edge])
  receiver_valid = (receiver >= 0) & (receiver < gout_ref.shape[0])
  receiver = jnp.clip(receiver, 0, gout_ref.shape[0] - 1)

  g0 = plt.load(
    gout_ref.at[receiver, 0, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g1 = plt.load(
    gout_ref.at[receiver, 1, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g2 = plt.load(
    gout_ref.at[receiver, 2, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g3 = plt.load(
    gout_ref.at[receiver, 3, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g4 = plt.load(
    gout_ref.at[receiver, 4, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g5 = plt.load(
    gout_ref.at[receiver, 5, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g6 = plt.load(
    gout_ref.at[receiver, 6, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g7 = plt.load(
    gout_ref.at[receiver, 7, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  g8 = plt.load(
    gout_ref.at[receiver, 8, c_offsets],
    mask=c_mask & receiver_valid,
    other=0.0,
  )
  x0 = plt.load(
    messages_ref.at[safe_edge, 0, c_offsets], mask=c_mask, other=0.0
  )
  x1 = plt.load(
    messages_ref.at[safe_edge, 5, c_offsets], mask=c_mask, other=0.0
  )
  x2 = plt.load(
    messages_ref.at[safe_edge, 1, c_offsets], mask=c_mask, other=0.0
  )
  x3 = plt.load(
    messages_ref.at[safe_edge, 3, c_offsets], mask=c_mask, other=0.0
  )
  x4 = plt.load(
    messages_ref.at[safe_edge, 8, c_offsets], mask=c_mask, other=0.0
  )
  x5 = plt.load(
    messages_ref.at[safe_edge, 6, c_offsets], mask=c_mask, other=0.0
  )
  x6 = plt.load(
    messages_ref.at[safe_edge, 2, c_offsets], mask=c_mask, other=0.0
  )
  x7 = plt.load(
    messages_ref.at[safe_edge, 4, c_offsets], mask=c_mask, other=0.0
  )
  x8 = plt.load(
    messages_ref.at[safe_edge, 7, c_offsets], mask=c_mask, other=0.0
  )

  plt.store(out_ref.at[edge, 0, 0], jnp.sum(g0 * x0))
  plt.store(out_ref.at[edge, 1, 1], jnp.sum(g1 * x1))
  plt.store(out_ref.at[edge, 1, 2], jnp.sum(g1 * x2))
  plt.store(out_ref.at[edge, 1, 3], jnp.sum(g1 * x3))
  plt.store(out_ref.at[edge, 2, 1], jnp.sum(g2 * x1))
  plt.store(out_ref.at[edge, 2, 2], jnp.sum(g2 * x2))
  plt.store(out_ref.at[edge, 2, 3], jnp.sum(g2 * x3))
  plt.store(out_ref.at[edge, 3, 1], jnp.sum(g3 * x1))
  plt.store(out_ref.at[edge, 3, 2], jnp.sum(g3 * x2))
  plt.store(out_ref.at[edge, 3, 3], jnp.sum(g3 * x3))
  plt.store(out_ref.at[edge, 4, 4], jnp.sum(g4 * x4))
  plt.store(out_ref.at[edge, 4, 5], jnp.sum(g4 * x5))
  plt.store(out_ref.at[edge, 4, 6], jnp.sum(g4 * x6))
  plt.store(out_ref.at[edge, 4, 7], jnp.sum(g4 * x7))
  plt.store(out_ref.at[edge, 4, 8], jnp.sum(g4 * x8))
  plt.store(out_ref.at[edge, 5, 4], jnp.sum(g5 * x4))
  plt.store(out_ref.at[edge, 5, 5], jnp.sum(g5 * x5))
  plt.store(out_ref.at[edge, 5, 6], jnp.sum(g5 * x6))
  plt.store(out_ref.at[edge, 5, 7], jnp.sum(g5 * x7))
  plt.store(out_ref.at[edge, 5, 8], jnp.sum(g5 * x8))
  plt.store(out_ref.at[edge, 6, 4], jnp.sum(g6 * x4))
  plt.store(out_ref.at[edge, 6, 5], jnp.sum(g6 * x5))
  plt.store(out_ref.at[edge, 6, 6], jnp.sum(g6 * x6))
  plt.store(out_ref.at[edge, 6, 7], jnp.sum(g6 * x7))
  plt.store(out_ref.at[edge, 6, 8], jnp.sum(g6 * x8))
  plt.store(out_ref.at[edge, 7, 4], jnp.sum(g7 * x4))
  plt.store(out_ref.at[edge, 7, 5], jnp.sum(g7 * x5))
  plt.store(out_ref.at[edge, 7, 6], jnp.sum(g7 * x6))
  plt.store(out_ref.at[edge, 7, 7], jnp.sum(g7 * x7))
  plt.store(out_ref.at[edge, 7, 8], jnp.sum(g7 * x8))
  plt.store(out_ref.at[edge, 8, 4], jnp.sum(g8 * x4))
  plt.store(out_ref.at[edge, 8, 5], jnp.sum(g8 * x5))
  plt.store(out_ref.at[edge, 8, 6], jnp.sum(g8 * x6))
  plt.store(out_ref.at[edge, 8, 7], jnp.sum(g8 * x7))
  plt.store(out_ref.at[edge, 8, 8], jnp.sum(g8 * x8))


def _edge_to_node_bwd_dwigner(
  messages: jnp.ndarray,
  gout: jnp.ndarray,
  edge_index: jnp.ndarray,
) -> jnp.ndarray:
  num_edges = messages.shape[0]
  if num_edges == 0:
    return jnp.zeros(
      (0, COEFFICIENT_DIM, COEFFICIENT_DIM), dtype=messages.dtype
    )
  block_c = next_power_of_two(messages.shape[2])
  grid_x = min(num_edges, MAX_GRID_EDGE_BLOCKS)
  grid_y = ceil_div(num_edges, grid_x)
  padded = grid_x * grid_y
  kernel = partial(
    edge_to_node_bwd_dwigner_kernel,
    block_c=block_c,
    grid_x=grid_x,
    num_edges=num_edges,
  )
  params = compiler_params(block_c)
  init = jnp.zeros(
    (padded, COEFFICIENT_DIM, COEFFICIENT_DIM), dtype=messages.dtype
  )
  result = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(init.shape, init.dtype),
    grid=(grid_x, grid_y),
    input_output_aliases={3: 0},
    compiler_params=params,
  )(messages, gout, edge_index, init)
  return result[:num_edges]


def _edge_to_node_scatter(
  messages: jnp.ndarray,
  edge_index: jnp.ndarray,
  wigner_and_m_mapping_inv: jnp.ndarray,
  *,
  num_nodes: int,
) -> jnp.ndarray:
  edge_messages = _inverse_wigner(messages, wigner_and_m_mapping_inv)
  receiver = edge_index[1]
  receiver = jnp.where(receiver >= 0, receiver, num_nodes)
  return (
    jnp.zeros(
      (num_nodes, COEFFICIENT_DIM, messages.shape[2]), dtype=messages.dtype
    )
    .at[receiver]
    .add(edge_messages, mode='drop')
  )


def _zero_tangent(tangent, primal):
  return jnp.zeros_like(primal) if type(tangent) is ad.Zero else tangent


def _zero_if_undefined(value):
  if ad.is_undefined_primal(value):
    return jnp.zeros(value.aval.shape, value.aval.dtype)
  return value


def _batch_size(vals, dims):
  for val, dim in zip(vals, dims):
    if dim is not batching.not_mapped:
      return val.shape[dim]
  raise ValueError('Expected at least one batched input.')


def _move_or_broadcast_batch(value, dim, batch_size):
  if dim is batching.not_mapped:
    return jnp.broadcast_to(value, (batch_size,) + value.shape)
  if dim != 0:
    return jnp.moveaxis(value, dim, 0)
  return value


def _batched_edge_index(edge_index, batch_size, num_nodes):
  edge = edge_index[None, :, :]
  offsets = (jnp.arange(batch_size, dtype=edge_index.dtype) * num_nodes)[
    :, None, None
  ]
  valid = (edge >= 0) & (edge < num_nodes)
  batched = jnp.where(valid, edge + offsets, -1)
  return jnp.transpose(batched, (1, 0, 2)).reshape(
    2, batch_size * edge_index.shape[1]
  )


def _node_to_edge_backward_impl(x, edge_index, wigner, gout):
  dx = _node_to_edge_bwd_dx_scatter(gout, edge_index, wigner, x.shape)
  dwigner = _node_to_edge_bwd_dwigner_gather(x, edge_index, gout)
  return dx, dwigner


node_to_edge_backward_p = core.Primitive('uma_node_to_edge_backward')
node_to_edge_backward_p.multiple_results = True


def _node_to_edge_backward_abstract_eval(
  x_aval, edge_index_aval, w_aval, gout_aval
):
  del edge_index_aval
  return (
    core.ShapedArray(x_aval.shape, gout_aval.dtype),
    core.ShapedArray(w_aval.shape, gout_aval.dtype),
  )


def _node_to_edge_backward_jvp_rule(primals, tangents):
  x, edge_index, wigner, gout = primals
  tx, _, twigner, tgout = tangents
  dx, dwigner = node_to_edge_backward_p.bind(x, edge_index, wigner, gout)
  tangent_dx, tangent_dw = node_to_edge_backward_jvp_p.bind(
    x,
    edge_index,
    wigner,
    gout,
    _zero_tangent(tx, x),
    _zero_tangent(twigner, wigner),
    _zero_tangent(tgout, gout),
  )
  return (dx, dwigner), (tangent_dx, tangent_dw)


def _node_to_edge_backward_transpose_rule(
  cotangents, x, edge_index, wigner, gout
):
  ddx, ddwigner = cotangents
  grad_x = None
  grad_wigner = None
  grad_gout = None

  gout = _zero_if_undefined(gout)
  x_value = _zero_if_undefined(x)
  wigner_value = _zero_if_undefined(wigner)

  if ad.is_undefined_primal(x) and type(ddwigner) is not ad.Zero:
    grad_x, _ = node_to_edge_backward_p.bind(
      jnp.zeros(x.aval.shape, dtype=ddwigner.dtype),
      edge_index,
      ddwigner,
      gout,
    )
  if ad.is_undefined_primal(wigner) and type(ddx) is not ad.Zero:
    _, grad_wigner = node_to_edge_backward_p.bind(
      ddx,
      edge_index,
      jnp.zeros(wigner.aval.shape, dtype=ddx.dtype),
      gout,
    )
  if ad.is_undefined_primal(gout):
    terms = []
    if type(ddx) is not ad.Zero:
      terms.append(node_to_edge_p.bind(ddx, edge_index, wigner_value))
    if type(ddwigner) is not ad.Zero:
      terms.append(node_to_edge_p.bind(x_value, edge_index, ddwigner))
    if terms:
      grad_gout = terms[0]
      for term in terms[1:]:
        grad_gout = grad_gout + term
  return grad_x, None, grad_wigner, grad_gout


def _node_to_edge_backward_batching_rule(vals, dims):
  x, edge_index, wigner, gout = vals
  x_bdim, edge_bdim, w_bdim, gout_bdim = dims
  if edge_bdim is not batching.not_mapped:
    raise NotImplementedError(
      'Batched edge_index is not supported by UMA kernels.'
    )
  if all(dim is batching.not_mapped for dim in (x_bdim, w_bdim, gout_bdim)):
    return node_to_edge_backward_p.bind(x, edge_index, wigner, gout), (
      batching.not_mapped,
      batching.not_mapped,
    )
  batch_size = _batch_size((x, wigner, gout), (x_bdim, w_bdim, gout_bdim))
  x = _move_or_broadcast_batch(x, x_bdim, batch_size)
  wigner = _move_or_broadcast_batch(wigner, w_bdim, batch_size)
  gout = _move_or_broadcast_batch(gout, gout_bdim, batch_size)
  num_nodes, num_edges = x.shape[1], wigner.shape[1]
  flat_edge_index = _batched_edge_index(edge_index, batch_size, num_nodes)
  dx, dwigner = node_to_edge_backward_p.bind(
    x.reshape(batch_size * num_nodes, *x.shape[2:]),
    flat_edge_index,
    wigner.reshape(batch_size * num_edges, *wigner.shape[2:]),
    gout.reshape(batch_size * num_edges, *gout.shape[2:]),
  )
  return (
    dx.reshape(batch_size, num_nodes, *dx.shape[1:]),
    dwigner.reshape(batch_size, num_edges, *dwigner.shape[1:]),
  ), (0, 0)


node_to_edge_backward_jvp_p = core.Primitive('uma_node_to_edge_backward_jvp')
node_to_edge_backward_jvp_p.multiple_results = True


def _node_to_edge_backward_jvp_impl(
  x, edge_index, wigner, gout, tx, twigner, tgout
):
  dx_from_gout, dw_from_gout = node_to_edge_backward_p.bind(
    x, edge_index, wigner, tgout
  )
  dx_from_wigner, _ = node_to_edge_backward_p.bind(x, edge_index, twigner, gout)
  _, dw_from_x = node_to_edge_backward_p.bind(
    tx, edge_index, jnp.zeros_like(wigner), gout
  )
  return dx_from_gout + dx_from_wigner, dw_from_gout + dw_from_x


def _node_to_edge_backward_jvp_abstract_eval(
  x_aval, edge_index_aval, w_aval, gout_aval, tx_aval, tw_aval, tgout_aval
):
  del edge_index_aval, tx_aval, tw_aval, tgout_aval
  return (
    core.ShapedArray(x_aval.shape, gout_aval.dtype),
    core.ShapedArray(w_aval.shape, gout_aval.dtype),
  )


def _node_to_edge_backward_jvp_transpose_rule(
  cotangents, x, edge_index, wigner, gout, tx, twigner, tgout
):
  ddx, ddwigner = cotangents
  x = _zero_if_undefined(x)
  wigner = _zero_if_undefined(wigner)
  gout = _zero_if_undefined(gout)

  grad_tx = None
  grad_twigner = None
  grad_tgout = None
  if ad.is_undefined_primal(tx) and type(ddwigner) is not ad.Zero:
    grad_tx, _ = node_to_edge_backward_p.bind(
      jnp.zeros(tx.aval.shape, dtype=ddwigner.dtype),
      edge_index,
      ddwigner,
      gout,
    )
  if ad.is_undefined_primal(twigner) and type(ddx) is not ad.Zero:
    _, grad_twigner = node_to_edge_backward_p.bind(
      ddx,
      edge_index,
      jnp.zeros(twigner.aval.shape, dtype=ddx.dtype),
      gout,
    )
  if ad.is_undefined_primal(tgout):
    terms = []
    if type(ddx) is not ad.Zero:
      terms.append(node_to_edge_p.bind(ddx, edge_index, wigner))
    if type(ddwigner) is not ad.Zero:
      terms.append(node_to_edge_p.bind(x, edge_index, ddwigner))
    if terms:
      grad_tgout = terms[0]
      for term in terms[1:]:
        grad_tgout = grad_tgout + term
  return None, None, None, None, grad_tx, grad_twigner, grad_tgout


def _node_to_edge_backward_jvp_jvp_rule(primals, tangents):
  x, edge_index, wigner, gout, tx, twigner, tgout = primals
  tx0, _, tw0, tgout0, ttx, ttw, ttgout = tangents
  diff_primals = (x, wigner, gout, tx, twigner, tgout)
  diff_tangents = tuple(
    jnp.zeros_like(primal) if type(tangent) is ad.Zero else tangent
    for primal, tangent in zip(
      diff_primals, (tx0, tw0, tgout0, ttx, ttw, ttgout)
    )
  )

  def func(x_, wigner_, gout_, tx_, twigner_, tgout_):
    return _node_to_edge_backward_jvp_impl(
      x_, edge_index, wigner_, gout_, tx_, twigner_, tgout_
    )

  return jax.jvp(func, diff_primals, diff_tangents)


def _node_to_edge_backward_jvp_batching_rule(vals, dims):
  x, edge_index, wigner, gout, tx, twigner, tgout = vals
  x_bdim, edge_bdim, w_bdim, gout_bdim, tx_bdim, tw_bdim, tgout_bdim = dims
  if edge_bdim is not batching.not_mapped:
    raise NotImplementedError(
      'Batched edge_index is not supported by UMA kernels.'
    )
  if all(
    dim is batching.not_mapped
    for dim in (x_bdim, w_bdim, gout_bdim, tx_bdim, tw_bdim, tgout_bdim)
  ):
    return node_to_edge_backward_jvp_p.bind(
      x, edge_index, wigner, gout, tx, twigner, tgout
    ), (batching.not_mapped, batching.not_mapped)

  batch_size = _batch_size(
    (x, wigner, gout, tx, twigner, tgout),
    (x_bdim, w_bdim, gout_bdim, tx_bdim, tw_bdim, tgout_bdim),
  )
  x = _move_or_broadcast_batch(x, x_bdim, batch_size)
  wigner = _move_or_broadcast_batch(wigner, w_bdim, batch_size)
  gout = _move_or_broadcast_batch(gout, gout_bdim, batch_size)
  tx = _move_or_broadcast_batch(tx, tx_bdim, batch_size)
  twigner = _move_or_broadcast_batch(twigner, tw_bdim, batch_size)
  tgout = _move_or_broadcast_batch(tgout, tgout_bdim, batch_size)
  num_nodes, num_edges = x.shape[1], wigner.shape[1]
  flat_edge_index = _batched_edge_index(edge_index, batch_size, num_nodes)
  dx, dwigner = node_to_edge_backward_jvp_p.bind(
    x.reshape(batch_size * num_nodes, *x.shape[2:]),
    flat_edge_index,
    wigner.reshape(batch_size * num_edges, *wigner.shape[2:]),
    gout.reshape(batch_size * num_edges, *gout.shape[2:]),
    tx.reshape(batch_size * num_nodes, *tx.shape[2:]),
    twigner.reshape(batch_size * num_edges, *twigner.shape[2:]),
    tgout.reshape(batch_size * num_edges, *tgout.shape[2:]),
  )
  return (
    dx.reshape(batch_size, num_nodes, *dx.shape[1:]),
    dwigner.reshape(batch_size, num_edges, *dwigner.shape[1:]),
  ), (0, 0)


node_to_edge_backward_jvp_p.def_impl(_node_to_edge_backward_jvp_impl)
node_to_edge_backward_jvp_p.def_abstract_eval(
  _node_to_edge_backward_jvp_abstract_eval
)
mlir.register_lowering(
  node_to_edge_backward_jvp_p,
  mlir.lower_fun(_node_to_edge_backward_jvp_impl, multiple_results=True),
)
ad.primitive_jvps[node_to_edge_backward_jvp_p] = (
  _node_to_edge_backward_jvp_jvp_rule
)
ad.primitive_transposes[node_to_edge_backward_jvp_p] = (
  _node_to_edge_backward_jvp_transpose_rule
)
batching.primitive_batchers[node_to_edge_backward_jvp_p] = (
  _node_to_edge_backward_jvp_batching_rule
)


node_to_edge_backward_p.def_impl(_node_to_edge_backward_impl)
node_to_edge_backward_p.def_abstract_eval(_node_to_edge_backward_abstract_eval)
mlir.register_lowering(
  node_to_edge_backward_p,
  mlir.lower_fun(_node_to_edge_backward_impl, multiple_results=True),
)
ad.primitive_jvps[node_to_edge_backward_p] = _node_to_edge_backward_jvp_rule
ad.primitive_transposes[node_to_edge_backward_p] = (
  _node_to_edge_backward_transpose_rule
)
batching.primitive_batchers[node_to_edge_backward_p] = (
  _node_to_edge_backward_batching_rule
)


node_to_edge_p = core.Primitive('uma_node_to_edge')
node_to_edge_p.multiple_results = False


def _node_to_edge_abstract_eval(x_aval, edge_index_aval, w_aval):
  del w_aval
  return core.ShapedArray(
    (edge_index_aval.shape[1], COEFFICIENT_DIM, 2 * x_aval.shape[-1]),
    x_aval.dtype,
  )


def _node_to_edge_jvp_rule(primals, tangents):
  x, edge_index, wigner = primals
  tx, _, twigner = tangents
  out = node_to_edge_p.bind(x, edge_index, wigner)
  tangent_out = node_to_edge_jvp_p.bind(
    x,
    edge_index,
    wigner,
    _zero_tangent(tx, x),
    _zero_tangent(twigner, wigner),
  )
  return out, tangent_out


def _node_to_edge_transpose_rule(cotangent, x, edge_index, wigner):
  if type(cotangent) is ad.Zero:
    return None, None, None
  grad_x = None
  grad_wigner = None
  if ad.is_undefined_primal(x):
    dummy_x = jnp.zeros(x.aval.shape, dtype=cotangent.dtype)
    grad_x, _ = node_to_edge_backward_p.bind(
      dummy_x, edge_index, wigner, cotangent
    )
  if ad.is_undefined_primal(wigner):
    dummy_w = jnp.zeros(wigner.aval.shape, dtype=cotangent.dtype)
    _, grad_wigner = node_to_edge_backward_p.bind(
      x, edge_index, dummy_w, cotangent
    )
  return grad_x, None, grad_wigner


def _node_to_edge_batching_rule(vals, dims):
  x, edge_index, wigner = vals
  x_bdim, edge_bdim, w_bdim = dims
  if edge_bdim is not batching.not_mapped:
    raise NotImplementedError(
      'Batched edge_index is not supported by UMA kernels.'
    )
  if all(dim is batching.not_mapped for dim in (x_bdim, w_bdim)):
    return node_to_edge_p.bind(x, edge_index, wigner), batching.not_mapped
  batch_size = _batch_size((x, wigner), (x_bdim, w_bdim))
  x = _move_or_broadcast_batch(x, x_bdim, batch_size)
  wigner = _move_or_broadcast_batch(wigner, w_bdim, batch_size)
  num_nodes, num_edges = x.shape[1], wigner.shape[1]
  flat_edge_index = _batched_edge_index(edge_index, batch_size, num_nodes)
  out = node_to_edge_p.bind(
    x.reshape(batch_size * num_nodes, *x.shape[2:]),
    flat_edge_index,
    wigner.reshape(batch_size * num_edges, *wigner.shape[2:]),
  )
  return out.reshape(batch_size, num_edges, *out.shape[1:]), 0


node_to_edge_jvp_p = core.Primitive('uma_node_to_edge_jvp')
node_to_edge_jvp_p.multiple_results = False


def _node_to_edge_jvp_impl(x, edge_index, wigner, tx, twigner):
  return node_to_edge_p.bind(tx, edge_index, wigner) + node_to_edge_p.bind(
    x, edge_index, twigner
  )


def _node_to_edge_jvp_abstract_eval(
  x_aval, edge_index_aval, w_aval, tx_aval, tw_aval
):
  del w_aval, tx_aval, tw_aval
  return core.ShapedArray(
    (edge_index_aval.shape[1], COEFFICIENT_DIM, 2 * x_aval.shape[-1]),
    x_aval.dtype,
  )


def _node_to_edge_jvp_transpose_rule(
  cotangent, x, edge_index, wigner, tx, twigner
):
  if type(cotangent) is ad.Zero:
    return None, None, None, None, None
  x = _zero_if_undefined(x)
  wigner = _zero_if_undefined(wigner)
  grad_tx = None
  grad_twigner = None
  if ad.is_undefined_primal(tx):
    grad_tx, _ = node_to_edge_backward_p.bind(
      jnp.zeros(tx.aval.shape, dtype=cotangent.dtype),
      edge_index,
      wigner,
      cotangent,
    )
  if ad.is_undefined_primal(twigner):
    _, grad_twigner = node_to_edge_backward_p.bind(
      x,
      edge_index,
      jnp.zeros(twigner.aval.shape, dtype=cotangent.dtype),
      cotangent,
    )
  return None, None, None, grad_tx, grad_twigner


def _node_to_edge_jvp_jvp_rule(primals, tangents):
  x, edge_index, wigner, tx, twigner = primals
  tx0, _, tw0, ttx, ttw = tangents
  diff_primals = (x, wigner, tx, twigner)
  diff_tangents = tuple(
    jnp.zeros_like(primal) if type(tangent) is ad.Zero else tangent
    for primal, tangent in zip(diff_primals, (tx0, tw0, ttx, ttw))
  )

  def func(x_, wigner_, tx_, twigner_):
    return _node_to_edge_jvp_impl(x_, edge_index, wigner_, tx_, twigner_)

  return jax.jvp(func, diff_primals, diff_tangents)


def _node_to_edge_jvp_batching_rule(vals, dims):
  x, edge_index, wigner, tx, twigner = vals
  x_bdim, edge_bdim, w_bdim, tx_bdim, tw_bdim = dims
  if edge_bdim is not batching.not_mapped:
    raise NotImplementedError(
      'Batched edge_index is not supported by UMA kernels.'
    )
  if all(
    dim is batching.not_mapped for dim in (x_bdim, w_bdim, tx_bdim, tw_bdim)
  ):
    return node_to_edge_jvp_p.bind(
      x, edge_index, wigner, tx, twigner
    ), batching.not_mapped

  batch_size = _batch_size(
    (x, wigner, tx, twigner), (x_bdim, w_bdim, tx_bdim, tw_bdim)
  )
  x = _move_or_broadcast_batch(x, x_bdim, batch_size)
  wigner = _move_or_broadcast_batch(wigner, w_bdim, batch_size)
  tx = _move_or_broadcast_batch(tx, tx_bdim, batch_size)
  twigner = _move_or_broadcast_batch(twigner, tw_bdim, batch_size)
  num_nodes, num_edges = x.shape[1], wigner.shape[1]
  flat_edge_index = _batched_edge_index(edge_index, batch_size, num_nodes)
  out = node_to_edge_jvp_p.bind(
    x.reshape(batch_size * num_nodes, *x.shape[2:]),
    flat_edge_index,
    wigner.reshape(batch_size * num_edges, *wigner.shape[2:]),
    tx.reshape(batch_size * num_nodes, *tx.shape[2:]),
    twigner.reshape(batch_size * num_edges, *twigner.shape[2:]),
  )
  return out.reshape(batch_size, num_edges, *out.shape[1:]), 0


node_to_edge_jvp_p.def_impl(_node_to_edge_jvp_impl)
node_to_edge_jvp_p.def_abstract_eval(_node_to_edge_jvp_abstract_eval)
mlir.register_lowering(
  node_to_edge_jvp_p,
  mlir.lower_fun(_node_to_edge_jvp_impl, multiple_results=False),
)
ad.primitive_jvps[node_to_edge_jvp_p] = _node_to_edge_jvp_jvp_rule
ad.primitive_transposes[node_to_edge_jvp_p] = _node_to_edge_jvp_transpose_rule
batching.primitive_batchers[node_to_edge_jvp_p] = (
  _node_to_edge_jvp_batching_rule
)


node_to_edge_p.def_impl(_node_to_edge_impl)
node_to_edge_p.def_abstract_eval(_node_to_edge_abstract_eval)
mlir.register_lowering(
  node_to_edge_p,
  mlir.lower_fun(_node_to_edge_impl, multiple_results=False),
)
ad.primitive_jvps[node_to_edge_p] = _node_to_edge_jvp_rule
ad.primitive_transposes[node_to_edge_p] = _node_to_edge_transpose_rule
batching.primitive_batchers[node_to_edge_p] = _node_to_edge_batching_rule


def node_to_edge_wigner_permute(
  x: jnp.ndarray,
  edge_index: jnp.ndarray,
  wigner_and_m_mapping: jnp.ndarray,
) -> jnp.ndarray:
  return node_to_edge_p.bind(x, edge_index, wigner_and_m_mapping)


def _edge_to_node_backward_impl(
  messages, edge_index, wigner, gout, *, num_nodes
):
  del num_nodes
  dmessages = _edge_to_node_bwd_dmessages(
    gout, edge_index, wigner, messages.shape
  )
  dwigner = _edge_to_node_bwd_dwigner(messages, gout, edge_index)
  return dmessages, dwigner


edge_to_node_backward_p = core.Primitive('uma_edge_to_node_backward')
edge_to_node_backward_p.multiple_results = True


def _edge_to_node_backward_abstract_eval(
  messages_aval, edge_index_aval, w_aval, gout_aval, *, num_nodes
):
  del edge_index_aval, num_nodes
  return (
    core.ShapedArray(messages_aval.shape, gout_aval.dtype),
    core.ShapedArray(w_aval.shape, gout_aval.dtype),
  )


def _edge_to_node_backward_jvp_rule(primals, tangents, *, num_nodes):
  messages, edge_index, wigner, gout = primals
  tmessages, _, twigner, tgout = tangents
  dmessages, dwigner = edge_to_node_backward_p.bind(
    messages, edge_index, wigner, gout, num_nodes=num_nodes
  )
  tangent_dmessages, tangent_dw = edge_to_node_backward_jvp_p.bind(
    messages,
    edge_index,
    wigner,
    gout,
    _zero_tangent(tmessages, messages),
    _zero_tangent(twigner, wigner),
    _zero_tangent(tgout, gout),
    num_nodes=num_nodes,
  )
  return (dmessages, dwigner), (tangent_dmessages, tangent_dw)


def _edge_to_node_backward_transpose_rule(
  cotangents, messages, edge_index, wigner, gout, *, num_nodes
):
  ddmessages, ddwigner = cotangents
  grad_messages = None
  grad_wigner = None
  grad_gout = None
  gout = _zero_if_undefined(gout)
  messages_value = _zero_if_undefined(messages)
  wigner_value = _zero_if_undefined(wigner)

  if ad.is_undefined_primal(messages) and type(ddwigner) is not ad.Zero:
    grad_messages, _ = edge_to_node_backward_p.bind(
      jnp.zeros(messages.aval.shape, dtype=ddwigner.dtype),
      edge_index,
      ddwigner,
      gout,
      num_nodes=num_nodes,
    )
  if ad.is_undefined_primal(wigner) and type(ddmessages) is not ad.Zero:
    _, grad_wigner = edge_to_node_backward_p.bind(
      ddmessages,
      edge_index,
      jnp.zeros(wigner.aval.shape, dtype=ddmessages.dtype),
      gout,
      num_nodes=num_nodes,
    )
  if ad.is_undefined_primal(gout):
    terms = []
    if type(ddmessages) is not ad.Zero:
      terms.append(
        edge_to_node_p.bind(
          ddmessages,
          edge_index,
          wigner_value,
          num_nodes=num_nodes,
        )
      )
    if type(ddwigner) is not ad.Zero:
      terms.append(
        edge_to_node_p.bind(
          messages_value,
          edge_index,
          ddwigner,
          num_nodes=num_nodes,
        )
      )
    if terms:
      grad_gout = terms[0]
      for term in terms[1:]:
        grad_gout = grad_gout + term
  return grad_messages, None, grad_wigner, grad_gout


def _edge_to_node_backward_batching_rule(vals, dims, *, num_nodes):
  messages, edge_index, wigner, gout = vals
  m_bdim, edge_bdim, w_bdim, gout_bdim = dims
  if edge_bdim is not batching.not_mapped:
    raise NotImplementedError(
      'Batched edge_index is not supported by UMA kernels.'
    )
  if all(dim is batching.not_mapped for dim in (m_bdim, w_bdim, gout_bdim)):
    return edge_to_node_backward_p.bind(
      messages, edge_index, wigner, gout, num_nodes=num_nodes
    ), (batching.not_mapped, batching.not_mapped)
  batch_size = _batch_size(
    (messages, wigner, gout), (m_bdim, w_bdim, gout_bdim)
  )
  messages = _move_or_broadcast_batch(messages, m_bdim, batch_size)
  wigner = _move_or_broadcast_batch(wigner, w_bdim, batch_size)
  gout = _move_or_broadcast_batch(gout, gout_bdim, batch_size)
  num_edges = messages.shape[1]
  flat_edge_index = _batched_edge_index(edge_index, batch_size, num_nodes)
  dmessages, dwigner = edge_to_node_backward_p.bind(
    messages.reshape(batch_size * num_edges, *messages.shape[2:]),
    flat_edge_index,
    wigner.reshape(batch_size * num_edges, *wigner.shape[2:]),
    gout.reshape(batch_size * num_nodes, *gout.shape[2:]),
    num_nodes=batch_size * num_nodes,
  )
  return (
    dmessages.reshape(batch_size, num_edges, *dmessages.shape[1:]),
    dwigner.reshape(batch_size, num_edges, *dwigner.shape[1:]),
  ), (0, 0)


edge_to_node_backward_jvp_p = core.Primitive('uma_edge_to_node_backward_jvp')
edge_to_node_backward_jvp_p.multiple_results = True


def _edge_to_node_backward_jvp_impl(
  messages, edge_index, wigner, gout, tmessages, twigner, tgout, *, num_nodes
):
  dmessages_from_gout, dw_from_gout = edge_to_node_backward_p.bind(
    messages, edge_index, wigner, tgout, num_nodes=num_nodes
  )
  dmessages_from_wigner, _ = edge_to_node_backward_p.bind(
    messages, edge_index, twigner, gout, num_nodes=num_nodes
  )
  _, dw_from_messages = edge_to_node_backward_p.bind(
    tmessages,
    edge_index,
    jnp.zeros_like(wigner),
    gout,
    num_nodes=num_nodes,
  )
  return (
    dmessages_from_gout + dmessages_from_wigner,
    dw_from_gout + dw_from_messages,
  )


def _edge_to_node_backward_jvp_abstract_eval(
  messages_aval,
  edge_index_aval,
  w_aval,
  gout_aval,
  tmessages_aval,
  tw_aval,
  tgout_aval,
  *,
  num_nodes,
):
  del edge_index_aval, tmessages_aval, tw_aval, tgout_aval, num_nodes
  return (
    core.ShapedArray(messages_aval.shape, gout_aval.dtype),
    core.ShapedArray(w_aval.shape, gout_aval.dtype),
  )


def _edge_to_node_backward_jvp_transpose_rule(
  cotangents,
  messages,
  edge_index,
  wigner,
  gout,
  tmessages,
  twigner,
  tgout,
  *,
  num_nodes,
):
  ddmessages, ddwigner = cotangents
  messages = _zero_if_undefined(messages)
  wigner = _zero_if_undefined(wigner)
  gout = _zero_if_undefined(gout)

  grad_tmessages = None
  grad_twigner = None
  grad_tgout = None
  if ad.is_undefined_primal(tmessages) and type(ddwigner) is not ad.Zero:
    grad_tmessages, _ = edge_to_node_backward_p.bind(
      jnp.zeros(tmessages.aval.shape, dtype=ddwigner.dtype),
      edge_index,
      ddwigner,
      gout,
      num_nodes=num_nodes,
    )
  if ad.is_undefined_primal(twigner) and type(ddmessages) is not ad.Zero:
    _, grad_twigner = edge_to_node_backward_p.bind(
      ddmessages,
      edge_index,
      jnp.zeros(twigner.aval.shape, dtype=ddmessages.dtype),
      gout,
      num_nodes=num_nodes,
    )
  if ad.is_undefined_primal(tgout):
    terms = []
    if type(ddmessages) is not ad.Zero:
      terms.append(
        edge_to_node_p.bind(ddmessages, edge_index, wigner, num_nodes=num_nodes)
      )
    if type(ddwigner) is not ad.Zero:
      terms.append(
        edge_to_node_p.bind(messages, edge_index, ddwigner, num_nodes=num_nodes)
      )
    if terms:
      grad_tgout = terms[0]
      for term in terms[1:]:
        grad_tgout = grad_tgout + term
  return None, None, None, None, grad_tmessages, grad_twigner, grad_tgout


def _edge_to_node_backward_jvp_jvp_rule(primals, tangents, *, num_nodes):
  messages, edge_index, wigner, gout, tmessages, twigner, tgout = primals
  tm0, _, tw0, tgout0, ttm, ttw, ttgout = tangents
  diff_primals = (messages, wigner, gout, tmessages, twigner, tgout)
  diff_tangents = tuple(
    jnp.zeros_like(primal) if type(tangent) is ad.Zero else tangent
    for primal, tangent in zip(
      diff_primals, (tm0, tw0, tgout0, ttm, ttw, ttgout)
    )
  )

  def func(messages_, wigner_, gout_, tmessages_, twigner_, tgout_):
    return _edge_to_node_backward_jvp_impl(
      messages_,
      edge_index,
      wigner_,
      gout_,
      tmessages_,
      twigner_,
      tgout_,
      num_nodes=num_nodes,
    )

  return jax.jvp(func, diff_primals, diff_tangents)


def _edge_to_node_backward_jvp_batching_rule(vals, dims, *, num_nodes):
  messages, edge_index, wigner, gout, tmessages, twigner, tgout = vals
  m_bdim, edge_bdim, w_bdim, gout_bdim, tm_bdim, tw_bdim, tgout_bdim = dims
  if edge_bdim is not batching.not_mapped:
    raise NotImplementedError(
      'Batched edge_index is not supported by UMA kernels.'
    )
  if all(
    dim is batching.not_mapped
    for dim in (m_bdim, w_bdim, gout_bdim, tm_bdim, tw_bdim, tgout_bdim)
  ):
    return edge_to_node_backward_jvp_p.bind(
      messages,
      edge_index,
      wigner,
      gout,
      tmessages,
      twigner,
      tgout,
      num_nodes=num_nodes,
    ), (batching.not_mapped, batching.not_mapped)

  batch_size = _batch_size(
    (messages, wigner, gout, tmessages, twigner, tgout),
    (m_bdim, w_bdim, gout_bdim, tm_bdim, tw_bdim, tgout_bdim),
  )
  messages = _move_or_broadcast_batch(messages, m_bdim, batch_size)
  wigner = _move_or_broadcast_batch(wigner, w_bdim, batch_size)
  gout = _move_or_broadcast_batch(gout, gout_bdim, batch_size)
  tmessages = _move_or_broadcast_batch(tmessages, tm_bdim, batch_size)
  twigner = _move_or_broadcast_batch(twigner, tw_bdim, batch_size)
  tgout = _move_or_broadcast_batch(tgout, tgout_bdim, batch_size)
  num_edges = messages.shape[1]
  flat_edge_index = _batched_edge_index(edge_index, batch_size, num_nodes)
  dmessages, dwigner = edge_to_node_backward_jvp_p.bind(
    messages.reshape(batch_size * num_edges, *messages.shape[2:]),
    flat_edge_index,
    wigner.reshape(batch_size * num_edges, *wigner.shape[2:]),
    gout.reshape(batch_size * num_nodes, *gout.shape[2:]),
    tmessages.reshape(batch_size * num_edges, *tmessages.shape[2:]),
    twigner.reshape(batch_size * num_edges, *twigner.shape[2:]),
    tgout.reshape(batch_size * num_nodes, *tgout.shape[2:]),
    num_nodes=batch_size * num_nodes,
  )
  return (
    dmessages.reshape(batch_size, num_edges, *dmessages.shape[1:]),
    dwigner.reshape(batch_size, num_edges, *dwigner.shape[1:]),
  ), (0, 0)


edge_to_node_backward_jvp_p.def_impl(_edge_to_node_backward_jvp_impl)
edge_to_node_backward_jvp_p.def_abstract_eval(
  _edge_to_node_backward_jvp_abstract_eval
)
mlir.register_lowering(
  edge_to_node_backward_jvp_p,
  mlir.lower_fun(_edge_to_node_backward_jvp_impl, multiple_results=True),
)
ad.primitive_jvps[edge_to_node_backward_jvp_p] = (
  _edge_to_node_backward_jvp_jvp_rule
)
ad.primitive_transposes[edge_to_node_backward_jvp_p] = (
  _edge_to_node_backward_jvp_transpose_rule
)
batching.primitive_batchers[edge_to_node_backward_jvp_p] = (
  _edge_to_node_backward_jvp_batching_rule
)


edge_to_node_backward_p.def_impl(_edge_to_node_backward_impl)
edge_to_node_backward_p.def_abstract_eval(_edge_to_node_backward_abstract_eval)
mlir.register_lowering(
  edge_to_node_backward_p,
  mlir.lower_fun(_edge_to_node_backward_impl, multiple_results=True),
)
ad.primitive_jvps[edge_to_node_backward_p] = _edge_to_node_backward_jvp_rule
ad.primitive_transposes[edge_to_node_backward_p] = (
  _edge_to_node_backward_transpose_rule
)
batching.primitive_batchers[edge_to_node_backward_p] = (
  _edge_to_node_backward_batching_rule
)


edge_to_node_p = core.Primitive('uma_edge_to_node')
edge_to_node_p.multiple_results = False


def _edge_to_node_abstract_eval(
  messages_aval, edge_index_aval, w_aval, *, num_nodes
):
  del edge_index_aval, w_aval
  return core.ShapedArray(
    (num_nodes, COEFFICIENT_DIM, messages_aval.shape[-1]),
    messages_aval.dtype,
  )


def _edge_to_node_jvp_rule(primals, tangents, *, num_nodes):
  messages, edge_index, wigner = primals
  tmessages, _, twigner = tangents
  out = edge_to_node_p.bind(messages, edge_index, wigner, num_nodes=num_nodes)
  tangent_out = edge_to_node_jvp_p.bind(
    messages,
    edge_index,
    wigner,
    _zero_tangent(tmessages, messages),
    _zero_tangent(twigner, wigner),
    num_nodes=num_nodes,
  )
  return out, tangent_out


def _edge_to_node_transpose_rule(
  cotangent, messages, edge_index, wigner, *, num_nodes
):
  if type(cotangent) is ad.Zero:
    return None, None, None
  grad_messages = None
  grad_wigner = None
  if ad.is_undefined_primal(messages):
    dummy_messages = jnp.zeros(messages.aval.shape, dtype=cotangent.dtype)
    grad_messages, _ = edge_to_node_backward_p.bind(
      dummy_messages, edge_index, wigner, cotangent, num_nodes=num_nodes
    )
  if ad.is_undefined_primal(wigner):
    dummy_w = jnp.zeros(wigner.aval.shape, dtype=cotangent.dtype)
    _, grad_wigner = edge_to_node_backward_p.bind(
      messages, edge_index, dummy_w, cotangent, num_nodes=num_nodes
    )
  return grad_messages, None, grad_wigner


def _edge_to_node_batching_rule(vals, dims, *, num_nodes):
  messages, edge_index, wigner = vals
  m_bdim, edge_bdim, w_bdim = dims
  if edge_bdim is not batching.not_mapped:
    raise NotImplementedError(
      'Batched edge_index is not supported by UMA kernels.'
    )
  if all(dim is batching.not_mapped for dim in (m_bdim, w_bdim)):
    return (
      edge_to_node_p.bind(messages, edge_index, wigner, num_nodes=num_nodes),
      batching.not_mapped,
    )
  batch_size = _batch_size((messages, wigner), (m_bdim, w_bdim))
  messages = _move_or_broadcast_batch(messages, m_bdim, batch_size)
  wigner = _move_or_broadcast_batch(wigner, w_bdim, batch_size)
  num_edges = messages.shape[1]
  flat_edge_index = _batched_edge_index(edge_index, batch_size, num_nodes)
  out = edge_to_node_p.bind(
    messages.reshape(batch_size * num_edges, *messages.shape[2:]),
    flat_edge_index,
    wigner.reshape(batch_size * num_edges, *wigner.shape[2:]),
    num_nodes=batch_size * num_nodes,
  )
  return out.reshape(batch_size, num_nodes, *out.shape[1:]), 0


edge_to_node_jvp_p = core.Primitive('uma_edge_to_node_jvp')
edge_to_node_jvp_p.multiple_results = False


def _edge_to_node_jvp_impl(
  messages, edge_index, wigner, tmessages, twigner, *, num_nodes
):
  return edge_to_node_p.bind(
    tmessages, edge_index, wigner, num_nodes=num_nodes
  ) + edge_to_node_p.bind(messages, edge_index, twigner, num_nodes=num_nodes)


def _edge_to_node_jvp_abstract_eval(
  messages_aval, edge_index_aval, w_aval, tmessages_aval, tw_aval, *, num_nodes
):
  del edge_index_aval, w_aval, tmessages_aval, tw_aval
  return core.ShapedArray(
    (num_nodes, COEFFICIENT_DIM, messages_aval.shape[-1]),
    messages_aval.dtype,
  )


def _edge_to_node_jvp_transpose_rule(
  cotangent, messages, edge_index, wigner, tmessages, twigner, *, num_nodes
):
  if type(cotangent) is ad.Zero:
    return None, None, None, None, None
  messages = _zero_if_undefined(messages)
  wigner = _zero_if_undefined(wigner)
  grad_tmessages = None
  grad_twigner = None
  if ad.is_undefined_primal(tmessages):
    grad_tmessages, _ = edge_to_node_backward_p.bind(
      jnp.zeros(tmessages.aval.shape, dtype=cotangent.dtype),
      edge_index,
      wigner,
      cotangent,
      num_nodes=num_nodes,
    )
  if ad.is_undefined_primal(twigner):
    _, grad_twigner = edge_to_node_backward_p.bind(
      messages,
      edge_index,
      jnp.zeros(twigner.aval.shape, dtype=cotangent.dtype),
      cotangent,
      num_nodes=num_nodes,
    )
  return None, None, None, grad_tmessages, grad_twigner


def _edge_to_node_jvp_jvp_rule(primals, tangents, *, num_nodes):
  messages, edge_index, wigner, tmessages, twigner = primals
  tm0, _, tw0, ttm, ttw = tangents
  diff_primals = (messages, wigner, tmessages, twigner)
  diff_tangents = tuple(
    jnp.zeros_like(primal) if type(tangent) is ad.Zero else tangent
    for primal, tangent in zip(diff_primals, (tm0, tw0, ttm, ttw))
  )

  def func(messages_, wigner_, tmessages_, twigner_):
    return _edge_to_node_jvp_impl(
      messages_, edge_index, wigner_, tmessages_, twigner_, num_nodes=num_nodes
    )

  return jax.jvp(func, diff_primals, diff_tangents)


def _edge_to_node_jvp_batching_rule(vals, dims, *, num_nodes):
  messages, edge_index, wigner, tmessages, twigner = vals
  m_bdim, edge_bdim, w_bdim, tm_bdim, tw_bdim = dims
  if edge_bdim is not batching.not_mapped:
    raise NotImplementedError(
      'Batched edge_index is not supported by UMA kernels.'
    )
  if all(
    dim is batching.not_mapped for dim in (m_bdim, w_bdim, tm_bdim, tw_bdim)
  ):
    return edge_to_node_jvp_p.bind(
      messages,
      edge_index,
      wigner,
      tmessages,
      twigner,
      num_nodes=num_nodes,
    ), batching.not_mapped

  batch_size = _batch_size(
    (messages, wigner, tmessages, twigner),
    (m_bdim, w_bdim, tm_bdim, tw_bdim),
  )
  messages = _move_or_broadcast_batch(messages, m_bdim, batch_size)
  wigner = _move_or_broadcast_batch(wigner, w_bdim, batch_size)
  tmessages = _move_or_broadcast_batch(tmessages, tm_bdim, batch_size)
  twigner = _move_or_broadcast_batch(twigner, tw_bdim, batch_size)
  num_edges = messages.shape[1]
  flat_edge_index = _batched_edge_index(edge_index, batch_size, num_nodes)
  out = edge_to_node_jvp_p.bind(
    messages.reshape(batch_size * num_edges, *messages.shape[2:]),
    flat_edge_index,
    wigner.reshape(batch_size * num_edges, *wigner.shape[2:]),
    tmessages.reshape(batch_size * num_edges, *tmessages.shape[2:]),
    twigner.reshape(batch_size * num_edges, *twigner.shape[2:]),
    num_nodes=batch_size * num_nodes,
  )
  return out.reshape(batch_size, num_nodes, *out.shape[1:]), 0


edge_to_node_jvp_p.def_impl(_edge_to_node_jvp_impl)
edge_to_node_jvp_p.def_abstract_eval(_edge_to_node_jvp_abstract_eval)
mlir.register_lowering(
  edge_to_node_jvp_p,
  mlir.lower_fun(_edge_to_node_jvp_impl, multiple_results=False),
)
ad.primitive_jvps[edge_to_node_jvp_p] = _edge_to_node_jvp_jvp_rule
ad.primitive_transposes[edge_to_node_jvp_p] = _edge_to_node_jvp_transpose_rule
batching.primitive_batchers[edge_to_node_jvp_p] = (
  _edge_to_node_jvp_batching_rule
)


edge_to_node_p.def_impl(_edge_to_node_scatter)
edge_to_node_p.def_abstract_eval(_edge_to_node_abstract_eval)
mlir.register_lowering(
  edge_to_node_p,
  mlir.lower_fun(_edge_to_node_scatter, multiple_results=False),
)
ad.primitive_jvps[edge_to_node_p] = _edge_to_node_jvp_rule
ad.primitive_transposes[edge_to_node_p] = _edge_to_node_transpose_rule
batching.primitive_batchers[edge_to_node_p] = _edge_to_node_batching_rule


def edge_to_node_wigner_inverse(
  messages: jnp.ndarray,
  edge_index: jnp.ndarray,
  wigner_and_m_mapping_inv: jnp.ndarray,
  num_nodes: int,
) -> jnp.ndarray:
  return edge_to_node_p.bind(
    messages, edge_index, wigner_and_m_mapping_inv, num_nodes=num_nodes
  )
