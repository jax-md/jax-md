"""UMA model blocks.

This module provides the building blocks for the UMA model, including
Edgewise message passing and Atomwise feed-forward operations.

Ported from FairChem's UMA implementation.
"""

from typing import List, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_md import util as jmd_util
from jax_md._nn.uma.common.so3 import create_so3_grid
from jax_md._nn.uma.nn import primitives
from jax_md._nn.uma.nn.so2_layers import SO2Convolution
from jax_md._nn.uma.nn.so3_layers import SO3Linear
from jax_md._nn.uma.nn.activation import (
  GateActivation,
  SeparableS2Activation,
)
from jax_md._nn.uma.nn.layer_norm import get_normalization_layer

Array = jmd_util.Array


class Edgewise(eqx.Module):
  """Edgewise message passing module.

  Performs SO(2) convolutions to pass messages along edges.

  Attributes:
      so2_conv_1: SO(2) convolution on the concatenated source/target features.
      so2_conv_2: SO(2) convolution producing the message.
      act: Activation applied between the two convolutions.
  """

  so2_conv_1: SO2Convolution
  so2_conv_2: SO2Convolution
  act: GateActivation | SeparableS2Activation
  use_kernels: bool = eqx.field(static=True)

  def __init__(
    self,
    sphere_channels: int,
    hidden_channels: int,
    lmax: int,
    mmax: int,
    edge_channels_list: List[int],
    m_size: Sequence[int],
    act_type: str = 'gate',
    grid_resolution: int | None = None,
    use_kernels: bool = False,
    *,
    key: Array,
  ):
    self.use_kernels = use_kernels

    keys = jax.random.split(key, 2)

    if act_type == 'gate':
      extra_m0_output_channels = lmax * hidden_channels
    else:
      extra_m0_output_channels = hidden_channels

    self.so2_conv_1 = SO2Convolution(
      sphere_channels=2 * sphere_channels,  # Concatenated source + target
      m_output_channels=hidden_channels,
      lmax=lmax,
      mmax=mmax,
      m_size=m_size,
      internal_weights=False,
      edge_channels_list=edge_channels_list,
      extra_m0_output_channels=extra_m0_output_channels,
      key=keys[0],
    )

    self.so2_conv_2 = SO2Convolution(
      sphere_channels=hidden_channels,
      m_output_channels=sphere_channels,
      lmax=lmax,
      mmax=mmax,
      m_size=m_size,
      internal_weights=True,
      edge_channels_list=None,
      extra_m0_output_channels=None,
      key=keys[1],
    )

    if act_type == 'gate':
      self.act = GateActivation(
        lmax=lmax,
        mmax=mmax,
        num_channels=hidden_channels,
        m_prime=True,
      )
    else:
      self.act = SeparableS2Activation(
        lmax=lmax,
        mmax=mmax,
        grid_resolution=grid_resolution,
        m_prime=True,
      )

  def __call__(
    self,
    x: jnp.ndarray,
    x_edge: jnp.ndarray,
    edge_index: jnp.ndarray,
    wigner_and_M_mapping: jnp.ndarray,
    wigner_and_M_mapping_inv: jnp.ndarray,
    wigner: jnp.ndarray | None,
    wigner_inv: jnp.ndarray | None,
    edge_envelope: jnp.ndarray,
    node_offset: int = 0,
  ) -> jnp.ndarray:
    """Apply edgewise message passing.

    Args:
        x: Node features, shape [num_nodes, num_m_coeffs, sphere_channels].
        x_edge: Edge scalar features, shape [num_edges, edge_channels].
        edge_index: Edge connectivity, shape [2, num_edges].
        wigner_and_M_mapping: Forward Wigner rotation + M mapping.
        wigner_and_M_mapping_inv: Inverse Wigner rotation + M mapping.
        edge_envelope: Edge envelope values, shape [num_edges, 1, 1].
        node_offset: Offset for node indices.

    Returns:
        Message contribution to add to nodes, shape [num_nodes, ...].
    """
    # Gather source/target node features and rotate to edge-aligned frames.
    if (
      self.use_kernels
      and wigner is not None
      and wigner.shape[-2:] == (9, 9)
      and x.shape[-2] == 9
      and jax.default_backend() == 'gpu'
    ):
      from jax_md._nn.uma import kernels

      x_message = kernels.node_to_edge_wigner_permute(x, edge_index, wigner)
    else:
      # Get source and target node features
      if x.shape[0] == 0:
        return jnp.zeros_like(x)
      sender = jnp.where(edge_index[0] >= 0, edge_index[0], x.shape[0])
      receiver = jnp.where(edge_index[1] >= 0, edge_index[1], x.shape[0])
      x_source = jnp.take(x, sender, axis=0, mode='fill', fill_value=0.0)
      x_target = jnp.take(x, receiver, axis=0, mode='fill', fill_value=0.0)

      # Concatenate source and target features
      x_message = jnp.concatenate([x_source, x_target], axis=2)

      # Rotate to edge-aligned frame
      x_message = jnp.einsum('nmj,njc->nmc', wigner_and_M_mapping, x_message)

    # First SO2 convolution
    x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)

    # Activation (m-prime)
    x_message = self.act(x_0_gating, x_message)

    # Second SO2 convolution
    x_message = self.so2_conv_2(x_message, x_edge)
    if isinstance(x_message, tuple):
      # Not reached: so2_conv_2 has no extra m0 outputs.
      x_message = x_message[0]

    # Apply envelope
    x_message = x_message * edge_envelope

    # Rotate back to global frame and aggregate to target nodes
    if (
      self.use_kernels
      and wigner_inv is not None
      and wigner_inv.shape[-2:] == (9, 9)
      and x_message.shape[-2] == 9
      and jax.default_backend() == 'gpu'
      and node_offset == 0
    ):
      from jax_md._nn.uma import kernels

      return kernels.edge_to_node_wigner_inverse(
        x_message, edge_index, wigner_inv, x.shape[0]
      )
    else:
      # Rotate back to global frame
      x_message = jnp.einsum(
        'njm,nmc->njc', wigner_and_M_mapping_inv, x_message
      )

      # Aggregate messages onto target nodes using scatter-add
      target_indices = edge_index[1] - node_offset
      target_indices = jnp.where(
        target_indices >= 0, target_indices, x.shape[0]
      )
      return jnp.zeros_like(x).at[target_indices].add(x_message, mode='drop')


class SpectralAtomwise(eqx.Module):
  """Spectral atomwise feed-forward module.

  Uses SO(3) linear layers with gated activation.

  Attributes:
      Dense: Scalar MLP layers producing the gating scalars.
      so3_linear_1: SO(3) linear layer applied before the gate.
      act: Gate activation.
      so3_linear_2: SO(3) linear layer applied after the gate.
  """

  Dense: Tuple[primitives.Dense, ...]
  so3_linear_1: SO3Linear
  act: GateActivation
  so3_linear_2: SO3Linear

  def __init__(
    self,
    sphere_channels: int,
    hidden_channels: int,
    lmax: int,
    *,
    key: Array,
  ):
    keys = jax.random.split(key, 3)

    self.Dense = (
      primitives.Dense(sphere_channels, lmax * hidden_channels, key=keys[0]),
    )

    self.so3_linear_1 = SO3Linear(
      sphere_channels,
      hidden_channels,
      lmax,
      key=keys[1],
    )

    self.act = GateActivation(
      lmax=lmax,
      mmax=lmax,
      num_channels=hidden_channels,
    )

    self.so3_linear_2 = SO3Linear(
      hidden_channels,
      sphere_channels,
      lmax,
      key=keys[2],
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply spectral atomwise FFN.

    Args:
        x: Node features, shape [num_nodes, (lmax+1)^2, sphere_channels].

    Returns:
        Updated features, same shape.
    """
    scalar_input = x[:, 0:1, :].squeeze(1)
    gating_scalars = jax.nn.silu(self.Dense[0](scalar_input))

    x = self.so3_linear_1(x)

    x = self.act(gating_scalars, x)

    x = self.so3_linear_2(x)

    return x


class GridAtomwise(eqx.Module):
  """Grid-based atomwise feed-forward module.

  Projects to grid, applies MLP, projects back.

  Attributes:
      Dense: Grid MLP layers.
      lmax: Maximum degree l.
      grid_resolution: SO(3) grid resolution, or None for the default.
  """

  Dense: Tuple[primitives.Dense, ...]
  lmax: int = eqx.field(static=True)
  grid_resolution: int | None = eqx.field(static=True)

  def __init__(
    self,
    sphere_channels: int,
    hidden_channels: int,
    lmax: int,
    grid_resolution: int | None = None,
    *,
    key: Array,
  ):
    self.lmax = lmax
    self.grid_resolution = grid_resolution

    keys = jax.random.split(key, 3)

    self.Dense = (
      primitives.Dense(
        sphere_channels, hidden_channels, key=keys[0], use_bias=False
      ),
      primitives.Dense(
        hidden_channels, hidden_channels, key=keys[1], use_bias=False
      ),
      primitives.Dense(
        hidden_channels, sphere_channels, key=keys[2], use_bias=False
      ),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply grid atomwise FFN.

    Args:
        x: Node features, shape [num_nodes, (lmax+1)^2, sphere_channels].

    Returns:
        Updated features, same shape.
    """
    # Constants under a trace, not leaves
    with jax.ensure_compile_time_eval():
      grid = create_so3_grid(
        self.lmax, self.lmax, resolution=self.grid_resolution, rescale=True
      )
    lat, lon, num_coeffs = grid.to_grid_mat.shape
    grid_points = lat * lon

    # Flatten grid dims for efficient matmul dispatch via cuBLAS.
    to_grid_flat = grid.to_grid_mat.reshape(grid_points, num_coeffs)
    from_grid_flat = grid.from_grid_mat.reshape(grid_points, num_coeffs)

    # Project to grid: [N, coeffs, C] -> [N, grid_points, C]
    x_grid = jnp.einsum('gi,nic->ngc', to_grid_flat, x)

    x_grid = self.Dense[0](x_grid)
    x_grid = jax.nn.silu(x_grid)
    x_grid = self.Dense[1](x_grid)
    x_grid = jax.nn.silu(x_grid)
    x_grid = self.Dense[2](x_grid)

    # Project back: [N, grid_points, C] -> [N, coeffs, C]
    x = jnp.einsum('gi,ngc->nic', from_grid_flat, x_grid)

    return x


class UMABlock(eqx.Module):
  """UMA transformer-like block.

  Combines edgewise message passing and atomwise FFN operations
  with residual connections and normalization.

  Attributes:
      norm_1: Pre-normalization for the edgewise residual block.
      edge_wise: Edgewise message passing.
      norm_2: Pre-normalization for the atomwise residual block.
      atom_wise: Atomwise feed-forward.
  """

  norm_1: eqx.Module
  edge_wise: Edgewise
  norm_2: eqx.Module
  atom_wise: SpectralAtomwise | GridAtomwise

  def __init__(
    self,
    sphere_channels: int,
    hidden_channels: int,
    lmax: int,
    mmax: int,
    m_size: Sequence[int],
    edge_channels_list: List[int],
    norm_type: str = 'rms_norm_sh',
    act_type: str = 'gate',
    ff_type: str = 'grid',
    grid_resolution: int | None = None,
    use_kernels: bool = False,
    *,
    key: Array,
  ):
    keys = jax.random.split(key, 2)

    self.norm_1 = get_normalization_layer(
      norm_type,
      lmax=lmax,
      num_channels=sphere_channels,
    )

    self.edge_wise = Edgewise(
      sphere_channels=sphere_channels,
      hidden_channels=hidden_channels,
      lmax=lmax,
      mmax=mmax,
      edge_channels_list=edge_channels_list,
      m_size=m_size,
      act_type=act_type,
      grid_resolution=grid_resolution,
      use_kernels=use_kernels,
      key=keys[0],
    )

    self.norm_2 = get_normalization_layer(
      norm_type,
      lmax=lmax,
      num_channels=sphere_channels,
    )

    if ff_type == 'spectral':
      self.atom_wise = SpectralAtomwise(
        sphere_channels=sphere_channels,
        hidden_channels=hidden_channels,
        lmax=lmax,
        key=keys[1],
      )
    else:
      self.atom_wise = GridAtomwise(
        sphere_channels=sphere_channels,
        hidden_channels=hidden_channels,
        lmax=lmax,
        grid_resolution=grid_resolution,
        key=keys[1],
      )

  def __call__(
    self,
    x: jnp.ndarray,
    x_edge: jnp.ndarray,
    edge_index: jnp.ndarray,
    wigner_and_M_mapping: jnp.ndarray,
    wigner_and_M_mapping_inv: jnp.ndarray,
    wigner: jnp.ndarray | None,
    wigner_inv: jnp.ndarray | None,
    edge_envelope: jnp.ndarray,
    sys_node_embedding: jnp.ndarray | None = None,
    node_offset: int = 0,
  ) -> jnp.ndarray:
    """Apply UMA block.

    Args:
        x: Node features, shape [num_nodes, (lmax+1)^2, sphere_channels].
        x_edge: Edge scalar features, shape [num_edges, edge_channels].
        edge_index: Edge connectivity, shape [2, num_edges].
        wigner_and_M_mapping: Forward Wigner rotation + M mapping.
        wigner_and_M_mapping_inv: Inverse Wigner rotation + M mapping.
        edge_envelope: Edge envelope values, shape [num_edges, 1, 1].
        sys_node_embedding: Per-system embedding to add, shape [num_nodes, sphere_channels].
        node_offset: Offset for node indices.

    Returns:
        Updated node features, same shape as x.
    """
    # First residual block: Edgewise
    x_res = x

    x = self.norm_1(x)  # ty: ignore[call-non-callable]

    # Add system embedding to scalars
    if sys_node_embedding is not None:
      x = x.at[:, 0, :].add(sys_node_embedding)

    x = self.edge_wise(
      x,
      x_edge,
      edge_index,
      wigner_and_M_mapping,
      wigner_and_M_mapping_inv,
      wigner,
      wigner_inv,
      edge_envelope,
      node_offset,
    )
    x = x + x_res

    # Second residual block: Atomwise
    x_res = x

    x = self.norm_2(x)  # ty: ignore[call-non-callable]

    x = self.atom_wise(x)
    x = x + x_res

    return x
