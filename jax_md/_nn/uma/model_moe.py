"""UMA Mixture-of-Experts backbone model.

This extends UMABackbone with Mixture-of-Linear-Experts, MOLE, layers
in the SO(2) convolutions. The pretrained UMA checkpoints use this
architecture.

The MoE mechanism:
1. A routing MLP takes system-level features and produces per-system
   expert mixing coefficients [num_systems, num_experts].
2. Expert weights in SO2 convolutions are mixed per-system:
   mixed_W = einsum('eoi,be->boi', expert_W, coefficients)
3. Each atom/edge uses its system's mixed weight matrix.
"""

import ast
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jax_md import util as jmd_util
from jax_md._nn.uma.model import UMAConfig
from jax_md._nn.uma.blocks import SpectralAtomwise, GridAtomwise
from jax_md._nn.uma.common.so3 import (
  create_coefficient_mapping,
  create_so3_grid,
  coefficient_idx,
)
from jax_md._nn.uma.common.rotation import (
  load_jacobi_matrices_from_file,
  init_edge_rot_euler_angles,
  eulers_to_wigner,
)
from jax_md._nn.uma.nn import primitives
from jax_md._nn.uma.nn.radial import (
  GaussianSmearing,
  PolynomialEnvelope,
  RadialMLP,
)
from jax_md._nn.uma.nn.embedding import (
  ChgSpinEmbedding,
  DatasetEmbedding,
  EdgeDegreeEmbedding,
  symmetric_uniform,
)
from jax_md._nn.uma.nn.layer_norm import get_normalization_layer
from jax_md._nn.uma.nn.activation import GateActivation, SeparableS2Activation
from jax_md._nn.uma.nn.mole import MOLELinear, RoutingMLP
from jax_md._nn.uma.pretrained import ConversionMetadata

Array = jmd_util.Array


@dataclass
class UMAMoEConfig(UMAConfig):
  """Configuration for UMA MoE model.

  Extends UMAConfig with MoE-specific parameters.

  Attributes:
      use_segment_mm_pallas: Use the Pallas segmented matmul.
      max_edge_segment_size: Static upper bound on the number of edges
        belonging to a single system.
  """

  num_experts: int = 32
  moe_dropout: float = 0.05
  use_composition_embedding: bool = True
  model_version: float = 1.1
  routing_hidden_channels: int = 64
  charge_balanced_channels: Tuple[int, ...] = ()
  spin_balanced_channels: Tuple[int, ...] = ()
  merged_mole: bool = False
  so2_block_gemm: bool = False
  use_segment_mm_pallas: bool = True
  max_edge_segment_size: int | None = None


def channel_range(channels: Tuple[int, ...]) -> Tuple[int, int]:
  if not channels:
    return 0, 0
  sorted_channels = sorted(channels)
  expected = list(range(sorted_channels[0], sorted_channels[-1] + 1))
  if sorted_channels != expected:
    raise ValueError(f'Balanced channels must be contiguous, got {channels}')
  return sorted_channels[0], sorted_channels[-1] + 1


def balance_channels(
  emb: jnp.ndarray,
  target: jnp.ndarray,
  batch: jnp.ndarray,
  start_idx: int,
  end_idx: int,
  target_offset: float = 0.0,
) -> jnp.ndarray:
  if end_idx <= start_idx:
    return emb
  channels = emb[:, 0, start_idx:end_idx]
  num_systems = target.shape[0]
  system_sums = (
    jnp.zeros((num_systems, end_idx - start_idx), dtype=emb.dtype)
    .at[batch]
    .add(channels)
  )
  natoms = jnp.zeros((num_systems,), dtype=emb.dtype).at[batch].add(1.0)
  target_sums = (target.astype(emb.dtype) - target_offset)[:, None]
  corrections = (system_sums - target_sums) / natoms[:, None]
  return emb.at[:, 0, start_idx:end_idx].set(channels - corrections[batch])


def merge_mole_params(params: Any, expert_coefficients: jnp.ndarray) -> Any:
  """Return a params tree with MOLE expert weights pre-mixed.

  This is intended for single-system inference. Any parameter named
  ``weights`` with shape ``[num_experts, out, in]`` is replaced by the
  corresponding mixed weight with shape ``[out, in]``.
  """
  coeff = expert_coefficients[0]
  num_experts = coeff.shape[0]

  def merge_leaf(key: str, value: Any) -> Any:
    if (
      key == 'weights'
      and hasattr(value, 'ndim')
      and value.ndim == 3
      and value.shape[0] == num_experts
    ):
      return jnp.einsum('e,eoi->oi', coeff, value)
    return value

  def recurse(tree: Any) -> Any:
    if isinstance(tree, dict):
      return {k: recurse_value(k, v) for k, v in tree.items()}
    return tree

  def recurse_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
      return recurse(value)
    return merge_leaf(key, value)

  return recurse(params)


class SO2MConvMoE(eqx.Module):
  """SO(2) convolution for a single order m, with MoE expert weights.

  Like SO2MConv but the fc weight has shape [num_experts, 2*out_half, in].
  Expert mixing is applied before the linear transformation.

  Attributes:
      m: Order of the spherical harmonic coefficients.
      m_output_channels: Number of output channels.
      lmax: Maximum degree l.
  """

  fc: MOLELinear
  m: int = eqx.field(static=True)
  m_output_channels: int = eqx.field(static=True)
  lmax: int = eqx.field(static=True)
  so2_block_gemm: bool = eqx.field(static=True)

  def __init__(
    self,
    m: int,
    sphere_channels: int,
    m_output_channels: int,
    lmax: int,
    num_experts: int,
    merged_mole: bool = False,
    so2_block_gemm: bool = False,
    use_segment_mm_pallas: bool = True,
    max_edge_segment_size: int | None = None,
    *,
    key: Array,
  ):
    self.m = m
    self.m_output_channels = m_output_channels
    self.lmax = lmax
    self.so2_block_gemm = so2_block_gemm

    num_coefficients = lmax - m + 1
    out_channels_half = m_output_channels * num_coefficients
    self.fc = MOLELinear(
      num_experts=num_experts,
      in_features=num_coefficients * sphere_channels,
      out_features=2 * out_channels_half,
      use_bias=False,
      merged=merged_mole,
      use_segment_mm_pallas=use_segment_mm_pallas,
      max_segment_size=max_edge_segment_size,
      assume_segment_contiguous_batches=use_segment_mm_pallas,
      key=key,
    )

  def block_gemm(
    self,
    x_m: jnp.ndarray,
    weights: jnp.ndarray,
    expert_coefficients: jnp.ndarray,
    edge_batch: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply SO2 complex multiply as one block GEMM."""
    num_coefficients = self.lmax - self.m + 1
    out_channels_half = self.m_output_channels * num_coefficients
    if weights.ndim == 3:
      if expert_coefficients.shape[0] == 1:
        weights = jnp.einsum('e,eoi->oi', expert_coefficients[0], weights)
      else:
        weights = jnp.einsum('eoi,be->boi', weights, expert_coefficients)

    w1, w2 = jnp.split(weights, 2, axis=-2)
    w_block = jnp.concatenate(
      [
        jnp.concatenate([w1, -w2], axis=-1),
        jnp.concatenate([w2, w1], axis=-1),
      ],
      axis=-2,
    )
    batch_size = x_m.shape[0]
    x_cat = x_m.reshape(batch_size, -1)
    if w_block.ndim == 2:
      out_cat = jnp.einsum('ni,oi->no', x_cat, w_block)
    else:
      # per-system block weights go through MOLELinear
      out_cat = self.fc(
        x_cat, expert_coefficients, edge_batch, mixed_weights=w_block
      )
    out = out_cat.reshape(batch_size, 2, out_channels_half)
    x_m_r = out[:, 0, :].reshape(
      batch_size, num_coefficients, self.m_output_channels
    )
    x_m_i = out[:, 1, :].reshape(
      batch_size, num_coefficients, self.m_output_channels
    )
    return x_m_r, x_m_i

  def __call__(
    self,
    x_m: jnp.ndarray,
    expert_coefficients: jnp.ndarray,
    edge_batch: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply MoE SO(2) convolution for order m.

    Args:
        x_m: Input, shape [num_edges, 2, num_coeffs * sphere_channels].
        expert_coefficients: Shape [num_systems, num_experts].
        edge_batch: System index per edge, shape [num_edges].

    Returns:
        Tuple of (real_output, imag_output).
    """
    num_coefficients = self.lmax - self.m + 1
    out_channels_half = self.m_output_channels * num_coefficients

    if self.so2_block_gemm:
      return self.block_gemm(
        x_m, self.fc.weights, expert_coefficients, edge_batch
      )

    # Apply MoE linear to last dim: (E, 2, C) -> (E, 2, 2*out_half)
    # MOLELinear handles 3D: [E, 2, C] with batch_indices for edges
    x_m = self.fc(x_m, expert_coefficients, edge_batch)

    batch_size = x_m.shape[0]
    x_m = x_m.reshape(batch_size, -1, out_channels_half)

    x_r_0 = x_m[:, 0, :]
    x_i_0 = x_m[:, 1, :]
    x_r_1 = x_m[:, 2, :]
    x_i_1 = x_m[:, 3, :]

    x_m_r = x_r_0 - x_i_1
    x_m_i = x_r_1 + x_i_0

    x_m_r = x_m_r.reshape(batch_size, num_coefficients, self.m_output_channels)
    x_m_i = x_m_i.reshape(batch_size, num_coefficients, self.m_output_channels)

    return x_m_r, x_m_i


class SO2ConvolutionMoE(eqx.Module):
  """SO(2) convolution block with MoE expert weights.

  Like SO2Convolution but fc_m0 and so2_m_conv use MOLE layers.

  Attributes:
      sphere_channels: Number of input spherical channels.
      m_output_channels: Number of output channels per m.
      lmax: Maximum degree l.
      mmax: Maximum order m.
      m_size: List of number of coefficients for each m.
      extra_m0_output_channels: Extra output channels for m=0 gating.
  """

  fc_m0: MOLELinear
  rad_func: RadialMLP | None
  so2_m_conv: Dict[str, SO2MConvMoE]
  sphere_channels: int = eqx.field(static=True)
  m_output_channels: int = eqx.field(static=True)
  lmax: int = eqx.field(static=True)
  mmax: int = eqx.field(static=True)
  m_size: Tuple[int, ...] = eqx.field(static=True)
  extra_m0_output_channels: int | None = eqx.field(static=True)

  def __init__(
    self,
    sphere_channels: int,
    m_output_channels: int,
    lmax: int,
    mmax: int,
    m_size: Sequence[int],
    num_experts: int,
    internal_weights: bool = True,
    edge_channels_list: List[int] | None = None,
    extra_m0_output_channels: int | None = None,
    merged_mole: bool = False,
    so2_block_gemm: bool = False,
    use_segment_mm_pallas: bool = True,
    max_edge_segment_size: int | None = None,
    *,
    key: Array,
  ):
    self.sphere_channels = sphere_channels
    self.m_output_channels = m_output_channels
    self.lmax = lmax
    self.mmax = mmax
    self.m_size = tuple(m_size)
    self.extra_m0_output_channels = extra_m0_output_channels

    keys = jax.random.split(key, 2 + mmax)

    num_channels_m0 = (lmax + 1) * sphere_channels
    m0_output_channels = m_output_channels * (lmax + 1)
    if extra_m0_output_channels is not None:
      m0_output_channels = m0_output_channels + extra_m0_output_channels

    self.fc_m0 = MOLELinear(
      num_experts=num_experts,
      in_features=num_channels_m0,
      out_features=m0_output_channels,
      use_bias=True,
      merged=merged_mole,
      use_segment_mm_pallas=use_segment_mm_pallas,
      max_segment_size=max_edge_segment_size,
      assume_segment_contiguous_batches=use_segment_mm_pallas,
      key=keys[0],
    )

    num_channels_rad = num_channels_m0
    for m in range(1, mmax + 1):
      num_channels_rad += (lmax - m + 1) * sphere_channels

    self.rad_func = None
    if not internal_weights:
      edge_channels = list(edge_channels_list or []) + [num_channels_rad]
      self.rad_func = RadialMLP(channels_list=edge_channels, key=keys[1])

    self.so2_m_conv = {
      f'so2_m_conv_{m}': SO2MConvMoE(
        m=m,
        sphere_channels=sphere_channels,
        m_output_channels=m_output_channels,
        lmax=lmax,
        num_experts=num_experts,
        merged_mole=merged_mole,
        so2_block_gemm=so2_block_gemm,
        use_segment_mm_pallas=use_segment_mm_pallas,
        max_edge_segment_size=max_edge_segment_size,
        key=keys[1 + m],
      )
      for m in range(1, mmax + 1)
    }

  def __call__(
    self,
    x: jnp.ndarray,
    x_edge: jnp.ndarray,
    expert_coefficients: jnp.ndarray,
    edge_batch: jnp.ndarray,
  ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply MoE SO(2) convolution.

    Args:
        x: Input in m-major ordering, shape [num_edges, num_m_coeffs, sphere_channels].
        x_edge: Edge scalar features, shape [num_edges, edge_channels].
        expert_coefficients: Shape [num_systems, num_experts].
        edge_batch: System index per edge, shape [num_edges].
    """
    num_channels_m0 = (self.lmax + 1) * self.sphere_channels

    m_split_sizes = [self.m_size[m] for m in range(self.mmax + 1)]
    edge_split_sizes = [num_channels_m0]
    for m in range(1, self.mmax + 1):
      edge_split_sizes.append((self.lmax - m + 1) * self.sphere_channels)

    num_edges = x.shape[0]

    use_static_l2_layout = (
      self.lmax == 2
      and self.mmax == 2
      and len(self.m_size) == 3
      and self.m_size[0] == 3
      and self.m_size[1] == 4
      and self.m_size[2] == 2
    )

    if self.rad_func is not None:
      x_edge_weights = self.rad_func(x_edge)
    else:
      x_edge_weights = None

    if x_edge_weights is not None:
      if use_static_l2_layout:
        edge_m0 = num_channels_m0
        edge_m1 = edge_m0 + 2 * self.sphere_channels
        x_edge_by_m = [
          x_edge_weights[:, :edge_m0],
          x_edge_weights[:, edge_m0:edge_m1],
          x_edge_weights[:, edge_m1:],
        ]
      else:
        x_edge_by_m = jnp.split(
          x_edge_weights,
          np.cumsum(edge_split_sizes[:-1]),
          axis=1,
        )
    else:
      x_edge_by_m = [None] * (self.mmax + 1)

    if use_static_l2_layout:
      x_by_m = [x[:, :3], x[:, 3:7], x[:, 7:9]]
    else:
      x_by_m = jnp.split(
        x,
        np.cumsum(m_split_sizes[:-1]),
        axis=1,
      )

    # m=0 with MoE
    x_0 = x_by_m[0].reshape(num_edges, -1)
    if x_edge_by_m[0] is not None:
      x_0 = x_0 * x_edge_by_m[0]
    x_0 = self.fc_m0(x_0, expert_coefficients, edge_batch)

    if self.extra_m0_output_channels is not None:
      x_0_extra = x_0[:, : self.extra_m0_output_channels]
      x_0 = x_0[:, self.extra_m0_output_channels :]

    out = [x_0.reshape(num_edges, -1, self.m_output_channels)]

    # m > 0 with MoE
    for m in range(1, self.mmax + 1):
      x_m = x_by_m[m].reshape(num_edges, 2, -1)
      x_edge_m = x_edge_by_m[m]
      if x_edge_m is not None:
        x_m = x_m * x_edge_m[:, None, :]

      x_m_r, x_m_i = self.so2_m_conv[f'so2_m_conv_{m}'](
        x_m, expert_coefficients, edge_batch
      )
      out.append(x_m_r)
      out.append(x_m_i)

    out = jnp.concatenate(out, axis=1)

    if self.extra_m0_output_channels is not None:
      return out, x_0_extra
    return out


class EdgewiseMoE(eqx.Module):
  """Edgewise message passing with MoE SO2 convolutions."""

  so2_conv_1: SO2ConvolutionMoE
  so2_conv_2: SO2ConvolutionMoE
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
    num_experts: int,
    act_type: str = 'gate',
    grid_resolution: int | None = None,
    use_kernels: bool = False,
    merged_mole: bool = False,
    so2_block_gemm: bool = False,
    use_segment_mm_pallas: bool = True,
    max_edge_segment_size: int | None = None,
    *,
    key: Array,
  ):
    self.use_kernels = use_kernels

    keys = jax.random.split(key, 2)

    if act_type == 'gate':
      extra_m0 = lmax * hidden_channels
    else:
      extra_m0 = hidden_channels

    self.so2_conv_1 = SO2ConvolutionMoE(
      sphere_channels=2 * sphere_channels,
      m_output_channels=hidden_channels,
      lmax=lmax,
      mmax=mmax,
      m_size=m_size,
      num_experts=num_experts,
      internal_weights=False,
      edge_channels_list=edge_channels_list,
      extra_m0_output_channels=extra_m0,
      merged_mole=merged_mole,
      so2_block_gemm=so2_block_gemm,
      use_segment_mm_pallas=use_segment_mm_pallas,
      max_edge_segment_size=max_edge_segment_size,
      key=keys[0],
    )

    self.so2_conv_2 = SO2ConvolutionMoE(
      sphere_channels=hidden_channels,
      m_output_channels=sphere_channels,
      lmax=lmax,
      mmax=mmax,
      m_size=m_size,
      num_experts=num_experts,
      internal_weights=True,
      merged_mole=merged_mole,
      so2_block_gemm=so2_block_gemm,
      use_segment_mm_pallas=use_segment_mm_pallas,
      max_edge_segment_size=max_edge_segment_size,
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
    edge_envelope,
    expert_coefficients: jnp.ndarray,
    edge_batch: jnp.ndarray,
    node_offset: int = 0,
  ) -> jnp.ndarray:
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
      if x.shape[0] == 0:
        return jnp.zeros_like(x)
      sender = jnp.where(edge_index[0] >= 0, edge_index[0], x.shape[0])
      receiver = jnp.where(edge_index[1] >= 0, edge_index[1], x.shape[0])
      x_source = jnp.take(x, sender, axis=0, mode='fill', fill_value=0.0)
      x_target = jnp.take(x, receiver, axis=0, mode='fill', fill_value=0.0)
      x_message = jnp.concatenate([x_source, x_target], axis=2)
      x_message = jnp.einsum('nmj,njc->nmc', wigner_and_M_mapping, x_message)

    x_message, x_0_gating = self.so2_conv_1(
      x_message, x_edge, expert_coefficients, edge_batch
    )

    x_message = self.act(x_0_gating, x_message)

    x_message = self.so2_conv_2(
      x_message, x_edge, expert_coefficients, edge_batch
    )

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
    edge_messages = jnp.einsum(
      'njm,nmc->njc', wigner_and_M_mapping_inv, x_message
    )
    target_indices = edge_index[1] - node_offset
    target_indices = jnp.where(target_indices >= 0, target_indices, x.shape[0])
    return jnp.zeros_like(x).at[target_indices].add(edge_messages, mode='drop')


class UMABlockMoE(eqx.Module):
  """UMA block with MoE edgewise and standard atomwise."""

  norm_1: eqx.Module
  edge_wise: EdgewiseMoE
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
    num_experts: int,
    norm_type: str = 'rms_norm_sh',
    act_type: str = 'gate',
    ff_type: str = 'spectral',
    grid_resolution: int | None = None,
    use_kernels: bool = False,
    merged_mole: bool = False,
    so2_block_gemm: bool = False,
    use_segment_mm_pallas: bool = True,
    max_edge_segment_size: int | None = None,
    *,
    key: Array,
  ):
    keys = jax.random.split(key, 2)

    self.norm_1 = get_normalization_layer(
      norm_type,
      lmax=lmax,
      num_channels=sphere_channels,
    )

    self.edge_wise = EdgewiseMoE(
      sphere_channels=sphere_channels,
      hidden_channels=hidden_channels,
      lmax=lmax,
      mmax=mmax,
      edge_channels_list=edge_channels_list,
      m_size=m_size,
      num_experts=num_experts,
      act_type=act_type,
      grid_resolution=grid_resolution,
      use_kernels=use_kernels,
      merged_mole=merged_mole,
      so2_block_gemm=so2_block_gemm,
      use_segment_mm_pallas=use_segment_mm_pallas,
      max_edge_segment_size=max_edge_segment_size,
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
    expert_coefficients: jnp.ndarray,
    edge_batch: jnp.ndarray,
    sys_node_embedding: jnp.ndarray | None = None,
    node_offset: int = 0,
  ) -> jnp.ndarray:
    x_res = x
    x = self.norm_1(x)  # ty: ignore[call-non-callable]

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
      expert_coefficients,
      edge_batch,
      node_offset,
    )
    x = x + x_res

    x_res = x
    x = self.norm_2(x)  # ty: ignore[call-non-callable]

    x = self.atom_wise(x)
    x = x + x_res
    return x


class UMAMoEBackbone(eqx.Module):
  """UMA backbone with Mixture-of-Experts.

  This is the architecture used by the pretrained UMA checkpoints.
  It extends UMABackbone with:
  - Expert weights in SO2 convolutions
  - A routing MLP for expert selection
  - Composition embedding for routing input

  Attributes:
      config: UMAMoEConfig instance.
  """

  sphere_embedding: primitives.Embed
  charge_embedding: ChgSpinEmbedding
  spin_embedding: ChgSpinEmbedding
  dataset_embedding: DatasetEmbedding | None
  mix_csd: primitives.Dense
  composition_embedding: primitives.Embed | None
  routing_mlp: RoutingMLP | None
  envelope: PolynomialEnvelope
  distance_expansion: GaussianSmearing
  source_embedding: primitives.Embed
  target_embedding: primitives.Embed
  edge_degree_embedding: EdgeDegreeEmbedding
  blocks: Tuple[UMABlockMoE, ...]
  norm: eqx.Module
  config: UMAMoEConfig = eqx.field(static=True)
  sph_feature_size: int = eqx.field(static=True)
  charge_channel_start: int = eqx.field(static=True)
  charge_channel_end: int = eqx.field(static=True)
  spin_channel_start: int = eqx.field(static=True)
  spin_channel_end: int = eqx.field(static=True)

  def __init__(self, config: UMAMoEConfig, *, key: Array):
    cfg = config
    self.config = config
    self.sph_feature_size = (cfg.lmax + 1) ** 2
    mapping = create_coefficient_mapping(cfg.lmax, cfg.mmax)
    self.charge_channel_start, self.charge_channel_end = channel_range(
      cfg.charge_balanced_channels
    )
    self.spin_channel_start, self.spin_channel_end = channel_range(
      cfg.spin_balanced_channels
    )
    edge_channels_list = [
      cfg.num_distance_basis + 2 * cfg.edge_channels,
      cfg.edge_channels,
      cfg.edge_channels,
    ]

    keys = jax.random.split(key, 10 + cfg.num_layers)

    self.sphere_embedding = primitives.Embed(
      num_embeddings=cfg.max_num_elements,
      features=cfg.sphere_channels,
      key=keys[0],
    )

    self.charge_embedding = ChgSpinEmbedding(
      embedding_type=cfg.chg_spin_emb_type,
      embedding_target='charge',
      embedding_size=cfg.sphere_channels,
      trainable=False,
      key=keys[1],
    )
    self.spin_embedding = ChgSpinEmbedding(
      embedding_type=cfg.chg_spin_emb_type,
      embedding_target='spin',
      embedding_size=cfg.sphere_channels,
      trainable=False,
      key=keys[2],
    )

    num_csd_embeddings = 2
    self.dataset_embedding = None
    if cfg.dataset_list:
      self.dataset_embedding = DatasetEmbedding(
        embedding_size=cfg.sphere_channels,
        num_datasets=len(cfg.dataset_list),
        trainable=False,
        key=keys[3],
      )
      num_csd_embeddings = 3

    self.mix_csd = primitives.Dense(
      num_csd_embeddings * cfg.sphere_channels,
      cfg.sphere_channels,
      key=keys[4],
    )

    self.composition_embedding = None
    self.routing_mlp = None
    if not cfg.merged_mole:
      routing_input_channels = 0

      if cfg.use_composition_embedding:
        self.composition_embedding = primitives.Embed(
          num_embeddings=cfg.max_num_elements,
          features=cfg.sphere_channels,
          key=keys[5],
        )
        routing_input_channels += cfg.sphere_channels

      routing_input_channels += cfg.sphere_channels

      self.routing_mlp = RoutingMLP(
        in_features=routing_input_channels,
        hidden_channels=cfg.routing_hidden_channels,
        num_experts=cfg.num_experts,
        dropout_rate=cfg.moe_dropout,
        key=keys[6],
      )

    self.envelope = PolynomialEnvelope(exponent=5)

    self.distance_expansion = GaussianSmearing(
      start=0.0,
      stop=cfg.cutoff,
      num_gaussians=cfg.num_distance_basis,
      basis_width_scalar=2.0,
    )

    self.source_embedding = primitives.Embed(
      num_embeddings=cfg.max_num_elements,
      features=cfg.edge_channels,
      embedding_init=symmetric_uniform(0.001),
      key=keys[7],
    )
    self.target_embedding = primitives.Embed(
      num_embeddings=cfg.max_num_elements,
      features=cfg.edge_channels,
      embedding_init=symmetric_uniform(0.001),
      key=keys[8],
    )

    self.edge_degree_embedding = EdgeDegreeEmbedding(
      sphere_channels=cfg.sphere_channels,
      lmax=cfg.lmax,
      mmax=cfg.mmax,
      edge_channels_list=edge_channels_list,
      rescale_factor=5.0,
      m_size=mapping.m_size,
      key=keys[9],
    )

    self.blocks = tuple(
      UMABlockMoE(
        sphere_channels=cfg.sphere_channels,
        hidden_channels=cfg.hidden_channels,
        lmax=cfg.lmax,
        mmax=cfg.mmax,
        m_size=mapping.m_size,
        edge_channels_list=edge_channels_list,
        num_experts=cfg.num_experts,
        norm_type=cfg.norm_type,
        act_type=cfg.act_type,
        ff_type=cfg.ff_type,
        grid_resolution=cfg.grid_resolution,
        use_kernels=cfg.use_kernels,
        merged_mole=cfg.merged_mole,
        so2_block_gemm=cfg.so2_block_gemm,
        use_segment_mm_pallas=cfg.use_segment_mm_pallas,
        max_edge_segment_size=cfg.max_edge_segment_size,
        key=keys[10 + i],
      )
      for i in range(cfg.num_layers)
    )

    self.norm = get_normalization_layer(
      cfg.norm_type,
      lmax=cfg.lmax,
      num_channels=cfg.sphere_channels,
    )

  def __call__(
    self,
    positions: jnp.ndarray,
    atomic_numbers: jnp.ndarray,
    batch: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_distance_vec: jnp.ndarray,
    charge: jnp.ndarray,
    spin: jnp.ndarray,
    dataset_idx: jnp.ndarray | None = None,
  ) -> Dict[str, jnp.ndarray]:
    cfg = self.config
    num_atoms = positions.shape[0]

    atom_emb = self.sphere_embedding(atomic_numbers)

    chg_emb = self.charge_embedding(charge)
    spin_emb = self.spin_embedding(spin)

    if self.dataset_embedding is not None:
      if dataset_idx is None:
        raise ValueError(
          'This model has a dataset embedding; pass dataset_idx.'
        )
      dataset_emb = self.dataset_embedding(dataset_idx)
      csd_cat = jnp.concatenate([chg_emb, spin_emb, dataset_emb], axis=1)
    else:
      csd_cat = jnp.concatenate([chg_emb, spin_emb], axis=1)

    csd_mixed_emb = jax.nn.silu(self.mix_csd(csd_cat))

    if self.routing_mlp is None:
      expert_coefficients = jnp.zeros(
        (charge.shape[0], cfg.num_experts), dtype=positions.dtype
      )
    else:
      routing_input_parts = []

      if self.composition_embedding is not None:
        comp_per_atom = self.composition_embedding(atomic_numbers)
        # Mean-pool composition per system via index_reduce mean
        # include_self depends on model_version:
        #   v1.0: include_self=True -> mean = sum / (N + 1)
        #   v1.1+: include_self=False -> mean = sum / N
        num_systems = charge.shape[0]
        comp_sum = (
          jnp.zeros((num_systems, cfg.sphere_channels))
          .at[batch]
          .add(comp_per_atom)
        )
        atom_counts = jnp.zeros(num_systems).at[batch].add(1.0)
        if cfg.model_version < 1.05:
          composition = comp_sum / jnp.maximum(atom_counts[:, None] + 1.0, 1.0)
        else:
          composition = comp_sum / jnp.maximum(atom_counts[:, None], 1.0)

        routing_input_parts.append(composition)

      routing_input_parts.append(csd_mixed_emb)
      routing_input = jnp.concatenate(routing_input_parts, axis=-1)

      expert_coefficients = self.routing_mlp(routing_input, deterministic=True)
    edge_distance = jnp.linalg.norm(edge_distance_vec, axis=-1)
    dist_scaled = edge_distance / cfg.cutoff

    edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)

    edge_distance_embedding = self.distance_expansion(edge_distance)

    source_emb = self.source_embedding(atomic_numbers[edge_index[0]])
    target_emb = self.target_embedding(atomic_numbers[edge_index[1]])

    x_edge = jnp.concatenate(
      [edge_distance_embedding, source_emb, target_emb],
      axis=1,
    )

    euler_angles = init_edge_rot_euler_angles(edge_distance_vec)
    # Constants under a trace, not leaves
    with jax.ensure_compile_time_eval():
      to_m = create_coefficient_mapping(cfg.lmax, cfg.mmax).to_m
      Jd_list = load_jacobi_matrices_from_file(cfg.lmax)
    wigner = eulers_to_wigner(euler_angles, 0, cfg.lmax, Jd_list)
    wigner_inv = jnp.transpose(wigner, (0, 2, 1))

    if cfg.mmax != cfg.lmax:
      with jax.ensure_compile_time_eval():
        grid = create_so3_grid(
          cfg.lmax, cfg.lmax, resolution=cfg.grid_resolution, rescale=True
        )
        coefficient_index = coefficient_idx(grid.mapping, cfg.lmax, cfg.mmax)
      # PT: wigner.index_select(1, idx) → select rows only → [E, m_dim, l_dim]
      # PT: wigner_inv.index_select(2, idx) → select cols only → [E, l_dim, m_dim]
      wigner = wigner[:, coefficient_index, :]  # [E, m_dim, l_dim]
      wigner_inv = wigner_inv[:, :, coefficient_index]  # [E, l_dim, m_dim]

    wigner_and_M_mapping = jnp.einsum('mk,nkj->nmj', to_m, wigner)
    wigner_and_M_mapping_inv = jnp.einsum('njk,mk->njm', wigner_inv, to_m)

    x_message = jnp.zeros(
      (num_atoms, self.sph_feature_size, cfg.sphere_channels),
      dtype=positions.dtype,
    )
    x_message = x_message.at[:, 0, :].set(atom_emb)

    sys_node_embedding = csd_mixed_emb[batch]
    x_message = x_message.at[:, 0, :].add(
      sys_node_embedding.astype(x_message.dtype)
    )

    x_message = self.edge_degree_embedding(
      x_message,
      x_edge,
      edge_index,
      wigner_and_M_mapping_inv,
      edge_envelope,
    )

    # Edge batch indices (for MoE)
    edge_batch = batch[edge_index[1]]

    for i in range(cfg.num_layers):
      block = self.blocks[i]
      if cfg.activation_checkpointing:
        block = eqx.filter_checkpoint(block)
      x_message = block(
        x_message,
        x_edge,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        wigner,
        wigner_inv,
        edge_envelope,
        expert_coefficients,
        edge_batch,
        sys_node_embedding=sys_node_embedding,
      )
      x_message = balance_channels(
        x_message,
        charge,
        batch,
        self.charge_channel_start,
        self.charge_channel_end,
        target_offset=0.0,
      )
      x_message = balance_channels(
        x_message,
        spin,
        batch,
        self.spin_channel_start,
        self.spin_channel_end,
        target_offset=1.0,
      )

    x_message = self.norm(x_message)  # ty: ignore[call-non-callable]

    return {
      'node_embedding': x_message,
      'batch': batch,
      'expert_coefficients': expert_coefficients,
    }


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def cache_native(
  model_name: str,
  config: UMAMoEConfig,
  params: Dict[str, Any],
  metadata: ConversionMetadata,
  ckpt: Any,
) -> None:
  """Cache converted params as native .npz for fast future loading."""
  import json

  try:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    params_file = os.path.join(CHECKPOINT_DIR, f'{model_name}.npz')
    if os.path.exists(params_file):
      return  # Already cached

    # Flatten and save params
    flat = jax.tree.leaves_with_path(params)
    param_dict: Dict[str, Any] = {}
    for path, value in flat:
      key = '/'.join(
        str(p.key) if hasattr(p, 'key') else str(p.idx) for p in path
      )
      param_dict[key] = np.array(value)
    np.savez_compressed(params_file, **param_dict)

    # Save head params
    ema_sd = {
      k.replace('module.', '', 1): v for k, v in ckpt.ema_state_dict.items()
    }
    prefix = 'output_heads.energyandforcehead.head.energy_block.'
    head_dict = {}
    for idx in [0, 2, 4]:
      for suffix in ['weight', 'weights', 'bias']:
        key = f'{prefix}{idx}.{suffix}'
        if key in ema_sd:
          head_dict[f'energy_block_{idx}_{suffix}'] = ema_sd[key].cpu().numpy()
    if head_dict:
      np.savez_compressed(
        os.path.join(CHECKPOINT_DIR, f'{model_name}_head.npz'), **head_dict
      )

    # Save config
    heads_cfg = (
      ckpt.model_config.get('heads', {})
      if isinstance(ckpt.model_config, dict)
      else {}
    )
    head_dm = None
    if isinstance(heads_cfg, dict):
      for _, hcfg in heads_cfg.items():
        if isinstance(hcfg, dict):
          dm = hcfg.get('dataset_mapping', None)
          if dm and isinstance(dm, dict):
            head_dm = dict(dm)

    cfg_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
    cfg_dict = {
      k: (list(v) if isinstance(v, list | tuple) else v)
      for k, v in cfg_dict.items()
    }
    if head_dm:
      cfg_dict['head_dataset_mapping'] = head_dm
    with open(
      os.path.join(CHECKPOINT_DIR, f'{model_name}_config.json'), 'w'
    ) as f:
      json.dump(cfg_dict, f, indent=2)

  except Exception:
    pass  # Caching is best-effort


def load_native(
  model_name: str, head_dataset: str, directory: str = CHECKPOINT_DIR
) -> Tuple[UMAMoEConfig, Dict[str, Any], Dict[str, Any] | None] | None:
  """Load from pre-converted native .npz checkpoint (no torch needed)."""
  import json

  params_file = os.path.join(directory, f'{model_name}.npz')
  config_file = os.path.join(directory, f'{model_name}_config.json')
  head_file = os.path.join(directory, f'{model_name}_head.npz')

  if not os.path.exists(params_file) or not os.path.exists(config_file):
    return None

  # Load config
  with open(config_file) as f:
    cfg = json.load(f)

  config = UMAMoEConfig(
    max_num_elements=cfg['max_num_elements'],
    sphere_channels=cfg['sphere_channels'],
    lmax=cfg['lmax'],
    mmax=cfg['mmax'],
    num_layers=cfg['num_layers'],
    hidden_channels=cfg['hidden_channels'],
    cutoff=cfg['cutoff'],
    edge_channels=cfg['edge_channels'],
    num_distance_basis=cfg['num_distance_basis'],
    norm_type=cfg['norm_type'],
    act_type=cfg['act_type'],
    ff_type=cfg['ff_type'],
    chg_spin_emb_type=cfg['chg_spin_emb_type'],
    dataset_list=cfg.get('dataset_list', []),
    num_experts=cfg.get('num_experts', 32),
    use_composition_embedding=cfg.get('use_composition_embedding', True),
    model_version=1.0
    if model_name == 'uma-s-1p2'
    else cfg.get('model_version', 1.0),
    routing_hidden_channels=cfg.get('routing_hidden_channels', 64),
    charge_balanced_channels=tuple(
      cfg.get(
        'charge_balanced_channels',
        (0, 1, 2) if model_name == 'uma-s-1p2' else (),
      )
    ),
    spin_balanced_channels=tuple(cfg.get('spin_balanced_channels', ())),
    merged_mole=cfg.get('merged_mole', False),
    so2_block_gemm=cfg.get('so2_block_gemm', False),
  )

  # Load params — unflatten from key/value pairs
  data = np.load(params_file, allow_pickle=False)
  params = {'params': {}}
  for key in data.files:
    parts = key.split('/')
    # Skip the leading 'params' prefix
    if parts[0] == 'params':
      parts = parts[1:]
    d = params['params']
    for p in parts[:-1]:
      if p not in d:
        d[p] = {}
      d = d[p]
    d[parts[-1]] = jnp.array(data[key])

  # Load head params
  head_params = None
  if os.path.exists(head_file):
    hd = np.load(head_file, allow_pickle=False)
    head_mapping = cfg.get('head_dataset_mapping', None)

    # Check if MoE head (3D weights)
    w0_key = (
      'energy_block_0_weights'
      if 'energy_block_0_weights' in hd.files
      else 'energy_block_0_weight'
    )
    is_moe_head = w0_key in hd.files and hd[w0_key].ndim == 3

    if is_moe_head:
      # Select the correct dataset expert
      if head_mapping:
        sorted_ds = sorted(head_mapping.keys())
      else:
        sorted_ds = sorted(config.dataset_list or [])
      ds_idx = sorted_ds.index(head_dataset) if head_dataset in sorted_ds else 0

      def sel(key_base: str) -> jnp.ndarray | None:
        for suffix in ['weights', 'weight']:
          k = f'{key_base}_{suffix}'
          if k in hd.files:
            w = hd[k]
            if w.ndim == 3:
              return jnp.array(w[ds_idx].T)  # [out, in] -> [in, out]
            return jnp.array(w.T) if w.ndim == 2 else jnp.array(w)
        return None

      head_params = {
        'params': {
          'linear_0': {
            'kernel': sel('energy_block_0'),
            'bias': jnp.array(hd['energy_block_0_bias']),
          },
          'linear_1': {
            'kernel': sel('energy_block_2'),
            'bias': jnp.array(hd['energy_block_2_bias']),
          },
          'linear_2': {
            'kernel': sel('energy_block_4'),
            'bias': jnp.array(hd['energy_block_4_bias']),
          },
        }
      }
    else:
      # Standard head — transpose [out, in] -> [in, out]
      head_params = {
        'params': {
          'linear_0': {
            'kernel': jnp.array(hd['energy_block_0_weight'].T),
            'bias': jnp.array(hd['energy_block_0_bias']),
          },
          'linear_1': {
            'kernel': jnp.array(hd['energy_block_2_weight'].T),
            'bias': jnp.array(hd['energy_block_2_bias']),
          },
          'linear_2': {
            'kernel': jnp.array(hd['energy_block_4_weight'].T),
            'bias': jnp.array(hd['energy_block_4_bias']),
          },
        }
      }

  return config, params, head_params


def load_pretrained(
  checkpoint_path: str,
  head_dataset: str = 'omat',
) -> Tuple[UMAMoEConfig, Dict[str, Any], Dict[str, Any] | None]:
  """Load a pretrained UMA checkpoint for use with UMAMoEBackbone.

  This is the primary way to load pretrained models. It preserves the
  full MoE architecture — all expert weights, the routing MLP, and
  the composition embedding. The model runs expert mixing during the
  forward pass, so it works for any system without re-loading.

  Tries to load from pre-converted native .npz checkpoints first
  with no torch dependency. Falls back to converting from .pt if needed.

  Args:
      checkpoint_path: Path to a checkpoint file, or a model name
          'uma-s-1p1', 'uma-s-1p2' or 'uma-m-1p1'.
      head_dataset: Dataset for the energy head. For models with MoE heads
          such as uma-s-1p2, this selects which dataset's head expert to use.
          For single-head models, uma-s-1p1 and uma-m-1p1, this is ignored.

  Returns:
      Tuple of (config, params, head_params) where:
        - config: UMAMoEConfig matching the checkpoint
        - params: nested weight dict for UMAMoEBackbone, keyed by
          checkpoint path
        - head_params: nested weight dict for MLPEnergyHead, taken from the
          checkpoint's energy head weights

      Both dicts are grafted onto a constructed module with
      :func:`jax_md._nn.uma.pretrained.load_params_into_module`.

  Example:
      >>> from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
      >>> from jax_md._nn.uma.heads import MLPEnergyHead
      >>> from jax_md._nn.uma.pretrained import load_params_into_module
      >>> config, params, head_params = load_pretrained('uma-s-1p1')
      >>> model = load_params_into_module(
      ...     params, UMAMoEBackbone(config=config, key=backbone_key))
      >>> head = load_params_into_module(
      ...     head_params,
      ...     MLPEnergyHead(sphere_channels=config.sphere_channels,
      ...                   hidden_channels=config.hidden_channels,
      ...                   key=head_key))
      >>> emb = model(positions, Z, batch, edge_index,
      ...             edge_vec, charge, spin, dataset_idx)
      >>> energy = head(emb['node_embedding'], batch, 1)
  """
  from jax_md._nn.uma.pretrained import PRETRAINED_MODELS

  # Try native .npz checkpoint first (no torch needed)
  if checkpoint_path in PRETRAINED_MODELS:
    result = load_native(checkpoint_path, head_dataset)
    if result is not None:
      return result

  # Also try if checkpoint_path points to a directory with .npz files
  if os.path.isdir(checkpoint_path):
    name = os.path.basename(os.path.normpath(checkpoint_path))
    result = load_native(name, head_dataset, checkpoint_path)
    if result is not None:
      return result

  # Fall back to .pt conversion (requires torch)
  from jax_md._nn.uma.pretrained import (
    download_pretrained,
    load_checkpoint_raw,
    extract_config,
    convert_checkpoint,
  )

  original_name = checkpoint_path  # Save for caching
  if checkpoint_path in PRETRAINED_MODELS:
    checkpoint_path = download_pretrained(checkpoint_path)

  ckpt = load_checkpoint_raw(checkpoint_path)
  cfg_base = extract_config(ckpt)
  mc = (
    ckpt.model_config['backbone']
    if isinstance(ckpt.model_config, dict)
    else ckpt.model_config
  )

  def get(k: str, d: Any = None) -> Any:
    if isinstance(mc, dict):
      return mc.get(k, d)
    return getattr(mc, k, d)

  config = UMAMoEConfig(
    max_num_elements=cfg_base.max_num_elements,
    sphere_channels=cfg_base.sphere_channels,
    lmax=cfg_base.lmax,
    mmax=cfg_base.mmax,
    num_layers=cfg_base.num_layers,
    hidden_channels=cfg_base.hidden_channels,
    cutoff=cfg_base.cutoff,
    edge_channels=cfg_base.edge_channels,
    num_distance_basis=cfg_base.num_distance_basis,
    norm_type=cfg_base.norm_type,
    act_type=cfg_base.act_type,
    ff_type=cfg_base.ff_type,
    chg_spin_emb_type=cfg_base.chg_spin_emb_type,
    dataset_list=cfg_base.dataset_list,
    num_experts=get('num_experts', 32),
    use_composition_embedding=get('use_composition_embedding', True),
    model_version=float(get('model_version', 1.0)),
    charge_balanced_channels=tuple(get('charge_balanced_channels', ())),
    spin_balanced_channels=tuple(get('spin_balanced_channels', ())),
    merged_mole=get('merged_mole', False),
    use_kernels=bool(get('use_kernels', True)),
    so2_block_gemm=get('so2_block_gemm', False),
  )

  _, params, metadata = convert_checkpoint(
    checkpoint_path,
    use_ema=True,
  )

  # Infer routing_hidden_channels from converted weights
  p = params.get('params', params)
  if 'routing_mlp' in p and 'layers_0' in p['routing_mlp']:
    kernel = p['routing_mlp']['layers_0']['kernel']
    if hasattr(kernel, 'shape'):
      config = UMAMoEConfig(
        **{
          f: getattr(config, f)
          for f in config.__dataclass_fields__
          if f != 'routing_hidden_channels'
        },
        routing_hidden_channels=kernel.shape[1],
      )

  # Build head params from checkpoint's energy_block weights
  # Conversion produces keys like energy_block_0_kernel, energy_block_0_bias, etc.
  # For standard heads: 2D weights [out, in], already transposed to [in, out]
  # For MoE heads: 3D weights [num_ds_experts, out, in] — select by dataset
  head_params = None
  if metadata.head_params:
    hp = metadata.head_params
    w0 = hp.get('energy_block_0_kernel', None)

    # Detect MoE head (3D weights)
    is_moe_head = w0 is not None and w0.ndim == 3
    if is_moe_head:
      # MoE head: select expert for the requested dataset
      # DatasetSpecificMoEWrapper uses sorted dataset names from dataset_mapping
      # The number of experts in head weights tells us the true dataset count
      n_head_experts = w0.shape[0]

      # Try to get the full dataset list from the checkpoint's head config
      heads_cfg = ckpt.model_config.get('heads', {})
      head_ds = None
      if isinstance(heads_cfg, dict):
        for _, hcfg in heads_cfg.items():
          if isinstance(hcfg, dict):
            dm = hcfg.get('dataset_mapping', None)
            if dm and isinstance(dm, dict):
              head_ds = sorted(dm.keys())
            dn = hcfg.get('dataset_names', None)
            if dn and not head_ds:
              head_ds = sorted(
                list(dn) if not isinstance(dn, str) else ast.literal_eval(dn)
              )

      if head_ds is None or len(head_ds) != n_head_experts:
        # Fallback: infer from state dict dataset_emb_dict keys
        ema_keys = list(ckpt.ema_state_dict.keys())
        ds_from_keys = set()
        for k in ema_keys:
          if 'dataset_emb_dict.' in k:
            parts = k.split('.')
            for i, p in enumerate(parts):
              if p == 'dataset_emb_dict' and i + 1 < len(parts):
                ds_from_keys.add(parts[i + 1])
        if len(ds_from_keys) == n_head_experts:
          head_ds = sorted(ds_from_keys)
        else:
          head_ds = sorted(config.dataset_list or [])

      ds_idx = head_ds.index(head_dataset) if head_dataset in head_ds else 0

      def select_expert(w: jnp.ndarray | None) -> jnp.ndarray | None:
        if w is not None and w.ndim == 3:
          # w is [num_experts, out, in] — select expert and transpose to [in, out]
          return w[ds_idx].T
        return w

      head_params = {
        'params': {
          'linear_0': {
            'kernel': select_expert(hp.get('energy_block_0_kernel')),
            'bias': hp.get('energy_block_0_bias'),
          },
          'linear_1': {
            'kernel': select_expert(hp.get('energy_block_2_kernel')),
            'bias': hp.get('energy_block_2_bias'),
          },
          'linear_2': {
            'kernel': select_expert(hp.get('energy_block_4_kernel')),
            'bias': hp.get('energy_block_4_bias'),
          },
        }
      }
    else:
      # Standard head: 2D weights (already in correct format)
      head_params = {
        'params': {
          'linear_0': {
            'kernel': hp.get('energy_block_0_kernel'),
            'bias': hp.get('energy_block_0_bias'),
          },
          'linear_1': {
            'kernel': hp.get('energy_block_2_kernel'),
            'bias': hp.get('energy_block_2_bias'),
          },
          'linear_2': {
            'kernel': hp.get('energy_block_4_kernel'),
            'bias': hp.get('energy_block_4_bias'),
          },
        }
      }

    # Remove None entries
    for layer in list(head_params['params'].keys()):
      head_params['params'][layer] = {
        k: v for k, v in head_params['params'][layer].items() if v is not None
      }

  # Cache as native .npz for instant loading next time
  if original_name in PRETRAINED_MODELS:
    cache_native(original_name, config, params, metadata, ckpt)

  return config, params, head_params
