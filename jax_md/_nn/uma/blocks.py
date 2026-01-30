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

"""UMA model blocks.

This module provides the building blocks for the UMA model, including
Edgewise (message passing) and Atomwise (feed-forward) operations.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_md._nn.uma.nn.so2_layers import SO2Convolution
from jax_md._nn.uma.nn.so3_layers import SO3Linear
from jax_md._nn.uma.nn.activation import (
  GateActivation,
  SeparableS2Activation,
)
from jax_md._nn.uma.nn.layer_norm import get_normalization_layer
from jax_md._nn.uma.nn.radial import PolynomialEnvelope


class Edgewise(nn.Module):
  """Edgewise (message passing) module.

  Performs SO(2) convolutions to pass messages along edges.

  Attributes:
      sphere_channels: Number of spherical channels.
      hidden_channels: Number of hidden channels.
      lmax: Maximum degree l.
      mmax: Maximum order m.
      edge_channels_list: Channel dimensions for edge features.
      m_size: List of number of coefficients for each m.
      cutoff: Distance cutoff.
      act_type: Type of activation ('gate' or 's2').
  """

  sphere_channels: int
  hidden_channels: int
  lmax: int
  mmax: int
  edge_channels_list: List[int]
  m_size: List[int]
  cutoff: float
  act_type: Literal['gate', 's2'] = 'gate'
  to_grid_mat: Optional[jnp.ndarray] = None
  from_grid_mat: Optional[jnp.ndarray] = None

  @nn.compact
  def __call__(
    self,
    x: jnp.ndarray,
    x_edge: jnp.ndarray,
    edge_distance: jnp.ndarray,
    edge_index: jnp.ndarray,
    wigner_and_M_mapping: jnp.ndarray,
    wigner_and_M_mapping_inv: jnp.ndarray,
    edge_envelope: jnp.ndarray,
    node_offset: int = 0,
  ) -> jnp.ndarray:
    """Apply edgewise message passing.

    Args:
        x: Node features, shape [num_nodes, num_m_coeffs, sphere_channels].
        x_edge: Edge scalar features, shape [num_edges, edge_channels].
        edge_distance: Edge distances, shape [num_edges].
        edge_index: Edge connectivity, shape [2, num_edges].
        wigner_and_M_mapping: Forward Wigner rotation + M mapping.
        wigner_and_M_mapping_inv: Inverse Wigner rotation + M mapping.
        edge_envelope: Edge envelope values, shape [num_edges, 1, 1].
        node_offset: Offset for node indices.

    Returns:
        Message contribution to add to nodes, shape [num_nodes, ...].
    """
    # Determine extra output channels for gating
    if self.act_type == 'gate':
      extra_m0_output_channels = self.lmax * self.hidden_channels
    else:
      extra_m0_output_channels = self.hidden_channels

    # First SO2 convolution
    so2_conv_1 = SO2Convolution(
      sphere_channels=2 * self.sphere_channels,  # Concatenated source + target
      m_output_channels=self.hidden_channels,
      lmax=self.lmax,
      mmax=self.mmax,
      m_size=self.m_size,
      internal_weights=False,
      edge_channels_list=self.edge_channels_list,
      extra_m0_output_channels=extra_m0_output_channels,
      name='so2_conv_1',
    )

    # Second SO2 convolution
    so2_conv_2 = SO2Convolution(
      sphere_channels=self.hidden_channels,
      m_output_channels=self.sphere_channels,
      lmax=self.lmax,
      mmax=self.mmax,
      m_size=self.m_size,
      internal_weights=True,
      edge_channels_list=None,
      extra_m0_output_channels=None,
      name='so2_conv_2',
    )

    # Activation
    if self.act_type == 'gate':
      act = GateActivation(
        lmax=self.lmax,
        mmax=self.mmax,
        num_channels=self.hidden_channels,
        m_prime=True,
        name='act',
      )
    else:
      act = SeparableS2Activation(
        lmax=self.lmax,
        mmax=self.mmax,
        to_grid_mat=self.to_grid_mat,
        from_grid_mat=self.from_grid_mat,
        name='act',
      )

    # Envelope function
    envelope = PolynomialEnvelope(exponent=5, name='envelope')

    # Get source and target node features
    x_source = x[edge_index[0]]
    x_target = x[edge_index[1]]

    # Concatenate source and target features
    x_message = jnp.concatenate([x_source, x_target], axis=2)

    # Rotate to edge-aligned frame
    x_message = jnp.einsum('nmj,njc->nmc', wigner_and_M_mapping, x_message)

    # First SO2 convolution
    x_message, x_0_gating = so2_conv_1(x_message, x_edge)

    # Activation (m-prime)
    x_message = act(x_0_gating, x_message)

    # Second SO2 convolution
    x_message = so2_conv_2(x_message, x_edge)

    # Apply envelope
    x_message = x_message * edge_envelope

    # Rotate back to global frame
    x_message = jnp.einsum('njm,nmc->njc', wigner_and_M_mapping_inv, x_message)

    # Aggregate messages onto target nodes
    target_indices = edge_index[1] - node_offset
    new_embedding = jax.ops.segment_sum(
      x_message,
      target_indices,
      num_segments=x.shape[0],
    )

    return new_embedding


class SpectralAtomwise(nn.Module):
  """Spectral atomwise feed-forward module.

  Uses SO(3) linear layers with gated activation.

  Attributes:
      sphere_channels: Number of spherical channels.
      hidden_channels: Number of hidden channels.
      lmax: Maximum degree l.
      mmax: Maximum order m.
  """

  sphere_channels: int
  hidden_channels: int
  lmax: int
  mmax: int

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply spectral atomwise FFN.

    Args:
        x: Node features, shape [num_nodes, (lmax+1)^2, sphere_channels].

    Returns:
        Updated features, same shape.
    """
    # Scalar MLP for gating
    scalar_input = x[:, 0:1, :].squeeze(1)
    scalar_mlp = nn.Sequential(
      [
        nn.Dense(self.lmax * self.hidden_channels),
        nn.silu,
      ]
    )
    gating_scalars = scalar_mlp(scalar_input)

    # First SO3 linear
    so3_linear_1 = SO3Linear(
      out_features=self.hidden_channels,
      lmax=self.lmax,
      name='so3_linear_1',
    )
    x = so3_linear_1(x)

    # Gate activation
    act = GateActivation(
      lmax=self.lmax,
      mmax=self.lmax,
      num_channels=self.hidden_channels,
      name='act',
    )
    x = act(gating_scalars, x)

    # Second SO3 linear
    so3_linear_2 = SO3Linear(
      out_features=self.sphere_channels,
      lmax=self.lmax,
      name='so3_linear_2',
    )
    x = so3_linear_2(x)

    return x


class GridAtomwise(nn.Module):
  """Grid-based atomwise feed-forward module.

  Projects to grid, applies MLP, projects back.

  Attributes:
      sphere_channels: Number of spherical channels.
      hidden_channels: Number of hidden channels.
      lmax: Maximum degree l.
      mmax: Maximum order m.
      to_grid_mat: Matrix for coefficients -> grid transformation.
      from_grid_mat: Matrix for grid -> coefficients transformation.
  """

  sphere_channels: int
  hidden_channels: int
  lmax: int
  mmax: int
  to_grid_mat: jnp.ndarray
  from_grid_mat: jnp.ndarray

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply grid atomwise FFN.

    Args:
        x: Node features, shape [num_nodes, (lmax+1)^2, sphere_channels].

    Returns:
        Updated features, same shape.
    """
    # Grid MLP
    grid_mlp = nn.Sequential(
      [
        nn.Dense(self.hidden_channels, use_bias=False),
        nn.silu,
        nn.Dense(self.hidden_channels, use_bias=False),
        nn.silu,
        nn.Dense(self.sphere_channels, use_bias=False),
      ]
    )

    # Project to grid
    x_grid = jnp.einsum('bai,zic->zbac', self.to_grid_mat, x)

    # Apply MLP in grid space
    x_grid = grid_mlp(x_grid)

    # Project back to coefficients
    x = jnp.einsum('bai,zbac->zic', self.from_grid_mat, x_grid)

    return x


class UMABlock(nn.Module):
  """UMA transformer-like block.

  Combines edgewise (message passing) and atomwise (FFN) operations
  with residual connections and normalization.

  Attributes:
      sphere_channels: Number of spherical channels.
      hidden_channels: Number of hidden channels.
      lmax: Maximum degree l.
      mmax: Maximum order m.
      m_size: List of number of coefficients for each m.
      edge_channels_list: Channel dimensions for edge features.
      cutoff: Distance cutoff.
      norm_type: Type of normalization.
      act_type: Type of activation for edgewise.
      ff_type: Type of feed-forward ('spectral' or 'grid').
      to_grid_mat: Matrix for coefficients -> grid transformation.
      from_grid_mat: Matrix for grid -> coefficients transformation.
  """

  sphere_channels: int
  hidden_channels: int
  lmax: int
  mmax: int
  m_size: List[int]
  edge_channels_list: List[int]
  cutoff: float
  norm_type: str = 'rms_norm_sh'
  act_type: Literal['gate', 's2'] = 'gate'
  ff_type: Literal['spectral', 'grid'] = 'grid'
  to_grid_mat: Optional[jnp.ndarray] = None
  from_grid_mat: Optional[jnp.ndarray] = None

  @nn.compact
  def __call__(
    self,
    x: jnp.ndarray,
    x_edge: jnp.ndarray,
    edge_distance: jnp.ndarray,
    edge_index: jnp.ndarray,
    wigner_and_M_mapping: jnp.ndarray,
    wigner_and_M_mapping_inv: jnp.ndarray,
    edge_envelope: jnp.ndarray,
    sys_node_embedding: Optional[jnp.ndarray] = None,
    node_offset: int = 0,
  ) -> jnp.ndarray:
    """Apply UMA block.

    Args:
        x: Node features, shape [num_nodes, (lmax+1)^2, sphere_channels].
        x_edge: Edge scalar features, shape [num_edges, edge_channels].
        edge_distance: Edge distances, shape [num_edges].
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

    # Pre-normalization
    norm_1 = get_normalization_layer(
      self.norm_type,
      lmax=self.lmax,
      num_channels=self.sphere_channels,
    )
    x = norm_1(x)

    # Add system embedding to scalars
    if sys_node_embedding is not None:
      x = x.at[:, 0, :].add(sys_node_embedding)

    # Edgewise message passing
    edgewise = Edgewise(
      sphere_channels=self.sphere_channels,
      hidden_channels=self.hidden_channels,
      lmax=self.lmax,
      mmax=self.mmax,
      edge_channels_list=self.edge_channels_list,
      m_size=self.m_size,
      cutoff=self.cutoff,
      act_type=self.act_type,
      to_grid_mat=self.to_grid_mat,
      from_grid_mat=self.from_grid_mat,
      name='edge_wise',
    )
    x = edgewise(
      x,
      x_edge,
      edge_distance,
      edge_index,
      wigner_and_M_mapping,
      wigner_and_M_mapping_inv,
      edge_envelope,
      node_offset,
    )
    x = x + x_res

    # Second residual block: Atomwise
    x_res = x

    # Pre-normalization
    norm_2 = get_normalization_layer(
      self.norm_type,
      lmax=self.lmax,
      num_channels=self.sphere_channels,
    )
    x = norm_2(x)

    # Atomwise FFN
    if self.ff_type == 'spectral':
      atomwise = SpectralAtomwise(
        sphere_channels=self.sphere_channels,
        hidden_channels=self.hidden_channels,
        lmax=self.lmax,
        mmax=self.mmax,
        name='atom_wise',
      )
    else:
      atomwise = GridAtomwise(
        sphere_channels=self.sphere_channels,
        hidden_channels=self.hidden_channels,
        lmax=self.lmax,
        mmax=self.mmax,
        to_grid_mat=self.to_grid_mat,
        from_grid_mat=self.from_grid_mat,
        name='atom_wise',
      )
    x = atomwise(x)
    x = x + x_res

    return x
