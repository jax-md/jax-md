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

"""Activation functions for equivariant neural networks.

This module provides activation functions that respect SO(3) equivariance,
including gated activations and S2 activations.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class ScaledSiLU(nn.Module):
  """Scaled SiLU (Swish) activation function.

  Applies SiLU with a scaling factor to maintain unit variance.
  """

  scale_factor: float = 1.6791767923989418

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return nn.silu(x) * self.scale_factor


class GateActivation(nn.Module):
  """Gated activation for equivariant features.

  Uses scalar features to gate vector/tensor features, preserving equivariance.
  Scalar (l=0) features pass through SiLU, while higher-l features are
  multiplied by sigmoid-gated scalars.

  Attributes:
      lmax: Maximum degree l.
      mmax: Maximum order m.
      num_channels: Number of channels per feature.
      m_prime: If True, use m-prime ordering (for SO(2) convolution output).
  """

  lmax: int
  mmax: int
  num_channels: int
  m_prime: bool = False

  def setup(self):
    # Compute expand index based on lmax and mmax
    num_components = 0
    for l in range(1, self.lmax + 1):
      num_m_components = min(2 * l + 1, 2 * self.mmax + 1)
      num_components += num_m_components

    expand_index = []
    if self.m_prime:
      # m-prime ordering: l0m0, l1m0, l2m0, ..., l1m1, l2m1, ...
      # m=0: lmax components
      for l in range(1, self.lmax + 1):
        expand_index.append(l - 1)

      # m > 0: pairs of real/imaginary
      for m in range(1, self.mmax + 1):
        for l in range(m, self.lmax + 1):
          expand_index.append(l - 1)
        for l in range(m, self.lmax + 1):
          expand_index.append(l - 1)
    else:
      # Standard l-major ordering
      for l in range(1, self.lmax + 1):
        length = min(2 * l + 1, 2 * self.mmax + 1)
        for _ in range(length):
          expand_index.append(l - 1)

    self.expand_index = jnp.array(expand_index)

  def __call__(
    self,
    gating_scalars: jnp.ndarray,
    input_tensors: jnp.ndarray,
  ) -> jnp.ndarray:
    """Apply gated activation.

    Args:
        gating_scalars: Scalar features for gating, shape [batch, lmax * num_channels].
        input_tensors: Equivariant features, shape [batch, num_m_coeffs, num_channels].

    Returns:
        Activated features, shape [batch, num_m_coeffs, num_channels].
    """
    batch_size = gating_scalars.shape[0]

    # Apply sigmoid to gating scalars and reshape
    gating_scalars = nn.sigmoid(gating_scalars)
    gating_scalars = gating_scalars.reshape(
      batch_size, self.lmax, self.num_channels
    )

    # Expand gating scalars to match input tensor dimensions
    gating_scalars = gating_scalars[:, self.expand_index, :]

    # Split input into scalars (l=0) and higher-l components
    input_scalars = input_tensors[:, :1, :]
    input_vectors = input_tensors[:, 1:, :]

    # Apply SiLU to scalars
    output_scalars = nn.silu(input_scalars)

    # Gate the vector components
    output_vectors = input_vectors * gating_scalars

    return jnp.concatenate([output_scalars, output_vectors], axis=1)


class SeparableS2Activation(nn.Module):
  """Separable S2 activation using grid representation.

  Applies nonlinearity in grid space while preserving equivariance.
  Scalar features are processed with SiLU, while vector features are
  projected to a grid, activated, and projected back.

  Attributes:
      lmax: Maximum degree l.
      mmax: Maximum order m.
      to_grid_mat: Matrix for coefficients -> grid transformation.
      from_grid_mat: Matrix for grid -> coefficients transformation.
  """

  lmax: int
  mmax: int
  to_grid_mat: jnp.ndarray
  from_grid_mat: jnp.ndarray

  @nn.compact
  def __call__(
    self,
    input_scalars: jnp.ndarray,
    input_tensors: jnp.ndarray,
  ) -> jnp.ndarray:
    """Apply S2 activation.

    Args:
        input_scalars: Scalar features, shape [batch, 1, channels].
        input_tensors: Equivariant features, shape [batch, num_coeffs, channels].

    Returns:
        Activated features, shape [batch, num_coeffs, channels].
    """
    # Process scalars with SiLU
    output_scalars = nn.silu(input_scalars)
    output_scalars = output_scalars.reshape(
      output_scalars.shape[0], 1, output_scalars.shape[-1]
    )

    # Project to grid
    x_grid = jnp.einsum('iba,zic->zbac', self.to_grid_mat, input_tensors)

    # Apply SiLU in grid space
    x_grid = nn.silu(x_grid)

    # Project back to coefficients
    output_tensors = jnp.einsum('bai,zbac->zic', self.from_grid_mat, x_grid)

    # Combine scalars and higher-l components
    return jnp.concatenate(
      [output_scalars, output_tensors[:, 1:, :]],
      axis=1,
    )
