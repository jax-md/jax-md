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

"""Radial basis functions for distance encoding.

This module provides radial basis functions used to encode interatomic
distances in the UMA model.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class PolynomialEnvelope(nn.Module):
  """Polynomial envelope function for smooth distance cutoff.

  Ensures contributions smoothly go to zero at the cutoff distance.

  Attributes:
      exponent: Exponent p for the polynomial (default 5).
  """

  exponent: int = 5

  def setup(self):
    p = float(self.exponent)
    self.a = -(p + 1) * (p + 2) / 2
    self.b = p * (p + 2)
    self.c = -p * (p + 1) / 2

  def __call__(self, d_scaled: jnp.ndarray) -> jnp.ndarray:
    """Apply polynomial envelope.

    Args:
        d_scaled: Scaled distances (d / cutoff), shape [num_edges].

    Returns:
        Envelope values, shape [num_edges]. Zero for d_scaled >= 1.
    """
    p = float(self.exponent)
    env_val = 1.0 + (d_scaled**p) * (
      self.a + d_scaled * (self.b + self.c * d_scaled)
    )
    return jnp.where(d_scaled < 1.0, env_val, 0.0)


class GaussianSmearing(nn.Module):
  """Gaussian basis expansion for distance encoding.

  Expands distances into a set of Gaussian basis functions.

  Attributes:
      start: Start of the distance range.
      stop: End of the distance range.
      num_gaussians: Number of Gaussian basis functions.
      basis_width_scalar: Scaling factor for basis width.
  """

  start: float = 0.0
  stop: float = 5.0
  num_gaussians: int = 50
  basis_width_scalar: float = 2.0

  def setup(self):
    offset = jnp.linspace(self.start, self.stop, self.num_gaussians)
    self.offset = offset
    self.coeff = -0.5 / (self.basis_width_scalar * (offset[1] - offset[0])) ** 2

  def __call__(self, dist: jnp.ndarray) -> jnp.ndarray:
    """Expand distances into Gaussian basis.

    Args:
        dist: Distances, shape [num_edges].

    Returns:
        Gaussian basis expansion, shape [num_edges, num_gaussians].
    """
    dist = dist.reshape(-1, 1) - self.offset.reshape(1, -1)
    return jnp.exp(self.coeff * dist**2)


class RadialMLP(nn.Module):
  """Multi-layer perceptron for radial features.

  Processes edge scalar features through linear layers with LayerNorm
  and SiLU activation.

  Attributes:
      channels_list: List of channel dimensions for each layer.
  """

  channels_list: Sequence[int]

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Apply radial MLP.

    Args:
        inputs: Input features, shape [num_edges, input_channels].

    Returns:
        Output features, shape [num_edges, channels_list[-1]].
    """
    x = inputs
    for i in range(1, len(self.channels_list)):
      x = nn.Dense(
        features=self.channels_list[i],
        use_bias=True,
        name=f'linear_{i}',
      )(x)

      # Apply LayerNorm and SiLU for all but the last layer
      if i < len(self.channels_list) - 1:
        x = nn.LayerNorm(name=f'norm_{i}')(x)
        x = nn.silu(x)

    return x
