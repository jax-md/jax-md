"""Activation functions for equivariant neural networks.

This module provides activation functions that respect SO(3) equivariance,
including gated activations and S2 activations.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations


import flax.linen as nn
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

  When ``mapping_to_m`` is provided the input is assumed to be in m-major
  ordering (the output of SO(2) convolutions) and is converted to l-major
  before the grid projection and back to m-major afterwards.  This matches
  the PyTorch ``SeparableS2Activation_M`` variant.

  Attributes:
      lmax: Maximum degree l.
      mmax: Maximum order m.
      to_grid_mat: Matrix for coefficients -> grid transformation (l-major).
      from_grid_mat: Matrix for grid -> coefficients transformation (l-major).
      mapping_to_m: Permutation matrix from l-major to m-major ordering.
          When set, the transpose is used to convert m-major inputs to
          l-major before the grid transform.
  """

  lmax: int
  mmax: int
  to_grid_mat: jnp.ndarray
  from_grid_mat: jnp.ndarray
  mapping_to_m: jnp.ndarray | None = None

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

    # Convert from m-major to l-major if needed (SO(2) conv output is m-major
    # but grid matrices are in l-major ordering)
    if self.mapping_to_m is not None:
      from_m = self.mapping_to_m.T
      input_tensors = jnp.einsum('mk,zkc->zmc', from_m, input_tensors)

    # Project to grid
    x_grid = jnp.einsum('bai,zic->zbac', self.to_grid_mat, input_tensors)

    # Apply SiLU in grid space
    x_grid = nn.silu(x_grid)

    # Project back to coefficients
    output_tensors = jnp.einsum('bai,zbac->zic', self.from_grid_mat, x_grid)

    # Convert from l-major back to m-major if needed
    if self.mapping_to_m is not None:
      output_tensors = jnp.einsum(
        'mk,zkc->zmc', self.mapping_to_m, output_tensors
      )

    # Combine scalars and higher-l components
    return jnp.concatenate(
      [output_scalars, output_tensors[:, 1:, :]],
      axis=1,
    )
