"""SO(2) convolution layers for message passing.

This module provides SO(2) convolution layers that operate on spherical
harmonic coefficients organized by order m.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

from typing import List, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax.nn import initializers

from jax_md._nn.uma.nn.radial import RadialMLP


class SO2MConv(nn.Module):
  """SO(2) convolution for a single order m.

  Performs an SO(2) convolution on features corresponding to +/- m,
  handling the real/imaginary pairing of spherical harmonic coefficients.

  Matches PyTorch SO2_m_Conv exactly:
  - Linear maps last dim of (E, 2, C) -> (E, 2, 2*out_half)
  - Reshape to (E, 4, out_half), split into 4 components
  - Complex multiply: real = r0 - i1, imag = r1 + i0

  Attributes:
      m: Order of the spherical harmonic coefficients.
      sphere_channels: Number of input spherical channels.
      m_output_channels: Number of output channels.
      lmax: Maximum degree l.
  """

  m: int
  sphere_channels: int
  m_output_channels: int
  lmax: int

  @nn.compact
  def __call__(self, x_m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply SO(2) convolution for order m.

    Args:
        x_m: Input features for order m, shape [num_edges, 2, num_coeffs * sphere_channels]
             where the second dimension is [real, imaginary].

    Returns:
        Tuple of (real_output, imag_output), each of shape
        [num_edges, num_coeffs, m_output_channels].
    """
    num_coefficients = self.lmax - self.m + 1

    # out_channels_half = m_output_channels * num_coefficients
    # (matching PyTorch: self.m_output_channels * (num_channels // self.sphere_channels))
    out_channels_half = self.m_output_channels * num_coefficients

    # Linear transformation: applied to last dim of 3D input (E, 2, C) -> (E, 2, 2*out_half)
    # PyTorch: Linear(num_channels, 2 * out_channels_half, bias=False)
    # PyTorch init: weight.data.mul_(1 / sqrt(2))
    fc = nn.Dense(
      features=2 * out_channels_half,
      use_bias=False,
      kernel_init=initializers.variance_scaling(
        scale=0.5,  # 1/sqrt(2) factor via variance scaling
        mode='fan_in',
        distribution='uniform',
      ),
      name='fc',
    )

    # Apply linear to last dim: (E, 2, num_channels) -> (E, 2, 2*out_half)
    x_m = fc(x_m)

    # Reshape: (E, 2, 2*out_half) -> (E, 4, out_half)
    batch_size = x_m.shape[0]
    x_m = x_m.reshape(batch_size, -1, out_channels_half)

    # Split into 4 components along dim 1
    x_r_0 = x_m[:, 0, :]  # (E, out_half)
    x_i_0 = x_m[:, 1, :]
    x_r_1 = x_m[:, 2, :]
    x_i_1 = x_m[:, 3, :]

    # Complex multiplication
    x_m_r = x_r_0 - x_i_1  # Real part
    x_m_i = x_r_1 + x_i_0  # Imaginary part

    # Reshape to (E, num_coefficients, m_output_channels)
    x_m_r = x_m_r.reshape(batch_size, num_coefficients, self.m_output_channels)
    x_m_i = x_m_i.reshape(batch_size, num_coefficients, self.m_output_channels)

    return x_m_r, x_m_i


class SO2Convolution(nn.Module):
  """SO(2) convolution block for all orders m.

  Performs SO(2) convolutions for all orders m from 0 to mmax.

  Attributes:
      sphere_channels: Number of input spherical channels.
      m_output_channels: Number of output channels per m.
      lmax: Maximum degree l.
      mmax: Maximum order m.
      m_size: List of number of coefficients for each m.
          For m>0 this includes both real and imaginary parts.
      internal_weights: If True, use internal weights instead of radial function.
      edge_channels_list: Channel dimensions for radial MLP (if not internal_weights).
      extra_m0_output_channels: Extra output channels for m=0 (for gating).
  """

  sphere_channels: int
  m_output_channels: int
  lmax: int
  mmax: int
  m_size: List[int]
  internal_weights: bool = True
  edge_channels_list: List[int] | None = None
  extra_m0_output_channels: int | None = None

  @nn.compact
  def __call__(
    self,
    x: jnp.ndarray,
    x_edge: jnp.ndarray,
  ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply SO(2) convolution for all orders.

    Args:
        x: Input features in m-major ordering, shape [num_edges, num_m_coeffs, sphere_channels].
        x_edge: Edge scalar features, shape [num_edges, edge_channels].

    Returns:
        Output features in m-major ordering. If extra_m0_output_channels is set,
        returns tuple of (output, extra_m0_features).
    """
    num_channels_m0 = (self.lmax + 1) * self.sphere_channels

    # Output channels for m=0
    m0_output_channels = self.m_output_channels * (self.lmax + 1)
    if self.extra_m0_output_channels is not None:
      m0_output_channels = m0_output_channels + self.extra_m0_output_channels

    # FC for m=0 (only real values, with bias)
    fc_m0 = nn.Dense(
      features=m0_output_channels,
      use_bias=True,
      name='fc_m0',
    )

    # Compute total radial channels needed
    # m=0 channels
    num_channels_rad = num_channels_m0
    # m>0 channels: each m uses num_coefficients * sphere_channels
    for m in range(1, self.mmax + 1):
      num_coeffs = self.lmax - m + 1
      num_channels_rad += num_coeffs * self.sphere_channels

    # Radial function (if using external weights)
    rad_func = None
    if not self.internal_weights:
      edge_channels = list(self.edge_channels_list) + [num_channels_rad]
      rad_func = RadialMLP(channels_list=edge_channels, name='rad_func')

    # m_size already includes both real and imaginary for m>0
    m_split_sizes = [self.m_size[m] for m in range(self.mmax + 1)]

    # Edge split sizes: one per m, based on input channels to fc
    edge_split_sizes = [num_channels_m0]  # m=0
    for m in range(1, self.mmax + 1):
      edge_split_sizes.append((self.lmax - m + 1) * self.sphere_channels)

    num_edges = x.shape[0]

    # Get radial weights if needed
    if rad_func is not None:
      x_edge_weights = rad_func(x_edge)
      x_edge_by_m = jnp.split(
        x_edge_weights,
        np.cumsum(edge_split_sizes[:-1]),
        axis=1,
      )
    else:
      x_edge_by_m = [None] * (self.mmax + 1)

    # Split input by m
    x_by_m = jnp.split(
      x,
      np.cumsum(m_split_sizes[:-1]),
      axis=1,
    )

    # Process m=0 (only real values)
    x_0 = x_by_m[0].reshape(num_edges, -1)
    if x_edge_by_m[0] is not None:
      x_0 = x_0 * x_edge_by_m[0]
    x_0 = fc_m0(x_0)

    # Extract extra m0 features if requested
    if self.extra_m0_output_channels is not None:
      x_0_extra = x_0[:, : self.extra_m0_output_channels]
      x_0 = x_0[:, self.extra_m0_output_channels :]

    out = [x_0.reshape(num_edges, -1, self.m_output_channels)]

    # Process m > 0
    for m in range(1, self.mmax + 1):
      # Reshape to (E, 2, num_coefficients * sphere_channels)
      x_m = x_by_m[m].reshape(num_edges, 2, -1)

      # Apply radial weighting (broadcast over real/imag dim)
      if x_edge_by_m[m] is not None:
        x_m = x_m * x_edge_by_m[m][:, None, :]

      # SO2 convolution for this m
      so2_m_conv = SO2MConv(
        m=m,
        sphere_channels=self.sphere_channels,
        m_output_channels=self.m_output_channels,
        lmax=self.lmax,
        name=f'so2_m_conv_{m}',
      )
      x_m_r, x_m_i = so2_m_conv(x_m)
      out.append(x_m_r)
      out.append(x_m_i)

    out = jnp.concatenate(out, axis=1)

    if self.extra_m0_output_channels is not None:
      return out, x_0_extra
    else:
      return out
