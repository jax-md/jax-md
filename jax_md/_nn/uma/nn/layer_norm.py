"""Equivariant normalization layers.

This module provides normalization layers that respect SO(3) equivariance,
including equivariant RMS normalization and layer normalization.

Ported from FairChem's UMA implementation.
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from jax_md import util as jmd_util
from jax_md._nn.uma.nn import primitives

Array = jmd_util.Array


def get_l_to_all_m_expand_index(lmax: int) -> jnp.ndarray:
  """Create index mapping from l to all m values for that l."""
  expand_index = []
  for l in range(lmax + 1):
    for m in range(2 * l + 1):
      expand_index.append(l)
  return jnp.array(expand_index)


def get_normalization_layer(
  norm_type: str,
  lmax: int,
  num_channels: int,
  eps: float = 1e-5,
  affine: bool = True,
) -> eqx.Module:
  """Get normalization layer by name.

  Args:
      norm_type: One of 'layer_norm', 'layer_norm_sh' or 'rms_norm_sh'.
      lmax: Maximum degree l.
      num_channels: Number of channels.
      eps: Small constant for numerical stability.
      affine: Whether to include learnable affine parameters.

  Returns:
      Normalization module.
  """
  kwargs: dict[str, Any] = dict(
    lmax=lmax, num_channels=num_channels, eps=eps, affine=affine
  )

  if norm_type == 'layer_norm':
    return EquivariantLayerNorm(**kwargs)
  elif norm_type == 'layer_norm_sh':
    return EquivariantLayerNormSH(**kwargs)
  elif norm_type == 'rms_norm_sh':
    return EquivariantRMSNorm(**kwargs)
  raise ValueError(f'Unknown norm_type: {norm_type}')


class EquivariantLayerNorm(eqx.Module):
  """Equivariant layer normalization.

  Normalizes features per degree l to respect equivariance.

  Attributes:
      affine_weight: Per-degree scale of shape [lmax+1, num_channels], or None.
      affine_bias: Shift for l=0 of shape [num_channels], or None.
      lmax: Maximum degree l.
      eps: Small constant for numerical stability.
  """

  affine_weight: Array | None
  affine_bias: Array | None
  lmax: int = eqx.field(static=True)
  eps: float = eqx.field(static=True)

  def __init__(
    self,
    lmax: int,
    num_channels: int,
    eps: float = 1e-5,
    affine: bool = True,
  ):
    self.lmax = lmax
    self.eps = eps

    if affine:
      self.affine_weight = jnp.ones((lmax + 1, num_channels))
      self.affine_bias = jnp.zeros((num_channels,))
    else:
      self.affine_weight = None
      self.affine_bias = None

  def __call__(self, node_input: Array) -> Array:
    """Apply equivariant layer normalization.

    Args:
        node_input: Input features, shape [batch, (lmax+1)^2, num_channels].

    Returns:
        Normalized features, same shape as input.
    """
    out = []
    for l in range(self.lmax + 1):
      start_idx = l**2
      length = 2 * l + 1

      feature = node_input[:, start_idx : start_idx + length, :]

      # For scalars (l=0), subtract mean
      if l == 0:
        feature_mean = jnp.mean(feature, axis=2, keepdims=True)
        feature = feature - feature_mean

      # Compute normalization factor
      feature_norm = jnp.mean(feature**2, axis=1, keepdims=True)
      feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)
      feature_norm = (feature_norm + self.eps) ** -0.5

      if self.affine_weight is not None:
        weight = self.affine_weight[l : l + 1, :].reshape(1, 1, -1)
        feature_norm = feature_norm * weight

      feature = feature * feature_norm

      if self.affine_bias is not None and l == 0:
        bias = self.affine_bias.reshape(1, 1, -1)
        feature = feature + bias

      out.append(feature)

    return jnp.concatenate(out, axis=1)


class EquivariantLayerNormSH(eqx.Module):
  """Equivariant layer normalization for spherical harmonics.

  Normalizes l=0 separately and all l>0 components together.

  Attributes:
      norm_l0: Standard layer normalization applied to the l=0 component.
      affine_weight: Per-degree scale of shape [lmax, num_channels], or None.
      lmax: Maximum degree l.
      eps: Small constant for numerical stability.
  """

  norm_l0: primitives.LayerNorm
  affine_weight: Array | None
  lmax: int = eqx.field(static=True)
  eps: float = eqx.field(static=True)

  def __init__(
    self,
    lmax: int,
    num_channels: int,
    eps: float = 1e-5,
    affine: bool = True,
  ):
    self.lmax = lmax
    self.eps = eps

    self.norm_l0 = primitives.LayerNorm(num_channels, eps=eps)

    if lmax > 0 and affine:
      self.affine_weight = jnp.ones((lmax, num_channels))
    else:
      self.affine_weight = None

  def __call__(self, node_input: Array) -> Array:
    """Apply equivariant layer normalization for spherical harmonics.

    Args:
        node_input: Input features, shape [batch, (lmax+1)^2, num_channels].

    Returns:
        Normalized features, same shape as input.
    """
    out = []

    # Normalize l=0 with standard layer norm
    feature_l0 = node_input[:, 0:1, :]
    feature_l0 = self.norm_l0(feature_l0)
    out.append(feature_l0)

    # Normalize l>0 together
    if self.lmax > 0:
      num_m_components = (self.lmax + 1) ** 2
      feature = node_input[:, 1:num_m_components, :]

      # Compute feature norm across all l>0
      feature_norm = jnp.mean(feature**2, axis=1, keepdims=True)
      feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)
      feature_norm = (feature_norm + self.eps) ** -0.5

      for l in range(1, self.lmax + 1):
        start_idx = l**2
        length = 2 * l + 1
        feat = node_input[:, start_idx : start_idx + length, :]

        if self.affine_weight is not None:
          weight = self.affine_weight[l - 1 : l, :].reshape(1, 1, -1)
          feat_scale = feature_norm * weight
        else:
          feat_scale = feature_norm

        feat = feat * feat_scale
        out.append(feat)

    return jnp.concatenate(out, axis=1)


class EquivariantRMSNorm(eqx.Module):
  """Equivariant RMS normalization for spherical harmonics.

  Uses RMS normalization across all spherical harmonic components with
  degree-balanced weighting.

  Attributes:
      affine_weight: Per-degree scale of shape [lmax+1, num_channels], or None.
      affine_bias: Shift for l=0 of shape [num_channels], or None.
      lmax: Maximum degree l.
      eps: Small constant for numerical stability.
      centering: Whether to center l=0 features.
      std_balance_degrees: Whether to balance standard deviation across degrees.
  """

  affine_weight: Array | None
  affine_bias: Array | None
  lmax: int = eqx.field(static=True)
  eps: float = eqx.field(static=True)
  centering: bool = eqx.field(static=True)
  std_balance_degrees: bool = eqx.field(static=True)

  def __init__(
    self,
    lmax: int,
    num_channels: int,
    eps: float = 1e-5,
    affine: bool = True,
    centering: bool = True,
    std_balance_degrees: bool = True,
  ):
    self.lmax = lmax
    self.eps = eps
    self.centering = centering
    self.std_balance_degrees = std_balance_degrees

    if affine:
      self.affine_weight = jnp.ones((lmax + 1, num_channels))
      if centering:
        self.affine_bias = jnp.zeros((num_channels,))
      else:
        self.affine_bias = None
    else:
      self.affine_weight = None
      self.affine_bias = None

  def __call__(self, node_input: Array) -> Array:
    """Apply equivariant RMS normalization.

    Args:
        node_input: Input features, shape [batch, (lmax+1)^2, num_channels].

    Returns:
        Normalized features, same shape as input.
    """
    feature = node_input

    # Center l=0 features
    if self.centering:
      feature_l0 = feature[:, 0:1, :]
      feature_l0_mean = jnp.mean(feature_l0, axis=2, keepdims=True)
      feature_l0 = feature_l0 - feature_l0_mean
      feature = jnp.concatenate(
        [feature_l0, feature[:, 1:, :]],
        axis=1,
      )

    # Compute RMS norm
    if self.std_balance_degrees:
      balance_weight = []
      for l in range(self.lmax + 1):
        length = 2 * l + 1
        for _ in range(length):
          balance_weight.append(1.0 / length)
      balance_weight = jnp.array(balance_weight) / (self.lmax + 1)
      feature_norm = feature**2
      feature_norm = jnp.einsum(
        'nic,ia->nac',
        feature_norm,
        balance_weight.reshape(-1, 1),
      )
    else:
      feature_norm = jnp.mean(feature**2, axis=1, keepdims=True)

    feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)
    feature_norm = (feature_norm + self.eps) ** -0.5

    # Apply affine transformation
    if self.affine_weight is not None:
      weight = self.affine_weight[None]
      weight = weight[:, get_l_to_all_m_expand_index(self.lmax), :]
      feature_norm = feature_norm * weight

    out = feature * feature_norm

    # Add bias to l=0
    if self.affine_bias is not None:
      out = out.at[:, 0:1, :].add(self.affine_bias.reshape(1, 1, -1))

    return out
