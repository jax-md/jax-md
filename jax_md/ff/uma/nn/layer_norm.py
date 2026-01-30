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

"""Equivariant normalization layers.

This module provides normalization layers that respect SO(3) equivariance,
including equivariant RMS normalization and layer normalization.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp


def get_l_to_all_m_expand_index(lmax: int) -> jnp.ndarray:
    """Create index mapping from l to all m values for that l."""
    expand_index = []
    for l in range(lmax + 1):
        for m in range(2 * l + 1):
            expand_index.append(l)
    return jnp.array(expand_index)


def get_normalization_layer(
    norm_type: Literal['layer_norm', 'layer_norm_sh', 'rms_norm_sh'],
    lmax: int,
    num_channels: int,
    eps: float = 1e-5,
    affine: bool = True,
) -> nn.Module:
    """Get normalization layer by name.

    Args:
        norm_type: Type of normalization ('layer_norm', 'layer_norm_sh', 'rms_norm_sh').
        lmax: Maximum degree l.
        num_channels: Number of channels.
        eps: Small constant for numerical stability.
        affine: Whether to include learnable affine parameters.

    Returns:
        Normalization module.
    """
    if norm_type == 'layer_norm':
        return EquivariantLayerNorm(
            lmax=lmax, num_channels=num_channels, eps=eps, affine=affine
        )
    elif norm_type == 'layer_norm_sh':
        return EquivariantLayerNormSH(
            lmax=lmax, num_channels=num_channels, eps=eps, affine=affine
        )
    elif norm_type == 'rms_norm_sh':
        return EquivariantRMSNorm(
            lmax=lmax, num_channels=num_channels, eps=eps, affine=affine
        )
    else:
        raise ValueError(f'Unknown norm_type: {norm_type}')


class EquivariantLayerNorm(nn.Module):
    """Equivariant layer normalization.

    Normalizes features per degree l to respect equivariance.

    Attributes:
        lmax: Maximum degree l.
        num_channels: Number of channels.
        eps: Small constant for numerical stability.
        affine: Whether to include learnable affine parameters.
    """
    lmax: int
    num_channels: int
    eps: float = 1e-5
    affine: bool = True

    @nn.compact
    def __call__(self, node_input: jnp.ndarray) -> jnp.ndarray:
        """Apply equivariant layer normalization.

        Args:
            node_input: Input features, shape [batch, (lmax+1)^2, num_channels].

        Returns:
            Normalized features, same shape as input.
        """
        if self.affine:
            affine_weight = self.param(
                'affine_weight',
                nn.initializers.ones,
                (self.lmax + 1, self.num_channels),
            )
            affine_bias = self.param(
                'affine_bias',
                nn.initializers.zeros,
                (self.num_channels,),
            )
        else:
            affine_weight = None
            affine_bias = None

        out = []
        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1

            feature = node_input[:, start_idx:start_idx + length, :]

            # For scalars (l=0), subtract mean
            if l == 0:
                feature_mean = jnp.mean(feature, axis=2, keepdims=True)
                feature = feature - feature_mean

            # Compute normalization factor
            feature_norm = jnp.mean(feature ** 2, axis=1, keepdims=True)
            feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)
            feature_norm = (feature_norm + self.eps) ** -0.5

            if self.affine:
                weight = affine_weight[l:l + 1, :].reshape(1, 1, -1)
                feature_norm = feature_norm * weight

            feature = feature * feature_norm

            if self.affine and l == 0:
                bias = affine_bias.reshape(1, 1, -1)
                feature = feature + bias

            out.append(feature)

        return jnp.concatenate(out, axis=1)


class EquivariantLayerNormSH(nn.Module):
    """Equivariant layer normalization for spherical harmonics.

    Normalizes l=0 separately and all l>0 components together.

    Attributes:
        lmax: Maximum degree l.
        num_channels: Number of channels.
        eps: Small constant for numerical stability.
        affine: Whether to include learnable affine parameters.
    """
    lmax: int
    num_channels: int
    eps: float = 1e-5
    affine: bool = True

    @nn.compact
    def __call__(self, node_input: jnp.ndarray) -> jnp.ndarray:
        """Apply equivariant layer normalization for spherical harmonics.

        Args:
            node_input: Input features, shape [batch, (lmax+1)^2, num_channels].

        Returns:
            Normalized features, same shape as input.
        """
        out = []

        # Normalize l=0 with standard layer norm
        feature_l0 = node_input[:, 0:1, :]
        norm_l0 = nn.LayerNorm(epsilon=self.eps, name='norm_l0')
        feature_l0 = norm_l0(feature_l0)
        out.append(feature_l0)

        # Normalize l>0 together
        if self.lmax > 0:
            if self.affine:
                affine_weight = self.param(
                    'affine_weight',
                    nn.initializers.ones,
                    (self.lmax, self.num_channels),
                )
            else:
                affine_weight = None

            num_m_components = (self.lmax + 1) ** 2
            feature = node_input[:, 1:num_m_components, :]

            # Compute feature norm across all l>0
            feature_norm = jnp.mean(feature ** 2, axis=1, keepdims=True)
            feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)
            feature_norm = (feature_norm + self.eps) ** -0.5

            for l in range(1, self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                feat = node_input[:, start_idx:start_idx + length, :]

                if self.affine:
                    weight = affine_weight[l - 1:l, :].reshape(1, 1, -1)
                    feat_scale = feature_norm * weight
                else:
                    feat_scale = feature_norm

                feat = feat * feat_scale
                out.append(feat)

        return jnp.concatenate(out, axis=1)


class EquivariantRMSNorm(nn.Module):
    """Equivariant RMS normalization for spherical harmonics.

    Uses RMS normalization across all spherical harmonic components with
    degree-balanced weighting.

    Attributes:
        lmax: Maximum degree l.
        num_channels: Number of channels.
        eps: Small constant for numerical stability.
        affine: Whether to include learnable affine parameters.
        centering: Whether to center l=0 features.
        std_balance_degrees: Whether to balance standard deviation across degrees.
    """
    lmax: int
    num_channels: int
    eps: float = 1e-5
    affine: bool = True
    centering: bool = True
    std_balance_degrees: bool = True

    def setup(self):
        self.expand_index = get_l_to_all_m_expand_index(self.lmax)

        if self.std_balance_degrees:
            # Compute degree-balanced weights
            balance_weight = []
            for l in range(self.lmax + 1):
                length = 2 * l + 1
                for _ in range(length):
                    balance_weight.append(1.0 / length)
            balance_weight = jnp.array(balance_weight) / (self.lmax + 1)
            self.balance_degree_weight = balance_weight.reshape(-1, 1)
        else:
            self.balance_degree_weight = None

    @nn.compact
    def __call__(self, node_input: jnp.ndarray) -> jnp.ndarray:
        """Apply equivariant RMS normalization.

        Args:
            node_input: Input features, shape [batch, (lmax+1)^2, num_channels].

        Returns:
            Normalized features, same shape as input.
        """
        if self.affine:
            affine_weight = self.param(
                'affine_weight',
                nn.initializers.ones,
                (self.lmax + 1, self.num_channels),
            )
            if self.centering:
                affine_bias = self.param(
                    'affine_bias',
                    nn.initializers.zeros,
                    (self.num_channels,),
                )
            else:
                affine_bias = None
        else:
            affine_weight = None
            affine_bias = None

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
        if self.std_balance_degrees and self.balance_degree_weight is not None:
            feature_norm = feature ** 2
            feature_norm = jnp.einsum(
                'nic,ia->nac',
                feature_norm,
                self.balance_degree_weight,
            )
        else:
            feature_norm = jnp.mean(feature ** 2, axis=1, keepdims=True)

        feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)
        feature_norm = (feature_norm + self.eps) ** -0.5

        # Apply affine transformation
        if self.affine:
            weight = affine_weight.reshape(1, self.lmax + 1, self.num_channels)
            weight = weight[:, self.expand_index, :]
            feature_norm = feature_norm * weight

        out = feature * feature_norm

        # Add bias to l=0
        if self.affine and self.centering and affine_bias is not None:
            out = out.at[:, 0:1, :].add(affine_bias.reshape(1, 1, -1))

        return out
