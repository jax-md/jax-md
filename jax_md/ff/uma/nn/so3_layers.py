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

"""SO(3) equivariant linear layers.

This module provides SO(3) equivariant linear layers that operate on
spherical harmonic representations.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.nn import initializers


class SO3Linear(nn.Module):
    """SO(3) equivariant linear layer.

    Applies a linear transformation that respects SO(3) equivariance by using
    separate weight matrices for each degree l.

    Attributes:
        out_features: Number of output channels.
        lmax: Maximum degree of spherical harmonics.
        use_bias: Whether to include bias (only applied to l=0 component).
    """
    out_features: int
    lmax: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, input_embedding: jnp.ndarray) -> jnp.ndarray:
        """Apply SO(3) linear transformation.

        Args:
            input_embedding: Input features of shape [batch, (lmax+1)^2, in_features].

        Returns:
            Output features of shape [batch, (lmax+1)^2, out_features].
        """
        in_features = input_embedding.shape[-1]

        # Initialize weight with shape [lmax+1, out_features, in_features]
        # Use uniform initialization similar to PyTorch
        bound = 1.0 / math.sqrt(in_features)
        weight = self.param(
            'weight',
            initializers.uniform(scale=2 * bound),
            (self.lmax + 1, self.out_features, in_features),
        )
        # Shift to [-bound, bound]
        weight = weight - bound

        # Create expand index to map l values to all m values
        expand_index = []
        for l in range(self.lmax + 1):
            for m in range(2 * l + 1):
                expand_index.append(l)
        expand_index = jnp.array(expand_index)

        # Expand weights: [lmax+1, out, in] -> [(lmax+1)^2, out, in]
        weight_expanded = weight[expand_index]

        # Apply linear transformation: [batch, (lmax+1)^2, in] x [(lmax+1)^2, out, in] -> [batch, (lmax+1)^2, out]
        out = jnp.einsum('bmi,moi->bmo', input_embedding, weight_expanded)

        # Add bias only to scalar (l=0) component
        if self.use_bias:
            bias = self.param(
                'bias',
                initializers.zeros,
                (self.out_features,),
            )
            # Add bias to l=0 component (index 0)
            out = out.at[:, 0, :].add(bias)

        return out
