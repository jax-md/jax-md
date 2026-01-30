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

"""Embedding layers for UMA model.

This module provides embedding layers for atomic numbers, charge, spin,
and dataset identifiers.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.nn import initializers

from jax_md.ff.uma.nn.radial import RadialMLP


class AtomicEmbedding(nn.Module):
    """Embedding layer for atomic numbers.

    Attributes:
        num_elements: Maximum number of elements to embed.
        embedding_size: Dimension of the embedding.
    """
    num_elements: int
    embedding_size: int

    @nn.compact
    def __call__(self, atomic_numbers: jnp.ndarray) -> jnp.ndarray:
        """Embed atomic numbers.

        Args:
            atomic_numbers: Atomic numbers, shape [num_atoms].

        Returns:
            Embeddings, shape [num_atoms, embedding_size].
        """
        embedding = nn.Embed(
            num_embeddings=self.num_elements,
            features=self.embedding_size,
            name='embedding',
        )
        return embedding(atomic_numbers)


class ChgSpinEmbedding(nn.Module):
    """Embedding for charge or spin values.

    Supports multiple embedding types:
    - pos_emb: Positional (sinusoidal) embedding
    - lin_emb: Linear embedding
    - rand_emb: Random learned embedding

    Attributes:
        embedding_type: Type of embedding ('pos_emb', 'lin_emb', 'rand_emb').
        embedding_target: What to embed ('charge' or 'spin').
        embedding_size: Dimension of the embedding.
        trainable: Whether the embedding is trainable.
        scale: Scale factor for positional embedding.
    """
    embedding_type: Literal['pos_emb', 'lin_emb', 'rand_emb']
    embedding_target: Literal['charge', 'spin']
    embedding_size: int
    trainable: bool = False
    scale: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embed charge or spin values.

        Args:
            x: Values to embed, shape [batch].

        Returns:
            Embeddings, shape [batch, embedding_size].
        """
        if self.embedding_type == 'pos_emb':
            # Positional (sinusoidal) embedding
            W = self.param(
                'W',
                initializers.normal(stddev=self.scale),
                (self.embedding_size // 2,),
            )
            if not self.trainable:
                W = jax.lax.stop_gradient(W)

            x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
            emb = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

            # For spin, zero out null spin (0) embeddings
            if self.embedding_target == 'spin':
                zero_mask = (x == 0)[:, None]
                emb = jnp.where(zero_mask, 0.0, emb)

            return emb

        elif self.embedding_type == 'lin_emb':
            # Linear embedding
            linear = nn.Dense(
                features=self.embedding_size,
                use_bias=True,
                name='linear',
            )
            x_input = x.astype(jnp.float32)[:, None]

            # For spin, use sentinel value for null
            if self.embedding_target == 'spin':
                x_input = jnp.where(x[:, None] == 0, -100.0, x_input)

            out = linear(x_input)
            if not self.trainable:
                out = jax.lax.stop_gradient(out)
            return out

        elif self.embedding_type == 'rand_emb':
            # Random learned embedding
            if self.embedding_target == 'charge':
                # Charge can be -100 to 100, offset to make positive indices
                num_embeddings = 201
                x_idx = x + 100
            else:
                # Spin is 0 to 100
                num_embeddings = 101
                x_idx = x

            x_idx = x_idx.astype(jnp.int32)
            embedding = nn.Embed(
                num_embeddings=num_embeddings,
                features=self.embedding_size,
                name='embedding',
            )
            out = embedding(x_idx)
            if not self.trainable:
                out = jax.lax.stop_gradient(out)
            return out

        else:
            raise ValueError(f'Unknown embedding type: {self.embedding_type}')


class DatasetEmbedding(nn.Module):
    """Embedding for dataset identifiers.

    Each dataset gets its own learned embedding vector.

    Attributes:
        embedding_size: Dimension of the embedding.
        dataset_list: List of dataset names.
        trainable: Whether the embeddings are trainable.
    """
    embedding_size: int
    dataset_list: List[str]
    trainable: bool = False

    @nn.compact
    def __call__(self, dataset_names: List[str]) -> jnp.ndarray:
        """Embed dataset names.

        Args:
            dataset_names: List of dataset names, length [batch].

        Returns:
            Embeddings, shape [batch, embedding_size].
        """
        # Create embedding for each unique dataset
        embeddings = {}
        for dataset in self.dataset_list:
            emb = nn.Embed(
                num_embeddings=1,
                features=self.embedding_size,
                name=f'emb_{dataset}',
            )
            embeddings[dataset] = emb(jnp.array([0]))

        # Look up embeddings for each dataset in batch
        emb_list = []
        for name in dataset_names:
            # Handle dataset aliases
            if name in ['mptrj', 'salex'] and 'omat' in self.dataset_list:
                emb = embeddings['omat']
            elif name in embeddings:
                emb = embeddings[name]
            else:
                # Default to first dataset
                emb = embeddings[self.dataset_list[0]]
            emb_list.append(emb)

        out = jnp.concatenate(emb_list, axis=0)
        if not self.trainable:
            out = jax.lax.stop_gradient(out)
        return out


class EdgeDegreeEmbedding(nn.Module):
    """Embedding based on edge degree (neighbor aggregation).

    Aggregates edge features onto nodes using Wigner rotation.

    Attributes:
        sphere_channels: Number of spherical channels.
        lmax: Maximum degree l.
        mmax: Maximum order m.
        edge_channels_list: Channel dimensions for radial MLP.
        rescale_factor: Factor to rescale aggregated features.
        m_size: List of number of coefficients for each m.
    """
    sphere_channels: int
    lmax: int
    mmax: int
    edge_channels_list: List[int]
    rescale_factor: float
    m_size: List[int]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        x_edge: jnp.ndarray,
        edge_index: jnp.ndarray,
        wigner_inv: jnp.ndarray,
        edge_envelope: jnp.ndarray,
        node_offset: int = 0,
    ) -> jnp.ndarray:
        """Apply edge degree embedding.

        Args:
            x: Node features, shape [num_nodes, (lmax+1)^2, sphere_channels].
            x_edge: Edge scalar features, shape [num_edges, edge_channels].
            edge_index: Edge connectivity, shape [2, num_edges].
            wigner_inv: Inverse Wigner rotation matrices.
            edge_envelope: Edge envelope values, shape [num_edges, 1, 1].
            node_offset: Offset for node indices.

        Returns:
            Updated node features, same shape as x.
        """
        m_0_num_coefficients = self.m_size[0]
        m_all_num_coefficients = sum(self.m_size)

        # Radial MLP
        edge_channels = list(self.edge_channels_list)
        edge_channels.append(m_0_num_coefficients * self.sphere_channels)
        rad_func = RadialMLP(channels_list=edge_channels, name='rad_func')

        # Process edge features
        x_edge_m_0 = rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(-1, m_0_num_coefficients, self.sphere_channels)

        # Pad to full size
        num_edges = x_edge_m_0.shape[0]
        x_edge_embedding = jnp.zeros(
            (num_edges, m_all_num_coefficients, self.sphere_channels)
        )
        x_edge_embedding = x_edge_embedding.at[:, :m_0_num_coefficients, :].set(x_edge_m_0)

        # Apply inverse Wigner rotation
        x_edge_embedding = jnp.einsum('njk,nkc->njc', wigner_inv, x_edge_embedding)

        # Apply envelope
        x_edge_embedding = x_edge_embedding * edge_envelope

        # Aggregate onto target nodes
        target_indices = edge_index[1] - node_offset
        x_update = jax.ops.segment_sum(
            x_edge_embedding,
            target_indices,
            num_segments=x.shape[0],
        )

        return x + x_update / self.rescale_factor
