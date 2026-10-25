"""Embedding layers for UMA model.

This module provides embedding layers for atomic numbers, charge, spin,
and dataset identifiers.

Ported from FairChem's UMA implementation.
"""

from typing import Callable, List, Literal, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_md import util as jmd_util
from jax_md._nn.uma.nn import primitives
from jax_md._nn.uma.nn.radial import RadialMLP

Array = jmd_util.Array


# Aliases: these datasets map to 'omat'
DATASET_ALIASES = {
  'mptrj': 'omat',
  'salex': 'omat',
}


def symmetric_uniform(scale: float) -> Callable[..., Array]:
  """Initializer producing values in [-scale, scale), matching PyTorch's
  ``nn.init.uniform_(-scale, scale)``."""

  def init(
    key: Array, shape: Sequence[int], dtype: jax.typing.DTypeLike = jnp.float32
  ) -> Array:
    return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

  return init


def dataset_name_to_idx(name: str, dataset_list: List[str] | None) -> int:
  """Convert dataset name to integer index; call outside JIT.

  Args:
      name: Dataset name string.
      dataset_list: List of known dataset names.

  Returns:
      Integer index into dataset_list.
  """
  resolved = DATASET_ALIASES.get(name, name)
  if dataset_list and resolved in dataset_list:
    return dataset_list.index(resolved)
  return 0  # Default to first dataset


def dataset_names_to_indices(
  names: List[str], dataset_list: List[str] | None
) -> jnp.ndarray:
  """Convert a list of dataset names to integer indices; call outside JIT.

  Args:
      names: List of dataset name strings.
      dataset_list: List of known dataset names.

  Returns:
      Integer array of shape [len(names)].
  """
  return jnp.array([dataset_name_to_idx(n, dataset_list) for n in names])


class AtomicEmbedding(eqx.Module):
  """Embedding layer for atomic numbers.

  Attributes:
      embedding: Lookup table of shape [num_elements, embedding_size].
  """

  embedding: primitives.Embed

  def __init__(
    self,
    num_elements: int,
    embedding_size: int,
    *,
    key: Array,
  ):
    self.embedding = primitives.Embed(num_elements, embedding_size, key=key)

  def __call__(self, atomic_numbers: jnp.ndarray) -> jnp.ndarray:
    """Embed atomic numbers.

    Args:
        atomic_numbers: Atomic numbers, shape [num_atoms].

    Returns:
        Embeddings, shape [num_atoms, embedding_size].
    """
    return self.embedding(atomic_numbers)


class ChgSpinEmbedding(eqx.Module):
  """Embedding for charge or spin values.

  Supports multiple embedding types:
  - pos_emb: Positional (sinusoidal) embedding
  - lin_emb: Linear embedding
  - rand_emb: Random learned embedding

  Attributes:
      embedding_type: Type of embedding ('pos_emb', 'lin_emb', 'rand_emb').
      embedding_target: What to embed ('charge' or 'spin').
      trainable: Whether the embedding is trainable.
  """

  W: Array | None
  linear: primitives.Dense | None
  embedding: primitives.Embed | None
  embedding_type: str = eqx.field(static=True)
  embedding_target: Literal['charge', 'spin'] = eqx.field(static=True)
  trainable: bool = eqx.field(static=True)

  def __init__(
    self,
    embedding_type: str,
    embedding_target: Literal['charge', 'spin'],
    embedding_size: int,
    trainable: bool = False,
    scale: float = 1.0,
    *,
    key: Array,
  ):
    self.embedding_type = embedding_type
    self.embedding_target = embedding_target
    self.trainable = trainable

    self.W = None
    self.linear = None
    self.embedding = None

    if embedding_type == 'pos_emb':
      self.W = jax.random.normal(key, (embedding_size // 2,)) * scale
    elif embedding_type == 'lin_emb':
      self.linear = primitives.Dense(1, embedding_size, key=key, use_bias=True)
    elif embedding_type == 'rand_emb':
      if embedding_target == 'charge':
        num_embeddings = 201
      else:
        num_embeddings = 101
      self.embedding = primitives.Embed(num_embeddings, embedding_size, key=key)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Embed charge or spin values.

    Args:
        x: Values to embed, shape [batch].

    Returns:
        Embeddings, shape [batch, embedding_size].
    """
    if self.embedding_type == 'pos_emb':
      W = self.W
      if not self.trainable:
        W = jax.lax.stop_gradient(W)

      x_proj = x[:, None] * W[None, :] * 2 * jnp.pi  # ty: ignore[not-subscriptable]
      emb = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

      if self.embedding_target == 'spin':
        zero_mask = (x == 0)[:, None]
        emb = jnp.where(zero_mask, 0.0, emb)

      return emb

    elif self.embedding_type == 'lin_emb':
      x_input = x.astype(jnp.float32)[:, None]

      if self.embedding_target == 'spin':
        x_input = jnp.where(x[:, None] == 0, -100.0, x_input)

      out = self.linear(x_input)  # ty: ignore[call-non-callable]
      if not self.trainable:
        out = jax.lax.stop_gradient(out)
      return out

    elif self.embedding_type == 'rand_emb':
      if self.embedding_target == 'charge':
        x_idx = x + 100
      else:
        x_idx = x

      x_idx = x_idx.astype(jnp.int32)
      out = self.embedding(x_idx)  # ty: ignore[call-non-callable]
      if not self.trainable:
        out = jax.lax.stop_gradient(out)
      return out

    raise ValueError(f'Unknown embedding type: {self.embedding_type}')


class DatasetEmbedding(eqx.Module):
  """Embedding for dataset identifiers.

  Uses integer indices instead of string names for JIT compatibility.
  Call dataset_names_to_indices() outside JIT to convert strings to ints.

  Attributes:
      trainable: Whether the embeddings are trainable.
  """

  embedding: primitives.Embed
  trainable: bool = eqx.field(static=True)

  def __init__(
    self,
    embedding_size: int,
    num_datasets: int,
    trainable: bool = False,
    *,
    key: Array,
  ):
    self.trainable = trainable
    self.embedding = primitives.Embed(num_datasets, embedding_size, key=key)

  def __call__(self, dataset_idx: jnp.ndarray) -> jnp.ndarray:
    """Embed dataset indices.

    Args:
        dataset_idx: Integer dataset indices, shape [num_systems].

    Returns:
        Embeddings, shape [num_systems, embedding_size].
    """
    out = self.embedding(dataset_idx)
    if not self.trainable:
      out = jax.lax.stop_gradient(out)
    return out


class EdgeDegreeEmbedding(eqx.Module):
  """Embedding based on edge degree (neighbor aggregation).

  Aggregates edge features onto nodes using Wigner rotation.

  Attributes:
      sphere_channels: Number of spherical channels.
      rescale_factor: Factor to rescale aggregated features.
      m_size: List of number of coefficients for each m.
  """

  rad_func: RadialMLP
  sphere_channels: int = eqx.field(static=True)
  rescale_factor: float = eqx.field(static=True)
  m_size: Tuple[int, ...] = eqx.field(static=True)

  def __init__(
    self,
    sphere_channels: int,
    lmax: int,
    mmax: int,
    edge_channels_list: List[int],
    rescale_factor: float,
    m_size: Sequence[int],
    *,
    key: Array,
  ):
    self.sphere_channels = sphere_channels
    self.rescale_factor = rescale_factor
    self.m_size = tuple(m_size)

    m_0_num_coefficients = m_size[0]

    edge_channels = list(edge_channels_list)
    edge_channels.append(m_0_num_coefficients * sphere_channels)
    self.rad_func = RadialMLP(channels_list=edge_channels, key=key)

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

    # Process edge features
    x_edge_m_0 = self.rad_func(x_edge)
    x_edge_m_0 = x_edge_m_0.reshape(
      -1, m_0_num_coefficients, self.sphere_channels
    )

    # Only m=0 radial features are nonzero here, so multiply by the matching
    # inverse-Wigner columns directly instead of padding to all coefficients.
    x_edge_embedding = jnp.einsum(
      'njk,nkc->njc',
      wigner_inv[:, :, :m_0_num_coefficients],
      x_edge_m_0,
    )

    # Apply envelope
    x_edge_embedding = x_edge_embedding * edge_envelope

    # Aggregate onto target nodes using scatter-add
    target_indices = edge_index[1] - node_offset
    x_update = jnp.zeros_like(x).at[target_indices].add(x_edge_embedding)

    return x + x_update / self.rescale_factor
