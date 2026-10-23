"""Prediction heads for UMA model.

This module provides prediction heads for energy and forces
that operate on UMA backbone embeddings.

For energy-conserving forces, use jax.grad of the energy function
(done automatically in the JAX-MD integration via uma_neighbor_list).

Ported from FairChem's UMA implementation.
"""

from typing import Dict, Literal, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_md import util as jmd_util
from jax_md._nn.uma.nn import primitives
from jax_md._nn.uma.nn.so3_layers import SO3Linear

Array = jmd_util.Array


class MLPEnergyHead(eqx.Module):
  """MLP head for energy prediction.

  Uses a 3-layer MLP to predict per-atom energies which are summed
  to get the total energy.

  Attributes:
      linear: The three dense layers, in application order.
      reduce: Reduction method, 'sum' or 'mean'.
  """

  linear: Tuple[primitives.Dense, ...]
  reduce: Literal['sum', 'mean'] = eqx.field(static=True)

  def __init__(
    self,
    sphere_channels: int,
    hidden_channels: int,
    reduce: Literal['sum', 'mean'] = 'sum',
    *,
    key: Array,
  ):
    keys = jax.random.split(key, 3)
    self.linear = (
      primitives.Dense(
        sphere_channels, hidden_channels, key=keys[0], use_bias=True
      ),
      primitives.Dense(
        hidden_channels, hidden_channels, key=keys[1], use_bias=True
      ),
      primitives.Dense(hidden_channels, 1, key=keys[2], use_bias=True),
    )
    self.reduce = reduce

  def __call__(
    self,
    node_embedding: Array,
    batch: Array,
    num_systems: int,
    natoms: Array | None = None,
  ) -> Dict[str, Array]:
    """Predict energy from embeddings.

    Args:
        node_embedding: Node embeddings, shape [num_atoms, (lmax+1)^2, sphere_channels].
        batch: Batch indices, shape [num_atoms].
        num_systems: Number of systems in batch.
        natoms: Number of atoms per system, shape [num_systems].

    Returns:
        Dictionary with 'energy' key, shape [num_systems].
    """
    # Extract scalar (l=0) component
    scalar_input = node_embedding[:, 0, :]

    # MLP: Linear -> SiLU -> Linear -> SiLU -> Linear
    x = self.linear[0](scalar_input)
    x = jax.nn.silu(x)
    x = self.linear[1](x)
    x = jax.nn.silu(x)
    x = self.linear[2](x)

    node_energy = x.squeeze(-1)

    # Aggregate per-atom energies via scatter-add
    energy = jnp.zeros(num_systems).at[batch].add(node_energy)

    if self.reduce == 'mean' and natoms is not None:
      energy = energy / natoms

    return {'energy': energy}


class LinearEnergyHead(eqx.Module):
  """Simple linear head for energy prediction.

  Uses a single linear layer for fast inference.

  Attributes:
      energy_block: Dense layer mapping scalars to per-atom energies.
      reduce: Reduction method, 'sum' or 'mean'.
  """

  energy_block: primitives.Dense
  reduce: Literal['sum', 'mean'] = eqx.field(static=True)

  def __init__(
    self,
    sphere_channels: int,
    reduce: Literal['sum', 'mean'] = 'sum',
    *,
    key: Array,
  ):
    self.energy_block = primitives.Dense(
      sphere_channels, 1, key=key, use_bias=True
    )
    self.reduce = reduce

  def __call__(
    self,
    node_embedding: Array,
    batch: Array,
    num_systems: int,
    natoms: Array | None = None,
  ) -> Dict[str, Array]:
    """Predict energy from embeddings.

    Args:
        node_embedding: Node embeddings, shape [num_atoms, (lmax+1)^2, sphere_channels].
        batch: Batch indices, shape [num_atoms].
        num_systems: Number of systems in batch.
        natoms: Number of atoms per system, shape [num_systems].

    Returns:
        Dictionary with 'energy' key, shape [num_systems].
    """
    scalar_input = node_embedding[:, 0, :]
    node_energy = self.energy_block(scalar_input).squeeze(-1)

    energy = jnp.zeros(num_systems).at[batch].add(node_energy)

    if self.reduce == 'mean' and natoms is not None:
      energy = energy / natoms

    return {'energy': energy}


class LinearForceHead(eqx.Module):
  """Linear head for direct force prediction.

  Uses an SO(3) linear layer to directly predict forces from
  the equivariant embeddings.

  Attributes:
      linear: SO(3) linear layer producing the force vectors.
  """

  linear: SO3Linear

  def __init__(self, sphere_channels: int, *, key: Array):
    self.linear = SO3Linear(
      sphere_channels,
      1,
      1,
      key=key,
      use_bias=False,
    )

  def __call__(
    self,
    node_embedding: Array,
  ) -> Dict[str, Array]:
    """Predict forces from embeddings.

    Args:
        node_embedding: Node embeddings, shape [num_atoms, (lmax+1)^2, sphere_channels].

    Returns:
        Dictionary with 'forces' key, shape [num_atoms, 3].
    """
    # Select l=0 and l=1 components
    forces = self.linear(node_embedding[:, :4, :])

    # l=1 components
    forces = forces[:, 1:4, 0]  # [num_atoms, 3]

    return {'forces': forces}
