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

"""Prediction heads for UMA model.

This module provides prediction heads for energy, forces, and stress
that operate on UMA backbone embeddings.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_md._nn.uma.nn.so3_layers import SO3Linear


class MLPEnergyHead(nn.Module):
  """MLP head for energy prediction.

  Uses a 3-layer MLP to predict per-atom energies which are summed
  to get the total energy.

  Attributes:
      sphere_channels: Number of input channels.
      hidden_channels: Number of hidden channels.
      reduce: Reduction method ('sum' or 'mean').
  """

  sphere_channels: int
  hidden_channels: int
  reduce: Literal['sum', 'mean'] = 'sum'

  @nn.compact
  def __call__(
    self,
    node_embedding: jnp.ndarray,
    batch: jnp.ndarray,
    num_systems: int,
    natoms: Optional[jnp.ndarray] = None,
  ) -> Dict[str, jnp.ndarray]:
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
    x = nn.Dense(self.hidden_channels, use_bias=True, name='linear_0')(
      scalar_input
    )
    x = nn.silu(x)
    x = nn.Dense(self.hidden_channels, use_bias=True, name='linear_1')(x)
    x = nn.silu(x)
    x = nn.Dense(1, use_bias=True, name='linear_2')(x)

    node_energy = x.squeeze(-1)

    # Aggregate per-atom energies
    energy = jax.ops.segment_sum(node_energy, batch, num_segments=num_systems)

    if self.reduce == 'mean' and natoms is not None:
      energy = energy / natoms

    return {'energy': energy}


class LinearEnergyHead(nn.Module):
  """Simple linear head for energy prediction.

  Uses a single linear layer for fast inference.

  Attributes:
      sphere_channels: Number of input channels.
      reduce: Reduction method ('sum' or 'mean').
  """

  sphere_channels: int
  reduce: Literal['sum', 'mean'] = 'sum'

  @nn.compact
  def __call__(
    self,
    node_embedding: jnp.ndarray,
    batch: jnp.ndarray,
    num_systems: int,
    natoms: Optional[jnp.ndarray] = None,
  ) -> Dict[str, jnp.ndarray]:
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

    # Single linear layer
    linear = nn.Dense(1, use_bias=True, name='energy_block')
    node_energy = linear(scalar_input).squeeze(-1)

    # Aggregate per-atom energies
    energy = jax.ops.segment_sum(node_energy, batch, num_segments=num_systems)

    if self.reduce == 'mean' and natoms is not None:
      energy = energy / natoms

    return {'energy': energy}


class LinearForceHead(nn.Module):
  """Linear head for direct force prediction.

  Uses an SO(3) linear layer to directly predict forces from
  the equivariant embeddings.

  Attributes:
      sphere_channels: Number of input channels.
  """

  sphere_channels: int

  @nn.compact
  def __call__(
    self,
    node_embedding: jnp.ndarray,
  ) -> Dict[str, jnp.ndarray]:
    """Predict forces from embeddings.

    Args:
        node_embedding: Node embeddings, shape [num_atoms, (lmax+1)^2, sphere_channels].

    Returns:
        Dictionary with 'forces' key, shape [num_atoms, 3].
    """
    # Use SO3 linear with lmax=1 to get vector output
    # Input: first 4 coefficients (l=0 and l=1)
    linear = SO3Linear(
      out_features=1,
      lmax=1,
      use_bias=True,
      name='linear',
    )

    # Select l=0 and l=1 components
    forces = linear(node_embedding[:, :4, :])

    # Extract l=1 (vector) components (indices 1, 2, 3)
    forces = forces[:, 1:4, 0]  # [num_atoms, 3]

    return {'forces': forces}


class GradientForceHead(nn.Module):
  """Force head using gradient of energy.

  Computes forces as the negative gradient of energy with respect
  to positions. This ensures energy conservation.

  Attributes:
      energy_head: Energy prediction head to use.
  """

  sphere_channels: int
  hidden_channels: int
  reduce: Literal['sum', 'mean'] = 'sum'

  def __call__(
    self,
    apply_fn,
    params,
    positions: jnp.ndarray,
    atomic_numbers: jnp.ndarray,
    batch: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_distance_vec: jnp.ndarray,
    charge: jnp.ndarray,
    spin: jnp.ndarray,
    num_systems: int,
    dataset: Optional[list] = None,
  ) -> Dict[str, jnp.ndarray]:
    """Predict energy and forces.

    Args:
        apply_fn: Function to apply model (backbone + energy head).
        params: Model parameters.
        positions: Atomic positions, shape [num_atoms, 3].
        atomic_numbers: Atomic numbers, shape [num_atoms].
        batch: Batch indices, shape [num_atoms].
        edge_index: Edge connectivity, shape [2, num_edges].
        edge_distance_vec: Edge vectors, shape [num_edges, 3].
        charge: System charges, shape [num_systems].
        spin: System spins, shape [num_systems].
        num_systems: Number of systems.
        dataset: Dataset names.

    Returns:
        Dictionary with 'energy' and 'forces' keys.
    """

    def energy_fn(pos):
      # Recompute edge vectors from positions
      edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
      output = apply_fn(
        params,
        pos,
        atomic_numbers,
        batch,
        edge_index,
        edge_vec,
        charge,
        spin,
        dataset,
      )
      emb = output['node_embedding']

      # Apply energy head
      scalar_input = emb[:, 0, :]
      x = nn.Dense(self.hidden_channels, use_bias=True)(scalar_input)
      x = nn.silu(x)
      x = nn.Dense(self.hidden_channels, use_bias=True)(x)
      x = nn.silu(x)
      x = nn.Dense(1, use_bias=True)(x)
      node_energy = x.squeeze(-1)
      energy = jax.ops.segment_sum(node_energy, batch, num_segments=num_systems)
      return energy.sum()

    energy, grad = jax.value_and_grad(energy_fn)(positions)
    forces = -grad

    return {
      'energy': energy,
      'forces': forces,
    }


def create_uma_energy_fn(
  backbone_config,
  head_type: Literal['mlp', 'linear'] = 'mlp',
  use_gradient_forces: bool = False,
):
  """Create a combined UMA energy function.

  This creates a function that takes atomic configurations and returns
  energies (and optionally forces via gradients).

  Args:
      backbone_config: UMAConfig for the backbone.
      head_type: Type of energy head ('mlp' or 'linear').
      use_gradient_forces: If True, compute forces as energy gradients.

  Returns:
      Tuple of (model, init_fn) where model is a Flax module and
      init_fn initializes parameters.
  """
  from jax_md._nn.uma.model import UMABackbone

  class UMAEnergyModel(nn.Module):
    config: backbone_config.__class__

    @nn.compact
    def __call__(
      self,
      positions: jnp.ndarray,
      atomic_numbers: jnp.ndarray,
      batch: jnp.ndarray,
      edge_index: jnp.ndarray,
      edge_distance_vec: jnp.ndarray,
      charge: jnp.ndarray,
      spin: jnp.ndarray,
      dataset=None,
    ):
      # Backbone
      backbone = UMABackbone(config=self.config, name='backbone')
      emb = backbone(
        positions,
        atomic_numbers,
        batch,
        edge_index,
        edge_distance_vec,
        charge,
        spin,
        dataset,
      )

      # Energy head
      if head_type == 'mlp':
        head = MLPEnergyHead(
          sphere_channels=self.config.sphere_channels,
          hidden_channels=self.config.hidden_channels,
          name='energy_head',
        )
      else:
        head = LinearEnergyHead(
          sphere_channels=self.config.sphere_channels,
          name='energy_head',
        )

      num_systems = charge.shape[0]
      return head(emb['node_embedding'], batch, num_systems)

  return UMAEnergyModel(config=backbone_config)
