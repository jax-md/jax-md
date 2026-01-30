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

"""UMA (Universal Models for Atoms) backbone model.

This module provides the main UMA backbone model that produces node embeddings
from atomic positions and properties.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.nn import initializers

from jax_md._nn.ff.uma.common.so3 import (
  CoefficientMapping,
  SO3Grid,
  create_coefficient_mapping,
  create_so3_grid,
  coefficient_idx,
)
from jax_md._nn.ff.uma.common.rotation import (
  compute_jacobi_matrices,
  load_jacobi_matrices_from_file,
  init_edge_rot_euler_angles,
  eulers_to_wigner,
)
from jax_md._nn.ff.uma.nn.radial import GaussianSmearing, PolynomialEnvelope
from jax_md._nn.ff.uma.nn.embedding import (
  AtomicEmbedding,
  ChgSpinEmbedding,
  DatasetEmbedding,
  EdgeDegreeEmbedding,
)
from jax_md._nn.ff.uma.nn.layer_norm import get_normalization_layer
from jax_md._nn.ff.uma.blocks import UMABlock


@dataclass
class UMAConfig:
  """Configuration for UMA model.

  Attributes:
      max_num_elements: Maximum number of atomic elements.
      sphere_channels: Number of spherical channels.
      lmax: Maximum degree of spherical harmonics.
      mmax: Maximum order of spherical harmonics.
      num_layers: Number of UMA blocks.
      hidden_channels: Number of hidden channels in MLPs.
      cutoff: Distance cutoff in Angstroms.
      edge_channels: Number of edge channels.
      num_distance_basis: Number of Gaussian basis functions.
      norm_type: Type of normalization layer.
      act_type: Type of activation function.
      ff_type: Type of feed-forward layer.
      grid_resolution: Resolution for SO3 grid (None for default).
      chg_spin_emb_type: Type of charge/spin embedding.
      dataset_list: List of dataset names for dataset embedding.
      use_dataset_embedding: Whether to use dataset embedding.
  """

  max_num_elements: int = 100
  sphere_channels: int = 128
  lmax: int = 2
  mmax: int = 2
  num_layers: int = 2
  hidden_channels: int = 128
  cutoff: float = 5.0
  edge_channels: int = 128
  num_distance_basis: int = 512
  norm_type: str = 'rms_norm_sh'
  act_type: str = 'gate'
  ff_type: str = 'grid'
  grid_resolution: Optional[int] = None
  chg_spin_emb_type: str = 'pos_emb'
  dataset_list: Optional[List[str]] = field(default=None)
  use_dataset_embedding: bool = True


def default_config() -> UMAConfig:
  """Get default UMA configuration."""
  return UMAConfig(
    max_num_elements=100,
    sphere_channels=128,
    lmax=2,
    mmax=2,
    num_layers=2,
    hidden_channels=128,
    cutoff=5.0,
    edge_channels=128,
    num_distance_basis=512,
    norm_type='rms_norm_sh',
    act_type='gate',
    ff_type='grid',
    grid_resolution=None,
    chg_spin_emb_type='pos_emb',
    dataset_list=['oc20', 'omol', 'omat', 'odac', 'omc'],
    use_dataset_embedding=True,
  )


class UMABackbone(nn.Module):
  """UMA backbone model.

  This is the main feature extractor that produces equivariant node embeddings
  from atomic positions and properties.

  Attributes:
      config: UMA configuration.
  """

  config: UMAConfig

  def setup(self):
    cfg = self.config

    # Spherical harmonic feature size
    self.sph_feature_size = (cfg.lmax + 1) ** 2

    # Create coefficient mapping
    self.mapping = create_coefficient_mapping(cfg.lmax, cfg.mmax)

    # Create SO3 grids
    self.so3_grid_lmax_lmax = create_so3_grid(
      cfg.lmax, cfg.lmax, resolution=cfg.grid_resolution, rescale=True
    )
    self.so3_grid_lmax_mmax = create_so3_grid(
      cfg.lmax, cfg.mmax, resolution=cfg.grid_resolution, rescale=True
    )

    # Load Jacobi matrices for Wigner D-matrix computation
    # Use precomputed values for exact compatibility with PyTorch weights
    self.Jd_list = load_jacobi_matrices_from_file(cfg.lmax)

    # Get coefficient index for lmax/mmax subset
    self.coefficient_index = coefficient_idx(
      self.so3_grid_lmax_lmax.mapping, cfg.lmax, cfg.mmax
    )

    # Edge channels list
    self.edge_channels_list = [
      cfg.num_distance_basis + 2 * cfg.edge_channels,
      cfg.edge_channels,
      cfg.edge_channels,
    ]

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
    dataset: Optional[List[str]] = None,
  ) -> Dict[str, jnp.ndarray]:
    """Apply UMA backbone.

    Args:
        positions: Atomic positions, shape [num_atoms, 3].
        atomic_numbers: Atomic numbers, shape [num_atoms].
        batch: Batch indices, shape [num_atoms].
        edge_index: Edge connectivity, shape [2, num_edges].
        edge_distance_vec: Edge vectors, shape [num_edges, 3].
        charge: System charges, shape [num_systems].
        spin: System spins, shape [num_systems].
        dataset: Dataset names, length [num_systems].

    Returns:
        Dictionary with:
            - node_embedding: Node embeddings, shape [num_atoms, (lmax+1)^2, sphere_channels].
            - batch: Batch indices.
    """
    cfg = self.config
    num_atoms = positions.shape[0]
    num_edges = edge_index.shape[1]

    # === Embeddings ===

    # Atomic number embedding
    sphere_embedding = nn.Embed(
      num_embeddings=cfg.max_num_elements,
      features=cfg.sphere_channels,
      name='sphere_embedding',
    )
    atom_emb = sphere_embedding(atomic_numbers)

    # Charge/spin embeddings
    charge_embedding = ChgSpinEmbedding(
      embedding_type=cfg.chg_spin_emb_type,
      embedding_target='charge',
      embedding_size=cfg.sphere_channels,
      trainable=False,
      name='charge_embedding',
    )
    spin_embedding = ChgSpinEmbedding(
      embedding_type=cfg.chg_spin_emb_type,
      embedding_target='spin',
      embedding_size=cfg.sphere_channels,
      trainable=False,
      name='spin_embedding',
    )

    chg_emb = charge_embedding(charge)
    spin_emb = spin_embedding(spin)

    # Dataset embedding and mixing
    if (
      cfg.use_dataset_embedding
      and cfg.dataset_list is not None
      and dataset is not None
    ):
      dataset_embedding = DatasetEmbedding(
        embedding_size=cfg.sphere_channels,
        dataset_list=cfg.dataset_list,
        trainable=False,
        name='dataset_embedding',
      )
      dataset_emb = dataset_embedding(dataset)
      csd_cat = jnp.concatenate([chg_emb, spin_emb, dataset_emb], axis=1)
      mix_csd = nn.Dense(cfg.sphere_channels, name='mix_csd')
      csd_mixed_emb = nn.silu(mix_csd(csd_cat))
    else:
      csd_cat = jnp.concatenate([chg_emb, spin_emb], axis=1)
      mix_csd = nn.Dense(cfg.sphere_channels, name='mix_csd')
      csd_mixed_emb = nn.silu(mix_csd(csd_cat))

    # === Edge distance embedding ===

    edge_distance = jnp.linalg.norm(edge_distance_vec, axis=-1)
    dist_scaled = edge_distance / cfg.cutoff

    # Polynomial envelope
    envelope = PolynomialEnvelope(exponent=5, name='envelope')
    edge_envelope = envelope(dist_scaled).reshape(-1, 1, 1)

    # Gaussian smearing
    distance_expansion = GaussianSmearing(
      start=0.0,
      stop=cfg.cutoff,
      num_gaussians=cfg.num_distance_basis,
      basis_width_scalar=2.0,
      name='distance_expansion',
    )
    edge_distance_embedding = distance_expansion(edge_distance)

    # Source/target embeddings for edges
    source_embedding = nn.Embed(
      num_embeddings=cfg.max_num_elements,
      features=cfg.edge_channels,
      embedding_init=initializers.uniform(scale=0.002),
      name='source_embedding',
    )
    target_embedding = nn.Embed(
      num_embeddings=cfg.max_num_elements,
      features=cfg.edge_channels,
      embedding_init=initializers.uniform(scale=0.002),
      name='target_embedding',
    )

    source_emb = source_embedding(atomic_numbers[edge_index[0]])
    target_emb = target_embedding(atomic_numbers[edge_index[1]])
    # Shift to [-0.001, 0.001]
    source_emb = source_emb - 0.001
    target_emb = target_emb - 0.001

    x_edge = jnp.concatenate(
      [edge_distance_embedding, source_emb, target_emb],
      axis=1,
    )

    # === Compute Wigner matrices ===

    euler_angles = init_edge_rot_euler_angles(edge_distance_vec)
    wigner = eulers_to_wigner(euler_angles, 0, cfg.lmax, self.Jd_list)
    wigner_inv = jnp.transpose(wigner, (0, 2, 1))

    # Select subset if mmax != lmax
    if cfg.mmax != cfg.lmax:
      wigner = wigner[:, self.coefficient_index, :][
        :, :, self.coefficient_index
      ]
      wigner_inv = wigner_inv[:, self.coefficient_index, :][
        :, :, self.coefficient_index
      ]

    # Combine with M mapping
    to_m = self.mapping.to_m
    wigner_and_M_mapping = jnp.einsum('mk,nkj->nmj', to_m, wigner)
    wigner_and_M_mapping_inv = jnp.einsum('njk,mk->njm', wigner_inv, to_m)

    # === Initialize node embeddings ===

    x_message = jnp.zeros(
      (num_atoms, self.sph_feature_size, cfg.sphere_channels),
      dtype=positions.dtype,
    )
    # Set scalar (l=0) component to atomic embedding
    x_message = x_message.at[:, 0, :].set(atom_emb)

    # Add system embedding to each atom
    sys_node_embedding = csd_mixed_emb[batch]
    x_message = x_message.at[:, 0, :].add(sys_node_embedding)

    # === Edge degree embedding ===

    edge_degree_embedding = EdgeDegreeEmbedding(
      sphere_channels=cfg.sphere_channels,
      lmax=cfg.lmax,
      mmax=cfg.mmax,
      edge_channels_list=self.edge_channels_list,
      rescale_factor=5.0,  # sqrt of avg degree
      m_size=self.mapping.m_size,
      name='edge_degree_embedding',
    )
    x_message = edge_degree_embedding(
      x_message,
      x_edge,
      edge_index,
      wigner_and_M_mapping_inv,
      edge_envelope,
      node_offset=0,
    )

    # === Message passing blocks ===

    for i in range(cfg.num_layers):
      block = UMABlock(
        sphere_channels=cfg.sphere_channels,
        hidden_channels=cfg.hidden_channels,
        lmax=cfg.lmax,
        mmax=cfg.mmax,
        m_size=self.mapping.m_size,
        edge_channels_list=self.edge_channels_list,
        cutoff=cfg.cutoff,
        norm_type=cfg.norm_type,
        act_type=cfg.act_type,
        ff_type=cfg.ff_type,
        to_grid_mat=self.so3_grid_lmax_lmax.to_grid_mat,
        from_grid_mat=self.so3_grid_lmax_lmax.from_grid_mat,
        name=f'blocks_{i}',
      )
      x_message = block(
        x_message,
        x_edge,
        edge_distance,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        edge_envelope,
        sys_node_embedding=sys_node_embedding,
        node_offset=0,
      )

    # === Final normalization ===

    norm = get_normalization_layer(
      cfg.norm_type,
      lmax=cfg.lmax,
      num_channels=cfg.sphere_channels,
    )
    x_message = norm(x_message)

    return {
      'node_embedding': x_message,
      'batch': batch,
    }
