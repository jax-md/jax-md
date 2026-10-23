"""UMA backbone model.

This module provides the main UMA backbone model that produces node embeddings
from atomic positions and properties.

Ported from FairChem's UMA implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_md import util as jmd_util
from jax_md._nn.uma.common.so3 import (
  create_coefficient_mapping,
  create_so3_grid,
  coefficient_idx,
)
from jax_md._nn.uma.common.rotation import (
  load_jacobi_matrices_from_file,
  init_edge_rot_euler_angles,
  eulers_to_wigner,
)
from jax_md._nn.uma.nn.radial import GaussianSmearing, PolynomialEnvelope
from jax_md._nn.uma.nn.embedding import (
  ChgSpinEmbedding,
  DatasetEmbedding,
  EdgeDegreeEmbedding,
  symmetric_uniform,
)
from jax_md._nn.uma.nn.layer_norm import get_normalization_layer
from jax_md._nn.uma.nn import primitives
from jax_md._nn.uma.blocks import UMABlock

Array = jmd_util.Array


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
      activation_checkpointing: Whether to rematerialize the blocks.
      use_kernels: Whether to use optimized GPU kernels for eligible UMA
        operations. Only takes effect on GPU with lmax = mmax = 2; elsewhere
        the einsum path runs.
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
  grid_resolution: int | None = None
  chg_spin_emb_type: str = 'pos_emb'
  dataset_list: List[str] | None = field(default=None)
  use_dataset_embedding: bool = True
  activation_checkpointing: bool = False
  use_kernels: bool = True


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
    use_kernels=True,
  )


class UMABackbone(eqx.Module):
  """UMA backbone model.

  This is the main feature extractor that produces equivariant node embeddings
  from atomic positions and properties.

  Attributes:
      sphere_embedding: Atomic number embedding table.
      charge_embedding: Charge embedding.
      spin_embedding: Spin embedding.
      dataset_embedding: Dataset embedding, or None.
      mix_csd: Dense layer mixing the charge/spin/dataset embeddings.
      envelope: Polynomial cutoff envelope.
      distance_expansion: Gaussian basis expansion of edge distances.
      source_embedding: Edge source atomic number embedding table.
      target_embedding: Edge target atomic number embedding table.
      edge_degree_embedding: Edge degree embedding.
      blocks: Message passing blocks.
      norm: Final normalization layer.
  """

  sphere_embedding: primitives.Embed
  charge_embedding: ChgSpinEmbedding
  spin_embedding: ChgSpinEmbedding
  dataset_embedding: DatasetEmbedding | None
  mix_csd: primitives.Dense
  envelope: PolynomialEnvelope
  distance_expansion: GaussianSmearing
  source_embedding: primitives.Embed
  target_embedding: primitives.Embed
  edge_degree_embedding: EdgeDegreeEmbedding
  blocks: Tuple[UMABlock, ...]
  norm: eqx.Module
  lmax: int = eqx.field(static=True)
  mmax: int = eqx.field(static=True)
  grid_resolution: int | None = eqx.field(static=True)
  cutoff: float = eqx.field(static=True)
  sphere_channels: int = eqx.field(static=True)
  sph_feature_size: int = eqx.field(static=True)
  activation_checkpointing: bool = eqx.field(static=True)

  def __init__(self, config: UMAConfig, *, key: Array):
    cfg = config
    keys = jax.random.split(key, 8)

    self.lmax = cfg.lmax
    self.mmax = cfg.mmax
    self.grid_resolution = cfg.grid_resolution
    self.cutoff = cfg.cutoff
    self.sphere_channels = cfg.sphere_channels
    self.activation_checkpointing = cfg.activation_checkpointing

    # Spherical harmonic feature size
    self.sph_feature_size = (cfg.lmax + 1) ** 2
    mapping = create_coefficient_mapping(cfg.lmax, cfg.mmax)

    edge_channels_list = [
      cfg.num_distance_basis + 2 * cfg.edge_channels,
      cfg.edge_channels,
      cfg.edge_channels,
    ]

    self.sphere_embedding = primitives.Embed(
      cfg.max_num_elements, cfg.sphere_channels, key=keys[0]
    )

    self.charge_embedding = ChgSpinEmbedding(
      embedding_type=cfg.chg_spin_emb_type,
      embedding_target='charge',
      embedding_size=cfg.sphere_channels,
      trainable=False,
      key=keys[1],
    )
    self.spin_embedding = ChgSpinEmbedding(
      embedding_type=cfg.chg_spin_emb_type,
      embedding_target='spin',
      embedding_size=cfg.sphere_channels,
      trainable=False,
      key=keys[2],
    )

    if cfg.use_dataset_embedding and cfg.dataset_list is not None:
      self.dataset_embedding = DatasetEmbedding(
        embedding_size=cfg.sphere_channels,
        num_datasets=len(cfg.dataset_list),
        trainable=False,
        key=keys[3],
      )
      mix_csd_in_features = 3 * cfg.sphere_channels
    else:
      self.dataset_embedding = None
      mix_csd_in_features = 2 * cfg.sphere_channels

    self.mix_csd = primitives.Dense(
      mix_csd_in_features, cfg.sphere_channels, key=keys[4]
    )

    self.envelope = PolynomialEnvelope(exponent=5)

    self.distance_expansion = GaussianSmearing(
      start=0.0,
      stop=cfg.cutoff,
      num_gaussians=cfg.num_distance_basis,
      basis_width_scalar=2.0,
    )

    source_key, target_key = jax.random.split(keys[5])
    self.source_embedding = primitives.Embed(
      cfg.max_num_elements,
      cfg.edge_channels,
      key=source_key,
      embedding_init=symmetric_uniform(0.001),
    )
    self.target_embedding = primitives.Embed(
      cfg.max_num_elements,
      cfg.edge_channels,
      key=target_key,
      embedding_init=symmetric_uniform(0.001),
    )

    self.edge_degree_embedding = EdgeDegreeEmbedding(
      sphere_channels=cfg.sphere_channels,
      lmax=cfg.lmax,
      mmax=cfg.mmax,
      edge_channels_list=edge_channels_list,
      rescale_factor=5.0,  # sqrt of avg degree
      m_size=mapping.m_size,
      key=keys[6],
    )

    block_keys = jax.random.split(keys[7], max(cfg.num_layers, 1))
    self.blocks = tuple(
      UMABlock(
        sphere_channels=cfg.sphere_channels,
        hidden_channels=cfg.hidden_channels,
        lmax=cfg.lmax,
        mmax=cfg.mmax,
        m_size=mapping.m_size,
        edge_channels_list=edge_channels_list,
        norm_type=cfg.norm_type,
        act_type=cfg.act_type,
        ff_type=cfg.ff_type,
        grid_resolution=cfg.grid_resolution,
        use_kernels=cfg.use_kernels,
        key=block_keys[i],
      )
      for i in range(cfg.num_layers)
    )

    self.norm = get_normalization_layer(
      cfg.norm_type,
      lmax=cfg.lmax,
      num_channels=cfg.sphere_channels,
    )

  def __call__(
    self,
    positions: Array,
    atomic_numbers: Array,
    batch: Array,
    edge_index: Array,
    edge_distance_vec: Array,
    charge: Array,
    spin: Array,
    dataset_idx: Array | None = None,
  ) -> Dict[str, Array]:
    """Apply UMA backbone.

    Args:
        positions: Atomic positions, shape [num_atoms, 3].
        atomic_numbers: Atomic numbers, shape [num_atoms].
        batch: Batch indices, shape [num_atoms].
        edge_index: Edge connectivity, shape [2, num_edges].
        edge_distance_vec: Edge vectors, shape [num_edges, 3].
        charge: System charges, shape [num_systems].
        spin: System spins, shape [num_systems].
        dataset_idx: Integer dataset indices, shape [num_systems]. Required
            when the model has a dataset embedding. Use
            embedding.dataset_names_to_indices() to convert strings.

    Returns:
        Dictionary with:
            - node_embedding: Node embeddings, shape [num_atoms, (lmax+1)^2, sphere_channels].
            - batch: Batch indices.
    """
    num_atoms = positions.shape[0]

    # Atomic number embedding
    atom_emb = self.sphere_embedding(atomic_numbers)

    # Charge/spin embeddings
    chg_emb = self.charge_embedding(charge)
    spin_emb = self.spin_embedding(spin)

    # Dataset embedding and mixing
    if self.dataset_embedding is not None:
      if dataset_idx is None:
        raise ValueError(
          'This model has a dataset embedding; pass dataset_idx.'
        )
      dataset_emb = self.dataset_embedding(dataset_idx)
      csd_cat = jnp.concatenate([chg_emb, spin_emb, dataset_emb], axis=1)
    else:
      csd_cat = jnp.concatenate([chg_emb, spin_emb], axis=1)

    csd_mixed_emb = jax.nn.silu(self.mix_csd(csd_cat))

    edge_distance = jnp.linalg.norm(edge_distance_vec, axis=-1)
    dist_scaled = edge_distance / self.cutoff

    # Polynomial envelope
    edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)

    # Gaussian smearing
    edge_distance_embedding = self.distance_expansion(edge_distance)

    source_emb = self.source_embedding(atomic_numbers[edge_index[0]])
    target_emb = self.target_embedding(atomic_numbers[edge_index[1]])

    x_edge = jnp.concatenate(
      [edge_distance_embedding, source_emb, target_emb],
      axis=1,
    )

    euler_angles = init_edge_rot_euler_angles(edge_distance_vec)
    # Constants under a trace, not leaves
    with jax.ensure_compile_time_eval():
      to_m = create_coefficient_mapping(self.lmax, self.mmax).to_m
      Jd_list = load_jacobi_matrices_from_file(self.lmax)
    wigner = eulers_to_wigner(euler_angles, 0, self.lmax, Jd_list)
    wigner_inv = jnp.transpose(wigner, (0, 2, 1))

    # Select subset if mmax != lmax
    if self.mmax != self.lmax:
      with jax.ensure_compile_time_eval():
        grid = create_so3_grid(
          self.lmax, self.lmax, resolution=self.grid_resolution, rescale=True
        )
        coefficient_index = coefficient_idx(grid.mapping, self.lmax, self.mmax)
      wigner = wigner[:, coefficient_index, :]  # [E, m_dim, l_dim]
      wigner_inv = wigner_inv[:, :, coefficient_index]  # [E, l_dim, m_dim]

    # Combine with M mapping
    wigner_and_M_mapping = jnp.einsum('mk,nkj->nmj', to_m, wigner)
    wigner_and_M_mapping_inv = jnp.einsum('njk,mk->njm', wigner_inv, to_m)

    x_message = jnp.zeros(
      (num_atoms, self.sph_feature_size, self.sphere_channels),
      dtype=positions.dtype,
    )
    # Set scalar (l=0) component to atomic embedding
    x_message = x_message.at[:, 0, :].set(atom_emb)

    # Add system embedding to each atom
    sys_node_embedding = csd_mixed_emb[batch]
    x_message = x_message.at[:, 0, :].add(
      sys_node_embedding.astype(x_message.dtype)
    )

    x_message = self.edge_degree_embedding(
      x_message,
      x_edge,
      edge_index,
      wigner_and_M_mapping_inv,
      edge_envelope,
    )

    for block in self.blocks:
      if self.activation_checkpointing:
        block = eqx.filter_checkpoint(block)

      x_message = block(
        x_message,
        x_edge,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        wigner,
        wigner_inv,
        edge_envelope,
        sys_node_embedding=sys_node_embedding,
      )

    x_message = self.norm(x_message)  # ty: ignore[call-non-callable]

    return {
      'node_embedding': x_message,
      'batch': batch,
    }
