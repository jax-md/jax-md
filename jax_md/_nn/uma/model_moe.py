"""UMA Mixture-of-Experts (MoE) backbone model.

This extends UMABackbone with MOLE (Mixture-of-Linear-Experts) layers
in the SO(2) convolutions. The pretrained UMA checkpoints use this
architecture.

The MoE mechanism:
1. A routing MLP takes system-level features and produces per-system
   expert mixing coefficients [num_systems, num_experts].
2. Expert weights in SO2 convolutions are mixed per-system:
   mixed_W = einsum('eoi,be->boi', expert_W, coefficients)
3. Each atom/edge uses its system's mixed weight matrix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import initializers

from jax_md._nn.uma.model import UMAConfig
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
from jax_md._nn.uma.nn.radial import (
  GaussianSmearing,
  PolynomialEnvelope,
  RadialMLP,
)
from jax_md._nn.uma.nn.embedding import (
  ChgSpinEmbedding,
  DatasetEmbedding,
  EdgeDegreeEmbedding,
)
from jax_md._nn.uma.nn.layer_norm import get_normalization_layer
from jax_md._nn.uma.nn.activation import GateActivation, SeparableS2Activation
from jax_md._nn.uma.nn.mole import MOLELinear, RoutingMLP


@dataclass
class UMAMoEConfig(UMAConfig):
  """Configuration for UMA MoE model.

  Extends UMAConfig with MoE-specific parameters.
  """

  num_experts: int = 32
  moe_dropout: float = 0.05
  use_composition_embedding: bool = True
  model_version: float = 1.1
  routing_hidden_channels: int = 64


class SO2MConvMoE(nn.Module):
  """SO(2) convolution for a single order m, with MoE expert weights.

  Like SO2MConv but the fc weight has shape [num_experts, 2*out_half, in].
  Expert mixing is applied before the linear transformation.

  Attributes:
      m: Order of the spherical harmonic coefficients.
      sphere_channels: Number of input spherical channels.
      m_output_channels: Number of output channels.
      lmax: Maximum degree l.
      mmax: Maximum order m.
      num_experts: Number of MoE experts.
  """

  m: int
  sphere_channels: int
  m_output_channels: int
  lmax: int
  mmax: int
  num_experts: int

  @nn.compact
  def __call__(
    self,
    x_m: jnp.ndarray,
    expert_coefficients: jnp.ndarray,
    edge_batch: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply MoE SO(2) convolution for order m.

    Args:
        x_m: Input, shape [num_edges, 2, num_coeffs * sphere_channels].
        expert_coefficients: Shape [num_systems, num_experts].
        edge_batch: System index per edge, shape [num_edges].

    Returns:
        Tuple of (real_output, imag_output).
    """
    num_coefficients = self.lmax - self.m + 1
    out_channels_half = self.m_output_channels * num_coefficients

    fc = MOLELinear(
      num_experts=self.num_experts,
      in_features=num_coefficients * self.sphere_channels,
      out_features=2 * out_channels_half,
      use_bias=False,
      name='fc',
    )

    # Apply MoE linear to last dim: (E, 2, C) -> (E, 2, 2*out_half)
    # MOLELinear handles 3D: [E, 2, C] with batch_indices for edges
    x_m = fc(x_m, expert_coefficients, edge_batch)

    batch_size = x_m.shape[0]
    x_m = x_m.reshape(batch_size, -1, out_channels_half)

    x_r_0 = x_m[:, 0, :]
    x_i_0 = x_m[:, 1, :]
    x_r_1 = x_m[:, 2, :]
    x_i_1 = x_m[:, 3, :]

    x_m_r = x_r_0 - x_i_1
    x_m_i = x_r_1 + x_i_0

    x_m_r = x_m_r.reshape(batch_size, num_coefficients, self.m_output_channels)
    x_m_i = x_m_i.reshape(batch_size, num_coefficients, self.m_output_channels)

    return x_m_r, x_m_i


class SO2ConvolutionMoE(nn.Module):
  """SO(2) convolution block with MoE expert weights.

  Like SO2Convolution but fc_m0 and so2_m_conv use MOLE layers.

  Attributes:
      sphere_channels: Number of input spherical channels.
      m_output_channels: Number of output channels per m.
      lmax: Maximum degree l.
      mmax: Maximum order m.
      m_size: List of number of coefficients for each m.
      num_experts: Number of MoE experts.
      internal_weights: If True, use internal weights.
      edge_channels_list: Channel dims for radial MLP.
      extra_m0_output_channels: Extra output channels for m=0 gating.
  """

  sphere_channels: int
  m_output_channels: int
  lmax: int
  mmax: int
  m_size: List[int]
  num_experts: int
  internal_weights: bool = True
  edge_channels_list: List[int] | None = None
  extra_m0_output_channels: int | None = None

  @nn.compact
  def __call__(
    self,
    x: jnp.ndarray,
    x_edge: jnp.ndarray,
    expert_coefficients: jnp.ndarray,
    edge_batch: jnp.ndarray,
  ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply MoE SO(2) convolution.

    Args:
        x: Input in m-major ordering, shape [num_edges, num_m_coeffs, sphere_channels].
        x_edge: Edge scalar features, shape [num_edges, edge_channels].
        expert_coefficients: Shape [num_systems, num_experts].
        edge_batch: System index per edge, shape [num_edges].
    """
    num_channels_m0 = (self.lmax + 1) * self.sphere_channels
    m0_output_channels = self.m_output_channels * (self.lmax + 1)
    if self.extra_m0_output_channels is not None:
      m0_output_channels = m0_output_channels + self.extra_m0_output_channels

    # MoE fc for m=0
    fc_m0 = MOLELinear(
      num_experts=self.num_experts,
      in_features=num_channels_m0,
      out_features=m0_output_channels,
      use_bias=True,
      name='fc_m0',
    )

    # Radial function (standard, not MoE)
    num_channels_rad = num_channels_m0
    for m in range(1, self.mmax + 1):
      num_channels_rad += (self.lmax - m + 1) * self.sphere_channels

    rad_func = None
    if not self.internal_weights:
      edge_channels = list(self.edge_channels_list) + [num_channels_rad]
      rad_func = RadialMLP(channels_list=edge_channels, name='rad_func')

    m_split_sizes = [self.m_size[m] for m in range(self.mmax + 1)]
    edge_split_sizes = [num_channels_m0]
    for m in range(1, self.mmax + 1):
      edge_split_sizes.append((self.lmax - m + 1) * self.sphere_channels)

    num_edges = x.shape[0]

    if rad_func is not None:
      x_edge_weights = rad_func(x_edge)
      x_edge_by_m = jnp.split(
        x_edge_weights,
        np.cumsum(edge_split_sizes[:-1]),
        axis=1,
      )
    else:
      x_edge_by_m = [None] * (self.mmax + 1)

    x_by_m = jnp.split(
      x,
      np.cumsum(m_split_sizes[:-1]),
      axis=1,
    )

    # m=0 with MoE
    x_0 = x_by_m[0].reshape(num_edges, -1)
    if x_edge_by_m[0] is not None:
      x_0 = x_0 * x_edge_by_m[0]
    x_0 = fc_m0(x_0, expert_coefficients, edge_batch)

    if self.extra_m0_output_channels is not None:
      x_0_extra = x_0[:, : self.extra_m0_output_channels]
      x_0 = x_0[:, self.extra_m0_output_channels :]

    out = [x_0.reshape(num_edges, -1, self.m_output_channels)]

    # m > 0 with MoE
    for m in range(1, self.mmax + 1):
      x_m = x_by_m[m].reshape(num_edges, 2, -1)
      if x_edge_by_m[m] is not None:
        x_m = x_m * x_edge_by_m[m][:, None, :]

      so2_m_conv = SO2MConvMoE(
        m=m,
        sphere_channels=self.sphere_channels,
        m_output_channels=self.m_output_channels,
        lmax=self.lmax,
        mmax=self.mmax,
        num_experts=self.num_experts,
        name=f'so2_m_conv_{m}',
      )
      x_m_r, x_m_i = so2_m_conv(x_m, expert_coefficients, edge_batch)
      out.append(x_m_r)
      out.append(x_m_i)

    out = jnp.concatenate(out, axis=1)

    if self.extra_m0_output_channels is not None:
      return out, x_0_extra
    return out


class EdgewiseMoE(nn.Module):
  """Edgewise message passing with MoE SO2 convolutions."""

  sphere_channels: int
  hidden_channels: int
  lmax: int
  mmax: int
  edge_channels_list: List[int]
  m_size: List[int]
  cutoff: float
  num_experts: int
  act_type: str = 'gate'
  to_grid_mat: jnp.ndarray | None = None
  from_grid_mat: jnp.ndarray | None = None

  @nn.compact
  def __call__(
    self,
    x,
    x_edge,
    edge_distance,
    edge_index,
    wigner_and_M_mapping,
    wigner_and_M_mapping_inv,
    edge_envelope,
    expert_coefficients,
    edge_batch,
    node_offset=0,
  ):
    if self.act_type == 'gate':
      extra_m0 = self.lmax * self.hidden_channels
    else:
      extra_m0 = self.hidden_channels

    so2_conv_1 = SO2ConvolutionMoE(
      sphere_channels=2 * self.sphere_channels,
      m_output_channels=self.hidden_channels,
      lmax=self.lmax,
      mmax=self.mmax,
      m_size=self.m_size,
      num_experts=self.num_experts,
      internal_weights=False,
      edge_channels_list=self.edge_channels_list,
      extra_m0_output_channels=extra_m0,
      name='so2_conv_1',
    )

    so2_conv_2 = SO2ConvolutionMoE(
      sphere_channels=self.hidden_channels,
      m_output_channels=self.sphere_channels,
      lmax=self.lmax,
      mmax=self.mmax,
      m_size=self.m_size,
      num_experts=self.num_experts,
      internal_weights=True,
      name='so2_conv_2',
    )

    if self.act_type == 'gate':
      act = GateActivation(
        lmax=self.lmax,
        mmax=self.mmax,
        num_channels=self.hidden_channels,
        m_prime=True,
        name='act',
      )
    else:
      act = SeparableS2Activation(
        lmax=self.lmax,
        mmax=self.mmax,
        to_grid_mat=self.to_grid_mat,
        from_grid_mat=self.from_grid_mat,
        name='act',
      )

    x_source = x[edge_index[0]]
    x_target = x[edge_index[1]]
    x_message = jnp.concatenate([x_source, x_target], axis=2)
    x_message = jnp.einsum('nmj,njc->nmc', wigner_and_M_mapping, x_message)

    x_message, x_0_gating = so2_conv_1(
      x_message, x_edge, expert_coefficients, edge_batch
    )
    x_message = act(x_0_gating, x_message)
    x_message = so2_conv_2(x_message, x_edge, expert_coefficients, edge_batch)

    x_message = x_message * edge_envelope
    x_message = jnp.einsum('njm,nmc->njc', wigner_and_M_mapping_inv, x_message)

    target_indices = edge_index[1] - node_offset
    new_embedding = jnp.zeros_like(x).at[target_indices].add(x_message)
    return new_embedding


class UMABlockMoE(nn.Module):
  """UMA block with MoE edgewise and standard atomwise."""

  sphere_channels: int
  hidden_channels: int
  lmax: int
  mmax: int
  m_size: List[int]
  edge_channels_list: List[int]
  cutoff: float
  num_experts: int
  norm_type: str = 'rms_norm_sh'
  act_type: str = 'gate'
  ff_type: str = 'spectral'
  to_grid_mat: jnp.ndarray | None = None
  from_grid_mat: jnp.ndarray | None = None

  @nn.compact
  def __call__(
    self,
    x,
    x_edge,
    edge_distance,
    edge_index,
    wigner_and_M_mapping,
    wigner_and_M_mapping_inv,
    edge_envelope,
    expert_coefficients,
    edge_batch,
    sys_node_embedding=None,
    node_offset=0,
  ):
    from jax_md._nn.uma.blocks import SpectralAtomwise, GridAtomwise

    # Edgewise with MoE
    x_res = x
    norm_1 = get_normalization_layer(
      self.norm_type,
      lmax=self.lmax,
      num_channels=self.sphere_channels,
      name='norm_1',
    )
    x = norm_1(x)

    if sys_node_embedding is not None:
      x = x.at[:, 0, :].add(sys_node_embedding)

    edgewise = EdgewiseMoE(
      sphere_channels=self.sphere_channels,
      hidden_channels=self.hidden_channels,
      lmax=self.lmax,
      mmax=self.mmax,
      edge_channels_list=self.edge_channels_list,
      m_size=self.m_size,
      cutoff=self.cutoff,
      num_experts=self.num_experts,
      act_type=self.act_type,
      to_grid_mat=self.to_grid_mat,
      from_grid_mat=self.from_grid_mat,
      name='edge_wise',
    )
    x = edgewise(
      x,
      x_edge,
      edge_distance,
      edge_index,
      wigner_and_M_mapping,
      wigner_and_M_mapping_inv,
      edge_envelope,
      expert_coefficients,
      edge_batch,
      node_offset,
    )
    x = x + x_res

    # Atomwise (standard, not MoE)
    x_res = x
    norm_2 = get_normalization_layer(
      self.norm_type,
      lmax=self.lmax,
      num_channels=self.sphere_channels,
      name='norm_2',
    )
    x = norm_2(x)

    if self.ff_type == 'spectral':
      atomwise = SpectralAtomwise(
        sphere_channels=self.sphere_channels,
        hidden_channels=self.hidden_channels,
        lmax=self.lmax,
        mmax=self.mmax,
        name='atom_wise',
      )
    else:
      atomwise = GridAtomwise(
        sphere_channels=self.sphere_channels,
        hidden_channels=self.hidden_channels,
        lmax=self.lmax,
        mmax=self.mmax,
        to_grid_mat=self.to_grid_mat,
        from_grid_mat=self.from_grid_mat,
        name='atom_wise',
      )
    x = atomwise(x)
    x = x + x_res
    return x


class UMAMoEBackbone(nn.Module):
  """UMA backbone with Mixture-of-Experts.

  This is the architecture used by the pretrained UMA checkpoints.
  It extends UMABackbone with:
  - Expert weights in SO2 convolutions
  - A routing MLP for expert selection
  - Composition embedding for routing input

  Attributes:
      config: UMAMoEConfig instance.
  """

  config: UMAMoEConfig

  def setup(self):
    cfg = self.config
    self.sph_feature_size = (cfg.lmax + 1) ** 2
    self.mapping = create_coefficient_mapping(cfg.lmax, cfg.mmax)
    self.so3_grid_lmax_lmax = create_so3_grid(
      cfg.lmax, cfg.lmax, resolution=cfg.grid_resolution, rescale=True
    )
    self.Jd_list = load_jacobi_matrices_from_file(cfg.lmax)
    self.coefficient_index = coefficient_idx(
      self.so3_grid_lmax_lmax.mapping, cfg.lmax, cfg.mmax
    )
    self.edge_channels_list = [
      cfg.num_distance_basis + 2 * cfg.edge_channels,
      cfg.edge_channels,
      cfg.edge_channels,
    ]

  @nn.compact
  def __call__(
    self,
    positions,
    atomic_numbers,
    batch,
    edge_index,
    edge_distance_vec,
    charge,
    spin,
    dataset_idx=None,
  ):
    cfg = self.config
    num_atoms = positions.shape[0]

    # === Embeddings ===
    sphere_embedding = nn.Embed(
      num_embeddings=cfg.max_num_elements,
      features=cfg.sphere_channels,
      name='sphere_embedding',
    )
    atom_emb = sphere_embedding(atomic_numbers)

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

    if cfg.dataset_list and dataset_idx is not None:
      dataset_embedding = DatasetEmbedding(
        embedding_size=cfg.sphere_channels,
        num_datasets=len(cfg.dataset_list),
        trainable=False,
        name='dataset_embedding',
      )
      dataset_emb = dataset_embedding(dataset_idx)
      csd_cat = jnp.concatenate([chg_emb, spin_emb, dataset_emb], axis=1)
    else:
      csd_cat = jnp.concatenate([chg_emb, spin_emb], axis=1)

    mix_csd = nn.Dense(cfg.sphere_channels, name='mix_csd')
    csd_mixed_emb = nn.silu(mix_csd(csd_cat))

    # === Routing MLP for MoE ===
    routing_input_parts = []

    if cfg.use_composition_embedding:
      composition_emb_layer = nn.Embed(
        num_embeddings=cfg.max_num_elements,
        features=cfg.sphere_channels,
        name='composition_embedding',
      )
      comp_per_atom = composition_emb_layer(atomic_numbers)
      # Mean-pool composition per system via index_reduce mean
      # include_self depends on model_version:
      #   v1.0: include_self=True -> mean = sum / (N + 1)
      #   v1.1+: include_self=False -> mean = sum / N
      num_systems = charge.shape[0]
      comp_sum = (
        jnp.zeros((num_systems, cfg.sphere_channels))
        .at[batch]
        .add(comp_per_atom)
      )
      atom_counts = jnp.zeros(num_systems).at[batch].add(1.0)
      if cfg.model_version < 1.05:
        composition = comp_sum / jnp.maximum(atom_counts[:, None] + 1.0, 1.0)
      else:
        composition = comp_sum / jnp.maximum(atom_counts[:, None], 1.0)

      routing_input_parts.append(composition)

    routing_input_parts.append(csd_mixed_emb)
    routing_input = jnp.concatenate(routing_input_parts, axis=-1)

    routing_mlp = RoutingMLP(
      hidden_channels=cfg.routing_hidden_channels,
      num_experts=cfg.num_experts,
      dropout_rate=cfg.moe_dropout,
      name='routing_mlp',
    )
    expert_coefficients = routing_mlp(routing_input, deterministic=True)

    # === Edge features ===
    edge_distance = jnp.linalg.norm(edge_distance_vec, axis=-1)
    dist_scaled = edge_distance / cfg.cutoff

    envelope = PolynomialEnvelope(exponent=5, name='envelope')
    edge_envelope = envelope(dist_scaled).reshape(-1, 1, 1)

    distance_expansion = GaussianSmearing(
      start=0.0,
      stop=cfg.cutoff,
      num_gaussians=cfg.num_distance_basis,
      basis_width_scalar=2.0,
      name='distance_expansion',
    )
    edge_distance_embedding = distance_expansion(edge_distance)

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
    source_emb = source_embedding(atomic_numbers[edge_index[0]]) - 0.001
    target_emb = target_embedding(atomic_numbers[edge_index[1]]) - 0.001

    x_edge = jnp.concatenate(
      [edge_distance_embedding, source_emb, target_emb],
      axis=1,
    )

    # === Wigner matrices ===
    euler_angles = init_edge_rot_euler_angles(edge_distance_vec)
    wigner = eulers_to_wigner(euler_angles, 0, cfg.lmax, self.Jd_list)
    wigner_inv = jnp.transpose(wigner, (0, 2, 1))

    if cfg.mmax != cfg.lmax:
      # PT: wigner.index_select(1, idx) → select rows only → [E, m_dim, l_dim]
      # PT: wigner_inv.index_select(2, idx) → select cols only → [E, l_dim, m_dim]
      wigner = wigner[:, self.coefficient_index, :]  # [E, m_dim, l_dim]
      wigner_inv = wigner_inv[:, :, self.coefficient_index]  # [E, l_dim, m_dim]

    to_m = self.mapping.to_m
    wigner_and_M_mapping = jnp.einsum('mk,nkj->nmj', to_m, wigner)
    wigner_and_M_mapping_inv = jnp.einsum('njk,mk->njm', wigner_inv, to_m)

    # === Node embeddings ===
    x_message = jnp.zeros(
      (num_atoms, self.sph_feature_size, cfg.sphere_channels),
      dtype=positions.dtype,
    )
    x_message = x_message.at[:, 0, :].set(atom_emb)

    sys_node_embedding = csd_mixed_emb[batch]
    x_message = x_message.at[:, 0, :].add(sys_node_embedding)

    # Edge degree embedding (standard, not MoE)
    edge_degree_embedding = EdgeDegreeEmbedding(
      sphere_channels=cfg.sphere_channels,
      lmax=cfg.lmax,
      mmax=cfg.mmax,
      edge_channels_list=self.edge_channels_list,
      rescale_factor=5.0,
      m_size=self.mapping.m_size,
      name='edge_degree_embedding',
    )
    x_message = edge_degree_embedding(
      x_message,
      x_edge,
      edge_index,
      wigner_and_M_mapping_inv,
      edge_envelope,
    )

    # Edge batch indices (for MoE)
    edge_batch = batch[edge_index[1]]

    # === MoE message passing blocks ===
    for i in range(cfg.num_layers):
      block = UMABlockMoE(
        sphere_channels=cfg.sphere_channels,
        hidden_channels=cfg.hidden_channels,
        lmax=cfg.lmax,
        mmax=cfg.mmax,
        m_size=self.mapping.m_size,
        edge_channels_list=self.edge_channels_list,
        cutoff=cfg.cutoff,
        num_experts=cfg.num_experts,
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
        expert_coefficients,
        edge_batch,
        sys_node_embedding=sys_node_embedding,
      )

    norm = get_normalization_layer(
      cfg.norm_type,
      lmax=cfg.lmax,
      num_channels=cfg.sphere_channels,
      name='norm',
    )
    x_message = norm(x_message)

    return {
      'node_embedding': x_message,
      'batch': batch,
    }


_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def _cache_native(model_name, config, params, metadata, ckpt):
  """Cache converted params as native .npz for fast future loading."""
  import json

  try:
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    params_file = os.path.join(_CHECKPOINT_DIR, f'{model_name}.npz')
    if os.path.exists(params_file):
      return  # Already cached

    # Flatten and save params
    flat = jax.tree.leaves_with_path(params)
    param_dict = {}
    for path, value in flat:
      key = '/'.join(
        str(p.key) if hasattr(p, 'key') else str(p.idx) for p in path
      )
      param_dict[key] = np.array(value)
    np.savez_compressed(params_file, **param_dict)

    # Save head params
    ema_sd = {
      k.replace('module.', '', 1): v for k, v in ckpt.ema_state_dict.items()
    }
    prefix = 'output_heads.energyandforcehead.head.energy_block.'
    head_dict = {}
    for idx in [0, 2, 4]:
      for suffix in ['weight', 'weights', 'bias']:
        key = f'{prefix}{idx}.{suffix}'
        if key in ema_sd:
          head_dict[f'energy_block_{idx}_{suffix}'] = ema_sd[key].cpu().numpy()
    if head_dict:
      np.savez_compressed(
        os.path.join(_CHECKPOINT_DIR, f'{model_name}_head.npz'), **head_dict
      )

    # Save config
    mc = (
      ckpt.model_config.get('backbone', ckpt.model_config)
      if isinstance(ckpt.model_config, dict)
      else ckpt.model_config
    )
    get = (
      mc.get if isinstance(mc, dict) else lambda k, d=None: getattr(mc, k, d)
    )
    heads_cfg = (
      ckpt.model_config.get('heads', {})
      if isinstance(ckpt.model_config, dict)
      else {}
    )
    head_dm = None
    if isinstance(heads_cfg, dict):
      for _, hcfg in heads_cfg.items():
        if isinstance(hcfg, dict):
          dm = hcfg.get('dataset_mapping', None)
          if dm and isinstance(dm, dict):
            head_dm = dict(dm)

    cfg_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
    cfg_dict = {
      k: (list(v) if isinstance(v, list | tuple) else v)
      for k, v in cfg_dict.items()
    }
    if head_dm:
      cfg_dict['head_dataset_mapping'] = head_dm
    with open(
      os.path.join(_CHECKPOINT_DIR, f'{model_name}_config.json'), 'w'
    ) as f:
      json.dump(cfg_dict, f, indent=2)

  except Exception:
    pass  # Caching is best-effort


def _load_native(model_name: str, head_dataset: str):
  """Load from pre-converted native .npz checkpoint (no torch needed)."""
  import json

  params_file = os.path.join(_CHECKPOINT_DIR, f'{model_name}.npz')
  config_file = os.path.join(_CHECKPOINT_DIR, f'{model_name}_config.json')
  head_file = os.path.join(_CHECKPOINT_DIR, f'{model_name}_head.npz')

  if not os.path.exists(params_file) or not os.path.exists(config_file):
    return None

  # Load config
  with open(config_file) as f:
    cfg = json.load(f)

  config = UMAMoEConfig(
    max_num_elements=cfg['max_num_elements'],
    sphere_channels=cfg['sphere_channels'],
    lmax=cfg['lmax'],
    mmax=cfg['mmax'],
    num_layers=cfg['num_layers'],
    hidden_channels=cfg['hidden_channels'],
    cutoff=cfg['cutoff'],
    edge_channels=cfg['edge_channels'],
    num_distance_basis=cfg['num_distance_basis'],
    norm_type=cfg['norm_type'],
    act_type=cfg['act_type'],
    ff_type=cfg['ff_type'],
    chg_spin_emb_type=cfg['chg_spin_emb_type'],
    dataset_list=cfg.get('dataset_list', []),
    num_experts=cfg.get('num_experts', 32),
    use_composition_embedding=cfg.get('use_composition_embedding', True),
    model_version=cfg.get('model_version', 1.1),
    routing_hidden_channels=cfg.get('routing_hidden_channels', 64),
  )

  # Load params — unflatten from key/value pairs
  data = np.load(params_file, allow_pickle=False)
  params = {'params': {}}
  for key in data.files:
    parts = key.split('/')
    # Skip the leading 'params' prefix
    if parts[0] == 'params':
      parts = parts[1:]
    d = params['params']
    for p in parts[:-1]:
      if p not in d:
        d[p] = {}
      d = d[p]
    d[parts[-1]] = jnp.array(data[key])

  # Load head params
  head_params = None
  if os.path.exists(head_file):
    hd = np.load(head_file, allow_pickle=False)
    head_mapping = cfg.get('head_dataset_mapping', None)

    # Check if MoE head (3D weights)
    w0_key = (
      'energy_block_0_weights'
      if 'energy_block_0_weights' in hd.files
      else 'energy_block_0_weight'
    )
    is_moe_head = w0_key in hd.files and hd[w0_key].ndim == 3

    if is_moe_head:
      # Select the correct dataset expert
      if head_mapping:
        sorted_ds = sorted(head_mapping.keys())
      else:
        sorted_ds = sorted(config.dataset_list)
      ds_idx = sorted_ds.index(head_dataset) if head_dataset in sorted_ds else 0

      def _sel(key_base):
        for suffix in ['weights', 'weight']:
          k = f'{key_base}_{suffix}'
          if k in hd.files:
            w = hd[k]
            if w.ndim == 3:
              return jnp.array(w[ds_idx].T)  # [out, in] -> [in, out]
            return jnp.array(w.T) if w.ndim == 2 else jnp.array(w)
        return None

      head_params = {
        'params': {
          'linear_0': {
            'kernel': _sel('energy_block_0'),
            'bias': jnp.array(hd['energy_block_0_bias']),
          },
          'linear_1': {
            'kernel': _sel('energy_block_2'),
            'bias': jnp.array(hd['energy_block_2_bias']),
          },
          'linear_2': {
            'kernel': _sel('energy_block_4'),
            'bias': jnp.array(hd['energy_block_4_bias']),
          },
        }
      }
    else:
      # Standard head — transpose [out, in] -> [in, out]
      head_params = {
        'params': {
          'linear_0': {
            'kernel': jnp.array(hd['energy_block_0_weight'].T),
            'bias': jnp.array(hd['energy_block_0_bias']),
          },
          'linear_1': {
            'kernel': jnp.array(hd['energy_block_2_weight'].T),
            'bias': jnp.array(hd['energy_block_2_bias']),
          },
          'linear_2': {
            'kernel': jnp.array(hd['energy_block_4_weight'].T),
            'bias': jnp.array(hd['energy_block_4_bias']),
          },
        }
      }

  return config, params, head_params


def load_pretrained(
  checkpoint_path: str,
  head_dataset: str = 'omat',
) -> Tuple[UMAMoEConfig, Dict]:
  """Load a pretrained UMA checkpoint for use with UMAMoEBackbone.

  This is the primary way to load pretrained models. It preserves the
  full MoE architecture — all expert weights, the routing MLP, and
  the composition embedding. The model runs expert mixing during the
  forward pass, so it works for any system without re-loading.

  Tries to load from pre-converted native .npz checkpoints first
  (no torch dependency). Falls back to converting from .pt if needed.

  Args:
      checkpoint_path: Path to a checkpoint file, or a model name
          ('uma-s-1p1', 'uma-s-1p2', 'uma-m-1p1').
      head_dataset: Dataset for the energy head. For models with MoE heads
          (uma-s-1p2), this selects which dataset's head expert to use.
          For single-head models (uma-s-1p1, uma-m-1p1), this is ignored.

  Returns:
      Tuple of (config, params, head_params) where:
        - config: UMAMoEConfig matching the checkpoint
        - params: Flax params dict for UMAMoEBackbone.apply()
        - head_params: Flax params dict for MLPEnergyHead.apply(),
          loaded from the checkpoint's energy head weights

  Example:
      >>> from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
      >>> from jax_md._nn.uma.heads import MLPEnergyHead
      >>> config, params, head_params = load_pretrained('uma-s-1p1')
      >>> model = UMAMoEBackbone(config=config)
      >>> head = MLPEnergyHead(sphere_channels=config.sphere_channels,
      ...                      hidden_channels=config.hidden_channels)
      >>> emb = model.apply(params, positions, Z, batch, edge_index,
      ...                   edge_vec, charge, spin, dataset_idx)
      >>> energy = head.apply(head_params, emb['node_embedding'], batch, 1)
  """
  from jax_md._nn.uma.pretrained import PRETRAINED_MODELS

  # Try native .npz checkpoint first (no torch needed)
  if checkpoint_path in PRETRAINED_MODELS:
    result = _load_native(checkpoint_path, head_dataset)
    if result is not None:
      return result

  # Also try if checkpoint_path points to a directory with .npz files
  if os.path.isdir(checkpoint_path):
    name = os.path.basename(checkpoint_path)
    result = _load_native(name, head_dataset)
    if result is not None:
      return result

  # Fall back to .pt conversion (requires torch)
  from jax_md._nn.uma.pretrained import (
    download_pretrained,
    load_checkpoint_raw,
    extract_config,
    convert_checkpoint,
    PRETRAINED_MODELS,
  )

  original_name = checkpoint_path  # Save for caching
  if checkpoint_path in PRETRAINED_MODELS:
    checkpoint_path = download_pretrained(checkpoint_path)

  ckpt = load_checkpoint_raw(checkpoint_path)
  cfg_base = extract_config(ckpt)
  mc = (
    ckpt.model_config['backbone']
    if isinstance(ckpt.model_config, dict)
    else ckpt.model_config
  )
  get = mc.get if isinstance(mc, dict) else lambda k, d=None: getattr(mc, k, d)

  config = UMAMoEConfig(
    max_num_elements=cfg_base.max_num_elements,
    sphere_channels=cfg_base.sphere_channels,
    lmax=cfg_base.lmax,
    mmax=cfg_base.mmax,
    num_layers=cfg_base.num_layers,
    hidden_channels=cfg_base.hidden_channels,
    cutoff=cfg_base.cutoff,
    edge_channels=cfg_base.edge_channels,
    num_distance_basis=cfg_base.num_distance_basis,
    norm_type=cfg_base.norm_type,
    act_type=cfg_base.act_type,
    ff_type=cfg_base.ff_type,
    chg_spin_emb_type=cfg_base.chg_spin_emb_type,
    dataset_list=cfg_base.dataset_list,
    num_experts=get('num_experts', 32),
    use_composition_embedding=get('use_composition_embedding', True),
    model_version=float(get('model_version', 1.1)),
  )

  _, params, metadata = convert_checkpoint(
    checkpoint_path,
    use_ema=True,
  )

  # Infer routing_hidden_channels from converted weights
  p = params.get('params', params)
  if 'routing_mlp' in p and 'layers_0' in p['routing_mlp']:
    kernel = p['routing_mlp']['layers_0']['kernel']
    if hasattr(kernel, 'shape'):
      config = UMAMoEConfig(
        **{
          f: getattr(config, f)
          for f in config.__dataclass_fields__
          if f != 'routing_hidden_channels'
        },
        routing_hidden_channels=kernel.shape[1],
      )

  # Build head params from checkpoint's energy_block weights
  # Conversion produces keys like energy_block_0_kernel, energy_block_0_bias, etc.
  # For standard heads: 2D weights [out, in] (already transposed to [in, out])
  # For MoE heads: 3D weights [num_ds_experts, out, in] — select by dataset
  head_params = None
  if metadata.head_params:
    hp = metadata.head_params
    w0 = hp.get('energy_block_0_kernel', None)

    # Detect MoE head (3D weights)
    is_moe_head = w0 is not None and w0.ndim == 3
    if is_moe_head:
      # MoE head: select expert for the requested dataset
      # DatasetSpecificMoEWrapper uses sorted dataset names from dataset_mapping
      # The number of experts in head weights tells us the true dataset count
      n_head_experts = w0.shape[0]

      # Try to get the full dataset list from the checkpoint's head config
      heads_cfg = ckpt.model_config.get('heads', {})
      head_ds = None
      if isinstance(heads_cfg, dict):
        for _, hcfg in heads_cfg.items():
          if isinstance(hcfg, dict):
            dm = hcfg.get('dataset_mapping', None)
            if dm and isinstance(dm, dict):
              head_ds = sorted(dm.keys())
            dn = hcfg.get('dataset_names', None)
            if dn and not head_ds:
              head_ds = sorted(
                list(dn) if not isinstance(dn, str) else eval(dn)
              )

      if head_ds is None or len(head_ds) != n_head_experts:
        # Fallback: infer from state dict dataset_emb_dict keys
        ema_keys = list(ckpt.ema_state_dict.keys())
        ds_from_keys = set()
        for k in ema_keys:
          if 'dataset_emb_dict.' in k:
            parts = k.split('.')
            for i, p in enumerate(parts):
              if p == 'dataset_emb_dict' and i + 1 < len(parts):
                ds_from_keys.add(parts[i + 1])
        if len(ds_from_keys) == n_head_experts:
          head_ds = sorted(ds_from_keys)
        else:
          head_ds = sorted(config.dataset_list or [])

      ds_idx = head_ds.index(head_dataset) if head_dataset in head_ds else 0

      def _select_expert(w):
        if w is not None and w.ndim == 3:
          # w is [num_experts, out, in] — select expert and transpose to [in, out]
          return w[ds_idx].T
        return w

      head_params = {
        'params': {
          'linear_0': {
            'kernel': _select_expert(hp.get('energy_block_0_kernel')),
            'bias': hp.get('energy_block_0_bias'),
          },
          'linear_1': {
            'kernel': _select_expert(hp.get('energy_block_2_kernel')),
            'bias': hp.get('energy_block_2_bias'),
          },
          'linear_2': {
            'kernel': _select_expert(hp.get('energy_block_4_kernel')),
            'bias': hp.get('energy_block_4_bias'),
          },
        }
      }
    else:
      # Standard head: 2D weights (already in correct format)
      head_params = {
        'params': {
          'linear_0': {
            'kernel': hp.get('energy_block_0_kernel'),
            'bias': hp.get('energy_block_0_bias'),
          },
          'linear_1': {
            'kernel': hp.get('energy_block_2_kernel'),
            'bias': hp.get('energy_block_2_bias'),
          },
          'linear_2': {
            'kernel': hp.get('energy_block_4_kernel'),
            'bias': hp.get('energy_block_4_bias'),
          },
        }
      }

    # Remove None entries
    for layer in list(head_params['params'].keys()):
      head_params['params'][layer] = {
        k: v for k, v in head_params['params'][layer].items() if v is not None
      }

  # Cache as native .npz for instant loading next time
  _cache_native(original_name, config, params, metadata, ckpt)

  return config, params, head_params
