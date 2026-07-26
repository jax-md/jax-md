# Credit to https://github.com/torchmd/torchmd-net
#
# Parameters are stored fp32 and cast to fp64 under jax_enable_x64.
# LocalEnergyHead rounds 1/3 in fp32 to match the upstream Warp kernels.

import json
from functools import partial
from importlib.resources import files
from os import PathLike
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition
from jax_md._nn import weights

jax.config.update('jax_default_matmul_precision', 'highest')

ACEFF_MODEL_PATHS = {
  'aceff-jax-1.1': files('jax_md._nn.aceff') / 'aceff_v1.1.eqx',
  'aceff-jax-2.0': files('jax_md._nn.aceff') / 'aceff_v2.0.eqx',
}


def neighbor_list_featurizer(displacement_fn, *, cutoff: float):
  def featurize(position, neighbor, **kwargs):
    num_atoms = position.shape[0]
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    mask = partition.neighbor_list_mask(neighbor, True)
    if partition.is_sparse(neighbor.format):
      receivers, senders = jnp.asarray(neighbor.idx, dtype=jnp.int32)
      src = jnp.where(mask, senders, 0)
      dst_raw = receivers
    else:
      idx = jnp.asarray(neighbor.idx, dtype=jnp.int32)
      src = jnp.repeat(atom_ids, idx.shape[1])
      dst_raw = idx.reshape(-1)
      mask = mask.reshape(-1)
    fallback = jnp.where(num_atoms > 1, (src + 1) % num_atoms, src)
    dst = jnp.where(mask, dst_raw, fallback)

    edge_src = jnp.concatenate([atom_ids, src])
    edge_dst = jnp.concatenate([atom_ids, dst])
    edge_mask = jnp.concatenate([jnp.ones((num_atoms,), dtype=bool), mask])

    raw_vec = position[edge_src] - position[edge_dst]
    d = jax.vmap(partial(displacement_fn, **kwargs))
    offsets = d(position[edge_src], position[edge_dst]) - raw_vec
    far_shift = jnp.zeros_like(raw_vec).at[:, 0].set(float(cutoff) + 1.0)
    offsets = jnp.where(edge_mask[:, None], offsets, far_shift - raw_vec)
    return edge_src, edge_dst, offsets, edge_mask

  return featurize


def unique_pairs(num_atoms: int):
  pair_src, pair_dst = np.triu_indices(int(num_atoms), k=1)
  return (
    jnp.asarray(pair_src, dtype=jnp.int32),
    jnp.asarray(pair_dst, dtype=jnp.int32),
  )


def safe_norm(x, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24):
  return jnp.sqrt(
    jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps)
  )


def cosine_cutoff(d, cutoff, cutoff_lower=0.0):
  if cutoff_lower > 0:
    x = 2.0 * (d - cutoff_lower) / (cutoff - cutoff_lower) + 1.0
    c = 0.5 * (jnp.cos(jnp.pi * x) + 1.0)
    return jnp.where((d < cutoff) & (d > cutoff_lower), c, 0.0)
  else:
    c = 0.5 * (jnp.cos(d * jnp.pi / cutoff) + 1.0)
    return jnp.where(d < cutoff, c, 0.0)


def decompose_tensor(X):
  """Decompose into scalar, antisymmetric, and symmetric components."""
  antisymmetric = 0.5 * (X - jnp.swapaxes(X, 1, 2))
  symmetric_full = X - antisymmetric
  scalar = jnp.diagonal(X, axis1=1, axis2=2).mean(axis=-1)
  identity = jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]
  symmetric = symmetric_full - scalar[:, None, None, :] * identity
  return scalar, antisymmetric, symmetric


def vector_to_skewtensor(vec):
  """[N, 3, F] -> [N, 3, 3, F] skew-symmetric."""
  N, _, F = vec.shape
  zero = jnp.zeros((N, F), dtype=vec.dtype)
  tensor = jnp.stack(
    [
      zero,
      -vec[:, 2, :],
      vec[:, 1, :],
      vec[:, 2, :],
      zero,
      -vec[:, 0, :],
      -vec[:, 1, :],
      vec[:, 0, :],
      zero,
    ],
    axis=1,
  )
  return tensor.reshape(N, 3, 3, F)


def skewtensor_to_vector(A):
  """[N, 3, 3, F] -> [N, 3, F]."""
  A_flat = A.reshape(A.shape[0], 9, A.shape[3])
  return 0.5 * jnp.stack(
    [
      A_flat[:, 7] - A_flat[:, 5],
      A_flat[:, 2] - A_flat[:, 6],
      A_flat[:, 3] - A_flat[:, 1],
    ],
    axis=1,
  )


def outer_to_symtensor(T):
  """Symmetrize and remove trace. [N, 3, 3, F] -> [N, 3, 3, F]."""
  symmetric = 0.5 * (T + jnp.swapaxes(T, 1, 2))
  scalar = jnp.diagonal(T, axis1=1, axis2=2).mean(axis=-1)
  identity = jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]
  return symmetric - scalar[:, None, None, :] * identity


def tensor_matmul_o3(Y, msg):
  """O(3)-equivariant contraction: Y*msg + msg*Y."""
  Yp = jnp.transpose(Y, (0, 3, 1, 2))
  Mp = jnp.transpose(msg, (0, 3, 1, 2))
  return jnp.transpose(Mp @ Yp, (0, 2, 3, 1)) + jnp.transpose(
    Yp @ Mp, (0, 2, 3, 1)
  )


def tensor_matmul_so3(Y, msg):
  """SO(3)-equivariant contraction: Y*msg."""
  Yp = jnp.transpose(Y, (0, 3, 1, 2))
  Mp = jnp.transpose(msg, (0, 3, 1, 2))
  return jnp.transpose(Yp @ Mp, (0, 2, 3, 1))


class Linear(eqx.Module):
  kernel: Any
  bias: Any

  def __init__(self, d_in, d_out, *, bias=True):
    self.kernel = jnp.zeros((d_in, d_out), dtype=jnp.float32)
    self.bias = jnp.zeros((d_out,), dtype=jnp.float32) if bias else None

  def __call__(self, x):
    out = x @ self.kernel
    if self.bias is not None:
      out = out + self.bias
    return out


class MLP(eqx.Module):
  layers: tuple[Linear, ...]

  def __init__(self, d_in, dims):
    layers = []
    for d_out in dims:
      layers.append(Linear(d_in, int(d_out)))
      d_in = int(d_out)
    self.layers = tuple(layers)

  def __call__(self, x):
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i < len(self.layers) - 1:
        x = jax.nn.silu(x)
    return x


class LayerNorm(eqx.Module):
  weight: Any
  bias: Any
  eps: float = eqx.field(static=True)

  def __init__(self, dim, *, eps: float = 1.0e-5):
    self.weight = jnp.zeros((dim,), dtype=jnp.float32)
    self.bias = jnp.zeros((dim,), dtype=jnp.float32)
    self.eps = float(eps)

  def __call__(self, x):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return self.weight * (x - mean) * jax.lax.rsqrt(var + self.eps) + self.bias


class TensorEmbedding(eqx.Module):
  distance_proj1: Linear
  distance_proj2: Linear
  distance_proj3: Linear
  emb: Any
  emb2: Linear
  init_norm: LayerNorm
  linears_scalar: tuple[Linear, ...]
  linears_tensor: tuple[Linear, ...]
  cutoff: float = eqx.field(static=True)
  cutoff_lower: float = eqx.field(static=True)

  def __init__(self, num_rbf, hidden, *, cutoff, cutoff_lower, eps):
    self.distance_proj1 = Linear(num_rbf, hidden)
    self.distance_proj2 = Linear(num_rbf, hidden)
    self.distance_proj3 = Linear(num_rbf, hidden)
    self.emb = jnp.zeros((hidden, hidden), dtype=jnp.float32)
    self.emb2 = Linear(2 * hidden, hidden)
    self.init_norm = LayerNorm(hidden, eps=eps)
    self.linears_scalar = (
      Linear(hidden, 2 * hidden),
      Linear(2 * hidden, 3 * hidden),
    )
    self.linears_tensor = tuple(
      Linear(hidden, hidden, bias=False) for _ in range(3)
    )
    self.cutoff = float(cutoff)
    self.cutoff_lower = float(cutoff_lower)

  def __call__(
    self,
    species,
    edge_src,
    edge_dst,
    distances,
    unit_vectors,
    radial_features,
    *,
    edge_mask=None,
  ):
    num_features = self.emb.shape[1]
    num_atoms = species.shape[0]

    Zi = self.emb[species[edge_src]]
    Zj = self.emb[species[edge_dst]]
    Zij = self.emb2(
      jnp.concatenate([Zi, Zj], axis=-1),
    )

    distance_projection_1 = self.distance_proj1(radial_features)
    distance_projection_2 = self.distance_proj2(radial_features)
    distance_projection_3 = self.distance_proj3(radial_features)

    cutoff_values = cosine_cutoff(distances, self.cutoff, self.cutoff_lower)
    if edge_mask is not None:
      cutoff_values = cutoff_values * edge_mask
    species_pair_features = cutoff_values[:, None] * Zij

    edge_features = species_pair_features[:, None, :] * jnp.stack(
      [distance_projection_1, distance_projection_2, distance_projection_3],
      axis=1,
    )

    scalar_messages = edge_features[:, 0, :]
    scalar = jnp.zeros((num_atoms, num_features), dtype=scalar_messages.dtype)
    scalar = scalar.at[edge_src].add(scalar_messages)

    antisymmetric_messages = (
      edge_features[:, 1, :][:, None, :] * unit_vectors[:, :, None]
    )
    antisymmetric_vectors = jax.ops.segment_sum(
      antisymmetric_messages, edge_src, num_segments=num_atoms
    )

    outer = unit_vectors[:, :, None] * unit_vectors[:, None, :]
    symmetric_messages = (
      edge_features[:, 2, :][:, None, None, :] * outer[:, :, :, None]
    )
    symmetric = jax.ops.segment_sum(
      symmetric_messages, edge_src, num_segments=num_atoms
    )

    antisymmetric = vector_to_skewtensor(antisymmetric_vectors)
    symmetric = outer_to_symtensor(symmetric)
    identity = jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]
    tensor_features = (
      scalar[:, None, None, :] * identity + antisymmetric + symmetric
    )

    norm = (tensor_features**2).sum(axis=(1, 2))
    norm = self.init_norm(norm)
    for layer in self.linears_scalar:
      norm = jax.nn.silu(layer(norm))
    norm = norm.reshape(-1, 3, num_features)

    scalar_norm = norm[:, 0, :]
    antisymmetric_norm = norm[:, 1, :][:, None, None, :]
    symmetric_norm = norm[:, 2, :][:, None, None, :]

    scalar = self.linears_tensor[0](scalar) * scalar_norm
    antisymmetric = self.linears_tensor[1](antisymmetric) * antisymmetric_norm
    symmetric = self.linears_tensor[2](symmetric) * symmetric_norm

    return scalar[:, None, None, :] * identity + antisymmetric + symmetric


class PartialChargesHead(eqx.Module):
  q_mlp: MLP
  q_norm: LayerNorm

  def __init__(self, in_dim, mlp_dims, *, eps):
    self.q_mlp = MLP(in_dim, mlp_dims)
    self.q_norm = LayerNorm(in_dim, eps=eps)

  def _neural_charge_equilibration(
    self,
    partial_charges,
    charge_weights,
    total_charge=0.0,
  ):
    weights = charge_weights**2
    weight_sum = jnp.sum(weights, axis=0, keepdims=True) + 1.0e-6
    predicted_charge = jnp.sum(partial_charges, axis=0, keepdims=True)
    return partial_charges + (weights / weight_sum) * (
      total_charge - predicted_charge
    )

  def __call__(self, tensor_features, total_charge=0.0):
    scalar, antisymmetric, symmetric = decompose_tensor(tensor_features)
    charge_features = jnp.concatenate(
      [
        scalar,
        (antisymmetric**2).sum(axis=(1, 2)),
        (symmetric**2).sum(axis=(1, 2)),
      ],
      axis=-1,
    )

    charge_features = self.q_norm(charge_features)
    charge_features = self.q_mlp(charge_features)

    ncharge = charge_features.shape[-1] // 2
    partial_charges = charge_features[:, :ncharge]
    charge_weights = charge_features[:, ncharge:]
    return self._neural_charge_equilibration(
      partial_charges,
      charge_weights,
      total_charge,
    )


class AceFFLayer(eqx.Module):
  linears_scalar: tuple[Linear, ...]
  linears_tensor: tuple[Linear, ...]
  cutoff: float = eqx.field(static=True)
  cutoff_lower: float = eqx.field(static=True)
  group: str = eqx.field(static=True)
  edge_charge_features: bool = eqx.field(static=True)
  total_charge_interaction_scale: bool = eqx.field(static=True)

  def __init__(
    self,
    scalar_in,
    hidden,
    *,
    cutoff: float,
    cutoff_lower: float,
    group: str,
    edge_charge_features: bool,
    total_charge_interaction_scale: bool,
  ):
    self.linears_scalar = (
      Linear(scalar_in, hidden),
      Linear(hidden, 2 * hidden),
      Linear(2 * hidden, 3 * hidden),
    )
    self.linears_tensor = tuple(
      Linear(hidden, hidden, bias=False) for _ in range(6)
    )
    self.cutoff = float(cutoff)
    self.cutoff_lower = float(cutoff_lower)
    self.group = str(group)
    self.edge_charge_features = bool(edge_charge_features)
    self.total_charge_interaction_scale = bool(total_charge_interaction_scale)

  def __call__(
    self,
    tensor_features,
    partial_charges,
    total_charge,
    edge_src,
    edge_dst,
    distances,
    radial_features,
    *,
    edge_mask=None,
  ):
    num_atoms = tensor_features.shape[0]
    num_features = tensor_features.shape[3]

    cutoff_values = cosine_cutoff(distances, self.cutoff, self.cutoff_lower)
    if edge_mask is not None:
      cutoff_values = cutoff_values * edge_mask

    if self.edge_charge_features:
      source_charges = partial_charges[edge_src]
      neighbor_charges = partial_charges[edge_dst]
      edge_features = jnp.concatenate(
        [radial_features, source_charges, neighbor_charges],
        axis=-1,
      )
    else:
      edge_features = radial_features

    for layer in self.linears_scalar:
      edge_features = jax.nn.silu(layer(edge_features))
    edge_features = (edge_features * cutoff_values[:, None]).reshape(
      -1,
      3,
      num_features,
    )

    tensor_features = (
      tensor_features
      / ((tensor_features**2).sum(axis=(1, 2)) + 1)[:, None, None, :]
    )

    scalar, antisymmetric, symmetric = decompose_tensor(tensor_features)
    scalar = self.linears_tensor[0](scalar)
    antisymmetric = self.linears_tensor[1](antisymmetric)
    symmetric = self.linears_tensor[2](symmetric)
    identity = jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]
    projected_features = (
      scalar[:, None, None, :] * identity + antisymmetric + symmetric
    )

    antisymmetric_vectors = skewtensor_to_vector(antisymmetric)

    scalar_weights = edge_features[:, 0, :]
    vector_weights = edge_features[:, 1, :][:, None, :]
    tensor_weights = edge_features[:, 2, :][:, None, None, :]

    scalar_messages = scalar_weights * scalar[edge_dst]
    scalar_message = jnp.zeros(
      (num_atoms, num_features), dtype=tensor_features.dtype
    )
    scalar_message = scalar_message.at[edge_src].add(scalar_messages)

    antisymmetric_messages = vector_weights * antisymmetric_vectors[edge_dst]
    antisymmetric_message_vectors = jnp.zeros(
      (num_atoms, 3, num_features),
      dtype=tensor_features.dtype,
    )
    antisymmetric_message_vectors = antisymmetric_message_vectors.at[
      edge_src
    ].add(
      antisymmetric_messages,
    )

    symmetric_messages = tensor_weights * symmetric[edge_dst]
    symmetric_message = jnp.zeros(
      (num_atoms, 3, 3, num_features),
      dtype=tensor_features.dtype,
    )
    symmetric_message = symmetric_message.at[edge_src].add(symmetric_messages)

    antisymmetric_message = vector_to_skewtensor(antisymmetric_message_vectors)
    messages = (
      scalar_message[:, None, None, :] * identity
      + antisymmetric_message
      + symmetric_message
    )

    charge_factor = 1.0
    if self.total_charge_interaction_scale:
      charge_factor = 1.0 + 0.1 * jnp.asarray(
        total_charge,
        dtype=tensor_features.dtype,
      )

    if self.group == 'O(3)':
      updates = charge_factor * tensor_matmul_o3(projected_features, messages)
    else:
      updates = 2 * tensor_matmul_so3(projected_features, messages)

    scalar_update, antisymmetric_update, symmetric_update = decompose_tensor(
      updates
    )

    update_norm = (updates**2).sum(axis=(1, 2)) + 1
    scalar_update = scalar_update / update_norm
    antisymmetric_update = antisymmetric_update / update_norm[:, None, None, :]
    symmetric_update = symmetric_update / update_norm[:, None, None, :]

    scalar_update = self.linears_tensor[3](scalar_update)
    antisymmetric_update = self.linears_tensor[4](antisymmetric_update)
    symmetric_update = self.linears_tensor[5](symmetric_update)
    delta_features = (
      scalar_update[:, None, None, :] * identity
      + antisymmetric_update
      + symmetric_update
    )

    return (
      tensor_features
      + delta_features
      + charge_factor * tensor_matmul_so3(delta_features, delta_features)
    )


class LocalEnergyHead(eqx.Module):
  out_norm: LayerNorm
  linear: Linear
  output_network: MLP

  def __init__(self, hidden, output_network_dims, *, eps):
    self.out_norm = LayerNorm(3 * hidden, eps=eps)
    self.linear = Linear(3 * hidden, hidden)
    self.output_network = MLP(hidden, output_network_dims)

  def __call__(self, tensor_features):
    _, antisymmetric, symmetric = decompose_tensor(tensor_features)
    trace = jnp.diagonal(tensor_features, axis1=1, axis2=2).sum(axis=-1)
    warp_one_third = jnp.asarray(
      float(np.float32(1.0 / 3.0)),
      dtype=trace.dtype,
    )
    scalar_norm = warp_one_third * trace * trace
    energy_features = jnp.concatenate(
      [
        scalar_norm,
        (antisymmetric**2).sum(axis=(1, 2)),
        (symmetric**2).sum(axis=(1, 2)),
      ],
      axis=-1,
    )
    energy_features = self.out_norm(energy_features)
    energy_features = jax.nn.silu(self.linear(energy_features))
    energy_features = self.output_network(energy_features)
    return energy_features.squeeze(-1)


class CoulombHead(eqx.Module):
  qweights: Any
  coulomb_factor: float = eqx.field(static=True)
  coulomb_damp_cutoff: float = eqx.field(static=True)
  coulomb_cutoff: float | None = eqx.field(static=True)
  coulomb_epsilon_solvent: float = eqx.field(static=True)
  exp_minus_1: float = eqx.field(static=True)

  def __init__(
    self,
    qweights_dim,
    *,
    coulomb_factor: float,
    coulomb_damp_cutoff: float,
    coulomb_cutoff: float | None,
    coulomb_epsilon_solvent: float,
    exp_minus_1: float,
  ):
    self.qweights = jnp.zeros((qweights_dim,), dtype=jnp.float32)
    self.coulomb_factor = float(coulomb_factor)
    self.coulomb_damp_cutoff = float(coulomb_damp_cutoff)
    self.coulomb_cutoff = (
      None if coulomb_cutoff is None else float(coulomb_cutoff)
    )
    self.coulomb_epsilon_solvent = float(coulomb_epsilon_solvent)
    self.exp_minus_1 = float(exp_minus_1)

  def __call__(
    self,
    positions,
    partial_charges,
    *,
    displacement_fn,
    coulomb_edges=None,
  ):
    partial_charges = jnp.concatenate(partial_charges, axis=-1)
    if coulomb_edges is None:
      # No Coulomb neighbor list supplied: sum over all unique atom pairs.
      pair_src, pair_dst = unique_pairs(int(positions.shape[0]))
      pair_vectors = jax.vmap(displacement_fn)(
        positions[pair_src], positions[pair_dst]
      )
      pair_mask = None
    else:
      # The Coulomb neighbor list lists each undirected pair twice; keep
      # edge_src < edge_dst so every pair is counted once.
      edge_src, edge_dst, pair_vectors, edge_mask = coulomb_edges
      pair_src, pair_dst = edge_src, edge_dst
      pair_mask = edge_mask & (edge_src < edge_dst)

    distances = safe_norm(pair_vectors, axis=-1)
    damping_x = jnp.clip(
      distances / self.coulomb_damp_cutoff,
      0.0,
      1.0 - 1e-6,
    )
    damping = jnp.exp(-1.0 / (1.0 - damping_x**2)) / self.exp_minus_1
    cutoff_values = 1.0 - damping
    charge_products = partial_charges[pair_src] * partial_charges[pair_dst]
    weighted_charge_products = (charge_products * self.qweights[None, :]).sum(
      axis=-1
    ) / self.qweights.sum()
    if self.coulomb_cutoff is None:
      pair_energies = cutoff_values * weighted_charge_products / distances
    else:
      cutoff = self.coulomb_cutoff
      epsilon = self.coulomb_epsilon_solvent
      k_rf = (1.0 / cutoff**3) * (epsilon - 1.0) / (2.0 * epsilon + 1.0)
      c_rf = (1.0 / cutoff) * (3.0 * epsilon) / (2.0 * epsilon + 1.0)
      pair_energies = (
        cutoff_values
        * weighted_charge_products
        * (1.0 / distances + k_rf * distances**2 - c_rf)
      )
      pair_energies = jnp.where(distances < cutoff, pair_energies, 0.0)
    if pair_mask is not None:
      pair_energies = jnp.where(pair_mask, pair_energies, 0.0)
    return self.coulomb_factor * pair_energies.sum()


class AceFF(eqx.Module):
  rbf_betas: Any
  rbf_means: Any
  tensor_embedding: TensorEmbedding
  charge_predictor_0: PartialChargesHead | None
  layers: tuple[AceFFLayer, ...]
  charge_predictors_by_layer: tuple[PartialChargesHead, ...]
  local_energy_head: LocalEnergyHead
  coulomb_head: CoulombHead | None
  cutoff: float = eqx.field(static=True)
  ev_to_kjmol: float = eqx.field(static=True)
  cutoff_lower: float = eqx.field(static=True)
  alpha: float = eqx.field(static=True)
  neighbor_cell_atom_threshold: int = eqx.field(static=True)
  neighbor_cell_capacity_multiplier: float = eqx.field(static=True)

  def __init__(self, config):
    self.cutoff = config['cutoff']
    self.ev_to_kjmol = config['ev_to_kjmol']
    self.cutoff_lower = config['cutoff_lower']
    self.alpha = config['alpha']
    group = config['group']
    charge_predictors = config['charge_predictors']
    edge_charge_features = config['edge_charge_features']
    total_charge_interaction_scale = config['total_charge_interaction_scale']
    coulomb_energy = config['coulomb_energy']
    coulomb_factor = config['coulomb_factor']
    coulomb_damp_cutoff = config['coulomb_damp_cutoff']
    coulomb_cutoff = config['coulomb_cutoff']
    coulomb_epsilon_solvent = config['coulomb_epsilon_solvent']
    exp_minus_1 = config['exp_minus_1']
    self.neighbor_cell_atom_threshold = config['neighbor_cell_atom_threshold']
    self.neighbor_cell_capacity_multiplier = config[
      'neighbor_cell_capacity_multiplier'
    ]

    hidden = config['hidden_channels']
    num_rbf = config['num_rbf']
    num_layers = config['num_interaction_layers']
    eps = config['layernorm_eps']
    charge_mlp_dims = config['charge_mlp_dims'] if charge_predictors else []
    n_charge = charge_mlp_dims[-1] // 2 if charge_predictors else 0
    scalar_in = num_rbf + (2 * n_charge if edge_charge_features else 0)

    self.rbf_betas = jnp.zeros((num_rbf,), dtype=jnp.float32)
    self.rbf_means = jnp.zeros((num_rbf,), dtype=jnp.float32)
    self.tensor_embedding = TensorEmbedding(
      num_rbf,
      hidden,
      cutoff=self.cutoff,
      cutoff_lower=self.cutoff_lower,
      eps=eps,
    )
    self.charge_predictor_0 = (
      PartialChargesHead(3 * hidden, charge_mlp_dims, eps=eps)
      if charge_predictors
      else None
    )
    self.layers = tuple(
      AceFFLayer(
        scalar_in,
        hidden,
        cutoff=self.cutoff,
        cutoff_lower=self.cutoff_lower,
        group=group,
        edge_charge_features=edge_charge_features,
        total_charge_interaction_scale=total_charge_interaction_scale,
      )
      for _ in range(num_layers)
    )
    self.charge_predictors_by_layer = tuple(
      PartialChargesHead(3 * hidden, charge_mlp_dims, eps=eps)
      for _ in range(num_layers if charge_predictors else 0)
    )
    self.local_energy_head = LocalEnergyHead(
      hidden, config['output_network_dims'], eps=eps
    )
    self.coulomb_head = (
      CoulombHead(
        config['coulomb_qweights_dim'],
        coulomb_factor=coulomb_factor,
        coulomb_damp_cutoff=coulomb_damp_cutoff,
        coulomb_cutoff=coulomb_cutoff,
        coulomb_epsilon_solvent=coulomb_epsilon_solvent,
        exp_minus_1=exp_minus_1,
      )
      if coulomb_energy
      else None
    )

  def edge_features(
    self,
    positions,
    edge_src,
    edge_dst,
    offsets,
    edge_mask,
  ):
    edge_vectors = positions[edge_src] - positions[edge_dst] + offsets
    distances = jnp.where(
      edge_src == edge_dst,
      0.0,
      safe_norm(edge_vectors, axis=-1, eps=1.0e-30),
    )
    unit_vectors = edge_vectors / jnp.where(
      (edge_src == edge_dst)[:, None],
      1.0,
      jnp.maximum(distances[:, None], 1e-8),
    )
    edge_mask = jnp.asarray(edge_mask, dtype=positions.dtype)
    distances_expanded = distances[..., None]
    cutoff_values = cosine_cutoff(distances_expanded, self.cutoff, 0.0)
    radial_features = (
      cutoff_values
      * jnp.exp(
        -self.rbf_betas
        * (
          jnp.exp(self.alpha * (-distances_expanded + self.cutoff_lower))
          - self.rbf_means
        )
        ** 2
      )
      * edge_mask[:, None]
    )
    return distances, unit_vectors, radial_features, edge_mask

  def local_node_energies_and_charges(
    self,
    positions,
    species,
    edge_src,
    edge_dst,
    offsets,
    edge_mask,
    total_charge,
  ):
    distances, unit_vectors, radial_features, edge_mask = self.edge_features(
      positions,
      edge_src,
      edge_dst,
      offsets,
      edge_mask,
    )
    tensor_features = self.tensor_embedding(
      species,
      edge_src,
      edge_dst,
      distances,
      unit_vectors,
      radial_features,
      edge_mask=edge_mask,
    )

    partial_charges = None
    charge_history = []
    if self.charge_predictor_0 is not None:
      partial_charges = self.charge_predictor_0(
        tensor_features,
        total_charge,
      )
      charge_history.append(partial_charges)

    for layer_index, layer in enumerate(self.layers):
      tensor_features = layer(
        tensor_features,
        partial_charges,
        total_charge,
        edge_src,
        edge_dst,
        distances,
        radial_features,
        edge_mask=edge_mask,
      )
      if self.charge_predictor_0 is not None:
        partial_charges = self.charge_predictors_by_layer[layer_index](
          tensor_features,
          total_charge,
        )
        charge_history.append(partial_charges)

    return self.local_energy_head(tensor_features), charge_history

  def __call__(
    self,
    positions,
    species,
    *,
    displacement_fn=None,
    neighbors=None,
    coulomb_neighbors=None,
    total_charge=0.0,
  ):
    if displacement_fn is None or neighbors is None:
      raise ValueError(
        'AceFF requires a displacement_fn and a neighbor list. Build them '
        'with energy.aceff_neighbor_list.'
      )
    featurize = neighbor_list_featurizer(
      displacement_fn, cutoff=float(self.cutoff)
    )
    edge_src, edge_dst, offsets, edge_mask = featurize(positions, neighbors)
    node_energies, partial_charges = self.local_node_energies_and_charges(
      positions,
      species,
      edge_src,
      edge_dst,
      offsets,
      edge_mask,
      total_charge,
    )
    local_energy = jnp.sum(node_energies)
    if self.coulomb_head is None:
      return local_energy

    coulomb_edges = None
    if coulomb_neighbors is not None:
      coulomb_cutoff = self.coulomb_head.coulomb_cutoff
      coulomb_featurize = neighbor_list_featurizer(
        displacement_fn,
        cutoff=float(coulomb_cutoff)
        if coulomb_cutoff is not None
        else float(self.cutoff),
      )
      c_src, c_dst, c_shifts, c_mask = coulomb_featurize(
        positions, coulomb_neighbors
      )
      c_vectors = positions[c_src] - positions[c_dst] + c_shifts
      coulomb_edges = (c_src, c_dst, c_vectors, c_mask)

    coulomb_energy = self.coulomb_head(
      positions,
      partial_charges,
      displacement_fn=displacement_fn,
      coulomb_edges=coulomb_edges,
    )
    return local_energy + coulomb_energy


def load_model(
  model: str = 'aceff-jax-2.0',
  *,
  model_path: str | PathLike | None = None,
  dtype=None,
):
  if model_path is not None:
    path = weights.resolve_checkpoint(str(model_path), allow_cache=False)
  else:
    path = weights.resolve_checkpoint(str(ACEFF_MODEL_PATHS[model]))

  if dtype is None:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  with path.open('rb') as handle:
    config = dict(json.loads(handle.readline().decode()))
    template = AceFF(config)
    loaded = eqx.tree_deserialise_leaves(handle, template)
    return jax.tree_util.tree_map(
      lambda x: (
        x.astype(dtype)
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
        else x
      ),
      loaded,
    )
