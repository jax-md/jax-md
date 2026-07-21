# Credit to https://github.com/torchmd/torchmd-net

import json
from functools import partial
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition

jax.config.update('jax_default_matmul_precision', 'highest')

ACEFF_MODEL_PATHS = {
  'aceff-jax-1.1': files('jax_md._nn.aceff') / 'aceff_v1.1.eqx',
  'aceff-jax-2.0': files('jax_md._nn.aceff') / 'aceff_v2.0.eqx',
}
ACEFF_MODEL_NAMES = tuple(ACEFF_MODEL_PATHS)


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
    pbc_shifts = d(position[edge_src], position[edge_dst]) - raw_vec
    far_shift = jnp.zeros_like(raw_vec).at[:, 0].set(float(cutoff) + 1.0)
    pbc_shifts = jnp.where(edge_mask[:, None], pbc_shifts, far_shift - raw_vec)
    return edge_src, edge_dst, pbc_shifts, edge_mask

  return featurize


def unique_pairs(num_atoms: int):
  pair_src, pair_dst = np.triu_indices(int(num_atoms), k=1)
  return (
    jnp.asarray(pair_src, dtype=jnp.int32),
    jnp.asarray(pair_dst, dtype=jnp.int32),
  )


def cosine_cutoff(d, cutoff, cutoff_lower=0.0):
  if cutoff_lower > 0:
    x = 2.0 * (d - cutoff_lower) / (cutoff - cutoff_lower) + 1.0
    c = 0.5 * (jnp.cos(jnp.pi * x) + 1.0)
    return jnp.where((d < cutoff) & (d > cutoff_lower), c, 0.0)
  else:
    c = 0.5 * (jnp.cos(d * jnp.pi / cutoff) + 1.0)
    return jnp.where(d < cutoff, c, 0.0)


def _scalar_to_tensor(scalar):
  """Scalar [N, F] -> diagonal tensor [N, 3, 3, F]."""
  eye = jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]
  return scalar[:, None, None, :] * eye


def decompose_tensor(X):
  """Decompose into scalar, antisymmetric, and symmetric components."""
  antisymmetric = 0.5 * (X - jnp.swapaxes(X, 1, 2))
  symmetric_full = X - antisymmetric
  scalar = jnp.diagonal(X, axis1=1, axis2=2).mean(axis=-1)
  symmetric = symmetric_full - _scalar_to_tensor(scalar)
  return scalar, antisymmetric, symmetric


def compose_tensor(scalar, antisymmetric, symmetric):
  return _scalar_to_tensor(scalar) + antisymmetric + symmetric


def tensor_norm(X):
  """Frobenius norm squared: [N, 3, 3, F] -> [N, F]."""
  return (X**2).sum(axis=(1, 2))


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
  return symmetric - _scalar_to_tensor(scalar)


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

  def __init__(self, kernel, bias=None):
    self.kernel = kernel
    self.bias = bias

  def __call__(self, x):
    out = x @ self.kernel
    if self.bias is not None:
      out = out + self.bias
    return out


class LayerNorm(eqx.Module):
  weight: Any
  bias: Any
  eps: float = eqx.field(static=True)

  def __init__(self, weight, bias, *, eps: float = 1.0e-5):
    self.weight = weight
    self.bias = bias
    self.eps = float(eps)

  def __call__(self, x):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return self.weight * (x - mean) * jax.lax.rsqrt(var + self.eps) + self.bias


def safe_norm(x, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24):
  return jnp.sqrt(
    jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps)
  )


class TensorEmbedding(eqx.Module):
  weights: Any
  cutoff: float = eqx.field(static=True)
  cutoff_lower: float = eqx.field(static=True)

  def __init__(self, weights, *, cutoff: float, cutoff_lower: float):
    self.weights = weights
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
    weights = self.weights
    num_features = weights['emb'].shape[1]
    num_atoms = species.shape[0]

    Zi = weights['emb'][species[edge_src]]
    Zj = weights['emb'][species[edge_dst]]
    Zij = weights['emb2'](
      jnp.concatenate([Zi, Zj], axis=-1),
    )

    distance_projection_1 = weights['distance_proj1'](radial_features)
    distance_projection_2 = weights['distance_proj2'](radial_features)
    distance_projection_3 = weights['distance_proj3'](radial_features)

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
    antisymmetric_vectors = jnp.zeros(
      (num_atoms, 3, num_features),
      dtype=scalar_messages.dtype,
    )
    antisymmetric_vectors = antisymmetric_vectors.at[edge_src].add(
      antisymmetric_messages
    )

    outer = unit_vectors[:, :, None] * unit_vectors[:, None, :]
    symmetric_messages = (
      edge_features[:, 2, :][:, None, None, :] * outer[:, :, :, None]
    )
    symmetric = jnp.zeros(
      (num_atoms, 3, 3, num_features),
      dtype=scalar_messages.dtype,
    )
    symmetric = symmetric.at[edge_src].add(symmetric_messages)

    antisymmetric = vector_to_skewtensor(antisymmetric_vectors)
    symmetric = outer_to_symtensor(symmetric)
    tensor_features = compose_tensor(scalar, antisymmetric, symmetric)

    norm = tensor_norm(tensor_features)
    norm = weights['init_norm'](norm)
    for layer in weights['linears_scalar']:
      norm = jax.nn.silu(layer(norm))
    norm = norm.reshape(-1, 3, num_features)

    scalar_norm = norm[:, 0, :]
    antisymmetric_norm = norm[:, 1, :][:, None, None, :]
    symmetric_norm = norm[:, 2, :][:, None, None, :]

    scalar = weights['linears_tensor'][0](scalar) * scalar_norm
    antisymmetric = (
      weights['linears_tensor'][1](antisymmetric) * antisymmetric_norm
    )
    symmetric = weights['linears_tensor'][2](symmetric) * symmetric_norm

    return compose_tensor(scalar, antisymmetric, symmetric)


class ChargePredictionHead(eqx.Module):
  weights: Any

  def __init__(self, weights):
    self.weights = weights

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
    weights = self.weights
    scalar, antisymmetric, symmetric = decompose_tensor(tensor_features)
    charge_features = jnp.concatenate(
      [scalar, tensor_norm(antisymmetric), tensor_norm(symmetric)],
      axis=-1,
    )

    charge_features = weights['q_norm'](charge_features)
    for i, layer in enumerate(weights['q_mlp']):
      charge_features = layer(charge_features)
      if i < len(weights['q_mlp']) - 1:
        charge_features = jax.nn.silu(charge_features)

    ncharge = charge_features.shape[-1] // 2
    partial_charges = charge_features[:, :ncharge]
    charge_weights = charge_features[:, ncharge:]
    return self._neural_charge_equilibration(
      partial_charges,
      charge_weights,
      total_charge,
    )


class AceFFLayer(eqx.Module):
  weights: Any
  cutoff: float = eqx.field(static=True)
  cutoff_lower: float = eqx.field(static=True)
  group: str = eqx.field(static=True)
  edge_charge_features: bool = eqx.field(static=True)
  total_charge_interaction_scale: bool = eqx.field(static=True)

  def __init__(
    self,
    weights,
    *,
    cutoff: float,
    cutoff_lower: float,
    group: str,
    edge_charge_features: bool,
    total_charge_interaction_scale: bool,
  ):
    self.weights = weights
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
    weights = self.weights
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

    for layer in weights['linears_scalar']:
      edge_features = jax.nn.silu(layer(edge_features))
    edge_features = (edge_features * cutoff_values[:, None]).reshape(
      -1,
      3,
      num_features,
    )

    tensor_features = (
      tensor_features / (tensor_norm(tensor_features) + 1)[:, None, None, :]
    )

    scalar, antisymmetric, symmetric = decompose_tensor(tensor_features)
    scalar = weights['linears_tensor'][0](scalar)
    antisymmetric = weights['linears_tensor'][1](antisymmetric)
    symmetric = weights['linears_tensor'][2](symmetric)
    projected_features = compose_tensor(scalar, antisymmetric, symmetric)

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
    messages = compose_tensor(
      scalar_message,
      antisymmetric_message,
      symmetric_message,
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

    update_norm = tensor_norm(updates) + 1
    scalar_update = scalar_update / update_norm
    antisymmetric_update = antisymmetric_update / update_norm[:, None, None, :]
    symmetric_update = symmetric_update / update_norm[:, None, None, :]

    scalar_update = weights['linears_tensor'][3](scalar_update)
    antisymmetric_update = weights['linears_tensor'][4](antisymmetric_update)
    symmetric_update = weights['linears_tensor'][5](symmetric_update)
    delta_features = compose_tensor(
      scalar_update,
      antisymmetric_update,
      symmetric_update,
    )

    return (
      tensor_features
      + delta_features
      + charge_factor * tensor_matmul_so3(delta_features, delta_features)
    )


class LocalEnergyHead(eqx.Module):
  out_norm: Any
  linear: Any
  output_network: Any

  def __init__(self, weights):
    self.out_norm = weights['out_norm']
    self.linear = weights['linear']
    self.output_network = weights['output_network']

  def __call__(self, tensor_features):
    _, antisymmetric, symmetric = decompose_tensor(tensor_features)
    trace = jnp.diagonal(tensor_features, axis1=1, axis2=2).sum(axis=-1)
    warp_one_third = jnp.asarray(
      # This comes from upstream using Warp kernels that round
      # 1/3 to float32 before casting to float64.
      float(np.float32(1.0 / 3.0)),
      dtype=trace.dtype,
    )
    scalar_norm = warp_one_third * trace * trace
    energy_features = jnp.concatenate(
      [scalar_norm, tensor_norm(antisymmetric), tensor_norm(symmetric)],
      axis=-1,
    )
    energy_features = self.out_norm(energy_features)
    energy_features = jax.nn.silu(self.linear(energy_features))
    for i, layer in enumerate(self.output_network):
      energy_features = layer(energy_features)
      if i < len(self.output_network) - 1:
        energy_features = jax.nn.silu(energy_features)
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
    qweights,
    *,
    coulomb_factor: float,
    coulomb_damp_cutoff: float,
    coulomb_cutoff: float | None,
    coulomb_epsilon_solvent: float,
    exp_minus_1: float,
  ):
    self.qweights = qweights
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
      # Free-space fallback: every unique pair contributes once.
      pair_src, pair_dst = unique_pairs(int(positions.shape[0]))
      pair_vectors = jax.vmap(displacement_fn)(
        positions[pair_src], positions[pair_dst]
      )
      pair_mask = None
    else:
      # A dedicated Coulomb neighbor list at coulomb_cutoff supplies edges.
      # Each undirected pair appears twice, so keep only edge_src < edge_dst.
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
  charge_predictor_0: ChargePredictionHead | None
  layers: tuple[AceFFLayer, ...]
  charge_predictors_by_layer: tuple[ChargePredictionHead, ...]
  local_energy_head: LocalEnergyHead
  coulomb_head: CoulombHead | None
  cutoff: float = eqx.field(static=True)
  ev_to_kjmol: float = eqx.field(static=True)
  cutoff_lower: float = eqx.field(static=True)
  alpha: float = eqx.field(static=True)
  neighbor_cell_atom_threshold: int = eqx.field(static=True)
  neighbor_cell_capacity_multiplier: float = eqx.field(static=True)

  def __init__(self, config):
    self.cutoff = float(config['cutoff'])
    self.ev_to_kjmol = float(config['ev_to_kjmol'])
    self.cutoff_lower = float(config['cutoff_lower'])
    self.alpha = float(config['alpha'])
    group = str(config['group'])
    charge_predictors = bool(config['charge_predictors'])
    edge_charge_features = bool(config['edge_charge_features'])
    total_charge_interaction_scale = bool(
      config['total_charge_interaction_scale']
    )
    coulomb_energy = bool(config['coulomb_energy'])
    coulomb_factor = float(config['coulomb_factor'])
    coulomb_damp_cutoff = float(config['coulomb_damp_cutoff'])
    coulomb_cutoff = config['coulomb_cutoff']
    coulomb_cutoff = None if coulomb_cutoff is None else float(coulomb_cutoff)
    coulomb_epsilon_solvent = float(config['coulomb_epsilon_solvent'])
    exp_minus_1 = float(config['exp_minus_1'])
    self.neighbor_cell_atom_threshold = int(
      config['neighbor_cell_atom_threshold']
    )
    self.neighbor_cell_capacity_multiplier = float(
      config['neighbor_cell_capacity_multiplier']
    )

    # Rebuild the zero-filled parameter tree from the architecture dimensions
    # in the config; eqx.tree_deserialise_leaves fills in the array leaves.
    hidden = int(config['hidden_channels'])
    num_rbf = int(config['num_rbf'])
    num_layers = int(config['num_interaction_layers'])
    norm_eps = float(config['layernorm_eps'])

    def zeros(*shape):
      return jnp.zeros(shape, dtype=jnp.float32)

    def linear(d_in, d_out, *, bias=True):
      return Linear(zeros(d_in, d_out), zeros(d_out) if bias else None)

    def layer_norm(dim):
      return LayerNorm(zeros(dim), zeros(dim), eps=norm_eps)

    def mlp(d_in, dims):
      layers = []
      for d_out in dims:
        layers.append(linear(d_in, int(d_out)))
        d_in = int(d_out)
      return layers

    charge_mlp_dims = (
      [int(x) for x in config['charge_mlp_dims']] if charge_predictors else []
    )
    n_charge = charge_mlp_dims[-1] // 2 if charge_predictors else 0
    layer_scalar_in = num_rbf + (2 * n_charge if edge_charge_features else 0)

    def charge_head():
      return {
        'q_norm': layer_norm(3 * hidden),
        'q_mlp': mlp(3 * hidden, charge_mlp_dims),
      }

    params = {
      'rbf_means': zeros(num_rbf),
      'rbf_betas': zeros(num_rbf),
      'tensor_embedding': {
        'distance_proj1': linear(num_rbf, hidden),
        'distance_proj2': linear(num_rbf, hidden),
        'distance_proj3': linear(num_rbf, hidden),
        'emb': zeros(hidden, hidden),
        'emb2': linear(2 * hidden, hidden),
        'init_norm': layer_norm(hidden),
        'linears_scalar': [
          linear(hidden, 2 * hidden),
          linear(2 * hidden, 3 * hidden),
        ],
        'linears_tensor': [
          linear(hidden, hidden, bias=False) for _ in range(3)
        ],
      },
      'layers': [
        {
          'linears_scalar': [
            linear(layer_scalar_in, hidden),
            linear(hidden, 2 * hidden),
            linear(2 * hidden, 3 * hidden),
          ],
          'linears_tensor': [
            linear(hidden, hidden, bias=False) for _ in range(6)
          ],
        }
        for _ in range(num_layers)
      ],
      'linear': linear(3 * hidden, hidden),
      'out_norm': layer_norm(3 * hidden),
      'output_network': mlp(hidden, config['output_network_dims']),
    }
    if charge_predictors:
      params['charge_predict_0'] = charge_head()
      params['charge_predicts'] = [charge_head() for _ in range(num_layers)]
    if coulomb_energy:
      params['qweights'] = zeros(int(config['coulomb_qweights_dim']))

    self.rbf_betas = params['rbf_betas']
    self.rbf_means = params['rbf_means']
    self.tensor_embedding = TensorEmbedding(
      params['tensor_embedding'],
      cutoff=self.cutoff,
      cutoff_lower=self.cutoff_lower,
    )
    self.charge_predictor_0 = (
      ChargePredictionHead(params['charge_predict_0'])
      if charge_predictors
      else None
    )
    self.layers = tuple(
      AceFFLayer(
        layer,
        cutoff=self.cutoff,
        cutoff_lower=self.cutoff_lower,
        group=group,
        edge_charge_features=edge_charge_features,
        total_charge_interaction_scale=total_charge_interaction_scale,
      )
      for layer in params['layers']
    )
    self.charge_predictors_by_layer = (
      tuple(
        ChargePredictionHead(weights) for weights in params['charge_predicts']
      )
      if charge_predictors
      else ()
    )
    self.local_energy_head = LocalEnergyHead(params)
    self.coulomb_head = (
      CoulombHead(
        params['qweights'],
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
    pbc_shifts,
    edge_mask,
  ):
    edge_vectors = positions[edge_src] - positions[edge_dst] + pbc_shifts
    is_self = edge_src == edge_dst
    distances = jnp.where(
      is_self,
      0.0,
      safe_norm(edge_vectors, axis=-1, eps=1.0e-30),
    )
    safe_denom = jnp.where(
      is_self[:, None],
      1.0,
      jnp.maximum(distances[:, None], 1e-8),
    )
    unit_vectors = edge_vectors / safe_denom
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
    pbc_shifts,
    edge_mask,
    total_charge,
  ):
    distances, unit_vectors, radial_features, edge_mask = self.edge_features(
      positions,
      edge_src,
      edge_dst,
      pbc_shifts,
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
    edge_src, edge_dst, pbc_shifts, edge_mask = featurize(positions, neighbors)
    node_energies, partial_charges = self.local_node_energies_and_charges(
      positions,
      species,
      edge_src,
      edge_dst,
      pbc_shifts,
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
  model: str | PathLike = 'aceff-jax-2.0',
  *,
  model_path: str | PathLike | None = None,
  dtype: Any = jnp.float32,
):
  if model_path is not None:
    path = Path(model_path)
  elif isinstance(model, PathLike):
    path = Path(model)
  elif model in ACEFF_MODEL_PATHS:
    path = ACEFF_MODEL_PATHS[model]
  else:
    path = Path(model)

  with path.open('rb') as handle:
    config = dict(json.loads(handle.readline().decode()))
    template = AceFF(config)
    loaded = eqx.tree_deserialise_leaves(handle, template)

  if jax.config.jax_enable_x64:
    dtype = jnp.float64
  if dtype == jnp.float32:
    return loaded
  return jax.tree_util.tree_map(
    lambda x: (
      x.astype(dtype)
      if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
      else x
    ),
    loaded,
  )
