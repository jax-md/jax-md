# Credit to https://github.com/orbital-materials/orb-models
#
# Parameters are stored fp32, matching upstream orb-models; computation runs
# in fp64 under jax_enable_x64.

import json
from functools import partial
from importlib.resources import files
from os import PathLike
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_md import partition
from jax_md._nn import weights
from jax_md.units import EV_TO_KJMOL

jax.config.update('jax_default_matmul_precision', 'highest')

ORB_MODEL_PATHS = {
  'orb-jax-v3-conservative-omol': (
    files('jax_md._nn.orb') / 'orb-v3-conservative-omol.eqx'
  ),
}


def neighbor_list_featurizer(displacement_fn, *, cutoff: float | None = None):
  def featurize(position, neighbor, **kwargs):
    num_atoms = position.shape[0]
    mask = partition.neighbor_list_mask(neighbor, True)
    if partition.is_sparse(neighbor.format):
      receivers, senders = jnp.asarray(neighbor.idx, dtype=jnp.int32)
    else:
      atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
      receivers = jnp.asarray(neighbor.idx, dtype=jnp.int32)
      senders = jnp.broadcast_to(atom_ids[:, None], receivers.shape)
      receivers = receivers.reshape(-1)
      senders = senders.reshape(-1)
      mask = mask.reshape(-1)

    senders = jnp.where(mask, senders, 0)
    receivers = jnp.where(mask, receivers, 0)

    d = jax.vmap(partial(displacement_fn, **kwargs))
    edge_vectors = d(position[receivers], position[senders])

    distances = jnp.linalg.norm(edge_vectors, axis=-1)
    edge_mask = mask & (distances > 1.0e-8)
    if cutoff is not None:
      edge_mask = edge_mask & (distances < cutoff)
    edge_vectors = jnp.where(edge_mask[:, None], edge_vectors, 0.0)
    return edge_vectors, senders, receivers, edge_mask

  return featurize


def safe_norm(
  x: Array, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24
) -> Array:
  return jnp.sqrt(
    jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps)
  )


def polynomial_cutoff(r: Array, r_max: float | Array, p: float) -> Array:
  ratio = r / r_max
  envelope = (
    1.0
    - ((p + 1.0) * (p + 2.0) / 2.0) * ratio**p
    + p * (p + 2.0) * ratio ** (p + 1.0)
    - (p * (p + 1.0) / 2.0) * ratio ** (p + 2.0)
  )
  return jnp.where(r < r_max, envelope, 0.0)


def bessel_basis(
  r: Array,
  bessel_weights: Array,
  prefactor: Array,
) -> Array:
  safe_r = jnp.maximum(r, 1.0e-7)
  return prefactor * (
    jnp.sin(bessel_weights[None, :] * safe_r[:, None]) / safe_r[:, None]
  )


def spherical_harmonics_0_to_3(edge_vectors: Array) -> Array:
  xyz = edge_vectors / safe_norm(edge_vectors, axis=-1, keepdims=True)
  x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
  sh_0_0 = jnp.ones_like(x)
  sh_1_0 = x
  sh_1_1 = y
  sh_1_2 = z
  sh_2_0 = jnp.sqrt(3.0) * x * z
  sh_2_1 = jnp.sqrt(3.0) * x * y
  y2 = y**2
  x2z2 = x**2 + z**2
  sh_2_2 = y2 - 0.5 * x2z2
  sh_2_3 = jnp.sqrt(3.0) * y * z
  sh_2_4 = jnp.sqrt(3.0) / 2.0 * (z**2 - x**2)
  sh_3_0 = jnp.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
  sh_3_1 = jnp.sqrt(5.0) * sh_2_0 * y
  sh_3_2 = jnp.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
  sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
  sh_3_4 = jnp.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
  sh_3_5 = jnp.sqrt(5.0) * sh_2_4 * y
  sh_3_6 = jnp.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
  sh = jnp.stack(
    [
      sh_0_0,
      sh_1_0,
      sh_1_1,
      sh_1_2,
      sh_2_0,
      sh_2_1,
      sh_2_2,
      sh_2_3,
      sh_2_4,
      sh_3_0,
      sh_3_1,
      sh_3_2,
      sh_3_3,
      sh_3_4,
      sh_3_5,
      sh_3_6,
    ],
    axis=-1,
  )
  component_scale = jnp.array(
    [1.0] + [jnp.sqrt(3.0)] * 3 + [jnp.sqrt(5.0)] * 5 + [jnp.sqrt(7.0)] * 7,
    dtype=sh.dtype,
  )
  return sh * component_scale


def condition_nodes(
  charge_embedding: Array,
  spin_embedding: Array,
  total_charge: Array,
  total_spin: Array,
  num_atoms: int,
) -> Array:
  charge_proj = (
    total_charge[:, None] * charge_embedding[None, :] * (2.0 * jnp.pi)
  )
  spin_proj = total_spin[:, None] * spin_embedding[None, :] * (2.0 * jnp.pi)
  charge_emb = jnp.concatenate(
    [jnp.sin(charge_proj), jnp.cos(charge_proj)], axis=-1
  )
  spin_emb = jnp.concatenate([jnp.sin(spin_proj), jnp.cos(spin_proj)], axis=-1)
  spin_emb = jnp.where(total_spin[:, None] == 0, 0.0, spin_emb)
  return jnp.repeat(
    jnp.concatenate([charge_emb, spin_emb], axis=-1), num_atoms, axis=0
  )


class Linear(eqx.Module):
  weight: Array
  bias: Array

  def __init__(
    self,
    config: dict[str, Any],
    prefix: str,
  ) -> None:
    pdtype = np.dtype(config['parameter_dtype'])
    self.weight = jnp.zeros(tuple(config['params'][f'{prefix}.weight']), pdtype)
    self.bias = jnp.zeros(tuple(config['params'][f'{prefix}.bias']), pdtype)

  def __call__(self, x: Array) -> Array:
    return x @ jnp.swapaxes(self.weight, -1, -2) + self.bias


class MLP(eqx.Module):
  layers: tuple[Linear, ...]

  def __init__(
    self,
    config: dict[str, Any],
    prefix: str,
    num_layers: int,
  ) -> None:
    self.layers = tuple(
      Linear(config, f'{prefix}.NN-{i}') for i in range(num_layers)
    )

  def __call__(self, x: Array) -> Array:
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i < len(self.layers) - 1:
        x = jax.nn.silu(x)
    return x


class RMSNorm(eqx.Module):
  weight: Array

  def __init__(
    self,
    config: dict[str, Any],
    prefix: str,
  ) -> None:
    self.weight = jnp.zeros(
      tuple(config['params'][f'{prefix}.weight']),
      np.dtype(config['parameter_dtype']),
    )

  def __call__(self, x: Array) -> Array:
    eps = jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype)
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * scale * self.weight


class MLPNorm(eqx.Module):
  mlp: MLP
  layer_norm: RMSNorm

  def __init__(
    self,
    config: dict[str, Any],
    prefix: str,
  ) -> None:
    self.mlp = MLP(
      config,
      f'{prefix}.mlp',
      int(config['mlp_num_layers']),
    )
    self.layer_norm = RMSNorm(config, f'{prefix}.layer_norm')

  def __call__(self, x: Array) -> Array:
    return self.layer_norm(self.mlp(x))


class AttentionBlock(eqx.Module):
  cond_node_proj: Linear
  receive_attn: Linear
  send_attn: Linear
  edge_mlp: MLPNorm
  node_mlp: MLPNorm

  def __init__(
    self,
    config: dict[str, Any],
    prefix: str,
  ) -> None:
    self.cond_node_proj = Linear(config, f'{prefix}._cond_node_proj')
    self.receive_attn = Linear(config, f'{prefix}._receive_attn')
    self.send_attn = Linear(config, f'{prefix}._send_attn')
    self.edge_mlp = MLPNorm(config, f'{prefix}._edge_mlp')
    self.node_mlp = MLPNorm(config, f'{prefix}._node_mlp')

  def __call__(
    self,
    nodes: Array,
    edges: Array,
    cond_nodes: Array,
    senders: Array,
    receivers: Array,
    cutoff: Array,
  ) -> tuple[Array, Array]:
    nodes_cond = nodes + self.cond_node_proj(cond_nodes)
    receive_attn = jax.nn.sigmoid(self.receive_attn(edges)) * cutoff
    send_attn = jax.nn.sigmoid(self.send_attn(edges)) * cutoff
    edge_features = jnp.concatenate(
      [edges, nodes_cond[senders], nodes_cond[receivers]],
      axis=1,
    )
    updated_edges = self.edge_mlp(edge_features)
    sent = jnp.zeros_like(nodes).at[senders].add(updated_edges * send_attn)
    received = (
      jnp.zeros_like(nodes).at[receivers].add(updated_edges * receive_attn)
    )
    node_features = jnp.concatenate([nodes_cond, received, sent], axis=1)
    updated_nodes = self.node_mlp(node_features)
    return nodes_cond + updated_nodes, edges + updated_edges


class ORBLayer(eqx.Module):
  rbf_bessel_weights: Array
  rbf_prefactor: Array
  atom_embedding: Array
  charge_embedding: Array
  spin_embedding: Array
  encoder_node_fn: MLPNorm
  encoder_edge_fn: MLPNorm
  blocks: tuple[AttentionBlock, ...]
  num_layers: int = eqx.field(static=True)
  mlp_num_layers: int = eqx.field(static=True)
  edge_feature_dim: int = eqx.field(static=True)
  cutoff: float = eqx.field(static=True)
  cutoff_polynomial_p: float = eqx.field(static=True)

  def __init__(
    self,
    *,
    config: dict[str, Any],
    cutoff: float,
    num_layers: int,
    mlp_num_layers: int,
    edge_feature_dim: int,
    cutoff_polynomial_p: float,
  ) -> None:
    self.num_layers = int(num_layers)
    self.mlp_num_layers = int(mlp_num_layers)
    self.edge_feature_dim = int(edge_feature_dim)
    self.cutoff = float(cutoff)
    self.cutoff_polynomial_p = float(cutoff_polynomial_p)

    pdtype = np.dtype(config['parameter_dtype'])
    self.rbf_bessel_weights = jnp.zeros(
      tuple(config['params']['model.rbf_transform.bessel_weights']), pdtype
    )
    self.rbf_prefactor = jnp.zeros(
      tuple(config['params']['model.rbf_transform.prefactor']), pdtype
    )
    self.atom_embedding = jnp.zeros(
      tuple(config['params']['model.atom_emb.embeddings.weight']), pdtype
    )
    self.charge_embedding = jnp.zeros(
      tuple(config['params']['model.conditioner.charge_embedding.W']), pdtype
    )
    self.spin_embedding = jnp.zeros(
      tuple(config['params']['model.conditioner.spin_embedding.W']), pdtype
    )
    self.encoder_node_fn = MLPNorm(config, 'model._encoder._node_fn')
    self.encoder_edge_fn = MLPNorm(config, 'model._encoder._edge_fn')
    self.blocks = tuple(
      AttentionBlock(config, f'model.gnn_stacks.{i}')
      for i in range(self.num_layers)
    )

  def __call__(
    self,
    edge_vectors: Array,
    species: Array,
    senders: Array,
    receivers: Array,
    edge_mask: Array,
    total_charge: Array,
    total_spin: Array,
  ) -> Array:
    distances = safe_norm(edge_vectors, axis=-1)
    rbfs = bessel_basis(distances, self.rbf_bessel_weights, self.rbf_prefactor)
    angular = spherical_harmonics_0_to_3(edge_vectors)
    cutoff = polynomial_cutoff(
      distances,
      self.cutoff,
      self.cutoff_polynomial_p,
    )
    cutoff = cutoff[:, None] * edge_mask[:, None].astype(cutoff.dtype)
    edges_in = (
      cutoff[:, :, None] * rbfs[:, :, None] * angular[:, None, :]
    ).reshape((senders.shape[0], self.edge_feature_dim))
    nodes_in = self.atom_embedding[species]
    cond_nodes = condition_nodes(
      self.charge_embedding,
      self.spin_embedding,
      total_charge,
      total_spin,
      species.shape[0],
    )

    nodes = self.encoder_node_fn(nodes_in)
    edges = self.encoder_edge_fn(edges_in)
    for block in self.blocks:
      nodes, edges = block(nodes, edges, cond_nodes, senders, receivers, cutoff)
    return nodes


class LocalEnergyHead(eqx.Module):
  energy_mlp: MLP
  energy_normalizer_var: Array
  energy_normalizer_mean: Array
  energy_reference_weight: Array
  energy_mlp_num_layers: int = eqx.field(static=True)

  def __init__(
    self,
    *,
    config: dict[str, Any],
    energy_mlp_num_layers: int,
  ) -> None:
    self.energy_mlp_num_layers = int(energy_mlp_num_layers)
    self.energy_mlp = MLP(
      config,
      'heads.energy.mlp',
      self.energy_mlp_num_layers,
    )
    pdtype = np.dtype(config['parameter_dtype'])
    self.energy_normalizer_var = jnp.zeros(
      tuple(config['params']['heads.energy.normalizer.bn.running_var']), pdtype
    )
    self.energy_normalizer_mean = jnp.zeros(
      tuple(config['params']['heads.energy.normalizer.bn.running_mean']), pdtype
    )
    self.energy_reference_weight = jnp.zeros(
      tuple(config['params']['heads.energy.reference.linear.weight']), pdtype
    )

  def __call__(self, node_features: Array, species: Array) -> Array:
    graph_features = jnp.mean(node_features, axis=0, keepdims=True)
    x = self.energy_mlp(graph_features).reshape(())
    x = x * jnp.sqrt(self.energy_normalizer_var[0])
    x = x + self.energy_normalizer_mean[0]
    x = x * species.shape[0]
    reference = jnp.sum(self.energy_reference_weight[species])
    return x + reference


class ZBLRepulsion(eqx.Module):
  covalent_radii: Array
  coulomb_ev_angstrom: float = eqx.field(static=True)
  zbl_polynomial_p: float = eqx.field(static=True)
  zbl_atomic_number_exponent: float = eqx.field(static=True)
  zbl_screening_length_scale: float = eqx.field(static=True)
  zbl_screening_weights: tuple[float, ...] = eqx.field(static=True)
  zbl_screening_exponents: tuple[float, ...] = eqx.field(static=True)

  def __init__(
    self,
    *,
    config: dict[str, Any],
  ) -> None:
    self.coulomb_ev_angstrom = float(config['zbl_coulomb_ev_angstrom'])
    self.zbl_polynomial_p = float(config['zbl_polynomial_p'])
    self.zbl_atomic_number_exponent = float(
      config['zbl_atomic_number_exponent']
    )
    self.zbl_screening_length_scale = float(
      config['zbl_screening_length_scale']
    )
    self.zbl_screening_weights = tuple(
      float(x) for x in config['zbl_screening_weights']
    )
    self.zbl_screening_exponents = tuple(
      float(x) for x in config['zbl_screening_exponents']
    )
    self.covalent_radii = jnp.zeros(
      tuple(config['params']['covalent_radii']),
      np.dtype(config['parameter_dtype']),
    )

  def __call__(
    self,
    species: Array,
    edge_vectors: Array,
    senders: Array,
    receivers: Array,
    edge_mask: Array,
  ) -> Array:
    num_atoms = species.shape[0]
    distances = safe_norm(edge_vectors, axis=-1)
    safe_distances = jnp.maximum(distances, 1.0e-7)
    z_sender = species[senders] + 1
    z_receiver = species[receivers] + 1
    z_sender_f = z_sender.astype(edge_vectors.dtype)
    z_receiver_f = z_receiver.astype(edge_vectors.dtype)
    zbl_exponent = jnp.asarray(
      self.zbl_atomic_number_exponent, dtype=edge_vectors.dtype
    )
    screening_length = self.zbl_screening_length_scale / (
      z_sender_f**zbl_exponent + z_receiver_f**zbl_exponent
    )
    scaled_distance = safe_distances / screening_length
    screening_weights = jnp.asarray(
      self.zbl_screening_weights,
      dtype=edge_vectors.dtype,
    )[:, None]
    screening_exponents = jnp.asarray(
      self.zbl_screening_exponents,
      dtype=edge_vectors.dtype,
    )[:, None]
    zbl_screening = jnp.sum(
      screening_weights
      * jnp.exp(-screening_exponents * scaled_distance[None, :]),
      axis=0,
    )
    bare_nuclear_repulsion = (
      self.coulomb_ev_angstrom * z_sender_f * z_receiver_f / safe_distances
    )
    cutoff_radius = (
      self.covalent_radii[z_sender] + self.covalent_radii[z_receiver]
    )
    orb_envelope = polynomial_cutoff(
      safe_distances,
      cutoff_radius,
      self.zbl_polynomial_p,
    )
    edge_repulsion = 0.5 * bare_nuclear_repulsion * zbl_screening * orb_envelope
    edge_repulsion = edge_repulsion * edge_mask.astype(edge_repulsion.dtype)
    return jnp.sum(edge_repulsion) / num_atoms


class Orb(eqx.Module):
  layer: ORBLayer
  energy_head: LocalEnergyHead
  zbl_repulsion: ZBLRepulsion
  cutoff: float = eqx.field(static=True)
  ev_to_kjmol: float = eqx.field(static=True)
  num_species_embeddings: int = eqx.field(static=True)
  num_layers: int = eqx.field(static=True)
  mlp_num_layers: int = eqx.field(static=True)
  energy_mlp_num_layers: int = eqx.field(static=True)
  edge_feature_dim: int = eqx.field(static=True)
  cutoff_polynomial_p: float = eqx.field(static=True)

  def __init__(
    self,
    *,
    config: dict[str, Any],
  ) -> None:
    self.cutoff = float(config['cutoff'])
    self.ev_to_kjmol = float(config.get('ev_to_kjmol', EV_TO_KJMOL))
    self.num_species_embeddings = int(
      config.get(
        'num_species_embeddings',
        config['params']['model.atom_emb.embeddings.weight'][0],
      )
    )
    self.num_layers = int(
      config.get('num_layers', config.get('num_gnn_stacks', 5))
    )
    self.mlp_num_layers = int(config.get('mlp_num_layers', 3))
    self.energy_mlp_num_layers = int(config.get('energy_mlp_num_layers', 2))
    self.edge_feature_dim = int(config.get('edge_feature_dim', 128))
    self.cutoff_polynomial_p = float(config.get('cutoff_polynomial_p', 4.0))

    self.layer = ORBLayer(
      config=config,
      cutoff=self.cutoff,
      num_layers=self.num_layers,
      mlp_num_layers=self.mlp_num_layers,
      edge_feature_dim=self.edge_feature_dim,
      cutoff_polynomial_p=self.cutoff_polynomial_p,
    )
    self.energy_head = LocalEnergyHead(
      config=config,
      energy_mlp_num_layers=self.energy_mlp_num_layers,
    )
    self.zbl_repulsion = ZBLRepulsion(config=config)

  def local_node_features(
    self,
    positions_angstrom: Array,
    species: Array,
    total_charge: Array,
    total_spin: Array,
    *,
    neighbors,
    displacement_fn,
  ) -> tuple[Array, Array, Array, Array, Array]:
    featurize = neighbor_list_featurizer(
      displacement_fn, cutoff=float(self.cutoff)
    )
    edge_vectors, senders, receivers, edge_mask = featurize(
      positions_angstrom, neighbors
    )
    node_features = self.layer(
      edge_vectors,
      species,
      senders,
      receivers,
      edge_mask,
      total_charge,
      total_spin,
    )
    return node_features, edge_vectors, senders, receivers, edge_mask

  def __call__(
    self,
    positions_angstrom: Array,
    species: Array,
    total_charge: Array,
    total_spin: Array,
    *,
    displacement_fn=None,
    neighbors=None,
  ) -> Array:
    if displacement_fn is None or neighbors is None:
      raise ValueError(
        'Orb requires a displacement_fn and a neighbor list. Build them '
        'with energy.orb_neighbor_list.'
      )
    node_features, edge_vectors, senders, receivers, edge_mask = (
      self.local_node_features(
        positions_angstrom,
        species,
        total_charge,
        total_spin,
        neighbors=neighbors,
        displacement_fn=displacement_fn,
      )
    )
    graph_energy = self.energy_head(node_features, species)
    zbl_energy = self.zbl_repulsion(
      species,
      edge_vectors,
      senders,
      receivers,
      edge_mask,
    )
    total_energy = graph_energy + zbl_energy
    return total_energy


def load_model(
  model: str = 'orb-jax-v3-conservative-omol',
  *,
  model_path: str | PathLike | None = None,
  dtype=None,
) -> Orb:
  if model_path is not None:
    path = weights.resolve_checkpoint(model_path, allow_cache=False)
  else:
    path = weights.resolve_checkpoint(str(ORB_MODEL_PATHS[model]))

  if dtype is None:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  with path.open('rb') as handle:
    config = dict(json.loads(handle.readline().decode()))
    model = eqx.tree_deserialise_leaves(handle, Orb(config=config))
    return jax.tree_util.tree_map(
      lambda x: (
        x.astype(dtype)
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
        else x
      ),
      model,
    )
