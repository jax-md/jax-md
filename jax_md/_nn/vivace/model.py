# Credit to https://github.com/microsoft/simpoly

import json
from functools import partial
from importlib.resources import files
from os import PathLike
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax_md import partition
from jax_md._nn import weights

jax.config.update('jax_default_matmul_precision', 'highest')

VIVACE_MODEL_PATHS = {
  'vivace-v0.1': files('jax_md._nn.vivace') / 'vivace_v0.1.eqx',
}


def neighbor_list_featurizer(displacement_fn, *, cutoff: float):
  def featurize(position, neighbor, **kwargs):
    num_atoms = position.shape[0]
    mask = partition.neighbor_list_mask(neighbor, mask_self=True)
    if partition.is_sparse(neighbor.format):
      if neighbor.format is partition.NeighborListFormat.OrderedSparse:
        raise ValueError('Vivace requires a full list; use format=Sparse.')
      receivers, senders = jnp.asarray(neighbor.idx, dtype=jnp.int32)
    else:
      idx = jnp.asarray(neighbor.idx, dtype=jnp.int32)
      receivers = jnp.repeat(
        jnp.arange(num_atoms, dtype=jnp.int32), idx.shape[1]
      )
      senders = idx.reshape(-1)
      mask = mask.reshape(-1)

    receivers = jnp.where(mask, receivers, 0)
    senders = jnp.where(mask, senders, 0)
    d = jax.vmap(partial(displacement_fn, **kwargs))
    # Neighbor minus center under minimum image, matching SimPoly export.
    vectors = d(position[senders], position[receivers])
    far = jnp.zeros((position.shape[-1],), dtype=position.dtype)
    far = far.at[0].set(2.0 * cutoff)
    vectors = jnp.where(mask[:, None], vectors, far)
    return receivers, senders, vectors, mask

  return featurize


def step_clamp(r: Array, clamp: list[float]) -> Array:
  r_max_inv, one_minus_offset, offset_inv = clamp
  u = r * r_max_inv
  t = (1.0 - u) * offset_inv
  smooth = (3.0 - 2.0 * t) * t * t
  return jnp.where(u > one_minus_offset, smooth, 1.0) * (u < 1.0)


def spherical_harmonics_0_to_2(vectors: Array) -> Array:
  norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
  unit = vectors / jnp.where(norm > 0.0, norm, 1.0)
  x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
  sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=vectors.dtype))
  sqrt5 = jnp.sqrt(jnp.asarray(5.0, dtype=vectors.dtype))
  sqrt15 = jnp.sqrt(jnp.asarray(15.0, dtype=vectors.dtype))
  return jnp.stack(
    [
      jnp.ones_like(x),
      sqrt3 * x,
      sqrt3 * y,
      sqrt3 * z,
      sqrt15 * x * z,
      sqrt15 * x * y,
      sqrt5 / 2.0 * (3.0 * y * y - 1.0),
      sqrt15 * y * z,
      sqrt15 / 2.0 * (z * z - x * x),
    ],
    axis=-1,
  )


class Linear(eqx.Module):
  weight: Array
  bias: Array | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    use_bias: bool,
    dtype: Any = jnp.float32,
  ):
    self.weight = jnp.zeros((out_features, in_features), dtype=dtype)
    self.bias = (
      None if not use_bias else jnp.zeros((out_features,), dtype=dtype)
    )

  def __call__(self, x: Array) -> Array:
    result = x @ self.weight.T
    return result if self.bias is None else result + self.bias


class MLP(eqx.Module):
  layers: list[Linear]

  def __init__(
    self,
    sizes: list[int],
    *,
    biases: list[bool],
    dtype: Any = jnp.float32,
  ):
    self.layers = [
      Linear(d_in, d_out, use_bias=use_bias, dtype=dtype)
      for d_in, d_out, use_bias in zip(
        sizes[:-1], sizes[1:], biases, strict=True
      )
    ]

  def __call__(self, x: Array) -> Array:
    for layer in self.layers[:-1]:
      x = jax.nn.silu(layer(x))
    return self.layers[-1](x)


class GaussianBasis(eqx.Module):
  means: Array
  betas: Array

  def __init__(self, num_radial: int, *, dtype: Any = jnp.float32):
    self.means = jnp.zeros((1, num_radial), dtype=dtype)
    self.betas = jnp.zeros((1, num_radial), dtype=dtype)

  def __call__(self, exp_r: Array) -> Array:
    return jnp.exp(self.betas * (exp_r - self.means) ** 2)


class InitialEmbedding(eqx.Module):
  atomic_embedding: Array
  radial_basis: GaussianBasis
  equivariant_radial_basis: GaussianBasis
  equivariant_radial_linear: Linear
  invariant_mlp: MLP
  equivariant_mlp: MLP
  spherical_irreps: Array
  radial_alpha: float = eqx.field(static=True)
  cutoff_clamp: list[float] = eqx.field(static=True)
  equivariant_clamp: list[float] = eqx.field(static=True)
  equivariant_cutoff: float = eqx.field(static=True)

  def __init__(self, config: dict[str, Any], *, dtype: Any = jnp.float32):
    num_species = int(config['num_species'])
    num_radial = int(config['num_radial'])
    invariant_features = int(config['num_invariant_features'])
    equivariant_features = int(config['num_equivariant_features'])
    edge_input = num_radial + 2 * invariant_features
    self.atomic_embedding = jnp.zeros(
      (num_species, invariant_features), dtype=dtype
    )
    self.radial_basis = GaussianBasis(num_radial, dtype=dtype)
    self.equivariant_radial_basis = GaussianBasis(num_radial, dtype=dtype)
    self.equivariant_radial_linear = Linear(
      num_radial, invariant_features, use_bias=False, dtype=dtype
    )
    self.invariant_mlp = MLP(
      [edge_input, invariant_features, invariant_features],
      biases=[True, True],
      dtype=dtype,
    )
    self.equivariant_mlp = MLP(
      [edge_input, invariant_features, 3 * equivariant_features],
      biases=[True, True],
      dtype=dtype,
    )
    self.spherical_irreps = jnp.zeros((9,), dtype=jnp.int32)
    self.radial_alpha = float(config['radial_alpha'])
    self.cutoff_clamp = list(config['cutoff_clamp'])
    self.equivariant_clamp = list(config['equivariant_clamp'])
    self.equivariant_cutoff = float(config['equivariant_cutoff'])

  def __call__(
    self,
    species,
    vectors,
    receivers,
    senders,
    edge_mask,
  ):
    num_atoms = species.shape[0]
    lengths = jnp.linalg.norm(vectors, axis=-1)
    envelope = step_clamp(lengths, self.cutoff_clamp) * edge_mask
    equivariant_envelope = (
      step_clamp(lengths, self.equivariant_clamp) * edge_mask
    )
    equivariant_mask = (lengths < self.equivariant_cutoff) & edge_mask
    exp_r = jnp.exp(-self.radial_alpha * lengths)[:, None]
    basis = self.radial_basis(exp_r)
    equivariant_basis = self.equivariant_radial_basis(exp_r)
    equivariant_embedding = (
      self.equivariant_radial_linear(equivariant_basis)
      * equivariant_envelope[:, None]
    )

    node_invariant = self.atomic_embedding[species]
    pair_embedding = [
      node_invariant[receivers],
      node_invariant[senders],
    ]
    edge_invariant = self.invariant_mlp(
      jnp.concatenate([basis] + pair_embedding, axis=-1)
    )
    edge_invariant = edge_invariant * envelope[:, None]

    harmonics = spherical_harmonics_0_to_2(vectors)
    equivariant_weights = self.equivariant_mlp(
      jnp.concatenate([equivariant_basis] + pair_embedding, axis=-1)
    )
    equivariant_weights = equivariant_weights * equivariant_envelope[:, None]
    num_channels = equivariant_weights.shape[-1] // 3
    edge_equivariant = (
      harmonics[:, :, None]
      * equivariant_weights.reshape(-1, 3, num_channels)[
        :, self.spherical_irreps
      ]
    )
    node_equivariant = (
      jnp.zeros((num_atoms, 9, num_channels), dtype=edge_equivariant.dtype)
      .at[receivers]
      .add(edge_equivariant)
    )
    return (
      lengths,
      envelope,
      equivariant_envelope,
      equivariant_mask,
      harmonics,
      equivariant_embedding,
      node_invariant,
      edge_invariant,
      node_equivariant,
    )


class EquivariantUpdate(eqx.Module):
  query: Linear
  key: Linear
  value: Linear
  environment_out: Linear
  coefficients: Array
  mixing_kernel: Array
  num_attention_heads: int = eqx.field(static=True)
  attention_head_dim: int = eqx.field(static=True)

  def __init__(
    self,
    config: dict[str, Any],
    *,
    input_base_dim: int,
    output_base_dim: int,
    dtype: Any = jnp.float32,
  ):
    invariant_features = int(config['num_invariant_features'])
    channels = int(config['num_equivariant_features'])
    self.num_attention_heads = int(config['num_attention_heads'])
    self.attention_head_dim = int(config['attention_head_dim'])
    latent = self.num_attention_heads * self.attention_head_dim
    self.query = Linear(invariant_features, latent, use_bias=False, dtype=dtype)
    self.key = Linear(invariant_features, latent, use_bias=False, dtype=dtype)
    self.value = Linear(invariant_features, latent, use_bias=False, dtype=dtype)
    self.environment_out = Linear(
      latent, 3 * channels, use_bias=False, dtype=dtype
    )
    self.coefficients = jnp.zeros(
      (output_base_dim, input_base_dim, 9), dtype=dtype
    )
    self.mixing_kernel = jnp.zeros(
      (output_base_dim, channels, channels), dtype=dtype
    )

  def __call__(
    self,
    node_invariant,
    edge_invariant,
    node_equivariant,
    receivers,
    equivariant_envelope,
    harmonics,
    spherical_irreps,
    attention_scale,
  ):
    num_atoms = node_invariant.shape[0]
    query = self.query(node_invariant)[receivers]
    shape = (-1, self.num_attention_heads, self.attention_head_dim)
    key = self.key(edge_invariant).reshape(shape)
    value = self.value(edge_invariant).reshape(shape)
    difference = query.reshape(shape) - key
    kernel = jnp.exp(
      -0.5 * attention_scale * jnp.sum(difference * difference, axis=-1)
    )
    attention = (kernel[:, :, None] * value).reshape(query.shape) * (
      equivariant_envelope[:, None]
    )
    environment_weights = self.environment_out(attention)
    num_channels = environment_weights.shape[-1] // 3
    edge_equivariant = (
      harmonics[:, :, None]
      * (environment_weights.reshape(-1, 3, num_channels)[:, spherical_irreps])
    )
    environment = (
      jnp.zeros((num_atoms, 9, num_channels), dtype=edge_equivariant.dtype)
      .at[receivers]
      .add(edge_equivariant)
    )
    new_equivariant = jnp.einsum(
      'aic,ajc,kij->akc', node_equivariant, environment, self.coefficients
    )
    new_equivariant = jnp.einsum(
      'kvu,aku->akv', self.mixing_kernel, new_equivariant
    )
    norm = jnp.mean(jnp.sum(new_equivariant**2, axis=1), axis=-1)
    return new_equivariant / (norm + 1.0e-5)[:, None, None]


class EdgeUpdate(eqx.Module):
  invariant_out: Linear
  scalar_selection: Array
  filter_mlp: MLP
  post_mlp: MLP

  def __init__(
    self,
    config: dict[str, Any],
    *,
    output_base_dim: int,
    dtype: Any = jnp.float32,
  ):
    invariant_features = int(config['num_invariant_features'])
    hidden_features = int(config['num_hidden_features'])
    channels = int(config['num_equivariant_features'])
    self.invariant_out = Linear(
      channels, invariant_features, use_bias=False, dtype=dtype
    )
    self.scalar_selection = jnp.zeros((output_base_dim, 9), dtype=dtype)
    self.filter_mlp = MLP(
      [invariant_features, invariant_features, invariant_features],
      biases=[False, False],
      dtype=dtype,
    )
    self.post_mlp = MLP(
      [invariant_features, hidden_features, invariant_features],
      biases=[False, False],
      dtype=dtype,
    )

  def __call__(
    self,
    node_invariant,
    edge_invariant,
    node_equivariant,
    receivers,
    equivariant_envelope,
    equivariant_mask,
    harmonics,
    equivariant_embedding,
  ):
    inner = jnp.einsum(
      'eku,kj,ej->eu',
      node_equivariant[receivers],
      self.scalar_selection,
      harmonics,
    )
    delta = self.filter_mlp(equivariant_embedding) * self.invariant_out(inner)
    update = edge_invariant + equivariant_envelope[:, None] * self.post_mlp(
      edge_invariant + delta
    )
    return node_invariant, jnp.where(
      equivariant_mask[:, None], update, edge_invariant
    )


class NodeUpdate(eqx.Module):
  invariant_out: Linear
  post_mlp: MLP

  def __init__(
    self,
    config: dict[str, Any],
    *,
    dtype: Any = jnp.float32,
  ):
    invariant_features = int(config['num_invariant_features'])
    hidden_features = int(config['num_hidden_features'])
    channels = int(config['num_equivariant_features'])
    self.invariant_out = Linear(
      channels, invariant_features, use_bias=False, dtype=dtype
    )
    self.post_mlp = MLP(
      [invariant_features, hidden_features, invariant_features],
      biases=[False, False],
      dtype=dtype,
    )

  def __call__(
    self,
    node_invariant,
    edge_invariant,
    node_equivariant,
    receivers,
    equivariant_envelope,
    equivariant_mask,
    harmonics,
    equivariant_embedding,
  ):
    scalars = self.invariant_out(node_equivariant[:, 0, :])
    mean = jnp.mean(scalars, axis=-1, keepdims=True)
    variance = jnp.var(scalars, axis=-1, keepdims=True)
    scalars = (scalars - mean) / jnp.sqrt(variance + 1.0e-5)
    return node_invariant + self.post_mlp(scalars), edge_invariant


class VivaceLayer(eqx.Module):
  equivariant_update: EquivariantUpdate
  invariant_update: EdgeUpdate | NodeUpdate

  def __init__(
    self,
    config: dict[str, Any],
    *,
    input_base_dim: int,
    output_base_dim: int,
    update_node: bool,
    dtype: Any = jnp.float32,
  ):
    self.equivariant_update = EquivariantUpdate(
      config,
      input_base_dim=input_base_dim,
      output_base_dim=output_base_dim,
      dtype=dtype,
    )
    self.invariant_update = (
      NodeUpdate(config, dtype=dtype)
      if update_node
      else EdgeUpdate(config, output_base_dim=output_base_dim, dtype=dtype)
    )

  def __call__(
    self,
    node_invariant,
    edge_invariant,
    node_equivariant,
    receivers,
    equivariant_envelope,
    equivariant_mask,
    harmonics,
    equivariant_embedding,
    spherical_irreps,
    attention_scale,
  ):
    node_equivariant = self.equivariant_update(
      node_invariant,
      edge_invariant,
      node_equivariant,
      receivers,
      equivariant_envelope,
      harmonics,
      spherical_irreps,
      attention_scale,
    )
    node_invariant, edge_invariant = self.invariant_update(
      node_invariant,
      edge_invariant,
      node_equivariant,
      receivers,
      equivariant_envelope,
      equivariant_mask,
      harmonics,
      equivariant_embedding,
    )
    return node_invariant, edge_invariant, node_equivariant


class ShortRangeRepulsion(eqx.Module):
  mlp: MLP
  powers: Array
  cutoff: float = eqx.field(static=True)
  envelope_cutoff: float = eqx.field(static=True)

  def __init__(self, config: dict[str, Any], *, dtype: Any = jnp.float32):
    features = int(config['num_invariant_features'])
    num_powers = int(config['num_short_range_powers'])
    self.mlp = MLP(
      [features, features, features, num_powers],
      biases=[True, True, True],
      dtype=dtype,
    )
    self.powers = jnp.zeros((num_powers,), dtype=dtype)
    self.cutoff = float(config['short_range_cutoff'])
    self.envelope_cutoff = float(config['short_range_envelope_cutoff'])

  def __call__(
    self,
    edge_invariant,
    receivers,
    lengths,
    edge_mask,
    num_atoms,
  ):
    short_mask = (lengths < self.cutoff) & edge_mask
    weights = self.mlp(edge_invariant) ** 2
    clamp = (
      0.5
      * (jnp.cos(lengths * (jnp.pi / self.envelope_cutoff)) + 1.0)
      * (lengths < self.envelope_cutoff)
    )
    safe_distance = jnp.where(short_mask, lengths, 1.0)
    repulsion = jnp.sum(
      weights * safe_distance[:, None] ** -self.powers, axis=-1
    )
    repulsion = repulsion * clamp * short_mask
    return (
      jnp.zeros((num_atoms,), dtype=repulsion.dtype)
      .at[receivers]
      .add(repulsion)
    )


class EnergyHead(eqx.Module):
  atomic_energies: Array
  node_mlp: MLP
  edge_mlp: MLP

  def __init__(self, config: dict[str, Any], *, dtype: Any = jnp.float32):
    num_species = int(config['num_species'])
    features = int(config['num_invariant_features'])
    hidden_features = int(config['num_hidden_features'])
    self.atomic_energies = jnp.zeros((num_species,), dtype=dtype)
    self.node_mlp = MLP(
      [features, hidden_features, 1], biases=[False, False], dtype=dtype
    )
    self.edge_mlp = MLP(
      [features, hidden_features, 1], biases=[False, False], dtype=dtype
    )

  def __call__(
    self,
    species,
    node_invariant,
    edge_invariant,
    receivers,
    envelope,
  ):
    node_energy = (
      self.atomic_energies[species] + self.node_mlp(node_invariant)[:, 0]
    )
    edge_energy = self.edge_mlp(edge_invariant * envelope[:, None])[:, 0]
    return node_energy.at[receivers].add(edge_energy)


class Vivace(eqx.Module):
  embedding: InitialEmbedding
  layers: list[VivaceLayer]
  energy_head: EnergyHead
  short_range_repulsion: ShortRangeRepulsion
  attention_scale: float = eqx.field(static=True)
  cutoff: float = eqx.field(static=True)

  def __init__(
    self,
    config: dict[str, Any],
    *,
    dtype: Any = jnp.float32,
  ):
    self.embedding = InitialEmbedding(config, dtype=dtype)
    base_dims = config['tensor_product_dims']
    self.layers = [
      VivaceLayer(
        config,
        input_base_dim=input_dim,
        output_base_dim=output_dim,
        update_node=layer_index == len(base_dims) - 1,
        dtype=dtype,
      )
      for layer_index, (input_dim, output_dim) in enumerate(base_dims)
    ]
    self.energy_head = EnergyHead(config, dtype=dtype)
    self.short_range_repulsion = ShortRangeRepulsion(config, dtype=dtype)
    self.attention_scale = float(config['attention_scale'])
    self.cutoff = float(config['cutoff'])

  def __call__(
    self,
    positions,
    species,
    *,
    displacement_fn=None,
    neighbors=None,
    **displacement_kwargs,
  ):
    if displacement_fn is None or neighbors is None:
      raise ValueError(
        'Vivace requires a displacement_fn and a neighbor list. Build them '
        'with energy.vivace_neighbor_list.'
      )
    featurize = neighbor_list_featurizer(displacement_fn, cutoff=self.cutoff)
    receivers, senders, vectors, edge_mask = featurize(
      positions, neighbors, **displacement_kwargs
    )
    species = jnp.asarray(species, dtype=jnp.int32)
    vectors = vectors.astype(self.embedding.atomic_embedding.dtype)
    (
      lengths,
      envelope,
      equivariant_envelope,
      equivariant_mask,
      harmonics,
      equivariant_embedding,
      node_invariant,
      edge_invariant,
      node_equivariant,
    ) = self.embedding(species, vectors, receivers, senders, edge_mask)
    for layer in self.layers:
      node_invariant, edge_invariant, node_equivariant = layer(
        node_invariant,
        edge_invariant,
        node_equivariant,
        receivers,
        equivariant_envelope,
        equivariant_mask,
        harmonics,
        equivariant_embedding,
        self.embedding.spherical_irreps,
        self.attention_scale,
      )
    repulsion = self.short_range_repulsion(
      edge_invariant,
      receivers,
      lengths,
      edge_mask,
      species.shape[0],
    )
    node_energy = self.energy_head(
      species,
      node_invariant,
      edge_invariant,
      receivers,
      envelope,
    )
    return jnp.sum(node_energy + repulsion)


def load_model(
  model: str = 'vivace-v0.1',
  *,
  model_path: str | PathLike | None = None,
  dtype=None,
) -> Vivace:
  """Load a Vivace checkpoint."""
  if model_path is not None:
    path = weights.resolve_checkpoint(model_path, allow_cache=False)
  else:
    path = weights.resolve_checkpoint(str(VIVACE_MODEL_PATHS[model]))

  if dtype is None:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

  with path.open('rb') as handle:
    config = json.loads(handle.readline().decode('utf-8'))
    template = Vivace(config, dtype=jnp.float32)
    model = eqx.tree_deserialise_leaves(handle, template)
    return jax.tree_util.tree_map(
      lambda x: (
        x.astype(dtype)
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
        else x
      ),
      model,
    )
