import json
from functools import partial
from importlib.resources import files
from os import PathLike
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space
from jax_md.units import HARTREE_EV

jax.config.update('jax_default_matmul_precision', 'highest')

DEFAULT_ENSEMBLE_MODEL_PATH = files('jax_md._nn.ani') / 'ani2x_ensemble.eqx'
DEFAULT_SINGLE_MODEL_PATH = files('jax_md._nn.ani') / 'ani2x_model0.eqx'
ANI2X_MODEL_PATHS = {
  'ani2x-jax-ensemble': DEFAULT_ENSEMBLE_MODEL_PATH,
  'ani2x-jax-model0': DEFAULT_SINGLE_MODEL_PATH,
}


def neighbor_list_featurizer(displacement_fn):
  def featurize(position, neighbor, **kwargs):
    num_atoms = position.shape[0]
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    idx = jnp.asarray(neighbor.idx, dtype=jnp.int32)
    mask = partition.neighbor_list_mask(neighbor)
    safe_idx = jnp.where(mask, idx, atom_ids[:, None])
    d = space.map_neighbor(partial(displacement_fn, **kwargs))
    edge_vectors = d(position, position[safe_idx])
    edge_vectors = jnp.where(mask[..., None], edge_vectors, 0.0)
    return edge_vectors, safe_idx, mask

  return featurize


def piecewise_cutoff(distance, cutoff: float):
  """ANI cosine cutoff."""
  return 0.5 * jnp.cos(distance * jnp.pi / cutoff) + 0.5


def species_index_table(species_order: tuple[int, ...]) -> tuple[int, ...]:
  table = [-1] * (max(species_order) + 1)
  for species_index, atomic_number in enumerate(species_order):
    table[atomic_number] = species_index
  return tuple(table)


def pair_index_table(num_species: int) -> tuple[tuple[int, ...], ...]:
  table = np.zeros((num_species, num_species), dtype=np.int32)
  pair_index = 0
  for species_i in range(num_species):
    for species_j in range(species_i, num_species):
      table[species_i, species_j] = pair_index
      table[species_j, species_i] = pair_index
      pair_index += 1
  return tuple(tuple(int(index) for index in row) for row in table)


class ANI2xCheckpoint(eqx.Module):
  """Full on-disk ANI2x checkpoint leaves before active-species pruning."""

  atom_energies: jnp.ndarray
  layer_weights: list
  layer_biases: list

  def __init__(self, config: dict):
    num_species = len(config['species_order'])
    num_models = config['num_models']
    network_sizes = tuple(config['network_sizes'])

    # Self-energies are large constants kept in f64; weights are f32.
    self.atom_energies = jnp.zeros(num_species, dtype=jnp.float64)
    self.layer_weights = []
    self.layer_biases = []
    for layer_index in range(len(network_sizes) - 1):
      d_in = network_sizes[layer_index]
      d_out = network_sizes[layer_index + 1]
      self.layer_weights.append(
        jnp.zeros((num_models, num_species, d_in, d_out), dtype=jnp.float32)
      )
      self.layer_biases.append(
        jnp.zeros((num_models, num_species, d_out), dtype=jnp.float32)
      )


def active_pair_ids(
  active_species: tuple[int, ...],
  pair_to_index: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
  active_species_np = np.asarray(active_species, dtype=np.int32)
  pair_to_index_np = np.asarray(pair_to_index, dtype=np.int32)
  return tuple(
    int(pair_id)
    for pair_id in np.unique(
      pair_to_index_np[np.ix_(active_species_np, active_species_np)]
    )
  )


def lookup_table(
  active_ids: tuple[int, ...], full_size: int
) -> tuple[int, ...]:
  active_ids_np = np.asarray(active_ids, dtype=np.int32)
  lookup = np.full(full_size, -1, dtype=np.int32)
  lookup[active_ids_np] = np.arange(len(active_ids_np), dtype=np.int32)
  return tuple(int(x) for x in lookup.tolist())


def basis_block_columns(
  active_ids: tuple[int, ...],
  block_width: int,
  *,
  offset: int = 0,
) -> np.ndarray:
  active_ids_np = np.asarray(active_ids, dtype=np.int32)
  block_offsets = active_ids_np[:, None] * block_width
  block_columns = np.arange(block_width, dtype=np.int32)
  return (offset + block_offsets + block_columns).reshape(-1)


def first_layer_columns(
  active_species: tuple[int, ...],
  active_pairs: tuple[int, ...],
  *,
  num_species: int,
  radial_divisions: int,
  angular_basis_width: int,
) -> np.ndarray:
  radial_cols = basis_block_columns(active_species, radial_divisions)
  angular_cols = basis_block_columns(
    active_pairs,
    angular_basis_width,
    offset=num_species * radial_divisions,
  )
  return np.concatenate((radial_cols, angular_cols))


class ANI2x(eqx.Module):
  # Runtime leaves pruned from the full checkpoint.
  atom_energies: jnp.ndarray
  layer_weights: list
  layer_biases: list

  radial_eta: float = eqx.field(static=True)
  angular_eta: float = eqx.field(static=True)
  zeta: float = eqx.field(static=True)
  radial_cutoff: float = eqx.field(static=True)
  angular_cutoff: float = eqx.field(static=True)
  celu_alpha: float = eqx.field(static=True)
  radial_shifts: tuple[float, ...] = eqx.field(static=True)
  angular_shifts: tuple[float, ...] = eqx.field(static=True)
  angular_radial_shifts: tuple[float, ...] = eqx.field(static=True)
  species_to_index: tuple[int, ...] = eqx.field(static=True)
  pair_to_index: tuple[tuple[int, ...], ...] = eqx.field(static=True)
  radial_divisions: int = eqx.field(static=True)
  angular_basis_width: int = eqx.field(static=True)
  species_lookup: tuple[int, ...] = eqx.field(static=True)
  pair_lookup: tuple[int, ...] = eqx.field(static=True)
  num_models: int = eqx.field(static=True)
  num_active_species: int = eqx.field(static=True)
  num_active_pairs: int = eqx.field(static=True)

  def __init__(
    self,
    *,
    config: dict,
    checkpoint: ANI2xCheckpoint,
    active_species: tuple[int, ...] | None = None,
  ):
    self.radial_eta = config['radial_eta']
    self.angular_eta = config['angular_eta']
    self.zeta = config['zeta']
    self.radial_cutoff = config['radial_cutoff']
    self.angular_cutoff = config['angular_cutoff']
    self.celu_alpha = config['celu_alpha']
    self.radial_shifts = tuple(config['radial_shifts'])
    self.angular_shifts = tuple(config['angular_shifts'])
    self.angular_radial_shifts = tuple(config['angular_radial_shifts'])
    species_order = tuple(int(z) for z in config['species_order'])
    num_species = len(species_order)
    self.species_to_index = species_index_table(species_order)
    self.pair_to_index = pair_index_table(num_species)
    num_species_pairs = num_species * (num_species + 1) // 2
    self.radial_divisions = len(self.radial_shifts)
    self.num_models = int(config['num_models'])
    self.angular_basis_width = len(self.angular_shifts) * len(
      self.angular_radial_shifts
    )

    if active_species is None:
      active_species = tuple(range(num_species))
    else:
      if not active_species:
        raise ValueError('ANI active species cannot be empty.')
      invalid = [x for x in active_species if x < 0 or x >= num_species]
      if invalid:
        raise ValueError(
          'ANI active species must be ANI species indices in '
          f'[0, {num_species}); got {invalid}.'
        )

    active_pairs = active_pair_ids(active_species, self.pair_to_index)

    self.species_lookup = lookup_table(active_species, num_species)
    self.pair_lookup = lookup_table(active_pairs, num_species_pairs)
    first_layer_cols = first_layer_columns(
      active_species,
      active_pairs,
      num_species=num_species,
      radial_divisions=self.radial_divisions,
      angular_basis_width=self.angular_basis_width,
    )

    self.num_active_species = len(active_species)
    self.num_active_pairs = len(active_pairs)

    active_species_idx = jnp.asarray(active_species, dtype=jnp.int32)
    first_layer_cols = jnp.asarray(first_layer_cols, dtype=jnp.int32)
    self.atom_energies = checkpoint.atom_energies[active_species_idx]
    layer_weights = []
    layer_biases = []
    for layer_index, checkpoint_weights in enumerate(checkpoint.layer_weights):
      weights = checkpoint_weights[:, active_species_idx]
      if layer_index == 0:
        weights = weights[:, :, first_layer_cols, :]
      layer_weights.append(weights)
      checkpoint_biases = checkpoint.layer_biases[layer_index]
      layer_biases.append(checkpoint_biases[:, active_species_idx])
    self.layer_weights = layer_weights
    self.layer_biases = layer_biases

  def local_node_energies(
    self,
    positions,
    species,
    *,
    neighbor,
    displacement_fn,
  ):
    species = jnp.asarray(self.species_to_index, dtype=jnp.int32)[
      jnp.asarray(species, dtype=jnp.int32)
    ]
    num_atoms = species.shape[0]
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    species_lookup = jnp.asarray(self.species_lookup, dtype=jnp.int32)
    pair_lookup = jnp.asarray(self.pair_lookup, dtype=jnp.int32)
    pair_to_index = jnp.asarray(self.pair_to_index, dtype=jnp.int32)
    radial_shifts = jnp.asarray(self.radial_shifts, dtype=positions.dtype)
    angular_shifts = jnp.asarray(self.angular_shifts, dtype=positions.dtype)
    angular_radial_shifts = jnp.asarray(
      self.angular_radial_shifts,
      dtype=positions.dtype,
    )

    local_species = species_lookup[species]

    # One list at the radial cutoff also feeds the angular terms; neighbors
    # outside the smaller angular cutoff are masked out below.
    featurize = neighbor_list_featurizer(displacement_fn)
    edge_displacements, safe_neighbors, edge_mask = featurize(
      positions, neighbor
    )
    radial_displacements = angular_displacements = edge_displacements
    radial_safe_neighbors = angular_safe_neighbors = safe_neighbors
    radial_neighbor_mask = angular_neighbor_mask = edge_mask
    local_radial_neighbor_species = local_species[radial_safe_neighbors]

    # R_ij is the distance between atom i and radial neighbor j.
    radial_distance2 = jnp.sum(radial_displacements**2, axis=-1)
    radial_distance = jnp.sqrt(jnp.clip(radial_distance2, min=1e-5))
    radial_real_neighbor = radial_neighbor_mask & (
      radial_safe_neighbors != atom_ids[:, None]
    )

    # Eq. 3: radial symmetry terms, then sum them by species.
    radial_mask = radial_real_neighbor & (radial_distance < self.radial_cutoff)
    radial_switch = (
      piecewise_cutoff(radial_distance, self.radial_cutoff) * radial_mask
    )
    radial_terms = (
      jnp.exp(
        -self.radial_eta * (radial_distance[..., None] - radial_shifts) ** 2
      )
      * (0.25 * radial_switch)[..., None]
    )

    radial_active = local_radial_neighbor_species >= 0
    radial_terms = jnp.where(radial_active[..., None], radial_terms, 0.0)
    radial_one_hot = jax.nn.one_hot(
      local_radial_neighbor_species,
      self.num_active_species,
      dtype=radial_terms.dtype,
    )
    radial_aev = jnp.einsum(
      'nkr,nks->nsr', radial_terms, radial_one_hot
    ).reshape(
      num_atoms,
      self.num_active_species * self.radial_divisions,
    )

    angular_neighbor_species = species[angular_safe_neighbors]

    # Eq. 4/5: angular symmetry terms over unique neighbor pairs around atom i.
    angular_distance2 = jnp.sum(angular_displacements**2, axis=-1)
    angular_distance = jnp.sqrt(jnp.clip(angular_distance2, min=1e-5))
    angular_real_neighbor = angular_neighbor_mask & (
      angular_safe_neighbors != atom_ids[:, None]
    )
    angular_mask = angular_real_neighbor & (
      angular_distance < self.angular_cutoff
    )
    angular_switch = (
      piecewise_cutoff(angular_distance, self.angular_cutoff) * angular_mask
    )
    direction = angular_displacements / jnp.clip(
      angular_distance[..., None], min=1e-5
    )

    neighbor_i_np, neighbor_j_np = np.triu_indices(
      int(neighbor.idx.shape[1]), k=1
    )
    neighbor_i = jnp.asarray(neighbor_i_np, dtype=jnp.int32)
    neighbor_j = jnp.asarray(neighbor_j_np, dtype=jnp.int32)
    pair_mask = angular_mask[:, neighbor_i] & angular_mask[:, neighbor_j]
    cos_angle = jnp.sum(
      direction[:, neighbor_i, :] * direction[:, neighbor_j, :],
      axis=-1,
    )
    angle = jnp.arccos(0.95 * cos_angle)

    angular_part = (1 + jnp.cos(angle[..., None] - angular_shifts)) ** self.zeta
    switch_scale = angular_switch * (2.0 * 0.5**self.zeta) ** 0.5
    angular_part = (
      angular_part
      * (switch_scale[:, neighbor_i] * switch_scale[:, neighbor_j])[..., None]
    )

    scaled_distance = 0.5 * jnp.sqrt(self.angular_eta) * angular_distance
    pair_distance = (
      scaled_distance[:, neighbor_i] + scaled_distance[:, neighbor_j]
    )[..., None]
    angular_radial_part = jnp.exp(
      -((pair_distance - angular_radial_shifts) ** 2)
    )
    angular_terms = (
      angular_part[..., None, :] * angular_radial_part[..., :, None]
    ).reshape(num_atoms, -1, self.angular_basis_width)

    pair_index = pair_to_index[
      angular_neighbor_species[:, neighbor_i],
      angular_neighbor_species[:, neighbor_j],
    ]
    active_pair = pair_lookup[pair_index]
    pair_active = active_pair >= 0
    angular_terms = jnp.where(
      (pair_mask & pair_active)[..., None], angular_terms, 0.0
    )
    pair_one_hot = jax.nn.one_hot(
      active_pair,
      self.num_active_pairs,
      dtype=angular_terms.dtype,
    )
    angular_aev = jnp.einsum(
      'npa,nps->nsa', angular_terms, pair_one_hot
    ).reshape(
      num_atoms,
      self.num_active_pairs * self.angular_basis_width,
    )

    species_selector = jax.nn.one_hot(
      local_species,
      self.num_active_species,
      dtype=positions.dtype,
    )

    def select_atom_species(values_by_species):
      return jnp.einsum('ns,nmso->nmo', species_selector, values_by_species)

    radial_width = radial_aev.shape[-1]
    weights = self.layer_weights[0]
    bias = self.layer_biases[0]

    # Evaluate per-species lanes to avoid materializing per-atom weight tensors.
    x = (
      jnp.einsum('ni,msio->nmso', radial_aev, weights[:, :, :radial_width, :])
      + jnp.einsum(
        'ni,msio->nmso', angular_aev, weights[:, :, radial_width:, :]
      )
      + bias[None, :, :, :]
    )
    x = select_atom_species(x)
    if len(self.layer_weights) > 1:
      x = jax.nn.celu(x, alpha=self.celu_alpha)

    for layer_index in range(1, len(self.layer_weights)):
      weights = self.layer_weights[layer_index]
      bias = self.layer_biases[layer_index]
      x = select_atom_species(jnp.einsum('nmi,msio->nmso', x, weights) + bias)
      if layer_index < len(self.layer_weights) - 1:
        x = jax.nn.celu(x, alpha=self.celu_alpha)

    mlp_energies = x.squeeze(-1)
    atom_energies = jax.lax.stop_gradient(self.atom_energies[local_species])
    return jnp.mean(mlp_energies + atom_energies[:, None], axis=1)

  def __call__(
    self,
    positions,
    species,
    *,
    displacement_fn=None,
    neighbors=None,
  ):
    if displacement_fn is None or neighbors is None:
      raise ValueError(
        'ANI requires a displacement_fn and a neighbor list. Build them '
        'with energy.ani_neighbor_list.'
      )
    node_energies = self.local_node_energies(
      positions,
      species,
      neighbor=neighbors,
      displacement_fn=displacement_fn,
    )
    local_energy = jnp.sum(node_energies)
    return HARTREE_EV * local_energy


def load_model(
  model: str = 'ani2x-jax-ensemble',
  *,
  atomic_numbers=None,
  model_path: str | PathLike | None = None,
  dtype=None,
) -> ANI2x:
  """Load an ANI-2x checkpoint, optionally specialized to a fixed atomic-number set."""

  path = (
    Path(model_path) if model_path is not None else ANI2X_MODEL_PATHS[model]
  )

  if dtype is None:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  with path.open('rb') as handle:
    config = json.loads(handle.readline().decode())
    config = dict(config)
    active_species = None
    if atomic_numbers is not None:
      species_order = tuple(int(z) for z in config['species_order'])
      requested = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
      unsupported = sorted(set(requested.tolist()) - set(species_order))
      if unsupported:
        raise ValueError(
          f'ANI2x does not support atomic numbers {unsupported}.'
        )
      species = jnp.asarray(
        species_index_table(species_order), dtype=jnp.int32
      )[jnp.asarray(atomic_numbers, dtype=jnp.int32)]
      active_species = tuple(
        int(x)
        for x in sorted(set(np.asarray(jax.device_get(species)).tolist()))
      )
    # Need to first load the whole model and then prune out weights for other species
    checkpoint = eqx.tree_deserialise_leaves(handle, ANI2xCheckpoint(config))
    checkpoint = jax.tree_util.tree_map(
      lambda x: (
        x.astype(dtype)
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
        else x
      ),
      checkpoint,
    )
    return ANI2x(
      config=config,
      checkpoint=checkpoint,
      active_species=active_species,
    )
