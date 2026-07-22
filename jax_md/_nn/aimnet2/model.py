# Credit to https://github.com/isayevlab/aimnetcentral for the docs

from __future__ import annotations

import json
import math
from functools import partial
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import erfc
from jax_md import partition, space

jax.config.update('jax_default_matmul_precision', 'highest')

AIMNET2_MODEL_PATHS = {
  'aimnet2-jax': files('jax_md._nn.aimnet2') / 'aimnet2.eqx',
}


def neighbor_list_featurizer(displacement_fn, *, cutoff: float):
  def featurize(position, neighbor, **kwargs):
    num_atoms = position.shape[0]
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    idx = jnp.asarray(neighbor.idx, dtype=jnp.int32)
    valid = partition.neighbor_list_mask(neighbor)
    safe_neighbors = jnp.where(valid, idx, atom_ids[:, None])
    d = space.map_neighbor(partial(displacement_fn, **kwargs))
    edge_vectors = d(position, position[safe_neighbors]).astype(position.dtype)
    distances = safe_norm(edge_vectors, axis=-1)
    edge_mask = (
      valid
      & (safe_neighbors != atom_ids[:, None])
      & (distances > 1.0e-8)
      & (distances < float(cutoff))
    )
    edge_vectors = jnp.where(edge_mask[..., None], edge_vectors, 0.0)
    return edge_vectors, safe_neighbors, edge_mask

  return featurize


def safe_norm(
  x: Array, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24
) -> Array:
  return jnp.sqrt(
    jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps)
  )


class Linear(eqx.Module):
  weight: Array
  bias: Array
  in_dim: int = eqx.field(static=True)

  def __init__(
    self, in_dim: int, out_dim: int, *, dtype: Any = jnp.float32, key: Array
  ):
    weight_key, bias_key = jax.random.split(key)
    lim = jnp.sqrt(1.0 / in_dim)
    self.weight = jax.random.uniform(
      weight_key,
      (out_dim, in_dim),
      dtype=dtype,
      minval=-lim,
      maxval=lim,
    )
    self.bias = jax.random.uniform(
      bias_key,
      (out_dim,),
      dtype=dtype,
      minval=-lim,
      maxval=lim,
    )
    self.in_dim = in_dim

  def __call__(self, x: Array) -> Array:
    if x.shape[-1] != self.in_dim:
      raise ValueError(
        f'Expected feature axis of size {self.in_dim}, got shape {x.shape}.'
      )
    x = x.astype(self.weight.dtype)
    return x @ self.weight.T + self.bias


class MLP(eqx.Module):
  layers: list[Linear]
  sizes: tuple[int, ...] = eqx.field(static=True)

  def __init__(
    self, sizes: tuple[int, ...], *, dtype: Any = jnp.float32, key: Array
  ):
    self.sizes = tuple(int(size) for size in sizes)
    keys = jax.random.split(key, len(sizes) - 1)
    self.layers = [
      Linear(in_dim, out_dim, dtype=dtype, key=subkey)
      for subkey, in_dim, out_dim in zip(keys, self.sizes[:-1], self.sizes[1:])
    ]

  def __call__(self, x: Array, *, last_linear: bool = True) -> Array:
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i < len(self.layers) - 1 or not last_linear:
        x = jax.nn.gelu(x, approximate=False)
    return x


def d3_pair_energy(
  pair_c6ab: Array,
  nci: Array,
  ncj: Array,
  rij: Array,
  r2r4_i: Array,
  r2r4_j: Array,
  *,
  d3_s6: float,
  d3_s8: float,
  d3_a1: float,
  d3_a2: float,
  d3_k3: float,
) -> Array:
  reference_c6 = pair_c6ab[..., 0]
  reference_cn_i = pair_c6ab[..., 1]
  reference_cn_j = pair_c6ab[..., 2]
  num_cn_references = reference_c6.shape[-2] * reference_c6.shape[-1]

  cn_distance2 = (reference_cn_i - nci[..., None, None]) ** 2 + (
    reference_cn_j - ncj[..., None, None]
  ) ** 2
  reference_weights = jnp.exp(d3_k3 * cn_distance2).reshape(
    *rij.shape,
    num_cn_references,
  )
  reference_c6 = reference_c6.reshape(*rij.shape, num_cn_references)
  weight_sum = jnp.sum(reference_weights, axis=-1)
  weighted_c6 = jnp.sum(reference_weights * reference_c6, axis=-1)
  weighted_c6 = jnp.where(weight_sum < 1.0e-5, 0.0, weighted_c6)
  c6 = weighted_c6 / jnp.maximum(weight_sum, 1.0e-5)
  c8 = 3.0 * c6 * r2r4_i * r2r4_j

  bj_radius = (
    d3_a1 * jnp.sqrt(jnp.maximum(c8 / jnp.maximum(c6, 1.0e-30), 0.0)) + d3_a2
  )
  bj_radius2 = bj_radius**2
  bj_radius6 = bj_radius2**3
  bj_radius8 = bj_radius6 * bj_radius2

  e6 = -d3_s6 * c6 / (rij**6 + bj_radius6)
  e8 = -d3_s8 * c8 / (rij**8 + bj_radius8)
  return e6 + e8


def radial_symmetry_functions(
  distance: Array, shifts: Array, eta: Array, cutoff: float
) -> Array:
  cutoff_values = cosine_cutoff(distance, cutoff)
  return (
    jnp.exp(-eta * (distance[..., None] - shifts) ** 2)
    * cutoff_values[..., None]
  )


def cosine_cutoff(distance: Array, cutoff: float) -> Array:
  distance = jnp.clip(distance, 1.0e-6, cutoff)
  return 0.5 * (jnp.cos(distance * jnp.pi / cutoff) + 1.0)


def exp_cutoff(d: Array, rc: float, exp_minus_1: float) -> Array:
  x = jnp.clip(d / rc, 0.0, 1.0 - 1.0e-6)
  return jnp.exp(-1.0 / (1.0 - x**2)) / exp_minus_1


def short_range_coulomb_dense(
  charges: Array,
  d: Array,
  neighbors: Array,
  edge_mask: Array,
  *,
  coulomb_rc: float,
  coulomb_factor: float,
  exp_minus_1: float,
) -> Array:
  q_ij = charges[:, None] * charges[neighbors]
  inv_d = 1.0 / jnp.maximum(d, 1.0e-8)
  fc = exp_cutoff(d, coulomb_rc, exp_minus_1)
  e = coulomb_factor * (fc * q_ij * inv_d).astype(jnp.float64)
  return jnp.sum(jnp.where(edge_mask, e, 0.0))


def dsf_coulomb_dense(
  charges: Array,
  positions: Array,
  neighbor,
  displacement_fn,
  *,
  cutoff: float,
  alpha: float,
  coulomb_factor: float,
) -> Array:
  featurize = neighbor_list_featurizer(displacement_fn, cutoff=float(cutoff))
  edge_vectors, safe_neighbors, edge_mask = featurize(positions, neighbor)
  d = safe_norm(edge_vectors, axis=-1)
  rc = float(cutoff)
  erfc_alpha_rc = float(math.erfc(float(alpha) * rc))
  c2 = erfc_alpha_rc / rc
  c3 = c2 / rc
  c4 = (
    2.0
    * float(alpha)
    * math.exp(-((float(alpha) * rc) ** 2))
    / (rc * math.sqrt(math.pi))
  )
  j_dsf = erfc(float(alpha) * d) / jnp.maximum(d, 1.0e-8)
  j_dsf = j_dsf - c2 + (d - rc) * (c3 + c4)
  q_ij = charges[:, None] * charges[safe_neighbors]
  e = coulomb_factor * (q_ij * j_dsf).astype(jnp.float64)
  return jnp.sum(jnp.where(edge_mask, e, 0.0))


def simple_coulomb_all_pairs(
  positions: Array,
  charges: Array,
  hartree_bohr: float,
) -> Array:
  num_atoms = positions.shape[0]
  delta = positions[:, None, :] - positions[None, :, :]
  distance = jnp.sqrt(jnp.maximum(jnp.sum(delta * delta, axis=-1), 1.0e-12))
  pair_mask = jnp.arange(num_atoms)[:, None] < jnp.arange(num_atoms)[None, :]
  pair_energy = (
    hartree_bohr
    * charges[:, None]
    * charges[None, :]
    / jnp.maximum(
      distance,
      1.0e-8,
    )
  )
  return jnp.sum(jnp.where(pair_mask, pair_energy, 0.0).astype(jnp.float64))


class AIMNet2Layer(eqx.Module):
  afv: Array
  shifts: Array
  eta: Array
  conv_a_agh: Array
  conv_q_agh: Array
  mlp0: MLP
  mlp1: MLP
  mlp2: MLP
  nfeature: int = eqx.field(static=True)
  nshifts: int = eqx.field(static=True)
  ncharge: int = eqx.field(static=True)
  ncomb_v: int = eqx.field(static=True)
  mlp_last_linear: tuple[bool, ...] = eqx.field(static=True)

  def __init__(
    self,
    *,
    config: dict[str, Any],
    dtype: Any = jnp.float32,
    keys: Array,
  ):
    self.nfeature = int(config['nfeature'])
    self.nshifts = int(config['nshifts'])
    self.ncharge = int(config['ncharge'])
    self.ncomb_v = int(config['ncomb_v'])
    self.mlp_last_linear = tuple(bool(x) for x in config['mlp_last_linear'])
    num_species = len(config['implemented_species'])
    self.afv = jnp.zeros(
      (num_species, self.nfeature * self.nshifts), dtype=dtype
    )
    self.shifts = jnp.zeros((self.nshifts,), dtype=dtype)
    self.eta = jnp.zeros((), dtype=dtype)
    self.conv_a_agh = jnp.zeros(
      (self.nfeature, self.nshifts, self.ncomb_v),
      dtype=dtype,
    )
    self.conv_q_agh = jnp.zeros(
      (self.ncharge, self.nshifts, self.ncomb_v),
      dtype=dtype,
    )
    self.mlp0 = MLP(config['mlp0_sizes'], dtype=dtype, key=keys[0])
    self.mlp1 = MLP(config['mlp1_sizes'], dtype=dtype, key=keys[1])
    self.mlp2 = MLP(config['mlp2_sizes'], dtype=dtype, key=keys[2])

  def _atomic_embedding_features(
    self,
    atomic_embeddings: Array,
    g_ijs: Array,
    unit_vectors: Array,
    neighbors: Array,
    edge_mask: Array,
  ) -> Array:
    neighbor_embeddings = atomic_embeddings[neighbors]
    neighbor_embeddings = jnp.where(
      edge_mask[..., None, None],
      neighbor_embeddings,
      0.0,
    )
    scalar_features = jnp.sum(
      neighbor_embeddings * g_ijs[:, :, None, :], axis=1
    )
    vector_features = jnp.einsum(
      'nkag,nkg,nkd->nagd',
      neighbor_embeddings,
      g_ijs,
      unit_vectors,
    )
    num_atoms = atomic_embeddings.shape[0]
    scalar_features = scalar_features.reshape(num_atoms, -1)
    vector_features = jnp.einsum(
      'agh,nagd->nahd',
      self.conv_a_agh,
      vector_features,
    )
    vector_features = jnp.sum(vector_features**2, axis=-1).reshape(
      num_atoms, -1
    )
    return jnp.concatenate([scalar_features, vector_features], axis=-1)

  def _charge_features(
    self,
    partial_charges: Array,
    g_ijs: Array,
    unit_vectors: Array,
    neighbors: Array,
    edge_mask: Array,
  ) -> Array:
    neighbor_charges = partial_charges[neighbors]
    neighbor_charges = jnp.where(edge_mask[..., None], neighbor_charges, 0.0)
    scalar_features = jnp.einsum('nka,nkg->nag', neighbor_charges, g_ijs)
    vector_features = jnp.einsum(
      'nka,nkg,nkd->nagd',
      neighbor_charges,
      g_ijs,
      unit_vectors,
    )
    num_atoms = partial_charges.shape[0]
    scalar_features = scalar_features.reshape(num_atoms, -1)
    vector_features = jnp.einsum(
      'agh,nagd->nahd',
      self.conv_q_agh,
      vector_features,
    )
    vector_features = jnp.sum(vector_features**2, axis=-1).reshape(
      num_atoms, -1
    )
    return jnp.concatenate([scalar_features, vector_features], axis=-1)

  def _neural_charge_equilibration(
    self,
    partial_charges: Array,
    charge_weights: Array,
    total_charge: Array | float = 0.0,
  ) -> Array:
    weights = charge_weights**2
    weight_sum = jnp.sum(weights, axis=0, keepdims=True) + 1.0e-6
    predicted_charge = jnp.sum(partial_charges, axis=0, keepdims=True)
    return partial_charges + (weights / weight_sum) * (
      total_charge - predicted_charge
    )

  def __call__(
    self,
    species,
    unit_vectors,
    g_ijs,
    neighbors,
    edge_mask,
    total_charge,
  ):
    num_atoms = species.shape[0]
    nfeature, nshifts, ncharge = self.nfeature, self.nshifts, self.ncharge

    atomic_embeddings = self.afv[species].reshape(num_atoms, nfeature, nshifts)
    embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
    out0 = self.mlp0(
      jnp.concatenate(
        [
          embedding_flat,
          self._atomic_embedding_features(
            atomic_embeddings,
            g_ijs,
            unit_vectors,
            neighbors,
            edge_mask,
          ),
        ],
        axis=-1,
      ),
      last_linear=self.mlp_last_linear[0],
    )
    partial_charges = self._neural_charge_equilibration(
      out0[:, :ncharge],
      out0[:, ncharge : 2 * ncharge],
      total_charge,
    )
    atomic_embeddings = (embedding_flat + out0[:, 2 * ncharge :]).reshape(
      num_atoms,
      nfeature,
      nshifts,
    )

    embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
    out1 = self.mlp1(
      jnp.concatenate(
        [
          embedding_flat,
          self._atomic_embedding_features(
            atomic_embeddings,
            g_ijs,
            unit_vectors,
            neighbors,
            edge_mask,
          ),
          partial_charges,
          self._charge_features(
            partial_charges,
            g_ijs,
            unit_vectors,
            neighbors,
            edge_mask,
          ),
        ],
        axis=-1,
      ),
      last_linear=self.mlp_last_linear[1],
    )
    partial_charges = self._neural_charge_equilibration(
      partial_charges + out1[:, :ncharge],
      out1[:, ncharge : 2 * ncharge],
      total_charge,
    )
    atomic_embeddings = (embedding_flat + out1[:, 2 * ncharge :]).reshape(
      num_atoms,
      nfeature,
      nshifts,
    )

    embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
    aim_vectors = self.mlp2(
      jnp.concatenate(
        [
          embedding_flat,
          self._atomic_embedding_features(
            atomic_embeddings,
            g_ijs,
            unit_vectors,
            neighbors,
            edge_mask,
          ),
          partial_charges,
          self._charge_features(
            partial_charges,
            g_ijs,
            unit_vectors,
            neighbors,
            edge_mask,
          ),
        ],
        axis=-1,
      ),
      last_linear=self.mlp_last_linear[2],
    )
    return aim_vectors, partial_charges


class LocalEnergyHead(eqx.Module):
  energy_mlp: MLP
  atomic_shifts: Array

  def __init__(
    self,
    *,
    config: dict[str, Any],
    dtype: Any = jnp.float32,
    key: Array,
  ):
    self.energy_mlp = MLP(config['energy_sizes'], dtype=dtype, key=key)
    # Self-energy shifts are large constants kept in f64; weights are f32.
    self.atomic_shifts = jnp.zeros(
      (len(config['implemented_species']),), dtype=jnp.float64
    )

  def __call__(self, aim_vectors: Array, species: Array) -> Array:
    atom_local_energy = self.energy_mlp(aim_vectors, last_linear=True).squeeze(
      -1
    )
    return atom_local_energy.astype(jnp.float64) + self.atomic_shifts[species]


class AIMNet2(eqx.Module):
  bohr_a: float = eqx.field(static=True)
  coulomb_factor: float = eqx.field(static=True)
  cutoff: float = eqx.field(static=True)
  nfeature: int = eqx.field(static=True)
  nshifts: int = eqx.field(static=True)
  ncharge: int = eqx.field(static=True)
  ncomb_v: int = eqx.field(static=True)
  coulomb_rc: float = eqx.field(static=True)
  d3_k1: float = eqx.field(static=True)
  d3_k3: float = eqx.field(static=True)
  ev_to_kjmol: float = eqx.field(static=True)
  exp_minus_1: float = eqx.field(static=True)
  hartree_bohr: float = eqx.field(static=True)
  hartree_ev: float = eqx.field(static=True)
  mlp_last_linear: tuple[bool, ...] = eqx.field(static=True)
  implemented_species: tuple[int, ...] = eqx.field(static=True)
  species_lookup: tuple[int, ...] = eqx.field(static=True)
  d3_s6: float = eqx.field(static=True)
  d3_s8: float = eqx.field(static=True)
  d3_a1: float = eqx.field(static=True)
  d3_a2: float = eqx.field(static=True)
  lr_cutoff: float = eqx.field(static=True)
  dsf_alpha: float = eqx.field(static=True)
  layer: AIMNet2Layer
  energy_head: LocalEnergyHead
  d3_c6ab: Array
  d3_rcov: Array
  d3_r2r4: Array

  def __init__(
    self,
    *,
    config: dict[str, Any],
    dtype: Any = jnp.float32,
    key: Array = jax.random.PRNGKey(0),
  ):
    keys = jax.random.split(key, 4)
    self.bohr_a = float(config['bohr_a'])
    self.coulomb_factor = float(config['coulomb_factor'])
    self.cutoff = float(config['cutoff'])
    self.nfeature = int(config['nfeature'])
    self.nshifts = int(config['nshifts'])
    self.ncharge = int(config['ncharge'])
    self.ncomb_v = int(config['ncomb_v'])
    self.coulomb_rc = float(config['coulomb_rc'])
    self.d3_k1 = float(config['d3_k1'])
    self.d3_k3 = float(config['d3_k3'])
    self.ev_to_kjmol = float(config['ev_to_kjmol'])
    self.exp_minus_1 = float(config['exp_minus_1'])
    self.hartree_bohr = float(config['hartree_bohr'])
    self.hartree_ev = float(config['hartree_ev'])
    self.mlp_last_linear = tuple(bool(x) for x in config['mlp_last_linear'])
    self.implemented_species = tuple(
      int(x) for x in config['implemented_species']
    )
    species_lookup = [-1] * (max(self.implemented_species) + 1)
    for index, atomic_number in enumerate(self.implemented_species):
      species_lookup[atomic_number] = index
    self.species_lookup = tuple(species_lookup)
    self.d3_s6 = float(config['d3_s6'])
    self.d3_s8 = float(config['d3_s8'])
    self.d3_a1 = float(config['d3_a1'])
    self.d3_a2 = float(config['d3_a2'])
    self.lr_cutoff = float(config['lr_cutoff'])
    self.dsf_alpha = float(config['dsf_alpha'])
    self.layer = AIMNet2Layer(
      config=config,
      dtype=dtype,
      keys=keys[:3],
    )
    self.energy_head = LocalEnergyHead(
      config=config,
      dtype=dtype,
      key=keys[3],
    )
    self.d3_c6ab = jnp.zeros(
      tuple(int(v) for v in config['d3_c6ab_shape']), dtype=dtype
    )
    self.d3_rcov = jnp.zeros(
      tuple(int(v) for v in config['d3_rcov_shape']), dtype=dtype
    )
    self.d3_r2r4 = jnp.zeros(
      tuple(int(v) for v in config['d3_r2r4_shape']), dtype=dtype
    )

  def dispersion_energy(
    self, positions, species_idx, neighbor, displacement_fn
  ):
    featurize = neighbor_list_featurizer(
      displacement_fn, cutoff=float(self.lr_cutoff)
    )
    edge_vectors, safe_neighbors, edge_mask = featurize(positions, neighbor)
    rij = safe_norm(edge_vectors, axis=-1) / float(self.bohr_a)
    sp_i = species_idx[:, None]
    sp_j = species_idx[safe_neighbors]

    rr = (self.d3_rcov[sp_i] + self.d3_rcov[sp_j]) / jnp.maximum(rij, 1.0e-8)
    damp = 1.0 / (1.0 + jnp.exp(-float(self.d3_k1) * (rr - 1.0)))
    cn = jnp.sum(jnp.where(edge_mask, damp, 0.0), axis=1)

    atom_ids = jnp.arange(positions.shape[0], dtype=jnp.int32)
    pair_mask = edge_mask & (atom_ids[:, None] < safe_neighbors)
    e_pair = d3_pair_energy(
      self.d3_c6ab[sp_i, sp_j],
      cn[:, None],
      cn[safe_neighbors],
      rij,
      self.d3_r2r4[sp_i],
      self.d3_r2r4[sp_j],
      d3_s6=float(self.d3_s6),
      d3_s8=float(self.d3_s8),
      d3_a1=float(self.d3_a1),
      d3_a2=float(self.d3_a2),
      d3_k3=float(self.d3_k3),
    )
    return jnp.sum(
      jnp.where(pair_mask, e_pair.astype(jnp.float64), 0.0)
    ) * float(self.hartree_ev)

  def coulomb_energy(
    self,
    partial_charges,
    positions,
    r_ij,
    neighbors,
    edge_mask,
    lr_neighbor,
    displacement_fn,
    periodic,
  ):
    partial_charges = partial_charges.squeeze(-1)
    local_coulomb = short_range_coulomb_dense(
      partial_charges,
      r_ij,
      neighbors,
      edge_mask,
      coulomb_rc=self.coulomb_rc,
      coulomb_factor=self.coulomb_factor,
      exp_minus_1=self.exp_minus_1,
    )
    if not periodic:
      total_coulomb = simple_coulomb_all_pairs(
        positions,
        partial_charges,
        hartree_bohr=self.hartree_bohr,
      )
    else:
      total_coulomb = dsf_coulomb_dense(
        partial_charges,
        positions,
        lr_neighbor,
        displacement_fn,
        cutoff=float(self.lr_cutoff),
        alpha=float(self.dsf_alpha),
        coulomb_factor=self.coulomb_factor,
      )
    return total_coulomb - local_coulomb

  def local_node_energies_and_charges(
    self,
    positions: Array,
    species: Array,
    *,
    neighbor,
    displacement_fn,
    total_charge: Array | float = 0.0,
  ) -> tuple[Array, Array, Array, Array, Array]:
    """Return local node energies plus intermediates needed by global terms."""

    model_dtype = self.layer.afv.dtype
    positions = positions.astype(model_dtype)
    total_charge = jnp.asarray(total_charge, dtype=model_dtype)
    species = jnp.asarray(self.species_lookup, dtype=jnp.int32)[
      jnp.asarray(species, dtype=jnp.int32)
    ]
    featurize = neighbor_list_featurizer(
      displacement_fn, cutoff=float(self.cutoff)
    )
    local_vectors, neighbors, edge_mask = featurize(positions, neighbor)
    r_ij = safe_norm(local_vectors, axis=-1)
    unit_vectors = local_vectors / jnp.maximum(r_ij[..., None], 1.0e-8)
    g_ijs = radial_symmetry_functions(
      r_ij,
      self.layer.shifts,
      self.layer.eta,
      self.cutoff,
    )
    g_ijs = jnp.where(edge_mask[..., None], g_ijs, 0.0)
    aim_vectors, partial_charges = self.layer(
      species,
      unit_vectors,
      g_ijs,
      neighbors,
      edge_mask,
      total_charge,
    )

    node_energies = self.energy_head(aim_vectors, species)
    return node_energies, partial_charges, r_ij, neighbors, edge_mask

  def __call__(
    self,
    positions: Array,
    species: Array,
    *,
    displacement_fn=None,
    neighbors=None,
    lr_neighbors=None,
    periodic: bool = True,
    total_charge: Array | float = 0.0,
  ) -> Array:
    if displacement_fn is None or neighbors is None or lr_neighbors is None:
      raise ValueError(
        'AIMNet2 requires a displacement_fn and short- and long-range '
        'neighbor lists. Build them with energy.aimnet2_neighbor_list.'
      )
    positions64 = positions.astype(jnp.float64)
    (
      node_energies,
      partial_charges,
      r_ij,
      neighbor_ids,
      edge_mask,
    ) = self.local_node_energies_and_charges(
      positions64,
      species,
      neighbor=neighbors,
      displacement_fn=displacement_fn,
      total_charge=total_charge,
    )
    local_energy = jnp.sum(node_energies)
    coulomb_energy = self.coulomb_energy(
      partial_charges,
      positions64,
      r_ij,
      neighbor_ids,
      edge_mask,
      lr_neighbors,
      displacement_fn,
      periodic,
    )
    species_idx = jnp.asarray(self.species_lookup, dtype=jnp.int32)[
      jnp.asarray(species, dtype=jnp.int32)
    ]
    dispersion_energy = self.dispersion_energy(
      positions64, species_idx, lr_neighbors, displacement_fn
    ).astype(jnp.float64)

    total_energy = local_energy + coulomb_energy + dispersion_energy
    return total_energy.astype(jnp.float64)


def load_model(
  model: str = 'aimnet2-jax',
  *,
  model_path: str | PathLike | None = None,
  dtype=None,
) -> AIMNet2:
  path = (
    Path(model_path) if model_path is not None else AIMNET2_MODEL_PATHS[model]
  )
  if dtype is None:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  with path.open('rb') as handle:
    config = dict(json.loads(handle.readline().decode()))
    template = AIMNet2(config=config, dtype=jnp.float32)
    model = eqx.tree_deserialise_leaves(handle, template)
    return jax.tree_util.tree_map(
      lambda x: (
        x.astype(dtype)
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
        else x
      ),
      model,
    )
