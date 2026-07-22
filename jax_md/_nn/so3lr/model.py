from __future__ import annotations

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
from jax import Array
from jax.ops import segment_sum
from jax_md import partition

jax.config.update('jax_default_matmul_precision', 'highest')

SO3LR_MODEL_PATHS = {
  'so3lr': files('jax_md._nn.so3lr') / 'so3lr.eqx',
}


def neighbor_list_featurizer(displacement_fn):
  def featurize(position, neighbor, **kwargs):
    if not partition.is_sparse(neighbor.format):
      raise ValueError('SO3LR requires a JAX-MD sparse neighbor list.')
    idx_j, idx_i = jnp.asarray(neighbor.idx, dtype=jnp.int32)
    num_atoms = position.shape[0]
    valid = (idx_i < num_atoms) & (idx_j < num_atoms)
    safe_idx_i = jnp.where(valid, idx_i, 0)
    safe_idx_j = jnp.where(valid, idx_j, 0)
    d = jax.vmap(partial(displacement_fn, **kwargs))
    edges = d(position[safe_idx_j], position[safe_idx_i])
    edges = jnp.where(valid[:, None], edges, 0.0).astype(position.dtype)
    return idx_i, idx_j, edges

  return featurize


def safe_mask(mask, fn, operand, placeholder=0.0):
  masked = jnp.where(mask, operand, 0.0)
  return jnp.where(mask, fn(masked), placeholder)


def safe_scale(x, scale, placeholder=0.0):
  return safe_mask(scale != 0.0, lambda y: scale * y, x, placeholder)


def safe_norm(x, axis=-1, placeholder=0.0):
  y = jnp.sum(jnp.square(x), axis=axis)
  return safe_mask(y > 0.0, jnp.sqrt, y, placeholder)


def bernstein_basis(r_ij, *, n_rbf: int, gamma: float = 0.9448630629184640):
  log_factorial_n = sum(np.log(np.arange(1, n_rbf)))
  b = jnp.asarray(
    [
      log_factorial_n
      - sum(np.log(np.arange(1, k + 1)))
      - sum(np.log(np.arange(1, n_rbf - k)))
      for k in range(n_rbf)
    ],
    dtype=r_ij.dtype,
  )
  k = jnp.arange(n_rbf, dtype=r_ij.dtype)
  k_rev = jnp.arange(n_rbf, dtype=r_ij.dtype)[::-1]
  gamma_arr = jnp.asarray(np.float32(gamma), dtype=r_ij.dtype)
  exp_r = jnp.exp(-gamma_arr * r_ij)

  def log_poly(x):
    k_x = jnp.where(k != 0.0, k * jnp.log(x), 0.0)
    kk_x = jnp.where(k_rev != 0.0, k_rev * jnp.log(1.0 - x), 0.0)
    return b + k_x + kk_x

  return safe_mask(
    (exp_r != 0.0) & (exp_r != 1.0), lambda y: jnp.exp(log_poly(y)), exp_r, 0.0
  )


def phys_cutoff(r, r_cut: float):
  return safe_mask(
    r < r_cut,
    lambda x: (
      1.0
      - 6.0 * (x / r_cut) ** 5
      + 15.0 * (x / r_cut) ** 4
      - 10.0 * (x / r_cut) ** 3
    ),
    r,
    0.0,
  )


def spherical_harmonics_1_to_4(r):
  x, y, z = jnp.split(r, 3, axis=-1)
  y1 = jnp.concatenate(
    [
      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * y,
      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * z,
      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * x,
    ],
    axis=-1,
  )
  y2 = jnp.concatenate(
    [
      0.5 * jnp.sqrt(15.0 / jnp.pi) * x * y,
      0.5 * jnp.sqrt(15.0 / jnp.pi) * y * z,
      0.25 * jnp.sqrt(5.0 / jnp.pi) * (3.0 * z**2 - 1.0),
      0.5 * jnp.sqrt(15.0 / jnp.pi) * x * z,
      0.25 * jnp.sqrt(15.0 / jnp.pi) * (x**2 - y**2),
    ],
    axis=-1,
  )
  y3 = jnp.concatenate(
    [
      0.25 * jnp.sqrt(35.0 / (2.0 * jnp.pi)) * y * (3.0 * x**2 - y**2),
      0.5 * jnp.sqrt(105.0 / jnp.pi) * x * y * z,
      0.25 * jnp.sqrt(21.0 / (2.0 * jnp.pi)) * y * (5.0 * z**2 - 1.0),
      0.25 * jnp.sqrt(7.0 / jnp.pi) * (5.0 * z**3 - 3.0 * z),
      0.25 * jnp.sqrt(21.0 / (2.0 * jnp.pi)) * x * (5.0 * z**2 - 1.0),
      0.25 * jnp.sqrt(105.0 / jnp.pi) * (x**2 - y**2) * z,
      0.25 * jnp.sqrt(35.0 / (2.0 * jnp.pi)) * x * (x**2 - 3.0 * y**2),
    ],
    axis=-1,
  )
  y4 = jnp.concatenate(
    [
      0.75 * jnp.sqrt(35.0 / jnp.pi) * x * y * (x**2 - y**2),
      0.75 * jnp.sqrt(35.0 / (2.0 * jnp.pi)) * y * (3.0 * x**2 - y**2) * z,
      0.75 * jnp.sqrt(5.0 / jnp.pi) * x * y * (7.0 * z**2 - 1.0),
      0.75 * jnp.sqrt(5.0 / (2.0 * jnp.pi)) * y * (7.0 * z**3 - 3.0 * z),
      3.0 / 16.0 * jnp.sqrt(1.0 / jnp.pi) * (35.0 * z**4 - 30.0 * z**2 + 3.0),
      0.75 * jnp.sqrt(5.0 / (2.0 * jnp.pi)) * x * (7.0 * z**3 - 3.0 * z),
      0.375 * jnp.sqrt(5.0 / jnp.pi) * (x**2 - y**2) * (7.0 * z**2 - 1.0),
      0.75 * jnp.sqrt(35.0 / (2.0 * jnp.pi)) * x * (x**2 - 3.0 * y**2) * z,
      3.0
      / 16.0
      * jnp.sqrt(35.0 / jnp.pi)
      * (x**2 * (x**2 - 3.0 * y**2) - y**2 * (3.0 * x**2 - y**2)),
    ],
    axis=-1,
  )
  return jnp.concatenate([y1, y2, y3, y4], axis=-1)


def sigma_switch(x):
  return safe_mask(x > 0.0, lambda u: jnp.exp(-1.0 / u), x, 0.0)


def switching_fn(x, x_on, x_off):
  c = (x - x_on) / (x_off - x_on)
  return sigma_switch(1.0 - c) / (sigma_switch(1.0 - c) + sigma_switch(c))


class Linear(eqx.Module):
  kernel: Array
  bias: Array | None

  def __init__(
    self,
    *,
    shape: tuple[int, int],
    use_bias: bool = True,
    dtype=jnp.float32,
  ):
    self.kernel = jnp.zeros(shape, dtype=dtype)
    self.bias = jnp.zeros(shape[-1], dtype=dtype) if use_bias else None

  def __call__(self, x: Array, *, use_bias: bool = True) -> Array:
    y = x @ self.kernel
    if use_bias and self.bias is not None:
      y = y + self.bias
    return y


class MLP(eqx.Module):
  layers: tuple[Linear, ...]

  def __init__(
    self,
    *,
    shapes: tuple[tuple[int, int], ...],
    biases: tuple[bool, ...],
    dtype=jnp.float32,
  ):
    self.layers = tuple(
      Linear(shape=shape, use_bias=use_bias, dtype=dtype)
      for shape, use_bias in zip(shapes, biases, strict=True)
    )

  def __call__(
    self,
    x: Array,
    *,
    use_bias: bool = True,
    final_use_bias: bool = True,
    last_linear: bool = True,
  ) -> Array:
    for i, layer in enumerate(self.layers):
      is_last = i == len(self.layers) - 1
      x = layer(x, use_bias=final_use_bias if is_last else use_bias)
      if not is_last or not last_linear:
        x = jax.nn.silu(x)
    return x


class LayerNorm(eqx.Module):
  scale: Array
  bias: Array
  eps: float = eqx.field(static=True)

  def __init__(
    self,
    *,
    num_features: int,
    eps: float = 1.0e-6,
    dtype=jnp.float32,
  ):
    self.scale = jnp.zeros(num_features, dtype=dtype)
    self.bias = jnp.zeros(num_features, dtype=dtype)
    self.eps = float(eps)

  def __call__(self, x: Array) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + jnp.asarray(self.eps, dtype=x.dtype))
    return y * self.scale + self.bias


class ChargeSpinEmbed(eqx.Module):
  embed_0: Array
  embed_1: Array
  embed_2: Array
  residual: MLP

  def __init__(
    self,
    *,
    num_species: int = 119,
    num_features: int = 128,
    dtype=jnp.float32,
  ):
    self.embed_0 = jnp.zeros((num_species, num_features), dtype=dtype)
    self.embed_1 = jnp.zeros((2, num_features), dtype=dtype)
    self.embed_2 = jnp.zeros((2, num_features), dtype=dtype)
    self.residual = MLP(
      shapes=(
        (num_features, num_features),
        (num_features, num_features),
      ),
      biases=(False, False),
      dtype=dtype,
    )

  def __call__(
    self,
    atomic_numbers: Array,
    psi: Array,
  ) -> Array:
    q = self.embed_0[atomic_numbers]
    psi_bucket = jnp.asarray(psi < 0.0, dtype=jnp.int32)
    k = self.embed_1[psi_bucket]
    v = self.embed_2[psi_bucket]
    q_x_k = (q * k).sum(axis=-1) / jnp.sqrt(float(q.shape[-1]))
    y = jax.nn.softplus(q_x_k)
    a = psi * y / jnp.sum(y)
    x = a[:, None] * v
    y = self.residual(jax.nn.silu(x), use_bias=False, final_use_bias=False)
    return x + y


def l0_contraction(ev):
  return jnp.stack(
    [
      jnp.sum(ev[..., 0:3] ** 2, axis=-1)
      / jnp.sqrt(jnp.asarray(3.0, dtype=ev.dtype)),
      jnp.sum(ev[..., 3:8] ** 2, axis=-1)
      / jnp.sqrt(jnp.asarray(5.0, dtype=ev.dtype)),
      jnp.sum(ev[..., 8:15] ** 2, axis=-1)
      / jnp.sqrt(jnp.asarray(7.0, dtype=ev.dtype)),
      jnp.sum(ev[..., 15:24] ** 2, axis=-1) / jnp.asarray(3.0, dtype=ev.dtype),
    ],
    axis=-1,
  )


class AttentionBlock(eqx.Module):
  radial_filter1: MLP
  radial_filter2: MLP
  spherical_filter1: MLP
  spherical_filter2: MLP
  wq1: Array
  wk1: Array
  wq2: Array
  wk2: Array
  wv1: Array

  def __init__(
    self,
    *,
    num_radial_basis_fn: int = 32,
    num_features: int = 128,
    num_heads: int = 4,
    dtype=jnp.float32,
  ):
    head_features = num_features // num_heads
    radial_shapes = (
      (num_radial_basis_fn, num_features),
      (num_features, num_features),
    )
    spherical_shapes = (
      (num_heads, head_features),
      (head_features, num_features),
    )
    self.radial_filter1 = MLP(
      shapes=radial_shapes, biases=(True, True), dtype=dtype
    )
    self.radial_filter2 = MLP(
      shapes=radial_shapes, biases=(True, True), dtype=dtype
    )
    self.spherical_filter1 = MLP(
      shapes=spherical_shapes, biases=(True, True), dtype=dtype
    )
    self.spherical_filter2 = MLP(
      shapes=spherical_shapes, biases=(True, True), dtype=dtype
    )
    attention_shape = (num_heads, head_features, head_features)
    self.wq1 = jnp.zeros(attention_shape, dtype=dtype)
    self.wk1 = jnp.zeros(attention_shape, dtype=dtype)
    self.wq2 = jnp.zeros(attention_shape, dtype=dtype)
    self.wk2 = jnp.zeros(attention_shape, dtype=dtype)
    self.wv1 = jnp.zeros(attention_shape, dtype=dtype)

  def __call__(
    self,
    x: Array,
    ev: Array,
    rbf_ij: Array,
    ylm_ij: Array,
    cut: Array,
    idx_i: Array,
    idx_j: Array,
    *,
    avg_num_neighbors: float,
  ) -> tuple[Array, Array]:
    w1_ij = self.radial_filter1(rbf_ij)
    w2_ij = self.radial_filter2(rbf_ij)
    ev_i = ev[idx_i]
    ev_j = ev[idx_j]
    contracted = l0_contraction(ev_j - ev_i)
    w1_ij = w1_ij + self.spherical_filter1(contracted)
    w2_ij = w2_ij + self.spherical_filter2(contracted)
    w1_ij = w1_ij.reshape(*w1_ij.shape[:-1], 4, -1)
    w2_ij = w2_ij.reshape(*w2_ij.shape[:-1], 4, -1)
    x_h = x.reshape(*x.shape[:-1], 4, -1)

    q1_i = jnp.einsum('Hij,NHj->NHi', self.wq1, x_h)[idx_i]
    k1_j = jnp.einsum('Hij,NHj->NHi', self.wk1, x_h)[idx_j]
    q2_i = jnp.einsum('Hij,NHj->NHi', self.wq2, x_h)[idx_i]
    k2_j = jnp.einsum('Hij,NHj->NHi', self.wk2, x_h)[idx_j]
    nc = jnp.asarray(avg_num_neighbors, dtype=x.dtype)
    alpha1_ij = safe_scale(
      (q1_i * w1_ij * k1_j).sum(axis=-1) / nc, cut[:, None]
    )
    alpha2_ij = safe_scale(
      (q2_i * w2_ij * k2_j).sum(axis=-1) / nc, cut[:, None]
    )
    v_j = jnp.einsum('hij,Nhj->Nhi', self.wv1, x_h)[idx_j]
    x_att = segment_sum(
      alpha1_ij[:, :, None] * v_j, idx_i, num_segments=x.shape[0]
    )
    x_att = x_att.reshape(*x.shape[:-1], -1)
    ev_att = segment_sum(
      jnp.repeat(
        alpha2_ij,
        repeats=jnp.asarray([3, 5, 7, 9], dtype=jnp.int32),
        axis=-1,
        total_repeat_length=24,
      )
      * ylm_ij,
      idx_i,
      num_segments=x.shape[0],
    )
    return x_att, ev_att


class ScalarEquivariantExchange(eqx.Module):
  mlp_layer_2: Linear

  def __init__(
    self,
    *,
    num_features: int = 128,
    num_heads: int = 4,
    dtype=jnp.float32,
  ):
    combined_features = num_features + num_heads
    self.mlp_layer_2 = Linear(
      shape=(combined_features, combined_features), dtype=dtype
    )

  def __call__(
    self,
    scalar_features: Array,
    equivariant_features: Array,
  ) -> tuple[Array, Array]:
    features = jnp.concatenate(
      [scalar_features, l0_contraction(equivariant_features)],
      axis=-1,
    )
    updates = self.mlp_layer_2(features)
    scalar_update, equivariant_update = jnp.split(
      updates,
      [scalar_features.shape[-1]],
      axis=-1,
    )
    equivariant_update = jnp.repeat(
      equivariant_update,
      repeats=jnp.asarray([3, 5, 7, 9], dtype=jnp.int32),
      axis=-1,
      total_repeat_length=24,
    )
    return scalar_update, equivariant_update * equivariant_features


class SO3LRLayer(eqx.Module):
  attention_block: AttentionBlock
  exchange_block: ScalarEquivariantExchange
  layer_normalization_1: LayerNorm
  layer_normalization_2: LayerNorm
  res_mlp_1: MLP

  def __init__(
    self,
    *,
    num_radial_basis_fn: int = 32,
    num_features: int = 128,
    num_heads: int = 4,
    dtype=jnp.float32,
  ):
    self.attention_block = AttentionBlock(
      num_radial_basis_fn=num_radial_basis_fn,
      num_features=num_features,
      num_heads=num_heads,
      dtype=dtype,
    )
    self.exchange_block = ScalarEquivariantExchange(
      num_features=num_features,
      num_heads=num_heads,
      dtype=dtype,
    )
    self.layer_normalization_1 = LayerNorm(
      num_features=num_features, dtype=dtype
    )
    self.layer_normalization_2 = LayerNorm(
      num_features=num_features, dtype=dtype
    )
    self.res_mlp_1 = MLP(
      shapes=((num_features, num_features), (num_features, num_features)),
      biases=(True, True),
      dtype=dtype,
    )

  def __call__(
    self,
    x: Array,
    ev: Array,
    rbf_ij: Array,
    ylm_ij: Array,
    cut: Array,
    idx_i: Array,
    idx_j: Array,
    *,
    avg_num_neighbors: float,
  ) -> tuple[Array, Array]:
    x_att, ev_att = self.attention_block(
      x,
      ev,
      rbf_ij,
      ylm_ij,
      cut,
      idx_i,
      idx_j,
      avg_num_neighbors=avg_num_neighbors,
    )
    x = x + x_att
    ev = ev + ev_att
    x = self.layer_normalization_1(x)
    y = self.res_mlp_1(jax.nn.silu(x))
    x = x + y
    scalar_update, equivariant_update = self.exchange_block(x, ev)
    x = x + scalar_update
    ev = ev + equivariant_update
    x = self.layer_normalization_2(x)
    return x, ev


class LocalNodeEnergyHead(eqx.Module):
  energy_offset: Array
  atomic_scales: Array
  energy_mlp: MLP

  def __init__(
    self,
    *,
    num_species: int = 119,
    num_features: int = 128,
    dtype=jnp.float32,
  ):
    self.energy_offset = jnp.zeros(num_species, dtype=dtype)
    self.atomic_scales = jnp.zeros(num_species, dtype=dtype)
    self.energy_mlp = MLP(
      shapes=((num_features, num_features), (num_features, 1)),
      biases=(True, False),
      dtype=dtype,
    )

  def __call__(
    self,
    *,
    x: Array,
    atomic_numbers: Array,
  ) -> Array:
    energy_offset = self.energy_offset[atomic_numbers]
    atomic_scales = self.atomic_scales[atomic_numbers]
    atomic_energy = self.energy_mlp(x, final_use_bias=False).squeeze(axis=-1)
    return atomic_energy * atomic_scales + energy_offset


class PartialChargesHead(eqx.Module):
  embed_0: Array
  charge_mlp: MLP

  def __init__(
    self,
    *,
    num_species: int = 100,
    num_features: int = 128,
    dtype=jnp.float32,
  ):
    self.embed_0 = jnp.zeros((num_species, 1), dtype=dtype)
    self.charge_mlp = MLP(
      shapes=((num_features, num_features), (num_features, 1)),
      biases=(True, True),
      dtype=dtype,
    )

  def __call__(
    self,
    x: Array,
    atomic_numbers: Array,
    total_charge: Array,
  ) -> Array:
    q_shift = self.embed_0[atomic_numbers].squeeze(axis=-1)
    x_q = self.charge_mlp(x).squeeze(axis=-1) + q_shift
    charge_conservation = (total_charge - jnp.sum(x_q)) / x_q.shape[0]
    return x_q + charge_conservation


class HirshfeldRatiosHead(eqx.Module):
  embed_0: Array
  embed_1: Array
  hirshfeld_mlp: MLP

  def __init__(
    self,
    *,
    num_species: int = 100,
    num_features: int = 128,
    latent_features: int = 64,
    dtype=jnp.float32,
  ):
    self.embed_0 = jnp.zeros((num_species, 1), dtype=dtype)
    self.embed_1 = jnp.zeros((num_species, latent_features), dtype=dtype)
    self.hirshfeld_mlp = MLP(
      shapes=(
        (num_features, latent_features),
        (latent_features, latent_features),
      ),
      biases=(True, True),
      dtype=dtype,
    )

  def __call__(self, x: Array, atomic_numbers: Array) -> Array:
    num_features = x.shape[-1]
    v_shift = self.embed_0[atomic_numbers].squeeze(axis=-1)
    q = self.embed_1[atomic_numbers]
    k = self.hirshfeld_mlp(x)
    qk = (q * k / jnp.sqrt(float(num_features // 2))).sum(axis=-1)
    return jnp.abs(v_shift + qk)


class ZBLRepulsion(eqx.Module):
  a1: Array
  a2: Array
  a3: Array
  a4: Array
  c1: Array
  c2: Array
  c3: Array
  c4: Array
  p: Array
  d: Array
  coulomb_ev_angstrom: float = eqx.field(static=True)

  def __init__(
    self,
    *,
    coulomb_ev_angstrom: float,
    dtype=jnp.float32,
  ):
    self.coulomb_ev_angstrom = float(coulomb_ev_angstrom)
    empty = jnp.zeros(1, dtype=dtype)
    self.a1 = empty
    self.a2 = empty
    self.a3 = empty
    self.a4 = empty
    self.c1 = empty
    self.c2 = empty
    self.c3 = empty
    self.c4 = empty
    self.p = empty
    self.d = empty

  def __call__(
    self,
    atomic_numbers: Array,
    cut: Array,
    idx_i: Array,
    idx_j: Array,
    d_ij: Array,
  ) -> Array:
    # Upstream keeps the published checkpoint leaves in float32, so these
    # parameter-only transforms remain float32 even for float64 inference.
    parameter_dtype = jnp.float32
    a1 = jax.nn.softplus(self.a1.astype(parameter_dtype))
    a2 = jax.nn.softplus(self.a2.astype(parameter_dtype))
    a3 = jax.nn.softplus(self.a3.astype(parameter_dtype))
    a4 = jax.nn.softplus(self.a4.astype(parameter_dtype))
    c1 = jax.nn.softplus(self.c1.astype(parameter_dtype))
    c2 = jax.nn.softplus(self.c2.astype(parameter_dtype))
    c3 = jax.nn.softplus(self.c3.astype(parameter_dtype))
    c4 = jax.nn.softplus(self.c4.astype(parameter_dtype))
    p_exp = jax.nn.softplus(self.p.astype(parameter_dtype))
    d = jax.nn.softplus(self.d.astype(parameter_dtype))
    c_sum = c1 + c2 + c3 + c4
    c1, c2, c3, c4 = c1 / c_sum, c2 / c_sum, c3 / c_sum, c4 / c_sum
    z_i = atomic_numbers[idx_i]
    z_j = atomic_numbers[idx_j]
    z_d_ij = safe_mask(d_ij != 0.0, lambda u: z_i * z_j / u, d_ij, 0.0)
    x = jnp.asarray(self.coulomb_ev_angstrom, dtype=d_ij.dtype) * cut * z_d_ij
    rzd = d_ij * (jnp.power(z_i, p_exp) + jnp.power(z_j, p_exp)) * d
    y = (
      c1 * jnp.exp(-a1 * rzd)
      + c2 * jnp.exp(-a2 * rzd)
      + c3 * jnp.exp(-a3 * rzd)
      + c4 * jnp.exp(-a4 * rzd)
    )
    w = switching_fn(d_ij, x_on=0.0, x_off=1.5)
    e_rep_edge = w * x * y / jnp.asarray(2.0, dtype=d_ij.dtype)
    return segment_sum(e_rep_edge, idx_i, num_segments=atomic_numbers.shape[0])


class EnergyHead(eqx.Module):
  local_node_energies: LocalNodeEnergyHead
  zbl_repulsion: ZBLRepulsion
  partial_charges: PartialChargesHead
  hirshfeld_ratios: HirshfeldRatiosHead
  bohr_angstrom: float = eqx.field(static=True)
  hartree_ev: float = eqx.field(static=True)
  coulomb_ev_angstrom: float = eqx.field(static=True)

  def __init__(
    self,
    *,
    num_features: int = 128,
    num_species: int = 119,
    observable_num_species: int = 100,
    hirshfeld_latent_features: int = 64,
    bohr_angstrom: float,
    hartree_ev: float,
    coulomb_ev_angstrom: float,
    dtype=jnp.float32,
  ):
    self.bohr_angstrom = float(bohr_angstrom)
    self.hartree_ev = float(hartree_ev)
    self.coulomb_ev_angstrom = float(coulomb_ev_angstrom)
    self.local_node_energies = LocalNodeEnergyHead(
      num_species=num_species,
      num_features=num_features,
      dtype=dtype,
    )
    self.zbl_repulsion = ZBLRepulsion(
      coulomb_ev_angstrom=self.coulomb_ev_angstrom,
      dtype=dtype,
    )
    self.partial_charges = PartialChargesHead(
      num_species=observable_num_species,
      num_features=num_features,
      dtype=dtype,
    )
    self.hirshfeld_ratios = HirshfeldRatiosHead(
      num_species=observable_num_species,
      num_features=num_features,
      latent_features=hirshfeld_latent_features,
      dtype=dtype,
    )

  def __call__(
    self,
    *,
    x: Array,
    atomic_numbers: Array,
    idx_i: Array,
    idx_j: Array,
    d_ij: Array,
    cut: Array,
    idx_i_lr: Array,
    idx_j_lr: Array,
    d_ij_lr: Array,
    total_charge: Array,
    cutoff_lr: float,
    dispersion_cutoff_lr_damping: float,
    electrostatic_energy_scale: float,
    dispersion_energy_scale: float,
    fine_structure: float,
    dispersion_alphas: Array,
    dispersion_c6: Array,
  ) -> Array:
    nn_energy = self.local_node_energies(
      x=x,
      atomic_numbers=atomic_numbers,
    )
    zbl = self.zbl_repulsion(atomic_numbers, cut, idx_i, idx_j, d_ij)
    partial_charges = self.partial_charges(
      x,
      atomic_numbers,
      total_charge,
    )
    electrostatic = electrostatic_energy(
      partial_charges,
      idx_i_lr,
      idx_j_lr,
      d_ij_lr,
      cutoff_lr=cutoff_lr,
      sigma=electrostatic_energy_scale,
      coulomb_ev_angstrom=self.coulomb_ev_angstrom,
    )
    hirshfeld = self.hirshfeld_ratios(x, atomic_numbers)
    dispersion = dispersion_energy(
      atomic_numbers,
      hirshfeld,
      idx_i_lr,
      idx_j_lr,
      d_ij_lr,
      cutoff_lr=cutoff_lr,
      cutoff_lr_damping=dispersion_cutoff_lr_damping,
      dispersion_energy_scale=dispersion_energy_scale,
      fine_structure=fine_structure,
      dispersion_alphas=dispersion_alphas,
      dispersion_c6=dispersion_c6,
      bohr_angstrom=self.bohr_angstrom,
      hartree_ev=self.hartree_ev,
    )
    return nn_energy + zbl + electrostatic + dispersion


def electrostatic_energy(
  partial_charges,
  idx_i_lr,
  idx_j_lr,
  d_ij_lr,
  *,
  cutoff_lr,
  sigma,
  coulomb_ev_angstrom,
):
  pairwise = coulomb_erf_shifted_force_smooth(
    partial_charges,
    d_ij_lr,
    idx_i_lr,
    idx_j_lr,
    ke=coulomb_ev_angstrom,
    sigma=sigma,
    cutoff=cutoff_lr,
    cuton=cutoff_lr * 0.45,
  )
  return segment_sum(pairwise, idx_i_lr, num_segments=partial_charges.shape[0])


def dispersion_energy(
  atomic_numbers,
  hirshfeld_ratios,
  idx_i_lr,
  idx_j_lr,
  d_ij_lr,
  *,
  cutoff_lr,
  cutoff_lr_damping,
  dispersion_energy_scale,
  fine_structure,
  dispersion_alphas,
  dispersion_c6,
  bohr_angstrom,
  hartree_ev,
):
  alpha_ij, c6_ij = dispersion_pair_parameters(
    atomic_numbers,
    idx_i_lr,
    idx_j_lr,
    hirshfeld_ratios,
    dispersion_alphas=dispersion_alphas,
    dispersion_c6=dispersion_c6,
  )
  gamma_ij = gamma_cubic_fit(alpha_ij, fine_structure=fine_structure)
  pairwise = vdw_qdo_disp_damp(
    d_ij_lr / jnp.asarray(bohr_angstrom, dtype=d_ij_lr.dtype),
    gamma_ij,
    c6_ij,
    alpha_ij,
    jnp.asarray(dispersion_energy_scale, dtype=d_ij_lr.dtype),
    hartree_ev=hartree_ev,
  )
  w = safe_mask(
    d_ij_lr > 0.0,
    partial(switching_fn, x_on=cutoff_lr - cutoff_lr_damping, x_off=cutoff_lr),
    d_ij_lr,
    0.0,
  )
  pairwise = safe_scale(pairwise, w)
  return segment_sum(pairwise, idx_i_lr, num_segments=atomic_numbers.shape[0])


def coulomb_erf_shifted_force_smooth(
  q,
  rij,
  idx_i,
  idx_j,
  *,
  ke,
  sigma,
  cutoff,
  cuton,
):
  dtype = rij.dtype
  c = jnp.asarray(0.5, dtype=dtype)
  ke = jnp.asarray(ke, dtype=dtype)
  sigma = jnp.asarray(sigma, dtype=dtype)
  cutoff = jnp.asarray(cutoff, dtype=dtype)
  cuton = jnp.asarray(cuton, dtype=dtype)
  valid = (rij > 0.0) & (rij < cutoff)
  rij_safe = jnp.where(valid, rij, 1.0)

  def potential(r):
    return jax.lax.erf(r / sigma) / r

  def force(r):
    return (
      2.0 * r * jnp.exp(-((r / sigma) ** 2)) / (jnp.sqrt(jnp.pi) * sigma)
      - jax.lax.erf(r / sigma)
    ) / r**2

  f = switching_fn(rij_safe, cuton, cutoff)
  pairwise = potential(rij_safe)
  shift = potential(cutoff)
  force_shift = force(cutoff)
  shifted = pairwise - shift - force_shift * (rij_safe - cutoff)
  return jnp.where(
    valid,
    c
    * ke
    * q[idx_i]
    * q[idx_j]
    * (f * (pairwise - shift) + (1.0 - f) * shifted),
    0.0,
  )


def dispersion_pair_parameters(
  atomic_numbers,
  idx_i,
  idx_j,
  hirshfeld_ratios,
  *,
  dispersion_alphas,
  dispersion_c6,
):
  dtype = hirshfeld_ratios.dtype
  zi = atomic_numbers[idx_i] - 1
  zj = atomic_numbers[idx_j] - 1
  vi = hirshfeld_ratios[idx_i]
  vj = hirshfeld_ratios[idx_j]
  alphas = jnp.asarray(dispersion_alphas, dtype=dtype)
  c6 = jnp.asarray(dispersion_c6, dtype=dtype)
  alpha_i = jnp.take(alphas, zi, axis=0) * vi
  c6_i = jnp.take(c6, zi, axis=0) * jnp.square(vi)
  alpha_j = jnp.take(alphas, zj, axis=0) * vj
  c6_j = jnp.take(c6, zj, axis=0) * jnp.square(vj)
  alpha_ij = (alpha_i + alpha_j) / 2.0
  c6_ij = (
    2.0
    * c6_i
    * c6_j
    * alpha_j
    * alpha_i
    / (alpha_i**2 * c6_j + alpha_j**2 * c6_i)
  )
  return alpha_ij, c6_ij


def gamma_cubic_fit(alpha, *, fine_structure):
  dtype = alpha.dtype
  vdw_radius = jnp.asarray(fine_structure, dtype=dtype) ** jnp.asarray(
    -4.0 / 21.0, dtype
  ) * alpha ** jnp.asarray(1.0 / 7.0, dtype)
  b0 = jnp.asarray(-0.00433008, dtype=dtype)
  b1 = jnp.asarray(0.24428889, dtype=dtype)
  b2 = jnp.asarray(0.04125273, dtype=dtype)
  b3 = jnp.asarray(-0.00078893, dtype=dtype)
  sigma = b3 * vdw_radius**3 + b2 * vdw_radius**2 + b1 * vdw_radius + b0
  return jnp.asarray(0.5, dtype=dtype) / jnp.square(sigma)


def vdw_qdo_disp_damp(
  distance, gamma, c6, alpha_ij, gamma_scale, *, hartree_ev
):
  dtype = distance.dtype
  half = jnp.asarray(0.5, dtype=dtype)
  c8 = 5.0 / gamma * c6
  c10 = 245.0 / 8.0 / gamma**2 * c6
  damping_length = gamma_scale * 2.0 * 2.54 * alpha_ij ** (1.0 / 7.0)
  disp = (
    -c6 / (distance**6 + damping_length**6)
    - c8 / (distance**8 + damping_length**8)
    - c10 / (distance**10 + damping_length**10)
  )
  return half * disp * jnp.asarray(hartree_ev, dtype=dtype)


class SO3LR(eqx.Module):
  cutoff: float = eqx.field(static=True)
  long_range_cutoff: float = eqx.field(static=True)
  dispersion_energy_cutoff_lr_damping: float = eqx.field(static=True)
  ev_to_kjmol: float = eqx.field(static=True)
  num_radial_basis_fn: int = eqx.field(static=True)
  avg_num_neighbors: float = eqx.field(static=True)
  electrostatic_energy_scale: float = eqx.field(static=True)
  dispersion_energy_scale: float = eqx.field(static=True)
  fine_structure: float = eqx.field(static=True)
  feature_embedding: Array
  charge_embedding: ChargeSpinEmbed
  spin_embedding: ChargeSpinEmbed
  layers: list[SO3LRLayer]
  energy_head: EnergyHead
  dispersion_alphas: tuple[float, ...] = eqx.field(static=True)
  dispersion_c6: tuple[float, ...] = eqx.field(static=True)

  def __init__(
    self,
    metadata: dict[str, Any],
    hyperparameters: dict[str, Any],
    *,
    dtype=jnp.float32,
  ):
    model_cfg = hyperparameters['model']
    data_cfg = hyperparameters['data']

    self.cutoff = float(metadata['cutoff'])
    self.long_range_cutoff = float(metadata['lr_cutoff'])
    self.dispersion_energy_cutoff_lr_damping = float(
      metadata['dispersion_energy_cutoff_lr_damping']
    )
    self.ev_to_kjmol = float(metadata['ev_to_kjmol'])
    bohr_angstrom = float(metadata['bohr_angstrom'])
    hartree_ev = float(metadata['hartree_ev'])
    coulomb_ev_angstrom = float(metadata['coulomb_ev_angstrom'])

    num_layers = int(model_cfg['num_layers'])
    num_features = int(model_cfg['num_features'])
    num_heads = int(model_cfg['num_heads'])
    num_species = int(model_cfg['num_species'])
    observable_num_species = int(model_cfg['observable_num_species'])
    hirshfeld_latent_features = int(model_cfg['hirshfeld_latent_features'])
    self.num_radial_basis_fn = int(model_cfg['num_radial_basis_fn'])
    self.avg_num_neighbors = float(data_cfg['avg_num_neighbors'])
    self.electrostatic_energy_scale = float(
      model_cfg['electrostatic_energy_scale']
    )
    self.dispersion_energy_scale = float(model_cfg['dispersion_energy_scale'])
    self.fine_structure = float(metadata['fine_structure'])
    self.dispersion_alphas = tuple(
      float(x) for x in metadata['dispersion_alphas']
    )
    self.dispersion_c6 = tuple(float(x) for x in metadata['dispersion_c6'])
    self.feature_embedding = jnp.zeros(
      (num_species, num_features),
      dtype=dtype,
    )
    self.charge_embedding = ChargeSpinEmbed(
      num_species=num_species,
      num_features=num_features,
      dtype=dtype,
    )
    self.spin_embedding = ChargeSpinEmbed(
      num_species=num_species,
      num_features=num_features,
      dtype=dtype,
    )
    self.layers = [
      SO3LRLayer(
        num_radial_basis_fn=self.num_radial_basis_fn,
        num_features=num_features,
        num_heads=num_heads,
        dtype=dtype,
      )
      for _ in range(num_layers)
    ]
    self.energy_head = EnergyHead(
      num_features=num_features,
      num_species=num_species,
      observable_num_species=observable_num_species,
      hirshfeld_latent_features=hirshfeld_latent_features,
      bohr_angstrom=bohr_angstrom,
      hartree_ev=hartree_ev,
      coulomb_ev_angstrom=coulomb_ev_angstrom,
      dtype=dtype,
    )

  def __call__(
    self,
    positions,
    species,
    *,
    displacement_fn=None,
    neighbors=None,
    neighbors_lr=None,
    total_charge=0.0,
    total_spin=0.0,
  ):
    if displacement_fn is None or neighbors is None or neighbors_lr is None:
      raise ValueError(
        'SO3LR requires a displacement_fn and short- and long-range neighbor '
        'lists. Build them with energy.so3lr_neighbor_list.'
      )
    atomic_numbers = jnp.asarray(species, dtype=jnp.int32)
    featurize = neighbor_list_featurizer(displacement_fn)
    idx_i, idx_j, displacements = featurize(positions, neighbors)
    idx_i_lr, idx_j_lr, displacements_lr = featurize(positions, neighbors_lr)
    total_charge = jnp.asarray(total_charge, dtype=positions.dtype)
    total_spin = jnp.asarray(total_spin, dtype=positions.dtype)

    num_nodes = atomic_numbers.shape[0]

    d_ij = safe_norm(displacements, axis=-1)
    d_ij_lr = safe_norm(displacements_lr, axis=-1)
    rbf_ij = bernstein_basis(d_ij[:, None], n_rbf=self.num_radial_basis_fn)
    cut = phys_cutoff(d_ij, self.cutoff)
    unit_r_ij = safe_mask(
      d_ij[:, None] > 0.0,
      lambda y: y / d_ij[:, None],
      displacements,
      0.0,
    )
    ylm_ij = spherical_harmonics_1_to_4(unit_r_ij)
    ev = jnp.zeros((num_nodes, ylm_ij.shape[-1]), dtype=displacements.dtype)

    embeds = [
      self.feature_embedding[atomic_numbers],
      self.charge_embedding(
        atomic_numbers,
        total_charge,
      ),
      self.spin_embedding(
        atomic_numbers,
        total_spin,
      ),
    ]
    x = jnp.stack(embeds, axis=-1).sum(axis=-1) / jnp.sqrt(float(len(embeds)))

    for layer in self.layers:
      x, ev = layer(
        x,
        ev,
        rbf_ij,
        ylm_ij,
        cut,
        idx_i,
        idx_j,
        avg_num_neighbors=self.avg_num_neighbors,
      )

    atomic_energy = self.energy_head(
      x=x,
      atomic_numbers=atomic_numbers,
      idx_i=idx_i,
      idx_j=idx_j,
      d_ij=d_ij,
      cut=cut,
      idx_i_lr=idx_i_lr,
      idx_j_lr=idx_j_lr,
      d_ij_lr=d_ij_lr,
      total_charge=total_charge,
      cutoff_lr=self.long_range_cutoff,
      dispersion_cutoff_lr_damping=self.dispersion_energy_cutoff_lr_damping,
      electrostatic_energy_scale=self.electrostatic_energy_scale,
      dispersion_energy_scale=self.dispersion_energy_scale,
      fine_structure=self.fine_structure,
      dispersion_alphas=jnp.asarray(
        self.dispersion_alphas, dtype=positions.dtype
      ),
      dispersion_c6=jnp.asarray(self.dispersion_c6, dtype=positions.dtype),
    )
    return jnp.sum(atomic_energy)


def load_model(
  model: str = 'so3lr',
  *,
  model_path: str | PathLike | None = None,
  dtype=None,
):
  path = (
    Path(model_path) if model_path is not None else SO3LR_MODEL_PATHS[model]
  )

  if dtype is None:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  with path.open('rb') as handle:
    header = json.loads(handle.readline().decode('utf-8'))
    metadata = header['metadata']
    hyperparameters = header['hyperparameters']
    template = SO3LR(metadata, hyperparameters, dtype=jnp.float32)
    model = eqx.tree_deserialise_leaves(handle, template)
    return jax.tree_util.tree_map(
      lambda x: (
        x.astype(dtype)
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
        else x
      ),
      model,
    )
