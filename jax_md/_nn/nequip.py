# Copyright 2022 Google LLC
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

"""NequIP interatomic potentials.

Port of mir-group/nequip 0.19. :func:`convert` reads its torch checkpoints;
:func:`from_config` builds an untrained model with the same layout. Parameters
are float32 unless the model is cast; :func:`load_model` restores the saved
dtype and inputs are cast to it.
"""

import functools
import json
import math
from collections.abc import Iterator
from itertools import count
from typing import Any, Callable, Dict, List, Tuple

import e3nn_jax as e3nn
from e3nn_jax import FunctionalLinear
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
from e3nn_jax.legacy import FunctionalTensorProduct
import equinox as eqx
import jax
from jax.random import fold_in
from jax.random import normal
import jax.numpy as jnp
import numpy as onp

from jax_md import util


Irreps = e3nn.Irreps
Array = util.Array


NONLINEARITY: Dict[str, Callable[[Array], Array]] = {
  'silu': jax.nn.silu,
  'tanh': jnp.tanh,
  'abs': jnp.abs,
  'none': lambda x: x,
  'null': lambda x: x,
}


@functools.cache
def clebsch_gordan(l1: int, l2: int, l3: int) -> onp.ndarray:
  """Clebsch-Gordan coefficients as float64 numpy, computed outside jit."""
  with jax.ensure_compile_time_eval():
    return onp.asarray(e3nn.clebsch_gordan(l1, l2, l3))


@functools.cache
def irreps_blocks(irreps: str) -> Tuple[Tuple[int, int], ...]:
  """``(multiplicity, 2l+1)`` per entry of an irreps string."""
  return tuple((mul, ir.dim) for mul, ir in e3nn.Irreps(irreps))


@functools.cache
def irreps_degrees(irreps: str) -> Tuple[int, ...]:
  return tuple(ir.l for _, ir in e3nn.Irreps(irreps))


def split_irreps(x: Array, irreps: str) -> Tuple[Array, ...]:
  """Split flat features into one ``(..., mul, 2l+1)`` block per entry."""
  blocks, offset = [], 0
  for mul, dim in irreps_blocks(irreps):
    width = mul * dim
    blocks.append(
      x[..., offset : offset + width].reshape(x.shape[:-1] + (mul, dim))
    )
    offset += width
  return tuple(blocks)


def join_irreps(blocks, irreps: str, leading) -> Array:
  filled = next((b for b in blocks if b is not None), None)
  dtype = jnp.float32 if filled is None else filled.dtype
  parts = []
  for block, (mul, dim) in zip(blocks, irreps_blocks(irreps)):
    if block is None:
      parts.append(jnp.zeros(leading + (mul * dim,), dtype))
    else:
      parts.append(block.reshape(leading + (mul * dim,)))
  return jnp.concatenate(parts, axis=-1)


class EquivariantLinear(eqx.Module):
  """``o3.Linear``. One weight per instruction, in e3nn-torch's flat order."""

  weights: Tuple[Array, ...]
  irreps_in: str = eqx.field(static=True)
  irreps_out: str = eqx.field(static=True)
  instructions: Tuple[Tuple[int, int, float], ...] = eqx.field(static=True)

  def __call__(self, x: Array) -> Array:
    blocks = split_irreps(x, self.irreps_in)
    out = [None] * len(irreps_blocks(self.irreps_out))
    for weight, (i_in, i_out, path_weight) in zip(
      self.weights, self.instructions
    ):
      term = path_weight * jnp.einsum('...ui,uv->...vi', blocks[i_in], weight)
      out[i_out] = term if out[i_out] is None else out[i_out] + term  # ty: ignore[unsupported-operator]
    return join_irreps(out, self.irreps_out, x.shape[:-1])


class UVUTensorProduct(eqx.Module):
  """``o3.TensorProduct`` in ``uvu`` mode with per-edge weights."""

  irreps_in1: str = eqx.field(static=True)
  irreps_in2: str = eqx.field(static=True)
  irreps_out: str = eqx.field(static=True)
  instructions: Tuple[Tuple[int, int, int, float], ...] = eqx.field(static=True)

  def __call__(self, x1: Array, x2: Array, weight: Array) -> Array:
    blocks1 = split_irreps(x1, self.irreps_in1)
    blocks2 = split_irreps(x2, self.irreps_in2)
    deg1 = irreps_degrees(self.irreps_in1)
    deg2 = irreps_degrees(self.irreps_in2)
    deg_out = irreps_degrees(self.irreps_out)
    muls = irreps_blocks(self.irreps_in1)

    out = [None] * len(irreps_blocks(self.irreps_out))
    offset = 0
    for i_in1, i_in2, i_out, path_weight in self.instructions:
      mul = muls[i_in1][0]
      if weight.shape[-1] < offset + mul:
        raise ValueError(
          f'edge weights are {weight.shape[-1]} wide, need at least '
          f'{offset + mul}'
        )
      w = weight[..., offset : offset + mul]
      offset += mul
      cg = clebsch_gordan(deg1[i_in1], deg2[i_in2], deg_out[i_out])
      term = path_weight * jnp.einsum(
        '...u,ijk,...ui,...j->...uk',
        w,
        cg.astype(x1.dtype),
        blocks1[i_in1],
        blocks2[i_in2][..., 0, :],
      )
      out[i_out] = term if out[i_out] is None else out[i_out] + term  # ty: ignore[unsupported-operator]
    return join_irreps(out, self.irreps_out, x1.shape[:-1])


class FullyConnectedTensorProduct(eqx.Module):
  """``o3.FullyConnectedTensorProduct``, the self-connection."""

  weights: Tuple[Array, ...]
  irreps_in1: str = eqx.field(static=True)
  irreps_in2: str = eqx.field(static=True)
  irreps_out: str = eqx.field(static=True)
  instructions: Tuple[Tuple[int, int, int, float], ...] = eqx.field(static=True)

  def __call__(self, x1: Array, x2: Array) -> Array:
    blocks1 = split_irreps(x1, self.irreps_in1)
    blocks2 = split_irreps(x2, self.irreps_in2)
    deg1 = irreps_degrees(self.irreps_in1)
    deg2 = irreps_degrees(self.irreps_in2)
    deg_out = irreps_degrees(self.irreps_out)

    out = [None] * len(irreps_blocks(self.irreps_out))
    for weight, (i_in1, i_in2, i_out, path_weight) in zip(
      self.weights, self.instructions
    ):
      cg = clebsch_gordan(deg1[i_in1], deg2[i_in2], deg_out[i_out])
      term = path_weight * jnp.einsum(
        'uvw,ijk,...ui,...vj->...wk',
        weight,
        cg.astype(x1.dtype),
        blocks1[i_in1],
        blocks2[i_in2],
      )
      out[i_out] = term if out[i_out] is None else out[i_out] + term  # ty: ignore[unsupported-operator]
    return join_irreps(out, self.irreps_out, x1.shape[:-1])


def fully_connected_tensor_product(
  irreps_out, irreps_in1, irreps_in2, *, keys
) -> FullyConnectedTensorProduct:
  """``o3.FullyConnectedTensorProduct`` with all e3nn paths."""
  irreps_in1 = Irreps(irreps_in1).simplify()
  irreps_in2 = Irreps(irreps_in2).simplify()
  irreps_out = Irreps(irreps_out)

  tp = FunctionalFullyConnectedTensorProduct(
    irreps_in1, irreps_in2, irreps_out.simplify()
  )

  weights, instructions = [], []
  for ins in tp.instructions:
    weights.append(
      ins.weight_std * normal(next(keys), ins.path_shape, jnp.float32)
    )
    instructions.append(
      (ins.i_in1, ins.i_in2, ins.i_out, float(ins.path_weight))
    )

  return FullyConnectedTensorProduct(
    weights=tuple(weights),
    irreps_in1=str(irreps_in1),
    irreps_in2=str(irreps_in2),
    irreps_out=str(irreps_out.simplify()),
    instructions=tuple(instructions),
  )


class ScalarMLP(eqx.Module):
  """``ScalarLinearLayer`` stack, ``x @ (weight * alpha)`` with no bias."""

  weights: Tuple[Array, ...]
  alphas: Tuple[float, ...] = eqx.field(static=True)
  nonlinearity: str = eqx.field(static=True)
  nonlinearity_scale: float = eqx.field(static=True)

  def __call__(self, x: Array) -> Array:
    phi = NONLINEARITY[self.nonlinearity]
    for weight, alpha in zip(self.weights[:-1], self.alphas[:-1]):
      x = self.nonlinearity_scale * phi(x @ (weight * alpha))
    return x @ (self.weights[-1] * self.alphas[-1])


class Gate(eqx.Module):
  """``o3.Gate``. Scalars get an activation, ``l > 0`` blocks get a gate.

  ``input_slices`` gives each declared block's place in the ``_Sortcut``
  order. Empty when the declared order is already sorted.
  """

  irreps_scalars: str = eqx.field(static=True)
  irreps_gates: str = eqx.field(static=True)
  irreps_gated: str = eqx.field(static=True)
  act_scalars: Tuple[str, ...] = eqx.field(static=True)
  act_scalar_scales: Tuple[float, ...] = eqx.field(static=True)
  act_gates: Tuple[str, ...] = eqx.field(static=True)
  act_gate_scales: Tuple[float, ...] = eqx.field(static=True)
  input_slices: Tuple[Tuple[int, int], ...] = eqx.field(static=True, default=())

  def __call__(self, x: Array) -> Array:
    if self.input_slices:
      x = jnp.concatenate([x[..., a:b] for a, b in self.input_slices], axis=-1)
    n_scalars = sum(
      mul * dim for mul, dim in irreps_blocks(self.irreps_scalars)
    )
    n_gates = sum(mul * dim for mul, dim in irreps_blocks(self.irreps_gates))

    scalars = x[..., :n_scalars]
    gates = x[..., n_scalars : n_scalars + n_gates]
    gated = x[..., n_scalars + n_gates :]

    out = []
    offset = 0
    for (mul, dim), act, scale in zip(
      irreps_blocks(self.irreps_scalars),
      self.act_scalars,
      self.act_scalar_scales,
    ):
      width = mul * dim
      out.append(
        scale * NONLINEARITY[act](scalars[..., offset : offset + width])
      )
      offset += width

    if n_gates:
      activated = []
      offset = 0
      for (mul, dim), act, scale in zip(
        irreps_blocks(self.irreps_gates), self.act_gates, self.act_gate_scales
      ):
        width = mul * dim
        activated.append(
          scale * NONLINEARITY[act](gates[..., offset : offset + width])
        )
        offset += width
      activated = jnp.concatenate(activated, axis=-1)

      offset = gate_offset = 0
      for mul, dim in irreps_blocks(self.irreps_gated):
        width = mul * dim
        block = gated[..., offset : offset + width].reshape(
          gated.shape[:-1] + (mul, dim)
        )
        scale = activated[..., gate_offset : gate_offset + mul]
        out.append(
          (block * scale[..., None]).reshape(gated.shape[:-1] + (width,))
        )
        offset += width
        gate_offset += mul

    return jnp.concatenate(out, axis=-1)


class ConvNetLayer(eqx.Module):
  """One interaction layer: linear, convolve, linear, self-connect, gate."""

  linear_1: EquivariantLinear
  tp: UVUTensorProduct
  linear_2: EquivariantLinear
  edge_mlp: ScalarMLP
  gate: Gate
  self_connection: FullyConnectedTensorProduct | None
  scatter_norm_factor: float = eqx.field(static=True)
  resnet: bool = eqx.field(static=True)

  def __call__(
    self,
    x: Array,
    node_attrs: Array,
    edge_attrs: Array,
    edge_embedding: Array,
    senders: Array,
    receivers: Array,
    edge_mask: Array | None,
  ) -> Array:
    sc = (
      None
      if self.self_connection is None
      else self.self_connection(x, node_attrs)
    )

    h = self.scatter_norm_factor * self.linear_1(x)

    weight = self.edge_mlp(edge_embedding)
    if edge_mask is not None:
      weight = jnp.where(edge_mask[:, None], weight, 0.0)

    messages = self.tp(h[senders], edge_attrs, weight)
    aggregated = (
      jnp.zeros((x.shape[0], messages.shape[-1]), messages.dtype)
      .at[receivers]
      .add(messages)
    )

    out = self.linear_2(aggregated)
    if sc is not None:
      out = out + sc
    out = self.gate(out)

    return out + x if self.resnet else out


class NequIP(eqx.Module):
  """NequIP energy model. Returns per-atom energies.

  ``senders`` is upstream's ``edge_index[1]`` and ``receivers`` its
  ``edge_index[0]``. Edge vectors are cast to the parameter dtype.
  """

  bessel_weights: Array
  type_embedding: Array
  layers: Tuple[ConvNetLayer, ...]
  readout: ScalarMLP
  scales: Array
  shifts: Array
  zbl_atomic_numbers: Tuple[float, ...] | None = eqx.field(static=True)

  r_max: float = eqx.field(static=True)
  cutoff_p: float = eqx.field(static=True)
  bessel_factor: float = eqx.field(static=True)
  irreps_edge_sh: str = eqx.field(static=True)
  type_names: Tuple[str, ...] = eqx.field(static=True)
  zbl_prefactor: float = eqx.field(static=True)
  zbl_cutoff_p: float = eqx.field(static=True)

  def __call__(
    self,
    position: Array,
    species: Array,
    senders: Array,
    receivers: Array,
    edge_mask: Array | None = None,
    edge_vector: Array | None = None,
  ) -> Array:
    if edge_vector is None:
      edge_vector = position[senders] - position[receivers]
    edge_vector = edge_vector.astype(self.type_embedding.dtype)

    if edge_mask is not None:
      # padded edges get r = r_max so bessel and ZBL stay finite
      pad = jnp.array([self.r_max, 0.0, 0.0], edge_vector.dtype)
      edge_vector = jnp.where(edge_mask[:, None], edge_vector, pad)

    lengths = jnp.linalg.norm(edge_vector, axis=-1, keepdims=True)
    normed = lengths / self.r_max
    edge_embedding = self.radial_embedding(lengths)

    edge_attrs = e3nn.spherical_harmonics(
      self.irreps_edge_sh,  # ty: ignore[invalid-argument-type]
      edge_vector,
      normalize=True,
      normalization='component',
    ).array

    node_attrs = self.type_embedding[species]
    x = node_attrs
    for layer in self.layers:
      x = layer(
        x,
        node_attrs,
        edge_attrs,
        edge_embedding,
        senders,
        receivers,
        edge_mask,
      )

    energy = self.readout(x)[:, 0]
    energy = self.shifts[species, 0] + self.scales[species, 0] * energy

    if self.zbl_atomic_numbers is not None:
      energy = energy + self.zbl_energy(
        lengths[:, 0], species, senders, receivers, normed[:, 0], edge_mask
      )
    return energy

  def radial_embedding(self, lengths: Array) -> Array:
    """Bessel basis under the polynomial cutoff, one row per edge."""
    normed = lengths / self.r_max
    bessel = jnp.sinc(normed * self.bessel_weights) * self.bessel_weights
    cutoff = polynomial_cutoff(normed, self.cutoff_p)
    return bessel * cutoff * self.bessel_factor

  def zbl_energy(
    self, r, species, senders, receivers, normed, edge_mask
  ) -> Array:
    # LAMMPS pair_zbl constants
    p_zbl, a0 = 0.23, 0.46850
    c = (0.02817, 0.28022, 0.50986, 0.18175)
    d = (-0.20162, -0.40290, -0.94229, -3.19980)

    z = jnp.asarray(self.zbl_atomic_numbers, r.dtype)[species]
    z_i, z_j = z[receivers], z[senders]
    x = ((z_i**p_zbl + z_j**p_zbl) * r) / a0
    psi = sum(ci * jnp.exp(di * x) for ci, di in zip(c, d))
    pair = self.zbl_prefactor * ((z_i * z_j) / r) * psi
    pair = pair * polynomial_cutoff(normed, self.zbl_cutoff_p)
    if edge_mask is not None:
      pair = jnp.where(edge_mask, pair, 0.0)
    return (
      jnp.zeros_like(psi, shape=(species.shape[0],)).at[receivers].add(pair)
    )


def polynomial_cutoff(x: Array, p: float) -> Array:
  """DimeNet polynomial envelope, zero at and beyond the cutoff."""
  out = 1.0
  out = out - ((p + 1.0) * (p + 2.0) / 2.0) * x**p
  out = out + p * (p + 2.0) * x ** (p + 1.0)
  out = out - (p * (p + 1.0) / 2.0) * x ** (p + 2.0)
  return out * (x < 1.0)


# e3nn wraps activations in ``normalize2mom``; probe to get name and scale
ACTIVATION_PROBES = {
  'silu': lambda x: x / (1.0 + onp.exp(-x)),
  'tanh': onp.tanh,
  'abs': onp.abs,
  'none': lambda x: x,
}


def activation_of(fn):
  """``(name, scale)`` such that ``fn(x) == scale * NONLINEARITY[name](x)``."""
  import torch

  probe = torch.linspace(-3.0, 3.0, 65, dtype=torch.float64)
  got = fn(probe).detach().cpu().numpy()
  x = probe.numpy()
  for name, reference in ACTIVATION_PROBES.items():
    ref = reference(x)
    denominator = float(ref @ ref)
    if denominator == 0.0:
      continue
    scale = float(got @ ref) / denominator
    if onp.abs(got - scale * ref).max() < 1e-9 * max(1.0, onp.abs(got).max()):
      return name, scale
  raise ValueError(f'unrecognized activation {fn}')


def slice_flat_weight(weight, instructions) -> List[onp.ndarray]:
  """Split e3nn-torch's flat weight into one array per instruction."""
  flat = weight.detach().cpu().numpy()
  out, offset = [], 0
  for ins in instructions:
    size = int(onp.prod(ins.path_shape))
    out.append(flat[offset : offset + size].reshape(ins.path_shape).copy())
    offset += size
  if offset != flat.size:
    raise ValueError(f'consumed {offset} of {flat.size} weights')
  return out


def linear_spec(module) -> Dict[str, Any]:
  if getattr(module, 'bias', None) is not None and module.bias.numel():
    raise ValueError('biased o3.Linear is not supported')
  return {
    'irreps_in': str(module.irreps_in),
    'irreps_out': str(module.irreps_out),
    'instructions': [
      [int(i.i_in), int(i.i_out), float(i.path_weight)]
      for i in module.instructions
    ],
    'weights': slice_flat_weight(module.weight, module.instructions),
  }


def uvu_tp_spec(module) -> Dict[str, Any]:
  for ins in module.instructions:
    if ins.connection_mode != 'uvu' or not ins.has_weight:
      raise ValueError(f'unsupported tp instruction {ins.connection_mode}')
  if any(mul != 1 for mul, _ in module.irreps_in2):
    raise ValueError('edge harmonics with multiplicity > 1 are not supported')
  return {
    'irreps_in1': str(module.irreps_in1),
    'irreps_in2': str(module.irreps_in2),
    'irreps_out': str(module.irreps_out),
    'instructions': [
      [int(i.i_in1), int(i.i_in2), int(i.i_out), float(i.path_weight)]
      for i in module.instructions
    ],
  }


def fctp_spec(module) -> Dict[str, Any]:
  for ins in module.instructions:
    if ins.connection_mode != 'uvw' or not ins.has_weight:
      raise ValueError(f'unsupported sc instruction {ins.connection_mode}')
  return {
    'irreps_in1': str(module.irreps_in1),
    'irreps_in2': str(module.irreps_in2),
    'irreps_out': str(module.irreps_out),
    'instructions': [
      [int(i.i_in1), int(i.i_in2), int(i.i_out), float(i.path_weight)]
      for i in module.instructions
    ],
    'weights': slice_flat_weight(module.weight, module.instructions),
  }


def scalar_mlp_spec(module) -> Dict[str, Any]:
  if hasattr(module.mlp, 'weights'):
    weights = [w.detach().cpu().numpy().copy() for w in module.mlp.weights]
    return {
      'alphas': [float(module.mlp.alpha)] + [1.0] * (len(weights) - 1),
      'nonlinearity': 'none',
      'nonlinearity_scale': 1.0,
      'weights': weights,
    }
  layers = [m for m in module.mlp if hasattr(m, 'weight')]
  acts = {activation_of(m) for m in module.mlp if not hasattr(m, 'weight')}
  if len(acts) > 1:
    raise ValueError(f'mixed activations in a scalar MLP: {acts}')
  for layer in layers:
    if getattr(layer, 'bias', None) is not None:
      raise ValueError('biased ScalarLinearLayer is not supported')
  name, scale = acts.pop() if acts else ('none', 1.0)
  return {
    'alphas': [float(m.alpha) for m in layers],
    'nonlinearity': name,
    'nonlinearity_scale': scale,
    'weights': [m.weight.detach().cpu().numpy().copy() for m in layers],
  }


def activations_of(activation) -> Tuple[str, List[Tuple[str, float]]]:
  """Blocks of an ``e3nn.nn.Activation`` with one ``(name, scale)`` each."""
  acts = [
    ('none', 1.0) if act is None else activation_of(act)
    for act in activation.acts
  ]
  return str(activation.irreps_in), acts


def gate_spec(module) -> Dict[str, Any]:
  if type(module).__name__ != 'Gate':
    raise ValueError(
      f'only o3.Gate is supported, found {type(module).__name__}'
    )
  irreps_scalars, scalars = activations_of(module.act_scalars)
  irreps_gates, gates = activations_of(module.act_gates)
  declared = module.irreps_scalars + module.irreps_gates + module.irreps_gated
  order = declared.sort().inv
  starts, offset = [0] * len(declared), 0
  for position in order:
    starts[position] = offset
    offset += declared[position].dim
  slices = [(s, s + mul_ir.dim) for s, mul_ir in zip(starts, declared)]
  return {
    'irreps_scalars': irreps_scalars,
    'irreps_gates': irreps_gates,
    'irreps_gated': str(module.irreps_gated),
    'input_slices': [] if slices == sorted(slices) else slices,
    'act_scalars': [n for n, _ in scalars],
    'act_scalar_scales': [s for _, s in scalars],
    'act_gates': [n for n, _ in gates],
    'act_gate_scales': [s for _, s in gates],
  }


def tensor_product_of(tp_scatter):
  """The e3nn tensor product inside a ``TensorProductScatter``.

  Also present with OpenEquivariance or CuEquivariance enabled.
  """
  tp = getattr(tp_scatter, 'tp', None)
  if tp is None or not hasattr(tp, 'instructions'):
    raise ValueError(
      f'{type(tp_scatter).__name__} has no e3nn tensor product. Rebuild the '
      'model without the acceleration modifier.'
    )
  return tp


def convert(torch_model) -> Dict[str, Any]:
  """Turn a loaded nequip model into a dict of specs and numpy arrays."""
  sequential = torch_model.model.func
  blocks = dict(sequential.named_children())

  conv_names = sorted(
    (n for n in blocks if n.endswith('_convnet')),
    key=lambda n: int(n[len('layer') : -len('_convnet')]),
  )
  if not conv_names:
    raise ValueError('no convnet layers found')
  if blocks['edge_norm']._per_edge_type:
    raise ValueError('per-edge-type cutoffs are not supported')

  layers = []
  for name in conv_names:
    convnet = blocks[name]
    conv = convnet.conv
    norm = getattr(conv, 'avg_num_neighbors_norm', None)
    if norm is not None and norm.norm_const.numel() != 1:
      raise ValueError('per-type avg_num_neighbors is not supported')
    factor = conv.scatter_norm_factor if norm is None else norm.norm_const
    layers.append(
      {
        'linear_1': linear_spec(conv.linear_1),
        'tp': uvu_tp_spec(tensor_product_of(conv.tp_scatter)),
        'linear_2': linear_spec(conv.linear_2),
        'edge_mlp': scalar_mlp_spec(conv.edge_mlp),
        'gate': gate_spec(convnet.equivariant_nonlin),
        'self_connection': None if conv.sc is None else fctp_spec(conv.sc),
        'scatter_norm_factor': 1.0 if factor is None else float(factor),
        'resnet': bool(convnet.resnet),
      }
    )

  bessel = blocks['bessel_encode']
  type_names = str(torch_model.metadata['type_names']).split()
  scale_shift = blocks['per_type_energy_scale_shift']

  def per_type(name, default):
    value = getattr(scale_shift, name, None)
    if value is None or value.numel() == 0:
      return onp.full((len(type_names), 1), default, onp.float32)
    value = value.detach().cpu().numpy().reshape(-1, 1)
    return onp.broadcast_to(value, (len(type_names), 1)).copy()

  zbl = blocks.get('pair_potential')
  if zbl is not None and type(zbl).__name__ != 'ZBL':
    raise ValueError(f'unsupported pair potential {type(zbl).__name__}')

  return {
    'r_max': float(torch_model.metadata['r_max']),
    'type_names': type_names,
    'cutoff_p': float(bessel.cutoff.p),
    'bessel_factor': float(blocks['factor'].factor),
    'irreps_edge_sh': str(blocks['spharm'].irreps_out['edge_attrs']),
    'bessel_weights': bessel.bessel_weights.detach().cpu().numpy().copy(),
    'type_embedding': blocks['type_embed']
    .embed_module.weight.detach()
    .cpu()
    .numpy()
    .copy(),
    'layers': layers,
    'readout': scalar_mlp_spec(blocks['per_atom_energy_readout'].mlp_module),
    'scales': per_type('scales', 1.0),
    'shifts': per_type('shifts', 0.0),
    'zbl': None
    if zbl is None
    else {
      'atomic_numbers': [float(z) for z in zbl.atomic_numbers.tolist()],
      'prefactor': float(zbl._qqr2exesquare),
      # packaged models have no cutoff attribute; use the bessel exponent
      'cutoff_p': float(getattr(zbl, 'cutoff', bessel.cutoff).p),
    },
  }


def strip_arrays(node):
  """Replace every array in a description with its shape."""
  if isinstance(node, onp.ndarray):
    return {'__shape__': list(node.shape)}
  if isinstance(node, dict):
    return {k: strip_arrays(v) for k, v in node.items()}
  if isinstance(node, list):
    return [strip_arrays(v) for v in node]
  return node


def zero_arrays(node):
  """Inverse of :func:`strip_arrays`; shapes become float32 zeros."""
  if isinstance(node, dict):
    if '__shape__' in node:
      return onp.zeros(node['__shape__'], onp.float32)
    return {k: zero_arrays(v) for k, v in node.items()}
  if isinstance(node, list):
    return [zero_arrays(v) for v in node]
  return node


def save_model(path: str, model: NequIP, config: Dict[str, Any]) -> None:
  """Write *model* as a JSON header followed by its leaves.

  ``config`` is either the :func:`from_config` keyword arguments or the
  :func:`convert` output; the header says which.
  """
  header: Dict[str, Any] = (
    {'description': strip_arrays(config)}
    if 'layers' in config
    else {'from_config': config}
  )
  header['dtype'] = str(model.type_embedding.dtype)
  line = (json.dumps(header) + '\n').encode()
  with open(path, 'wb') as f:
    f.write(line)
    eqx.tree_serialise_leaves(f, model)


def load_model(path: str) -> Tuple[NequIP, Dict[str, Any]]:
  """Read a file written by :func:`save_model`. Needs no torch."""
  with open(path, 'rb') as f:
    header = json.loads(f.readline().decode())
    if 'from_config' in header:
      config = header['from_config']
      template = from_config(jax.random.PRNGKey(0), **config)
    else:
      config = header['description']
      template = build(zero_arrays(config))
    dtype = header.get('dtype', 'float32')
    template = jax.tree.map(lambda x: x.astype(dtype), template)
    return eqx.tree_deserialise_leaves(f, template), config


def zbl_numbers(zbl: Dict[str, Any]) -> Tuple[float, ...]:
  """Atomic numbers from a :func:`convert` ZBL block, as floats."""
  numbers = tuple(float(v) for v in zbl['atomic_numbers'])
  if not numbers or min(numbers) < 1.0:
    raise ValueError(f'ZBL atomic numbers must be a list of Z >= 1: {numbers}')
  return numbers


def build(description: Dict[str, Any]) -> NequIP:
  """Build the model from :func:`convert` output."""
  layers = []
  for spec in description['layers']:
    sc = spec['self_connection']
    layers.append(
      ConvNetLayer(
        linear_1=EquivariantLinear(
          weights=tuple(
            jnp.asarray(w, jnp.float32) for w in spec['linear_1']['weights']
          ),
          irreps_in=spec['linear_1']['irreps_in'],
          irreps_out=spec['linear_1']['irreps_out'],
          instructions=tuple(
            (int(a), int(b), float(c))
            for a, b, c in spec['linear_1']['instructions']
          ),
        ),
        tp=UVUTensorProduct(
          irreps_in1=spec['tp']['irreps_in1'],
          irreps_in2=spec['tp']['irreps_in2'],
          irreps_out=spec['tp']['irreps_out'],
          instructions=tuple(
            (int(a), int(b), int(c), float(d))
            for a, b, c, d in spec['tp']['instructions']
          ),
        ),
        linear_2=EquivariantLinear(
          weights=tuple(
            jnp.asarray(w, jnp.float32) for w in spec['linear_2']['weights']
          ),
          irreps_in=spec['linear_2']['irreps_in'],
          irreps_out=spec['linear_2']['irreps_out'],
          instructions=tuple(
            (int(a), int(b), float(c))
            for a, b, c in spec['linear_2']['instructions']
          ),
        ),
        edge_mlp=ScalarMLP(
          weights=tuple(
            jnp.asarray(w, jnp.float32) for w in spec['edge_mlp']['weights']
          ),
          alphas=tuple(float(a) for a in spec['edge_mlp']['alphas']),
          nonlinearity=spec['edge_mlp']['nonlinearity'],
          nonlinearity_scale=float(spec['edge_mlp']['nonlinearity_scale']),
        ),
        gate=Gate(
          irreps_scalars=spec['gate']['irreps_scalars'],
          irreps_gates=spec['gate']['irreps_gates'],
          irreps_gated=spec['gate']['irreps_gated'],
          act_scalars=tuple(spec['gate']['act_scalars']),
          act_scalar_scales=tuple(
            float(v) for v in spec['gate']['act_scalar_scales']
          ),
          act_gates=tuple(spec['gate']['act_gates']),
          act_gate_scales=tuple(
            float(v) for v in spec['gate']['act_gate_scales']
          ),
          input_slices=tuple(
            (int(a), int(b)) for a, b in spec['gate'].get('input_slices', ())
          ),
        ),
        self_connection=None
        if sc is None
        else FullyConnectedTensorProduct(
          weights=tuple(jnp.asarray(w, jnp.float32) for w in sc['weights']),
          irreps_in1=sc['irreps_in1'],
          irreps_in2=sc['irreps_in2'],
          irreps_out=sc['irreps_out'],
          instructions=tuple(
            (int(a), int(b), int(c), float(d))
            for a, b, c, d in sc['instructions']
          ),
        ),
        scatter_norm_factor=float(spec['scatter_norm_factor']),
        resnet=bool(spec['resnet']),
      )
    )

  zbl = description['zbl']
  return NequIP(
    bessel_weights=jnp.asarray(description['bessel_weights'], jnp.float32),
    type_embedding=jnp.asarray(description['type_embedding'], jnp.float32),
    layers=tuple(layers),
    readout=ScalarMLP(
      weights=tuple(
        jnp.asarray(w, jnp.float32) for w in description['readout']['weights']
      ),
      alphas=tuple(float(a) for a in description['readout']['alphas']),
      nonlinearity=description['readout']['nonlinearity'],
      nonlinearity_scale=float(description['readout']['nonlinearity_scale']),
    ),
    scales=jnp.asarray(description['scales'], jnp.float32),
    shifts=jnp.asarray(description['shifts'], jnp.float32),
    zbl_atomic_numbers=None if zbl is None else zbl_numbers(zbl),
    r_max=float(description['r_max']),
    cutoff_p=float(description['cutoff_p']),
    bessel_factor=float(description['bessel_factor']),
    irreps_edge_sh=description['irreps_edge_sh'],
    type_names=tuple(description['type_names']),
    zbl_prefactor=0.0 if zbl is None else float(zbl['prefactor']),
    zbl_cutoff_p=0.0 if zbl is None else float(zbl['cutoff_p']),
  )


def equivariant_linear(irreps_in, irreps_out, *, keys) -> EquivariantLinear:
  """``o3.Linear`` with all e3nn paths between the two irreps."""
  irreps_in, irreps_out = Irreps(irreps_in), Irreps(irreps_out)
  weights, instructions = [], []
  for ins in FunctionalLinear(irreps_in, irreps_out).instructions:
    weights.append(
      ins.weight_std * normal(next(keys), ins.path_shape, jnp.float32)
    )
    instructions.append((ins.i_in, ins.i_out, float(ins.path_weight)))
  return EquivariantLinear(
    weights=tuple(weights),
    irreps_in=str(irreps_in),
    irreps_out=str(irreps_out),
    instructions=tuple(instructions),
  )


def uvu_tensor_product(irreps_in1, irreps_in2, keep) -> UVUTensorProduct:
  """All ``uvu`` paths into *keep*. Outputs grouped by irrep, unsimplified."""
  irreps_in1, irreps_in2 = Irreps(irreps_in1), Irreps(irreps_in2)
  if any(mul != 1 for mul, _ in irreps_in2):
    raise ValueError('edge harmonics with multiplicity > 1 are not supported')
  keep = {ir for _, ir in Irreps(keep)}

  paths = []
  for i_1, (mul, ir_1) in enumerate(irreps_in1):
    for i_2, (_, ir_2) in enumerate(irreps_in2):
      for ir_out in ir_1 * ir_2:
        if ir_out in keep:
          paths.append((i_1, i_2, mul, ir_out))
  # instructions stay in generation order; the edge MLP is sliced that way
  order = sorted(
    range(len(paths)), key=lambda k: (paths[k][3].l, paths[k][3].p)
  )
  position = {k: i_out for i_out, k in enumerate(order)}

  irreps_out = Irreps([(paths[k][2], paths[k][3]) for k in order])
  instructions = [
    (i_1, i_2, position[k], 'uvu', True)
    for k, (i_1, i_2, _, _) in enumerate(paths)
  ]
  tp = FunctionalTensorProduct(irreps_in1, irreps_in2, irreps_out, instructions)  # ty: ignore[invalid-argument-type]
  return UVUTensorProduct(
    irreps_in1=str(irreps_in1),
    irreps_in2=str(irreps_in2),
    irreps_out=str(irreps_out),
    instructions=tuple(
      (ins.i_in1, ins.i_in2, ins.i_out, float(ins.path_weight))
      for ins in tp.instructions
    ),
  )


def scalar_mlp(dims, nonlinearity, *, keys) -> ScalarMLP:
  """MLP without bias. ``alpha`` is ``1/sqrt(fan_in)``, Kaiming after an act."""
  weights, alphas = [], []
  for i, (fan_in, fan_out) in enumerate(zip(dims, dims[1:])):
    weights.append(normal(next(keys), (fan_in, fan_out), jnp.float32))
    alphas.append(math.sqrt(2.0 / fan_in) if i else 1.0 / math.sqrt(fan_in))
  name, scale = ('none', 1.0) if len(dims) == 2 else (nonlinearity, 1.0)
  return ScalarMLP(
    weights=tuple(weights),
    alphas=tuple(alphas),
    nonlinearity=name,
    nonlinearity_scale=scale,
  )


def reachable(irreps_in, edge_sh, irreps) -> Irreps:
  """Blocks of ``irreps`` reachable from ``irreps_in ⊗ edge_sh``."""
  irreps_in, edge_sh = Irreps(irreps_in), Irreps(edge_sh)
  return Irreps(
    [
      (mul, ir)
      for mul, ir in Irreps(irreps)
      if any(
        ir in list(ir1 * ir2) for _, ir1 in irreps_in for _, ir2 in edge_sh
      )
    ]
  )


def gate_irreps(hidden_irreps) -> Tuple[str, str, str]:
  """``(scalars, gates, gated)`` of hidden irreps, a ``0e`` gate per channel."""
  hidden_irreps = Irreps(hidden_irreps)
  scalars = Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l == 0])
  gated = Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l > 0])
  gates = Irreps([(mul, '0e') for mul, _ in gated])  # ty: ignore[invalid-argument-type]
  # e3nn prints empty irreps as 'Irreps()', which it cannot parse back
  return (
    str(scalars) if scalars else '',
    str(gates) if gates else '',
    str(gated) if gated else '',
  )


ELEMENTS = (
  'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni '
  'Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I '
  'Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt '
  'Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr '
  'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'
).split()

# LAMMPS qqr2e in metal units, halved because every pair is visited twice
ZBL_PREFACTOR = 14.399645 / 2


def atomic_numbers(type_names) -> Tuple[float, ...]:
  """Atomic numbers of chemical symbols, as floats."""
  unknown = [name for name in type_names if name not in ELEMENTS]
  if unknown:
    raise ValueError(f'ZBL needs chemical symbols as type names, got {unknown}')
  return tuple(float(ELEMENTS.index(name) + 1) for name in type_names)


@functools.cache
def second_moment_scale(nonlinearity: str) -> float:
  """e3nn's ``normalize2mom`` constant for an activation, by quadrature."""
  nodes, weights = onp.polynomial.hermite_e.hermegauss(100)
  with jax.ensure_compile_time_eval():
    values = onp.asarray(NONLINEARITY[nonlinearity](jnp.asarray(nodes)))
  return float((weights @ values**2 / weights.sum()) ** -0.5)


def from_config(
  key,
  type_names: Tuple[str, ...] = ('H', 'C', 'O'),
  *,
  hidden_irreps: str = '32x0e + 16x1o',
  irreps_edge_sh: str = '1x0e + 1x1o',
  num_layers: int = 3,
  num_type_features: int = 32,
  num_bessel: int = 8,
  r_max: float = 5.0,
  cutoff_p: float = 6.0,
  radial_mlp_width: int = 64,
  avg_num_neighbors: float = 1.0,
  nonlinearity: str = 'silu',
  self_connection: bool = True,
  resnet: bool = False,
  shift: float = 0.0,
  scale: float = 1.0,
  zbl: bool = False,
) -> NequIP:
  """Build an untrained :class:`NequIP` with the layout of a converted one.

  Args:
    key: PRNG key.
    type_names: Chemical species; the model's `species` argument indexes them.
    hidden_irreps: Node feature irreps.
    irreps_edge_sh: Edge spherical harmonics irreps.
    num_layers: Number of interaction layers.
    num_type_features: Width of the species embedding.
    num_bessel: Number of radial basis functions.
    r_max: Cutoff radius.
    cutoff_p: Exponent of the polynomial cutoff envelope.
    radial_mlp_width: Hidden width of the radial MLP.
    avg_num_neighbors: Mean neighbor count; the scatter is divided by its
      square root.
    nonlinearity: Gate and radial MLP activation.
    self_connection: Self-connection in every layer after the first.
    resnet: Residual connection in every layer after the first, wherever the
      irreps match.
    shift: Initial per-type energy shift; trains with the network.
    scale: Initial per-type energy scale; trains with the network.
    zbl: Add the ZBL pair repulsion. Needs chemical symbols as type names;
      the atomic numbers are fixed.
  """
  keys = (
    key if isinstance(key, Iterator) else (fold_in(key, i) for i in count())
  )
  type_names = tuple(type_names)
  hidden = Irreps(hidden_irreps)
  edge_sh = Irreps(irreps_edge_sh)

  layers = []
  irreps_in = Irreps(f'{num_type_features}x0e')
  for layer in range(num_layers):
    # upstream drops blocks no path reaches from the previous layer
    scalars, gates, gated = gate_irreps(reachable(irreps_in, edge_sh, hidden))
    if layer == num_layers - 1:
      # the readout is a plain MLP, so only even scalars may feed it
      even = [(mul, ir) for mul, ir in Irreps(scalars) if ir.p == 1]
      if not even:
        raise ValueError(
          f'hidden_irreps {hidden_irreps!r} leave no even scalars for the '
          'readout.'
        )
      scalars, gates, gated = str(Irreps(even)), '', ''  # ty: ignore[invalid-argument-type]
    # even scalars take the nonlinearity, odd ones tanh
    acts = tuple(
      nonlinearity if ir.p == 1 else 'tanh' for _, ir in Irreps(scalars)
    )
    act_scales = tuple(second_moment_scale(act) for act in acts)
    num_gate_blocks = len(Irreps(gates))
    target = Irreps(scalars) + Irreps(gated)

    linear_1 = equivariant_linear(irreps_in, irreps_in, keys=keys)
    tp = uvu_tensor_product(irreps_in, edge_sh, target)
    after_tp = Irreps(tp.irreps_out).simplify()

    gate = Gate(
      irreps_scalars=scalars,
      irreps_gates=gates,
      irreps_gated=gated,
      act_scalars=acts,
      act_scalar_scales=act_scales,
      act_gates=(nonlinearity,) * num_gate_blocks,
      act_gate_scales=(second_moment_scale(nonlinearity),) * num_gate_blocks,
    )
    # adjacent 0e blocks merge; the gate splits them by multiplicity
    linear_2_out = (
      (Irreps(scalars) + Irreps(gates)).simplify() if gates else Irreps(scalars)
    ) + Irreps(gated)

    weight_numel = sum(
      irreps_blocks(tp.irreps_in1)[i_1][0] for i_1, _, _, _ in tp.instructions
    )
    sc = (
      fully_connected_tensor_product(
        linear_2_out, irreps_in, f'{num_type_features}x0e', keys=keys
      )
      if self_connection and layer > 0
      else None
    )

    layers.append(
      ConvNetLayer(
        linear_1=linear_1,
        tp=tp,
        linear_2=equivariant_linear(after_tp, linear_2_out, keys=keys),
        edge_mlp=scalar_mlp(
          (num_bessel, radial_mlp_width, weight_numel),
          nonlinearity,
          keys=keys,
        ),
        gate=gate,
        self_connection=sc,
        scatter_norm_factor=1.0 / math.sqrt(avg_num_neighbors),
        resnet=bool(resnet) and layer > 0 and irreps_in == target,
      )
    )
    irreps_in = target

  return NequIP(
    bessel_weights=jnp.arange(1, num_bessel + 1, dtype=jnp.float32)[None],
    type_embedding=normal(
      next(keys), (len(type_names), num_type_features), jnp.float32
    ),
    layers=tuple(layers),
    readout=scalar_mlp((Irreps(scalars).dim, 1), nonlinearity, keys=keys),
    scales=jnp.full((len(type_names), 1), float(scale), jnp.float32),
    shifts=jnp.full((len(type_names), 1), float(shift), jnp.float32),
    zbl_atomic_numbers=atomic_numbers(type_names) if zbl else None,
    r_max=float(r_max),
    cutoff_p=float(cutoff_p),
    bessel_factor=2.0 * math.pi / float(r_max) ** 2,
    irreps_edge_sh=str(edge_sh),
    type_names=type_names,
    zbl_prefactor=ZBL_PREFACTOR if zbl else 0.0,
    zbl_cutoff_p=float(cutoff_p) if zbl else 0.0,
  )
