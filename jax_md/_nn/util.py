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

"""Neural Network Primitives."""

from collections.abc import Mapping
from typing import Union, Dict, Callable, Tuple, Optional

import functools

from jraph import GraphsTuple
import jraph

from jax import vmap
from jax.nn import initializers
import jax.numpy as jnp

from jax_md import util
from jax_md import energy
from jax_md import partition

from ml_collections import ConfigDict

import flax.linen as nn


Array = util.Array
FeaturizerFn = Callable[[GraphsTuple, Array, Array, Array | None], GraphsTuple]

f32 = jnp.float32

partial = functools.partial
normal = lambda var: initializers.variance_scaling(var, 'fan_in', 'normal')


# Nonlinearities:


UnaryFn = Callable[[Array], Array]


class BetaSwish(nn.Module):
  @nn.compact
  def __call__(self, x):
    features = x.shape[-1]
    beta = self.param('Beta', nn.initializers.ones, (features,))
    return x * nn.sigmoid(beta * x)


NONLINEARITY = {
  'none': lambda x: x,
  'relu': nn.relu,
  'swish': lambda x: BetaSwish()(x),
  'raw_swish': nn.swish,
  'tanh': nn.tanh,
  'sigmoid': nn.sigmoid,
  'silu': nn.silu,
}


def get_nonlinearity_by_name(name: str) -> UnaryFn:
  if name in NONLINEARITY:
    return NONLINEARITY[name]
  raise ValueError(f'Nonlinearity "{name}" not found.')


# Fully-Connected Networks


class MLP(nn.Module):
  features: Tuple[int, ...]
  nonlinearity: str

  use_bias: bool = True
  scalar_mlp_std: float | None = None

  @nn.compact
  def __call__(self, x):
    features = self.features

    dense = partial(nn.Dense, use_bias=self.use_bias)
    phi = get_nonlinearity_by_name(self.nonlinearity)

    kernel_init = normal(self.scalar_mlp_std)

    for h in features[:-1]:
      x = phi(dense(h, kernel_init=kernel_init)(x))

    return dense(features[-1], kernel_init=normal(1.0))(x)


def mlp(
  hidden_features: Union[int, Tuple[int, ...]], nonlinearity: str, **kwargs
) -> Callable[..., Array]:
  if isinstance(hidden_features, int):
    hidden_features = (hidden_features,)

  def mlp_fn(*args):
    fn = MLP(hidden_features, nonlinearity, **kwargs)
    return jraph.concatenated_args(fn)(*args)

  return mlp_fn


# Featurizers


def neighbor_list_featurizer(displacement_fn):
  def featurize(atoms, position, neighbor, **kwargs):
    N = position.shape[0]
    graph = partition.to_jraph(neighbor, nodes=atoms)
    mask = partition.neighbor_list_mask(neighbor, True)

    Rb = position[graph.senders]
    Ra = position[graph.receivers]

    d = vmap(partial(displacement_fn, **kwargs))
    dR = d(Ra, Rb)
    dR = jnp.where(mask[:, None], dR, 1)

    return graph._replace(edges=dR)

  return featurize


# Bessel Functions


# Similar to the original Behler-Parinello features. Used by Nequip [1] and
# Schnet [2] to encode distance information.


def bessel(r_c, frequencies, r):
  rp = jnp.where(r > f32(1e-5), r, f32(1000.0))
  b = 2 / r_c * jnp.sin(frequencies * rp / r_c) / rp
  return jnp.where(r > f32(1e-5), b, 0)


class BesselEmbedding(nn.Module):
  count: int
  inner_cutoff: float
  outer_cutoff: float

  @nn.compact
  def __call__(self, rs: Array) -> Array:
    def init_fn(key, shape):
      del key
      assert len(shape) == 1
      n = shape[0]
      return jnp.arange(1, n + 1) * jnp.pi

    frequencies = self.param('frequencies', init_fn, (self.count,))
    bessel_fn = partial(bessel, self.outer_cutoff, frequencies)
    bessel_fn = vmap(
      energy.multiplicative_isotropic_cutoff(
        bessel_fn, self.inner_cutoff, self.outer_cutoff
      )
    )
    return bessel_fn(rs)


# Scale and Shifts

# TODO(schsam): Currently, we refer to a global dataset-level scale / shift
# if no scale / shift is specified by the config. At some point, it would be
# nice to remove this.

DATASET_SHIFT_SCALE = {'harder_silicon': (2.2548, 0.8825)}


def get_shift_and_scale(cfg: ConfigDict) -> Tuple[float, float]:
  if hasattr(cfg, 'scale') and hasattr(cfg, 'shift'):
    return cfg.shift, cfg.scale
  elif hasattr(cfg, 'train_dataset'):
    return DATASET_SHIFT_SCALE[cfg.train_dataset[0]]
  else:
    raise ValueError()


# Checkpoint conversion


_HAIKU_TO_NNX = {
  'linear': 'layers',
  'EdgeFunction': 'edge_fns',
  'NodeFunction': 'node_fns',
  'GlobalFunction': 'global_fns',
}


def _haiku_segment_to_nnx(segment):
  """Map a Haiku-era path segment to the NNX attribute naming convention.

  ``'linear'`` -> ``'layers_0'``,  ``'linear_1'`` -> ``'layers_1'``,
  ``'EdgeFunction'`` -> ``'edge_fns_0'``, ``'EdgeFunction_1'`` -> ``'edge_fns_1'``.
  Segments not in the map are returned as-is.
  """
  for haiku_name, nnx_name in _HAIKU_TO_NNX.items():
    if segment == haiku_name:
      return (f'{nnx_name}_0',)
    if segment.startswith(haiku_name + '_'):
      suffix = segment[len(haiku_name) + 1 :]
      if suffix.isdigit():
        return (f'{nnx_name}_{suffix}',)
  return (segment,)


def _checkpoint_module_name_to_param_path(module_name):
  prefix = 'Energy/~/'
  if not module_name.startswith(prefix):
    raise ValueError(f'Unexpected checkpoint module path: {module_name}')
  segments = module_name[len(prefix) :].replace('/~/', '/').split('/')
  path = ()
  for seg in segments:
    path += _haiku_segment_to_nnx(seg)
  return path


def _flatten_mapping(tree, prefix=()):
  flat = {}
  for key, value in tree.items():
    path = prefix + (key,)
    if isinstance(value, Mapping):
      flat.update(_flatten_mapping(value, path))
    else:
      flat[path] = value
  return flat


def _unflatten_mapping(flat):
  tree = {}
  for path, value in flat.items():
    cursor = tree
    for key in path[:-1]:
      cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value
  return tree


def convert_checkpoint_to_params(checkpoint_params, template_params):
  """Convert a legacy Haiku checkpoint to the current NNX parameter tree.

  Takes the raw ``{module_name: {w, b, ...}}`` dict produced by loading a
  Haiku-era ``si_gnn.pickle`` and remaps it onto the nested dict layout
  expected by the migrated ``graph_network_neighbor_list`` runtime.

  Args:
    checkpoint_params: The raw Haiku checkpoint dict, keyed by module paths
      like ``'Energy/~/GraphNetEncoder/~/...'`` with leaf names ``w`` / ``b``.
    template_params: A reference parameter tree (e.g. from ``init_fn``) whose
      keys, shapes, and dtypes define the target layout.

  Returns:
    A new nested dict with the same structure as *template_params*, populated
    with the checkpoint values (cast to the template dtypes).

  Raises:
    ValueError: If the checkpoint and template have mismatched keys or shapes.
  """
  checkpoint_flat = {}
  for module_name, module_params in checkpoint_params.items():
    base_path = _checkpoint_module_name_to_param_path(module_name)
    for leaf_name, value in module_params.items():
      if leaf_name == 'w':
        mapped_leaf_name = 'kernel'
      elif leaf_name == 'b':
        mapped_leaf_name = 'bias'
      else:
        mapped_leaf_name = leaf_name
      checkpoint_flat[base_path + (mapped_leaf_name,)] = value

  template_flat = _flatten_mapping(template_params)
  missing = sorted(set(template_flat) - set(checkpoint_flat))
  extra = sorted(set(checkpoint_flat) - set(template_flat))
  if missing or extra:
    lines = ['Checkpoint and template parameter tree have different keys.']
    if missing:
      lines.append(f'Missing in checkpoint: {missing[:5]}')
    if extra:
      lines.append(f'Extra in checkpoint: {extra[:5]}')
    raise ValueError('\n'.join(lines))

  converted_flat = {}
  for path, template_leaf in template_flat.items():
    checkpoint_leaf = checkpoint_flat[path]
    if checkpoint_leaf.shape != template_leaf.shape:
      raise ValueError(
        f'Shape mismatch at {path}: '
        f'{checkpoint_leaf.shape} vs {template_leaf.shape}'
      )
    converted_flat[path] = jnp.asarray(
      checkpoint_leaf, dtype=template_leaf.dtype
    )

  return _unflatten_mapping(converted_flat)
