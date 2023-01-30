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
FeaturizerFn = Callable[[GraphsTuple, Array, Array, Optional[Array]], GraphsTuple]

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
  scalar_mlp_std: Optional[float] = None

  @nn.compact
  def __call__(self, x):
    features = self.features

    dense = partial(nn.Dense, use_bias=self.use_bias)
    phi = get_nonlinearity_by_name(self.nonlinearity)

    kernel_init = normal(self.scalar_mlp_std)

    for h in features[:-1]:
      x = phi(dense(h, kernel_init=kernel_init)(x))

    return dense(features[-1], kernel_init=normal(1.0))(x)


def mlp(hidden_features: Union[int, Tuple[int, ...]],
        nonlinearity: str,
        **kwargs) -> Callable[..., Array]:
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
    bessel_fn = vmap(energy.multiplicative_isotropic_cutoff(bessel_fn,
                                                            self.inner_cutoff,
                                                            self.outer_cutoff))
    return bessel_fn(rs)


# Scale and Shifts

# TODO(schsam): Currently, we refer to a global dataset-level scale / shift
# if no scale / shift is specified by the config. At some point, it would be
# nice to remove this.

DATASET_SHIFT_SCALE = {
    'harder_silicon': (2.2548, 0.8825)
}


def get_shift_and_scale(cfg: ConfigDict) -> Tuple[float, float]:
  if hasattr(cfg, 'scale') and hasattr(cfg, 'shift'):
    return cfg.shift, cfg.scale
  elif hasattr(cfg, 'train_dataset'):
    return DATASET_SHIFT_SCALE[cfg.train_dataset[0]]
  else:
    raise ValueError()
