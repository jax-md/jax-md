"""Dense and normalization primitives matching the pretrained checkpoints.

The parameter dtype, the variance form and the order the scale is folded in
all have to match the reference implementation for the checkpoints to
reproduce.
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_md import util as jmd_util

Array = jmd_util.Array


DEFAULT_KERNEL_INIT = jax.nn.initializers.lecun_normal()
DEFAULT_EMBED_INIT = jax.nn.initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)


class Dense(eqx.Module):
  """``x @ kernel + bias`` with float32 parameters.

  ``kernel_init`` defaults to ``lecun_normal``, as ``nn.Dense`` did.
  """

  kernel: Array
  bias: Array | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    key: Array,
    use_bias: bool = True,
    kernel_init: Callable[..., Array] = DEFAULT_KERNEL_INIT,
  ):
    self.kernel = kernel_init(key, (in_features, out_features), jnp.float32)
    self.bias = jnp.zeros((out_features,), jnp.float32) if use_bias else None

  def __call__(self, x: Array) -> Array:
    y = jax.lax.dot_general(x, self.kernel, (((x.ndim - 1,), (0,)), ((), ())))
    return y if self.bias is None else y + self.bias


class LayerNorm(eqx.Module):
  """Layer normalization over the trailing axis."""

  scale: Array
  bias: Array
  eps: float = eqx.field(static=True)

  def __init__(self, num_features: int, *, eps: float = 1e-6):
    self.scale = jnp.ones((num_features,), jnp.float32)
    self.bias = jnp.zeros((num_features,), jnp.float32)
    self.eps = eps

  def __call__(self, x: Array) -> Array:
    x = x.astype(jnp.promote_types(jnp.result_type(x), jnp.float32))
    mu = x.mean(-1, keepdims=True)
    var = jnp.maximum(
      0.0, jnp.square(x).mean(-1, keepdims=True) - jnp.square(mu)
    )
    return (x - mu) * (jax.lax.rsqrt(var + self.eps) * self.scale) + self.bias


class Embed(eqx.Module):
  """Lookup table, holding its weights under the name ``embedding``.

  A child declared as ``sphere_embedding`` thus keeps the checkpoint path
  ``sphere_embedding/embedding``. ``embedding_init`` defaults to the
  initializer ``nn.Embed`` used.
  """

  embedding: Array

  def __init__(
    self,
    num_embeddings: int,
    features: int,
    *,
    key: Array,
    embedding_init: Callable[..., Array] = DEFAULT_EMBED_INIT,
  ):
    self.embedding = embedding_init(
      key, (num_embeddings, features), jnp.float32
    )

  def __call__(self, idx: Array) -> Array:
    return jnp.take(self.embedding, idx, axis=0)
