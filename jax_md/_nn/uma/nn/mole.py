"""Mixture-of-Linear-Experts (MOLE) layers for UMA MoE backbone.

The MOLE approach mixes expert weight matrices per-system:
  1. Routing MLP: system features -> mixing coefficients [num_systems, num_experts]
  2. Weight mixing: einsum('eoi,be->boi', expert_weights, coefficients)
  3. Application: each atom uses its system's mixed weight matrix
"""

import math
from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_md import util as jmd_util
from jax_md._nn.uma.kernels import segment_mm
from jax_md._nn.uma.nn import primitives

Array = jmd_util.Array


class MOLELinear(eqx.Module):
  """Mixture-of-Linear-Experts layer.

  Holds [num_experts, out_features, in_features] weight tensor.
  At forward time, mixes experts using per-system coefficients,
  then applies the mixed weight to each atom/edge.

  Attributes:
      weights: Expert weight tensor.
      bias: Shared bias term, or None.
      out_features: Output dimension.
  """

  weights: Array
  bias: Array | None
  out_features: int = eqx.field(static=True)
  merged: bool = eqx.field(static=True)
  assume_equal_contiguous_batches: bool = eqx.field(static=True)
  use_segment_mm_pallas: bool = eqx.field(static=True)
  max_segment_size: int | None = eqx.field(static=True)
  assume_segment_contiguous_batches: bool = eqx.field(static=True)

  def __init__(
    self,
    num_experts: int,
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    merged: bool = False,
    assume_equal_contiguous_batches: bool = False,
    use_segment_mm_pallas: bool = False,
    max_segment_size: int | None = None,
    assume_segment_contiguous_batches: bool = False,
    *,
    key: Array,
  ):
    self.out_features = out_features
    self.merged = merged
    self.assume_equal_contiguous_batches = assume_equal_contiguous_batches
    self.use_segment_mm_pallas = use_segment_mm_pallas
    self.max_segment_size = max_segment_size
    self.assume_segment_contiguous_batches = assume_segment_contiguous_batches

    bound = math.sqrt(1.0 / in_features)
    weight_shape = (
      (out_features, in_features)
      if merged
      else (num_experts, out_features, in_features)
    )
    weight_key, bias_key = jax.random.split(key)
    self.weights = jax.random.uniform(
      weight_key, weight_shape, minval=-bound, maxval=bound
    )
    self.bias = (
      jax.random.uniform(bias_key, (out_features,), minval=-bound, maxval=bound)
      if use_bias
      else None
    )

  def __call__(
    self,
    x: Array,
    expert_coefficients: Array,
    batch_indices: Array,
    mixed_weights: Array | None = None,
  ) -> Array:
    """Apply MOLE linear layer.

    Args:
        x: Input tensor, shape [N, ..., in_features].
        expert_coefficients: Mixing coefficients, shape [num_systems, num_experts].
        batch_indices: System index for each atom/edge, shape [N].
        mixed_weights: Per-system weights, shape [num_systems, out, in]. When
            given they replace the expert mixing of ``weights``.

    Returns:
        Output tensor, shape [N, ..., out_features].
    """
    weights = self.weights

    if self.merged:
      if x.ndim == 2:
        out = jnp.einsum('ni,oi->no', x, weights)
      elif x.ndim == 3:
        out = jnp.einsum('nci,oi->nco', x, weights)
      else:
        raise ValueError(f'MOLELinear: unsupported input ndim={x.ndim}')
    elif expert_coefficients.shape[0] == 1 and mixed_weights is None:
      mixed_weight = jnp.einsum('e,eoi->oi', expert_coefficients[0], weights)
      if x.ndim == 2:
        out = jnp.einsum('ni,oi->no', x, mixed_weight)
      elif x.ndim == 3:
        out = jnp.einsum('nci,oi->nco', x, mixed_weight)
      else:
        raise ValueError(f'MOLELinear: unsupported input ndim={x.ndim}')
    else:
      # Mix experts per system: [E, O, I] x [B, E] -> [B, O, I].
      if mixed_weights is None:
        mixed_weights = jnp.einsum('eoi,be->boi', weights, expert_coefficients)
      B = mixed_weights.shape[0]
      N = x.shape[0]

      if self.assume_equal_contiguous_batches and N % B == 0:
        # The caller guarantees batch_indices are [0,0,...,1,1,...] with exactly N/B rows per system.
        R = N // B
        if x.ndim == 2:
          out = jnp.einsum(
            'bni,boi->bno', x.reshape(B, R, -1), mixed_weights
          ).reshape(N, -1)
        elif x.ndim == 3:
          C = x.shape[1]
          out = jnp.einsum(
            'bnci,boi->bnco', x.reshape(B, R, C, -1), mixed_weights
          ).reshape(N, C, -1)
        else:
          raise ValueError(f'MOLELinear: unsupported input ndim={x.ndim}')
      else:
        # segment_mm uses sizes as contiguous row segments; callers must
        # guarantee batch_indices are grouped by system before enabling it.
        use_segment_mm = (
          self.assume_segment_contiguous_batches
          and self.use_segment_mm_pallas
          and self.max_segment_size is not None
          and jax.default_backend() == 'gpu'
        )
        if use_segment_mm:
          out = segment_mm(
            x,
            mixed_weights,
            jnp.bincount(batch_indices, length=B),
            use_pallas=True,
            max_size=self.max_segment_size,
          )
        else:
          # General multi-system case: per-system matmul via scan. Each step
          # multiplies all rows by one system's weight and masks the rows that
          # belong to that system. This handles uneven and interleaved batches.
          if x.ndim == 2:

            def apply_system(acc, inputs):
              system_weights, system_idx = inputs
              system_out = jnp.einsum('ni,oi->no', x, system_weights)
              mask = batch_indices == system_idx
              return acc + jnp.where(mask[:, None], system_out, 0.0), None

          elif x.ndim == 3:

            def apply_system(acc, inputs):
              system_weights, system_idx = inputs
              system_out = jnp.einsum('nci,oi->nco', x, system_weights)
              mask = batch_indices == system_idx
              return acc + jnp.where(mask[:, None, None], system_out, 0.0), None

          else:
            raise ValueError(f'MOLELinear: unsupported input ndim={x.ndim}')

          init = jnp.zeros(x.shape[:-1] + (self.out_features,), dtype=x.dtype)
          system_ids = jnp.arange(B, dtype=batch_indices.dtype)
          out, _ = jax.lax.scan(apply_system, init, (mixed_weights, system_ids))

    if self.bias is not None:
      out = out + self.bias

    return out


class RoutingMLP(eqx.Module):
  """Routing MLP that computes expert mixing coefficients.

  Takes system-level features (composition + charge/spin/dataset)
  and produces softmax'd expert weights.

  Attributes:
      layers: Dense layers keyed by their checkpoint name.
      dropout_rate: Dropout rate for expert selection.
  """

  layers: Dict[str, primitives.Dense]
  dropout_rate: float = eqx.field(static=True)

  def __init__(
    self,
    in_features: int,
    hidden_channels: int,
    num_experts: int,
    dropout_rate: float = 0.0,
    *,
    key: Array,
  ):
    keys = jax.random.split(key, 3)
    self.layers = {
      'layers_0': primitives.Dense(in_features, hidden_channels, key=keys[0]),
      'layers_2': primitives.Dense(
        hidden_channels, hidden_channels, key=keys[1]
      ),
      'layers_4': primitives.Dense(hidden_channels, num_experts, key=keys[2]),
    }
    self.dropout_rate = dropout_rate

  def __call__(
    self,
    system_features: Array,
    deterministic: bool = True,
    *,
    key: Array | None = None,
  ) -> Array:
    """Compute expert mixing coefficients.

    Args:
        system_features: Per-system features, shape [num_systems, feature_dim].
        deterministic: If False, apply dropout.
        key: Dropout key, required when deterministic is False.

    Returns:
        Expert coefficients, shape [num_systems, num_experts].
    """
    x = self.layers['layers_0'](system_features)
    x = jax.nn.silu(x)
    x = self.layers['layers_2'](x)
    x = jax.nn.silu(x)
    x = self.layers['layers_4'](x)
    x = jax.nn.silu(x)  # PT Sequential has SiLU after last Linear too

    if not deterministic and self.dropout_rate > 0:
      if key is None:
        raise ValueError('Dropout needs a key when deterministic is False.')
      keep_prob = 1.0 - self.dropout_rate
      mask = jax.random.bernoulli(key, keep_prob, x.shape)
      x = jnp.where(mask, x / keep_prob, 0.0)

    # Softmax + small epsilon for numerical stability
    coefficients = jax.nn.softmax(x, axis=-1) + 0.005

    return coefficients
