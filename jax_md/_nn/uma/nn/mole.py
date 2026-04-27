"""Mixture-of-Linear-Experts (MOLE) layers for UMA MoE backbone.

The MOLE approach mixes expert weight matrices per-system:
  1. Routing MLP: system features -> mixing coefficients [num_systems, num_experts]
  2. Weight mixing: einsum('eoi,be->boi', expert_weights, coefficients)
  3. Application: each atom uses its system's mixed weight matrix

This is the JAX equivalent of fairchem's MOLE/MOLEDGL modules.
"""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp


class MOLELinear(nn.Module):
  """Mixture-of-Linear-Experts layer.

  Holds [num_experts, out_features, in_features] weight tensor.
  At forward time, mixes experts using per-system coefficients,
  then applies the mixed weight to each atom/edge.

  Attributes:
      num_experts: Number of expert weight matrices.
      in_features: Input dimension.
      out_features: Output dimension.
      use_bias: Whether to include a shared bias term.
  """

  num_experts: int
  in_features: int
  out_features: int
  use_bias: bool = True

  @nn.compact
  def __call__(
    self,
    x: jnp.ndarray,
    expert_coefficients: jnp.ndarray,
    batch_indices: jnp.ndarray,
  ) -> jnp.ndarray:
    """Apply MOLE linear layer.

    Args:
        x: Input tensor, shape [N, ..., in_features].
        expert_coefficients: Mixing coefficients, shape [num_systems, num_experts].
        batch_indices: System index for each atom/edge, shape [N].

    Returns:
        Output tensor, shape [N, ..., out_features].
    """
    bound = math.sqrt(1.0 / self.in_features)

    weights = self.param(
      'weights',
      lambda key, shape: jax.random.uniform(
        key, shape, minval=-bound, maxval=bound
      ),
      (self.num_experts, self.out_features, self.in_features),
    )

    # Mix experts per system: [E, O, I] x [B, E] -> [B, O, I]
    mixed_weights = jnp.einsum('eoi,be->boi', weights, expert_coefficients)

    # Avoid materializing [N, O, I] per-item weights
    if x.ndim == 2:
      if mixed_weights.shape[0] == 1:
        out = jnp.einsum('ni,oi->no', x, mixed_weights[0])
      else:

        def apply_system(acc, inputs):
          system_weights, system_idx = inputs
          system_out = jnp.einsum('ni,oi->no', x, system_weights)
          mask = batch_indices == system_idx
          return acc + jnp.where(mask[:, None], system_out, 0.0), None

        init = jnp.zeros(x.shape[:-1] + (self.out_features,), dtype=x.dtype)
        system_ids = jnp.arange(
          mixed_weights.shape[0], dtype=batch_indices.dtype
        )
        out, _ = jax.lax.scan(apply_system, init, (mixed_weights, system_ids))
    elif x.ndim == 3:
      if mixed_weights.shape[0] == 1:
        out = jnp.einsum('nci,oi->nco', x, mixed_weights[0])
      else:

        def apply_system(acc, inputs):
          system_weights, system_idx = inputs
          system_out = jnp.einsum('nci,oi->nco', x, system_weights)
          mask = batch_indices == system_idx
          return acc + jnp.where(mask[:, None, None], system_out, 0.0), None

        init = jnp.zeros(x.shape[:-1] + (self.out_features,), dtype=x.dtype)
        system_ids = jnp.arange(
          mixed_weights.shape[0], dtype=batch_indices.dtype
        )
        out, _ = jax.lax.scan(apply_system, init, (mixed_weights, system_ids))
    else:
      raise ValueError(f'MOLELinear: unsupported input ndim={x.ndim}')

    if self.use_bias:
      bias = self.param(
        'bias',
        lambda key, shape: jax.random.uniform(
          key, shape, minval=-bound, maxval=bound
        ),
        (self.out_features,),
      )
      out = out + bias

    return out


class RoutingMLP(nn.Module):
  """Routing MLP that computes expert mixing coefficients.

  Takes system-level features (composition + charge/spin/dataset)
  and produces softmax'd expert weights.

  Attributes:
      hidden_channels: Hidden layer size.
      num_experts: Number of experts to route to.
      dropout_rate: Dropout rate for expert selection.
  """

  hidden_channels: int
  num_experts: int
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(
    self, system_features: jnp.ndarray, deterministic: bool = True
  ) -> jnp.ndarray:
    """Compute expert mixing coefficients.

    Args:
        system_features: Per-system features, shape [num_systems, feature_dim].
        deterministic: If False, apply dropout.

    Returns:
        Expert coefficients, shape [num_systems, num_experts].
    """
    x = nn.Dense(self.hidden_channels, name='layers_0')(system_features)
    x = nn.silu(x)
    x = nn.Dense(self.hidden_channels, name='layers_2')(x)
    x = nn.silu(x)
    x = nn.Dense(self.num_experts, name='layers_4')(x)
    x = nn.silu(x)  # PT Sequential has SiLU after last Linear too

    if not deterministic and self.dropout_rate > 0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

    # Softmax + small epsilon for numerical stability
    coefficients = jax.nn.softmax(x, axis=-1) + 0.005

    return coefficients
