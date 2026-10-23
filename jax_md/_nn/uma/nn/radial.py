"""Radial basis functions for distance encoding.

This module provides radial basis functions used to encode interatomic
distances in the UMA model.

Ported from FairChem's UMA implementation.
"""

from typing import Dict, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_md import util as jmd_util
from jax_md._nn.uma.nn import primitives

Array = jmd_util.Array


class PolynomialEnvelope(eqx.Module):
  """Polynomial envelope function for smooth distance cutoff.

  Ensures contributions smoothly go to zero at the cutoff distance.

  Attributes:
      exponent: Exponent p for the polynomial, default 5.
  """

  exponent: int = eqx.field(static=True)
  a: float = eqx.field(static=True)
  b: float = eqx.field(static=True)
  c: float = eqx.field(static=True)

  def __init__(self, exponent: int = 5):
    self.exponent = exponent
    p = float(exponent)
    self.a = -(p + 1) * (p + 2) / 2
    self.b = p * (p + 2)
    self.c = -p * (p + 1) / 2

  def __call__(self, d_scaled: Array) -> Array:
    """Apply polynomial envelope.

    Args:
        d_scaled: Scaled distances (d / cutoff), shape [num_edges].

    Returns:
        Envelope values, shape [num_edges]. Zero for d_scaled >= 1.
    """
    p = float(self.exponent)
    env_val = 1.0 + (d_scaled**p) * (
      self.a + d_scaled * (self.b + self.c * d_scaled)
    )
    return jnp.where(d_scaled < 1.0, env_val, 0.0)


class GaussianSmearing(eqx.Module):
  """Gaussian basis expansion for distance encoding.

  Expands distances into a set of Gaussian basis functions.

  Attributes:
      coeff: Shared exponent coefficient.
  """

  start: float = eqx.field(static=True)
  stop: float = eqx.field(static=True)
  num_gaussians: int = eqx.field(static=True)
  coeff: float = eqx.field(static=True)

  def __init__(
    self,
    start: float = 0.0,
    stop: float = 5.0,
    num_gaussians: int = 50,
    basis_width_scalar: float = 2.0,
  ):
    self.start = start
    self.stop = stop
    self.num_gaussians = num_gaussians
    # Concrete even under eqx.filter_eval_shape
    with jax.ensure_compile_time_eval():
      offset = jnp.linspace(start, stop, num_gaussians)
      self.coeff = float(
        -0.5 / (basis_width_scalar * (offset[1] - offset[0])) ** 2
      )

  def __call__(self, dist: Array) -> Array:
    """Expand distances into Gaussian basis.

    Args:
        dist: Distances, shape [num_edges].

    Returns:
        Gaussian basis expansion, shape [num_edges, num_gaussians].
    """
    # Constants under a trace, not leaves
    with jax.ensure_compile_time_eval():
      offset = jnp.linspace(self.start, self.stop, self.num_gaussians)
    dist = dist.reshape(-1, 1) - offset.reshape(1, -1)
    return jnp.exp(self.coeff * dist**2)


class RadialMLP(eqx.Module):
  """Multi-layer perceptron for radial features.

  Processes edge scalar features through linear layers with LayerNorm
  and SiLU activation.

  Attributes:
      linear: Dense layers keyed by their checkpoint name.
      norm: Layer norms keyed by their checkpoint name.
  """

  linear: Dict[str, primitives.Dense]
  norm: Dict[str, primitives.LayerNorm]
  n_layers: int = eqx.field(static=True)

  def __init__(self, channels_list: Sequence[int], *, key: Array):
    n = len(channels_list)
    keys = jax.random.split(key, max(n - 1, 1))
    self.n_layers = n
    self.linear = {
      f'linear_{i}': primitives.Dense(
        channels_list[i - 1], channels_list[i], key=keys[i - 1]
      )
      for i in range(1, n)
    }
    self.norm = {
      f'norm_{i}': primitives.LayerNorm(channels_list[i])
      for i in range(1, n - 1)
    }

  def __call__(self, inputs: Array) -> Array:
    """Apply radial MLP.

    Args:
        inputs: Input features, shape [num_edges, input_channels].

    Returns:
        Output features, shape [num_edges, channels_list[-1]].
    """
    x = inputs
    for i in range(1, self.n_layers):
      x = self.linear[f'linear_{i}'](x)

      # Apply LayerNorm and SiLU for all but the last layer
      if i < self.n_layers - 1:
        x = self.norm[f'norm_{i}'](x)
        x = jax.nn.silu(x)

    return x
