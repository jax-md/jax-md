"""SO(3) equivariant linear layers.

This module provides SO(3) equivariant linear layers that operate on
spherical harmonic representations.

Ported from FairChem's UMA implementation.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_md import util as jmd_util

Array = jmd_util.Array


class SO3Linear(eqx.Module):
  """SO(3) equivariant linear layer.

  Applies a linear transformation that respects SO(3) equivariance by using
  separate weight matrices for each degree l.

  Attributes:
      weight: Per-degree weights of shape [lmax+1, out_features, in_features].
      bias: Bias applied to the l=0 component only, or None.
      lmax: Maximum degree of spherical harmonics.
  """

  weight: Array
  bias: Array | None
  lmax: int = eqx.field(static=True)

  def __init__(
    self,
    in_features: int,
    out_features: int,
    lmax: int,
    *,
    key: Array,
    use_bias: bool = True,
  ):
    self.lmax = lmax
    # PyTorch's uniform(-1/sqrt(in), 1/sqrt(in)) init
    bound = 1.0 / math.sqrt(in_features)
    self.weight = jax.random.uniform(
      key, (lmax + 1, out_features, in_features), jnp.float32, -bound, bound
    )
    self.bias = jnp.zeros((out_features,)) if use_bias else None

  def __call__(self, input_embedding: Array) -> Array:
    """Apply SO(3) linear transformation.

    Args:
        input_embedding: Input features of shape [batch, (lmax+1)^2, in_features].

    Returns:
        Output features of shape [batch, (lmax+1)^2, out_features].
    """
    # Create expand index to map l values to all m values
    expand_index = jnp.array(
      [l for l in range(self.lmax + 1) for _ in range(2 * l + 1)]
    )

    # Expand weights: [lmax+1, out, in] -> [(lmax+1)^2, out, in]
    weight_expanded = self.weight[expand_index]

    out = jnp.einsum('bmi,moi->bmo', input_embedding, weight_expanded)

    # Add bias only to scalar (l=0) component
    if self.bias is not None:
      out = out.at[:, 0, :].add(self.bias)

    return out
