# Copyright 2019 Google LLC
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

"""Defines utility functions."""

from typing import Iterable, Union, Optional, Any

import jax
from brainunit import Quantity
from jax.tree_util import register_pytree_node
from jax.lib import xla_bridge
import jax.numpy as jnp
from jax import jit

from functools import partial

import numpy as onp

Array = jnp.ndarray
PyTree = Any

i16 = jnp.int16
i32 = jnp.int32
i64 = jnp.int64

f32 = jnp.float32
f64 = jnp.float64


CUSTOM_SIMULATION_TYPE = []


def register_custom_simulation_type(t: Any):
  global CUSTOM_SIMULATION_TYPE
  CUSTOM_SIMULATION_TYPE += [t]


def check_custom_simulation_type(x: Any) -> bool:
  if type(x) in CUSTOM_SIMULATION_TYPE:
    raise ValueError()


def static_cast(*xs):
  """Function to cast a value to the lowest dtype that can express it."""
  # NOTE(schsam): static_cast is so named because it cannot be jit.
  if xla_bridge.get_backend().platform == 'tpu':
    return (Quantity(jnp.array(x.mantissa, jnp.float32), unit=x.unit) if isinstance(x, Quantity) \
              else (jnp.array(x, jnp.float32)) for x in xs)
  else:
    return (Quantity(jnp.array(x.mantissa, dtype=onp.min_scalar_type(x.mantissa)), unit=x.unit) if isinstance(x, Quantity) \
      else jnp.array(x, dtype=onp.min_scalar_type(x)) for x in xs)


def register_pytree_namedtuple(cls):
  register_pytree_node(
      cls,
      lambda xs: (tuple(xs), None),
      lambda _, xs: cls(*xs))


def merge_dicts(a, b, ignore_unused_parameters=False):
  if not ignore_unused_parameters:
    return {**a, **b}

  merged = dict(a)
  for key in merged.keys():
    b_val = b.get(key)
    if b_val is not None:
      merged[key] = b_val
  return merged


@partial(jit, static_argnums=(1,))
def safe_mask(mask, fn, operand, placeholder=0):
  masked = jnp.where(mask, operand, 0)
  return jnp.where(mask, fn(masked), placeholder)


def high_precision_sum(X: Array,
                       axis: Optional[Union[Iterable[int], int]]=None,
                       keepdims: bool=False):
  """Sums over axes at 64-bit precision then casts back to original dtype."""
  X_unit = None
  if isinstance(X, Quantity):
    X_unit = X.unit
    X = X.mantissa

  if jnp.issubdtype(X.dtype, jnp.integer):
    dtyp = jnp.int64
  elif jnp.issubdtype(X.dtype, jnp.complexfloating):
    dtyp = jnp.complex128
  else:
    dtyp = jnp.float64

  if X_unit is None:
    return jnp.array(
        jnp.sum(X, axis=axis, dtype=dtyp, keepdims=keepdims), dtype=X.dtype)
  else:
    return Quantity(
        jnp.array(
            jnp.sum(X, axis=axis, dtype=dtyp, keepdims=keepdims), dtype=X.dtype),
        unit=X_unit)


def maybe_downcast(x):
  x_unit = None
  if isinstance(x, Quantity):
    x_unit = x.unit
    x = x.mantissa
  if isinstance(x, jnp.ndarray) and x.dtype is jnp.dtype('float64'):
    return x
  return jnp.array(x, f32) if x_unit is None else Quantity(jnp.array(x, f32), unit=x_unit)


def is_array(x: Any) -> bool:
  return isinstance(x, (jnp.ndarray, onp.ndarray))
