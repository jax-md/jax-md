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

"""`mm.py` utilities"""

from functools import wraps, partial

from typing import Callable, Tuple, TextIO, Dict, Any, Optional, Iterable, NamedTuple

import jax
import jax.numpy as jnp
from jax import ops
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk
from jax_md import space, smap, partition, nn, quantity, interpolate, util, dataclasses, energy

maybe_downcast = util.maybe_downcast

# Types


f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
MetricFn = space.MetricFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat
MaskFn = Callable[[Array], Array]

def
