# Copyright 2024 Google LLC
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

"""Neural network layers for UMA model."""

from jax_md.ff.uma.nn.so3_layers import SO3Linear
from jax_md.ff.uma.nn.so2_layers import SO2Convolution
from jax_md.ff.uma.nn.so2_layers import SO2MConv
from jax_md.ff.uma.nn.radial import GaussianSmearing
from jax_md.ff.uma.nn.radial import PolynomialEnvelope
from jax_md.ff.uma.nn.radial import RadialMLP
from jax_md.ff.uma.nn.activation import GateActivation
from jax_md.ff.uma.nn.activation import ScaledSiLU
from jax_md.ff.uma.nn.activation import SeparableS2Activation
from jax_md.ff.uma.nn.layer_norm import EquivariantRMSNorm
from jax_md.ff.uma.nn.layer_norm import EquivariantLayerNorm
from jax_md.ff.uma.nn.layer_norm import get_normalization_layer
from jax_md.ff.uma.nn.embedding import AtomicEmbedding
from jax_md.ff.uma.nn.embedding import ChgSpinEmbedding
from jax_md.ff.uma.nn.embedding import DatasetEmbedding
from jax_md.ff.uma.nn.embedding import EdgeDegreeEmbedding

__all__ = [
    'SO3Linear',
    'SO2Convolution',
    'SO2MConv',
    'GaussianSmearing',
    'PolynomialEnvelope',
    'RadialMLP',
    'GateActivation',
    'ScaledSiLU',
    'SeparableS2Activation',
    'EquivariantRMSNorm',
    'EquivariantLayerNorm',
    'get_normalization_layer',
    'AtomicEmbedding',
    'ChgSpinEmbedding',
    'DatasetEmbedding',
    'EdgeDegreeEmbedding',
]
