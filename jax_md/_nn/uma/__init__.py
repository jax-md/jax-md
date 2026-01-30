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

"""UMA (Universal Models for Atoms) force field implementation.

This is a JAX port of the UMA model from FairChem, designed to enable
loading of pretrained PyTorch weights.

The UMA model is an equivariant graph neural network for atomic systems
featuring SO(3) symmetry via Wigner D-matrices and spherical harmonic
representations.

Example usage:
    from jax_md._nn import uma

    # Load pretrained weights from a PyTorch checkpoint
    params = uma.load_pytorch_checkpoint('uma_model.pt')

    # Create model
    model = uma.UMABackbone(config)

    # Run inference
    embeddings = model.apply(params, positions, atomic_numbers, ...)
"""

from jax_md._nn.uma.model import UMABackbone
from jax_md._nn.uma.model import UMAConfig
from jax_md._nn.uma.blocks import UMABlock
from jax_md._nn.uma.heads import MLPEnergyHead
from jax_md._nn.uma.heads import LinearEnergyHead
from jax_md._nn.uma.heads import LinearForceHead
from jax_md._nn.uma.weight_conversion import load_pytorch_checkpoint
from jax_md._nn.uma.weight_conversion import convert_pytorch_state_dict

__all__ = [
  'UMABackbone',
  'UMAConfig',
  'UMABlock',
  'MLPEnergyHead',
  'LinearEnergyHead',
  'LinearForceHead',
  'load_pytorch_checkpoint',
  'convert_pytorch_state_dict',
]
