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

"""Weight conversion utilities for loading PyTorch checkpoints.

This module provides utilities to convert PyTorch UMA weights to JAX format
so that pretrained models can be used in JAX-MD.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def load_pytorch_checkpoint(
    checkpoint_path: str,
    map_location: str = 'cpu',
) -> Dict[str, Any]:
    """Load a PyTorch UMA checkpoint and convert to JAX format.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint file (.pt).
        map_location: Device to load the checkpoint to.

    Returns:
        Dictionary of JAX-compatible parameters that can be used with
        the UMA Flax model.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            'PyTorch is required to load PyTorch checkpoints. '
            'Install with: pip install torch'
        )

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Convert to numpy
    np_state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}

    # Convert to JAX params format
    jax_params = convert_pytorch_state_dict(np_state_dict)

    return jax_params


def convert_pytorch_state_dict(
    state_dict: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Convert PyTorch state dict to JAX/Flax params format.

    This handles the mapping between PyTorch module naming conventions
    and Flax naming conventions.

    Args:
        state_dict: PyTorch state dict with numpy arrays.

    Returns:
        Nested dictionary of parameters in Flax format.
    """
    jax_params = {}

    # Mapping rules for converting PyTorch names to Flax names
    name_mappings = {
        # Embeddings
        'sphere_embedding.weight': ('sphere_embedding', 'embedding'),
        'source_embedding.weight': ('source_embedding', 'embedding'),
        'target_embedding.weight': ('target_embedding', 'embedding'),

        # Charge/spin/dataset embeddings
        'charge_embedding.W': ('charge_embedding', 'W'),
        'spin_embedding.W': ('spin_embedding', 'W'),
        'dataset_embedding': 'dataset_embedding',

        # Mix CSD
        'mix_csd.weight': ('mix_csd', 'kernel'),
        'mix_csd.bias': ('mix_csd', 'bias'),

        # Distance expansion
        'distance_expansion.offset': ('distance_expansion', 'offset'),

        # Edge degree embedding
        'edge_degree_embedding.rad_func': 'edge_degree_embedding/rad_func',

        # Blocks
        'blocks': 'blocks',

        # Final norm
        'norm.affine_weight': ('norm', 'affine_weight'),
        'norm.affine_bias': ('norm', 'affine_bias'),
    }

    for pt_key, pt_value in state_dict.items():
        # Skip buffer tensors (Jd matrices, etc.)
        if 'Jd_' in pt_key or pt_key.endswith('expand_index') or pt_key.endswith('coefficient_idx'):
            continue

        # Convert key to Flax format
        flax_key = _convert_key(pt_key)

        # Convert weight format (PyTorch: [out, in], Flax: [in, out])
        flax_value = _convert_weight(pt_key, pt_value)

        # Set nested dict value
        _set_nested(jax_params, flax_key, flax_value)

    return {'params': jax_params}


def _convert_key(pt_key: str) -> Tuple[str, ...]:
    """Convert PyTorch parameter key to Flax key tuple."""
    # Replace dots with tuple elements
    parts = pt_key.split('.')

    flax_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]

        # Handle numbered modules (e.g., blocks.0 -> blocks_0)
        if part.isdigit():
            # Combine with previous part
            if flax_parts:
                flax_parts[-1] = f'{flax_parts[-1]}_{part}'
            i += 1
            continue

        # Handle specific conversions
        if part == 'weight':
            flax_parts.append('kernel')
        elif part == 'fc':
            flax_parts.append('Dense_0')
        elif part == 'fc_m0':
            flax_parts.append('fc_m0')
        elif part.startswith('so2_m_conv'):
            # so2_m_conv.0 -> so2_m_conv_1
            flax_parts.append(part)
        elif part == 'net':
            # RadialMLP net -> just pass through
            pass
        elif part == 'rad_func':
            flax_parts.append('rad_func')
        elif part == 'affine_weight':
            flax_parts.append('affine_weight')
        elif part == 'affine_bias':
            flax_parts.append('affine_bias')
        elif part == 'norm_l0':
            flax_parts.append('norm_l0')
        elif part == 'scale' or part == 'bias':
            flax_parts.append(part)
        else:
            flax_parts.append(part)

        i += 1

    return tuple(flax_parts)


def _convert_weight(key: str, value: np.ndarray) -> jnp.ndarray:
    """Convert weight tensor from PyTorch to JAX format."""
    # Linear layer weights: PyTorch [out, in] -> JAX [in, out]
    if '.weight' in key and len(value.shape) == 2:
        # Check if it's a linear layer weight (not embedding)
        if 'embedding' not in key.lower():
            value = value.T

    # SO3Linear weights: [lmax+1, out, in] -> [lmax+1, in, out]
    if 'so3_linear' in key.lower() and '.weight' in key and len(value.shape) == 3:
        value = np.transpose(value, (0, 2, 1))

    return jnp.array(value)


def _set_nested(d: Dict, keys: Tuple[str, ...], value: Any) -> None:
    """Set a value in a nested dictionary."""
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def extract_config_from_checkpoint(
    checkpoint_path: str,
) -> Optional[Dict[str, Any]]:
    """Extract model configuration from a PyTorch checkpoint.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint file.

    Returns:
        Configuration dictionary if available, None otherwise.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            'PyTorch is required to load PyTorch checkpoints. '
            'Install with: pip install torch'
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Try different config locations
    if 'config' in checkpoint:
        return checkpoint['config']
    if 'hyper_parameters' in checkpoint:
        return checkpoint['hyper_parameters']
    if 'model_config' in checkpoint:
        return checkpoint['model_config']

    return None


def config_from_pytorch_checkpoint(checkpoint_path: str):
    """Create UMAConfig from a PyTorch checkpoint.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint file.

    Returns:
        UMAConfig instance matching the checkpoint.
    """
    from jax_md.ff.uma.model import UMAConfig

    pt_config = extract_config_from_checkpoint(checkpoint_path)

    if pt_config is None:
        # Return default config
        return UMAConfig()

    # Map PyTorch config to JAX config
    return UMAConfig(
        max_num_elements=pt_config.get('max_num_elements', 100),
        sphere_channels=pt_config.get('sphere_channels', 128),
        lmax=pt_config.get('lmax', 2),
        mmax=pt_config.get('mmax', 2),
        num_layers=pt_config.get('num_layers', 2),
        hidden_channels=pt_config.get('hidden_channels', 128),
        cutoff=pt_config.get('cutoff', 5.0),
        edge_channels=pt_config.get('edge_channels', 128),
        num_distance_basis=pt_config.get('num_distance_basis', 512),
        norm_type=pt_config.get('norm_type', 'rms_norm_sh'),
        act_type=pt_config.get('act_type', 'gate'),
        ff_type=pt_config.get('ff_type', 'grid'),
        grid_resolution=pt_config.get('grid_resolution'),
        chg_spin_emb_type=pt_config.get('chg_spin_emb_type', 'pos_emb'),
        dataset_list=pt_config.get('dataset_list'),
        use_dataset_embedding=pt_config.get('use_dataset_embedding', True),
    )


def verify_weight_loading(
    jax_params: Dict[str, Any],
    pytorch_state_dict: Dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Dict[str, bool]:
    """Verify that weights were loaded correctly.

    Args:
        jax_params: Converted JAX parameters.
        pytorch_state_dict: Original PyTorch state dict.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        Dictionary mapping parameter names to whether they match.
    """
    results = {}

    for pt_key, pt_value in pytorch_state_dict.items():
        # Skip buffers
        if 'Jd_' in pt_key:
            continue

        flax_key = _convert_key(pt_key)

        # Try to get the value from jax_params
        try:
            jax_value = jax_params['params']
            for k in flax_key:
                jax_value = jax_value[k]

            # Convert and compare
            expected = _convert_weight(pt_key, pt_value)
            matches = np.allclose(np.array(jax_value), np.array(expected), rtol=rtol, atol=atol)
            results[pt_key] = matches
        except (KeyError, TypeError):
            results[pt_key] = False

    return results
