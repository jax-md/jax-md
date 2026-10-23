"""UMA force field implementation, Universal Models for Atoms.

This is a JAX port of the UMA model from FairChem, designed to enable
loading of pretrained PyTorch weights.

The UMA model is an equivariant graph neural network for atomic systems
featuring SO(3) symmetry via Wigner D-matrices and spherical harmonic
representations.

Example usage:
    from jax_md._nn import uma
    from jax_md._nn.uma.pretrained import load_params_into_module

    config, params, head_params = uma.load_pretrained('uma-s-1p2')
    model = load_params_into_module(
        params, uma.UMAMoEBackbone(config=config, key=key))

    # Run inference
    embeddings = model(positions, atomic_numbers, ...)
"""

from jax_md._nn.uma.model import UMABackbone
from jax_md._nn.uma.model import UMAConfig
from jax_md._nn.uma.model import default_config
from jax_md._nn.uma.model_moe import UMAMoEBackbone
from jax_md._nn.uma.model_moe import UMAMoEConfig
from jax_md._nn.uma.model_moe import load_pretrained
from jax_md._nn.uma.blocks import UMABlock
from jax_md._nn.uma.heads import MLPEnergyHead
from jax_md._nn.uma.heads import LinearEnergyHead
from jax_md._nn.uma.heads import LinearForceHead
from jax_md._nn.uma.featurizer import uma_featurizer
from jax_md._nn.uma.weight_conversion import load_pytorch_checkpoint
from jax_md._nn.uma.weight_conversion import convert_pytorch_state_dict
from jax_md._nn.uma.weight_conversion import config_from_pytorch_checkpoint
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices
from jax_md._nn.uma.pretrained import (
  download_pretrained,
  convert_checkpoint,
  load_checkpoint_raw,
  print_conversion_report,
  PRETRAINED_MODELS,
)
