# UMA (Universal Models for Atoms) - JAX Port

A JAX/Flax implementation of the UMA model from [FairChem](https://github.com/FAIR-Chem/fairchem), designed for molecular dynamics simulations with JAX-MD.

## Overview

UMA is a state-of-the-art equivariant graph neural network for predicting atomic energies and forces. This JAX port faithfully replicates the PyTorch implementation, enabling:

- **Loading pretrained PyTorch weights** for immediate use
- **JAX's JIT compilation** for fast inference
- **Automatic differentiation** for computing forces as energy gradients
- **Integration with JAX-MD** for molecular dynamics simulations

## Installation

The UMA module is included with JAX-MD. Ensure you have the required dependencies:

```bash
pip install jax jaxlib flax
pip install torch  # Only needed for loading PyTorch checkpoints
```

## Quick Start

### Loading a Pretrained Model

```python
from jax_md._nn import uma
import jax.numpy as jnp

# Load pretrained weights from a PyTorch checkpoint
params = uma.load_pytorch_checkpoint('path/to/uma_checkpoint.pt')

# Create model configuration
config = uma.UMAConfig(
    sphere_channels=128,
    lmax=2,
    mmax=2,
    num_layers=4,
    hidden_channels=128,
    cutoff=6.0,
    num_distance_basis=512,
)

# Create model
model = uma.UMABackbone(config=config)

# Run inference
output = model.apply(
    params,
    positions,           # [num_atoms, 3]
    atomic_numbers,      # [num_atoms]
    batch,               # [num_atoms] - batch index for each atom
    edge_index,          # [2, num_edges]
    edge_distance_vec,   # [num_edges, 3]
    charge,              # [num_systems]
    spin,                # [num_systems]
    dataset=['omat'],    # Dataset name for each system
)

node_embedding = output['node_embedding']  # [num_atoms, (lmax+1)^2, sphere_channels]
```

### Computing Energies and Forces

```python
from jax_md._nn.uma.heads import MLPEnergyHead, LinearForceHead
import jax

# Energy prediction
energy_head = MLPEnergyHead(
    sphere_channels=config.sphere_channels,
    hidden_channels=config.hidden_channels,
)

energy_output = energy_head.apply(
    energy_params,
    output['node_embedding'],
    batch,
    num_systems=len(charge),
)
energy = energy_output['energy']  # [num_systems]

# Force prediction via gradient
def energy_fn(pos):
    # Recompute edge vectors
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    emb = model.apply(params, pos, atomic_numbers, batch,
                      edge_index, edge_vec, charge, spin, dataset)
    e = energy_head.apply(energy_params, emb['node_embedding'],
                          batch, num_systems)
    return e['energy'].sum()

forces = -jax.grad(energy_fn)(positions)  # [num_atoms, 3]
```

### Using with JAX-MD

```python
from jax_md import space, simulate

# Create displacement function
displacement_fn, shift_fn = space.periodic(box_size)

# Create energy function for JAX-MD
def uma_energy_fn(positions, neighbor):
    # Build edges from neighbor list
    edge_index = build_edges_from_neighbor(neighbor)
    edge_vec = displacement_fn(positions[edge_index[0]], positions[edge_index[1]])

    emb = model.apply(params, positions, atomic_numbers, batch,
                      edge_index, edge_vec, charge, spin, dataset)
    energy = energy_head.apply(energy_params, emb['node_embedding'],
                               batch, num_systems)
    return energy['energy'].sum()

# Run NVE simulation
init_fn, apply_fn = simulate.nve(uma_energy_fn, shift_fn, dt=0.001)
state = init_fn(key, positions)

for _ in range(num_steps):
    state = apply_fn(state)
```

## Architecture

UMA uses an SO(3)-equivariant architecture with the following components:

```
Input: positions, atomic_numbers, edges
         │
         ▼
┌─────────────────────────────┐
│   Atomic Embedding          │  Embed atomic numbers
│   + Charge/Spin/Dataset Emb │  System-level conditioning
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Edge Degree Embedding     │  Initial edge aggregation
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   UMA Blocks (×N)           │
│   ├── Norm                  │
│   ├── Edgewise (SO2 Conv)   │  Message passing with Wigner rotation
│   ├── Residual              │
│   ├── Norm                  │
│   ├── Atomwise (Grid FFN)   │  Per-atom feed-forward
│   └── Residual              │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Final Norm                │
└─────────────────────────────┘
         │
         ▼
Output: node_embedding [N, (lmax+1)², C]
```

### Key Components

| Component | Description |
|-----------|-------------|
| `SO3Linear` | Equivariant linear layer with per-degree weights |
| `SO2Convolution` | SO(2) convolution in edge-aligned frame |
| `GateActivation` | Gated nonlinearity preserving equivariance |
| `EquivariantRMSNorm` | RMS normalization respecting SO(3) symmetry |
| `WignerD` | Rotation matrices for edge frame transformation |

## Configuration

### UMAConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_num_elements` | 100 | Maximum atomic number to embed |
| `sphere_channels` | 128 | Number of feature channels |
| `lmax` | 2 | Maximum spherical harmonic degree |
| `mmax` | 2 | Maximum spherical harmonic order |
| `num_layers` | 2 | Number of UMA blocks |
| `hidden_channels` | 128 | Hidden dimension in MLPs |
| `cutoff` | 5.0 | Distance cutoff (Angstroms) |
| `edge_channels` | 128 | Edge feature dimension |
| `num_distance_basis` | 512 | Number of Gaussian basis functions |
| `norm_type` | 'rms_norm_sh' | Normalization type |
| `act_type` | 'gate' | Activation type ('gate' or 's2') |
| `ff_type` | 'grid' | FFN type ('grid' or 'spectral') |
| `dataset_list` | ['oc20', ...] | Supported dataset names |

## Weight Loading

### From PyTorch Checkpoint

```python
# Load full checkpoint
params = uma.load_pytorch_checkpoint('checkpoint.pt')

# Extract config from checkpoint (if available)
from jax_md._nn.uma.weight_conversion import config_from_pytorch_checkpoint
config = config_from_pytorch_checkpoint('checkpoint.pt')
```

### Manual Conversion

```python
import torch
from jax_md._nn.uma.weight_conversion import convert_pytorch_state_dict

# Load PyTorch state dict
pt_state = torch.load('checkpoint.pt', map_location='cpu')['state_dict']
np_state = {k: v.numpy() for k, v in pt_state.items()}

# Convert to JAX format
jax_params = convert_pytorch_state_dict(np_state)
```

### Verifying Weights

```python
from jax_md._nn.uma.weight_conversion import verify_weight_loading

results = verify_weight_loading(jax_params, np_state)
for key, matches in results.items():
    if not matches:
        print(f"Mismatch: {key}")
```

## Prediction Heads

### MLPEnergyHead
3-layer MLP for energy prediction from scalar features.

```python
head = MLPEnergyHead(sphere_channels=128, hidden_channels=128)
output = head.apply(params, node_embedding, batch, num_systems)
# output['energy']: [num_systems]
```

### LinearEnergyHead
Single linear layer for fast inference.

```python
head = LinearEnergyHead(sphere_channels=128)
output = head.apply(params, node_embedding, batch, num_systems)
```

### LinearForceHead
Direct force prediction using SO(3) linear layer on vector components.

```python
head = LinearForceHead(sphere_channels=128)
output = head.apply(params, node_embedding)
# output['forces']: [num_atoms, 3]
```

## File Structure

```
jax_md/ff/uma/
├── __init__.py              # Public API
├── model.py                 # UMABackbone, UMAConfig
├── blocks.py                # UMABlock, Edgewise, Atomwise
├── heads.py                 # Energy/Force prediction heads
├── weight_conversion.py     # PyTorch → JAX conversion
├── Jd.pt                    # Precomputed Jacobi matrices
├── common/
│   ├── so3.py              # CoefficientMapping, SO3Grid
│   └── rotation.py         # Wigner D-matrix computation
└── nn/
    ├── so3_layers.py       # SO3Linear
    ├── so2_layers.py       # SO2Convolution
    ├── radial.py           # Distance basis functions
    ├── activation.py       # Equivariant activations
    ├── layer_norm.py       # Equivariant normalization
    └── embedding.py        # Atomic/system embeddings
```

## Differences from PyTorch Implementation

| Aspect | PyTorch | JAX |
|--------|---------|-----|
| Module system | `nn.Module` with mutable state | Flax `nn.Module` with explicit params |
| Buffers | `register_buffer()` | Stored in config/dataclass |
| Weight shapes | `[out, in]` for Linear | `[in, out]` (transposed on load) |
| Graph parallel | `gp_utils` | Not implemented (single device) |
| Activation checkpointing | Supported | Use `jax.checkpoint` manually |

## Citation

If you use this code, please cite the original UMA paper:

```bibtex
@article{uma2024,
  title={Universal Models for Atoms},
  author={Meta AI},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This JAX port is released under the Apache 2.0 License, consistent with JAX-MD.
The original UMA implementation is from FairChem (MIT License).
