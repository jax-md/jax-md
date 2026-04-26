# UMA (Universal Models for Atoms) - JAX Port

A JAX/Flax implementation of the UMA model from [FairChem](https://github.com/FAIR-Chem/fairchem), designed for molecular dynamics simulations with JAX-MD.

## Quick Start

### With JAX-MD (recommended)

```python
from jax_md import space, energy, simulate, quantity
import jax
import jax.numpy as jnp
from jax_md._nn.uma.model import UMAConfig

# Set up periodic system
box_size = 10.0
displacement_fn, shift_fn = space.periodic(box_size)

# Atomic numbers (e.g., silicon)
atoms = jnp.array([14, 14, 14, 14])

# Create UMA energy function (follows JAX-MD convention)
cfg = UMAConfig(sphere_channels=64, num_layers=2, hidden_channels=64)
neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
    displacement_fn, box_size, cfg=cfg, atoms=atoms
)

# Initialize
key = jax.random.PRNGKey(0)
positions = jax.random.uniform(key, (4, 3)) * box_size
nbrs = neighbor_fn.allocate(positions)
params = init_fn(key, positions, nbrs)

# Compute energy and forces
E = energy_fn(params, positions, nbrs)
forces = -jax.grad(energy_fn, argnums=1)(params, positions, nbrs)
```

### Loading Pretrained Weights

```python
from jax_md._nn.uma import load_pretrained
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices

model_name = 'uma-s-1p1'
config, _, _ = load_pretrained(model_name)
dataset_idx = dataset_names_to_indices(['omat'], config.dataset_list)

neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
    displacement_fn, box_size,
    checkpoint_path=model_name,  # loads native weights or falls back to HF
    atoms=atoms,
    dataset_idx=dataset_idx,
)
params = init_fn(key, positions, nbrs)  # Returns pretrained weights
```

### Pretrained MoE (direct usage)

```python
from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
from jax_md._nn.uma.heads import MLPEnergyHead
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices

config, params, head_params = load_pretrained('uma-s-1p1')
model = UMAMoEBackbone(config=config)
head = MLPEnergyHead(
    sphere_channels=config.sphere_channels,
    hidden_channels=config.hidden_channels,
)
dataset_idx = dataset_names_to_indices(['omat'], config.dataset_list)

emb = model.apply(params, positions, atomic_numbers, batch,
                   edge_index, edge_vec, charge, spin, dataset_idx)
result = head.apply(head_params, emb['node_embedding'], batch, num_systems)
energy = result['energy']
```

### Direct Model Usage

```python
from jax_md._nn import uma

config = uma.default_config()
model = uma.UMABackbone(config=config)

# Dataset names must be converted to integer indices outside JIT
dataset_idx = uma.dataset_names_to_indices(['omat'], config.dataset_list)

# `params` can come from model.init(...) or converted pretrained weights.
output = model.apply(
    params,
    positions,           # [num_atoms, 3]
    atomic_numbers,      # [num_atoms]
    batch,               # [num_atoms]
    edge_index,          # [2, num_edges]
    edge_distance_vec,   # [num_edges, 3]
    charge,              # [num_systems]
    spin,                # [num_systems]
    dataset_idx,         # [num_systems] (integer)
)
node_embedding = output['node_embedding']  # [N, (lmax+1)^2, C]
```

## Key Design Decisions

- **Integer dataset indices**: Dataset names are converted to integers via `dataset_names_to_indices()` outside jitted code.
- **Scatter-add aggregation**: Standard backbone and head aggregations use `jnp.zeros(...).at[idx].add(...)`; optional fused edgewise kernels may use segment reductions.
- **Custom safe_acos**: Uses `jax.custom_jvp` with exact `arccos` in the forward pass and a clamped derivative to match PyTorch's `Safeacos` behavior near +/-1.
- **Grid matrices**: Uses PyTorch e3nn when available for FairChem-compatible SO(3) grid matrices; falls back to a NumPy implementation that may differ slightly.
- **Energy-conserving forces**: The `uma_neighbor_list` path exposes a scalar energy, so forces are computed with `-jax.grad(energy_fn)`. Direct force heads remain lower-level utilities.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sphere_channels` | 128 | Feature channels |
| `lmax` | 2 | Max spherical harmonic degree |
| `mmax` | 2 | Max spherical harmonic order |
| `num_layers` | 2 | Number of UMA blocks |
| `hidden_channels` | 128 | Hidden MLP dimension |
| `cutoff` | 5.0 | Distance cutoff (Angstroms) |
| `num_distance_basis` | 512 | Gaussian basis functions |
| `norm_type` | 'rms_norm_sh' | Normalization type |
| `act_type` | 'gate' | Activation ('gate' or 's2') |
| `ff_type` | 'grid' | FFN type ('grid' or 'spectral') |
| `activation_checkpointing` | False | Use `nn.remat` for memory savings |

Pretrained checkpoints load their own architecture values from checkpoint
metadata. For example, the pretrained UMA cutoff and distance-basis size may
differ from the random `UMAConfig()` defaults above.

## File Structure

```
jax_md/_nn/uma/
├── model.py                 # UMABackbone, UMAConfig
├── model_moe.py             # UMAMoEBackbone, UMAMoEConfig, load_pretrained
├── blocks.py                # UMABlock, Edgewise, Atomwise
├── heads.py                 # Energy/Force prediction heads
├── featurizer.py            # JAX-MD neighbor list -> UMA input
├── pretrained.py            # Download, convert, cache pretrained checkpoints
├── weight_conversion.py     # PyTorch -> JAX weight conversion
├── checkpoints/             # JSON configs for pretrained models
├── common/
│   ├── so3.py              # CoefficientMapping, SO3Grid
│   └── rotation.py         # Wigner D-matrix computation
└── nn/
    ├── so3_layers.py       # SO3Linear
    ├── so2_layers.py       # SO2Convolution, SO2MConv
    ├── mole.py             # MOLE (Mixture of Layer Experts)
    ├── radial.py           # GaussianSmearing, PolynomialEnvelope
    ├── activation.py       # GateActivation, S2Activation
    ├── layer_norm.py       # Equivariant normalization
    └── embedding.py        # Atomic/charge/spin/dataset embeddings
```
