# API Reference

Complete API documentation for all JAX MD modules, auto-generated from
source docstrings.

---

## Core

Fundamental building blocks for defining simulation geometry and units.

| Module | Description |
|--------|-------------|
| [`jax_md.space`](space.md) | Displacement functions, periodic boundaries, coordinate transforms |
| [`jax_md.units`](units.md) | Physical unit systems (metal, real, SI) |
| [`jax_md.util`](util.md) | Array utilities, precision helpers, high-precision reductions |
| [`jax_md.dataclasses`](dataclasses.md) | JAX-compatible dataclass decorator with static fields |
| [`jax_md.tpu`](tpu.md) | TPU-specific utilities |

## Spatial Partitioning

Efficient spatial data structures for short-range interactions.

| Module | Description |
|--------|-------------|
| [`jax_md.partition`](partition.md) | Neighbor lists (Dense, Sparse, OrderedSparse) and cell lists |
| [`jax_md.custom_partition`](custom_partition.md) | Multi-image neighbor lists for small boxes where cutoff > L/2 |

## Potentials

Energy functions for particle interactions.

| Module | Description |
|--------|-------------|
| [`jax_md.energy`](energy.md) | Classical potentials (LJ, Morse, Tersoff, EAM, SW, ...) and NN wrappers |
| [`jax_md.smap`](smap.md) | Higher-order functions: `pair`, `bond`, `triplet` combinators |
| [`jax_md.custom_smap`](custom_smap.md) | Structure maps for multi-image neighbor lists |
| [`jax_md.interpolate`](interpolate.md) | Spline interpolation for tabulated potentials |

## Force Fields

Reactive and molecular mechanics force fields.

| Module | Description |
|--------|-------------|
| [`jax_md.reaxff`](reaxff.md) | ReaxFF reactive force field |

## Neural Networks

Machine-learned interatomic potentials.

| Module | Description |
|--------|-------------|
| [`jax_md.nn`](nn.md) | Graph neural network layers (MLP, GraphNetEncoder) |
| [`jax_md._nn.nequip`](nequip.md) | NequIP equivariant model with tensor product convolutions |
| [`jax_md._nn.gnome`](gnome.md) | GNoME pre-trained universal potential |
| [`jax_md._nn.behler_parrinello`](behler_parrinello.md) | Behler-Parrinello symmetry functions (radial + angular) |
| [`jax_md._nn.e3nn_layer`](e3nn_layer.md) | Equivariant tensor product layers (Flax Linen) |
| [`jax_md._nn.util`](nn_util.md) | Featurizers, checkpoint conversion, MLP builders |

## Dynamics

Simulation integrators, minimizers, and physical observables.

| Module | Description |
|--------|-------------|
| [`jax_md.simulate`](simulate.md) | NVE, NVT, NPT, Langevin, Brownian, swap MC integrators |
| [`jax_md.minimize`](minimize.md) | FIRE descent, gradient descent minimizers |
| [`jax_md.quantity`](quantity.md) | Forces, temperature, pressure, stress, pair correlation |
| [`jax_md.elasticity`](elasticity.md) | Athermal elastic moduli, Born terms |
| [`jax_md.rigid_body`](rigid_body.md) | Quaternion-based rigid body dynamics |

## Workflows

Higher-level simulation pipelines.

| Module | Description |
|--------|-------------|
| [`jax_md.a2c`](a2c.md) | Amorphous-to-crystalline structure prediction |
