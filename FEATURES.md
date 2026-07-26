# JAX MD Features

This document summarizes the main components of the library.

## Spaces ([`space.py`](https://jax-md.readthedocs.io/en/main/jax_md.space.html))

In general we must have a way of computing the pairwise distance between atoms. We must also have efficient strategies for moving atoms in some space that may or may not be globally isomorphic to R^N. For example, periodic boundary conditions are commonplace in simulations and must be respected. Spaces are defined as a pair of functions, `(displacement_fn, shift_fn)`. Given two points `displacement_fn(R_1, R_2)` computes the displacement vector between the two points. If you would like to compute displacement vectors between all pairs of points in a given `(N, dim)` matrix the function [`space.map_product`](https://jax-md.readthedocs.io/en/main/jax_md.space.html#jax_md.space.map_product) appropriately vectorizes `displacement_fn`. It is often useful to define a metric instead of a displacement function in which case you can use the helper function [`space.metric`](https://jax-md.readthedocs.io/en/main/jax_md.space.html#jax_md.space.metric) to convert a displacement function to a metric function. Given a point and a shift `shift_fn(R, dR)` displaces the point `R` by an amount `dR`.

The following spaces are currently supported:
- [`space.free()`](https://jax-md.readthedocs.io/en/main/jax_md.space.html?highlight=free#jax_md.space.free) specifies a space with free boundary conditions.
- [`space.periodic(box_size)`](https://jax-md.readthedocs.io/en/main/jax_md.space.html?highlight=periodic#jax_md.space.periodic) specifies a space with periodic boundary conditions of side length `box_size`.
- [`space.periodic_general(box)`](https://jax-md.readthedocs.io/en/main/jax_md.space.html?highlight=periodic_general#jax_md.space.periodic_general) specifies a space as a periodic parallelepiped formed by transforming the unit cube by an affine transformation `box`.

Spaces also include mapping helpers:
- [`space.map_product()`](https://jax-md.readthedocs.io/en/main/jax_md.space.html#jax_md.space.map_product) vectorizes a displacement or metric over all pairs.
- [`space.map_bond()`](https://jax-md.readthedocs.io/en/main/jax_md.space.html#jax_md.space.map_bond) maps a displacement or metric over indexed bonds.
- [`space.map_neighbor()`](https://jax-md.readthedocs.io/en/main/jax_md.space.html#jax_md.space.map_neighbor) maps a displacement or metric over neighbor-list entries.
- [`space.inverse()`](jax_md/space.py#L110) computes the inverse of a scalar, vector, or matrix box.

Example:

```python
from jax_md import space
box_size = 25.0
displacement_fn, shift_fn = space.periodic(box_size)
```

## Units ([`units.py`](jax_md/units.py))

JAX MD computations are unitless by default, but the library includes unit-system dictionaries that make it easier to convert physical inputs into the units expected by simulations. The unit systems follow conventions similar to LAMMPS. The conversion factors are constructed in float64, so enable 64-bit precision, for example with `JAX_ENABLE_X64=1`, before using them to avoid silent truncation to float32.

The following unit systems are currently supported:
- [`units.metal_unit_system()`](jax_md/units.py#L44) provides conversion factors for metal units, with Angstrom, eV, amu, picoseconds, bar, and related quantities.
- [`units.real_unit_system()`](jax_md/units.py#L93) provides conversion factors for real units, with Angstrom, kcal/mol, grams/mol, femtoseconds, atm, and related quantities.

Example:

```python
from jax_md import units

metal = units.metal_unit_system()
dt = 1e-3 * metal['time']
kT = 300.0 * metal['temperature']
```

## Potential Energy ([`energy.py`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html))

In the simplest case, molecular dynamics calculations are often based on a pair potential that is defined by a user. This then is used to compute a total energy whose negative gradient gives forces. One of the very nice things about JAX is that we get forces for free. The second part of the code is devoted to computing energies.

We provide the following classical potentials:
- [`energy.soft_sphere`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=soft_sphere#jax_md.energy.soft_sphere) a soft sphere whose energy increases as the overlap of the spheres to some power, `alpha`.
- [`energy.lennard_jones`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=lennard_jones#jax_md.energy.lennard_jones) a standard 12-6 Lennard-Jones potential.
- [`energy.morse`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=morse#jax_md.energy.morse) a Morse potential.
- [`energy.tersoff`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=tersoff#jax_md.energy.tersoff) the Tersoff potential for simulating semiconducting materials. Can load parameters from LAMMPS files.
- [`energy.eam`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=eam#jax_md.energy.eam) embedded atom model potential with ability to load parameters from LAMMPS files.
- [`energy.stillinger_weber`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=stillinger_weber#jax_md.energy.stillinger_weber) used to model Silicon-like systems.
- [`energy.bks`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=bks#jax_md.energy.bks) Beest-Kramer-van Santen potential used to model silica.
- [`energy.gupta_potential()`](jax_md/energy.py#L449) implements the Gupta many-body potential, and [`energy.gupta_gold55()`](jax_md/energy.py#L523) provides a gold-cluster convenience wrapper.
- [`energy.edip()`](jax_md/energy.py#L1506) and [`energy.edip_neighbor_list()`](jax_md/energy.py#L1594) implement the Environment Dependent Interatomic Potential.

For finite-ranged potentials it is often useful to consider only interactions within a certain neighborhood. Most of the above potentials have a `_neighbor_list` variant such as `energy.lennard_jones_neighbor_list` that uses a neighbor list, described below, for optimization. The Gupta potentials currently do not.

Example:

```python
import jax.numpy as np
from jax import random
from jax_md import energy, quantity, space

N = 1000
spatial_dimension = 2
key = random.PRNGKey(0)
box_size = 32.0
R = box_size * random.uniform(key, (N, spatial_dimension))
displacement_fn, shift_fn = space.periodic(box_size)
energy_fn = energy.soft_sphere_pair(displacement_fn)
print('E = {}'.format(energy_fn(R)))
force_fn = quantity.force(energy_fn)
print('Total Squared Force = {}'.format(np.sum(force_fn(R) ** 2)))
```

## Electrostatics ([`energy.py`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html))

Charged systems often require splitting the Coulomb interaction into short-range and reciprocal-space pieces. JAX MD includes direct-space Coulomb terms, Ewald sums, and particle mesh Ewald (PME) utilities.

These functions are available from the public `jax_md.energy` namespace:
- [`energy.coulomb_direct_pair()`](jax_md/_energy/electrostatics.py#L56) for direct pairwise screened Coulomb interactions.
- [`energy.coulomb_direct_neighbor_list()`](jax_md/_energy/electrostatics.py#L71) for direct Coulomb interactions with a neighbor list.
- [`energy.coulomb_recip_ewald()`](jax_md/_energy/electrostatics.py#L105) for reciprocal Ewald sums.
- [`energy.coulomb_recip_pme()`](jax_md/_energy/electrostatics.py#L139) for reciprocal PME sums.
- [`energy.coulomb()`](jax_md/_energy/electrostatics.py#L205) and [`energy.coulomb_neighbor_list()`](jax_md/_energy/electrostatics.py#L227) convenience wrappers that combine direct and reciprocal terms.
- [`energy.coulomb_ewald_neighbor_list()`](jax_md/_energy/electrostatics.py#L186) for Ewald electrostatics with neighbor-listed direct-space terms.

Example:

```python
import jax.numpy as np
from jax_md import energy, space

# PME electrostatics for a 3D periodic box.
displacement_fn, shift_fn = space.periodic(10.0)
charge = np.ones((1000,))
energy_fn = energy.coulomb(displacement_fn, 10.0, charge, grid_points=32)
```

## Membrane Potentials ([`energy.py`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html))

JAX MD includes potentials for triangulated surfaces, useful for modeling membranes and vesicles whose geometry is represented by vertices and triangular faces.

The membrane potentials are:
- [`energy.triangle_area_potential()`](jax_md/energy.py#L2526) for local triangle-area conservation.
- [`energy.volume_potential()`](jax_md/energy.py#L2589) for global volume conservation of a closed triangulated surface.
- [`energy.bending_potential()`](jax_md/energy.py#L2646) for bending energy on a triangulated surface.

Example:

```python
from jax_md import energy

area_energy = energy.triangle_area_potential(
    R_mem, triangles, displacement_fn, A_0=1.0, k=10.0)
```

## Neural Network Potentials ([`energy.py`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html), [`nn.py`](jax_md/nn.py), [`_nn/`](jax_md/_nn))

JAX MD provides several neural-network potential interfaces. These follow the same general pattern as classical energies: build an energy function, use autodiff to obtain forces, and use neighbor lists when a finite cutoff is required.

We provide the following neural network potentials and helpers:
- [`energy.behler_parrinello`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=behler_parrinello#jax_md.energy.behler_parrinello) a fixed-feature neural network architecture for molecular systems.
- [`energy.BehlerParrinelloEnergy`](jax_md/energy.py#L1918), an object-oriented wrapper for Behler-Parrinello-style models.
- [`energy.graph_network`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=graph_network#jax_md.energy.graph_network) a graph neural network designed for energy fitting.
- [`energy.nequip_neighbor_list()`](jax_md/energy.py#L2412) and [`energy.load_gnome_model_neighbor_list()`](jax_md/energy.py#L2470) wrappers for NequIP/GNoME-style neighbor-list graph models.
- [`energy.uma_neighbor_list()`](jax_md/energy.py#L2716) for UMA force-field models, including pretrained checkpoint loading, MoE heads, charge/spin/dataset embeddings, atom-reference corrections, and optional optimized kernels.
- [`energy.mace_neighbor_list()`](jax_md/energy.py#L3055) for wrapping MACE-JAX models as JAX MD neighbor-list energies.

The neural network subpackage also includes model-building pieces:
- [`jax_md.nn.MLP`](jax_md/nn.py#L49) and graph-network helpers such as [`apply_node_fn()`](jax_md/nn.py#L181), [`apply_edge_fn()`](jax_md/nn.py#L209), and [`apply_global_fn()`](jax_md/nn.py#L235).
- [`jax_md._nn.mace`](jax_md/_nn/mace) utilities for MACE featurization, JAX model construction, and torch-to-JAX conversion.
- [`jax_md._nn.uma`](jax_md/_nn/uma) modules for UMA backbones, heads, featurizers, SO3/SO2 layers, MoE routing, pretrained checkpoint conversion, and Pallas/segment-matrix-multiplication kernels.

Example:

```python
from jax_md import energy

neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
    displacement_fn, box, atoms=atomic_numbers)
neighbor = neighbor_fn.allocate(R)
params = init_fn(key, R, neighbor)
E = energy_fn(params, R, neighbor)
```

## Force Fields ([`mm_forcefields/`](jax_md/mm_forcefields))

JAX MD includes a molecular-mechanics force-field framework for bonded, nonbonded, and reactive force fields. These modules are useful when systems are specified by a topology and parameter set rather than by a hand-written potential.

The force-field framework includes:
- [`mm_forcefields.base.Topology`](jax_md/mm_forcefields/base.py#L11), bonded parameter containers, nonbonded options, combination rules, cutoff functions, and geometric helpers for angles and dihedrals.
- [`mm_forcefields.neighbor.create_neighbor_list()`](jax_md/mm_forcefields/neighbor.py#L10) and exclusion/1-4 table helpers for force-field neighbor lists.
- [`mm_forcefields.nonbonded.electrostatics`](jax_md/mm_forcefields/nonbonded/electrostatics.py) handlers for cutoff, Ewald, and PME electrostatics.

Supported force-field families include:
- ReaxFF, including force-field file parsing, reactive neighbor-list generation, bond-order terms, valence and torsion terms, hydrogen-bond terms, van der Waals terms, EEM/ACKS2 charge equilibration, and stress support.
- OPLS-AA, including topology construction, parameter validation, CHARMM-style loading helpers, and an OPLS-AA energy function.
- AMBER, including bonded terms, periodic torsions, improper torsions, CMAP terms, Lennard-Jones and Coulomb terms, softcore nonbonded terms, and SETTLE/CCMA constraints.

The OpenMM IO helpers can convert or load several system sources:
- [`load_amber_system()`](jax_md/mm_forcefields/io/openmm.py#L187)
- [`load_charmm_system()`](jax_md/mm_forcefields/io/openmm.py#L232)
- [`load_gromacs_system()`](jax_md/mm_forcefields/io/openmm.py#L264)
- [`load_parmed_system()`](jax_md/mm_forcefields/io/openmm.py#L349)
- [`load_generator_system()`](jax_md/mm_forcefields/io/openmm.py#L296)

Example:

```python
from jax_md.mm_forcefields.oplsaa import energy as opls_energy
from jax_md.mm_forcefields.nonbonded.electrostatics import PMECoulomb

coulomb = PMECoulomb(grid_size=32, alpha=0.3, r_cut=12.0)
energy_fn, neighbor_fn, displacement_fn = opls_energy.energy(
    topology, parameters, box, coulomb)
```

## Dynamics ([`simulate.py`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html), [`minimize.py`](https://jax-md.readthedocs.io/en/main/jax_md.minimize.html))

Given an energy function and a system, there are a number of dynamics that are useful to simulate. The simulation code is based on the structure of the optimizers found in JAX. In particular, each simulation function returns an initialization function and an update function. The initialization function takes a set of positions and creates the necessary dynamical state variables. The update function does a single step of dynamics to the dynamical state variables and returns an updated state.

We provide the following dynamics:
- [`simulate.nve`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html) Constant energy simulation; numerically integrates Newton's laws directly.
- [`simulate.nvt_nose_hoover`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html) Uses Nose-Hoover chains to simulate a constant temperature system.
- [`simulate.npt_nose_hoover`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html) Uses Nose-Hoover chains to simulate a system at constant pressure and temperature.
- [`simulate.nvt_langevin`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html) Simulates a system by numerically integrating the Langevin stochastic differential equation.
- [`simulate.hybrid_swap_mc`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html) Alternates NVT dynamics with Monte Carlo swapping moves to generate low energy glasses.
- [`simulate.brownian`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html) Simulates Brownian motion.
- [`simulate.temp_rescale()`](jax_md/simulate.py#L1433) performs explicit velocity rescaling.
- [`simulate.temp_berendsen()`](jax_md/simulate.py#L1503) implements the Berendsen weak-coupling thermostat.
- [`simulate.nvk()`](jax_md/simulate.py#L1569) samples the isokinetic NVK ensemble with a Gaussian thermostat.
- [`simulate.temp_csvr()`](jax_md/simulate.py#L1687) implements canonical sampling through velocity rescaling.

Simulation initializers that use momenta can initialize them from a random key or accept user-provided momenta, which is useful when restarting or coupling JAX MD to external workflows.

It is often desirable to find an energy minimum of the system. JAX MD provides:
- [`minimize.gradient_descent`](https://jax-md.readthedocs.io/en/main/jax_md.minimize.html) simple gradient descent minimization.
- [`minimize.fire_descent`](https://jax-md.readthedocs.io/en/main/jax_md.minimize.html) minimization with the fast inertial relaxation engine.
- [`minimize.fire_descent_box()`](jax_md/minimize.py#L819) FIRE relaxation of both particle positions and simulation cell.
- [`minimize.exp_preconditioner()`](jax_md/minimize.py#L270) and [`minimize.c1_preconditioner()`](jax_md/minimize.py#L421) graph-based preconditioners.
- [`minimize.estimate_exp_mu()`](jax_md/minimize.py#L450) for estimating an exponential preconditioner energy scale.
- [`minimize.precon_fire_descent()`](jax_md/minimize.py#L522) and [`minimize.precon_fire_descent_box()`](jax_md/minimize.py#L1064) preconditioned FIRE minimizers for atomic and atom-plus-cell relaxation.

Example:

```python
from jax_md import simulate
temperature = 1.0
dt = 1e-3
init, update = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, temperature)
state = init(key, R)
for _ in range(100):
  state = update(state)
R = state.position
```

## Spatial Partitioning ([`partition.py`](https://jax-md.readthedocs.io/en/main/jax_md.partition.html), [`custom_partition.py`](jax_md/custom_partition.py), [`custom_smap.py`](jax_md/custom_smap.py))

In many applications, it is useful to construct spatial partitions of particles or objects in a simulation. JAX MD supports standard cell lists and neighbor lists, plus custom multi-image neighbor lists for small periodic cells or models that need explicit periodic image shifts.

We provide the following methods:
- [`partition.cell_list`](https://jax-md.readthedocs.io/en/main/jax_md.partition.html?highlight=cell_list#jax_md.partition.cell_list) partitions objects and metadata into a grid of cells.
- [`partition.neighbor_list`](https://jax-md.readthedocs.io/en/main/jax_md.partition.html?highlight=neighbor_list#jax_md.partition.neighbor_list) constructs a set of neighbors within some cutoff distance for each object in a simulation.
- [`partition.neighbor_list_mask()`](jax_md/partition.py#L1167) masks invalid neighbor-list entries.
- [`partition.to_jraph()`](jax_md/partition.py#L1185) converts sparse neighbor lists into graph tuples for graph neural networks.
- [`partition.shift_array()`](jax_md/partition.py#L253) and [`partition.unflatten_cell_buffer()`](jax_md/partition.py#L278) expose helper operations used by downstream partitioning workflows.
- [`custom_partition.neighbor_list_multi_image()`](jax_md/custom_partition.py#L800) constructs sparse neighbor lists that retain explicit periodic image shifts.
- [`custom_partition.estimate_max_neighbors()`](jax_md/custom_partition.py#L627) and [`custom_partition.estimate_max_neighbors_from_box()`](jax_md/custom_partition.py#L688) estimate neighbor-list capacity.
- [`custom_partition.graph_featurizer()`](jax_md/custom_partition.py#L1125) creates graph features that use stored multi-image shifts.
- [`custom_smap.pair_neighbor_list_multi_image()`](jax_md/custom_smap.py#L32) maps pair energies over multi-image neighbor lists.

Cell List Example:

```python
from jax_md import partition

cell_size = 5.0
cell_list_fn = partition.cell_list(box_size, cell_size)
cell_list_data = cell_list_fn.allocate(R)
```

Neighbor List Example:

```python
from jax_md import partition

r_cutoff = 5.0
neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, r_cutoff)
neighbors = neighbor_list_fn.allocate(R) # Create a new neighbor list.

# Do some simulating....

neighbors = neighbors.update(R)  # Update the neighbor list without resizing.
if neighbors.did_buffer_overflow:  # Couldn't fit all the neighbors into the list.
  neighbors = neighbor_list_fn.allocate(R)  # So create a new neighbor list.
```

There are three different formats of neighbor list supported: `Dense`, `Sparse`, and `OrderedSparse`. `Dense` neighbor lists store neighbors in a `(particle_count, neighbors_per_particle)` array; `Sparse` neighbor lists store neighbors in a `(2, total_neighbors)` array of pairs; `OrderedSparse` neighbor lists are like `Sparse` neighbor lists, but they only store pairs such that `i < j`.

## Mapping Interactions ([`smap.py`](jax_md/smap.py))

The `smap` module maps scalar interaction functions over molecular structures. This is the layer that turns a pair, bond, angle, torsion, or triplet interaction into an energy function over a full system.

The interaction mapping helpers include:
- [`smap.bond()`](jax_md/smap.py#L188) for indexed bond interactions.
- [`smap.angle()`](jax_md/smap.py#L440) for indexed angle interactions.
- [`smap.torsion()`](jax_md/smap.py#L492) for indexed torsion interactions.
- [`smap.pair()`](jax_md/smap.py#L548) for all-pairs or species-aware pair interactions.
- [`smap.pair_neighbor_list()`](jax_md/smap.py#L856) for pair interactions evaluated over neighbor lists.
- [`smap.triplet()`](jax_md/smap.py#L982) for three-body interactions.

## Quantities ([`quantity.py`](jax_md/quantity.py), [`rigid_body.py`](jax_md/rigid_body.py))

The `quantity` module defines measurements derived from configurations, velocities, energies, and forces. These quantities are often used in analysis, thermodynamic control, and elastic or barostatted simulations.

The main quantities include:
- [`quantity.force()`](jax_md/quantity.py#L58) to transform an energy function into a force function.
- [`quantity.kinetic_energy()`](jax_md/quantity.py#L124) and [`quantity.temperature()`](jax_md/quantity.py#L162) for dynamical states.
- [`quantity.pressure()`](jax_md/quantity.py#L202) and [`quantity.stress()`](jax_md/quantity.py#L238) for thermodynamic and mechanical response.
- [`quantity.pair_correlation()`](jax_md/quantity.py#L351) and [`quantity.pair_correlation_neighbor_list()`](jax_md/quantity.py#L440) for radial distribution style measurements.
- [`quantity.volume()`](jax_md/quantity.py#L111), [`quantity.volume_fraction()`](jax_md/quantity.py#L627), and number-density/volume-fraction box helpers.
- [`quantity.bulk_modulus()`](jax_md/quantity.py#L677) and elastic-tensor helpers.
- [`quantity.gamma_from_stokes_law_3d()`](jax_md/quantity.py#L764) for hydrodynamic drag estimates.

The rigid-body module includes point-union utilities and point-based rigid-body energies:
- [`rigid_body.point_union_shape()`](jax_md/rigid_body.py#L884) builds rigid bodies from point masses.
- [`rigid_body.union_to_points()`](jax_md/rigid_body.py#L965) expands rigid bodies back into point clouds.
- [`rigid_body.point_energy()`](jax_md/rigid_body.py#L1020) and [`rigid_body.point_energy_neighbor_list()`](jax_md/rigid_body.py#L1054) evaluate pointwise energies on rigid bodies.

## Workflows and Utilities ([`a2c/`](jax_md/a2c), [`util.py`](jax_md/util.py))

JAX MD includes workflow helpers and numerical utilities that are used by examples and downstream projects.

The A2C utilities include:
- [`a2c.crystallizer_utils.get_subcells_to_crystallize()`](jax_md/a2c/crystallizer_utils.py#L54) and [`get_subcells_to_crystallize_parallel()`](jax_md/a2c/crystallizer_utils.py#L219) for subcell generation.
- [`a2c.crystallizer_utils.subcells_to_structures()`](jax_md/a2c/crystallizer_utils.py#L310) and [`valid_subcell()`](jax_md/a2c/crystallizer_utils.py#L368) for converting and filtering candidate structures.
- [`a2c.make_amorphous_utils.random_packed_structure()`](jax_md/a2c/make_amorphous_utils.py#L55) for amorphous structure generation.

General utilities include:
- [`util.high_precision_sum()`](jax_md/util.py#L91) for numerically stable reductions.
- [`util.safe_norm()`](jax_md/util.py#L128), [`util.safe_arccos()`](jax_md/util.py#L146), and [`util.normalize()`](jax_md/util.py#L160) for guarded vector operations.
- [`util.x64_enabled()`](jax_md/util.py#L115) for checking whether JAX double precision is enabled.
