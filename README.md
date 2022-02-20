# JAX, M.D.

### Accelerated, Differentiable, Molecular Dynamics
[**Quickstart**](#getting-started)
| [**Reference docs**](https://jax-md.readthedocs.io/en/main/)
| [**Paper**](https://arxiv.org/pdf/1912.04232.pdf)
| [**NeurIPS 2020**](https://neurips.cc/virtual/2020/public/poster_83d3d4b6c9579515e1679aca8cbc8033.html)

![Build Status](https://github.com/google/jax-md/workflows/Build/badge.svg?branch=main) [![Coverage](https://codecov.io/gh/google/jax-md/branch/main/graph/badge.svg?token=JYQpbNyICv)](https://codecov.io/gh/google/jax-md)
 [![PyPI](https://img.shields.io/pypi/v/jax-md)](https://pypi.org/project/jax-md/) [![PyPI - License](https://img.shields.io/pypi/l/jax_md)](https://github.com/google/jax-md/blob/main/LICENSE)

Molecular dynamics is a workhorse of modern computational condensed matter
physics. It is frequently used to simulate materials to observe how small scale
interactions can give rise to complex large-scale phenomenology. Most molecular
dynamics packages (e.g. HOOMD Blue or LAMMPS) are complicated, specialized
pieces of code that are many thousands of lines long. They typically involve
significant code duplication to allow for running simulations on CPU and GPU.
Additionally, large amounts of code is often devoted to taking derivatives
of quantities to compute functions of interest (e.g. gradients of energies
to compute forces).

However, recent work in machine learning has led to significant software
developments that might make it possible to write more concise
molecular dynamics simulations that offer a range of benefits. Here we target
JAX, which allows us to write python code that gets compiled to XLA and allows
us to run on CPU, GPU, or TPU. Moreover, JAX allows us to take derivatives of
python code. Thus, not only is this molecular dynamics simulation automatically
hardware accelerated, it is also __end-to-end__ differentiable. This should
allow for some interesting experiments that we're excited to explore.

JAX, MD is a research project that is currently under development. Expect
sharp edges and possibly some API breaking changes as we continue to support
a broader set of simulations. JAX MD is a functional and data driven library.
Data is stored in arrays or tuples of arrays and functions transform data from
one state to another.

### Getting Started

For a video introducing JAX MD along with a [demo](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/talk_demo.ipynb), check out this talk from the Physics meets Machine Learning series:

[![Science Meets ML Talk](https://img.youtube.com/vi/Bkm8tGET7-w/0.jpg)](https://www.youtube.com/watch?v=Bkm8tGET7-w)

To get started playing around with JAX MD check out the following colab notebooks on Google Cloud without needing to install anything. For a very simple introduction, I would recommend the Minimization example. For an example of a bunch of the features of JAX MD, check out the JAX MD cookbook.

- [JAX MD Cookbook](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/jax_md_cookbook.ipynb)
- [Minimization](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/minimization.ipynb)
- [NVE Simulation](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/nve_simulation.ipynb)
- [NVT Simulation](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/nvt_simulation.ipynb)
- [NPT Simulation](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/npt_simulation.ipynb)
- [NVE with Neighbor Lists](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/nve_neighbor_list.ipynb)
- [Custom Potentials](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/customizing_potentials_cookbook.ipynb)
- [Neural Network Potentials](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/neural_networks.ipynb)
- [Flocking](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/flocking.ipynb)
- [Meta Optimization](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/meta_optimization.ipynb)
- [Swap Monte Carlo (Cargese Summer School)](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/cargese_swap_mc.ipynb)
- [Implicit Differentiation](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/implicit_differentiation.ipynb)
- [Athermal Linear Elasticity](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/athermal_linear_elasticity.ipynb)
- [Smash a Sand Castle](https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/sand_castle.ipynb)

You can install JAX MD locally with pip,
```
pip install jax-md --upgrade
```
If you want to build the latest version then you can grab the most recent version from head,
```
git clone https://github.com/google/jax-md
pip install -e jax-md
```

# Overview

We now summarize the main components of the library.

## Spaces ([`space.py`](https://jax-md.readthedocs.io/en/latest/jax_md.space.html))

In general we must have a way of computing the pairwise distance between atoms.
We must also have efficient strategies for moving atoms in some space that may
or may not be globally isomorphic to R^N. For example, periodic boundary
conditions are commonplace in simulations and must be respected. Spaces are defined as a pair of functions, `(displacement_fn, shift_fn)`. Given two points `displacement_fn(R_1, R_2)` computes the displacement vector between the two points. If you would like to compute displacement vectors between all pairs of points in a given `(N, dim)` matrix the function `space.map_product` appropriately vectorizes `displacement_fn`. It is often useful to define a metric instead of a displcement function in which case you can use the helper function `space.metric` to convert a displacement function to a metric function. Given a point and a shift `shift_fn(R, dR)` displaces the point `R` by an amount `dR`.

The following spaces are currently supported:
- `space.free()` specifies a space with free boundary conditions.
- `space.periodic(box_size)` specifies a space with periodic boundary conditions of side length `box_size`.
- `space.periodic_general(box)` specifies a space as a periodic parellelopiped formed by transforming the unit cube by an affine transformation `box`.

Example:

```python
from jax_md import space
box_size = 25.0
displacement_fn, shift_fn = space.periodic(box_size)
```

## Potential Energy ([`energy.py`](https://jax-md.readthedocs.io/en/latest/jax_md.energy.html))

In the simplest case, molecular dynamics calculations are often based on a pair
potential that is defined by a user. This then is used to compute a total energy
whose negative gradient gives forces. One of the very nice things about JAX is
that we get forces for free! The second part of the code is devoted to computing
energies.

We provide the following classical potentials:
- `energy.soft_sphere` a soft sphere whose energy incrases as the overlap of the spheres to some power, `alpha`.
- `energy.lennard_jones` a standard 12-6 lennard-jones potential.
- `energy.morse` a morse potential.
- `energy.eam` embedded atom model potential with ability to load parameters from LAMMPS files.
- `energy.stillinger_weber` used to model Silicon-like systems.
- `energy.bks` Beest-Kramer-van Santen potential used to model silica.
- `energy.gupta` used to model gold nanoclusters.

We also provide the following neural network potentials:
- `energy.behler_parrinello` a widely used fixed-feature neural network architecture for molecular systems.
- `energy.graph_network` a deep graph neural network designed for energy fitting.

For finite-ranged potentials it is often useful to consider only interactions within a certain neighborhood. We include the `_neighbor_list` modifier to the above potentials that uses a list of neighbors (see below) for optimization.

Example:

```python
import jax.numpy as np
from jax import random
from jax_md import energy, quantity
N = 1000
spatial_dimension = 2
key = random.PRNGKey(0)
R = random.uniform(key, (N, spatial_dimension), minval=0.0, maxval=1.0)
energy_fn = energy.lennard_jones_pair(displacement_fn)
print('E = {}'.format(energy_fn(R)))
force_fn = quantity.force(energy_fn)
print('Total Squared Force = {}'.format(np.sum(force_fn(R) ** 2)))
```

## Dynamics ([`simulate.py`](https://jax-md.readthedocs.io/en/latest/jax_md.simulate.html), [`minimize.py`](https://jax-md.readthedocs.io/en/latest/jax_md.minimize.html))

Given an energy function and a system, there are a number of dynamics are useful
to simulate. The simulation code is based on the structure of the optimizers
found in JAX. In particular, each simulation function returns an initialization
function and an update function. The initialization function takes a set of
positions and creates the necessary dynamical state variables. The update
function does a single step of dynamics to the dynamical state variables and
returns an updated state.

We include a several different kinds of dynamics. However, there is certainly room
to add more for e.g. constaint strain simulations.

It is often desirable to find an energy minimum of the system. We provide
two methods to do this. We provide simple gradient descent minimization. This is
mostly for pedagogical purposes, since it often performs poorly. We additionally
include the FIRE algorithm which often sees significantly faster convergence. Moreover a common experiment to run in the context of molecular dynamics is to simulate a system with a fixed volume and temperature.

We provide the following dynamics:
- `simulate.nve` Constant energy simulation; numerically integrates Newton's laws directly.
- `simulate.nvt_nose_hoover` Uses Nose-Hoover chain to simulate a constant temperature system.
- `simulate.npt_nose_hoover` Uses Nose-Hoover chain to simulate a system at constant pressure and temperature.
- `simulate.nvt_langevin` Simulates a system by numerically integrating the Langevin stochistic differential equation.
- `simulate.hybrid_swap_mc` Alternates NVT dynamics with Monte-Carlo swapping moves to generate low energy glasses.
- `simulate.brownian` Simulates brownian motion.
- `minimize.gradient_descent` Mimimizes a system using gradient descent.
- `minimize.fire_descent` Minimizes a system using the fast inertial relaxation engine.

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

## Spatial Partitioning ([`partition.py`](https://jax-md.readthedocs.io/en/latest/jax_md.partition.html))

In many applications, it is useful to construct spatial partitions of particles / objects in a simulation.

We provide the following methods:
- `partition.cell_list` Partitions objects (and metadata) into a grid of cells.
- `partition.neighbor_list` Constructs a set of neighbors within some cutoff distance for each object in a simulation.

Cell List Example:
```python
from jax_md import partition

cell_size = 5.0
capacity = 10
cell_list_fn = partition.cell_list(box_size, cell_size, capacity)
cell_list_data = cell_list_fn(R)
```

Neighbor List Example:
```python
from jax_md import partition

neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, cell_size)
neighbors = neighbor_list_fn.allocate(R) # Create a new neighbor list.

# Do some simulating....

neighbors = neighbors.update(R)  # Update the neighbor list without resizing.
if neighbors.did_buffer_overflow:  # Couldn't fit all the neighbors into the list.
  neighbors = neighbor_list_fn.allocate(R)  # So create a new neighbor list.
```

There are three different formats of neighbor list supported: `Dense`, `Sparse`, and `OrderedSparse`. `Dense` neighbor lists store neighbors in an `(particle_count, neighbors_per_particle)` array, `Sparse` neighbor lists store neighbors in a `(2, total_neighbors)` array of pairs, `OrderedSparse` neighbor lists are like `Sparse` neighbor lists, but they only store pairs such that `i < j`.

# Development

JAX MD is under active development. We have very limited development resources and so we typically focus on adding features that will have high impact to researchers using JAX MD (including us). Please don't hesitate to open feature requests to help us guide development. We more than welcome contributions!

## Technical gotchas

### GPU

You must follow [JAX's](https://www.github.com/google/jax/) GPU installation instructions to enable GPU support.


### 64-bit precision
To enable 64-bit precision, set the respective JAX flag _before_ importing `jax_md` (see the JAX [guide](https://colab.research.google.com/github/google/jax/blob/main/notebooks/Common_Gotchas_in_JAX.ipynb#scrollTo=YTktlwTTMgFl)), for example:

```python
from jax.config import config
config.update("jax_enable_x64", True)
```

# Publications

JAX MD has been used in the following publications. If you don't see your paper on the list, but you used JAX MD let us know and we'll add it to the list!

1. [Improved tracking of particles with highly correlated motion.](https://arxiv.org/abs/2201.09752)<br> E. M. King, Z. Wang, D. A. Weitz, F. Spaepen, and M. P. Brenner
2. [Optimal Control of Nonequilibrium Systems Through Automatic Differentiation.](https://arxiv.org/abs/2201.00098)<br> M. C. Engel, J. A. Smith, and M. P. Brenner
3. [Graph Neural Networks Accelerated Molecular Dynamics.](https://arxiv.org/abs/2112.03383)<br> Z. Li, K. Meidani, P. Yadav, and A. B. Farimani
4. [Gradients are Not All You Need.](https://arxiv.org/abs/2111.05803)<br> L. Metz, C. D. Freeman, S. S. Schoenholz, and T. Kachman
5. [Lagrangian Neural Network with Differential Symmetries and Relational Inductive Bias.](https://arxiv.org/abs/2110.03266)<br> R. Bhattoo, S. Ranu, and N. M. A. Krishnan
6. [Efficient and Modular Implicit Differentiation.](https://arxiv.org/abs/2105.15183)<br> M. Blondel, Q. Berthet, M. Cuturi, R. Frostig, S. Hoyer, F. Llinares-López, F. Pedregosa, and J.-P. Vert
7. [Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting.<br>(Nature Communications 2021)](https://www.nature.com/articles/s41467-021-27241-4)<br> S. Thaler and J. Zavadlav
8. [Learn2Hop: Learned Optimization on Rough Landscapes. (ICML 2021)](http://proceedings.mlr.press/v139/merchant21a.html)<br> A. Merchant, L. Metz, S. S. Schoenholz, and E. D. Cubuk
9. [Designing self-assembling kinetics with differentiable statistical physics models. (PNAS 2021)](https://www.pnas.org/content/118/10/e2024083118.short)<br> C. P. Goodrich, E. M. King, S. S. Schoenholz, E. D. Cubuk, and  M. P. Brenner

# Citation

If you use the code in a publication, please cite the repo using the .bib,

```
@inproceedings{jaxmd2020,
 author = {Schoenholz, Samuel S. and Cubuk, Ekin D.},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 title = {JAX M.D. A Framework for Differentiable Physics},
 url = {https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
