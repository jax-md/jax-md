# JAX, M.D.

# Accelerated, Differentiable, Molecular Dynamics

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
a broader set of simulations.

### Getting Started

To get started playing around with JAX, MD check out the following colab
notebooks on Google Cloud without needing to install anything.

- [Minimization](https://colab.research.google.com/github/google/jax-md/blob/master/notebooks/minimization.ipynb)
- [NVE Simulation](https://colab.research.google.com/github/google/jax-md/blob/master/notebooks/nve_simulation.ipynb)
- [NVT Simulation](https://colab.research.google.com/github/google/jax-md/blob/master/notebooks/nvt_simulation.ipynb)

Alternatively, you can install JAX, MD by first following the [JAX's](https://www.github.com/google/jax/)
installation instructions. Then installing JAX, MD should be as easy as,

```
git clone https://github.com/google/jax-md
pip install -e jax-md
```

# Overview

There are several aspects of the library.

## Spaces

In general we must have a way of computing the pairwise distance between atoms.
We must also have efficient strategies for moving atoms in some space that may
or may not be globally isomorphic to R^N. For example, periodic boundary
conditions are commonplace in simulations and must be respected. This part of
the code implements these functions.

Example:

```python
box_size = 25.0
displacement_fn, shift_fn = periodic(box_size)
```

## Potential Energy

In the simplest case, molecular dynamics calculations are often based on a pair
potential that is defined by a user. This then is used to compute a total energy
whose negative gradient gives forces. One of the very nice things about JAX is
that we get forces for free! The second part of the code is devoted to computing
energies. We provide a Soft Sphere potential and a Lennard Jones potential. We
also offer a convenience wrapper to compute the force.

Example:

```python
N = 1000
spatial_dimension = 2
key = random.PRNGKey(0)
R = random.uniform(key, (N, spatial_dimension), minval=0.0, maxval=1.0)
energy_fn = energy.lennard_jones_pairwise(displacement)
print('E = {}'.format(energy(R)))
force_fn = quantity.force(energy_fn)
print('Total Squared Force = {}'.format(np.sum(force_fn(R) ** 2)))
```

## Dynamics

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
include the FIRE algorithm which often sees significantly faster convergence.

Moreover a common experiment to run in the context of molecular dynamics is to
simulate a system with a fixed volume and temperature. We provide the function
`nvt_nose_hoover` to do this.

Example:

```python
temperature = 1.0
dt = 1e-3
init, update = nvt_nose_hoover(energy, wrap_fn, dt, temperature)
state = init(R)
for _ in range(100):
  state = update(state)
R = state.positions
```

