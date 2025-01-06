<h1 align='center'>JAX, M,D,U</h1>
<h2 align='center'>Unit-aware, Accelerated, Differentiable, Molecular Dynamics modeling in JAX.</h2>

Molecular dynamics is a workhorse of modern computational condensed matter physics. It is frequently used to simulate materials to observe how small scale interactions can give rise to complex large-scale phenomenology. Most molecular dynamics packages (e.g. HOOMD Blue or LAMMPS) are complicated, specialized pieces of code that are many thousands of lines long. They typically involve significant code duplication to allow for running simulations on CPU and GPU. Additionally, large amounts of code is often devoted to taking derivatives of quantities to compute functions of interest (e.g. gradients of energies to compute forces).

However, recent work in machine learning has led to significant software developments that might make it possible to write more concise molecular dynamics simulations that offer a range of benefits. Here we target JAX, which allows us to write python code that gets compiled to XLA and allows us to run on CPU, GPU, or TPU. Moreover, JAX allows us to take derivatives of python code. Thus, not only is this molecular dynamics simulation automatically hardware accelerated, it is also __end-to-end__ differentiable. This should allow for some interesting experiments that we're excited to explore.

JAX, MD is a research project that is currently under development. Expect sharp edges and possibly some API breaking changes as we continue to support a broader set of simulations. JAX MD is a functional and data driven library. Data is stored in arrays or tuples of arrays and functions transform data from one state to another.

🚧 Please note that only some APIs have been modified to support [`brainunit`](https://github.com/chaobrain/brainunit), because it is not easy to modify all the complex codes of JAX, M,D. 🚧


## Getting Started

You can install JAX MD with pip,
```bash
pip install git+https://github.com/routhleck/jax-md.git
```

## Quickstart(Minimization)
```python
import jax.numpy as jnp
import brainunit as u
import brainstate as bst
import matplotlib.pyplot as plt

def plot(x, y, *args):
  plt.plot(x, y, *args, linewidth=3)
  plt.gca().set_facecolor([1, 1, 1])

# Energy and Automatic Differentiation
@u.assign_units(r=u.angstrom, result=u.eV)
def soft_sphere(r):
  return jnp.where(r < 1, 
                   1/3 * (1 - r) ** 3,
                   0.)

print(soft_sphere(0.5 * u.angstrom))

r = u.math.linspace(0 * u.angstrom, 2. * u.angstrom, 200)
plot(r, soft_sphere(r))

from brainunit.autograd import grad

# We can compute its derivative automatically
du_dr = grad(soft_sphere)

print(du_dr(0.5 * u.angstrom))

# Randomly Initialize a System
from jax import random

key = random.PRNGKey(1)

particle_count = 128
number_density = 1.2 / u.angstrom ** 2
dim = 2

from jax_md import quantity

# number_density = N / V
box_size = quantity.box_size_at_number_density(particle_count = particle_count, 
                                               number_density = number_density, 
                                               spatial_dimension = dim)

R = bst.random._random_for_unit.uniform_for_unit(key, (particle_count, dim), minval=0*u.angstrom, maxval=box_size)

from jax_md.colab_tools import renderer
renderer.render(box_size, renderer.Disk(R), resolution=(512, 512))

# Displacements and Distances
from jax_md import space

displacement, shift = space.periodic(box_size)

print(displacement(R[0], R[1]))

metric = space.metric(displacement)

print(metric(R[0], R[1]))

# Compute distances between pairs of points
v_displacement = space.map_product(displacement)
v_metric = space.map_product(metric)

print(v_metric(R[:3], R[:3]))

# Total Energy of a System
def energy_fn(R):
  dr = v_metric(R, R)
  return 0.5 * u.math.sum(soft_sphere(dr))

print(energy_fn(R))

print(grad(energy_fn)(R).shape)

# Minimization
from jax_md import minimize

init_fn, apply_fn = minimize.fire_descent(energy_fn, shift)

state = init_fn(R)

trajectory = []

while u.math.max(u.math.abs(state.force)) > 1e-4 * u.IMF:
  state = apply_fn(state)
  trajectory += [state.position]


trajectory = u.math.stack(trajectory)

renderer.render(box_size,
                renderer.Disk(trajectory),
                resolution=(512, 512))
```

More examples about unit-aware features in jax-md please see [unit-aware examples](https://github.com/Routhleck/jax-md/tree/main/examples-with-unit).
