from jax_md import space, energy
import itertools
import jax.numpy as jnp
from jax import random
from jax import jit, grad

box = 2.0
displacement, _ = space.periodic(box)
tiled_displacement, _ = space.periodic(box * 3)

R = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [1.0, 1.0, 0.0], [0.0, 0.0, 1.75]])

dx = [list(range(tiles)) for tiles in (3, 3, 3)]
shift = jnp.array(list(itertools.product(*dx)))

R_tiled = R[:, None, :] + shift[None, :, :] * box
R_tiled = jnp.reshape(R_tiled, (-1, 3))

energy_fn = energy.lennard_jones_pair(displacement)
tiled_energy_fn = energy.lennard_jones_pair(tiled_displacement)

print(energy_fn(R))
print(tiled_energy_fn(R_tiled) / 27)

print(grad(energy_fn)(R))
print(grad(tiled_energy_fn)(R_tiled)[::27])
print(jnp.max(jnp.abs(grad(energy_fn)(R) - grad(tiled_energy_fn)(R_tiled)[::27])))

@jit
def do_thing(R):
  energy_fn = energy.lennard_jones_pair(displacement, sigma=sigma)
  return energy_fn(R)

sigma = 1.0
#do_thing(R)
