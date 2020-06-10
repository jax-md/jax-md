import jax.numpy as np
from jax import random
from jax_md import space, energy

key = random.PRNGKey(0)

R = random.uniform(key, (32, 2), dtype=np.float32)

d, _ = space.free()

init_fn, model_fn = energy.graph_network(d, 0.2)

params = init_fn(key, R)

neighbor_fn, _, model_neighbor_fn = energy.graph_network_neighbor_list(d, 1.0, 0.2, 0.05)

neighbors = neighbor_fn(R)

print(neighbors.idx.shape)

print(model_fn(params, R).dtype,
      model_neighbor_fn(params, R, neighbors).dtype)
