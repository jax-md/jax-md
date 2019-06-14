# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example showing the simple minimization of a two-dimensional system."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

from jax import random
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import jit
from jax_md import space, energy, minimize, quantity, smap
from jax_md.util import f32, i32

def main(unused_argv):
  key = random.PRNGKey(0)

  # Setup some variables describing the system.
  N = 500
  dimension = 2
  box_size = f32(25.0)

  # Create helper functions to define a periodic box of some size.
  displacement, shift = space.periodic(box_size)

  # Use JAX's random number generator to generate random initial positions.
  key, split = random.split(key)
  R = random.uniform(
    split, (N, dimension), minval=0.0, maxval=box_size, dtype=f32)

  # The system ought to be a 50:50 mixture of two types of particles, one
  # large and one small.
  sigma = np.array([[1.0, 1.2], [1.2, 1.4]], dtype=f32)
  N_2 = int(N / 2)
  species = np.array([0] * N_2 + [1] * N_2, dtype=i32)

  # Create an energy function.
  energy_fn = energy.soft_sphere_pair(displacement, species, sigma)
  force_fn = quantity.force(energy_fn)

  # Create a minimizer.
  init_fn, apply_fn = minimize.fire_descent(energy_fn, shift)
  opt_state = init_fn(R)

  # Minimize the system.
  minimize_steps = 50
  print_every = 10

  print('Minimizing.')
  print('Step\tEnergy\tMax Force')
  print('-----------------------------------')
  for step in range(minimize_steps):
    opt_state = apply_fn(opt_state)

    if step % print_every == 0:
      R = opt_state.position
      print('{:.2f}\t{:.2f}\t{:.2f}'.format(
          step, energy_fn(R), np.max(force_fn(R))))


if __name__ == '__main__':
  app.run(main)
