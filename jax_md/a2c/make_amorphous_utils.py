# Copyright 2024 Google LLC
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

"""Utilities for generating amorphous structures."""

import itertools
import json
from typing import Any, Sequence
from jax import random, jit
import jax.numpy as jnp
from jax_md import energy
from jax_md import minimize
from jax_md import space
from jax_md import util as jax_md_util
import numpy as onp
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure


def get_diameter(composition) -> float:
  """Auto diamater tries to calculate min separation from radii."""
  if len(composition.elements) > 1:
    diameter = onp.array(list(itertools.combinations(
            [e.average_ionic_radius for e in composition.elements], 2)))
    diameter = float(min(diameter.sum(axis=1)))
  else:
    elem = composition.elements[0]
    if elem.is_metal:
      diameter = float(elem.metallic_radius)*2
    else:
      diameter = (
          float(elem.atomic_radius)*2 if elem.atomic_radius
          else float(elem.average_ionic_radius)*2
          )
  return diameter


def random_packed_structure(
    composition: Composition,
    lattice: Sequence[Sequence[onp.float32]],
    seed: int = 42,
    diameter: float | None = None,
    auto_diameter: bool = False,
    max_iter: int = 30,
    distance_tolerance: float = 0.0001,
) -> Structure:
  """Generates random packed structure in a box, optionally minimizing overlap.

  Args:
    composition: target formula with actual atom counts: e.g. Fe80B20
    lattice: dimensions of the triclinic box in angstrom. need not be cubic.
    seed: random seed
    diameter: interatomic distances below this value considered overlapping
    auto_diameter: if True, attempts to calcualte diameter for soft-sphere pot.
    max_iter: number of fire desent steps applied in overlap minimization
    distance_tolerance: used to mask out identical atoms in the distance matrix

  Returns:
    A structure object
  """
  element_symbols, element_counts = zip(*composition.as_dict().items())
  element_counts = [int(i) for i in element_counts]

  species = []
  for i, el in enumerate(element_symbols):
    for _ in range(element_counts[i]):
      species.append(el)
  lattice = jnp.array(lattice)
  key = random.PRNGKey(seed)
  _, split = random.split(key)
  R = random.uniform(split, (sum(element_counts), 3), dtype=onp.float32)

  def min_distance(s):
    return onp.where(
        s.distance_matrix < distance_tolerance, 10, s.distance_matrix
    ).min()

  if auto_diameter:
    diameter = get_diameter(composition)
  print('Using random pack diameter of ', str(diameter))

  if diameter is not None:
    print('Reduce atom overlap using the soft_sphere potential')
    R_cart = jnp.dot(R, lattice)
    box = jnp.array(lattice.T)
    displacement, shift = space.periodic_general(box,
                                                 fractional_coordinates=False,
                                                 wrapped=False)
    diameter = jax_md_util.maybe_downcast(diameter)
    energy_fn = energy.soft_sphere_pair(displacement, sigma=diameter)  # pytype: disable=wrong-arg-types
    fire_init, fire_apply = minimize.fire_descent(energy_fn, shift)
    fire_apply = jit(fire_apply)
    fire_state = fire_init(R_cart)

    for _ in range(max_iter):
      fire_state = fire_apply(fire_state)
      # We break the loop early if desired min distance is reasonble.
      s = Structure(lattice, species, fire_state.position,
                    coords_are_cartesian=True)  # pytype: disable=wrong-arg-types
      if min_distance(s) > diameter * 0.95:
        break
    R = fire_state.position

  template = """melt {composition}
  1.0
      {lattice}
      {element_symbols}
      {element_counts}
  Cartesian
  """.format(
      composition=composition,
      lattice=onp.array_str(lattice).replace('[', '').replace(']', ''),
      element_symbols=' '.join(element_symbols),
      element_counts=' '.join([str(i) for i in element_counts]),
  )
  b = '\n'.join([' '.join([str(i) for i in y.tolist()]) for y in R])
  template += b
  return Structure.from_str(template, fmt='poscar')

