# Copyright 2022 Google LLC
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


from typing import Callable, Tuple, Dict, Any, Optional

import numpy as onp

import jax
from jax import vmap, jit
import jax.numpy as jnp

from jax_md import space, dataclasses, quantity, partition, smap, util
import haiku as hk

from collections import namedtuple
from functools import partial, reduce
from jax.tree_util import tree_map
from jax import ops

import jraph


# Typing


Array = util.Array
f32 = util.f32
f64 = util.f64

InitFn = Callable[..., Array]
CallFn = Callable[..., Array]

DisplacementOrMetricFn = space.DisplacementOrMetricFn
DisplacementFn = space.DisplacementFn
NeighborList = partition.NeighborList

"""
Our neural network force field is based on the Behler-Parrinello neural network
architecture (BP-NN) [1]. The BP-NN architecture uses a relatively simple,
fully connected neural network to predict the local energy for each atom. Then
the total energy is the sum of local energies due to each atom. Atoms of the
same type use the same NN to predict energy.

Each atomic NN is applied to hand-crafted features called symmetry functions.
There are two kinds of symmetry functions: radial and angular. Radial symmetry
functions represent information about two-body interactions of the central
atom, whereas angular symmetry functions represent information about three-body
interactions. Below we implement radial and angular symmetry functions for
arbitrary number of types of atoms (Note that most applications of BP-NN limit
their systems to 1 to 3 types of atoms). We also present a convenience wrapper
that returns radial and angular symmetry functions with symmetry function
parameters that should work reasonably for most systems (the symmetry functions
are taken from reference [2]). Please see references [1, 2] for details about
how the BP-NN works.

[1] Behler, Jörg, and Michele Parrinello. "Generalized neural-network
representation of high-dimensional potential-energy surfaces." Physical
Review Letters 98.14 (2007): 146401.

[2] Artrith, Nongnuch, Björn Hiller, and Jörg Behler. "Neural network
potentials for metals and oxides–First applications to copper clusters at zinc
oxide."
Physica Status Solidi (b) 250.6 (2013): 1191-1203.
"""

def _cutoff_fn(dr: Array,
                                 cutoff_distance: float=8.0) -> Array:
  """Function of pairwise distance that smoothly goes to zero at the cutoff."""
  # Also returns zero if the pairwise distance is zero,
  # to prevent a particle from interacting with itself.
  return jnp.where((dr < cutoff_distance) & (dr > 1e-7),
                  0.5 * (jnp.cos(jnp.pi * dr / cutoff_distance) + 1), 0)


def radial_symmetry_functions(displacement_or_metric: DisplacementOrMetricFn,
                              species: Optional[Array],
                              etas: Array,
                              cutoff_distance: float
                              ) -> Callable[[Array], Array]:
  """Returns a function that computes radial symmetry functions.


  Args:
    displacement: A function that produces an `[N_atoms, M_atoms,
      spatial_dimension]` of particle displacements from particle positions
      specified as an `[N_atoms, spatial_dimension] and `[M_atoms,
      spatial_dimension]` respectively.
    species: An `[N_atoms]` that contains the species of each particle.
    etas: List of radial symmetry function parameters that control the spatial
      extension.
    cutoff_distance: Neighbors whose distance is larger than `cutoff_distance` do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.

  Returns:
    A function that computes the radial symmetry function from input `[N_atoms,
    spatial_dimension]` and returns `[N_atoms, N_etas * N_types]` where `N_etas`
    is the number of eta parameters, `N_types` is the number of types of
    particles in the system.
  """
  metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

  def radial_fn(eta: Array, dr: Array) -> Array:
    return jnp.exp(-eta * dr**2) * _cutoff_fn(dr, cutoff_distance)
  radial_fn = vmap(radial_fn, (0, None))

  if species is None:
    def compute_fn(R: Array, **kwargs) -> Array:
      _metric = partial(metric, **kwargs)
      _metric = space.map_product(_metric)
      return util.high_precision_sum(radial_fn(etas, _metric(R, R)), axis=1).T
  elif isinstance(species, jnp.ndarray):
    species = onp.array(species)
    def compute_fn(R: Array, **kwargs) -> Array:
      _metric = partial(metric, **kwargs)
      _metric = space.map_product(_metric)
      def return_radial(atom_type):
        """Returns the radial symmetry functions for neighbor type atom_type."""
        R_neigh = R[species == atom_type, :]
        dr = _metric(R, R_neigh)
        return util.high_precision_sum(radial_fn(etas, dr), axis=1).T
      return jnp.hstack([return_radial(atom_type) for
                        atom_type in onp.unique(species)])
  return compute_fn


def radial_symmetry_functions_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    species: Array,
    etas: Array,
    cutoff_distance: float) -> Callable[[Array, NeighborList], Array]:
  """Returns a function that computes radial symmetry functions.


  Args:
    displacement: A function that produces an `[N_atoms, M_atoms,
      spatial_dimension]` of particle displacements from particle positions
      specified as an `[N_atoms, spatial_dimension] and `[M_atoms,
      spatial_dimension]` respectively.
    species: An `[N_atoms]` that contains the species of each particle.
    etas: List of radial symmetry function parameters that control the spatial
      extension.
    cutoff_distance: Neighbors whose distance is larger than `cutoff_distance` do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.

  Returns:
    A function that computes the radial symmetry function from input `[N_atoms,
    spatial_dimension]` and returns `[N_atoms, N_etas * N_types]` where `N_etas`
    is the number of eta parameters, `N_types` is the number of types of
    particles in the system.
  """
  metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

  def radial_fn(eta, dr):
    return jnp.exp(-eta * dr**2) * _cutoff_fn(dr, cutoff_distance)
  radial_fn = vmap(radial_fn, (0, None))

  def sym_fn(R: Array, neighbor: NeighborList, mask: Array=None,
             **kwargs) -> Array:
    _metric = partial(metric, **kwargs)
    if neighbor.format is partition.Dense:
      _metric = space.map_neighbor(_metric)
      R_neigh = R[neighbor.idx]
      mask = True if mask is None else mask[neighbor.idx]
      mask = (neighbor.idx < R.shape[0])[None, :, :] & mask
      dr = _metric(R, R_neigh)
      return util.high_precision_sum(radial_fn(etas, dr) * mask, axis=2).T
    elif neighbor.format is partition.Sparse:
      _metric = space.map_bond(_metric)
      dr = _metric(R[neighbor.idx[0]], R[neighbor.idx[1]])
      radial = radial_fn(etas, dr).T
      N = R.shape[0]
      mask = True if mask is None else mask[neighbor.idx[1]]
      mask = (neighbor.idx[0] < N) & mask
      return ops.segment_sum(radial * mask[:, None], neighbor.idx[0], N)
    else:
      raise ValueError()

  if species is None:
    def compute_fn(R: Array, neighbor: NeighborList, **kwargs) -> Array:
      return sym_fn(R, neighbor, **kwargs)
    return compute_fn

  def compute_fn(R: Array, neighbor: NeighborList, **kwargs) -> Array:
    _metric = partial(metric, **kwargs)
    def return_radial(atom_type):
      """Returns the radial symmetry functions for neighbor type atom_type."""
      return sym_fn(R, neighbor, species==atom_type, **kwargs)
    return jnp.hstack([return_radial(atom_type) for
                     atom_type in onp.unique(species)])

  return compute_fn


def single_pair_angular_symmetry_function(dR12: Array,
                                          dR13: Array,
                                          eta: Array,
                                          lam: Array,
                                          zeta: Array,
                                          cutoff_distance: float
                                          ) -> Array:
  """Computes the angular symmetry function due to one pair of neighbors."""

  dR23 = dR12 - dR13
  dr12_2 = space.square_distance(dR12)
  dr13_2 = space.square_distance(dR13)
  dr23_2 = space.square_distance(dR23)
  dr12 = space.distance(dR12)
  dr13 = space.distance(dR13)
  dr23 = space.distance(dR23)
  triplet_squared_distances = dr12_2 + dr13_2 + dr23_2
  triplet_cutoff = reduce(
      lambda x, y: x * _cutoff_fn(y, cutoff_distance),
      [dr12, dr13, dr23], 1.0)
  z = zeta
  result = 2.0 ** (1.0 - zeta) * ((
    1.0 + lam * quantity.cosine_angle_between_two_vectors(dR12, dR13)) ** z *
    jnp.exp(-eta * triplet_squared_distances) * triplet_cutoff)
  return result


def angular_symmetry_functions(displacement: DisplacementFn,
                               species: Array,
                               etas: Array,
                               lambdas: Array,
                               zetas: Array,
                               cutoff_distance: float
                               ) -> Callable[[Array], Array]:
  """Returns a function that computes angular symmetry functions.

  Args:
    displacement: A function that produces an `[N_atoms, M_atoms,
      spatial_dimension]` of particle displacements from particle positions
      specified as an `[N_atoms, spatial_dimension] and `[M_atoms,
      spatial_dimension]` respectively.
    species: An `[N_atoms]` that contains the species of each particle.
    eta: Parameter of angular symmetry function that controls the spatial
      extension.
    lam:
    zeta:
    cutoff_distance: Neighbors whose distance is larger than `cutoff_distance` do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.
  Returns:
    A function that computes the angular symmetry function from input `[N_atoms,
    spatial_dimension]` and returns `[N_atoms, N_types * (N_types + 1) / 2]`
    where `N_types` is the number of types of particles in the system.
  """

  _angular_fn = vmap(single_pair_angular_symmetry_function,
                     (None, None, 0, 0, 0, None))

  _batched_angular_fn = lambda dR12, dR13: _angular_fn(dR12,
                                                       dR13,
                                                       etas,
                                                       lambdas,
                                                       zetas,
                                                       cutoff_distance)
  _all_pairs_angular = vmap(
      vmap(vmap(_batched_angular_fn, (0, None)), (None, 0)), 0)

  if species is None:
   def compute_fn(R, **kwargs):
     D_fn = partial(displacement, **kwargs)
     D_fn = space.map_product(D_fn)
     dR = D_fn(R, R)
     return jnp.sum(_all_pairs_angular(dR, dR), axis=[1, 2])
   return compute_fn

  if isinstance(species, jnp.ndarray):
    species = onp.array(species)

  def compute_fn(R, **kwargs):
    atom_types = onp.unique(species)
    D_fn = partial(displacement, **kwargs)
    D_fn = space.map_product(D_fn)
    D_different_types = [D_fn(R[species == s, :], R) for s in atom_types]
    out = []
    for i in range(len(atom_types)):
      for j in range(i, len(atom_types)):
        out += [
            jnp.sum(
                _all_pairs_angular(D_different_types[i], D_different_types[j]),
                axis=[1, 2])
        ]
    return jnp.hstack(out)
  return compute_fn

def angular_symmetry_functions_neighbor_list(
    displacement: DisplacementFn,
    species: Array,
    etas: Array,
    lambdas: Array,
    zetas: Array,
    cutoff_distance: float
) -> Callable[[Array, NeighborList], Array]:
  """Returns a function that computes angular symmetry functions.

  Args:
    displacement: A function that produces an `[N_atoms, M_atoms,
      spatial_dimension]` of particle displacements from particle positions
      specified as an `[N_atoms, spatial_dimension] and `[M_atoms,
      spatial_dimension]` respectively.
    species: An `[N_atoms]` that contains the species of each particle.
    eta: Parameter of angular symmetry function that controls the spatial
      extension.
    lam:
    zeta:
    cutoff_distance: Neighbors whose distance is larger than `cutoff_distance` do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.
  Returns:
    A function that computes the angular symmetry function from input
    `[N_atoms, spatial_dimension]` and returns
    `[N_atoms, N_types * (N_types + 1) / 2]` where `N_types` is the number of
    types of particles in the system.
  """

  _angular_fn = vmap(single_pair_angular_symmetry_function,
                     (None, None, 0, 0, 0, None))

  _batched_angular_fn = lambda dR12, dR13: _angular_fn(dR12,
                                                       dR13,
                                                       etas,
                                                       lambdas,
                                                       zetas,
                                                       cutoff_distance)
  _all_pairs_angular = vmap(
      vmap(vmap(_batched_angular_fn, (0, None)), (None, 0)), 0)

  def sym_fn(R: Array, neighbor: NeighborList,
             mask_i: Array=None, mask_j: Array=None,
             **kwargs) -> Array:
    D_fn = partial(displacement, **kwargs)

    if neighbor.format is partition.Dense:
      D_fn = space.map_neighbor(D_fn)

      R_neigh = R[neighbor.idx]

      dR = D_fn(R, R_neigh)
      _all_pairs_angular = vmap(
        vmap(vmap(_batched_angular_fn, (0, None)), (None, 0)), 0)
      all_angular = _all_pairs_angular(dR, dR)

      mask_i = True if mask_i is None else mask_i[neighbor.idx]
      mask_j = True if mask_j is None else mask_j[neighbor.idx]

      mask_i = (neighbor.idx < R.shape[0]) & mask_i
      mask_i = mask_i[:, :, jnp.newaxis, jnp.newaxis]
      mask_j = (neighbor.idx < R.shape[0]) & mask_j
      mask_j = mask_j[:, jnp.newaxis, :, jnp.newaxis]

      return util.high_precision_sum(all_angular * mask_i * mask_j,
                                     axis=[1, 2])
    elif neighbor.format is partition.Sparse:
      D_fn = space.map_bond(D_fn)
      dR = D_fn(R[neighbor.idx[0]], R[neighbor.idx[1]])
      _all_pairs_angular = vmap(vmap(_batched_angular_fn, (0, None)),
                                (None, 0))
      all_angular = _all_pairs_angular(dR, dR)

      N = R.shape[0]
      mask_i = True if mask_i is None else mask_i[neighbor.idx[1]]
      mask_j = True if mask_j is None else mask_j[neighbor.idx[1]]
      mask_i = (neighbor.idx[0] < N) & mask_i
      mask_j = (neighbor.idx[0] < N) & mask_j

      mask = mask_i[:, None] & mask_j[None, :]
      mask = mask[:, :, None, None]
      all_angular = jnp.reshape(all_angular, (-1,) + all_angular.shape[2:])
      neighbor_idx = jnp.repeat(neighbor.idx[0], len(neighbor.idx[0]))
      out = ops.segment_sum(all_angular, neighbor_idx, N)
      return out
    else:
      raise ValueError()

  if species is None:
    def compute_fn(R: Array, neighbor: NeighborList, **kwargs) -> Array:
      return sym_fn(R, neighbor, **kwargs)
    return compute_fn

  def compute_fn(R: Array, neighbor: NeighborList, **kwargs) -> Array:
    atom_types = onp.unique(species)
    out = []
    for i in range(len(atom_types)):
      mask_i = species == i
      for j in range(i, len(atom_types)):
        mask_j = species == j
        out += [
            sym_fn(R, neighbor, mask_i, mask_j)
        ]
    return jnp.hstack(out)
  return compute_fn


def symmetry_functions_neighbor_list(
    displacement: DisplacementFn,
    species: Array,
    radial_etas: Optional[Array]=None,
    angular_etas: Optional[Array]=None,
    lambdas: Optional[Array]=None,
    zetas: Optional[Array]=None,
    cutoff_distance: float=8.0) -> Callable[[Array, NeighborList], Array]:
  if radial_etas is None:
    radial_etas = jnp.array([9e-4, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.4],
                    f32) / f32(0.529177 ** 2)

  if angular_etas is None:
    angular_etas = jnp.array([1e-4] * 4 + [0.003] * 4 + [0.008] * 2 +
                            [0.015] * 4 + [0.025] * 4 + [0.045] * 4,
                            f32) / f32(0.529177 ** 2)

  if lambdas is None:
    lambdas = jnp.array([-1, 1] * 4 + [1] * 14, f32)

  if zetas is None:
    zetas = jnp.array([1, 1, 2, 2] * 2 + [1, 2] + [1, 2, 4, 16] * 3, f32)

  radial_fn = radial_symmetry_functions_neighbor_list(
    displacement,
    species,
    etas=radial_etas,
    cutoff_distance=cutoff_distance)
  angular_fn = angular_symmetry_functions_neighbor_list(
    displacement,
    species,
    etas=angular_etas,
    lambdas=lambdas,
    zetas=zetas,
    cutoff_distance=cutoff_distance)
  return (lambda R, neighbor, **kwargs:
          jnp.hstack((radial_fn(R, neighbor, **kwargs),
                     angular_fn(R, neighbor, **kwargs))))


def symmetry_functions(displacement: DisplacementFn,
                       species: Optional[Array]=None,
                       radial_etas: Optional[Array]=None,
                       angular_etas: Optional[Array]=None,
                       lambdas: Optional[Array]=None,
                       zetas: Optional[Array]=None,
                       cutoff_distance: float=8.0
                       ) -> Callable[[Array], Array]:
  if radial_etas is None:
    radial_etas = jnp.array([9e-4, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.4],
                    f32) / f32(0.529177 ** 2)

  if angular_etas is None:
    angular_etas = jnp.array([1e-4] * 4 + [0.003] * 4 + [0.008] * 2 +
                            [0.015] * 4 + [0.025] * 4 + [0.045] * 4,
                            f32) / f32(0.529177 ** 2)

  if lambdas is None:
    lambdas = jnp.array([-1, 1] * 4 + [1] * 14, f32)

  if zetas is None:
    zetas = jnp.array([1, 1, 2, 2] * 2 + [1, 2] + [1, 2, 4, 16] * 3, f32)

  radial_fn = radial_symmetry_functions(displacement,
                                        species,
                                        etas=radial_etas,
                                        cutoff_distance=cutoff_distance)
  angular_fn = angular_symmetry_functions(displacement,
                                          species,
                                          etas=angular_etas,
                                          lambdas=lambdas,
                                          zetas=zetas,
                                          cutoff_distance=cutoff_distance)
  symmetry_fn = lambda R, **kwargs: jnp.hstack((radial_fn(R, **kwargs),
                                               angular_fn(R, **kwargs)))

  return symmetry_fn
