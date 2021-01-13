# Copyright 2020 Google LLC
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

"""Neural Network Primitives."""

from typing import Callable, Tuple, Dict, Any, Optional

import numpy as onp

import jax
from jax import vmap, jit
import jax.numpy as np

from jax_md import space, dataclasses, quantity, partition, smap, util
import haiku as hk

from collections import namedtuple
from functools import partial, reduce
from jax.tree_util import tree_multimap, tree_map


# Typing


Array = util.Array
f32 = util.f32
f64 = util.f64

InitFn = Callable[..., Array]
CallFn = Callable[..., Array]

DisplacementOrMetricFn = space.DisplacementOrMetricFn
DisplacementFn = space.DisplacementFn
NeighborList = partition.NeighborList


# Features used in fixed feature methods


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

def _behler_parrinello_cutoff_fn(dr: Array,
                                 cutoff_distance: float=8.0) -> Array:
  """Function of pairwise distance that smoothly goes to zero at the cutoff."""
  # Also returns zero if the pairwise distance is zero,
  # to prevent a particle from interacting with itself.
  return np.where((dr < cutoff_distance) & (dr > 1e-7),
                  0.5 * (np.cos(np.pi * dr / cutoff_distance) + 1), 0)


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
    cutoff_distance: Neighbors whose distance is larger than cutoff_distance do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.

  Returns:
    A function that computes the radial symmetry function from input `[N_atoms,
    spatial_dimension]` and returns `[N_atoms, N_etas * N_types]` where N_etas
    is the number of eta parameters, N_types is the number of types of
    particles in the system.
  """
  metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

  radial_fn = lambda eta, dr: (np.exp(-eta * dr**2) *
                               _behler_parrinello_cutoff_fn(dr, cutoff_distance))
  radial_fn = vmap(radial_fn, (0, None))

  if species is None:
    def compute_fn(R: Array, **kwargs) -> Array:
      _metric = partial(metric, **kwargs)
      _metric = space.map_product(_metric)
      return util.high_precision_sum(radial_fn(etas, _metric(R, R)), axis=1).T
  elif isinstance(species, np.ndarray):
    species = onp.array(species)
    def compute_fn(R: Array, **kwargs) -> Array:
      _metric = partial(metric, **kwargs)
      _metric = space.map_product(_metric)
      def return_radial(atom_type):
        """Returns the radial symmetry functions for neighbor type atom_type."""
        R_neigh = R[species == atom_type, :]
        dr = _metric(R, R_neigh)
        return util.high_precision_sum(radial_fn(etas, dr), axis=1).T
      return np.hstack([return_radial(atom_type) for
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
    cutoff_distance: Neighbors whose distance is larger than cutoff_distance do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.

  Returns:
    A function that computes the radial symmetry function from input `[N_atoms,
    spatial_dimension]` and returns `[N_atoms, N_etas * N_types]` where N_etas
    is the number of eta parameters, N_types is the number of types of
    particles in the system.
  """
  metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

  def radial_fn(eta, dr):
    return (np.exp(-eta * dr**2) *
            _behler_parrinello_cutoff_fn(dr, cutoff_distance))
  radial_fn = vmap(radial_fn, (0, None))

  if species is None:
    def compute_fn(R: Array, neighbor: NeighborList, **kwargs) -> Array:
      _metric = partial(metric, **kwargs)
      _metric = space.map_neighbor(_metric)
      R_neigh = R[neighbor.idx]
      mask = (neighbor.idx < R.shape[0])[np.newaxis, :, :]
      dr = _metric(R, R_neigh)
      return util.high_precision_sum(radial_fn(etas, dr) * mask, axis=2).T
    return compute_fn

  def compute_fn(R: Array, neighbor: NeighborList, **kwargs) -> Array:
    _metric = partial(metric, **kwargs)
    _metric = space.map_neighbor(_metric)
    radial_fn = lambda eta, dr: (np.exp(-eta * dr**2) *
                _behler_parrinello_cutoff_fn(dr, cutoff_distance))
    def return_radial(atom_type):
      """Returns the radial symmetry functions for neighbor type atom_type."""
      R_neigh = R[neighbor.idx]
      species_neigh = species[neighbor.idx]
      mask = np.logical_and(neighbor.idx < R.shape[0],
                            species_neigh == atom_type)
      dr = _metric(R, R_neigh)

      radial = vmap(radial_fn, (0, None))(etas, dr)
      return util.high_precision_sum(radial * mask[np.newaxis, :, :], axis=2).T

    return np.hstack([return_radial(atom_type) for
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
      lambda x, y: x * _behler_parrinello_cutoff_fn(y, cutoff_distance),
      [dr12, dr13, dr23], 1.0)
  result = 2.0 ** (1.0 - zeta) * (
      1.0 + lam * quantity.angle_between_two_vectors(dR12, dR13)) ** zeta * \
      np.exp(-eta * triplet_squared_distances) * triplet_cutoff
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
    cutoff_distance: Neighbors whose distance is larger than cutoff_distance do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.
  Returns:
    A function that computes the angular symmetry function from input `[N_atoms,
    spatial_dimension]` and returns `[N_atoms, N_types * (N_types + 1) / 2]`
    where N_types is the number of types of particles in the system.
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
     return np.sum(_all_pairs_angular(dR, dR), axis=[1, 2])
   return compute_fn

  if isinstance(species, np.ndarray):
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
            np.sum(
                _all_pairs_angular(D_different_types[i], D_different_types[j]),
                axis=[1, 2])
        ]
    return np.hstack(out)
  return compute_fn

def angular_symmetry_functions_neighbor_list(
    displacement: DisplacementFn,
    species: Array,
    etas: Array,
    lambdas: Array,
    zetas: Array,
    cutoff_distance: float) -> Callable[[Array, NeighborList], Array]:
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
    cutoff_distance: Neighbors whose distance is larger than cutoff_distance do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.
  Returns:
    A function that computes the angular symmetry function from input
    `[N_atoms, spatial_dimension]` and returns
    `[N_atoms, N_types * (N_types + 1) / 2]` where N_types is the number of
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

  if species is None:
    def compute_fn(R: Array, neighbor: NeighborList, **kwargs) -> Array:
      D_fn = partial(displacement, **kwargs)
      D_fn = space.map_neighbor(D_fn)

      R_neigh = R[neighbor.idx]
      mask = neighbor.idx < R.shape[0]

      dR = D_fn(R, R_neigh)
      all_angular = _all_pairs_angular(dR, dR)

      mask_i = mask[:, :, np.newaxis, np.newaxis]
      mask_j = mask[:, np.newaxis, :, np.newaxis]

      return util.high_precision_sum(all_angular * mask_i * mask_j,
                                     axis=[1, 2])
    return compute_fn

  def compute_fn(R: Array, neighbor: NeighborList, **kwargs) -> Array:
    D_fn = partial(displacement, **kwargs)
    D_fn = space.map_neighbor(D_fn)

    R_neigh = R[neighbor.idx]
    species_neigh = species[neighbor.idx]

    atom_types = onp.unique(species)

    base_mask = neighbor.idx < len(R)
    mask = [np.logical_and(base_mask, species_neigh == t) for t in atom_types]

    out = []
    dR = D_fn(R, R_neigh)
    all_angular = _all_pairs_angular(dR, dR)

    for i in range(len(atom_types)):
      mask_i = mask[i][:, :, np.newaxis, np.newaxis]
      for j in range(i, len(atom_types)):
        mask_j = mask[j][:, np.newaxis, :, np.newaxis]
        out += [
            util.high_precision_sum(all_angular * mask_i * mask_j, axis=[1, 2])
        ]
    return np.hstack(out)
  return compute_fn


def behler_parrinello_symmetry_functions_neighbor_list(
    displacement: DisplacementFn,
    species: Array,
    radial_etas: Array=None,
    angular_etas: Array=None,
    lambdas: Array=None,
    zetas: Array=None,
    cutoff_distance: float=8.0) -> Callable[[Array, NeighborList], Array]:
  if radial_etas is None:
    radial_etas = np.array([9e-4, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.4],
                    f32) / f32(0.529177 ** 2)

  if angular_etas is None:
    angular_etas = np.array([1e-4] * 4 + [0.003] * 4 + [0.008] * 2 + 
                            [0.015] * 4 + [0.025] * 4 + [0.045] * 4,
                            f32) / f32(0.529177 ** 2)

  if lambdas is None:
    lambdas = np.array([-1, 1] * 4 + [1] * 14, f32)

  if zetas is None:
    zetas = np.array([1, 1, 2, 2] * 2 + [1, 2] + [1, 2, 4, 16] * 3, f32)

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
          np.hstack((radial_fn(R, neighbor, **kwargs),
                     angular_fn(R, neighbor, **kwargs))))


def behler_parrinello_symmetry_functions(displacement: DisplacementFn,
                                         species: Array=None,
                                         radial_etas: Array=None, 
                                         angular_etas: Array=None, 
                                         lambdas: Array=None,
                                         zetas: Array=None, 
                                         cutoff_distance: float=8.0
                                         ) -> Callable[[Array], Array]:
  if radial_etas is None:
    radial_etas = np.array([9e-4, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.4],
                    f32) / f32(0.529177 ** 2)
  
  if angular_etas is None:
    angular_etas = np.array([1e-4] * 4 + [0.003] * 4 + [0.008] * 2 + 
                            [0.015] * 4 + [0.025] * 4 + [0.045] * 4,
                            f32) / f32(0.529177 ** 2)
  
  if lambdas is None:
    lambdas = np.array([-1, 1] * 4 + [1] * 14, f32)
  
  if zetas is None:
    zetas = np.array([1, 1, 2, 2] * 2 + [1, 2] + [1, 2, 4, 16] * 3, f32)

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
  symmetry_fn = lambda R, **kwargs: np.hstack((radial_fn(R, **kwargs),
                                               angular_fn(R, **kwargs)))

  return symmetry_fn


# Graph neural network primitives

"""
  Our implementation here is based off the outstanding GraphNets library by
  DeepMind at, www.github.com/deepmind/graph_nets. This implementation was also
  heavily influenced by work done by Thomas Keck. We implement a subset of the
  functionality from the graph nets library to be compatable with jax-md
  states and neighbor lists, end-to-end jit compilation, and easy batching.

  Graphs are described by node states, edge states, a global state, and
  outgoing / incoming edges.  

  We provide two components:

    1) A GraphIndependent layer that applies a neural network separately to the
       node states, the edge states, and the globals. This is often used as an
       encoding or decoding step.
    2) A GraphNetwork layer that transforms the nodes, edges, and globals using
       neural networks following Battaglia et al. (). Here, we use
       sum-message-aggregation. 

  The graphs network components implemented here implement identical functions
  to the DeepMind library. However, to be compatible with jax-md, there are
  significant differences in the graph layout used here to the reference
  implementation. See `GraphTuple` for details.
"""

@dataclasses.dataclass
class GraphTuple(object):
    """A struct containing graph data.

    Attributes:
      nodes: For a graph with N_nodes, this is an `[N_nodes, node_dimension]`
        array containing the state of each node in the graph.
      edges: For a graph whose degree is bounded by max_degree, this is an
        `[N_nodes, max_degree, edge_dimension]`. Here `edges[i, j]` is the
        state of the outgoing edge from node `i` to node `edge_idx[i, j]`.
      globals: An array of shape `[global_dimension]`.
      edge_idx: An integer array of shape `[N_nodes, max_degree]` where
        `edge_idx[i, j]` is the id of the jth outgoing edge from node `i`.
        Empty entries (that don't contain an edge) are denoted by
        `edge_idx[i, j] == N_nodes`.
    """
    nodes: np.ndarray
    edges: np.ndarray
    globals: np.ndarray
    edge_idx: np.ndarray


def concatenate_graph_features(graphs: Tuple[GraphTuple, ...]) -> GraphTuple:
  """Given a list of GraphTuple returns a new concatenated GraphTuple.

  Note that currently we do not check that the graphs have consistent edge
  connectivity.
  """
  return GraphTuple(
      nodes=np.concatenate([g.nodes for g in graphs], axis=-1),
      edges=np.concatenate([g.edges for g in graphs], axis=-1),
      globals=np.concatenate([g.globals for g in graphs], axis=-1),
      edge_idx=graphs[0].edge_idx,  # pytype: disable=wrong-keyword-args
  )


def GraphIndependent(edge_fn: Callable[[Array], Array],
                     node_fn: Callable[[Array], Array],
                     global_fn: Callable[[Array], Array]
                     ) -> Callable[[GraphTuple], GraphTuple]:
  """Applies functions independently to the nodes, edges, and global states.
  """
  identity = lambda x: x
  _node_fn = vmap(node_fn) if node_fn is not None else identity
  _edge_fn = vmap(vmap(edge_fn)) if edge_fn is not None else identity
  _global_fn = global_fn if global_fn is not None else identity

  def embed_fn(graph):
    return dataclasses.replace(
        graph,
        nodes=_node_fn(graph.nodes),
        edges=_edge_fn(graph.edges),
        globals=_global_fn(graph.globals)
    )
  return embed_fn


def _apply_node_fn(graph: GraphTuple,
                   node_fn: Callable[[Array,Array, Array, Array], Array]
                   ) -> Array:
  mask = graph.edge_idx < graph.nodes.shape[0]
  mask = mask[:, :, np.newaxis]

  if graph.edges is not None:
    # TODO: Should we also have outgoing edges?
    flat_edges = np.reshape(graph.edges, (-1, graph.edges.shape[-1]))
    edge_idx = np.reshape(graph.edge_idx, (-1,))
    incoming_edges = jax.ops.segment_sum(
        flat_edges, edge_idx, graph.nodes.shape[0] + 1)[:-1]
    outgoing_edges = np.sum(graph.edges * mask, axis=1)
  else:
    incoming_edges = None
    outgoing_edges = None

  if graph.globals is not None:
    _globals = np.broadcast_to(graph.globals[np.newaxis, :],
                               graph.nodes.shape[:1] + graph.globals.shape)
  else:
    _globals = None

  return node_fn(graph.nodes, incoming_edges, outgoing_edges, _globals)


def _apply_edge_fn(graph: GraphTuple,
                   edge_fn: Callable[[Array, Array, Array, Array], Array]
                   ) -> Array:
  if graph.nodes is not None:
    incoming_nodes = graph.nodes[graph.edge_idx]
    outgoing_nodes = np.broadcast_to(
        graph.nodes[:, np.newaxis, :],
        graph.edge_idx.shape + graph.nodes.shape[-1:])
  else:
    incoming_nodes = None
    outgoing_nodes = None

  if graph.globals is not None:
    _globals = np.broadcast_to(graph.globals[np.newaxis, np.newaxis, :],
                               graph.edge_idx.shape + graph.globals.shape)
  else:
    _globals = None

  mask = graph.edge_idx < graph.nodes.shape[0]
  mask = mask[:, :, np.newaxis]
  return edge_fn(graph.edges, incoming_nodes, outgoing_nodes, _globals) * mask


def _apply_global_fn(graph: GraphTuple,
                     global_fn: Callable[[Array, Array, Array], Array]
                     ) -> Array:
  nodes = None if graph.nodes is None else np.sum(graph.nodes, axis=0)

  if graph.edges is not None:
    mask = graph.edge_idx < graph.nodes.shape[0]
    mask = mask[:, :, np.newaxis]
    edges = np.sum(graph.edges * mask, axis=(0, 1))
  else:
    edges = None

  return global_fn(nodes, edges, graph.globals)


class GraphNetwork:
  """Implementation of a Graph Network.

  See https://arxiv.org/abs/1806.01261 for more details.
  """
  def __init__(self,
               edge_fn: Callable[[Array], Array],
               node_fn: Callable[[Array], Array],
               global_fn: Callable[[Array], Array]):
    self._node_fn = (None if node_fn is None else
                     partial(_apply_node_fn, node_fn=vmap(node_fn)))

    self._edge_fn = (None if edge_fn is None else
                     partial(_apply_edge_fn, edge_fn=vmap(vmap(edge_fn))))

    self._global_fn = (None if global_fn is None else
                       partial(_apply_global_fn, global_fn=global_fn))

  def __call__(self, graph: GraphTuple) -> GraphTuple:
    if self._edge_fn is not None:
      graph = dataclasses.replace(graph, edges=self._edge_fn(graph))

    if self._node_fn is not None:
      graph = dataclasses.replace(graph, nodes=self._node_fn(graph))

    if self._global_fn is not None:
      graph = dataclasses.replace(graph, globals=self._global_fn(graph))

    return graph


# Prefab Networks


class GraphNetEncoder(hk.Module):
  """Implements a Graph Neural Network for energy fitting.

  Based on the network used in "Unveiling the predictive power of static
  structure in glassy systems"; Bapst et al.
  (https://www.nature.com/articles/s41567-020-0842-8). This network first
  embeds edges, nodes, and global state. Then `n_recurrences` of GraphNetwork
  layers are applied. Unlike in Bapst et al. this network does not include a
  readout, which should be added separately depending on the application.

  For example, when predicting particle mobilities, one would use a decoder
  only on the node states while a model of energies would decode only the node
  states.
  """
  def __init__(self,
               n_recurrences: int,
               mlp_sizes: Tuple[int, ...],
               mlp_kwargs: Dict[str, Any]=None,
               name: str='GraphNetEncoder'):
    super(GraphNetEncoder, self).__init__(name=name)

    if mlp_kwargs is None:
      mlp_kwargs = {}

    self._n_recurrences = n_recurrences

    embedding_fn = lambda name: hk.nets.MLP(
        output_sizes=mlp_sizes,
        activate_final=True,
        name=name,
        **mlp_kwargs)

    model_fn = lambda name: lambda *args: hk.nets.MLP(
        output_sizes=mlp_sizes,
        activate_final=True,
        name=name,
        **mlp_kwargs)(np.concatenate(args, axis=-1))

    self._encoder = GraphIndependent(
        embedding_fn('EdgeEncoder'),
        embedding_fn('NodeEncoder'),
        embedding_fn('GlobalEncoder'))
    self._propagation_network = lambda: GraphNetwork(
        model_fn('EdgeFunction'),
        model_fn('NodeFunction'),
        model_fn('GlobalFunction'))

  def __call__(self, graph: GraphTuple) -> GraphTuple:
    encoded = self._encoder(graph)
    outputs = encoded

    for _ in range(self._n_recurrences):
      inputs = concatenate_graph_features((outputs, encoded))
      outputs = self._propagation_network()(inputs)

    return outputs
