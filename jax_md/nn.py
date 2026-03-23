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

from flax import nnx
import jax
from jax import vmap, jit
import jax.numpy as jnp

from jax_md import space, dataclasses, quantity, partition, smap
from jax_md import util as jmd_util

from collections import namedtuple
from functools import partial, reduce
from jax.tree_util import tree_map
from jax import ops

import jraph

from ._nn import behler_parrinello
from ._nn import nequip
from ._nn import gnome
from ._nn import util

# Typing


Array = jmd_util.Array
f32 = jmd_util.f32
f64 = jmd_util.f64

CallFn = Callable[..., Array]

DisplacementOrMetricFn = space.DisplacementOrMetricFn
DisplacementFn = space.DisplacementFn
NeighborList = partition.NeighborList

ActivationFn = Callable[[Array], Array]

DEFAULT_KERNEL_INIT = jax.nn.initializers.variance_scaling(
  1.0, 'fan_avg', 'truncated_normal'
)
DEFAULT_BIAS_INIT = jax.nn.initializers.zeros


class MLP(nnx.Module):
  """Multi-layer perceptron with configurable activation."""

  def __init__(
    self,
    in_features: int,
    output_sizes: Tuple[int, ...],
    *,
    rngs: nnx.Rngs,
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Callable = DEFAULT_KERNEL_INIT,
    bias_init: Callable = DEFAULT_BIAS_INIT,
    use_bias: bool = True,
    activate_final: bool = True,
  ):
    self.activation = activation
    self.activate_final = activate_final
    sizes = (in_features,) + tuple(output_sizes)
    self.layers = nnx.List(
      [
        nnx.Linear(
          sizes[i],
          sizes[i + 1],
          use_bias=use_bias,
          kernel_init=kernel_init,
          bias_init=bias_init,
          rngs=rngs,
        )
        for i in range(len(sizes) - 1)
      ]
    )

  def __call__(self, x: Array) -> Array:
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if self.activate_final or i < len(self.layers) - 1:
        x = self.activation(x)
    return x


# TO BE DELETED BELOW:
# Graph neural network primitives

"""
  Our implementation here is based off the outstanding GraphNets library by
  DeepMind at, www.github.com/deepmind/graph_nets. This implementation was also
  heavily influenced by work done by Thomas Keck. We implement a subset of the
  functionality from the graph nets library to be compatible with jax-md
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
  implementation. See `GraphsTuple` for details.
"""


@dataclasses.dataclass
class GraphsTuple(object):
  """A struct containing graph data.

  Attributes:
    nodes: For a graph with `N_nodes`, this is an `[N_nodes, node_dimension]`
      array containing the state of each node in the graph.
    edges: For a graph whose degree is bounded by max_degree, this is an
      `[N_nodes, max_degree, edge_dimension]`. Here `edges[i, j]` is the
      state of the outgoing edge from node `i` to node `edge_idx[i, j]`.
    globals: An array of shape `[global_dimension]`.
    edge_idx: An integer array of shape `[N_nodes, max_degree]` where
      `edge_idx[i, j]` is the id of the j-th outgoing edge from node `i`.
      Empty entries (that don't contain an edge) are denoted by
      `edge_idx[i, j] == N_nodes`.
  """

  nodes: jnp.ndarray
  edges: jnp.ndarray
  globals: jnp.ndarray
  edge_idx: jnp.ndarray

  _replace = dataclasses.replace


def concatenate_graph_features(graphs: Tuple[GraphsTuple, ...]) -> GraphsTuple:
  """Given a list of GraphsTuple returns a new concatenated GraphsTuple.

  Note that currently we do not check that the graphs have consistent edge
  connectivity.
  """
  graph = graphs[0]
  return graph._replace(
    nodes=jnp.concatenate([g.nodes for g in graphs], axis=-1),
    edges=jnp.concatenate([g.edges for g in graphs], axis=-1),
    globals=jnp.concatenate(
      [g.globals for g in graphs], axis=-1
    ),  # pytype: disable=missing-parameter
  )


def GraphMapFeatures(
  edge_fn: Callable[[Array], Array],
  node_fn: Callable[[Array], Array],
  global_fn: Callable[[Array], Array],
) -> Callable[[GraphsTuple], GraphsTuple]:
  """Applies functions independently to the nodes, edges, and global states."""
  identity = lambda x: x
  _node_fn = vmap(node_fn) if node_fn is not None else identity
  _edge_fn = vmap(vmap(edge_fn)) if edge_fn is not None else identity
  _global_fn = global_fn if global_fn is not None else identity

  def embed_fn(graph):
    return graph._replace(
      nodes=_node_fn(graph.nodes),
      edges=_edge_fn(graph.edges),
      globals=_global_fn(graph.globals),
    )

  return embed_fn


def apply_node_fn(
  graph: GraphsTuple, node_fn: Callable[[Array, Array, Array, Array], Array]
) -> Array:
  mask = graph.edge_idx < graph.nodes.shape[0]
  mask = mask[:, :, jnp.newaxis]

  if graph.edges is not None:
    # TODO: Should we also have outgoing edges?
    flat_edges = jnp.reshape(graph.edges, (-1, graph.edges.shape[-1]))
    edge_idx = jnp.reshape(graph.edge_idx, (-1,))
    incoming_edges = jax.ops.segment_sum(
      flat_edges, edge_idx, graph.nodes.shape[0] + 1
    )[:-1]
    outgoing_edges = jnp.sum(graph.edges * mask, axis=1)
  else:
    incoming_edges = None
    outgoing_edges = None

  if graph.globals is not None:
    _globals = jnp.broadcast_to(
      graph.globals[jnp.newaxis, :], graph.nodes.shape[:1] + graph.globals.shape
    )
  else:
    _globals = None

  return node_fn(graph.nodes, incoming_edges, outgoing_edges, _globals)


def apply_edge_fn(
  graph: GraphsTuple, edge_fn: Callable[[Array, Array, Array, Array], Array]
) -> Array:
  if graph.nodes is not None:
    incoming_nodes = graph.nodes[graph.edge_idx]
    outgoing_nodes = jnp.broadcast_to(
      graph.nodes[:, jnp.newaxis, :],
      graph.edge_idx.shape + graph.nodes.shape[-1:],
    )
  else:
    incoming_nodes = None
    outgoing_nodes = None

  if graph.globals is not None:
    _globals = jnp.broadcast_to(
      graph.globals[jnp.newaxis, jnp.newaxis, :],
      graph.edge_idx.shape + graph.globals.shape,
    )
  else:
    _globals = None

  mask = graph.edge_idx < graph.nodes.shape[0]
  mask = mask[:, :, jnp.newaxis]
  return edge_fn(graph.edges, incoming_nodes, outgoing_nodes, _globals) * mask


def apply_global_fn(
  graph: GraphsTuple, global_fn: Callable[[Array, Array, Array], Array]
) -> Array:
  nodes = None if graph.nodes is None else jnp.sum(graph.nodes, axis=0)

  if graph.edges is not None:
    mask = graph.edge_idx < graph.nodes.shape[0]
    mask = mask[:, :, jnp.newaxis]
    edges = jnp.sum(graph.edges * mask, axis=(0, 1))
  else:
    edges = None

  return global_fn(nodes, edges, graph.globals)


class GraphNetwork:
  """Implementation of a Graph Network.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(
    self,
    edge_fn: Callable[[Array, Array, Array, Array], Array],
    node_fn: Callable[[Array, Array, Array, Array], Array],
    global_fn: Callable[[Array, Array, Array], Array],
  ):
    self.node_fn = (
      None if node_fn is None else partial(apply_node_fn, node_fn=vmap(node_fn))
    )

    self.edge_fn = (
      None
      if edge_fn is None
      else partial(apply_edge_fn, edge_fn=vmap(vmap(edge_fn)))
    )

    self.global_fn = (
      None
      if global_fn is None
      else partial(apply_global_fn, global_fn=global_fn)
    )

  def __call__(self, graph: GraphsTuple) -> GraphsTuple:
    if self.edge_fn is not None:
      graph = graph._replace(edges=self.edge_fn(graph))

    if self.node_fn is not None:
      graph = graph._replace(nodes=self.node_fn(graph))

    if self.global_fn is not None:
      graph = graph._replace(globals=self.global_fn(graph))

    return graph


# Prefab Networks


class GraphNetEncoder(nnx.Module):
  """Implements a Graph Neural Network for energy fitting.

  Based on the network used in "Unveiling the predictive power of static
  structure in glassy systems"; Bapst et al.
  (https://www.nature.com/articles/s41567-020-0842-8). This network first
  embeds edges, nodes, and global state. Then ``n_recurrences`` of GraphNetwork
  layers are applied. Unlike in Bapst et al. this network does not include a
  readout, which should be added separately depending on the application.

  For example, when predicting particle mobilities, one would use a decoder
  only on the node states while a model of energies would decode only the node
  states.
  """

  def __init__(
    self,
    in_node_features: int,
    in_edge_features: int,
    in_global_features: int,
    n_recurrences: int,
    mlp_sizes: Tuple[int, ...],
    *,
    rngs: nnx.Rngs,
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Callable = DEFAULT_KERNEL_INIT,
    bias_init: Callable = DEFAULT_BIAS_INIT,
    format: partition.NeighborListFormat = partition.Dense,
  ):
    self.n_recurrences = n_recurrences
    self.format = format

    kw = dict(
      rngs=rngs,
      activation=activation,
      kernel_init=kernel_init,
      bias_init=bias_init,
      activate_final=True,
    )
    m = mlp_sizes[-1]

    self.EdgeEncoder = MLP(in_edge_features, mlp_sizes, **kw)
    self.NodeEncoder = MLP(in_node_features, mlp_sizes, **kw)
    self.GlobalEncoder = MLP(in_global_features, mlp_sizes, **kw)

    self.edge_fns = nnx.List(
      [MLP(8 * m, mlp_sizes, **kw) for _ in range(n_recurrences)]
    )
    self.node_fns = nnx.List(
      [MLP(6 * m, mlp_sizes, **kw) for _ in range(n_recurrences)]
    )
    self.global_fns = nnx.List(
      [MLP(4 * m, mlp_sizes, **kw) for _ in range(n_recurrences)]
    )

  def __call__(self, graph: GraphsTuple) -> GraphsTuple:
    if self.format is partition.Dense:
      graph_map_features = GraphMapFeatures
      graph_network = GraphNetwork
    elif self.format is partition.Sparse:
      graph_map_features = jraph.GraphMapFeatures
      graph_network = jraph.GraphNetwork
    else:
      raise ValueError()

    encoded = graph_map_features(
      self.EdgeEncoder, self.NodeEncoder, self.GlobalEncoder
    )(graph)

    outputs = encoded

    for edge_mlp, node_mlp, global_mlp in zip(
      self.edge_fns, self.node_fns, self.global_fns
    ):

      def edge_update(edges, sent, received, globals_, mlp=edge_mlp):
        return mlp(jnp.concatenate((edges, sent, received, globals_), axis=-1))

      def node_update(nodes, sent, received, globals_, mlp=node_mlp):
        return mlp(jnp.concatenate((nodes, sent, received, globals_), axis=-1))

      def global_update(nodes, edges, globals_, mlp=global_mlp):
        return mlp(jnp.concatenate((nodes, edges, globals_), axis=-1))

      inputs = concatenate_graph_features((outputs, encoded))
      outputs = graph_network(edge_update, node_update, global_update)(inputs)

    return outputs
