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


from typing import Dict, Union, Tuple


import functools

import e3nn_jax as e3nn

import flax.linen as nn

import jax
import jax.numpy as jnp
from jax import tree_util
from jax import vmap

from jax_md import space
from jax_md import util

import jraph

from . import util as nn_util

from .e3nn_layer import FullyConnectedTensorProductE3nn
from .e3nn_layer import Linear

import operator

from ml_collections import ConfigDict


# Types


Irreps = e3nn.Irreps
IrrepsArray = e3nn.IrrepsArray
Irrep = e3nn.Irrep
Array = util.Array
FeaturizerFn = nn_util.FeaturizerFn


get_nonlinearity_by_name = nn_util.get_nonlinearity_by_name
partial = functools.partial


# Code


def featurizer(sh_irreps: str)-> FeaturizerFn:
  """Node, edge, global features for NequIP."""
  def node_feature_fn(graph):
    """ NequIP node features are simply the input graph's nodes. """
    return graph.nodes

  def edge_feature_fn(dR):
    """ NequIP edge features.

     Builds the following:
     - the relative interatomic positions \vec{r}_{ij}
     - interatomic distances r_{ij}
     - normalized spherical harmonics Y(\hat{r}_{ij})
     """
    features = []
    edge_sh_irreps = Irreps(sh_irreps)
    edge_sh = e3nn.spherical_harmonics(edge_sh_irreps, dR, normalize=True)
    return dR, space.distance(dR), edge_sh

  def global_feature_fn(g):
    """ NequIP does not use global features. """
    return g

  return node_feature_fn, edge_feature_fn, global_feature_fn


def prod(xs):
  """From e3nn_jax/util/__init__.py."""
  return functools.reduce(operator.mul, xs, 1)


def tp_path_exists(arg_in1, arg_in2, arg_out):
  """Check if a tensor product path is viable.

  This helper function is similar to the one used in:
  https://github.com/e3nn/e3nn
  """
  arg_in1 = Irreps(arg_in1).simplify()
  arg_in2 = Irreps(arg_in2).simplify()
  arg_out = Irrep(arg_out)

  for multiplicity_1, irreps_1 in arg_in1:
    for multiplicity_2, irreps_2 in arg_in2:
      if arg_out in irreps_1 * irreps_2:
        return True
  return False


class NequIPConvolution(nn.Module):
  """NequIP Convolution.

  Implementation follows the original paper by Batzner et al.

  nature.com/articles/s41467-022-29939-5 and partially
  https://github.com/mir-group/nequip.

  Args:
        hidden_irreps: irreducible representation of hidden/latent features
        use_sc: use self-connection in network (recommended)
        nonlinearities: nonlinearities to use for even/odd irreps
        radial_net_nonlinearity: nonlinearity to use in radial MLP
        radial_net_n_hidden: number of hidden neurons in radial MLP
        radial_net_n_layers: number of hidden layers for radial MLP
        num_basis: number of Bessel basis functions to use
        n_neighbors: constant number of per-atom neighbors, used for internal
        normalization
        scalar_mlp_std: standard deviation of weight init of radial MLP

    Returns:
        Updated node features h after the convolution.
  """
  hidden_irreps: Irreps
  use_sc: bool
  nonlinearities: Union[str, Dict[str, str]]
  radial_net_nonlinearity: str = 'raw_swish'
  radial_net_n_hidden: int = 64
  radial_net_n_layers: int = 2
  num_basis: int = 8
  n_neighbors: float = 1.
  scalar_mlp_std: float = 4.0

  @nn.compact
  def __call__(
      self,
      node_features: IrrepsArray,
      node_attributes: IrrepsArray,
      edge_sh: Array,
      edge_src: Array,
      edge_dst: Array,
      edge_embedded: Array
      ) -> IrrepsArray:
    # Convolution outline in NequIP is:
    # Linear on nodes
    # TP + aggregate
    # divide by average number of neighbors
    # Concatenation
    # Linear on nodes
    # Self-connection
    # Gate
    irreps_scalars = []
    irreps_nonscalars = []
    irreps_gate_scalars = []

    # get scalar target irreps
    for multiplicity, irrep in self.hidden_irreps:
      # need the additional Irrep() here for the build, even though irrep is
      # already of type Irrep()
      if (Irrep(irrep).l == 0 and
          tp_path_exists(node_features.irreps, edge_sh.irreps, irrep)):
        irreps_scalars += [(multiplicity, irrep)]

    irreps_scalars = Irreps(irreps_scalars)

    # get non-scalar target irreps
    for multiplicity, irrep in self.hidden_irreps:
      # need the additional Irrep() here for the build, even though irrep is
      # already of type Irrep()
      if (Irrep(irrep).l > 0 and
          tp_path_exists(node_features.irreps, edge_sh.irreps, irrep)):
        irreps_nonscalars += [(multiplicity, irrep)]

    irreps_nonscalars = Irreps(irreps_nonscalars)

    # get gate scalar irreps
    if tp_path_exists(node_features.irreps, edge_sh.irreps, '0e'):
      gate_scalar_irreps_type = '0e'
    else:
      gate_scalar_irreps_type = '0o'

    for multiplicity, irreps in irreps_nonscalars:
      irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]

    irreps_gate_scalars = Irreps(irreps_gate_scalars)

    # final layer output irreps are all three
    # note that this order is assumed by the gate function later, i.e.
    # scalars left, then gate scalar, then non-scalars
    h_out_irreps = irreps_scalars + irreps_gate_scalars + irreps_nonscalars

    # self-connection: TP between node features and node attributes
    # this can equivalently be seen as a matrix multiplication acting on
    # the node features where the weight matrix is indexed by the node
    # attributes (typically the chemical species), i.e. a linear transform
    # that is a function of the species of the central atom
    if self.use_sc:
      self_connection = FullyConnectedTensorProductE3nn(
          h_out_irreps,
          # node_features.irreps,
          # node_attributes.irreps
          )(node_features, node_attributes)

    h = node_features

    # first linear, stays in current h-space
    h = Linear(node_features.irreps)(h)

    # map node features onto edges for tp
    edge_features = jax.tree_map(lambda x: x[edge_src], h)

    # we gather the instructions for the tp as well as the tp output irreps
    mode = 'uvu'
    trainable = 'True'
    irreps_after_tp = []
    instructions = []

    # iterate over both arguments, i.e. node irreps and edge irreps
    # if they have a valid TP path for any of the target irreps,
    # add to instructions and put in appropriate position
    # we use uvu mode (where v is a single-element sum) and weights will
    # be provide externally by the scalar MLP
    # this triple for loop is similar to the one used in e3nn and nequip
    for i, (mul_in1, irreps_in1) in enumerate(node_features.irreps):
      for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
        for curr_irreps_out in irreps_in1 * irreps_in2:
          if curr_irreps_out in h_out_irreps:
            k = len(irreps_after_tp)
            irreps_after_tp += [(mul_in1, curr_irreps_out)]
            instructions += [(i, j, k, mode, trainable)]

    # we will likely have constructed irreps in a non-l-increasing order
    # so we sort them to be in a l-increasing order
    irreps_after_tp, p, _ = Irreps(irreps_after_tp).sort()

    # if we sort the target irreps, we will have to sort the instructions
    # acoordingly, using the permutation indices
    sorted_instructions = []

    for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
      sorted_instructions += [
          (irreps_in1, irreps_in2, p[irreps_out], mode, trainable)]

    # TP between spherical harmonics embedding of the edge vector
    # Y_ij(\hat{r}) and neighboring node h_j, weighted on a per-element basis
    # by the radial network R(r_ij)
    tp = e3nn.FunctionalTensorProduct(
        irreps_in1=edge_features.irreps,
        irreps_in2=edge_sh.irreps,
        irreps_out=irreps_after_tp,
        instructions=sorted_instructions
    )

    # scalar radial network, number of output neurons is the total number of
    # tensor product paths, nonlinearity must have f(0)=0 and MLP must not
    # have biases
    n_tp_weights = 0

    # get output dim of radial MLP / number of TP weights
    for ins in tp.instructions:
      if ins.has_weight:
        n_tp_weights += prod(ins.path_shape)

    # build radial MLP R(r) that maps from interatomic distances to TP weights
    # must not use bias to that R(0)=0
    fc = nn_util.MLP(
        (self.radial_net_n_hidden,) * self.radial_net_n_layers + (n_tp_weights,),
        self.radial_net_nonlinearity,
        use_bias=False,
        scalar_mlp_std=self.scalar_mlp_std
        )

    # the TP weights (v dimension) are given by the FC
    weight = fc(edge_embedded)

    # tp between node features that have been mapped onto edges and edge RSH
    # weighted by FC weight, we vmap over the dimension of the edges
    edge_features = jax.vmap(tp.left_right)(weight, edge_features, edge_sh)

    # aggregate edges onto nodes after tp using e3nn-jax's index_add
    h = jax.tree_map(
        lambda x: e3nn.index_add(edge_dst, x, out_dim=h.shape[0]),
        edge_features
        )

    # normalize by the average (not local) number of neighbors
    h = h / self.n_neighbors

    # second linear, now we create extra gate scalars by mapping to h-out
    h = Linear(h_out_irreps)(h)

    # self-connection, similar to a resnet-update that sums the output from
    # the TP to chemistry-weighted h
    if self.use_sc:
      h = h + self_connection

    # gate nonlinearity, applied to gate data, consisting of:
    # a) regular scalars,
    # b) gate scalars, and
    # c) non-scalars to be gated
    # in this order
    gate_fn = partial(
        e3nn.gate,
        even_act=get_nonlinearity_by_name(self.nonlinearities['e']),
        odd_act=get_nonlinearity_by_name(self.nonlinearities['o']),
        even_gate_act=get_nonlinearity_by_name(self.nonlinearities['e']),
        odd_gate_act=get_nonlinearity_by_name(self.nonlinearities['o'])
        )

    h = gate_fn(h)

    return h


class NequIPEnergyModel(nn.Module):
  """NequIP.

  Implementation follows the original paper by Batzner et al.

  nature.com/articles/s41467-022-29939-5 and partially
  https://github.com/mir-group/nequip.

    Args:
        graph_net_steps: number of NequIP convolutional layers
        use_sc: use self-connection in network (recommended)
        nonlinearities: nonlinearities to use for even/odd irreps
        n_element: number of chemical elements in input data
        hidden_irreps: irreducible representation of hidden/latent features
        sh_irreps: irreducible representations on the edges
        num_basis: number of Bessel basis functions to use
        r_max: radial cutoff used in length units
        radial_net_nonlinearity: nonlinearity to use in radial MLP
        radial_net_n_hidden: number of hidden neurons in radial MLP
        radial_net_n_layers: number of hidden layers for radial MLP
        shift: per-atom energy shift
        scale: per-atom energy scale
        n_neighbors: constant number of per-atom neighbors, used for internal
        normalization
        scalar_mlp_std: standard deviation of weight init of radial MLP

    Returns:
        Potential energy of the inputs.
  """

  graph_net_steps: int
  use_sc: bool
  nonlinearities: Union[str, Dict[str, str]]
  n_elements: int

  hidden_irreps: str
  sh_irreps: str

  num_basis: int = 8
  r_max: float = 4.

  radial_net_nonlinearity: str = 'raw_swish'
  radial_net_n_hidden: int = 64
  radial_net_n_layers: int = 2

  shift: float = 0.
  scale: float = 1.
  n_neighbors: float = 1.
  scalar_mlp_std: float = 4.0

  @nn.compact
  def __call__(self, graph):
    r_max = jnp.float32(self.r_max)
    hidden_irreps = Irreps(self.hidden_irreps)

    # get src/dst from graph
    edge_src = graph.senders
    edge_dst = graph.receivers

    # node features
    embedding_irreps = Irreps(f'{self.n_elements}x0e')
    node_attrs = IrrepsArray(embedding_irreps, graph.nodes)

    # edge embedding
    _, scalar_dr_edge, edge_sh = graph.edges

    embedded_dr_edge = nn_util.BesselEmbedding(
        count=self.num_basis,
        inner_cutoff=r_max - 0.5,
        outer_cutoff=r_max
        )(scalar_dr_edge)

    # embedding layer
    h_node = Linear(irreps_out=Irreps(hidden_irreps))(node_attrs)

    # convolutions
    for _ in range(self.graph_net_steps):
      h_node = NequIPConvolution(
          hidden_irreps=hidden_irreps,
          use_sc=self.use_sc,
          nonlinearities=self.nonlinearities,
          radial_net_nonlinearity=self.radial_net_nonlinearity,
          radial_net_n_hidden=self.radial_net_n_hidden,
          radial_net_n_layers=self.radial_net_n_layers,
          num_basis=self.num_basis,
          n_neighbors=self.n_neighbors,
          scalar_mlp_std=self.scalar_mlp_std
          )(h_node,
            node_attrs,
            edge_sh,
            edge_src,
            edge_dst,
            embedded_dr_edge
            )

    # output block, two Linears that decay dimensions from h to h//2 to 1
    for mul, ir in h_node.irreps:
      if ir == Irrep('0e'):
        mul_second_to_final = mul // 2

    second_to_final_irreps = Irreps(f'{mul_second_to_final}x0e')
    final_irreps = Irreps('1x0e')

    h_node = Linear(irreps_out=second_to_final_irreps)(h_node)
    atomic_output = Linear(irreps_out=final_irreps)(h_node).array

    # shift + scale atomic energies
    atomic_output = self.scale * atomic_output + self.shift

    # this aggregation follows jraph/_src/models.py
    n_graph = graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = tree_util.tree_leaves(graph.nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx,
        graph.n_node,
        axis=0,
        total_repeat_length=sum_n_node
        )

    global_output = tree_util.tree_map(
        lambda n: jraph.segment_sum(
            n,
            node_gr_idx,
            n_graph
            ), atomic_output)

    return global_output


def model_from_config(cfg: ConfigDict
                      ) -> Tuple[FeaturizerFn, NequIPEnergyModel]:
  """Model replication of NequIP.

  Implementation follows the original paper by Batzner et al.

  nature.com/articles/s41467-022-29939-5 and partially
  https://github.com/mir-group/nequip.
  """
  shift, scale = nn_util.get_shift_and_scale(cfg)

  model = NequIPEnergyModel(
      graph_net_steps=cfg.graph_net_steps,
      use_sc=cfg.use_sc,
      nonlinearities=cfg.nonlinearities,
      n_elements=cfg.n_elements,
      hidden_irreps=cfg.hidden_irreps,
      sh_irreps=cfg.sh_irreps,
      num_basis=cfg.num_basis,
      r_max=cfg.r_max,
      radial_net_nonlinearity=cfg.radial_net_nonlinearity,
      radial_net_n_hidden=cfg.radial_net_n_hidden,
      radial_net_n_layers=cfg.radial_net_n_layers,
      shift=shift,
      scale=scale,
      n_neighbors=cfg.n_neighbors,
      scalar_mlp_std=cfg.scalar_mlp_std,
      )

  return featurizer(cfg.sh_irreps), model
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


from typing import Dict, Union, Tuple


import functools

import e3nn_jax as e3nn

import flax.linen as nn

import jax
import jax.numpy as jnp
from jax import tree_util
from jax import vmap

from jax_md import space
from jax_md import util

import jraph

from . import util as nn_util

from .e3nn_layer import FullyConnectedTensorProductE3nn
from .e3nn_layer import Linear

import operator

from ml_collections import ConfigDict


# Types


Irreps = e3nn.Irreps
IrrepsArray = e3nn.IrrepsArray
Irrep = e3nn.Irrep
Array = util.Array
FeaturizerFn = nn_util.FeaturizerFn


get_nonlinearity_by_name = nn_util.get_nonlinearity_by_name
partial = functools.partial


# Code


def featurizer(sh_irreps: str)-> FeaturizerFn:
  """Node, edge, global features for NequIP."""
  def node_feature_fn(graph):
    """ NequIP node features are simply the input graph's nodes. """
    return graph.nodes

  def edge_feature_fn(dR):
    """ NequIP edge features.

     Builds the following:
     - the relative interatomic positions \vec{r}_{ij}
     - interatomic distances r_{ij}
     - normalized spherical harmonics Y(\hat{r}_{ij})
     """
    features = []
    edge_sh_irreps = Irreps(sh_irreps)
    edge_sh = e3nn.spherical_harmonics(edge_sh_irreps, dR, normalize=True)
    return dR, space.distance(dR), edge_sh

  def global_feature_fn(g):
    """ NequIP does not use global features. """
    return g

  return node_feature_fn, edge_feature_fn, global_feature_fn


def prod(xs):
  """From e3nn_jax/util/__init__.py."""
  return functools.reduce(operator.mul, xs, 1)


def tp_path_exists(arg_in1, arg_in2, arg_out):
  """Check if a tensor product path is viable.

  This helper function is similar to the one used in:
  https://github.com/e3nn/e3nn
  """
  arg_in1 = Irreps(arg_in1).simplify()
  arg_in2 = Irreps(arg_in2).simplify()
  arg_out = Irrep(arg_out)

  for multiplicity_1, irreps_1 in arg_in1:
    for multiplicity_2, irreps_2 in arg_in2:
      if arg_out in irreps_1 * irreps_2:
        return True
  return False


class NequIPConvolution(nn.Module):
  """NequIP Convolution.

  Implementation follows the original paper by Batzner et al.

  nature.com/articles/s41467-022-29939-5 and partially
  https://github.com/mir-group/nequip.

  Args:
        hidden_irreps: irreducible representation of hidden/latent features
        use_sc: use self-connection in network (recommended)
        nonlinearities: nonlinearities to use for even/odd irreps
        radial_net_nonlinearity: nonlinearity to use in radial MLP
        radial_net_n_hidden: number of hidden neurons in radial MLP
        radial_net_n_layers: number of hidden layers for radial MLP
        num_basis: number of Bessel basis functions to use
        n_neighbors: constant number of per-atom neighbors, used for internal
        normalization
        scalar_mlp_std: standard deviation of weight init of radial MLP

    Returns:
        Updated node features h after the convolution.
  """
  hidden_irreps: Irreps
  use_sc: bool
  nonlinearities: Union[str, Dict[str, str]]
  radial_net_nonlinearity: str = 'raw_swish'
  radial_net_n_hidden: int = 64
  radial_net_n_layers: int = 2
  num_basis: int = 8
  n_neighbors: float = 1.
  scalar_mlp_std: float = 4.0

  @nn.compact
  def __call__(
      self,
      node_features: IrrepsArray,
      node_attributes: IrrepsArray,
      edge_sh: Array,
      edge_src: Array,
      edge_dst: Array,
      edge_embedded: Array
      ) -> IrrepsArray:
    # Convolution outline in NequIP is:
    # Linear on nodes
    # TP + aggregate
    # divide by average number of neighbors
    # Concatenation
    # Linear on nodes
    # Self-connection
    # Gate
    irreps_scalars = []
    irreps_nonscalars = []
    irreps_gate_scalars = []

    # get scalar target irreps
    for multiplicity, irrep in self.hidden_irreps:
      # need the additional Irrep() here for the build, even though irrep is
      # already of type Irrep()
      if (Irrep(irrep).l == 0 and
          tp_path_exists(node_features.irreps, edge_sh.irreps, irrep)):
        irreps_scalars += [(multiplicity, irrep)]

    irreps_scalars = Irreps(irreps_scalars)

    # get non-scalar target irreps
    for multiplicity, irrep in self.hidden_irreps:
      # need the additional Irrep() here for the build, even though irrep is
      # already of type Irrep()
      if (Irrep(irrep).l > 0 and
          tp_path_exists(node_features.irreps, edge_sh.irreps, irrep)):
        irreps_nonscalars += [(multiplicity, irrep)]

    irreps_nonscalars = Irreps(irreps_nonscalars)

    # get gate scalar irreps
    if tp_path_exists(node_features.irreps, edge_sh.irreps, '0e'):
      gate_scalar_irreps_type = '0e'
    else:
      gate_scalar_irreps_type = '0o'

    for multiplicity, irreps in irreps_nonscalars:
      irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]

    irreps_gate_scalars = Irreps(irreps_gate_scalars)

    # final layer output irreps are all three
    # note that this order is assumed by the gate function later, i.e.
    # scalars left, then gate scalar, then non-scalars
    h_out_irreps = irreps_scalars + irreps_gate_scalars + irreps_nonscalars

    # self-connection: TP between node features and node attributes
    # this can equivalently be seen as a matrix multiplication acting on
    # the node features where the weight matrix is indexed by the node
    # attributes (typically the chemical species), i.e. a linear transform
    # that is a function of the species of the central atom
    if self.use_sc:
      self_connection = FullyConnectedTensorProductE3nn(
          h_out_irreps,
          # node_features.irreps,
          # node_attributes.irreps
          )(node_features, node_attributes)

    h = node_features

    # first linear, stays in current h-space
    h = Linear(node_features.irreps)(h)

    # map node features onto edges for tp
    edge_features = jax.tree_map(lambda x: x[edge_src], h)

    # we gather the instructions for the tp as well as the tp output irreps
    mode = 'uvu'
    trainable = 'True'
    irreps_after_tp = []
    instructions = []

    # iterate over both arguments, i.e. node irreps and edge irreps
    # if they have a valid TP path for any of the target irreps,
    # add to instructions and put in appropriate position
    # we use uvu mode (where v is a single-element sum) and weights will
    # be provide externally by the scalar MLP
    # this triple for loop is similar to the one used in e3nn and nequip
    for i, (mul_in1, irreps_in1) in enumerate(node_features.irreps):
      for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
        for curr_irreps_out in irreps_in1 * irreps_in2:
          if curr_irreps_out in h_out_irreps:
            k = len(irreps_after_tp)
            irreps_after_tp += [(mul_in1, curr_irreps_out)]
            instructions += [(i, j, k, mode, trainable)]

    # we will likely have constructed irreps in a non-l-increasing order
    # so we sort them to be in a l-increasing order
    irreps_after_tp, p, _ = Irreps(irreps_after_tp).sort()

    # if we sort the target irreps, we will have to sort the instructions
    # acoordingly, using the permutation indices
    sorted_instructions = []

    for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
      sorted_instructions += [
          (irreps_in1, irreps_in2, p[irreps_out], mode, trainable)]

    # TP between spherical harmonics embedding of the edge vector
    # Y_ij(\hat{r}) and neighboring node h_j, weighted on a per-element basis
    # by the radial network R(r_ij)
    tp = e3nn.FunctionalTensorProduct(
        irreps_in1=edge_features.irreps,
        irreps_in2=edge_sh.irreps,
        irreps_out=irreps_after_tp,
        instructions=sorted_instructions
    )

    # scalar radial network, number of output neurons is the total number of
    # tensor product paths, nonlinearity must have f(0)=0 and MLP must not
    # have biases
    n_tp_weights = 0

    # get output dim of radial MLP / number of TP weights
    for ins in tp.instructions:
      if ins.has_weight:
        n_tp_weights += prod(ins.path_shape)

    # build radial MLP R(r) that maps from interatomic distances to TP weights
    # must not use bias to that R(0)=0
    fc = nn_util.MLP(
        (self.radial_net_n_hidden,) * self.radial_net_n_layers + (n_tp_weights,),
        self.radial_net_nonlinearity,
        use_bias=False,
        scalar_mlp_std=self.scalar_mlp_std
        )

    # the TP weights (v dimension) are given by the FC
    weight = fc(edge_embedded)

    # tp between node features that have been mapped onto edges and edge RSH
    # weighted by FC weight, we vmap over the dimension of the edges
    edge_features = jax.vmap(tp.left_right)(weight, edge_features, edge_sh)
    # TODO: It's not great that e3nn_jax automatically upcasts internally,
    # but this would need to be fixed at the e3nn level.
    edge_features = jax.tree_map(lambda x: x.astype(h.dtype), edge_features)

    # aggregate edges onto nodes after tp using e3nn-jax's index_add
    h_type = h.dtype
    h = jax.tree_map(
        lambda x: e3nn.index_add(edge_dst, x, out_dim=h.shape[0]),
        edge_features
        )
    # TODO: Remove this once e3nn_jax doesn't upcast inputs.
    h = jax.tree_map(lambda x: x.astype(h_type), h)

    # normalize by the average (not local) number of neighbors
    h = h / self.n_neighbors

    # second linear, now we create extra gate scalars by mapping to h-out
    h = Linear(h_out_irreps)(h)

    # self-connection, similar to a resnet-update that sums the output from
    # the TP to chemistry-weighted h
    if self.use_sc:
      h = h + self_connection

    # gate nonlinearity, applied to gate data, consisting of:
    # a) regular scalars,
    # b) gate scalars, and
    # c) non-scalars to be gated
    # in this order
    gate_fn = partial(
        e3nn.gate,
        even_act=get_nonlinearity_by_name(self.nonlinearities['e']),
        odd_act=get_nonlinearity_by_name(self.nonlinearities['o']),
        even_gate_act=get_nonlinearity_by_name(self.nonlinearities['e']),
        odd_gate_act=get_nonlinearity_by_name(self.nonlinearities['o'])
        )

    h = gate_fn(h)
    # TODO: Remove this once e3nn_jax doesn't upcast inputs.
    h = jax.tree_map(lambda x: x.astype(h_type), h)

    return h


class NequIPEnergyModel(nn.Module):
  """NequIP.

  Implementation follows the original paper by Batzner et al.

  nature.com/articles/s41467-022-29939-5 and partially
  https://github.com/mir-group/nequip.

    Args:
        graph_net_steps: number of NequIP convolutional layers
        use_sc: use self-connection in network (recommended)
        nonlinearities: nonlinearities to use for even/odd irreps
        n_element: number of chemical elements in input data
        hidden_irreps: irreducible representation of hidden/latent features
        sh_irreps: irreducible representations on the edges
        num_basis: number of Bessel basis functions to use
        r_max: radial cutoff used in length units
        radial_net_nonlinearity: nonlinearity to use in radial MLP
        radial_net_n_hidden: number of hidden neurons in radial MLP
        radial_net_n_layers: number of hidden layers for radial MLP
        shift: per-atom energy shift
        scale: per-atom energy scale
        n_neighbors: constant number of per-atom neighbors, used for internal
        normalization
        scalar_mlp_std: standard deviation of weight init of radial MLP

    Returns:
        Potential energy of the inputs.
  """

  graph_net_steps: int
  use_sc: bool
  nonlinearities: Union[str, Dict[str, str]]
  n_elements: int

  hidden_irreps: str
  sh_irreps: str

  num_basis: int = 8
  r_max: float = 4.

  radial_net_nonlinearity: str = 'raw_swish'
  radial_net_n_hidden: int = 64
  radial_net_n_layers: int = 2

  shift: float = 0.
  scale: float = 1.
  n_neighbors: float = 1.
  scalar_mlp_std: float = 4.0

  @nn.compact
  def __call__(self, graph):
    r_max = jnp.float32(self.r_max)
    hidden_irreps = Irreps(self.hidden_irreps)

    # get src/dst from graph
    edge_src = graph.senders
    edge_dst = graph.receivers

    # node features
    embedding_irreps = Irreps(f'{self.n_elements}x0e')
    node_attrs = IrrepsArray(embedding_irreps, graph.nodes)

    # edge embedding
    _, scalar_dr_edge, edge_sh = graph.edges

    embedded_dr_edge = nn_util.BesselEmbedding(
        count=self.num_basis,
        inner_cutoff=r_max - 0.5,
        outer_cutoff=r_max
        )(scalar_dr_edge)

    # embedding layer
    h_node = Linear(irreps_out=Irreps(hidden_irreps))(node_attrs)

    # convolutions
    for _ in range(self.graph_net_steps):
      h_node = NequIPConvolution(
          hidden_irreps=hidden_irreps,
          use_sc=self.use_sc,
          nonlinearities=self.nonlinearities,
          radial_net_nonlinearity=self.radial_net_nonlinearity,
          radial_net_n_hidden=self.radial_net_n_hidden,
          radial_net_n_layers=self.radial_net_n_layers,
          num_basis=self.num_basis,
          n_neighbors=self.n_neighbors,
          scalar_mlp_std=self.scalar_mlp_std
          )(h_node,
            node_attrs,
            edge_sh,
            edge_src,
            edge_dst,
            embedded_dr_edge
            )

    # output block, two Linears that decay dimensions from h to h//2 to 1
    for mul, ir in h_node.irreps:
      if ir == Irrep('0e'):
        mul_second_to_final = mul // 2

    second_to_final_irreps = Irreps(f'{mul_second_to_final}x0e')
    final_irreps = Irreps('1x0e')

    h_node = Linear(irreps_out=second_to_final_irreps)(h_node)
    atomic_output = Linear(irreps_out=final_irreps)(h_node).array

    # shift + scale atomic energies
    atomic_output = self.scale * atomic_output + self.shift

    # this aggregation follows jraph/_src/models.py
    n_graph = graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = tree_util.tree_leaves(graph.nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx,
        graph.n_node,
        axis=0,
        total_repeat_length=sum_n_node
        )

    global_output = tree_util.tree_map(
        lambda n: jraph.segment_sum(
            n,
            node_gr_idx,
            n_graph
            ), atomic_output)

    return global_output


def model_from_config(cfg: ConfigDict
                      ) -> Tuple[FeaturizerFn, NequIPEnergyModel]:
  """Model replication of NequIP.

  Implementation follows the original paper by Batzner et al.

  nature.com/articles/s41467-022-29939-5 and partially
  https://github.com/mir-group/nequip.
  """
  shift, scale = nn_util.get_shift_and_scale(cfg)

  model = NequIPEnergyModel(
      graph_net_steps=cfg.graph_net_steps,
      use_sc=cfg.use_sc,
      nonlinearities=cfg.nonlinearities,
      n_elements=cfg.n_elements,
      hidden_irreps=cfg.hidden_irreps,
      sh_irreps=cfg.sh_irreps,
      num_basis=cfg.num_basis,
      r_max=cfg.r_max,
      radial_net_nonlinearity=cfg.radial_net_nonlinearity,
      radial_net_n_hidden=cfg.radial_net_n_hidden,
      radial_net_n_layers=cfg.radial_net_n_layers,
      shift=shift,
      scale=scale,
      n_neighbors=cfg.n_neighbors,
      scalar_mlp_std=cfg.scalar_mlp_std,
      )

  return featurizer(cfg.sh_irreps), model


def default_config() -> ConfigDict:
  config = ConfigDict()

  config.graph_net_steps = 5
  config.nonlinearities = {'e': 'raw_swish', 'o': 'tanh'}
  config.use_sc = True
  config.n_elements = 94
  config.hidden_irreps = '128x0e + 64x1e + 4x2e'
  config.sh_irreps = '1x0e + 1x1e + 1x2e'
  config.num_basis = 8
  config.r_max = 5.
  config.radial_net_nonlinearity = 'raw_swish'
  config.radial_net_n_hidden = 64
  config.radial_net_n_layers = 2

  # average number of neighbors per atom, used to divide activations are sum
  # in the nequip convolution, helpful for internal normalization.
  config.n_neighbors = 10.

  # Standard deviation used for the initializer of the weight matrix in the
  # radial scalar MLP
  config.scalar_mlp_std = 4.

  return config
