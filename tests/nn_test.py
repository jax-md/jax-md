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

"""Tests jax_md.nn."""

from absl.testing import absltest
from absl.testing import parameterized

from jax.config import config as jax_config
from jax import random
import jax.numpy as np

import numpy as onp

from jax import jit, grad
from jax_md import space, quantity, nn, dataclasses, partition
from jax_md.util import f32, f64
from jax_md.test_util import update_test_tolerance

from jax import test_util as jtu

jax_config.parse_flags_with_absl()
jax_config.enable_omnistaging()
FLAGS = jax_config.FLAGS

if FLAGS.jax_enable_x64:
  DTYPES = [f32, f64]
else:
  DTYPES = [f32]

N_TYPES_TO_TEST = [1, 2]
N_ETAS_TO_TEST = [1, 2]


@jtu.with_config(jax_numpy_rank_promotion='allow')
class SymmetryFunctionTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_N_types={}_N_etas={}_d_type={}'.format(
              N_types, N_etas, dtype.__name__),
          'dtype': dtype,
          'N_types': N_types,
          'N_etas': N_etas,
      } for N_types in N_TYPES_TO_TEST
        for N_etas in N_ETAS_TO_TEST
        for dtype in DTYPES))
  def test_radial_symmetry_functions(self, N_types, N_etas, dtype):
    displacement, shift = space.free()
    rs = np.linspace(1.0, 2.0, N_etas, dtype=dtype)
    gr = nn.radial_symmetry_functions(displacement,
                                      np.array([1, 1, N_types]),
                                      rs,
                                      4)
    R = np.array([[0,0,0], [1,1,1], [1,1,0]], dtype)
    gr_out = gr(R)
    self.assertAllClose(gr_out.shape, (3, N_types * N_etas))
    self.assertAllClose(gr_out[2, 0], dtype(0.411717), rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_N_types={}_N_etas={}_dtype={}_dim={}'.format(
              N_types, N_etas, dtype.__name__, dim),
          'dtype': dtype,
          'N_types': N_types,
          'N_etas': N_etas,
          'dim': dim,
      } for N_types in N_TYPES_TO_TEST
        for N_etas in N_ETAS_TO_TEST
        for dtype in DTYPES
        for dim in [2, 3]))
  def test_radial_symmetry_functions_neighbor_list(self,
                                                   N_types,
                                                   N_etas,
                                                   dtype,
                                                   dim):
    key = random.PRNGKey(0)

    N = 128
    box_size = 12.0
    r_cutoff = 3.

    displacement, shift = space.periodic(box_size)
    R_key, species_key = random.split(key)
    R = box_size * random.uniform(R_key, (N, dim))
    species = random.choice(species_key, N_types, (N,))

    neighbor_fn = partition.neighbor_list(displacement, box_size, r_cutoff, 0.)

    rs = np.linspace(1.0, 2.0, N_etas, dtype=dtype)
    gr = nn.radial_symmetry_functions(displacement, species, rs, r_cutoff)
    gr_neigh = nn.radial_symmetry_functions_neighbor_list(
      displacement,
      species,
      np.linspace(1.0, 2.0, N_etas, dtype=dtype),
      r_cutoff)
    nbrs = neighbor_fn(R)
    gr_exact = gr(R)
    gr_nbrs = gr_neigh(R, neighbor=nbrs)

    tol = 1e-13 if FLAGS.jax_enable_x64 else 1e-6
    self.assertAllClose(gr_exact, gr_nbrs, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_N_types={}_N_etas={}_dtype={}_dim={}'.format(
              N_types, N_etas, dtype.__name__, dim),
          'dtype': dtype,
          'N_types': N_types,
          'N_etas': N_etas,
          'dim': dim,
      } for N_types in N_TYPES_TO_TEST
        for N_etas in N_ETAS_TO_TEST
        for dtype in DTYPES
        for dim in [2, 3]))
  def test_angular_symmetry_functions_neighbor_list(self,
                                                    N_types,
                                                    N_etas,
                                                    dtype,
                                                    dim):
    key = random.PRNGKey(0)

    N = 128
    box_size = 12.0
    r_cutoff = 3.

    displacement, shift = space.periodic(box_size)
    R_key, species_key = random.split(key)
    R = box_size * random.uniform(R_key, (N, dim))
    species = random.choice(species_key, N_types, (N,))

    neighbor_fn = partition.neighbor_list(displacement, box_size, r_cutoff, 0.)

    etas = np.linspace(1., 2., N_etas, dtype=dtype)
    lam = np.array([-1.0] * N_etas, dtype)
    gr = nn.angular_symmetry_functions(displacement,
                                       species,
                                       etas=etas,
                                       lambdas=lam,
                                       zetas=np.array([1.0] * N_etas, dtype),
                                       cutoff_distance=r_cutoff)

    gr_neigh = nn.angular_symmetry_functions_neighbor_list(
      displacement,
      species,
      etas=etas,
      lambdas=np.array([-1.0] * N_etas, dtype),
      zetas=np.array([1.0] * N_etas, dtype),
      cutoff_distance=r_cutoff)

    nbrs = neighbor_fn(R)
    gr_exact = gr(R)
    gr_nbrs = gr_neigh(R, neighbor=nbrs)

    tol = 1e-13 if FLAGS.jax_enable_x64 else 1e-6
    self.assertAllClose(gr_exact, gr_nbrs, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_N_types={}_N_etas={}_d_type={}'.format(
              N_types, N_etas, dtype.__name__),
          'dtype': dtype,
          'N_types': N_types,
          'N_etas': N_etas,
      } for N_types in N_TYPES_TO_TEST
        for N_etas in N_ETAS_TO_TEST
        for dtype in DTYPES))
  def test_angular_symmetry_functions(self, N_types, N_etas, dtype):
    displacement, shift = space.free()
    etas = np.array([1e-4/(0.529177 ** 2)] * N_etas, dtype)
    lam = np.array([-1.0] * N_etas, dtype)
    gr = nn.angular_symmetry_functions(displacement,np.array([1, 1, N_types]),
                                       etas=etas,
                                       lambdas=lam,
                                       zetas=np.array([1.0] * N_etas, dtype),
                                       cutoff_distance=8.0)
    R = np.array([[0,0,0], [1,1,1], [1,1,0]], dtype)
    gr_out = gr(R)
    self.assertAllClose(gr_out.shape,
                        (3, N_etas *  N_types * (N_types + 1) // 2))
    self.assertAllClose(gr_out[2, 0], dtype(1.577944), rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_N_types={}_N_etas={}_d_type={}'.format(
              N_types, N_etas, dtype.__name__),
          'dtype': dtype,
          'N_types': N_types,
          'N_etas': N_etas,
      } for N_types in N_TYPES_TO_TEST
        for N_etas in N_ETAS_TO_TEST
        for dtype in DTYPES))
  def test_behler_parrinello_symmetry_functions(self, N_types, N_etas, dtype):
    displacement, shift = space.free()
    gr = nn.behler_parrinello_symmetry_functions(
            displacement,np.array([1, 1, N_types]),
            radial_etas=np.array([1e-4/(0.529177 ** 2)] * N_etas, dtype),
            angular_etas=np.array([1e-4/(0.529177 ** 2)] * N_etas, dtype),
            lambdas=np.array([-1.0] * N_etas, dtype),
            zetas=np.array([1.0] * N_etas, dtype),
            cutoff_distance=8.0)
    R = np.array([[0,0,0], [1,1,1], [1,1,0]], dtype)
    gr_out = gr(R)
    self.assertAllClose(gr_out.shape,
                        (3,
                         N_etas *  (N_types + N_types * (N_types + 1) // 2)))
    self.assertAllClose(gr_out[2, 0], dtype(1.885791), rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_N_types={}_N_etas={}_d_type={}'.format(
              N_types, N_etas, dtype.__name__),
          'dtype': dtype,
          'N_types': N_types,
          'N_etas': N_etas,
      } for N_types in N_TYPES_TO_TEST
        for N_etas in N_ETAS_TO_TEST
        for dtype in DTYPES))
  def test_behler_parrinello_symmetry_functions_neighbor_list(self,
                                                              N_types,
                                                              N_etas,
                                                              dtype):
    displacement, shift = space.free()
    neighbor_fn = partition.neighbor_list(displacement, 10.0, 8.0, 0.0)
    gr = nn.behler_parrinello_symmetry_functions_neighbor_list(
            displacement,np.array([1, 1, N_types]),
            radial_etas=np.array([1e-4/(0.529177 ** 2)] * N_etas, dtype),
            angular_etas=np.array([1e-4/(0.529177 ** 2)] * N_etas, dtype),
            lambdas=np.array([-1.0] * N_etas, dtype),
            zetas=np.array([1.0] * N_etas, dtype),
            cutoff_distance=8.0)
    R = np.array([[0,0,0], [1,1,1], [1,1,0]], dtype)
    nbrs = neighbor_fn(R)
    gr_out = gr(R, neighbor=nbrs)
    self.assertAllClose(gr_out.shape,
                        (3, N_etas *  (N_types + N_types * (N_types + 1) // 2)))
    self.assertAllClose(gr_out[2, 0], dtype(1.885791), rtol=1e-6, atol=1e-6)

def _graph_network(graph_tuple):
  update_node_fn = lambda n, se, re, g: n
  update_edge_fn = lambda e, sn, rn, g: e
  update_global_fn = lambda gn, ge, g: g

  net = nn.GraphNetwork(update_edge_fn,
                        update_node_fn,
                        update_global_fn)

  return net(graph_tuple)

def _graph_network_no_node_update(graph_tuple):
  update_node_fn = None
  update_edge_fn = lambda e, sn, rn, g: e
  update_global_fn = lambda gn, ge, g: g

  net = nn.GraphNetwork(update_edge_fn,
                        update_node_fn,
                        update_global_fn)

  return net(graph_tuple)

def _graph_network_no_edge_update(graph_tuple):
  update_node_fn = lambda n, se, re, g: n
  update_edge_fn = None
  update_global_fn = lambda gn, ge, g: g

  net = nn.GraphNetwork(update_edge_fn,
                        update_node_fn,
                        update_global_fn)

  return net(graph_tuple)

def _graph_network_no_global_update(graph_tuple):
  update_node_fn = lambda n, se, re, g: n
  update_edge_fn = lambda e, sn, rn, g: e
  update_global_fn = None

  net = nn.GraphNetwork(update_edge_fn,
                        update_node_fn,
                        update_global_fn)

  return net(graph_tuple)

def _graph_independent(graph_tuple):
  id = lambda x: x
  net = nn.GraphMapFeatures(id, id, id)
  return net(graph_tuple)

GRAPH_NETWORKS = [
    _graph_network,
    _graph_network_no_node_update,
    _graph_network_no_edge_update,
    _graph_network_no_global_update,
    _graph_independent
]


def _get_graphs():
    return [
    nn.GraphsTuple(
        nodes=np.array([[1.0], [2.0]]),
        edges=np.array([[[1.0], [2.0]],
                        [[3.0], [4.0]]]),
        globals=np.array([1.0]),
        edge_idx=np.array([[0, 1,], [0, 1]])
    ),
    nn.GraphsTuple(
        nodes=np.array([[1.0], [2.0]]),
        edges=np.array([[[1.0], [2.0]],
                        [[3.0], [4.0]]]),
        globals=np.array([1.0]),
        edge_idx=np.array([[0, 1,], [2, 1]])
    )
  ]


@jtu.with_config(jax_numpy_rank_promotion="allow")
class NeuralNetworkTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_fn={}_dtype={}'.format(fn.__name__, dtype.__name__),
          'network_fn': fn,
          'dtype': dtype
      } for fn in GRAPH_NETWORKS for dtype in DTYPES))
  def test_connect_graph_network(self, network_fn, dtype):
    for g in _get_graphs():
      g = dataclasses.replace(
          g,
          nodes=np.array(g.nodes, dtype),
          edges=np.array(g.edges, dtype),
          globals=np.array(g.globals, dtype))
      with self.subTest('nojit'):
        out = network_fn(g)
        self.assertGraphsTuplesClose(out, g)
      with self.subTest('jit'):
        out = jit(network_fn)(g)
        self.assertGraphsTuplesClose(out, g)

  def assertGraphsTuplesClose(self, a, b, tol=1e-6):
    a_mask = (a.edge_idx < a.nodes.shape[0]).reshape(a.edge_idx.shape + (1,))
    b_mask = (b.edge_idx < b.nodes.shape[0]).reshape(b.edge_idx.shape + (1,))

    a = dataclasses.replace(a, edges=a.edges * a_mask)
    b = dataclasses.replace(b, edges=b.edges * b_mask)

    a = dataclasses.asdict(a)
    b = dataclasses.asdict(b)

    self.assertAllClose(a, b)


if __name__ == '__main__':
  absltest.main()
