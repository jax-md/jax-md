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

import dataclasses
from types import SimpleNamespace
from typing import Any, Dict

import jax
import os

from absl.testing import absltest
import e3nn_jax as e3nn
import equinox as eqx
from jax import random
from jax_md import test_util
from jax_md import util
from jax_md import space
from jax_md import energy
from jax_md import custom_partition
from jax_md.nn import nequip


import jax.numpy as jnp
import numpy as onp

# Full precision for the parity and checkpoint comparisons
jax.config.update('jax_default_matmul_precision', 'highest')


def cfg_for(type_names=('Si',), **overrides):
  return dict(type_names=type_names, num_layers=2, r_max=4.0) | overrides


class NequIPFromConfigTest(test_util.JAXMDTestCase):
  """Models built with nequip.from_config."""

  def test_edge_mlp_width(self):
    model = nequip.from_config(random.PRNGKey(0), **cfg_for())
    for layer in model.layers:
      needed = sum(
        nequip.irreps_blocks(layer.tp.irreps_in1)[i_1][0]
        for i_1, _, _, _ in layer.tp.instructions
      )
      self.assertEqual(layer.edge_mlp.weights[-1].shape[1], needed)

  def test_last_layer_scalar_output(self):
    model = nequip.from_config(random.PRNGKey(0), **cfg_for())
    out = e3nn.Irreps(model.layers[-1].tp.irreps_out)
    self.assertTrue(all(ir.l == 0 for _, ir in out))

  def test_equivariance(self):
    cfg = cfg_for(type_names=('Si', 'O', 'C'))
    model = nequip.from_config(random.PRNGKey(0), **cfg)
    position, species, senders, receivers = cluster(r_max=cfg['r_max'])

    # float32 sums reassociate under rotation; use float64 when enabled
    if util.x64_enabled():
      model = jax.tree.map(lambda x: x.astype(jnp.float64), model)
      position = position.astype(jnp.float64)
      tol = dict(atol=1e-10, rtol=1e-10)
    else:
      tol = dict(atol=2e-3, rtol=2e-3)

    def total(R):
      return jnp.sum(model(R, species, senders, receivers))

    rotation = e3nn.rand_matrix(random.PRNGKey(0), dtype=position.dtype)
    shifted = position + jnp.array([0.3, -0.7, 1.1], position.dtype)
    self.assertAllClose(total(position), total(shifted), **tol)
    self.assertAllClose(total(position), total(position @ rotation.T), **tol)

    forces = -jax.grad(total)(position)
    rotated = -jax.grad(total)(position @ rotation.T)
    self.assertAllClose(forces @ rotation.T, rotated, **tol)

  def test_neighbor_list_energy(self):
    cfg = cfg_for(type_names=('Si', 'O', 'C'))
    box = 30.0
    displacement_fn, _ = space.periodic(box)
    position, species, senders, receivers = cluster(r_max=cfg['r_max'])
    position = position - position.mean(0) + box / 2

    neighbor_fn, init_fn, energy_fn = energy.nequip_neighbor_list(
      displacement_fn, box, cfg, species=species
    )
    neighbor = neighbor_fn.allocate(position)
    model = init_fn(random.PRNGKey(0), position, neighbor)

    direct = jnp.sum(model(position, species, senders, receivers))
    self.assertAllClose(
      energy_fn(model, position, neighbor), direct, atol=1e-4, rtol=1e-4
    )

  def test_neighbor_list_model_and_featurizer(self):
    cfg = cfg_for(type_names=('Si', 'O', 'C'))
    model = nequip.from_config(random.PRNGKey(0), **cfg)
    box = 30.0 * jnp.eye(3)
    position, species, senders, receivers = cluster(r_max=cfg['r_max'])
    fractional = (position - position.mean(0) + 15.0) / 30.0
    direct = jnp.sum(model(position, species, senders, receivers))
    displacement_fn, _ = space.periodic_general(box)
    multi_image: Dict[str, Any] = dict(
      neighbor_list_fn=custom_partition.neighbor_list_multi_image,
      featurizer_fn=custom_partition.graph_featurizer,
    )
    # the featurizer applies the lattice shifts itself
    for displacement, kwargs in ((displacement_fn, {}), (None, multi_image)):
      neighbor_fn, init_fn, energy_fn = energy.nequip_neighbor_list(
        displacement,
        box,
        model=model,
        species=species,
        fractional_coordinates=True,
        **kwargs,
      )
      neighbor = neighbor_fn.allocate(fractional)
      self.assertIs(init_fn(random.PRNGKey(0), fractional, neighbor), model)
      self.assertAllClose(
        energy_fn(model, fractional, neighbor), direct, atol=1e-4, rtol=1e-4
      )
    with self.assertRaisesRegex(ValueError, 'displacement_fn=None'):
      energy.nequip_neighbor_list(
        displacement_fn, box, model=model, **multi_image
      )

  def test_self_connection_flag(self):
    cfg = cfg_for(num_layers=3)
    on = nequip.from_config(random.PRNGKey(0), **cfg, self_connection=True)
    off = nequip.from_config(random.PRNGKey(0), **cfg, self_connection=False)
    # Layer 0 has no previous features to connect
    self.assertIsNone(on.layers[0].self_connection)
    self.assertIsNotNone(on.layers[1].self_connection)
    self.assertTrue(all(l.self_connection is None for l in off.layers))

  def test_hidden_irreps_order(self):
    args = cluster(r_max=4.0)
    scalars_first = nequip.from_config(
      random.PRNGKey(0), **cfg_for(hidden_irreps='16x0e + 8x1o')
    )
    gated_first = nequip.from_config(
      random.PRNGKey(0), **cfg_for(hidden_irreps='8x1o + 16x0e')
    )
    self.assertAllClose(scalars_first(*args), gated_first(*args))

  def test_mixed_parity_scalar_activations(self):
    # Odd scalars first arise from 1e ⊗ 1o, so they appear in the third layer.
    model = nequip.from_config(
      random.PRNGKey(0),
      **cfg_for(hidden_irreps='16x0e + 8x0o + 8x1o + 8x1e', num_layers=4),
    )
    self.assertEqual(model.layers[0].gate.act_scalars, ('silu',))
    gate = model.layers[2].gate
    self.assertEqual(gate.act_scalars, ('silu', 'tanh'))
    # e3nn's constants for silu and tanh, estimated from 1e6 seeded samples.
    silu, tanh = gate.act_scalar_scales
    self.assertAlmostEqual(silu, 1.679177, delta=4e-3)
    self.assertAlmostEqual(tanh, 1.593733, delta=4e-3)
    self.assertEqual(gate.act_gate_scales, (silu, silu))
    self.assertAlmostEqual(nequip.second_moment_scale('none'), 1.0, places=5)
    energy = model(*cluster(r_max=4.0))
    self.assertTrue(onp.all(onp.isfinite(onp.asarray(energy))))

  def test_cutoff_continuity(self):
    cfg = cfg_for(type_names=('Si', 'O', 'C'))
    model = nequip.from_config(random.PRNGKey(0), **cfg)
    species = jnp.zeros(2, jnp.int32)
    senders, receivers = jnp.array([0, 1]), jnp.array([1, 0])
    pair = lambda r: jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]], jnp.float32)
    total = lambda r: jnp.sum(model(pair(r), species, senders, receivers))

    # Past r_max the edge weight is zero, so the pair decouples smoothly
    apart = total(cfg['r_max'] + 1.0)
    for r in (cfg['r_max'] - 1e-3, cfg['r_max'], cfg['r_max'] + 1e-3):
      self.assertAllClose(total(r), apart, atol=1e-5)

  def test_force_sum(self):
    cfg = cfg_for(type_names=('Si', 'O', 'C'))
    model = nequip.from_config(random.PRNGKey(0), **cfg)
    position, species, senders, receivers = cluster(r_max=cfg['r_max'])
    force = -jax.grad(lambda p: jnp.sum(model(p, species, senders, receivers)))(
      position
    )
    self.assertTrue(onp.all(onp.isfinite(onp.asarray(force))))
    self.assertAllClose(
      jnp.sum(force, axis=0), jnp.zeros(3, force.dtype), atol=1e-4
    )

  def test_bessel_basis(self):
    model = nequip.from_config(random.PRNGKey(0), **cfg_for(num_bessel=4))
    r = jnp.array([[0.7], [1.9], [3.2]], jnp.float32)
    # Upstream's (2 / r_max) sin(n pi r / r_max) / r, rearranged as sinc
    n = jnp.arange(1, 5, dtype=jnp.float32)
    bessel = 2.0 / model.r_max * jnp.sin(n * jnp.pi * r / model.r_max) / r
    envelope = nequip.polynomial_cutoff(r / model.r_max, model.cutoff_p)
    self.assertAllClose(
      model.radial_embedding(r), bessel * envelope, rtol=1e-5, atol=1e-6
    )

  def test_zbl_repulsion(self):
    plain = nequip.from_config(random.PRNGKey(0), **cfg_for())
    with_zbl = nequip.from_config(random.PRNGKey(0), **cfg_for(), zbl=True)
    self.assertEqual(with_zbl.zbl_atomic_numbers, (14.0,))
    # Fixed atomic numbers add no leaf
    self.assertEqual(
      len(jax.tree.leaves(with_zbl)), len(jax.tree.leaves(plain))
    )
    species = jnp.zeros(2, jnp.int32)
    senders, receivers = jnp.array([0, 1]), jnp.array([1, 0])
    pair = lambda r: jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]], jnp.float32)
    total = lambda m, r: float(jnp.sum(m(pair(r), species, senders, receivers)))
    # Same key, so only the pair term differs
    self.assertGreater(total(with_zbl, 0.5) - total(plain, 0.5), 100.0)
    self.assertAllClose(total(with_zbl, 3.9), total(plain, 3.9), atol=1e-2)
    with self.assertRaisesRegex(ValueError, 'chemical symbols'):
      nequip.from_config(
        random.PRNGKey(0), **cfg_for(type_names=('Si_a',)), zbl=True
      )

  def test_float64_positions(self):
    model = nequip.from_config(random.PRNGKey(0), **cfg_for())
    species = jnp.zeros(2, jnp.int32)
    senders, receivers = jnp.array([0, 1]), jnp.array([1, 0])
    position = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3]], jnp.float64)
    energies = model(position, species, senders, receivers)
    self.assertEqual(energies.dtype, model.type_embedding.dtype)
    total = lambda R: jnp.sum(model(R, species, senders, receivers))
    self.assertEqual(jax.grad(total)(position).dtype, position.dtype)

  def test_from_config_under_jit(self):
    nequip.second_moment_scale.cache_clear()
    build = lambda key: nequip.from_config(key, **cfg_for())
    traced = jax.jit(build)(random.PRNGKey(0))
    eager = build(random.PRNGKey(0))
    for a, b in zip(jax.tree.leaves(traced), jax.tree.leaves(eager)):
      self.assertAllClose(a, b)

  def test_per_layer_irreps_pruning(self):
    model = nequip.from_config(
      random.PRNGKey(0),
      **cfg_for(
        hidden_irreps='8x0e + 8x0o + 8x1o + 8x1e',
        irreps_edge_sh='1x0e + 1x1o',
        num_layers=4,
      ),
    )
    gates = [layer.gate for layer in model.layers]
    # 1e needs 1o ⊗ 1o first, then 0o needs 1o ⊗ 1e.
    self.assertEqual(e3nn.Irreps(gates[0].irreps_gated), e3nn.Irreps('8x1o'))
    self.assertEqual(
      e3nn.Irreps(gates[1].irreps_gated), e3nn.Irreps('8x1o + 8x1e')
    )
    self.assertEqual(
      e3nn.Irreps(gates[2].irreps_scalars), e3nn.Irreps('8x0e + 8x0o')
    )
    # only even scalars reach the readout
    self.assertEqual(e3nn.Irreps(gates[3].irreps_scalars), e3nn.Irreps('8x0e'))
    self.assertEqual(model.readout.weights[0].shape[0], 8)

  def test_scalar_only_irreps(self):
    for overrides in (
      dict(hidden_irreps='16x0e'),
      dict(hidden_irreps='16x0e + 8x0o'),
      dict(irreps_edge_sh='1x0e'),
    ):
      model = nequip.from_config(random.PRNGKey(0), **cfg_for(**overrides))
      energy = model(*cluster(r_max=4.0))
      self.assertTrue(onp.all(onp.isfinite(onp.asarray(energy))))
    with self.assertRaisesRegex(ValueError, 'no even scalars'):
      nequip.from_config(
        random.PRNGKey(0), **cfg_for(hidden_irreps='8x0o + 8x1o')
      )

  def test_parity_invariance(self):
    model = nequip.from_config(
      random.PRNGKey(1),
      **cfg_for(
        type_names=('Li', 'O', 'P'),
        hidden_irreps='16x0e + 8x0o + 8x1o + 8x1e',
        irreps_edge_sh='1x0e + 1x1o',
        num_layers=4,
      ),
    )
    position, species, senders, receivers = cluster(n=8)
    energy = model(position, species, senders, receivers)
    for flipped in (-position, position * jnp.array([1.0, 1.0, -1.0])):
      self.assertAllClose(
        energy,
        model(flipped, species, senders, receivers),
        atol=1e-4,
        rtol=1e-4,
      )

  def test_float64_round_trip(self):
    if not util.x64_enabled():
      self.skipTest('needs x64')
    cfg = cfg_for()
    model = jax.tree.map(
      lambda x: x.astype(jnp.float64),
      nequip.from_config(random.PRNGKey(0), **cfg),
    )
    path = os.path.join(absltest.get_default_test_tmpdir(), 'f64.eqx')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nequip.save_model(path, model, cfg)
    loaded, _ = nequip.load_model(path)
    self.assertEqual(loaded.type_embedding.dtype, jnp.float64)
    position, species, senders, receivers = cluster()
    self.assertAllClose(
      model(position, species, senders, receivers),
      loaded(position, species, senders, receivers),
    )

  def test_zbl_round_trip(self):
    cfg = cfg_for(type_names=('Si', 'O', 'C'), zbl=True)
    model = nequip.from_config(random.PRNGKey(0), **cfg)
    path = os.path.join(absltest.get_default_test_tmpdir(), 'nequip_zbl.eqx')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nequip.save_model(path, model, cfg)
    loaded, config = nequip.load_model(path)
    self.assertTrue(config['zbl'])
    args = cluster(r_max=cfg['r_max'])
    self.assertAllClose(model(*args), loaded(*args))


PATH = os.environ.get('JAX_MD_NEQUIP_GNN_MODEL')
PACKAGE = os.environ.get('JAX_MD_NEQUIP_PACKAGE')


def load_package(path):
  """The eager torch model inside a ``nequip-package`` archive, on the CPU."""
  import torch.package

  importer = torch.package.PackageImporter(path)
  models = importer.load_pickle('model', 'eager_model.pkl', map_location='cpu')
  return next(iter(models.values())).eval()


def skeleton(node):
  """A module's structure: static fields, nesting and leaf shapes."""
  if isinstance(node, eqx.Module):
    return {
      field.name: skeleton(getattr(node, field.name))
      for field in dataclasses.fields(node)
    }
  if isinstance(node, (tuple, list)):
    return [skeleton(value) for value in node]
  if hasattr(node, 'shape'):
    return tuple(node.shape)
  return node


def assert_same_structure(test, a, b, path='model'):
  if isinstance(a, dict):
    test.assertEqual(a.keys(), b.keys(), path)
    for key in a:
      assert_same_structure(test, a[key], b[key], f'{path}.{key}')
  elif isinstance(a, list):
    test.assertEqual(len(a), len(b), path)
    for i, (x, y) in enumerate(zip(a, b)):
      assert_same_structure(test, x, y, f'{path}[{i}]')
  elif isinstance(a, float):
    # e3nn-torch's path weights and its sampled activation constants
    test.assertAlmostEqual(a, b, delta=5e-3 * max(1.0, abs(a)), msg=path)
  else:
    test.assertEqual(a, b, path)


def small_description(seed=0):
  """A one-layer model, small enough to build by hand."""
  rng = onp.random.RandomState(seed)
  n_types, width = 3, 8

  def normal(*shape):
    return rng.randn(*shape).astype(onp.float32)

  return {
    'r_max': 4.0,
    'type_names': ['H', 'C', 'O'],
    'cutoff_p': 5.0,
    'bessel_factor': 0.3,
    'irreps_edge_sh': '1x0e+1x1o',
    'bessel_weights': normal(1, 4),
    'type_embedding': normal(n_types, width),
    'layers': [
      {
        'linear_1': {
          'irreps_in': '8x0e',
          'irreps_out': '8x0e',
          'instructions': [[0, 0, 0.35]],
          'weights': [normal(width, width)],
        },
        'tp': {
          'irreps_in1': '8x0e',
          'irreps_in2': '1x0e+1x1o',
          'irreps_out': '8x0e+8x1o',
          'instructions': [[0, 0, 0, 1.0], [0, 1, 1, 1.7320508]],
        },
        'linear_2': {
          'irreps_in': '8x0e+8x1o',
          'irreps_out': '8x0e',
          'instructions': [[0, 0, 0.35]],
          'weights': [normal(width, width)],
        },
        'edge_mlp': {
          'alphas': [0.5, 0.25],
          'nonlinearity': 'silu',
          'nonlinearity_scale': 1.6,
          'weights': [normal(4, 16), normal(16, 2 * width)],
        },
        'gate': {
          'irreps_scalars': '8x0e',
          'irreps_gates': '',
          'irreps_gated': '',
          'act_scalars': ['silu'],
          'act_scalar_scales': [1.6],
          'act_gates': [],
          'act_gate_scales': [],
        },
        'self_connection': None,
        'scatter_norm_factor': 0.2,
        'resnet': False,
      }
    ],
    'readout': {
      'alphas': [0.35],
      'nonlinearity': 'none',
      'nonlinearity_scale': 1.0,
      'weights': [normal(width, 1)],
    },
    'scales': onp.abs(normal(n_types, 1)) + 0.5,
    'shifts': normal(n_types, 1),
    'zbl': {
      'atomic_numbers': [1.0, 6.0, 8.0],
      'prefactor': 7.1998224,
      'cutoff_p': 5.0,
    },
  }


def cluster(seed=0, n=7, spread=3.0, r_max=4.0, position=None):
  rng = onp.random.RandomState(seed)
  if position is None:
    position = rng.rand(n, 3) * spread
  position = onp.asarray(position, onp.float32)
  n = position.shape[0]
  species = rng.randint(0, 3, size=n).astype(onp.int32)
  distance = onp.linalg.norm(position[:, None] - position[None], axis=-1)
  senders, receivers = onp.nonzero((distance < r_max) & ~onp.eye(n, dtype=bool))
  return (
    jnp.asarray(position),
    jnp.asarray(species),
    jnp.asarray(senders, jnp.int32),
    jnp.asarray(receivers, jnp.int32),
  )


class AcceleratedTensorProductTest(absltest.TestCase):
  """The accelerated TensorProductScatter subclasses keep ``self.tp``."""

  def test_accelerated_paths(self):
    path = lambda i_in2, i_out, w: SimpleNamespace(
      i_in1=0,
      i_in2=i_in2,
      i_out=i_out,
      path_weight=w,
      connection_mode='uvu',
      has_weight=True,
    )
    tp = SimpleNamespace(
      irreps_in1=e3nn.Irreps('8x0e'),
      irreps_in2=e3nn.Irreps('1x0e+1x1o'),
      irreps_out=e3nn.Irreps('8x0e+8x1o'),
      instructions=(path(0, 0, 1.0), path(1, 1, 1.7320508)),
    )
    accelerated = nequip.tensor_product_of(
      SimpleNamespace(tp=tp, tp_conv=object())
    )
    self.assertIs(accelerated, tp)
    self.assertEqual(
      nequip.uvu_tp_spec(accelerated)['instructions'],
      [[0, 0, 0, 1.0], [0, 1, 1, 1.7320508]],
    )

  def test_wrapper_without_tp_rejected(self):
    with self.assertRaisesRegex(ValueError, 'acceleration modifier'):
      nequip.tensor_product_of(SimpleNamespace(tp_conv=object()))


class NequIPTest(test_util.JAXMDTestCase):
  def test_per_atom_energies_shape(self):
    model = nequip.build(small_description())
    position, species, senders, receivers = cluster()
    energy = model(position, species, senders, receivers)
    self.assertEqual(energy.shape, (position.shape[0],))
    self.assertTrue(onp.all(onp.isfinite(onp.asarray(energy))))

  def test_equivariance(self):
    model = nequip.build(small_description())
    position, species, senders, receivers = cluster()
    total = lambda p: jnp.sum(model(p, species, senders, receivers))
    rotation = e3nn.rand_matrix(random.PRNGKey(0), dtype=position.dtype)
    moved = position @ rotation.T + jnp.array([1.5, -2.0, 0.25], position.dtype)
    # float32 sums reassociate under rotation
    self.assertAllClose(
      model(position, species, senders, receivers),
      model(moved, species, senders, receivers),
      atol=1e-4,
      rtol=1e-4,
    )
    force = -jax.grad(total)(position)
    self.assertAllClose(
      force @ rotation.T, -jax.grad(total)(moved), atol=1e-4, rtol=1e-4
    )

  def test_edge_mask(self):
    model = nequip.build(small_description())
    position, species, senders, receivers = cluster()

    padding = 5
    pad = lambda idx: jnp.pad(idx, (0, padding))
    mask = jnp.arange(senders.shape[0] + padding) < senders.shape[0]
    self.assertAllClose(
      model(position, species, senders, receivers),
      model(position, species, pad(senders), pad(receivers), edge_mask=mask),
    )

  def test_message_direction(self):
    """A single directed edge must leave the sender untouched."""
    model = nequip.build(small_description())
    position = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], jnp.float32)
    species = jnp.array([1, 2], jnp.int32)

    zero, one, empty = (jnp.array(v, jnp.int32) for v in ([0], [1], []))
    one_edge = model(position, species, zero, one)
    isolated = model(position, species, empty, empty)

    # atom 0 sends and receives nothing, atom 1 receives the message
    self.assertAllClose(one_edge[0], isolated[0])
    self.assertGreater(float(jnp.abs(one_edge[1] - isolated[1])), 1e-6)

  def test_jit(self):
    model = nequip.build(small_description())
    position, species, senders, receivers = cluster()
    jitted = jax.jit(lambda p: model(p, species, senders, receivers))
    # fusion reassociates the float32 sums
    self.assertAllClose(
      model(position, species, senders, receivers),
      jitted(position),
      atol=1e-4,
      rtol=1e-4,
    )

  def test_save_load_round_trip(self):
    description = small_description()
    path = os.path.join(absltest.get_default_test_tmpdir(), 'nequip.eqx')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nequip.save_model(path, nequip.build(description), description)

    position, species, senders, receivers = cluster()
    self.assertAllClose(
      nequip.build(description)(position, species, senders, receivers),
      nequip.load_model(path)[0](position, species, senders, receivers),
    )

  def test_array_atomic_numbers_rejected(self):
    description = small_description()
    description['zbl']['atomic_numbers'] = onp.asarray(
      description['zbl']['atomic_numbers']
    )
    path = os.path.join(absltest.get_default_test_tmpdir(), 'array_z.eqx')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The header keeps only shapes of arrays, so the numbers would be lost.
    nequip.save_model(path, nequip.build(description), description)
    with self.assertRaisesRegex(ValueError, 'atomic numbers'):
      nequip.load_model(path)

  @absltest.skipIf(PATH is None, 'JAX_MD_NEQUIP_GNN_MODEL is not set.')
  def test_converted_model_runs(self):
    model = nequip.load_model(PATH)[0]  # ty: ignore[invalid-argument-type]
    n_types = model.type_embedding.shape[0]
    position, _, senders, receivers = cluster(r_max=model.r_max, spread=4.0)
    species = jnp.zeros(position.shape[0], jnp.int32) + min(12, n_types - 1)

    energy = model(position, species, senders, receivers)
    self.assertEqual(energy.shape, (position.shape[0],))
    self.assertTrue(onp.all(onp.isfinite(onp.asarray(energy))))

    force = -jax.grad(lambda p: jnp.sum(model(p, species, senders, receivers)))(
      position
    )
    self.assertTrue(onp.all(onp.isfinite(onp.asarray(force))))

  @absltest.skipIf(PACKAGE is None, 'JAX_MD_NEQUIP_PACKAGE is not set.')
  def test_torch_parity(self):
    import torch

    torch_model = load_package(PACKAGE)
    description = nequip.convert(torch_model)
    model = nequip.build(description)

    # Jittered 2.4 angstrom grid, so bonding rather than ZBL dominates
    rng = onp.random.RandomState(1)
    grid = onp.indices((2, 2, 2)).reshape(3, -1).T[:7]
    position, _, senders, receivers = cluster(
      r_max=model.r_max, position=2.4 * grid + 0.2 * rng.randn(7, 3)
    )
    n_types = len(description['type_names'])
    species = jnp.asarray(rng.randint(0, n_types, size=position.shape[0]))

    dtype = next(torch_model.parameters()).dtype
    out = torch_model(
      {
        'pos': torch.tensor(onp.asarray(position), dtype=dtype),
        # nequip numbers its edges the other way round; see NequIP.
        'edge_index': torch.tensor(
          onp.stack([onp.asarray(receivers), onp.asarray(senders)]),
          dtype=torch.long,
        ),
        'atom_types': torch.tensor(onp.asarray(species), dtype=torch.long),
        'batch': torch.zeros(position.shape[0], dtype=torch.long),
        'num_atoms': torch.tensor([position.shape[0]]),
      }
    )

    energy = model(position, species, senders, receivers)
    force = -jax.grad(lambda p: jnp.sum(model(p, species, senders, receivers)))(
      position
    )
    self.assertAllClose(
      energy,
      out['atomic_energy'].detach().numpy().ravel().astype(onp.float32),
      rtol=1e-5,
      atol=1e-3,
    )
    self.assertAllClose(
      force,
      out['forces'].detach().numpy().astype(onp.float32),
      rtol=1e-4,
      atol=5e-3,
    )

  @absltest.skipIf(PACKAGE is None, 'JAX_MD_NEQUIP_PACKAGE is not set.')
  def test_from_config_structure(self):
    description = nequip.convert(load_package(PACKAGE))
    layers = description['layers']
    first = layers[0]
    # The union over layers undoes upstream's pruning
    blocks = []
    for layer in layers[:-1]:
      for part in ('irreps_scalars', 'irreps_gated'):
        for mul, ir in e3nn.Irreps(layer['gate'][part]):
          if (mul, ir) not in blocks:
            blocks.append((mul, ir))
    hidden = e3nn.Irreps(blocks)  # ty: ignore[invalid-argument-type]
    built = nequip.from_config(
      random.PRNGKey(0),
      type_names=description['type_names'],
      hidden_irreps=str(hidden),
      irreps_edge_sh=description['irreps_edge_sh'],
      num_layers=len(layers),
      num_type_features=description['type_embedding'].shape[1],
      num_bessel=description['bessel_weights'].shape[-1],
      r_max=description['r_max'],
      cutoff_p=description['cutoff_p'],
      radial_mlp_width=first['edge_mlp']['weights'][0].shape[1],
      avg_num_neighbors=1.0 / first['scatter_norm_factor'] ** 2,
      self_connection=any(
        layer['self_connection'] is not None for layer in layers
      ),
      resnet=any(layer['resnet'] for layer in layers),
      zbl=description['zbl'] is not None,
    )
    converted = nequip.build(description)
    assert_same_structure(self, skeleton(converted), skeleton(built))
    if description['zbl'] is not None:
      # The symbol table has to agree with upstream's atomic numbers.
      self.assertAllClose(
        built.zbl_atomic_numbers, converted.zbl_atomic_numbers
      )


SI_CHECKPOINT = os.path.join(
  os.path.dirname(__file__), '..', 'examples', 'models', 'si_equivariant.eqx'
)


class ShippedCheckpointTest(test_util.JAXMDTestCase):
  @absltest.skipIf(
    not os.path.exists(SI_CHECKPOINT),
    'needs examples/models/si_equivariant.eqx',
  )
  def test_si_equivariant(self):
    model, config = nequip.load_model(SI_CHECKPOINT)
    self.assertEqual(model.zbl_atomic_numbers, (14.0,))
    self.assertTrue(config['zbl'])
    box = 5.43 * jnp.eye(3)
    diamond = (
      onp.array(
        [
          [0, 0, 0],
          [2, 2, 0],
          [2, 0, 2],
          [0, 2, 2],
          [1, 1, 1],
          [3, 3, 1],
          [3, 1, 3],
          [1, 3, 3],
        ],
        onp.float32,
      )
      / 4
    )
    diamond[0, 0] += 0.02
    position = jnp.asarray(diamond)
    # r_max exceeds half the cell, so the list needs periodic images
    neighbor_fn, _, energy_fn = energy.nequip_neighbor_list(
      None,
      box,
      model=model,
      species=jnp.zeros(8, jnp.int32),
      fractional_coordinates=True,
      neighbor_list_fn=custom_partition.neighbor_list_multi_image,
      featurizer_fn=custom_partition.graph_featurizer,
    )
    neighbor = neighbor_fn.allocate(position)
    total = lambda R: energy_fn(model, R, neighbor)
    force = -jax.grad(total)(position)
    # reference values of this checkpoint; atom 0 moved along x keeps Fy = Fz = 0
    self.assertAlmostEqual(float(total(position)) / 8, -5.94669, delta=2e-3)
    self.assertAllClose(
      force[0], jnp.array([-1.4052, 0.0, 0.0], force.dtype), atol=2e-3
    )
    self.assertAllClose(
      force[4], jnp.array([0.3635, 0.2829, 0.2829], force.dtype), atol=2e-3
    )


class GateInputOrderTest(test_util.JAXMDTestCase):
  """``e3nn.nn.Gate`` sorts its input, so a converted gate has to undo that."""

  DECLARED: Dict[str, Any] = dict(
    irreps_scalars='8x0e',
    irreps_gates='4x0e',
    irreps_gated='4x1o',
    act_scalars=('silu',),
    act_scalar_scales=(1.0,),
    act_gates=('silu',),
    act_gate_scales=(1.0,),
  )

  def test_input_slices(self):
    x = jnp.asarray(onp.random.RandomState(0).randn(3, 24), jnp.float32)
    # A sort that puts the gated block in front of the two scalar blocks.
    shuffled = jnp.concatenate([x[:, 12:], x[:, :12]], axis=-1)
    reordered = nequip.Gate(
      **self.DECLARED, input_slices=((12, 20), (20, 24), (0, 12))
    )
    self.assertAllClose(nequip.Gate(**self.DECLARED)(x), reordered(shuffled))

  def test_sorted_gate_no_slices(self):
    self.assertEqual(nequip.Gate(**self.DECLARED).input_slices, ())


class ScalarMLPSpecTest(test_util.JAXMDTestCase):
  """``scalar_mlp_spec`` on nequip's ``DeepLinearMLP`` layout."""

  def test_deep_linear_mlp(self):
    try:
      import torch
    except ImportError:
      self.skipTest('needs torch')
    torch.manual_seed(0)
    weights = [torch.randn(4, 8), torch.randn(8, 3)]
    mlp = SimpleNamespace(weights=weights, alpha=torch.tensor(0.25))
    spec = nequip.scalar_mlp_spec(SimpleNamespace(mlp=mlp))
    model = nequip.ScalarMLP(
      weights=tuple(jnp.asarray(w) for w in spec['weights']),
      alphas=tuple(spec['alphas']),
      nonlinearity=spec['nonlinearity'],
      nonlinearity_scale=spec['nonlinearity_scale'],
    )
    x = torch.randn(5, 4)
    expected = torch.mm(x, torch.linalg.multi_dot(weights) * mlp.alpha)
    self.assertAllClose(model(jnp.asarray(x.numpy())), expected.numpy())


if __name__ == '__main__':
  absltest.main()
