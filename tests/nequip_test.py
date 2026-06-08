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

from absl.testing import parameterized

from ml_collections import ConfigDict

import jax
import functools
from absl.testing import absltest
import e3nn_jax as e3nn
from jax import jit
from jax import random
from jax_md import test_util
from jax_md import space
from jax_md import partition
from jax_md.nn import nequip
from jax_md.nn import util as nn_util
from jax_md.nn import gnome

import jax.numpy as jnp
import numpy as onp


def tile(position, box, nodes, tile_count):
  tiles = jnp.arange(tile_count)
  dx, dy, dz = jnp.meshgrid(tiles, tiles, tiles)
  dR = jnp.stack((dx, dy, dz), axis=-1)
  dR = jnp.reshape(dR, (-1, 3))

  position = (position[:, None, :] + dR[None, :, :]) / tile_count
  position = jnp.reshape(position, (-1, 3))
  box *= tile_count

  nodes = jnp.broadcast_to(
    nodes[:, None, :], (nodes.shape[0], len(dR), nodes.shape[1])
  )
  nodes = jnp.reshape(nodes, (-1, nodes.shape[-1]))

  return position, box, nodes


def test_config() -> ConfigDict:
  config = ConfigDict()

  config.multihost = False

  config.seed = 0
  config.split_seed = 0
  config.shuffle_seed = 0

  config.epochs = 1000000
  config.epochs_per_eval = 1
  config.epochs_per_checkpoint = 1

  # keep checkpoint every n checkpoints, -1 only keeps last
  # works only on non-multihost, will error out on multi-host training
  # note that this works together with config.epochs_per_checkpoint, i.e. the
  # checkpointing will only be called on multiples of epochs_per_checkpoint
  config.keep_every_n_checkpoints = -1

  config.learning_rate = 1e-3
  config.schedule = 'constant'
  config.max_lr_plateau_epochs = 200

  config.train_batch_size = 2
  config.test_batch_size = 2

  config.model_family = 'nequip'

  # network
  config.graph_net_steps = 2
  config.nonlinearities = {'e': 'raw_swish', 'o': 'tanh'}
  config.use_sc = True
  config.n_elements = 94
  config.hidden_irreps = '4x0e + 2x1e'
  config.sh_irreps = '1x0e + 1x1e'
  config.num_basis = 8
  config.r_max = 2.5
  config.radial_net_nonlinearity = 'raw_swish'
  config.radial_net_n_hidden = 8
  config.radial_net_n_layers = 2

  # average number of neighbors per atom, used to divide activations are sum
  # in the nequip convolution, helpful for internal normalization.
  config.n_neighbors = 10.0

  # Standard deviation used for the initializer of the weight matrix in the
  # radial scalar MLP
  config.scalar_mlp_std = 4.0

  config.train_dataset = ['harder_silicon']
  config.test_dataset = ['harder_silicon']
  config.validation_dataset = ['harder_silicon']

  config.pretraining_checkpoint = None
  # start from last or a specific pretraining checkpoint
  # options: 'last', or 'ckpt_number', where 'ckpt_number' is string, e.g. '10'
  # for multi-host training, can only be 'last'
  config.pretraining_checkpoint_to_start_from = 'last'

  # The loss is computed as three terms E + lam_F * F + lam_S * S where the
  # first term computes the MSE of the energy, the second computes the MSE of
  # the forces and the last term computes the MSE of the stress. The
  # `force_lambda` and `stress_lambda` parameters determine the relative
  # weighting of the terms.
  config.energy_lambda = 1.0
  config.force_lambda = 1.0
  config.stress_lambda = 0.0
  config.bandgap_lambda = 0.0

  config.energy_loss = ('huber', 0.01)
  config.force_loss = ('huber', 0.01)
  config.stress_loss = ('huber', 10.0)
  config.bandgap_loss = 'L2'

  # 'norm_by_n', 'norm_by_3n', or 'unnormed', applies both to loss and
  # metrics computation
  config.force_loss_normalization = 'norm_by_3n'

  # If L2 regularization is used, then the optimizer is switched to AdamW.
  config.l2_regularization = 0.0

  config.optimizer = 'adam'

  # The epoch size controls the number of crystals in one epoch of data. If the
  # `epoch_size` is set to -1 then it is equal to the size of the dataset. This
  # is useful for large datasets where it is inconvenient to wait for a whole
  # pass through the training data to finish before outputting statistics. It
  # also allows datasets of different sizes to be compared on equal footing.
  config.epoch_size = -1
  config.eval_size = -1

  # By default, we do not restrict the number of atoms for the current system.
  config.max_atoms = -1

  return config


class NequIPTest(test_util.JAXMDTestCase):
  def setup(self):
    c = test_config()

    # dummy system
    atoms = onp.zeros((3, 94))
    atoms[0][5] = 1.0
    atoms[1][5] = 1.0
    atoms[2][5] = 1.0
    atoms = jnp.array(atoms)

    box = jnp.eye(3) * 1
    pos = jnp.array(
      [[0.0, 0.0, 0.0], [0.239487, 0.0, 0.0], [0.5234, 0.78234, 0.0]]
    )

    pos, box, atoms = tile(pos, box, atoms, 5)

    displacement, _ = space.periodic_general(box)

    # build network and initialize
    net = nequip.model_from_config(c)
    neighbor = partition.neighbor_list(
      displacement,
      jnp.diag(box),
      r_cutoff=2.5,
      format=partition.Sparse,
      fractional_coordinates=True,
    )
    params = None

    @jax.jit
    def init_fn(key, atoms, position, nbrs):
      f = nn_util.neighbor_list_featurizer(displacement)
      return net.init(key, f(atoms, pos, nbrs))

    @jax.jit
    def apply_fn(params, atoms, position, nbrs):
      f = nn_util.neighbor_list_featurizer(displacement)
      return net.apply(params, f(atoms, pos, nbrs))[0, 0]

    nbrs = neighbor.allocate(pos)
    params = init_fn(random.PRNGKey(c.seed), atoms, pos, nbrs)
    return apply_fn, (atoms, pos, nbrs), params

  def test_nequip_equivariance(self):
    """Test that the energy output is invariant and forces are equivariant."""
    net, example, params = self.setup()

    atoms, pos, box = example

    e_f_fn = jax.value_and_grad(net, argnums=2)

    energy, force = e_f_fn(params, *example)

    # get random rotation + inversion
    rand_rot_inv = -e3nn.rand_matrix(jax.random.PRNGKey(0), ())
    rotated_pos = (rand_rot_inv @ pos.T).T

    # rotated force output
    force_rotated_output = (rand_rot_inv @ force.T).T

    # eval on rotated input
    energy_rotated_input, force_rotated_input = e_f_fn(
      params, atoms, rotated_pos, box
    )

    # energy should be invariant
    self.assertAllClose(energy_rotated_input, energy)
    self.assertAllClose(force_rotated_input, force_rotated_output)

  def test_nequip_momentum_conservation(self):
    """Test that the sum over all atomic forces is close to 0 in x, y, z."""
    net, example, params = self.setup()

    force = jax.grad(net, argnums=2)(params, *example)

    f_sum = onp.abs(onp.sum(force, axis=0))
    self.assertAllClose(f_sum, onp.zeros_like(f_sum), atol=1e-7)

  def test_nequipcutoff_smoothness(self):
    """Test that the energy changes smoothly as an atom enters the nbh."""
    c = test_config()

    c.r_max = 4.0

    # dummy C-C dimer
    atoms = onp.zeros((2, 94))
    atoms[0][5] = 1.0
    atoms[1][5] = 1.0
    atoms = jnp.array(atoms)

    box = jnp.eye(3) * 10
    pos = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    displacement, _ = space.periodic_general(box)

    # build network and initialize
    net = nequip.model_from_config(c)
    neighbor = partition.neighbor_list(
      displacement,
      jnp.diag(box),
      r_cutoff=5.0,
      format=partition.Sparse,
      fractional_coordinates=True,
    )
    params = None

    @jax.jit
    def init_fn(key, atoms, position, nbrs):
      f = nn_util.neighbor_list_featurizer(displacement)
      return net.init(key, f(atoms, pos, nbrs))

    @jax.jit
    def apply_fn(params, atoms, position, nbrs):
      f = nn_util.neighbor_list_featurizer(displacement)
      return net.apply(params, f(atoms, pos, nbrs))[0, 0]

    nbrs = neighbor.allocate(pos)
    params = init_fn(random.PRNGKey(c.seed), atoms, pos, nbrs)

    energy_force_fn = jax.value_and_grad(apply_fn, argnums=2)

    r0 = 3.999 / 10.0
    r1 = 4.000 / 10.0
    r2 = 4.001 / 10.0

    es = []
    for r in [r0, r1, r2]:
      position = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
      e = energy_force_fn(params, atoms, position, nbrs)
      es.append(e[0])

    es = jnp.array(es)

    self.assertAllClose(es[0] * jnp.ones_like(es), es, atol=1e-7)

  def test_zero_outside_cutoff(self):
    """Test that basis + envelope give 0 if two atoms don't see each other."""
    bessel = nn_util.BesselEmbedding(
      count=8, inner_cutoff=4 - 0.5, outer_cutoff=4
    )

    scalar_dr_edge = jnp.expand_dims(jnp.array(4.1), 0)

    variables = bessel.init(random.PRNGKey(0), scalar_dr_edge)
    embedded_dr_edge = bessel.apply(variables, scalar_dr_edge)
    self.assertAllClose(embedded_dr_edge, jnp.zeros_like(embedded_dr_edge))

  def test_radial_net_zero_to_zero(self):
    """Test that the radial net gives f(0) = 0."""
    fc = nn_util.MLP(
      (64, 64, 1), 'raw_swish', use_bias=False, scalar_mlp_std=4.0
    )
    x = jnp.array([0.0])
    variables = fc.init(random.PRNGKey(0), x)
    fc_out = fc.apply(variables, x)

    self.assertAllClose(fc_out[0], 0.0)


if __name__ == '__main__':
  absltest.main()
