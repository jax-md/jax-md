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

"""Definitions of various standard energy functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps, partial

import jax
import jax.numpy as np
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk

from jax_md import space, smap, partition, nn
from jax_md.interpolate import spline
from jax_md.util import f32, f64
from jax_md.util import check_kwargs_time_dependence


def simple_spring(dr, length=1, epsilon=1, alpha=2, **unused_kwargs):
  """Isotropic spring potential with a given rest length.

  We define `simple_spring` to be a generalized Hookian spring with
  agreement when alpha = 2.
  """
  check_kwargs_time_dependence(unused_kwargs)
  return epsilon / alpha * (dr - length) ** alpha


def simple_spring_bond(
    displacement_or_metric, bond, bond_type=None, length=1, epsilon=1, alpha=2):
  """Convenience wrapper to compute energy of particles bonded by springs."""
  length = np.array(length, f32)
  epsilon = np.array(epsilon, f32)
  alpha = np.array(alpha, f32)
  return smap.bond(
    simple_spring,
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    bond,
    bond_type,
    length=length,
    epsilon=epsilon,
    alpha=alpha)


def soft_sphere(dr, sigma=1, epsilon=1, alpha=2, **unused_kwargs):
  """Finite ranged repulsive interaction between soft spheres.

  Args:
    dr: An ndarray of shape [n, m] of pairwise distances between particles.
    sigma: Particle diameter. Should either be a floating point scalar or an
      ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
    alpha: Exponent specifying interaction stiffness. Should either be a float
      point scalar or an ndarray whose shape is [n, m].
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
  Returns:
    Matrix of energies whose shape is [n, m].
  """
  check_kwargs_time_dependence(unused_kwargs)
  dr = dr / sigma
  U = epsilon * np.where(
    dr < 1.0, f32(1.0) / alpha * (f32(1.0) - dr) ** alpha, f32(0.0))
  return U


def soft_sphere_pair(
    displacement_or_metric, species=None, sigma=1.0, epsilon=1.0, alpha=2.0):
  """Convenience wrapper to compute soft sphere energy over a system."""
  sigma = np.array(sigma, dtype=f32)
  epsilon = np.array(epsilon, dtype=f32)
  alpha = np.array(alpha, dtype=f32)
  return smap.pair(
      soft_sphere,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      alpha=alpha)


def soft_sphere_neighbor_list(
    displacement_or_metric,
    box_size,
    species=None,
    sigma=1.0,
    epsilon=1.0,
    alpha=2.0,
    dr_threshold=0.2):
  """Convenience wrapper to compute soft spheres using a neighbor list."""
  sigma = np.array(sigma, dtype=f32)
  epsilon = np.array(epsilon, dtype=f32)
  alpha = np.array(alpha, dtype=f32)
  list_cutoff = f32(np.max(sigma))
  dr_threshold = f32(list_cutoff * dr_threshold)

  neighbor_fn = partition.neighbor_list(
    displacement_or_metric, box_size, list_cutoff, dr_threshold)
  energy_fn = smap.pair_neighbor_list(
    soft_sphere,
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha)

  return neighbor_fn, energy_fn


def lennard_jones(dr, sigma=1, epsilon=1, **unused_kwargs):
  """Lennard-Jones interaction between particles with a minimum at sigma.

  Args:
    dr: An ndarray of shape [n, m] of pairwise distances between particles.
    sigma: Distance between particles where the energy has a minimum. Should
      either be a floating point scalar or an ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
  Returns:
    Matrix of energies of shape [n, m].
  """
  check_kwargs_time_dependence(unused_kwargs)
  idr = (sigma / dr)
  idr = idr * idr
  idr6 = idr * idr * idr
  idr12 = idr6 * idr6
  # TODO(schsam): This seems potentially dangerous. We should do ErrorChecking
  # here.
  return np.nan_to_num(f32(4) * epsilon * (idr12 - idr6))


def lennard_jones_pair(
    displacement_or_metric,
    species=None, sigma=1.0, epsilon=1.0, r_onset=2.0, r_cutoff=2.5):
  """Convenience wrapper to compute Lennard-Jones energy over a system."""
  sigma = np.array(sigma, dtype=f32)
  epsilon = np.array(epsilon, dtype=f32)
  r_onset = r_onset * np.max(sigma)
  r_cutoff = r_cutoff * np.max(sigma)
  return smap.pair(
    multiplicative_isotropic_cutoff(lennard_jones, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    sigma=sigma,
    epsilon=epsilon)


def lennard_jones_neighbor_list(
    displacement_or_metric,
    box_size,
    species=None,
    sigma=1.0,
    epsilon=1.0,
    alpha=2.0,
    r_onset=2.0,
    r_cutoff=2.5,
    dr_threshold=0.5): # TODO(schsam) Optimize this.
  """Convenience wrapper to compute lennard-jones using a neighbor list."""
  sigma = np.array(sigma, f32)
  epsilon = np.array(epsilon, f32)
  r_onset = np.array(r_onset * np.max(sigma), f32)
  r_cutoff = np.array(r_cutoff * np.max(sigma), f32)
  dr_threshold = np.array(np.max(sigma) * dr_threshold, f32)

  neighbor_fn = partition.neighbor_list(
    displacement_or_metric, box_size, r_cutoff, dr_threshold)
  energy_fn = smap.pair_neighbor_list(
    multiplicative_isotropic_cutoff(lennard_jones, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    sigma=sigma,
    epsilon=epsilon)

  return neighbor_fn, energy_fn


def morse(dr, sigma=1.0, epsilon=5.0, alpha=5.0, **unused_kwargs):
  """Morse interaction between particles with a minimum at r0.
  Args:
    dr: An ndarray of shape [n, m] of pairwise distances between particles.
    sigma: Distance between particles where the energy has a minimum. Should
      either be a floating point scalar or an ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
    alpha: Range parameter. Should either be a floating point scalar or an 
      ndarray whose shape is [n, m].
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
  Returns:
    Matrix of energies of shape [n, m].
  """
  check_kwargs_time_dependence(unused_kwargs)
  U = epsilon * (f32(1) - np.exp(-alpha * (dr - sigma)))**f32(2) - epsilon
  # TODO(cpgoodri): ErrorChecking following lennard_jones
  return np.nan_to_num(np.array(U, dtype=dr.dtype))
  
def morse_pair(
    displacement_or_metric,
    species=None, sigma=1.0, epsilon=5.0, alpha=5.0, r_onset=2.0, r_cutoff=2.5):
  """Convenience wrapper to compute Morse energy over a system."""
  sigma = np.array(sigma, dtype=f32)
  epsilon = np.array(epsilon, dtype=f32)
  alpha = np.array(alpha, dtype=f32)
  return smap.pair(
    multiplicative_isotropic_cutoff(morse, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha)
  
def morse_neighbor_list(
    displacement_or_metric,
    box_size,
    species=None,
    sigma=1.0,
    epsilon=5.0,
    alpha=5.0,
    r_onset=2.0,
    r_cutoff=2.5,
    dr_threshold=0.5): # TODO(cpgoodri) Optimize this.
  """Convenience wrapper to compute Morse using a neighbor list."""
  sigma = np.array(sigma, f32)
  epsilon = np.array(epsilon, f32)
  alpha = np.array(alpha, f32)
  r_onset = np.array(r_onset, f32)
  r_cutoff = np.array(r_cutoff, f32)
  dr_threshold = np.array(dr_threshold, f32)

  neighbor_fn = partition.neighbor_list(
    displacement_or_metric, box_size, r_cutoff, dr_threshold)
  energy_fn = smap.pair_neighbor_list(
    multiplicative_isotropic_cutoff(morse, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha)

  return neighbor_fn, energy_fn


def multiplicative_isotropic_cutoff(fn, r_onset, r_cutoff):
  """Takes an isotropic function and constructs a truncated function.

  Given a function f:R -> R, we construct a new function f':R -> R such that
  f'(r) = f(r) for r < r_onset, f'(r) = 0 for r > r_cutoff, and f(r) is C^1
  everywhere. To do this, we follow the approach outlined in HOOMD Blue [1]
  (thanks to Carl Goodrich for the pointer). We construct a function S(r) such
  that S(r) = 1 for r < r_onset, S(r) = 0 for r > r_cutoff, and S(r) is C^1.
  Then f'(r) = S(r)f(r).

  Args:
    fn: A function that takes an ndarray of distances of shape [n, m] as well
      as varargs.
    r_onset: A float specifying the distance marking the onset of deformation.
    r_cutoff: A float specifying the cutoff distance.

  Returns:
    A new function with the same signature as fn, with the properties outlined
    above.

  [1] HOOMD Blue documentation. Accessed on 05/31/2019.
      https://hoomd-blue.readthedocs.io/en/stable/module-md-pair.html#hoomd.md.pair.pair
  """

  r_c = r_cutoff ** f32(2)
  r_o = r_onset ** f32(2)

  def smooth_fn(dr):
    r = dr ** f32(2)

    return np.where(
      dr < r_onset,
      f32(1),
      np.where(
        dr < r_cutoff,
        (r_c - r) ** f32(2) * (r_c + f32(2) * r - f32(3) * r_o) / (
          r_c - r_o) ** f32(3),
        f32(0)
      )
    )

  @wraps(fn)
  def cutoff_fn(dr, *args, **kwargs):
    return smooth_fn(dr) * fn(dr, *args, **kwargs)

  return cutoff_fn


def load_lammps_eam_parameters(f):
  """Reads EAM parameters from a LAMMPS file and returns relevant spline fits.

  This function reads single-element EAM potential fit parameters from a file
  in DYNAMO funcl format. In summary, the file contains:
  Line 1-3: comments,
  Line 4: Number of elements and the element type,
  Line 5: The number of charge values that the embedding energy is evaluated
  on (num_drho), interval between the charge values (drho), the number of
  distances the pairwise energy and the charge density is evaluated on (num_dr),
  the interval between these distances (dr), and the cutoff distance (cutoff).
  The lines that come after are the embedding function evaluated on num_drho
  charge values, charge function evaluated at num_dr distance values, and
  pairwise energy evaluated at num_dr distance values. Note that the pairwise
  energy is multiplied by distance (in units of eV x Angstroms). For more
  details of the DYNAMO file format, see:
  https://sites.google.com/a/ncsu.edu/cjobrien/tutorials-and-guides/eam
  Args:
    f: File handle for the EAM parameters text file.

  Returns:
    charge_fn: A function that takes an ndarray of shape [n, m] of distances
      between particles and returns a matrix of charge contributions.
    embedding_fn: Function that takes an ndarray of shape [n] of charges and
      returns an ndarray of shape [n] of the energy cost of embedding an atom
      into the charge.
    pairwise_fn: A function that takes an ndarray of shape [n, m] of distances
      and returns an ndarray of shape [n, m] of pairwise energies.
    cutoff: Cutoff distance for the embedding_fn and pairwise_fn.
  """
  raw_text = f.read().split('\n')
  if 'setfl' not in raw_text[0]:
    raise ValueError('File format is incorrect, expected LAMMPS setfl format.')
  temp_params = raw_text[4].split()
  num_drho, num_dr = int(temp_params[0]), int(temp_params[2])
  drho, dr, cutoff = float(temp_params[1]), float(temp_params[3]), float(
      temp_params[4])
  data = np.array(map(float, raw_text[6:-1]))
  embedding_fn = spline(data[:num_drho], drho)
  charge_fn = spline(data[num_drho:num_drho + num_dr], dr)
  # LAMMPS EAM parameters file lists pairwise energies after multiplying by
  # distance, in units of eV*Angstrom. We divide the energy by distance below,
  distances = np.arange(num_dr) * dr
  # Prevent dividing by zero at zero distance, which will not
  # affect the calculation
  distances = np.where(distances == 0, f32(0.001), distances)
  pairwise_fn = spline(
      data[num_dr + num_drho:num_drho + f32(2) * num_dr] / distances,
      dr)
  return charge_fn, embedding_fn, pairwise_fn, cutoff


def eam(displacement, charge_fn, embedding_fn, pairwise_fn, axis=None):
  """Interatomic potential as approximated by embedded atom model (EAM).

  This code implements the EAM approximation to interactions between metallic
  atoms. In EAM, the potential energy of an atom is given by two terms: a
  pairwise energy and an embedding energy due to the interaction between the
  atom and background charge density. The EAM potential for a single atomic
  species is often
  determined by three functions:
    1) Charge density contribution of an atom as a function of distance.
    2) Energy of embedding an atom in the background charge density.
    3) Pairwise energy.
  These three functions are usually provided as spline fits, and we follow the
  implementation and spline fits given by [1]. Note that in current
  implementation, the three functions listed above can also be expressed by a
  any function with the correct signature, including neural networks.

  Args:
    displacement: A function that produces an ndarray of shape [n, m,
      spatial_dimension] of particle displacements from particle positions
      specified as an ndarray of shape [n, spatial_dimension] and [m,
      spatial_dimension] respectively.
    charge_fn: A function that takes an ndarray of shape [n, m] of distances
      between particles and returns a matrix of charge contributions.
    embedding_fn: Function that takes an ndarray of shape [n] of charges and
      returns an ndarray of shape [n] of the energy cost of embedding an atom
      into the charge.
    pairwise_fn: A function that takes an ndarray of shape [n, m] of distances
      and returns an ndarray of shape [n, m] of pairwise energies.
    axis: Specifies which axis the total energy should be summed over.

  Returns:
    A function that computes the EAM energy of a set of atoms with positions
    given by an [n, spatial_dimension] ndarray.

  [1] Y. Mishin, D. Farkas, M.J. Mehl, DA Papaconstantopoulos, "Interatomic
  potentials for monoatomic metals from experimental data and ab initio
  calculations." Physical Review B, 59 (1999)
  """
  metric = space.map_product(space.metric(displacement))

  def energy(R, **kwargs):
    dr = metric(R, R, **kwargs)
    total_charge = smap._high_precision_sum(charge_fn(dr), axis=1)
    embedding_energy = embedding_fn(total_charge)
    pairwise_energy = smap._high_precision_sum(smap._diagonal_mask(
        pairwise_fn(dr)), axis=1) / f32(2.0)
    return smap._high_precision_sum(
        embedding_energy + pairwise_energy, axis=axis)

  return energy


def eam_from_lammps_parameters(displacement, f):
  """Convenience wrapper to compute EAM energy over a system."""
  return eam(displacement, *load_lammps_eam_parameters(f)[:-1])


def behler_parrinello(displacement, 
                      species=None,
                      mlp_sizes=(30, 30), 
                      mlp_kwargs=None, 
                      sym_kwargs=None):
  if sym_kwargs is None:
    sym_kwargs = {}
  if mlp_kwargs is None:
    mlp_kwargs = {
        'activation': np.tanh
    }

  sym_fn = nn.behler_parrinello_symmetry_functions(displacement, 
                                                   species, 
                                                   **sym_kwargs)

  @hk.transform
  def model(R, **kwargs):
    embedding_fn = hk.nets.MLP(output_sizes=mlp_sizes+(1,),
                               activate_final=False,
                               name='BPEncoder',
                               **mlp_kwargs)
    embedding_fn = vmap(embedding_fn)
    sym = sym_fn(R, **kwargs)
    readout = embedding_fn(sym)
    return np.sum(readout)
  return model.init, model.apply




class EnergyGraphNet(hk.Module):
  """Implements a Graph Neural Network for energy fitting.

  This model uses a GraphNetEmbedding combined with a decoder applied to the
  global state.
  """
  def __init__(self, n_recurrences, mlp_sizes, mlp_kwargs=None, name='Energy'):
    super(EnergyGraphNet, self).__init__(name=name)

    if mlp_kwargs is None:
      mlp_kwargs = {
        'w_init': hk.initializers.VarianceScaling(),
        'b_init': hk.initializers.VarianceScaling(0.1),
        'activation': jax.nn.softplus
      }

    self._graph_net = nn.GraphNetEncoder(n_recurrences, mlp_sizes, mlp_kwargs)
    self._decoder = hk.nets.MLP(output_sizes=mlp_sizes + (1,),
                                activate_final=False,
                                name='GlobalDecoder',
                                **mlp_kwargs)

  def __call__(self, graph: nn.GraphTuple) -> np.ndarray:
    output = self._graph_net(graph)
    return np.squeeze(self._decoder(output.globals), axis=-1)


def _canonicalize_node_state(nodes):
  if nodes is None:
    return nodes

  if nodes.ndim == 1:
    nodes = nodes[:, np.newaxis]

  if nodes.ndim != 2:
    raise ValueError(
      'Nodes must be a [N, node_dim] array. Found {}.'.format(nodes.shape))

  return nodes


def graph_network(displacement_fn,
                  r_cutoff,
                  nodes=None,
                  n_recurrences=2,
                  mlp_sizes=(64, 64),
                  mlp_kwargs=None):
  """Convenience wrapper around EnergyGraphNet model.

  Args:
    displacement_fn: Function to compute displacement between two positions.
    r_cutoff: A floating point cutoff; Edges will be added to the graph
      for pairs of particles whose separation is smaller than the cutoff.
    nodes: None or an ndarray of shape `[N, node_dim]` specifying the state
      of the nodes. If None this is set to the zeroes vector. Often, for a
      system with multiple species, this could be the species id.
    n_recurrences: The number of steps of message passing in the graph network.
    mlp_sizes: A tuple specifying the layer-widths for the fully-connected
      networks used to update the states in the graph network.
    mlp_kwargs: A dict specifying args for the fully-connected networks used to
      update the states in the graph network.

  Returns:
    A tuple of functions. An `params = init_fn(key, R)` that instantiates the
    model parameters and an `E = apply_fn(params, R)` that computes the energy
    for a particular state.
  """

  nodes = _canonicalize_node_state(nodes)

  @hk.without_apply_rng
  @partial(hk.transform, apply_rng=True)
  def model(R, **kwargs):
    N = R.shape[0]

    d = partial(displacement_fn, **kwargs)
    d = space.map_product(d)
    dR = d(R, R)

    dr_2 = space.square_distance(dR)

    if 'nodes' in kwargs:
      _nodes = _canonicalize_node_state(kwargs['nodes'])
    else:
      _nodes = np.zeros((N, 1), R.dtype) if nodes is None else nodes

    edge_idx = np.broadcast_to(np.arange(N)[np.newaxis, :], (N, N))
    edge_idx = np.where(dr_2 < r_cutoff ** 2, edge_idx, N)

    _globals = np.zeros((1,), R.dtype) 

    net = EnergyGraphNet(n_recurrences, mlp_sizes, mlp_kwargs)
    return net(nn.GraphTuple(_nodes, dR, _globals, edge_idx))

  return model.init, model.apply 


def graph_network_neighbor_list(displacement_fn,
                                box_size,
                                r_cutoff,
                                dr_threshold,
                                nodes=None,
                                n_recurrences=2,
                                mlp_sizes=(64, 64),
                                mlp_kwargs=None):
  """Convenience wrapper around EnergyGraphNet model using neighbor lists.

  Args:
    displacement_fn: Function to compute displacement between two positions.
    box_size: The size of the simulation volume, used to construct neighbor
      list.
    r_cutoff: A floating point cutoff; Edges will be added to the graph
      for pairs of particles whose separation is smaller than the cutoff.
    dr_threshold: A floating point number specifying a "halo" radius that we use
      for neighbor list construction. See `neighbor_list` for details.
    nodes: None or an ndarray of shape `[N, node_dim]` specifying the state
      of the nodes. If None this is set to the zeroes vector. Often, for a
      system with multiple species, this could be the species id.
    n_recurrences: The number of steps of message passing in the graph network.
    mlp_sizes: A tuple specifying the layer-widths for the fully-connected
      networks used to update the states in the graph network.
    mlp_kwargs: A dict specifying args for the fully-connected networks used to
      update the states in the graph network.

  Returns:
    A pair of functions. An `params = init_fn(key, R)` that instantiates the
    model parameters and an `E = apply_fn(params, R)` that computes the energy
    for a particular state.
  """

  nodes = _canonicalize_node_state(nodes)

  @hk.without_apply_rng
  @partial(hk.transform, apply_rng=True)
  def model(R, neighbor, **kwargs):
    N = R.shape[0]

    d = partial(displacement_fn, **kwargs)
    d = space.map_neighbor(d)
    R_neigh = R[neighbor.idx]
    dR = d(R, R_neigh)

    if 'nodes' in kwargs:
      _nodes = _canonicalize_node_state(kwargs['nodes'])
    else:
      _nodes = np.zeros((N, 1), R.dtype) if nodes is None else nodes

    _globals = np.zeros((1,), R.dtype) 

    dr_2 = space.square_distance(dR)
    edge_idx = np.where(dr_2 < r_cutoff ** 2, neighbor.idx, N)

    net = EnergyGraphNet(n_recurrences, mlp_sizes, mlp_kwargs)
    return net(nn.GraphTuple(_nodes, dR, _globals, edge_idx))

  neighbor_fn = partition.neighbor_list(displacement_fn,
                                        box_size,
                                        r_cutoff,
                                        dr_threshold,
                                        mask_self=False)
  init_fn, apply_fn = model.init, model.apply

  return neighbor_fn, init_fn, apply_fn
