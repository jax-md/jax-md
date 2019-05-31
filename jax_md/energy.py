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

import jax.numpy as np

from jax_md import space, smap
from jax_md.interpolate import spline


def soft_sphere(dR, sigma=1.0, epsilon=1.0, alpha=2.0):
  """Finite ranged repulsive interaction between soft spheres.

  Args:
    dR: An ndarray of shape [n, m, spatial_dimension] of displacement vectors
      between particles.
    sigma: Particle radii. Should either be a floating point scalar or an
      ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
    alpha: Exponent specifying interaction stiffness. Should either be a float
      point scalar or an ndarray whose shape is [n, m].

  Returns:
    Matrix of energies whose shape is [n, m].
  """
  dr = space.distance(dR)
  dr = dr / sigma
  U = epsilon * np.where(dr < 1.0, 1.0 / alpha * (1.0 - dr) ** alpha, 0.0)
  # NOTE(schsam): This seems a little bit janky. However, it seems possibly
  # necessary because constants seemed to be upcast to float64.
  return np.array(U, dtype=dr.dtype)


def soft_sphere_pairwise(
    metric, species=None, sigma=1.0, epsilon=1.0, alpha=2.0):
  """Convenience wrapper to compute soft sphere energy over a system."""
  return smap.pairwise(
      soft_sphere,
      metric,
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      alpha=alpha)


def lennard_jones(dR, sigma, epsilon):
  """Lennard-Jones interaction between particles with a minimum at sigma.

  Args:
    dR: An ndarray of shape [n, m, spatial_dimension] of displacement vectors
      between particles.
    sigma: Distance between particles where the energy has a minimum. Should
      either be a floating point scalar or an ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
  Returns:
    Matrix of energies of shape [n, m].
  """
  dr = space.square_distance(dR)
  dr = sigma ** 2 / dr
  idr6 = dr ** 3.0
  idr12 = idr6 ** 2.0
  return epsilon * (idr12 - 2 * idr6)


def lennard_jones_pairwise(
    metric, species=None, sigma=1.0, epsilon=1.0):
  """Convenience wrapper to compute Lennard-Jones energy over a system."""
  return smap.pairwise(
      lennard_jones, metric, species=species, sigma=sigma, epsilon=epsilon)


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
  distances = np.where(distances == 0, 0.001, distances)
  pairwise_fn = spline(
      data[num_dr + num_drho:num_drho + 2 * num_dr] / distances,
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
  def energy(R, **kwargs):
    dr = space.distance(displacement(R, R, **kwargs))
    total_charge = smap._high_precision_sum(charge_fn(dr), axis=1)
    embedding_energy = embedding_fn(total_charge)
    pairwise_energy = smap._high_precision_sum(smap._diagonal_mask(
        pairwise_fn(dr)), axis=1) / 2.0
    return smap._high_precision_sum(
        embedding_energy + pairwise_energy, axis=axis)

  return energy


def eam_from_lammps_parameters(displacement, f):
  """Convenience wrapper to compute EAM energy over a system."""
  return eam(displacement, *load_lammps_eam_parameters(f)[:-1])
