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

"""Describes different physical quantities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import grad
import jax.numpy as np

from jax_md.util import *

def force(energy):
  """Computes the force as the negative gradient of an energy."""
  return grad(lambda R, *args, **kwargs: -energy(R, *args, **kwargs))


def canonicalize_force(energy_or_force, quantity):
  if quantity is Force:
    return energy_or_force
  elif quantity is Energy:
    return force(energy_or_force)

  raise ValueError(
      'Expected quantity to be Energy or Force, but found {}'.format(quantity))


class Force(object):
  """Dummy object to denote whether a quantity is a force."""
  pass
Force = Force()


class Energy(object):
  """Dummy object to denote whether a quantity is an energy."""
  pass
Energy = Energy()


class Dynamic(object):
  """Object used to denote dynamic shapes and species."""
  pass
Dynamic = Dynamic()


def kinetic_energy(V, mass=1.0):
  """Computes the kinetic energy of a system with some velocities."""
  return 0.5 * np.sum(mass * V ** 2)


def temperature(V, mass=1.0):
  """Computes the temperature of a system with some velocities."""
  N, dim = V.shape
  return np.sum(mass * V ** 2) / (N * dim)


def canonicalize_mass(mass):
  if isinstance(mass, float):
    return mass
  elif isinstance(mass, np.ndarray):
    if len(mass.shape) == 2 and mass.shape[1] == 1:
      return mass
    elif len(mass.shape) == 1:
      return np.reshape(mass, (mass.shape[0], 1))
    elif len(mass.shape) == 0:
      return mass
  elif (isinstance(mass, f16) or
        isinstance(mass, f32) or
        isinstance(mass, f64)):
    return mass
  msg = (
      'Expected mass to be either a floating point number or a one-dimensional'
      'ndarray. Found {}.'.format(mass)
      )
  raise ValueError(msg)
