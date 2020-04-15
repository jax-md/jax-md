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

"""Code to transform functions on individual tuples of particles to sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce, partial
from collections import namedtuple
import math
from operator import mul

import numpy as onp

from jax import lax, ops, vmap, eval_shape
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
import jax.numpy as np

from jax_md import quantity, space
from jax_md.util import *


# Mapping potential functional forms to bonds.


# pylint: disable=invalid-name
def _get_bond_type_parameters(params, bond_type):
  """Get parameters for interactions for bonds indexed by a bond-type."""
  # TODO(schsam): We should do better error checking here.
  assert isinstance(bond_type, np.ndarray)
  assert len(bond_type.shape) == 1

  if isinstance(params, np.ndarray):
    if len(params.shape) == 1:
      return params[bond_type]
    elif len(params.shape) == 0:
      return params
    else:
      raise ValueError(
          'Params must be a scalar or a 1d array if using a bond-type lookup.')
  elif(isinstance(params, int) or isinstance(params, float) or
       np.issubdtype(params, np.integer) or np.issubdtype(params, np.floating)):
    return params
  raise NotImplementedError


def _kwargs_to_bond_parameters(bond_type, kwargs):
  """Extract parameters from keyword arguments."""
  # NOTE(schsam): We could pull out the species case from the generic case.
  for k, v in kwargs.items():
    if bond_type is not None:
      kwargs[k] = _get_bond_type_parameters(v, bond_type)
  return kwargs


def bond(fn, metric, static_bonds=None, static_bond_types=None, **kwargs):
  """Promotes a function that acts on a single pair to one on a set of bonds.

  TODO(schsam): It seems like bonds might potentially have poor memory access.
    Should think about this a bit and potentially optimize.

  Args:
    fn: A function that takes an ndarray of pairwise distances or displacements
      of shape [n, m] or [n, m, d_in] respectively as well as kwargs specifying
      parameters for the function. fn returns an ndarray of evaluations of shape
      [n, m, d_out].
    metric: A function that takes two ndarray of positions of shape
      [spatial_dimension] and [spatial_dimension] respectively and returns
      an ndarray of distances or displacements of shape [] or [d_in]
      respectively. The metric can optionally take a floating point time as a
      third argument.
    static_bonds: An ndarray of integer pairs wth shape [b, 2] where each pair
      specifies a bond. static_bonds are baked into the returned compute
      function statically and cannot be changed after the fact.
    static_bond_types: An ndarray of integers of shape [b] specifying the type
      of each bond. Only specify bond types if you want to specify bond
      parameters by type. One can also specify constant or per-bond parameters
      (see below).
    kwargs: Arguments providing parameters to the mapped function. In cases
      where no bond type information is provided these should be either 1) a
      scalar or 2) an ndarray of shape [b]. If bond type information is
      provided then the parameters should be specified as either 1) a scalar or
      2) an ndarray of shape [max_bond_type].

  Returns:
    A function fn_mapped. Note that fn_mapped can take arguments bonds and
    bond_types which will be bonds that are specified dynamically. This will
    incur a recompilation when the number of bonds changes. Improving this
    state of affairs I will leave as a TODO until someone actually uses this
    feature and runs into speed issues.
  """

  # Each call to vmap adds a single batch dimension. Here, we would like to
  # promote the metric function from one that computes the distance /
  # displacement between two vectors to one that acts on two lists of vectors.
  # Thus, we apply a single application of vmap.
  # TODO: Uncomment this once JAX supports vmap over kwargs.
  # metric = vmap(metric, (0, 0), 0)

  def compute_fn(R, bonds, bond_types, static_kwargs, dynamic_kwargs):
    Ra = R[bonds[:, 0]]
    Rb = R[bonds[:, 1]]
    _kwargs = merge_dicts(static_kwargs, dynamic_kwargs)
    _kwargs = _kwargs_to_bond_parameters(bond_types, _kwargs)
    # NOTE(schsam): This pattern is needed due to JAX issue #912. 
    _metric = vmap(partial(metric, **dynamic_kwargs), 0, 0)
    dr = _metric(Ra, Rb)
    return _high_precision_sum(fn(dr, **_kwargs))

  def mapped_fn(R, bonds=None, bond_types=None, **dynamic_kwargs):
    accum = f32(0)

    if bonds is not None:
      accum = accum + compute_fn(R, bonds, bond_types, kwargs, dynamic_kwargs)

    if static_bonds is not None:
      accum = accum + compute_fn(
          R, static_bonds, static_bond_types, kwargs, dynamic_kwargs)

    return accum
  return mapped_fn


# Mapping potential functional forms to pairwise interactions.


def _get_species_parameters(params, species):
  """Get parameters for interactions between species pairs."""
  # TODO(schsam): We should do better error checking here.
  if isinstance(params, np.ndarray):
    if len(params.shape) == 2:
      return params[species]
    elif len(params.shape) == 0:
      return params
    else:
      raise ValueError(
          'Params must be a scalar or a 2d array if using a species lookup.')
  return params


def _get_matrix_parameters(params):
  """Get an NxN parameter matrix from per-particle parameters."""
  if isinstance(params, np.ndarray):
    if len(params.shape) == 1:
      # NOTE(schsam): get_parameter_matrix only supports additive parameters.
      return 0.5 * (params[:, np.newaxis] + params[np.newaxis, :])
    elif len(params.shape) == 0 or len(params.shape) == 2:
      return params
    else:
      raise NotImplementedError
  elif(isinstance(params, int) or isinstance(params, float) or
       np.issubdtype(params, np.integer) or np.issubdtype(params, np.floating)):
    return params
  else:
    raise NotImplementedError


def _kwargs_to_parameters(species=None, **kwargs):
  """Extract parameters from keyword arguments."""
  # NOTE(schsam): We could pull out the species case from the generic case.
  s_kwargs = kwargs
  for k, v in s_kwargs.items():
    if species is None:
      s_kwargs[k] = _get_matrix_parameters(v)
    else:
      s_kwargs[k] = _get_species_parameters(v, species)
  return s_kwargs


def _diagonal_mask(X):
  """Sets the diagonal of a matrix to zero."""
  if X.shape[0] != X.shape[1]:
    raise ValueError(
        'Diagonal mask can only mask square matrices. Found {}x{}.'.format(
            X.shape[0], X.shape[1]))
  if len(X.shape) > 3:
    raise ValueError(
        ('Diagonal mask can only mask rank-2 or rank-3 tensors. '
         'Found {}.'.format(len(X.shape))))
  N = X.shape[0]
  # NOTE(schsam): It seems potentially dangerous to set nans to 0 here. However,
  # masking nans also doesn't seem to work. So it also seems necessary. At the
  # very least we should do some @ErrorChecking.
  X = np.nan_to_num(X)
  mask = f32(1.0) - np.eye(N, dtype=X.dtype)
  if len(X.shape) == 3:
    mask = np.reshape(mask, (N, N, 1))
  return mask * X


def _high_precision_sum(X, axis=None, keepdims=False):
  """Sums over axes at 64-bit precision then casts back to original dtype."""
  return np.array(
      np.sum(X, axis=axis, dtype=f64, keepdims=keepdims), dtype=X.dtype)


def _check_species_dtype(species):
  if species.dtype == i32 or species.dtype == i64:
    return
  msg = 'Species has wrong dtype. Expected integer but found {}.'.format(
      species.dtype)
  raise ValueError(msg)


def pair(
    fn, metric, species=None, reduce_axis=None, keepdims=False, **kwargs):
  """Promotes a function that acts on a pair of particles to one on a system.

  Args:
    fn: A function that takes an ndarray of pairwise distances or displacements
      of shape [n, m] or [n, m, d_in] respectively as well as kwargs specifying
      parameters for the function. fn returns an ndarray of evaluations of shape
      [n, m, d_out].
    metric: A function that takes two ndarray of positions of shape
      [spatial_dimension] and [spatial_dimension] respectively and returns
      an ndarray of distances or displacements of shape [] or [d_in]
      respectively. The metric can optionally take a floating point time as a
      third argument.
    species: A list of species for the different particles. This should either
      be None (in which case it is assumed that all the particles have the same
      species), an integer ndarray of shape [n] with species data, or Dynamic
      in which case the species data will be specified dynamically. Note: that
      dynamic species specification is less efficient, because we cannot
      specialize shape information.
    reduce_axis: A list of axes to reduce over. This is supplied to np.sum and
      so the same convention is used.
    keepdims: A boolean specifying whether the empty dimensions should be kept
      upon reduction. This is supplied to np.sum and so the same convention is
      used.
    kwargs: Arguments providing parameters to the mapped function. In cases
      where no species information is provided these should be either 1) a
      scalar, 2) an ndarray of shape [n], 3) an ndarray of shape [n, n]. If
      species information is provided then the parameters should be specified as
      either 1) a scalar or 2) an ndarray of shape [max_species, max_species].

  Returns:
    A function fn_mapped.

    If species is None or statically specified then fn_mapped takes as arguments
    an ndarray of positions of shape [n, spatial_dimension].

    If species is Dynamic then fn_mapped takes as input an ndarray of shape
    [n, spatial_dimension], an integer ndarray of species of shape [n], and an
    integer specifying the maximum species.

    The mapped function can also optionally take keyword arguments that get
    threaded through the metric.
  """

  # Each application of vmap adds a single batch dimension. For computations
  # over all pairs of particles, we would like to promote the metric function
  # from one that computes the displacement / distance between two vectors to
  # one that acts over the cartesian product of two sets of vectors. This is
  # equivalent to two applications of vmap adding one batch dimension for the
  # first set and then one for the second.
  # TODO: Uncomment this once vmap supports kwargs.
  #metric = vmap(vmap(metric, (0, None), 0), (None, 0), 0)

  if species is None:
    def fn_mapped(R, **dynamic_kwargs):
      _metric = space.map_product(partial(metric, **dynamic_kwargs))
      _kwargs = merge_dicts(kwargs, dynamic_kwargs)
      _kwargs = _kwargs_to_parameters(species, **_kwargs)
      dr = _metric(R, R)
      # NOTE(schsam): Currently we place a diagonal mask no matter what function
      # we are mapping. Should this be an option?
      return _high_precision_sum(
          _diagonal_mask(fn(dr, **_kwargs)),
          axis=reduce_axis,
          keepdims=keepdims) * f32(0.5)
  elif isinstance(species, np.ndarray):
    _check_species_dtype(species)
    species_count = int(np.max(species))
    if reduce_axis is not None or keepdims:
      # TODO(schsam): Support reduce_axis with static species.
      raise ValueError
    def fn_mapped(R, **dynamic_kwargs):
      U = f32(0.0)
      _metric = space.map_product(partial(metric, **dynamic_kwargs))
      for i in range(species_count + 1):
        for j in range(i, species_count + 1):
          _kwargs = merge_dicts(kwargs, dynamic_kwargs)
          s_kwargs = _kwargs_to_parameters((i, j), **_kwargs)
          Ra = R[species == i]
          Rb = R[species == j]
          dr = _metric(Ra, Rb)
          if j == i:
            dU = _high_precision_sum(_diagonal_mask(fn(dr, **s_kwargs)))
            U = U + f32(0.5) * dU
          else:
            dU = _high_precision_sum(fn(dr, **s_kwargs))
            U = U + dU
      return U
  elif species is quantity.Dynamic:
    def fn_mapped(R, species, species_count, **dynamic_kwargs):
      _check_species_dtype(species)
      U = f32(0.0)
      N = R.shape[0]
      _metric = space.map_product(partial(metric, **dynamic_kwargs))
      _kwargs = merge_dicts(kwargs, dynamic_kwargs)
      dr = _metric(R, R)
      for i in range(species_count):
        for j in range(species_count):
          s_kwargs = _kwargs_to_parameters((i, j), **_kwargs)
          mask_a = np.array(np.reshape(species == i, (N,)), dtype=R.dtype)
          mask_b = np.array(np.reshape(species == j, (N,)), dtype=R.dtype)
          mask = mask_a[:, np.newaxis] * mask_b[np.newaxis, :]
          if i == j:
            mask = mask * _diagonal_mask(mask)
          dU = mask * fn(dr, **s_kwargs)
          U = U + _high_precision_sum(dU, axis=reduce_axis, keepdims=keepdims)
      return U / f32(2.0)
  else:
    raise ValueError(
        'Species must be None, an ndarray, or Dynamic. Found {}.'.format(
            species))
  return fn_mapped


# Mapping pairwise functional forms to systems using neighbor lists.

def _get_neighborhood_matrix_params(idx, params):
  if isinstance(params, np.ndarray):
    if len(params.shape) == 1:
      return 0.5 * (np.reshape(params, params.shape + (1,)) + params[idx])
    elif len(params.shape) == 2:
      def query(id_a, id_b):
        return params[id_a, id_b]
      query = vmap(vmap(query, (None, 0)))
      return query(np.arange(idx.shape[0], dtype=np.int32), idx)
    elif len(params.shape) == 0:
      return params
    else:
      raise NotImplementedError()
  elif(isinstance(params, int) or isinstance(params, float) or
       np.issubdtype(params, np.integer) or np.issubdtype(params, np.floating)):
    return params
  else:
    raise NotImplementedError 

def _get_neighborhood_species_params(idx, species, params):
  """Get parameters for interactions between species pairs."""
  # TODO(schsam): We should do better error checking here.
  def lookup(species_a, species_b, params):
    return params[species_a, species_b]
  lookup = vmap(vmap(lookup, (None, 0, None)), (0, 0, None))

  neighbor_species = np.reshape(species[idx], idx.shape)
  if isinstance(params, np.ndarray):
    if len(params.shape) == 2:
      return lookup(species, neighbor_species, params)
    elif len(params.shape) == 0:
      return params
    else:
      raise ValueError(
          'Params must be a scalar or a 2d array if using a species lookup.')
  return params

def _neighborhood_kwargs_to_params(idx, species=None, **kwargs):
  out_dict = {}
  for k in kwargs:
    if species is None or (
        isinstance(kwargs[k], np.ndarray) and kwargs[k].ndim == 1):
      out_dict[k] = _get_neighborhood_matrix_params(idx, kwargs[k])
    else:
      out_dict[k] = _get_neighborhood_species_params(idx, species, kwargs[k])
  return out_dict

def _vectorized_cond(pred, fn, operand):
  masked = np.where(pred, operand, 1)
  return np.where(pred, fn(masked), 0)

def pair_neighbor_list(
    fn, metric, species=None, reduce_axis=None, keepdims=False, **kwargs):
  """Promotes a function acting on pairs of particles to use neighbor lists.

  Args:
    fn: A function that takes an ndarray of pairwise distances or displacements
      of shape [n, m] or [n, m, d_in] respectively as well as kwargs specifying
      parameters for the function. fn returns an ndarray of evaluations of shape
      [n, m, d_out].
    metric: A function that takes two ndarray of positions of shape
      [spatial_dimension] and [spatial_dimension] respectively and returns
      an ndarray of distances or displacements of shape [] or [d_in]
      respectively. The metric can optionally take a floating point time as a
      third argument.
    species: A list of species for the different particles. This should either
      be None (in which case it is assumed that all the particles have the same
      species), an integer ndarray of shape [n] with species data, or Dynamic
      in which case the species data will be specified dynamically. Note: that
      dynamic species specification is less efficient, because we cannot
      specialize shape information.
    reduce_axis: A list of axes to reduce over. This is supplied to np.sum and
      so the same convention is used.
    keepdims: A boolean specifying whether the empty dimensions should be kept
      upon reduction. This is supplied to np.sum and so the same convention is
      used.
    kwargs: Arguments providing parameters to the mapped function. In cases
      where no species information is provided these should be either 1) a
      scalar, 2) an ndarray of shape [n], 3) an ndarray of shape [n, n]. If
      species information is provided then the parameters should be specified as
      either 1) a scalar or 2) an ndarray of shape [max_species, max_species].

  Returns:
    A function fn_mapped that takes an ndarray of floats of shape [N, d_in] of
    positions and and ndarray of integers of shape [N, max_neighbors]
    specifying neighbors.
  """
  def fn_mapped(R, neighbor_idx, **dynamic_kwargs):
    d = partial(metric, **dynamic_kwargs)
    d = vmap(vmap(d, (None, 0)))
    mask = neighbor_idx != R.shape[0]
    R_neigh = R[neighbor_idx]
    dR = d(R, R_neigh)
    merged_kwargs = _neighborhood_kwargs_to_params(neighbor_idx, species,
        **merge_dicts(kwargs, dynamic_kwargs))
    out = fn(dR, **merged_kwargs)
    if out.ndim > mask.ndim:
      ddim = out.ndim - mask.ndim
      mask = np.reshape(mask, mask.shape + (1,) * ddim)
    out = np.where(mask, out, 0.)
    return _high_precision_sum(out, reduce_axis, keepdims) / 2.
  return fn_mapped
