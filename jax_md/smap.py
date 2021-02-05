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

from functools import reduce, partial

from typing import Dict, Callable, List, Tuple, Union

from collections import namedtuple
import math
from operator import mul

import numpy as onp

from jax import lax, ops, vmap, eval_shape
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp

from jax_md import quantity, space, util

high_precision_sum = util.high_precision_sum

# Typing

Array = util.Array
f32 = util.f32
f64 = util.f64

i32 = util.i32
i64 = util.i64

DisplacementOrMetricFn = space.DisplacementOrMetricFn


# Mapping potential functional forms to bonds.


def _get_bond_type_parameters(params: Array, bond_type: Array) -> Array:
  """Get parameters for interactions for bonds indexed by a bond-type."""
  # TODO(schsam): We should do better error checking here.
  assert isinstance(bond_type, jnp.ndarray)
  assert len(bond_type.shape) == 1

  if isinstance(params, jnp.ndarray):
    if len(params.shape) == 1:
      return params[bond_type]
    elif len(params.shape) == 0:
      return params
    else:
      raise ValueError(
          'Params must be a scalar or a 1d array if using a bond-type lookup.')
  elif(isinstance(params, int) or isinstance(params, float) or
       jnp.issubdtype(params, jnp.integer) or jnp.issubdtype(params, jnp.floating)):
    return params
  raise NotImplementedError


def _kwargs_to_bond_parameters(bond_type: Array,
                               kwargs: Dict[str, Array]) -> Dict[str, Array]:
  """Extract parameters from keyword arguments."""
  # NOTE(schsam): We could pull out the species case from the generic case.
  for k, v in kwargs.items():
    if bond_type is not None:
      kwargs[k] = _get_bond_type_parameters(v, bond_type)
  return kwargs


def bond(fn: Callable[..., Array],
         displacement_or_metric: DisplacementOrMetricFn,
         static_bonds: Array=None,
         static_bond_types: Array=None,
         ignore_unused_parameters: bool=False,
         **kwargs) -> Callable[..., Array]:
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
    ignore_unused_parameters: A boolean that denotes whether dynamically
      specified keyword arguments passed to the mapped function get ignored
      if they were not first specified as keyword arguments when calling
      `smap.bond(...)`.
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

  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)

  def compute_fn(R, bonds, bond_types, static_kwargs, dynamic_kwargs):
    Ra = R[bonds[:, 0]]
    Rb = R[bonds[:, 1]]
    _kwargs = merge_dicts(static_kwargs, dynamic_kwargs)
    _kwargs = _kwargs_to_bond_parameters(bond_types, _kwargs)
    # NOTE(schsam): This pattern is needed due to JAX issue #912.
    d = vmap(partial(displacement_or_metric, **dynamic_kwargs), 0, 0)
    dr = d(Ra, Rb)
    return high_precision_sum(fn(dr, **_kwargs))

  def mapped_fn(R: Array,
                bonds: Array=None,
                bond_types: Array=None,
                **dynamic_kwargs) -> Array:
    accum = f32(0)

    if bonds is not None:
      accum = accum + compute_fn(R, bonds, bond_types, kwargs, dynamic_kwargs)

    if static_bonds is not None:
      accum = accum + compute_fn(
          R, static_bonds, static_bond_types, kwargs, dynamic_kwargs)

    return accum
  return mapped_fn


# Mapping potential functional forms to pairwise interactions.


def _get_species_parameters(params: Array, species: Array) -> Array:
  """Get parameters for interactions between species pairs."""
  # TODO(schsam): We should do better error checking here.
  if isinstance(params, jnp.ndarray):
    if len(params.shape) == 2:
      return params[species]
    elif len(params.shape) == 0:
      return params
    else:
      raise ValueError(
          'Params must be a scalar or a 2d array if using a species lookup.')
  return params


def _get_matrix_parameters(params: Array) -> Array:
  """Get an NxN parameter matrix from per-particle parameters."""
  if isinstance(params, jnp.ndarray):
    if len(params.shape) == 1:
      # NOTE(schsam): get_parameter_matrix only supports additive parameters.
      return 0.5 * (params[:, jnp.newaxis] + params[jnp.newaxis, :])
    elif len(params.shape) == 0 or len(params.shape) == 2:
      return params
    else:
      raise NotImplementedError
  elif(isinstance(params, int) or isinstance(params, float) or
       jnp.issubdtype(params, jnp.integer) or jnp.issubdtype(params, jnp.floating)):
    return params
  else:
    raise NotImplementedError


def _kwargs_to_parameters(species: Array=None, **kwargs) -> Dict[str, Array]:
  """Extract parameters from keyword arguments."""
  # NOTE(schsam): We could pull out the species case from the generic case.
  s_kwargs = kwargs
  for k, v in s_kwargs.items():
    if species is None:
      s_kwargs[k] = _get_matrix_parameters(v)
    else:
      s_kwargs[k] = _get_species_parameters(v, species)
  return s_kwargs


def _diagonal_mask(X: Array) -> Array:
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
  X = jnp.nan_to_num(X)
  mask = f32(1.0) - jnp.eye(N, dtype=X.dtype)
  if len(X.shape) == 3:
    mask = jnp.reshape(mask, (N, N, 1))
  return mask * X


def _check_species_dtype(species):
  if species.dtype == i32 or species.dtype == i64:
    return
  msg = 'Species has wrong dtype. Expected integer but found {}.'.format(
      species.dtype)
  raise ValueError(msg)


def pair(fn: Callable[..., Array],
         displacement_or_metric: DisplacementOrMetricFn,
         species: Array=None,
         reduce_axis: Tuple[int, ...]=None,
         keepdims: bool=False,
         ignore_unused_parameters: bool=False,
         **kwargs) -> Callable[..., Array]:
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
    reduce_axis: A list of axes to reduce over. This is supplied to jnp.sum and
      so the same convention is used.
    keepdims: A boolean specifying whether the empty dimensions should be kept
      upon reduction. This is supplied to jnp.sum and so the same convention is
      used.
    ignore_unused_parameters: A boolean that denotes whether dynamically
      specified keyword arguments passed to the mapped function get ignored
      if they were not first specified as keyword arguments when calling
      `smap.pair(...)`.
    kwargs: Arguments providing parameters to the mapped function. In cases
      where no species information is provided these should be either 1) a
      scalar, 2) an ndarray of shape [n], 3) an ndarray of shape [n, n]. If
      species information is provided then the parameters should be specified as
      either 1) a scalar or 2) an ndarray of shape [max_species, max_species].

  Returns:
    A function fn_mapped.

    If species is None or statically specified then fn_mapped takes as arguments
    an ndarray of positions of shape [n, spatial_dimension].

    If species is Dynamic then fn_mapped takes as ijnput an ndarray of shape
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

  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)

  if species is None:
    def fn_mapped(R: Array, **dynamic_kwargs) -> Array:
      d = space.map_product(partial(displacement_or_metric, **dynamic_kwargs))
      _kwargs = merge_dicts(kwargs, dynamic_kwargs)
      _kwargs = _kwargs_to_parameters(species, **_kwargs)
      dr = d(R, R)
      # NOTE(schsam): Currently we place a diagonal mask no matter what function
      # we are mapping. Should this be an option?
      return high_precision_sum(_diagonal_mask(fn(dr, **_kwargs)),
                                axis=reduce_axis, keepdims=keepdims) * f32(0.5)
  elif isinstance(species, jnp.ndarray):
    species = onp.array(species)
    _check_species_dtype(species)
    species_count = int(onp.max(species))
    if reduce_axis is not None or keepdims:
      # TODO(schsam): Support reduce_axis with static species.
      raise ValueError
    def fn_mapped(R, **dynamic_kwargs):
      U = f32(0.0)
      d = space.map_product(partial(displacement_or_metric, **dynamic_kwargs))
      for i in range(species_count + 1):
        for j in range(i, species_count + 1):
          _kwargs = merge_dicts(kwargs, dynamic_kwargs)
          s_kwargs = _kwargs_to_parameters((i, j), **_kwargs)
          Ra = R[species == i]
          Rb = R[species == j]
          dr = d(Ra, Rb)
          if j == i:
            dU = high_precision_sum(_diagonal_mask(fn(dr, **s_kwargs)))
            U = U + f32(0.5) * dU
          else:
            dU = high_precision_sum(fn(dr, **s_kwargs))
            U = U + dU
      return U
  elif isinstance(species, int):
    species_count = species
    def fn_mapped(R, species, **dynamic_kwargs):
      _check_species_dtype(species)
      U = f32(0.0)
      N = R.shape[0]
      d = space.map_product(partial(displacement_or_metric, **dynamic_kwargs))
      _kwargs = merge_dicts(kwargs, dynamic_kwargs)
      dr = d(R, R)
      for i in range(species_count):
        for j in range(species_count):
          s_kwargs = _kwargs_to_parameters((i, j), **_kwargs)
          mask_a = jnp.array(jnp.reshape(species == i, (N,)), dtype=R.dtype)
          mask_b = jnp.array(jnp.reshape(species == j, (N,)), dtype=R.dtype)
          mask = mask_a[:, jnp.newaxis] * mask_b[jnp.newaxis, :]
          if i == j:
            mask = mask * _diagonal_mask(mask)
          dU = mask * fn(dr, **s_kwargs)
          U = U + high_precision_sum(dU, axis=reduce_axis, keepdims=keepdims)
      return U / f32(2.0)
  else:
    raise ValueError(
        'Species must be None, an ndarray, or Dynamic. Found {}.'.format(
          species))
  return fn_mapped


# Mapping pairwise functional forms to systems using neighbor lists.

def _get_neighborhood_matrix_params(idx: Array, params: Array) -> Array:
  if isinstance(params, jnp.ndarray):
    if len(params.shape) == 1:
      return 0.5 * (jnp.reshape(params, params.shape + (1,)) + params[idx])
    elif len(params.shape) == 2:
      def query(id_a, id_b):
        return params[id_a, id_b]
      query = vmap(vmap(query, (None, 0)))
      return query(jnp.arange(idx.shape[0], dtype=jnp.int32), idx)
    elif len(params.shape) == 0:
      return params
    else:
      raise NotImplementedError()
  elif(isinstance(params, int) or isinstance(params, float) or
       jnp.issubdtype(params, jnp.integer) or jnp.issubdtype(params, jnp.floating)):
    return params
  else:
    raise NotImplementedError 

def _get_neighborhood_species_params(idx: Array,
                                     species: Array,
                                     params: Array) -> Array:
  """Get parameters for interactions between species pairs."""
  # TODO(schsam): We should do better error checking here.
  def lookup(species_a, species_b, params):
    return params[species_a, species_b]
  lookup = vmap(vmap(lookup, (None, 0, None)), (0, 0, None))

  neighbor_species = jnp.reshape(species[idx], idx.shape)
  if isinstance(params, jnp.ndarray):
    if len(params.shape) == 2:
      return lookup(species, neighbor_species, params)
    elif len(params.shape) == 0:
      return params
    else:
      raise ValueError(
          'Params must be a scalar or a 2d array if using a species lookup.')
  return params

def _neighborhood_kwargs_to_params(idx: Array,
                                   species: Array=None,
                                   **kwargs) -> Dict[str, Array]:
  out_dict = {}
  for k in kwargs:
    if species is None or (
        isinstance(kwargs[k], jnp.ndarray) and kwargs[k].ndim == 1):
      out_dict[k] = _get_neighborhood_matrix_params(idx, kwargs[k])
    else:
      out_dict[k] = _get_neighborhood_species_params(idx, species, kwargs[k])
  return out_dict

def _vectorized_cond(pred: Array,
                     fn: Callable[[Array], Array],
                     operand: Array) -> Array:
  masked = jnp.where(pred, operand, 1)
  return jnp.where(pred, fn(masked), 0)

def pair_neighbor_list(fn: Callable[..., Array],
                       displacement_or_metric: DisplacementOrMetricFn,
                       species: Union[Array, int]=None,
                       reduce_axis: Tuple[int, ...]=None,
                       keepdims: bool=False,
                       ignore_unused_parameters: bool=False,
                       **kwargs) -> Callable[..., Array]:
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
    species: Species information for the different particles. This should either
      be None (in which case it is assumed that all the particles have the same
      species), an integer ndarray of shape [n] with species data, or an integer
      in which case the species data will be specified dynamically in the
      mapped function with at most `species` types of particles. Note: that
      dynamic species specification is less efficient, because we cannot
      specialize shape information.
    reduce_axis: A list of axes to reduce over. This is supplied to jnp.sum and
      so the same convention is used.
    keepdims: A boolean specifying whether the empty dimensions should be kept
      upon reduction. This is supplied to jnp.sum and so the same convention is
      used.
    ignore_unused_parameters: A boolean that denotes whether dynamically
      specified keyword arguments passed to the mapped function get ignored
      if they were not first specified as keyword arguments when calling
      `smap.pair_neighbor_list(...)`.
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

  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)

  def fn_mapped(R, neighbor, **dynamic_kwargs):
    d = partial(displacement_or_metric, **dynamic_kwargs)
    d = vmap(vmap(d, (None, 0)))
    mask = neighbor.idx != R.shape[0]
    R_neigh = R[neighbor.idx]
    dR = d(R, R_neigh)
    merged_kwargs = merge_dicts(kwargs, dynamic_kwargs)
    merged_kwargs = _neighborhood_kwargs_to_params(neighbor.idx,
                                                   species,
                                                   **merged_kwargs)
    out = fn(dR, **merged_kwargs)
    if out.ndim > mask.ndim:
      ddim = out.ndim - mask.ndim
      mask = jnp.reshape(mask, mask.shape + (1,) * ddim)
    out = jnp.where(mask, out, 0.)
    return high_precision_sum(out, reduce_axis, keepdims) / 2.
  return fn_mapped

def triplet(fn: Callable[..., Array],
            displacement_or_metric: DisplacementOrMetricFn,
            species: Array=None,
            reduce_axis: Tuple[int, ...]=None,
            keepdims: bool=False,
            ignore_unused_parameters: bool=False,
            **kwargs) -> Callable[..., Array]:
  """Promotes a function that acts on triples of particles to one on a system.

  Many empirical potentials in jax_md include three-body angular terms (e.g.
  Stillinger Weber). This utility function simplifies the loss computation
  in such cases by converting a function that takes in two pairwise displacements
  or distances to one that only requires the system as input.

  Args:
    fn: A function that takes an ndarray of two distances or displacements
        from a central atom, both of shape [n, m] or [n, m, d_in] respectively,
        as well as kwargs specifying parameters for the function.
    metric: A function that takes two ndarray of positions of shape
        [spatial_dimensions] and [spatial_dimensions] respectively and
        returns an ndarray of distances or displacements of shape [] or [d_in]
        respectively. The metric can optionally take a floating point time as a
        third argument.
    species: A list of species for the different particles. This should either
        be None (in which case it is assumed that all particles have the same
        species), an integer of shape [n] with species data, or Dynamic in
        which case the species data will be specified dynamically. Note:
        that dynamic species specification is less efficient, because we cannot
        specialize shape information.
    reduce_axis: A list of axis to reduce over. This is supplied to np.sum and
        the same convention is used.
    keepdims: A boolean specifying whether the empty dimensions should be kept
        upon reduction. This is supplied to np.sum and so the same convention
        is used.
    ignore_unused_parameters: A boolean that denotes whether dynamically
      specified keyword arguments passed to the mapped function get ignored
      if they were not first specified as keyword arguments when calling
      `smap.triplet(...)`.
    kwargs: Arguement providing parameters to the mapped function. In cases
        where no species information is provided, these should either be 1)
        a scalar, 2) an ndarray of shape [n] based on the central atom,
        3) an ndarray of shape [n, n, n] defining triplet interactions.
        If species information is provided, then the parameters should
        be specified as either 1) a scalar,  2) an ndarray of shape
        [max_species], 3) an ndarray of shape [max_species, max_species,
        max_species] defining triplet interactions.

  Returns:
    A function fn_mapped.

    If species is None or statically specified, then fn_mapped takes as
    arguments an ndarray of positions of shape [n, spatial_dimension].

    If species is Dynamic then fn_mapped takes as input an ndarray of shape
    [n, spatial_dimension], an integer ndarray of species of shape [n], and
    an integer specifying the maximum species.

    The mapped function can also optionally take keyword arguments that get
    threaded through the metric.
  """
  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)

  def extract_parameters_by_dim(kwargs, dim: Union[int, List[int]] = 0):
    """Helper function that extract parameters from a dictionary via dimension."""
    if isinstance(dim, int):
      dim = [dim]
    return {name: value for name, value in kwargs.items() if value.ndim in dim}

  if species is None:
    def fn_mapped(R, **dynamic_kwargs) -> Array:
      d = space.map_product(partial(displacement_or_metric, **dynamic_kwargs))
      _kwargs = merge_dicts(kwargs, dynamic_kwargs)
      _kwargs = _kwargs_to_parameters(species, **_kwargs)
      dR = d(R, R)
      compute_triplet = partial(fn, **_kwargs)
      output = vmap(vmap(vmap(compute_triplet, (None, 0)), (0, None)), 0)(dR, dR)
      return high_precision_sum(output,
                                axis=reduce_axis,
                                keepdims=keepdims) / 2.
  elif isinstance(species, jnp.ndarray):
    def fn_mapped(R, **dynamic_kwargs):
      d = partial(displacement_or_metric, **dynamic_kwargs)
      idx = onp.tile(onp.arange(R.shape[0]), [R.shape[0], 1])
      dR = vmap(vmap(d, (None, 0)))(R, R[idx])

      _kwargs = merge_dicts(kwargs, dynamic_kwargs)

      mapped_args = extract_parameters_by_dim(_kwargs, [3])
      mapped_args = {arg_name: arg_value[species]
          for arg_name, arg_value in mapped_args.items()}
      # While we support 2 dimensional inputs, these often make less sense
      # as the parameters do not depend on the central atom
      unmapped_args = extract_parameters_by_dim(_kwargs, [0])

      if extract_parameters_by_dim(_kwargs, [1, 2]):
        assert ValueError('Improper argument dimensions (1 or 2) not well defined for triplets.')

      def compute_triplet(dR, mapped_args, unmapped_args):
        paired_args = extract_parameters_by_dim(mapped_args, 2)
        paired_args.update(extract_parameters_by_dim(unmapped_args, 2))

        unpaired_args = extract_parameters_by_dim(mapped_args, 0)
        unpaired_args.update(extract_parameters_by_dim(unmapped_args, 0))

        output_fn = lambda dR1, dR2, paired_args: fn(dR1, dR2, **unpaired_args, **paired_args)
        neighbor_args = _neighborhood_kwargs_to_params(idx, species, **paired_args)
        output_fn = vmap(vmap(output_fn, (None, 0, 0)), (0, None, 0))
        return output_fn(dR, dR, neighbor_args)

      output_fn = partial(compute_triplet, unmapped_args=unmapped_args)
      output = vmap(output_fn)(dR, mapped_args)
      return high_precision_sum(output,
                                axis=reduce_axis,
                                keepdims=keepdims) / 2.
  elif species is quantity.Dynamic:
    raise NotImplementedError
  else:
    raise ValueError(
        'Species must be None, an ndarray, or Dynamic. Found {}.'.format(
            species))
  return fn_mapped
