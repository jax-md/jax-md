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

"""Implementation of Particle Mesh Ewald sums following Essmann et al. 1995.



"""

from functools import wraps, partial

from typing import Callable, Tuple, TextIO, Dict, Any, Optional

import jax
import jax.numpy as jnp

import numpy as onp

from jax.scipy.special import erfc  # error function
from jax_md import space, smap, partition, quantity, util



# Types


f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat


# Implementation


# Direct space code


def coulomb_direct(dr: Array, charge_sq: Array, alpha: float) -> Array:
  return charge_sq * erfc(alpha * dr) / dr


def coulomb_direct_pair(
        displacement_fn: DisplacementOrMetricFn,
        charge: Array,
        species: Array=None,
        alpha: float=0.35) -> Callable[[Array], Array]:
  return smap.pair(
      coulomb_direct,
      space.canonicalize_displacement_or_metric(displacement_fn),
      species=species,
      charge_sq=(lambda q1, q2: q1 * q2, charge),
      alpha=alpha
  )


def coulomb_direct_neighbor_list(
        displacement_or_metric: DisplacementOrMetricFn,
        box: Box,
        charge: Array,
        species: Array=None,
        alpha: float=0.35,
        cutoff: float=9.0,
        **neighbor_kwargs) -> Tuple[NeighborFn,
                                    Callable[[Array, NeighborList], Array]]:
  neighbor_fn = partition.neighbor_list(
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      box,
      cutoff,
      **neighbor_kwargs
  )

  masked_energy_fn = lambda dr, **kwargs: jnp.where(
      dr < cutoff,
      coulomb_direct(dr, **kwargs),
      0.)

  energy_fn = smap.pair_neighbor_list(
      masked_energy_fn,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      charge_sq=(lambda q1, q2: q1 * q2, charge),
      alpha=alpha
  )

  return neighbor_fn, energy_fn


# Reciprocal Space code.


def coulomb_recip_ewald(charge: Array,
                        side_length: Array,
                        alpha: float,
                        g_max: float) -> Callable[[Array], Array]:
  def energy_fn(position, **kwargs):
    dim = position.shape[-1]
    V = side_length**dim

    dg = 2 * onp.pi / side_length
    # Just to make the sum inclusive.
    g_range = onp.arange(0, g_max + dg/2, dg)
    g_range = onp.concatenate((-g_range[::-1], g_range[1:]))

    gx, gy, gz = jnp.meshgrid(g_range, g_range, g_range)
    g = jnp.reshape(jnp.stack((gx, gy, gz), axis=-1), (-1, dim))
    g2 = jnp.sum(g**2, axis=-1)
    mask = (g2 < g_max**2) & (g2 > 1e-7)

    Z = (4 * jnp.pi) / V
    S2 = jnp.abs(structure_factor(g, position, charge))**2
    fn = lambda g2: jnp.exp(-g2 / (4*alpha**2)) / g2 * S2

    return Z * util.high_precision_sum(safe_mask(mask, fn, g2, 1))
  return energy_fn


def coulomb_recip_pme(charge: Array,
                      box: Box,
                      grid_points: Array,
                      fractional_coordinates: bool=False,
                      alpha: float=0.34
                      ) -> Callable[[Array], Array]:
  _ibox = space.inverse(box)

  def energy_fn(R, **kwargs):
    q = kwargs.pop('charge', charge)
    _box = kwargs.pop('box', box)
    ibox = space.inverse(kwargs['box']) if 'box' in kwargs else _ibox

    dim = R.shape[-1]
    grid_dimensions = onp.array((grid_points,) * dim)

    grid = map_charges_to_grid(R, q, ibox, grid_dimensions,
                               fractional_coordinates)
    Fgrid = jnp.fft.fftn(grid)

    mx, my, mz = jnp.meshgrid(*[jnp.fft.fftfreq(g) for g in grid_dimensions])
    if jnp.isscalar(_box):
      m_2 = (mx**2 + my**2 + mz**2) * (grid_dimensions[0] * ibox)**2
      V = _box**dim
    else:
      m = (ibox[None, None, None, 0] * mx[:, :, :, None] * grid_dimensions[0] +
           ibox[None, None, None, 1] * my[:, :, :, None] * grid_dimensions[1] +
           ibox[None, None, None, 2] * mz[:, :, :, None] * grid_dimensions[2])
      m_2 = jnp.sum(m**2, axis=-1)
      V = jnp.linalg.det(_box)
    mask = m_2 != 0

    exp_m = 1 / (2 * jnp.pi * V) * jnp.exp(-jnp.pi**2 * m_2 / alpha**2) / m_2
    return util.high_precision_sum(
        mask * exp_m * B(mx, my, mz) * jnp.abs(Fgrid)**2)
  return energy_fn


# Coulomb energy functions.



def coulomb_ewald_neighbor_list(
        displacement_fn: Array,
        box: Array,
        charge: Array,
        species: Array=None,
        alpha: float=0.34,
        g_max: float=5.0
) -> Tuple[NeighborFn,
           Callable[[Array, NeighborList], Array]]:
  neighbor_fn, direct_fn = coulomb_direct_neighbor_list(
      displacement_fn, box, charge, species=species, alpha=alpha)
  recip_fn = coulomb_recip_ewald(charge, box, alpha, g_max)
  def total_energy(R, neighbor, **kwargs):
    return direct_fn(R, neighbor=neighbor, **kwargs) + recip_fn(R, **kwargs)
  return neighbor_fn, total_energy


def coulomb(
        displacement_fn: DisplacementOrMetricFn,
        box: Box,
        charge: Array,
        grid_points: Array,
        species: Array=None,
        alpha: float=0.34,
        fractional_coordinates: bool=False
) -> Callable[[Array], Array]:
  direct_fn = coulomb_direct_pair(
      displacement_fn, charge, species=species, alpha=alpha)
  recip_fn = coulomb_recip_pme(
      charge, box, grid_points, fractional_coordinates, alpha)
  def total_energy(R, **kwargs):
    return direct_fn(R, **kwargs) + recip_fn(R, **kwargs)
  return total_energy


def coulomb_neighbor_list(
        displacement_fn: DisplacementOrMetricFn,
        box: Box,
        charge: Array,
        grid_points: Array,
        species: Array=None,
        alpha: float=0.34,
        cutoff: float=9.0,
        fractional_coordinates: bool=False
) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  nbr_box = jnp.diag(box) if (isinstance(box, jnp.ndarray) and box.ndim == 2) else box
  neighbor_fn, direct_fn = coulomb_direct_neighbor_list(
      displacement_fn, nbr_box, charge, species=species, alpha=alpha, 
      fractional_coordinates=fractional_coordinates, cutoff=cutoff)
  recip_fn = coulomb_recip_pme(
      charge, box, grid_points, fractional_coordinates, alpha)
  def total_energy(R, neighbor, **kwargs):
    return direct_fn(R, neighbor=neighbor, **kwargs) + recip_fn(R, **kwargs)
  return neighbor_fn, total_energy


# Utility functions.


def structure_factor(g, R, q=1):
  if isinstance(q, jnp.ndarray):
    q = q[None, :]
  return util.high_precision_sum(
      q * jnp.exp(1j * jnp.einsum('id,jd->ij', g, R)),
      axis=1
  )


# B-Spline and charge (or structure factor) smearing code.
# TODO(schsam,  samarjeet): For now, we only include support for a fast fourth
# order spline. If you are interested in higher order b-splines or different
# interpolating functions, please raise an issue.


@partial(jnp.vectorize, signature='()->(p)')
def optimized_bspline_4(w):
  coeffs = jnp.zeros((4,))

  coeffs = coeffs.at[2].set(0.5 * w * w)
  coeffs = coeffs.at[0].set(0.5 * (1.0-w) * (1.0-w))
  coeffs = coeffs.at[1].set(1.0 - coeffs[0] - coeffs[2])

  coeffs = coeffs.at[3].set(w * coeffs[2] / 3.0)
  coeffs = coeffs.at[2].set(((1.0 + w) * coeffs[1] + (3.0 - w) * coeffs[2])/3.0)
  coeffs = coeffs.at[0].set((1.0 - w) * coeffs[0] / 3.0)
  coeffs = coeffs.at[1].set(1.0 - coeffs[0] - coeffs[2] - coeffs[3])

  return coeffs


def map_charges_to_grid(
    position: Array,
    charge: Array,
    inverse_box: Box,
    grid_dimensions: Array,
    fractional_coordinates: bool
  ) -> Array:
  """Smears charges over a grid of specified dimensions."""

  Q = jnp.zeros(grid_dimensions)
  N = position.shape[0]

  @partial(jnp.vectorize, signature='(),()->(p)')
  def grid_position(u, K):
    grid = jnp.floor(u).astype(jnp.int32)
    grid = jnp.arange(0, 4) + grid
    return jnp.mod(grid, K)

  @partial(jnp.vectorize, signature='(d),()->(p,p,p,d),(p,p,p)')
  def map_particle_to_grid(position, charge):
    if fractional_coordinates:
      u = transform_gradients(inverse_box, position) * grid_dimensions
    else:
      u = space.raw_transform(inverse_box, position) * grid_dimensions

    w = u - jnp.floor(u)
    coeffs = optimized_bspline_4(w)

    grid_pos = grid_position(u, grid_dimensions)

    accum = charge * (coeffs[0, :, None, None] *
                      coeffs[1, None, :, None] *
                      coeffs[2, None, None, :])
    grid_pos = jnp.concatenate(
        (jnp.broadcast_to(grid_pos[[0], :, None, None], (1, 4, 4, 4)),
         jnp.broadcast_to(grid_pos[[1], None, :, None], (1, 4, 4, 4)),
         jnp.broadcast_to(grid_pos[[2], None, None, :], (1, 4, 4, 4))), axis=0)
    grid_pos = jnp.transpose(grid_pos, (1, 2, 3, 0))

    return grid_pos, accum

  gp, ac = map_particle_to_grid(position, charge)
  gp = jnp.reshape(gp, (-1, 3))
  ac = jnp.reshape(ac, (-1,))

  return Q.at[gp[:, 0], gp[:, 1], gp[:, 2]].add(ac)


@partial(jnp.vectorize, signature='()->()')
def b(m, n=4):
  assert(n == 4)
  k = jnp.arange(n - 1)
  M = optimized_bspline_4(1.0)[1:][::-1]
  prefix = jnp.exp(2 * jnp.pi * 1j * (n - 1) * m)
  return prefix / jnp.sum(M * jnp.exp(2 * jnp.pi * 1j * m * k))


def B(mx, my, mz, n=4):
  """Compute the B factors from Essmann et al. equation 4.7."""
  b_x = b(mx)
  b_y = b(my)
  b_z = b(mz)
  return jnp.abs(b_x)**2 * jnp.abs(b_y)**2 * jnp.abs(b_z)**2


@jax.custom_jvp
def transform_gradients(box, coords):
  # This function acts as a no-op in the forward pass, but it transforms the
  # gradients into fractional coordinates in the backward pass.
  return coords


@transform_gradients.defjvp
def _(primals, tangents):
  box, coords = primals
  dbox, dcoords = tangents
  return coords, space.transform(dbox, coords) + space.transform(box, dcoords)
