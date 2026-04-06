"""SO(3) utilities for spherical harmonic operations.

This module provides utilities for working with spherical harmonic coefficients,
including coefficient mapping between different orderings and grid-based
transformations.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

import math
from typing import Tuple, List, Dict, NamedTuple

import jax.numpy as jnp
import numpy as np


class CoefficientMapping(NamedTuple):
  """Helper class for coefficients used to reshape l <--> m ordering.

  Attributes:
      lmax: Maximum degree of the spherical harmonics.
      mmax: Maximum order of the spherical harmonics.
      to_m: Matrix to convert from l-major to m-major ordering.
      l_harmonic: Degree (l) for each coefficient index.
      m_harmonic: Absolute order |m| for each coefficient index.
      m_complex: Complex order (signed m) for each coefficient index.
      m_size: Number of coefficients for each m value.
          NOTE: For m>0, this includes BOTH real and imaginary parts,
          i.e. m_size[m] = 2 * (lmax - m + 1) for m > 0.
          This differs from PyTorch which stores only the real count.
      res_size: Total number of coefficients.
  """

  lmax: int
  mmax: int
  to_m: jnp.ndarray
  l_harmonic: jnp.ndarray
  m_harmonic: jnp.ndarray
  m_complex: jnp.ndarray
  m_size: Tuple[int, ...]
  res_size: int
  coefficient_idx_cache: Dict[Tuple[int, int], jnp.ndarray]


def create_coefficient_mapping(lmax: int, mmax: int) -> CoefficientMapping:
  """Create a CoefficientMapping instance.

  Args:
      lmax: Maximum degree of the spherical harmonics.
      mmax: Maximum order of the spherical harmonics.

  Returns:
      CoefficientMapping instance with precomputed arrays.
  """
  l_harmonic_list = []
  m_harmonic_list = []
  m_complex_list = []

  for l in range(lmax + 1):
    mmax_l = min(mmax, l)
    for m in range(-mmax_l, mmax_l + 1):
      m_complex_list.append(m)
      m_harmonic_list.append(abs(m))
      l_harmonic_list.append(l)

  # Use numpy throughout (not jax) since this is precomputation at init time.
  # This avoids TracerArrayConversionError when setup() is called under JIT.
  l_harmonic_np = np.array(l_harmonic_list, dtype=np.int32)
  m_harmonic_np = np.array(m_harmonic_list, dtype=np.int32)
  m_complex_np = np.array(m_complex_list, dtype=np.int32)
  res_size = len(l_harmonic_list)

  num_coefficients = res_size
  to_m = np.zeros([num_coefficients, num_coefficients])
  m_size_list = [0] * (mmax + 1)

  offset = 0
  for m in range(mmax + 1):
    idx_r, idx_i = _complex_idx(m, lmax, mmax, m_complex_np, l_harmonic_np)

    for idx_out, idx_in in enumerate(idx_r):
      to_m[idx_out + offset, idx_in] = 1.0
    offset = offset + len(idx_r)

    for idx_out, idx_in in enumerate(idx_i):
      to_m[idx_out + offset, idx_in] = 1.0
    offset = offset + len(idx_i)

    m_size_list[m] = len(idx_r) + len(idx_i)

  # Pre-compute coefficient indices (numpy)
  coefficient_idx_cache = {}
  for l in range(lmax + 1):
    for m_val in range(lmax + 1):
      mask = (l_harmonic_np <= l) & (m_harmonic_np <= m_val)
      indices = np.arange(len(mask))
      mask_indices = indices[mask]
      coefficient_idx_cache[(l, m_val)] = jnp.array(mask_indices)

  # Convert to JAX arrays only for the final result
  return CoefficientMapping(
    lmax=lmax,
    mmax=mmax,
    to_m=jnp.array(to_m),
    l_harmonic=jnp.array(l_harmonic_np),
    m_harmonic=jnp.array(m_harmonic_np),
    m_complex=jnp.array(m_complex_np),
    m_size=tuple(m_size_list),
    res_size=res_size,
    coefficient_idx_cache=coefficient_idx_cache,
  )


def _complex_idx(
  m: int,
  lmax: int,
  mmax: int,
  m_complex,
  l_harmonic,
) -> Tuple[List[int], List[int]]:
  """Get indices for real and imaginary parts of order m coefficients."""
  # Ensure numpy arrays (not JAX arrays) for index computation
  m_complex_np = np.asarray(m_complex)
  l_harmonic_np = np.asarray(l_harmonic)
  indices = np.arange(len(l_harmonic_np))

  mask_r = (l_harmonic_np <= lmax) & (m_complex_np == m)
  mask_idx_r = indices[mask_r].tolist()

  mask_idx_i = []
  if m != 0:
    mask_i = (l_harmonic_np <= lmax) & (m_complex_np == -m)
    mask_idx_i = indices[mask_i].tolist()

  return mask_idx_r, mask_idx_i


def coefficient_idx(
  mapping: CoefficientMapping, lmax: int, mmax: int
) -> jnp.ndarray:
  """Get indices of coefficients with degree <= lmax and order <= mmax."""
  if (lmax, mmax) in mapping.coefficient_idx_cache:
    return mapping.coefficient_idx_cache[(lmax, mmax)]

  mask = (mapping.l_harmonic <= lmax) & (mapping.m_harmonic <= mmax)
  indices = jnp.arange(len(mask))
  return indices[mask]


class SO3Grid(NamedTuple):
  """Helper class for grid representation of spherical harmonic irreps.

  Attributes:
      lmax: Maximum degree of the spherical harmonics.
      mmax: Maximum order of the spherical harmonics.
      lat_resolution: Latitude resolution of the grid.
      long_resolution: Longitude resolution of the grid.
      mapping: CoefficientMapping for index manipulations.
      to_grid_mat: Matrix to convert coefficients to grid, shape [lat, long, num_coeffs].
      from_grid_mat: Matrix to convert grid to coefficients, shape [lat, long, num_coeffs].
      rescale: Whether rescaling was applied.
  """

  lmax: int
  mmax: int
  lat_resolution: int
  long_resolution: int
  mapping: CoefficientMapping
  to_grid_mat: jnp.ndarray
  from_grid_mat: jnp.ndarray
  rescale: bool


def create_so3_grid(
  lmax: int,
  mmax: int,
  resolution: int | None = None,
  rescale: bool = True,
  normalization: str = 'integral',
) -> SO3Grid:
  """Create an SO3Grid instance for grid-based spherical harmonic operations.

  Uses PyTorch's e3nn library for exact numerical matching with pretrained
  weights when available. Falls back to numpy computation otherwise.

  Args:
      lmax: Maximum degree of the spherical harmonics.
      mmax: Maximum order of the spherical harmonics.
      resolution: Grid resolution (if None, computed from lmax/mmax).
      rescale: Whether to rescale based on mmax.
      normalization: Spherical harmonic normalization convention.

  Returns:
      SO3Grid instance with precomputed transformation matrices.
  """
  lat_resolution = 2 * (lmax + 1)
  if lmax == mmax:
    long_resolution = 2 * (mmax + 1) + 1
  else:
    long_resolution = 2 * mmax + 1

  if resolution is not None:
    lat_resolution = resolution
    long_resolution = resolution

  mapping = create_coefficient_mapping(lmax, lmax)

  to_grid_mat, from_grid_mat = _compute_grid_matrices(
    lmax, mmax, lat_resolution, long_resolution, rescale, normalization, mapping
  )

  return SO3Grid(
    lmax=lmax,
    mmax=mmax,
    lat_resolution=lat_resolution,
    long_resolution=long_resolution,
    mapping=mapping,
    to_grid_mat=to_grid_mat,
    from_grid_mat=from_grid_mat,
    rescale=rescale,
  )


def _compute_grid_matrices(
  lmax: int,
  mmax: int,
  lat_resolution: int,
  long_resolution: int,
  rescale: bool,
  normalization: str,
  mapping: CoefficientMapping,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute transformation matrices between coefficients and grid.

  Tries e3nn (via torch) first for exact PyTorch compatibility,
  falls back to numpy computation.
  """
  # Try using PyTorch e3nn for exact matching with pretrained weights
  try:
    return _compute_grid_matrices_e3nn(
      lmax,
      mmax,
      lat_resolution,
      long_resolution,
      rescale,
      normalization,
      mapping,
    )
  except ImportError:
    pass

  # Fall back to numpy computation
  return _compute_grid_matrices_numpy(
    lmax, mmax, lat_resolution, long_resolution, rescale, mapping
  )


def _compute_grid_matrices_e3nn(
  lmax: int,
  mmax: int,
  lat_resolution: int,
  long_resolution: int,
  rescale: bool,
  normalization: str,
  mapping: CoefficientMapping,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute grid matrices using e3nn (exact match with PyTorch)."""
  import torch
  from e3nn.o3 import FromS2Grid, ToS2Grid

  to_grid_obj = ToS2Grid(
    lmax,
    (lat_resolution, long_resolution),
    normalization=normalization,
  )
  to_grid_mat = torch.einsum(
    'mbi, am -> bai', to_grid_obj.shb, to_grid_obj.sha
  ).detach()

  # Rescale based on mmax
  if rescale and lmax != mmax:
    for lval in range(lmax + 1):
      if lval <= mmax:
        continue
      start_idx = lval**2
      length = 2 * lval + 1
      rescale_factor = math.sqrt(length / (2 * mmax + 1))
      to_grid_mat[:, :, start_idx : (start_idx + length)] *= rescale_factor

  # Get coefficient indices for mmax subset
  pt_mapping = _PyTorchCoefficientMapping(lmax, lmax)
  coef_idx = pt_mapping.coefficient_idx(lmax, mmax)
  to_grid_mat = to_grid_mat[:, :, coef_idx]

  from_grid_obj = FromS2Grid(
    (lat_resolution, long_resolution),
    lmax,
    normalization=normalization,
  )
  from_grid_mat = torch.einsum(
    'am, mbi -> bai', from_grid_obj.sha, from_grid_obj.shb
  ).detach()

  # Rescale based on mmax
  if rescale and lmax != mmax:
    for lval in range(lmax + 1):
      if lval <= mmax:
        continue
      start_idx = lval**2
      length = 2 * lval + 1
      rescale_factor = math.sqrt(length / (2 * mmax + 1))
      from_grid_mat[:, :, start_idx : (start_idx + length)] *= rescale_factor

  from_grid_mat = from_grid_mat[:, :, coef_idx]

  return (
    jnp.array(to_grid_mat.numpy()),
    jnp.array(from_grid_mat.numpy()),
  )


class _PyTorchCoefficientMapping:
  """Minimal reimplementation of PyTorch CoefficientMapping for grid matrix indexing."""

  def __init__(self, lmax: int, mmax: int):
    self.lmax = lmax
    self.mmax = mmax

    l_harmonic = []
    m_harmonic = []
    m_complex_list = []

    for l in range(lmax + 1):
      mmax_l = min(mmax, l)
      for m in range(-mmax_l, mmax_l + 1):
        m_complex_list.append(m)
        m_harmonic.append(abs(m))
        l_harmonic.append(l)

    self._l_harmonic = np.array(l_harmonic)
    self._m_harmonic = np.array(m_harmonic)

  def coefficient_idx(self, lmax: int, mmax: int):
    """Get indices matching PyTorch CoefficientMapping.coefficient_idx()."""
    import torch

    mask = (self._l_harmonic <= lmax) & (self._m_harmonic <= mmax)
    indices = np.arange(len(mask))
    return torch.tensor(indices[mask]).long()


def _compute_grid_matrices_numpy(
  lmax: int,
  mmax: int,
  lat_resolution: int,
  long_resolution: int,
  rescale: bool,
  mapping: CoefficientMapping,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute transformation matrices using numpy (fallback).

  Note: This may produce slightly different numerical results than e3nn
  due to different spherical harmonic conventions. Use e3nn version
  for exact compatibility with pretrained PyTorch weights.
  """
  theta = np.linspace(0, np.pi, lat_resolution, endpoint=False)
  theta = theta + np.pi / (2 * lat_resolution)

  phi = np.linspace(0, 2 * np.pi, long_resolution, endpoint=False)

  sph_size = (lmax + 1) ** 2
  to_grid = np.zeros((lat_resolution, long_resolution, sph_size))
  from_grid = np.zeros((lat_resolution, long_resolution, sph_size))

  cos_theta = np.cos(theta)

  for l in range(lmax + 1):
    for m in range(-l, l + 1):
      idx = l * l + l + m

      plm = _associated_legendre(l, abs(m), cos_theta)

      norm = np.sqrt(
        (2 * l + 1)
        / (4 * np.pi)
        * math.factorial(l - abs(m))
        / math.factorial(l + abs(m))
      )

      if m >= 0:
        azimuth = np.cos(m * phi)
      else:
        azimuth = np.sin(abs(m) * phi)

      for i in range(lat_resolution):
        for j in range(long_resolution):
          to_grid[i, j, idx] = norm * plm[i] * azimuth[j]
          from_grid[i, j, idx] = norm * plm[i] * azimuth[j]

  sin_theta = np.sin(theta)
  quad_weight = 2 * np.pi / long_resolution * np.pi / lat_resolution
  for i in range(lat_resolution):
    from_grid[i, :, :] *= sin_theta[i] * quad_weight

  if rescale and lmax != mmax:
    for lval in range(lmax + 1):
      if lval <= mmax:
        continue
      start_idx = lval**2
      length = 2 * lval + 1
      rescale_factor = np.sqrt(length / (2 * mmax + 1))
      to_grid[:, :, start_idx : start_idx + length] *= rescale_factor
      from_grid[:, :, start_idx : start_idx + length] *= rescale_factor

  coef_idx = coefficient_idx(mapping, lmax, mmax)
  coef_idx_np = np.array(coef_idx)
  to_grid = to_grid[:, :, coef_idx_np]
  from_grid = from_grid[:, :, coef_idx_np]

  return jnp.array(to_grid), jnp.array(from_grid)


def _associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
  """Compute associated Legendre polynomial P_l^m(x)."""
  if m > l:
    return np.zeros_like(x)

  pmm = np.ones_like(x)
  if m > 0:
    somx2 = np.sqrt((1 - x) * (1 + x))
    fact = 1.0
    for i in range(1, m + 1):
      pmm = -pmm * fact * somx2
      fact += 2.0

  if l == m:
    return pmm

  pmmp1 = x * (2 * m + 1) * pmm

  if l == m + 1:
    return pmmp1

  pll = np.zeros_like(x)
  for ll in range(m + 2, l + 1):
    pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
    pmm = pmmp1
    pmmp1 = pll

  return pll


def to_grid(
  grid: SO3Grid, embedding: jnp.ndarray, lmax: int, mmax: int
) -> jnp.ndarray:
  """Convert spherical harmonic coefficients to grid representation.

  Args:
      grid: SO3Grid instance.
      embedding: Coefficients of shape [batch, num_coefficients, channels].
      lmax: Maximum degree to use.
      mmax: Maximum order to use.

  Returns:
      Grid representation of shape [batch, lat, long, channels].
  """
  coef_idx = coefficient_idx(grid.mapping, lmax, mmax)
  to_grid_mat = grid.to_grid_mat[:, :, : len(coef_idx)]
  # [lat, long, coef] x [batch, coef, channels] -> [batch, lat, long, channels]
  return jnp.einsum('bai,zic->zbac', to_grid_mat, embedding)


def from_grid(
  grid: SO3Grid, grid_vals: jnp.ndarray, lmax: int, mmax: int
) -> jnp.ndarray:
  """Convert grid representation to spherical harmonic coefficients.

  Args:
      grid: SO3Grid instance.
      grid_vals: Grid values of shape [batch, lat, long, channels].
      lmax: Maximum degree to use.
      mmax: Maximum order to use.

  Returns:
      Coefficients of shape [batch, num_coefficients, channels].
  """
  coef_idx = coefficient_idx(grid.mapping, lmax, mmax)
  from_grid_mat = grid.from_grid_mat[:, :, : len(coef_idx)]
  # [lat, long, coef] x [batch, lat, long, channels] -> [batch, coef, channels]
  return jnp.einsum('bai,zbac->zic', from_grid_mat, grid_vals)
