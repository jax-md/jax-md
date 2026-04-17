"""Rotation utilities for Wigner D-matrix computation.

This module provides functions for computing Wigner D-matrices which are used
to rotate spherical harmonic representations in equivariant neural networks.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-7

# Path to Jacobi matrices file (PyTorch format)
_JD_FILE = os.path.join(os.path.dirname(__file__), '..', 'Jd.pt')
# Path to Jacobi matrices file (numpy format)
_JD_NPY_DIR = os.path.join(os.path.dirname(__file__), '..', 'Jd_npy')


@jax.custom_jvp
def safe_acos(x: jnp.ndarray) -> jnp.ndarray:
  """Numerically stable arccos matching PyTorch's Safeacos.

  Forward pass: exact arccos(x).
  Backward pass: gradient uses clamped x to avoid NaN near ±1.
  """
  return jnp.arccos(x)


@safe_acos.defjvp
def _safe_acos_jvp(primals, tangents):
  (x,) = primals
  (x_dot,) = tangents
  primal_out = safe_acos(x)
  x_clamped = jnp.clip(x, -1 + EPS, 1 - EPS)
  denom = jnp.sqrt(1 - x_clamped**2)
  denom = jnp.maximum(denom, EPS)
  tangent_out = -x_dot / denom
  return primal_out, tangent_out


@jax.custom_jvp
def safe_atan2(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
  """Numerically stable atan2 with safe backward pass.

  Forward: exact atan2(y, x).
  Backward: clamps denominator x^2+y^2 to avoid NaN at the origin.
  """
  return jnp.arctan2(y, x)


@safe_atan2.defjvp
def _safe_atan2_jvp(primals, tangents):
  y, x = primals
  y_dot, x_dot = tangents
  primal_out = safe_atan2(y, x)
  denom = jnp.maximum(x**2 + y**2, EPS**2)
  tangent_out = (x * y_dot - y * x_dot) / denom
  return primal_out, tangent_out


def init_edge_rot_euler_angles(
  edge_distance_vec: jnp.ndarray,
  rng_key: jax.random.PRNGKey | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute Euler angles for rotating to edge-aligned frame.

  Given edge distance vectors, computes the Euler angles (ZYZ convention)
  needed to rotate the coordinate frame to align with each edge.

  Args:
      edge_distance_vec: Edge vectors of shape [num_edges, 3].
      rng_key: Optional PRNG key for random gamma angles (training mode).
          If None, gamma is set to zero (deterministic inference mode).
          The PyTorch implementation uses random gamma during training.

  Returns:
      Tuple of (gamma, beta, alpha) Euler angles, each of shape [num_edges].
  """
  # Normalize the edge vectors
  norm = jnp.linalg.norm(edge_distance_vec, axis=-1, keepdims=True)
  xyz = edge_distance_vec / jnp.maximum(norm, EPS)
  xyz = jnp.clip(xyz, -1.0, 1.0)

  x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

  # Latitude (beta) - angle from y-axis
  beta = safe_acos(y)

  # Longitude (alpha) - angle in xz plane
  alpha = safe_atan2(x, z)

  # Gamma (roll) - random for training, zero for inference.
  # For deterministic inference, gamma=0 is valid because the SO(2)
  # convolution is axially symmetric and gamma cancels out.
  if rng_key is not None:
    gamma = jax.random.uniform(
      rng_key, alpha.shape, minval=0.0, maxval=2 * jnp.pi
    )
  else:
    gamma = jnp.zeros_like(alpha)

  # Intrinsic to extrinsic swap (negative signs)
  return -gamma, -beta, -alpha


def load_jacobi_matrices_from_file(lmax: int) -> List[jnp.ndarray]:
  """Load precomputed Jacobi matrices.

  Tries loading in order:
  1. Numpy .npy files (no torch dependency)
  2. PyTorch .pt file (requires torch)
  3. Computed from scratch (fallback)

  Args:
      lmax: Maximum degree l for which to load matrices.

  Returns:
      List of Jacobi matrices for l = 0, 1, ..., lmax.
  """
  # Try numpy files first (no torch dependency)
  if os.path.isdir(_JD_NPY_DIR):
    try:
      Jd_list = []
      for l in range(lmax + 1):
        path = os.path.join(_JD_NPY_DIR, f'Jd_{l}.npy')
        if os.path.exists(path):
          Jd_list.append(jnp.array(np.load(path)))
        else:
          break
      if len(Jd_list) == lmax + 1:
        return Jd_list
    except Exception:
      pass

  # Try PyTorch file
  try:
    import torch

    Jd_torch = torch.load(_JD_FILE, map_location='cpu', weights_only=False)
    Jd_list = [jnp.array(Jd_torch[l].numpy()) for l in range(lmax + 1)]

    # Cache as numpy for future torch-free loading
    _save_jacobi_as_numpy(Jd_list)

    return Jd_list
  except (ImportError, FileNotFoundError):
    pass

  # Fall back to computing
  return compute_jacobi_matrices(lmax)


def _save_jacobi_as_numpy(Jd_list: List[jnp.ndarray]) -> None:
  """Cache Jacobi matrices as numpy files for torch-free loading."""
  try:
    os.makedirs(_JD_NPY_DIR, exist_ok=True)
    for l, Jd in enumerate(Jd_list):
      path = os.path.join(_JD_NPY_DIR, f'Jd_{l}.npy')
      np.save(path, np.array(Jd))
  except Exception:
    pass  # Non-critical: caching is best-effort


def compute_jacobi_matrices(lmax: int) -> List[jnp.ndarray]:
  """Compute Jacobi matrices (J matrices) for Wigner D-matrix computation.

  The Jacobi matrices are used in the computation of Wigner D-matrices
  via the relation D = X_a @ J @ X_b @ J @ X_c where X are z-rotation
  matrices.

  Note: For best compatibility with pretrained PyTorch weights, use
  load_jacobi_matrices_from_file() instead.

  Args:
      lmax: Maximum degree l for which to compute matrices.

  Returns:
      List of Jacobi matrices for l = 0, 1, ..., lmax.
  """
  Jd_list = []

  for l in range(lmax + 1):
    J = _compute_wigner_d_small(l, np.pi / 2)
    Jd_list.append(jnp.array(J))

  return Jd_list


def _compute_wigner_d_small(l: int, beta: float) -> np.ndarray:
  """Compute small Wigner d-matrix d^l(beta).

  Args:
      l: Degree of the representation.
      beta: Rotation angle.

  Returns:
      d-matrix of shape [2l+1, 2l+1].
  """
  if l == 0:
    return np.array([[1.0]])

  size = 2 * l + 1
  d = np.zeros((size, size))

  c = np.cos(beta / 2)
  s = np.sin(beta / 2)

  for m in range(-l, l + 1):
    for mp in range(-l, l + 1):
      idx_m = m + l
      idx_mp = mp + l

      val = 0.0
      s_min = max(0, m - mp)
      s_max = min(l + m, l - mp)

      for s_idx in range(s_min, s_max + 1):
        sign = (-1) ** (mp - m + s_idx)
        num = (
          math.factorial(l + m)
          * math.factorial(l - m)
          * math.factorial(l + mp)
          * math.factorial(l - mp)
        )
        denom = (
          math.factorial(l + m - s_idx)
          * math.factorial(s_idx)
          * math.factorial(mp - m + s_idx)
          * math.factorial(l - mp - s_idx)
        )
        power_c = 2 * l + m - mp - 2 * s_idx
        power_s = mp - m + 2 * s_idx

        term = sign * np.sqrt(num) / denom
        if power_c >= 0 and power_s >= 0:
          term *= c**power_c * s**power_s
        else:
          term = 0.0
        val += term

      d[idx_m, idx_mp] = val

  return d


def _z_rot_mat(angle: jnp.ndarray, l: int) -> jnp.ndarray:
  """Compute z-rotation matrix for degree l representation.

  Matches PyTorch implementation: frequencies go from l to -l,
  cos on diagonal, sin on anti-diagonal.

  Args:
      angle: Rotation angles of shape [batch].
      l: Degree of the representation.

  Returns:
      Rotation matrices of shape [batch, 2l+1, 2l+1].
  """
  batch_size = angle.shape[0]
  size = 2 * l + 1

  M = jnp.zeros((batch_size, size, size))

  # Match PyTorch: inds 0..2l, reversed_inds 2l..0, frequencies l..-l
  # IMPORTANT: sin must be set BEFORE cos, because when i == j (middle
  # element, freq=0) the cos value must overwrite the sin value.
  for i in range(size):
    freq = l - i  # frequencies go from l to -l
    j = size - 1 - i  # anti-diagonal index

    M = M.at[:, i, j].set(jnp.sin(freq * angle))
    M = M.at[:, i, i].set(jnp.cos(freq * angle))

  return M


def wigner_D(
  l: int,
  alpha: jnp.ndarray,
  beta: jnp.ndarray,
  gamma: jnp.ndarray,
  Jd: jnp.ndarray,
) -> jnp.ndarray:
  """Compute Wigner D-matrix for degree l.

  D^l(alpha, beta, gamma) = X_alpha @ J @ X_beta @ J @ X_gamma

  where X_angle is a z-rotation matrix and J is the Jacobi matrix.

  Args:
      l: Degree of the representation.
      alpha: First Euler angle, shape [batch].
      beta: Second Euler angle, shape [batch].
      gamma: Third Euler angle, shape [batch].
      Jd: Jacobi matrix for degree l, shape [2l+1, 2l+1].

  Returns:
      Wigner D-matrices of shape [batch, 2l+1, 2l+1].
  """
  Xa = _z_rot_mat(alpha, l)
  Xb = _z_rot_mat(beta, l)
  Xc = _z_rot_mat(gamma, l)

  # D = Xa @ J @ Xb @ J @ Xc
  result = jnp.einsum('bij,jk->bik', Xa, Jd)
  result = jnp.einsum('bij,bjk->bik', result, Xb)
  result = jnp.einsum('bij,jk->bik', result, Jd)
  result = jnp.einsum('bij,bjk->bik', result, Xc)

  return result


def eulers_to_wigner(
  eulers: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
  start_lmax: int,
  end_lmax: int,
  Jd_list: List[jnp.ndarray],
) -> jnp.ndarray:
  """Compute block-diagonal Wigner D-matrix from Euler angles.

  Args:
      eulers: Tuple of (alpha, beta, gamma) Euler angles, each [num_edges].
      start_lmax: Starting degree (usually 0).
      end_lmax: Ending degree (inclusive).
      Jd_list: List of Jacobi matrices for each degree.

  Returns:
      Block-diagonal Wigner matrix of shape [num_edges, size, size]
      where size = (end_lmax + 1)^2 - start_lmax^2.
  """
  alpha, beta, gamma = eulers
  num_edges = alpha.shape[0]

  size = (end_lmax + 1) ** 2 - start_lmax**2
  wigner = jnp.zeros((num_edges, size, size))

  start = 0
  for l in range(start_lmax, end_lmax + 1):
    block = wigner_D(l, alpha, beta, gamma, Jd_list[l])
    block_size = 2 * l + 1
    end = start + block_size

    wigner = wigner.at[:, start:end, start:end].set(block)
    start = end

  return wigner


def get_wigner_and_mapping(
  edge_distance_vec: jnp.ndarray,
  lmax: int,
  mmax: int,
  Jd_list: List[jnp.ndarray],
  to_m: jnp.ndarray,
  coefficient_index: jnp.ndarray,
  rng_key: jax.random.PRNGKey | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute Wigner matrices with M-mapping for SO(2) convolutions.

  This combines the Wigner rotation with the coefficient mapping to
  facilitate efficient SO(2) convolutions in the edge-aligned frame.

  Args:
      edge_distance_vec: Edge vectors of shape [num_edges, 3].
      lmax: Maximum degree l.
      mmax: Maximum order m.
      Jd_list: List of Jacobi matrices.
      to_m: Coefficient mapping matrix from l-major to m-major ordering.
      coefficient_index: Indices for selecting lmax/mmax subset.
      rng_key: Optional PRNG key for random gamma (training mode).

  Returns:
      Tuple of (wigner_and_M_mapping, wigner_and_M_mapping_inv).
  """
  euler_angles = init_edge_rot_euler_angles(edge_distance_vec, rng_key=rng_key)
  wigner = eulers_to_wigner(euler_angles, 0, lmax, Jd_list)
  wigner_inv = jnp.transpose(wigner, (0, 2, 1))

  # Select subset of coefficients if mmax != lmax
  if mmax != lmax:
    wigner = wigner[:, coefficient_index, :]  # [E, m_dim, l_dim]
    wigner_inv = wigner_inv[:, :, coefficient_index]  # [E, l_dim, m_dim]

  # Combine with M mapping
  to_m_selected = to_m
  wigner_and_M_mapping = jnp.einsum('mk,nkj->nmj', to_m_selected, wigner)
  wigner_and_M_mapping_inv = jnp.einsum(
    'njk,mk->njm', wigner_inv, to_m_selected
  )

  return wigner_and_M_mapping, wigner_and_M_mapping_inv
