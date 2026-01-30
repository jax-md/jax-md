# Copyright 2024 Google LLC
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

"""Rotation utilities for Wigner D-matrix computation.

This module provides functions for computing Wigner D-matrices which are used
to rotate spherical harmonic representations in equivariant neural networks.

Ported from FairChem's UMA implementation.
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-7

# Path to Jacobi matrices file
_JD_FILE = os.path.join(os.path.dirname(__file__), '..', 'Jd.pt')


def safe_acos(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable arccos with gradient clipping."""
    x_clamped = jnp.clip(x, -1 + EPS, 1 - EPS)
    return jnp.arccos(x_clamped)


def safe_atan2(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable atan2."""
    return jnp.arctan2(y, x)


def init_edge_rot_euler_angles(
    edge_distance_vec: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Euler angles for rotating to edge-aligned frame.

    Given edge distance vectors, computes the Euler angles (ZYZ convention)
    needed to rotate the coordinate frame to align with each edge.

    Args:
        edge_distance_vec: Edge vectors of shape [num_edges, 3].

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

    # Random gamma (roll) - set to random for training stability
    # For inference, this should be deterministic
    gamma = jnp.zeros_like(alpha)

    # Intrinsic to extrinsic swap (negative signs)
    return -gamma, -beta, -alpha


def load_jacobi_matrices_from_file(lmax: int) -> List[jnp.ndarray]:
    """Load precomputed Jacobi matrices from the bundled Jd.pt file.

    This ensures exact compatibility with the PyTorch implementation.

    Args:
        lmax: Maximum degree l for which to load matrices.

    Returns:
        List of Jacobi matrices for l = 0, 1, ..., lmax.
    """
    try:
        import torch
        Jd_torch = torch.load(_JD_FILE, map_location='cpu', weights_only=False)
        return [jnp.array(Jd_torch[l].numpy()) for l in range(lmax + 1)]
    except ImportError:
        # Fall back to computing if torch not available
        return compute_jacobi_matrices(lmax)
    except FileNotFoundError:
        # Fall back to computing if file not found
        return compute_jacobi_matrices(lmax)


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
        # Use the actual Jacobi polynomial approach
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

            # Compute d^l_{m,m'}(beta) using the formula
            val = 0.0
            s_min = max(0, m - mp)
            s_max = min(l + m, l - mp)

            for s_idx in range(s_min, s_max + 1):
                sign = (-1) ** (mp - m + s_idx)
                num = (
                    math.factorial(l + m) *
                    math.factorial(l - m) *
                    math.factorial(l + mp) *
                    math.factorial(l - mp)
                )
                denom = (
                    math.factorial(l + m - s_idx) *
                    math.factorial(s_idx) *
                    math.factorial(mp - m + s_idx) *
                    math.factorial(l - mp - s_idx)
                )
                power_c = 2 * l + m - mp - 2 * s_idx
                power_s = mp - m + 2 * s_idx

                term = sign * np.sqrt(num) / denom
                if power_c >= 0 and power_s >= 0:
                    term *= c ** power_c * s ** power_s
                else:
                    term = 0.0
                val += term

            d[idx_m, idx_mp] = val

    return d


def _z_rot_mat(angle: jnp.ndarray, l: int) -> jnp.ndarray:
    """Compute z-rotation matrix for degree l representation.

    Args:
        angle: Rotation angles of shape [batch].
        l: Degree of the representation.

    Returns:
        Rotation matrices of shape [batch, 2l+1, 2l+1].
    """
    batch_size = angle.shape[0]
    size = 2 * l + 1

    # Initialize as zeros
    M = jnp.zeros((batch_size, size, size))

    # Build the matrix using a loop (will be traced by JAX)
    for i in range(size):
        m = l - i  # m goes from l to -l
        freq = m

        # Diagonal: cos(m * angle)
        M = M.at[:, i, i].set(jnp.cos(freq * angle))

        # Anti-diagonal: sin(m * angle)
        j = size - 1 - i
        M = M.at[:, i, j].set(jnp.sin(freq * angle))

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
    # J is not batched, so we need to handle broadcasting
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

    size = (end_lmax + 1) ** 2 - start_lmax ** 2
    wigner = jnp.zeros((num_edges, size, size))

    start = 0
    for l in range(start_lmax, end_lmax + 1):
        block = wigner_D(l, alpha, beta, gamma, Jd_list[l])
        block_size = 2 * l + 1
        end = start + block_size

        # Set the diagonal block
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

    Returns:
        Tuple of (wigner_and_M_mapping, wigner_and_M_mapping_inv).
    """
    euler_angles = init_edge_rot_euler_angles(edge_distance_vec)
    wigner = eulers_to_wigner(euler_angles, 0, lmax, Jd_list)
    wigner_inv = jnp.transpose(wigner, (0, 2, 1))

    # Select subset of coefficients if mmax != lmax
    if mmax != lmax:
        wigner = wigner[:, coefficient_index, :][:, :, coefficient_index]
        wigner_inv = wigner_inv[:, coefficient_index, :][:, :, coefficient_index]

    # Combine with M mapping
    # wigner_and_M_mapping: [num_m_coeffs, num_edges, num_m_coeffs]
    to_m_selected = to_m
    wigner_and_M_mapping = jnp.einsum(
        'mk,nkj->nmj', to_m_selected, wigner
    )
    wigner_and_M_mapping_inv = jnp.einsum(
        'njk,mk->njm', wigner_inv, to_m_selected
    )

    return wigner_and_M_mapping, wigner_and_M_mapping_inv
