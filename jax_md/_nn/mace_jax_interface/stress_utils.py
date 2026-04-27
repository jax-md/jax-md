# stress_utils.py
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import grad
from jax_md import quantity


def box_to_cell(box):
  """
  Convert a box representation to a 3x3 cell matrix.

  Accepts:
    - shape (3,)   : orthorhombic box lengths
    - shape (3, 3) : full cell matrix
  """
  box = jnp.asarray(box)
  if box.ndim == 1:
    if box.shape != (3,):
      raise ValueError(f'Unexpected 1D box shape {box.shape}, expected (3,)')
    return jnp.diag(box)
  if box.shape != (3, 3):
    raise ValueError(
      f'Unexpected box shape {box.shape}, expected (3,) or (3,3)'
    )
  return box


def frac_coords(R, box):
  """
  Convert Cartesian coordinates to fractional coordinates with respect to box.
  """
  H = box_to_cell(box)
  return R @ jnp.linalg.inv(H)


@partial(jax.jit, static_argnames=('energy_fn',))
def configurational_stress_and_pressure(energy_fn, R, box, _unused=None):
  """
  Compute configurational stress tensor and pressure from a fixed-graph energy
  function by differentiating with respect to an infinitesimal strain.

  Parameters
  ----------
  energy_fn
      Energy function with signature:
          energy_fn(R, *, box=H) -> scalar
      where H is the runtime cell matrix.
  R
      Cartesian coordinates, shape (N, 3)
  box
      Box in shape (3,) or (3, 3)

  Returns
  -------
  sigma_conf : array, shape (3, 3)
      Symmetrized configurational stress tensor
  P_conf : scalar
      Configurational pressure = -tr(sigma_conf)/3
  """
  H0 = box_to_cell(box)
  V0 = jnp.linalg.det(H0)
  S = frac_coords(R, H0)

  def strained_energy(eps_flat):
    eps = eps_flat.reshape(3, 3)
    H = H0 @ (jnp.eye(3, dtype=H0.dtype) + eps)
    R_strained = S @ H
    return energy_fn(R_strained, box=H)

  dE_deps = grad(strained_energy)(jnp.zeros((9,), dtype=H0.dtype)).reshape(3, 3)
  sigma_conf = dE_deps / V0
  sigma_conf = 0.5 * (sigma_conf + sigma_conf.T)
  P_conf = -jnp.trace(sigma_conf) / 3.0
  return sigma_conf, P_conf


@partial(jax.jit, static_argnames=('energy_fn',))
def total_pressure_from_stress(energy_fn, R, box, _unused, momentum, mass):
  """
  Compute total pressure = configurational + kinetic.
  """
  _, P_conf = configurational_stress_and_pressure(energy_fn, R, box, None)
  H = box_to_cell(box)
  V = jnp.linalg.det(H)
  K = quantity.kinetic_energy(momentum=momentum, mass=mass)
  dim = R.shape[1]
  P_kin = 2.0 * K / (dim * V)
  return P_conf + P_kin


def make_pressure_snapshot_fn(
  *,
  freeze_graph_fn,
  make_fixed_graph_energy_fn,
  mass,
):
  """
  Build a convenient pressure snapshot function for MD scripts.

  Returned function signature:
      pressure_snapshot_fn(state, box, nbrs) -> pressure
  """

  def pressure_snapshot_fn(state, box, nbrs):
    fixed_graph = freeze_graph_fn(state.position, box=box, neighbors=nbrs)
    fixed_energy_fn = make_fixed_graph_energy_fn(fixed_graph)

    return total_pressure_from_stress(
      fixed_energy_fn,
      state.position,
      box,
      None,
      state.momentum,
      mass,
    )

  return pressure_snapshot_fn
