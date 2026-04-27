# npt_bridge_utils.py
from __future__ import annotations

import jax
import jax.numpy as jnp


def box_to_cell(box):
  box = jnp.asarray(box)
  if box.ndim == 1:
    return jnp.diag(box)
  if box.shape != (3, 3):
    raise ValueError(
      f'Unexpected box shape {box.shape}, expected (3,) or (3,3)'
    )
  return box


def apply_perturbation(box, perturbation):
  """
  Apply a JAX-MD-style box perturbation.

  - scalar perturbation: isotropic scaling
  - matrix perturbation: general box deformation
  """
  H0 = box_to_cell(box)

  if perturbation is None:
    return H0

  pert = jnp.asarray(perturbation)

  if pert.ndim == 0:
    return H0 * pert

  if pert.shape == (3, 3):
    return H0 @ pert

  raise ValueError(
    f'Unexpected perturbation shape {pert.shape}, expected scalar or (3,3)'
  )


def make_perturbation_compatible_energy_fn(
  *,
  fixed_graph,
  make_fixed_graph_energy_fn,
):
  """
  Wrap a fixed-graph bridge energy so it matches the API expected by
  jax_md.quantity.pressure / stress and by NPT machinery.

  Returned signature:
      energy_fn(R, *, box, perturbation=None, **kwargs) -> scalar
  """
  fixed_energy_fn = make_fixed_graph_energy_fn(fixed_graph)

  @jax.jit
  def energy_fn(R, *, box, perturbation=None, **kwargs):
    del kwargs
    H = apply_perturbation(box, perturbation)
    return fixed_energy_fn(R, box=H)

  return energy_fn
