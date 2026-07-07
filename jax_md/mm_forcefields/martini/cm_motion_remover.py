from jax import lax

import jax.numpy as jnp
from jax_md import dataclasses


def remove_cm_motion(state):
  """Remove center-of-mass momentum from a JaxMD simulation state.

  Intended to operate similar to OpenMM.CMMotionRemover. Works with
  NVEState, NVTNoseHooverState, and any state with .momentum and .mass fields.

  Args:
    state: JaxMD simulation state with .momentum and .mass fields.

  Returns:
    Updated state with center-of-mass momentum removed.
  """
  mass = state.mass
  momentum = state.momentum

  if mass.ndim == 1:
    mass_col = mass[:, None]
  else:
    mass_col = mass

  total_mass = jnp.sum(mass_col)
  p_com = jnp.sum(momentum, axis=0)
  v_com = p_com / total_mass

  correction = mass_col * v_com[None, :]
  corrected_momentum = momentum - correction

  return dataclasses.replace(state, momentum=corrected_momentum)


def make_cm_remover(apply_fn, freq=1):
  """Wrap a JaxMD apply_fn to periodically remove CM motion. Call the
  new apply_fn with the iteration number.

  Args:
    apply_fn: The JaxMD integrator's apply function.
    freq: How often (in steps) to remove CM motion.

  Returns:
    A new apply_fn with CM motion removal baked in.
  """

  def apply_with_cm_removal(i, state, **kwargs):
    state = apply_fn(state, **kwargs)
    state = lax.cond(
      i % freq == 0,
      remove_cm_motion,
      lambda s: s,
      state,
    )
    return state

  return apply_with_cm_removal
