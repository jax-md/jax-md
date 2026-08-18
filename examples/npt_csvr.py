"""Run a small Lennard-Jones liquid at two NPT state points using CSVR."""

import os

import jax
import jax.numpy as jnp
from jax import lax
from jax import random

from jax_md import energy
from jax_md import quantity
from jax_md import simulate
from jax_md import space

jax.config.update('jax_enable_x64', True)

SMOKE_TEST = os.environ.get('READTHEDOCS', False)
NSTEPS_SIM = 1000 if SMOKE_TEST else 50000
WRITE_EVERY = 100
STATE_POINTS = ((1.0, 0.4), (1.0, 0.8))

N = 64
box = (N / 0.7) ** (1 / 3)
grid = jnp.stack(jnp.meshgrid(*[jnp.arange(4)] * 3, indexing='ij'), axis=-1)
position = (grid.reshape(-1, 3) + 0.5) / 4
displacement, shift = space.periodic_general(box)
energy_fn = energy.lennard_jones_pair(
  displacement, sigma=1.0, epsilon=1.0, r_onset=1.8, r_cutoff=2.0
)

for index, (temperature, pressure) in enumerate(STATE_POINTS):
  init_fn, apply_fn = simulate.npt_csvr(
    energy_fn,
    shift,
    dt=0.002,
    pressure=pressure,
    kT=temperature,
    tau_p=1.0,
    tau_t=0.1,
  )
  state = init_fn(random.PRNGKey(index), position, box=box, mass=1.0)

  def sample(state, _):
    """Advance 100 steps and record temperature, pressure, and volume."""
    state = lax.fori_loop(0, WRITE_EVERY, lambda _, s: apply_fn(s), state)
    kinetic = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
    volume = quantity.volume(3, state.box)
    values = (
      quantity.temperature(momentum=state.momentum, mass=state.mass),
      (2 * kinetic - state.dUdV) / (3 * volume),
      volume,
    )
    return state, jnp.asarray(values)

  state, samples = jax.jit(
    lambda state: lax.scan(
      sample, state, None, length=NSTEPS_SIM // WRITE_EVERY
    )
  )(state)
  mean_temperature, mean_pressure, mean_volume = jnp.mean(
    samples[len(samples) // 2 :], axis=0
  )
  print(
    f'T={temperature:.1f}, P={pressure:.1f}: '
    f'<T>={mean_temperature:.3f}, '
    f'<P>={mean_pressure:.3f}, '
    f'<V>={mean_volume:.3f}'
  )
