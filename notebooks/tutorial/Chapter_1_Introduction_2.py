#@title Import & Util

import jax.numpy as np
from jax import device_put
from jax import config
# TODO: Uncomment this and enable warnings when XLA bug is fixed.
import warnings; warnings.simplefilter('ignore')
config.update('jax_enable_x64', True)
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import brainunit as u

import warnings
warnings.simplefilter("ignore")

sns.set_style(style='white')
background_color = [56 / 256] * 3
def plot(x, y, *args):
  plt.plot(x, y, *args, linewidth=3)
  plt.gca().set_facecolor([1, 1, 1])
def draw(R, **kwargs):
  if 'c' not in kwargs:
    kwargs['color'] = [1, 1, 0.9]
  ax = plt.axes(xlim=(0, float(jnp.max(R[:, 0]))), 
                ylim=(0, float(jnp.max(R[:, 1]))))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_facecolor(background_color)
  plt.scatter(R[:, 0], R[:, 1],  marker='o', s=1024, **kwargs)
  plt.gcf().patch.set_facecolor(background_color)
  plt.gcf().set_size_inches(6, 6)
  plt.tight_layout()
def draw_big(R, **kwargs):
  if 'c' not in kwargs:
    kwargs['color'] = [1, 1, 0.9]
  fig = plt.figure(dpi=128)
  ax = plt.axes(xlim=(0, float(jnp.max(R[:, 0]))),
                ylim=(0, float(jnp.max(R[:, 1]))))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_facecolor(background_color)
  s = plt.scatter(R[:, 0], R[:, 1], marker='o', s=0.5, **kwargs)
  s.set_rasterized(True)
  plt.gcf().patch.set_facecolor(background_color)
  plt.gcf().set_size_inches(10, 10)
  plt.tight_layout()
def draw_displacement(R, dR):
  plt.quiver(R[:, 0], R[:, 1], dR[:, 0], dR[:, 1], color=[1, 0.5, 0.5])

# Progress Bars

from IPython.display import HTML, display
import time

def ProgressIter(iter_fun, iter_len=0):
  if not iter_len:
    iter_len = len(iter_fun)
  out = display(progress(0, iter_len), display_id=True)
  for i, it in enumerate(iter_fun):
    yield it
    out.update(progress(i + 1, iter_len))

def progress(value, max):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 45%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))
    
import jax.numpy as jnp
from jax import random
from jax_md import units as ju
from jax_md import simulate, minimize
from jax import lax
from jax_md import space, quantity
from jax import jit

key = random.PRNGKey(1)

particle_count = 128
dim = 2

box_size = quantity.box_size_at_number_density(particle_count = particle_count,
                                               number_density = 1.2,
                                               spatial_dimension = dim)

R = random.uniform(key, (particle_count, dim), maxval=box_size.to_decimal(box_size.unit)) * ju.angstrom

@u.assign_units(r=ju.angstrom, result=ju.eV_unit)
def soft_sphere(r):
  return u.math.where(r < 1,
                   1/3 * (1 - r) ** 3,
                   0.)

displacement, shift = space.periodic(box_size)

@u.check_dims(R=ju.angstrom.dim, result=ju.eV.dim)
def energy_fn(R):
  dr = v_metric(R, R)
  return 0.5 * u.math.sum(soft_sphere(dr))

sim_steps = 10000
steps_per_write = 100
write_steps = sim_steps // steps_per_write


metric = space.metric(displacement)

v_displacement = space.map_product(displacement)
v_metric = space.map_product(metric)

init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt=5e-3 * ju.fsecond, kT=1e-4 * u.kelvin)

from jax import lax

@jit
def min(R):
  init_fn, apply_fn = minimize.fire_descent(energy_fn, shift)

  state = init_fn(R)
  # Using a JAX loop reduces compilation cost
  state = lax.while_loop(cond_fun=cond_fn,
                         body_fun=apply_fn,
                         init_val=state)

  return state.position

cond_fn = lambda state: u.math.max(u.math.abs(state.force)) > 1e-4 * ju.force_unit

R_is = min(R)

@jit
def sim(key):
  state = init_fn(key, R_is)

  def sim_fn(i, state_positions):
    state, positions = state_positions
    positions = positions.at[i].set(state.position)
    state = lax.fori_loop(0, steps_per_write, lambda _, s: step_fn(s), state)
    return state, positions

  positions = u.math.zeros((write_steps, particle_count, dim), unit=ju.angstrom)

  state, positions = lax.fori_loop(0, write_steps, sim_fn, (state, positions))
  return positions

key = random.PRNGKey(0)
positions = sim(key)