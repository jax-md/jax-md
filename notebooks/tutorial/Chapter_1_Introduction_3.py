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
from jax_md import space, quantity, energy
from jax import jit

key = random.PRNGKey(0)

particle_count = 128000
dim = 2
box_size = quantity.box_size_at_number_density(particle_count    = particle_count,
                                               number_density    = 1.0,
                                               spatial_dimension = dim)


R = random.uniform(key, (particle_count, dim)) * box_size

displacement, shift = space.periodic(box_size)

neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(displacement, box_size)

init_fn, apply_fn = minimize.fire_descent(energy_fn, shift)

nbrs = neighbor_fn.allocate(R)
print(nbrs.idx.shape)