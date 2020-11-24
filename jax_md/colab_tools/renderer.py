# Copyright 2020 Google LLC
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

"""Kernel-side code for an IPython based visualization tool."""

import base64

from google.colab import output

import IPython

import jax.numpy as jnp
from jax_md import dataclasses

import json

import numpy as onp

import time

renderer_code = IPython.display.HTML(
  url=('https://raw.githubusercontent.com/google/jax-md/master/'
       'jax_md/colab_tools/visualization.html')
)

SIMULATION_IDX = 0


@dataclasses.dataclass
class Disk:
  """Disk geometry elements.

  Args:
    position: An array of shape `(steps, count, dim)` or `(count, dim)` 
      specifying possibly time varying positions. Here `dim` is the spatial
      dimension.
    size: An array of shape (steps, count)`, `(count,)`, or `()` specifying
      possibly time-varying / per-disk diameters.
    color: An array of shape `(steps, count, 3)` or `(count,)` specifying
      possibly time-varying / per-disk RGB colors.
    count: The number of disks.
  """
  position: jnp.ndarray
  size: jnp.ndarray
  color: jnp.ndarray
  count: int = dataclasses.static_field()

  def __init__(self, position, diameter=1.0, color=None):
    if color is None:
      color = jnp.array([0.8, 0.8, 1.0])
    
    object.__setattr__(self, 'position', position)
    object.__setattr__(self, 'size', diameter)
    object.__setattr__(self, 'color', color)
    object.__setattr__(self, 'count', position.shape[-2])

  def __repr__(self):
    return 'Disk'


@dataclasses.dataclass
class Sphere:
  """Sphere geometry elements.

  Args:
    position: An array of shape `(steps, count, dim)` or `(count, dim)` 
      specifying possibly time varying positions. Here `dim` is the spatial
      dimension.
    size: An array of shape (steps, count)`, `(count,)`, or `()` specifying
      possibly time-varying / per-sphere diameters.
    color: An array of shape `(steps, count, 3)` or `(count,)` specifying
      possibly time-varying / per-sphere RGB colors.
    count: The number of spheres.
  """
  position: jnp.ndarray
  size: jnp.ndarray
  color: jnp.ndarray
  count: int

  def __init__(self, position, diameter=1.0, color=None):
    if color is None:
      color = jnp.array([0.8, 0.8, 1.0])
    
    object.__setattr__(self, 'position', position)
    object.__setattr__(self, 'size', diameter)
    object.__setattr__(self, 'color', color)
    object.__setattr__(self, 'count', position.shape[-2])

  def __repr__(self):
    return 'Sphere'


@dataclasses.dataclass
class Bond:
  """Bonds are lines between geometric objects.

  Args:
    reference_geometery: The name of the geometry object to draw bonds between.
    neighbor_idx: An array of ids of objects that should have bonds drawn 
      between them. This uses the same encoding as in `partition.neighbor_list`.
      Essentially, ids is a `(steps, count, max_neighbors)` or 
      `(count, max_neighbors)` array of integers. `neighbor_idx[i, j] < count`
      denotes a bond between object `i` and `neighbor[i, j]`.  
    diameter: The width of the line between the objects.
    color: An array of shape `(3,)` specifying the RGB color of the bonds.
    count: The number of objects.
    max_neighbors: The maximum number of bonds a central object can have.
  """
  reference_geometry: str
  neighbor_idx: jnp.ndarray
  diameter: jnp.ndarray
  color: jnp.ndarray
  count: int
  max_neighbors: int

  def __init__(self, reference_geometry, idx, diameter=1.0, color=None):
    if color is None:
      color = jnp.array([0.8, 0.8, 1.0])

    object.__setattr__(self, 'reference_geometry', reference_geometry)
    object.__setattr__(self, 'neighbor_idx', idx)
    object.__setattr__(self, 'diameter', diameter)
    object.__setattr__(self, 'color', color)
    object.__setattr__(self, 'count', idx.shape[-2])
    object.__setattr__(self, 'max_neighbors', idx.shape[-1])

  def __repr__(self):
    return 'Bond'


TYPE_DIMENSIONS = {
    'position': 2,
    'size': 1,
    'color': 2,
    'neighbor_idx': 2,
    'diameter': 1,
}


def _encode(R):
  dtype = R.dtype
  if dtype == jnp.float64:
    dtype = jnp.float32
  if dtype == jnp.int64:
    dtype = jnp.int32
  dtype = jnp.float32
  return base64.b64encode(onp.array(R, dtype).tobytes()).decode('utf-8')

def _to_json(data):
  try:
    return IPython.display.JSON(data=data)
  except:
    return IPython.display.JSON(data=json.dumps(data))

def render(box_size, 
           geometry,
           buffer_size=None, 
           background_color=None, 
           resolution=None):
  """Creates a rendering front-end along with callbacks in the host program.

  Args:
    box_size: A float or an array of shape `(spatial_dimension,)`. Specifies
      the size of the simulation volume. Used to position the camera.
    geometry: A dictionary containing names paired with geometric objects such
      as Disk, Sphere, or Bond.
    buffer_size: The maximum number of timesteps to send to the font-end in a
      single call.
    background_color: An array of shape (3,) specifying the background color of
      the visualization.
    resolution: The resolution of the renderer.
  """
  global SIMULATION_IDX
  
  simulation_idx = SIMULATION_IDX

  frame_count = None
  dimension = None

  for geom in geometry.values():
    if hasattr(geom, 'position'):
      assert dimension is None or goem.position.shape[-1] == dimension
      dimension = geom.position.shape[-1]

      if geom.position.ndim == 3:
        assert frame_count is None or frame_count == geom.position.shape[0]
        frame_count = geom.position.shape[0]

  assert dimension is not None

  if isinstance(box_size, jnp.ndarray):
    if box_size.shape:
      assert box_size.shape == (dimension,)
      box_size = list(box_size)
    else:
      box_size = [float(box_size),] * dimension
  elif isinstance(box_size, float) or isinstance(box_size, int):
    box_size = [box_size,] * dimension

  def get_metadata():
    metadata = {
        'box_size': box_size,
        'dimension': dimension,
        'geometry': [k for k in geometry.keys()],
        'simulation_idx': simulation_idx
    }

    if frame_count is not None:
      metadata['frame_count'] = frame_count

    if buffer_size is not None:
      metadata['buffer_size'] = buffer_size

    if background_color is not None:
      metadata['background_color'] = background_color

    if resolution is not None:
      metadata['resolution'] = resolution

    return _to_json(metadata)
  output.register_callback('GetSimulationMetadata', get_metadata)

  def get_dynamic_geometry_metadata(name):
    assert name in geometry

    geom = geometry[name]
    geom_dict = dataclasses.asdict(geom)

    geom_metadata = {
        'shape': str(geom),
        'fields': {},
    }

    for field in geom_dict:
      if not isinstance(geom_dict[field], onp.ndarray):
        geom_metadata[field] = geom_dict[field]
        continue
      if len(geom_dict[field].shape) == TYPE_DIMENSIONS[field] + 1:
        geom_metadata['fields'][field] = 'dynamic'
      elif len(geom_dict[field].shape) == TYPE_DIMENSIONS[field]:
        geom_metadata['fields'][field] = 'static'
      elif len(geom_dict[field].shape) == TYPE_DIMENSIONS[field] - 1:
        geom_metadata['fields'][field] = 'global'
    return _to_json(geom_metadata)
  output.register_callback(f'GetGeometryMetadata{SIMULATION_IDX}',
                          get_dynamic_geometry_metadata)

  def get_array_chunk(name, field, offset, size):
    assert name in geometry

    geom = dataclasses.asdict(geometry[name])
    assert field in geom
    array = geom[field]

    return _to_json({
        'array_chunk': _encode(array[offset:(offset + size)])
    })
  output.register_callback(f'GetArrayChunk{SIMULATION_IDX}', get_array_chunk)

  def get_array(name, field):
    assert name in geometry

    geom = dataclasses.asdict(geometry[name])
    assert field in geom
    array = geom[field]

    return _to_json({ 'array': _encode(array) })
  output.register_callback(f'GetArray{SIMULATION_IDX}', get_array)

  SIMULATION_IDX += 1

  IPython.display.display(renderer_code)
