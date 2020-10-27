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

try:
  import IPython

  import base64

  import numpy as onp

  import jax.numpy as jnp
  from jax_md import dataclasses

  from google.colab import output


  import time

  renderer_code = IPython.display.HTML(url=('https://raw.githubusercontent.com/'
                                            'google/jax-md/visualization_2/jax_md/'
                                            'colab_tools/visualization.html'))

  SIMULATION_IDX = 0

  @dataclasses.dataclass
  class Disk:
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


  def encode(R):
    dtype = R.dtype
    if dtype == jnp.float64:
      dtype = jnp.float32
    if dtype == jnp.int64:
      dtype = jnp.int32
    dtype = jnp.float32
    return base64.b64encode(onp.array(R, dtype).tobytes()).decode('utf-8')

  import json

  def to_json(data):
    try:
      return IPython.display.JSON(data=data)
    except:
      return IPython.display.JSON(data=json.dumps(data))

  def render(box_size, 
            geometry,
            buffer_size=None, 
            background_color=None, 
            resolution=None):
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
      box_size = float(box_size)

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

      return to_json(metadata)
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
      return to_json(geom_metadata)
    output.register_callback(f'GetGeometryMetadata{SIMULATION_IDX}',
                            get_dynamic_geometry_metadata)

    def get_array_chunk(name, field, offset, size):
      assert name in geometry

      geom = dataclasses.asdict(geometry[name])
      assert field in geom
      array = geom[field]

      return to_json({
          'array_chunk': encode(array[offset:(offset + size)])
      })
    output.register_callback(f'GetArrayChunk{SIMULATION_IDX}', get_array_chunk)

    def get_array(name, field):
      assert name in geometry

      geom = dataclasses.asdict(geometry[name])
      assert field in geom
      array = geom[field]

      return to_json({ 'array': encode(array) })
    output.register_callback(f'GetArray{SIMULATION_IDX}', get_array)

    SIMULATION_IDX += 1

    display(renderer_code)

except Exception as e:
  print(e)
