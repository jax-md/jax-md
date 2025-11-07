# Copyright 2019 Google LLC
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

"""Utilities for defining dataclasses that can be used with jax transformations.

This code was copied and adapted from https://github.com/google/flax/struct.py.

Accessed on 04/29/2020.
"""

import dataclasses
import jax


def dataclass(clz):
  """Create a class which can be passed to functional transformations.

  Jax transformations such as `jax.jit` and `jax.grad` require objects that are
  immutable and can be mapped over using the `jax.tree_util` methods.

  The `dataclass` decorator makes it easy to define custom classes that can be
  passed safely to Jax.

  Args:
    clz: the class that will be transformed by the decorator.
  Returns:
    The new class.
  """
  clz.set = lambda self, **kwargs: dataclasses.replace(self, **kwargs)
  data_clz = dataclasses.dataclass(frozen=True)(clz)
  meta_fields = []
  data_fields = []
  for name, field_info in data_clz.__dataclass_fields__.items():
    is_static = field_info.metadata.get('static', False)
    if is_static:
      meta_fields.append(name)
    else:
      data_fields.append(name)

  def iterate_clz(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple(getattr(x, name) for name in data_fields)
    return data, meta

  def clz_from_iterable(meta, data):
    meta_args = tuple(zip(meta_fields, meta))
    data_args = tuple(zip(data_fields, data))
    kwargs = dict(meta_args + data_args)
    return data_clz(**kwargs)

  jax.tree_util.register_pytree_node(data_clz, iterate_clz, clz_from_iterable)

  return data_clz


def static_field():
  return dataclasses.field(metadata={'static': True})


replace = dataclasses.replace
asdict = dataclasses.asdict
astuple = dataclasses.astuple
is_dataclass = dataclasses.is_dataclass
fields = dataclasses.fields
field = dataclasses.field


def unpack(dc) -> tuple:
  return tuple(getattr(dc, field.name) for field in dataclasses.fields(dc))
# Copyright 2019 Google LLC
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

"""Utilities for defining dataclasses that can be used with jax transformations.

This code was copied and adapted from https://github.com/google/flax/struct.py.

Accessed on 04/29/2020.
"""

import dataclasses
from dataclasses import Field as _Field
from dataclasses import asdict as _asdict
from dataclasses import astuple as _astuple
from dataclasses import field as _field
from dataclasses import fields as _fields
from dataclasses import is_dataclass as _is_dataclass
from dataclasses import replace as _replace
from typing import Any, Callable, Optional, TypeVar, overload

import jax

__all__ = (
    'dataclass',
    'static_field',
    'unpack',
    'replace',
    'asdict',
    'astuple',
    'is_dataclass',
    'fields',
    'field',
)


T = TypeVar('T', bound=type[Any])


@overload
def dataclass(clz: T, *, frozen: bool = True, **dataclass_kwargs: Any) -> T:
    ...


@overload
def dataclass(*, frozen: bool = True, **dataclass_kwargs: Any) -> Callable[[T], T]:
    ...


def dataclass(clz: Optional[T] = None,
              *,
              frozen: bool = True,
              **dataclass_kwargs: Any) -> T | Callable[[T], T]:
    """Create a class which can be passed to functional transformations.

    Jax transformations such as `jax.jit` and `jax.grad` require objects that are
    immutable and can be mapped over using the `jax.tree_util` methods.

    The `dataclass` decorator makes it easy to define custom classes that can be
    passed safely to Jax by relying on `jax.tree_util.register_dataclass`.

    Args:
        clz: the class that will be transformed by the decorator.
        frozen: whether the resulting dataclass should be frozen. Defaults to True.
        **dataclass_kwargs: additional keyword arguments forwarded to
            `dataclasses.dataclass`.
    Returns:
        The new class.
    """

    if 'frozen' in dataclass_kwargs:
        requested_frozen = dataclass_kwargs.pop('frozen')
        if requested_frozen != frozen:
            raise TypeError(
                "'frozen' must match the decorator argument when provided in dataclass_kwargs"
            )

    def decorate(target_clz: T) -> T:
        data_clz = dataclasses.dataclass(frozen=frozen, **dataclass_kwargs)(target_clz)
        registered_clz = jax.tree_util.register_dataclass(data_clz)

        def _set(self, **kwargs):
            return _replace(self, **kwargs)

        setattr(registered_clz, 'set', _set)
        return registered_clz

    if clz is None:
        return decorate

    return decorate(clz)


def static_field(*,
                 metadata: Optional[dict[str, Any]] = None,
                 **field_kwargs: Any) -> _Field[Any]:
    """Create a field that is treated as static (non-pytree) by JAX."""
    combined_metadata = dict(metadata or {})
    combined_metadata.setdefault('static', True)
    combined_metadata['pytree_node'] = False
    return _field(metadata=combined_metadata, **field_kwargs)


def unpack(dc: Any) -> tuple[Any, ...]:
    """Return a tuple of dataclass attribute values.

    This is a lightweight alternative to :func:`dataclasses.astuple` that avoids
    recursion and respects custom attribute access defined on the dataclass.
    """
    return tuple(getattr(dc, field.name) for field in _fields(dc))


replace = _replace
asdict = _asdict
astuple = _astuple
is_dataclass = _is_dataclass
fields = _fields
field = _field
