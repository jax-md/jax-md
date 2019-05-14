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

"""Defines utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.tree_util import register_pytree_node


def register_pytree_namedtuple(cls):
  register_pytree_node(
      cls,
      lambda xs: (tuple(xs), None),
      lambda _, xs: cls(*xs))
