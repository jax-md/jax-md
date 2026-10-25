# Copyright 2022 Google LLC
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

"""GNoME checkpoints are not supported; :func:`load_model` raises."""

from typing import NoReturn


def load_model(directory: str) -> NoReturn:
  """Not supported; see google-deepmind/materials_discovery."""
  raise NotImplementedError(
    f'jax-md does not read GNoME checkpoints such as {directory}: they use '
    'the earlier NequIP architecture. The checkpoint layout and a loader are '
    'at https://github.com/google-deepmind/materials_discovery under model/.'
  )
